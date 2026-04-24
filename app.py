"""
app.py – command-line entry point for the English → Spanish voice translator.

Usage
-----
Translate a pre-recorded file::

    python app.py --input path/to/english.wav

Record from the microphone for N seconds, then translate::

    python app.py --record --duration 10 --output-dir ./output

Environment variables
---------------------
OPENAI_API_KEY
    Required for the Whisper speech-to-text stage.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional microphone recording helper
# ---------------------------------------------------------------------------

def record_from_microphone(duration: int, output_path: str) -> str:
    """
    Record *duration* seconds of audio from the default microphone.

    Requires ``sounddevice`` and ``soundfile`` (both listed in
    ``requirements.txt``).

    Parameters
    ----------
    duration:
        Recording length in seconds.
    output_path:
        Path where the recorded WAV file is saved.

    Returns
    -------
    str
        Absolute path to the recorded WAV file.
    """
    try:
        import sounddevice as sd  # type: ignore
        import soundfile as sf    # type: ignore
    except ImportError as exc:
        logger.error(
            "Microphone recording requires 'sounddevice' and 'soundfile'. "
            "Install them with:  pip install sounddevice soundfile\n"
            "Original error: %s",
            exc,
        )
        sys.exit(1)

    sample_rate = 16_000  # 16 kHz – optimal for Whisper
    logger.info("Recording %ds of audio (press Ctrl+C to stop early) …", duration)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sample_rate)
    logger.info("Recorded audio saved to '%s'", out)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time English→Spanish voice translator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Path to an existing English audio file (WAV, MP3, M4A, …).",
    )
    source.add_argument(
        "--record", "-r",
        action="store_true",
        help="Record audio from the default microphone before translating.",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Recording duration in seconds (only used with --record).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        metavar="DIR",
        help="Directory where the translated MP3 is written.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        metavar="FILENAME",
        help="Name for the output MP3 (auto-derived from input when omitted).",
    )
    parser.add_argument(
        "--whisper-model",
        default="whisper-1",
        metavar="MODEL",
        help="OpenAI Whisper model identifier.",
    )
    parser.add_argument(
        "--slow-tts",
        action="store_true",
        help="Generate Spanish speech at reduced speed.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="OpenAI API key (overrides OPENAI_API_KEY env variable).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    """
    Entry point.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    from pipeline import VoiceTranslationPipeline  # local import for testability

    parser = _build_parser()
    args = parser.parse_args(argv)

    api_key: Optional[str] = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "No OpenAI API key found.  Set OPENAI_API_KEY or pass --api-key."
        )
        return 1

    # --- Resolve / record input audio ---
    if args.record:
        recorded_path = Path(args.output_dir) / "recorded_input.wav"
        audio_path = record_from_microphone(args.duration, str(recorded_path))
    else:
        audio_path = args.input

    # --- Run pipeline ---
    pipeline = VoiceTranslationPipeline(
        openai_api_key=api_key,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        tts_slow=args.slow_tts,
    )

    start = time.perf_counter()
    result = pipeline.run(audio_path, output_filename=args.output_filename)
    wall_time = time.perf_counter() - start

    # --- Report ---
    if result.success:
        print("\n" + "=" * 60)
        print(f"  English  : {result.english_text}")
        print(f"  Spanish  : {result.spanish_text}")
        print(f"  Output   : {result.output_audio}")
        print(f"  Latency  : STT={result.latency_stt:.2f}s  "
              f"Translate={result.latency_translate:.2f}s  "
              f"TTS={result.latency_tts:.2f}s  "
              f"Total={wall_time:.2f}s")
        print("=" * 60 + "\n")
        return 0
    else:
        for err in result.errors:
            logger.error("Pipeline error: %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
