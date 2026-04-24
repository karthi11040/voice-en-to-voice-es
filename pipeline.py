"""
pipeline.py – orchestrates the three-stage voice-translation pipeline.

Stages
------
1. Speech-to-Text   : English audio  →  English text   (Whisper API)
2. Translation      : English text   →  Spanish text   (Google Translate)
3. Text-to-Speech   : Spanish text   →  Spanish audio  (gTTS)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modules.stt import SpeechToText
from modules.translate import Translator
from modules.tts import TextToSpeech

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for the artefacts produced by a single pipeline run."""

    input_audio: str
    english_text: str = ""
    spanish_text: str = ""
    output_audio: str = ""
    latency_stt: float = 0.0
    latency_translate: float = 0.0
    latency_tts: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def total_latency(self) -> float:
        """Sum of all stage latencies in seconds."""
        return self.latency_stt + self.latency_translate + self.latency_tts

    @property
    def success(self) -> bool:
        """``True`` when the pipeline completed without errors."""
        return len(self.errors) == 0 and bool(self.output_audio)


class VoiceTranslationPipeline:
    """
    End-to-end pipeline: English voice → Spanish voice.

    Parameters
    ----------
    openai_api_key:
        OpenAI API key for the Whisper STT stage.  Falls back to the
        ``OPENAI_API_KEY`` environment variable when omitted.
    output_dir:
        Directory where translated audio files are written.
        Defaults to ``./output``.
    whisper_model:
        Whisper model identifier (e.g. ``"whisper-1"``).
    tts_slow:
        Synthesise Spanish speech at reduced speed when ``True``.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "output",
        whisper_model: str = "whisper-1",
        tts_slow: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._stt = SpeechToText(api_key=openai_api_key, model=whisper_model)
        self._translator = Translator(source="en", target="es")
        self._tts = TextToSpeech(language="es", slow=tts_slow)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, audio_path: str, output_filename: Optional[str] = None) -> PipelineResult:
        """
        Run the full pipeline on *audio_path*.

        Parameters
        ----------
        audio_path:
            Path to an English audio file (WAV, MP3, M4A, …).
        output_filename:
            Name for the generated Spanish MP3.  When omitted a name is
            derived from the input file.

        Returns
        -------
        PipelineResult
            All intermediate artefacts and per-stage latency figures.
        """
        result = PipelineResult(input_audio=audio_path)
        stem = Path(audio_path).stem

        if output_filename is None:
            output_filename = f"{stem}_es.mp3"

        output_path = str(self.output_dir / output_filename)

        logger.info("PIPELINE | starting for '%s'", audio_path)
        pipeline_start = time.perf_counter()

        # --- Stage 1: Speech-to-Text ---
        try:
            t0 = time.perf_counter()
            result.english_text = self._stt.transcribe(audio_path)
            result.latency_stt = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            logger.error("PIPELINE | STT failed: %s", exc)
            result.errors.append(f"STT: {exc}")
            return result

        # --- Stage 2: Translation ---
        try:
            t0 = time.perf_counter()
            result.spanish_text = self._translator.translate(result.english_text)
            result.latency_translate = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            logger.error("PIPELINE | Translation failed: %s", exc)
            result.errors.append(f"Translation: {exc}")
            return result

        # --- Stage 3: Text-to-Speech ---
        try:
            t0 = time.perf_counter()
            result.output_audio = self._tts.synthesise(result.spanish_text, output_path)
            result.latency_tts = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            logger.error("PIPELINE | TTS failed: %s", exc)
            result.errors.append(f"TTS: {exc}")
            return result

        total = time.perf_counter() - pipeline_start
        logger.info(
            "PIPELINE | completed in %.2fs  "
            "(STT=%.2fs  TRANSLATE=%.2fs  TTS=%.2fs)",
            total,
            result.latency_stt,
            result.latency_translate,
            result.latency_tts,
        )
        return result
