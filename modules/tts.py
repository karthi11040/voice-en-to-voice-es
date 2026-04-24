"""
Text-to-Speech module.

Converts Spanish text to an MP3 audio file using Google Text-to-Speech
(``gTTS``).  The resulting audio can be saved to disk and/or played back
immediately.
"""

import time
import logging
from pathlib import Path

from gtts import gTTS

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Synthesise Spanish speech from text."""

    def __init__(self, language: str = "es", slow: bool = False) -> None:
        """
        Initialise the TTS module.

        Parameters
        ----------
        language:
            BCP-47 language code for the synthesised speech.
        slow:
            When ``True`` the speech is generated at a reduced speed,
            which can improve comprehension for learners.
        """
        self.language = language
        self.slow = slow

    def synthesise(self, text: str, output_path: str) -> str:
        """
        Convert *text* to speech and write the result to *output_path*.

        Parameters
        ----------
        text:
            Spanish text to synthesise.
        output_path:
            Destination path for the generated MP3 file.

        Returns
        -------
        str
            Absolute path to the generated audio file.

        Raises
        ------
        ValueError
            When *text* is empty.
        gtts.tts.gTTSError
            On TTS API-level failures.
        """
        if not text.strip():
            raise ValueError("Cannot synthesise empty text.")

        logger.info("TTS | synthesising %d chars …", len(text))
        t0 = time.perf_counter()

        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tts.save(str(out))

        elapsed = time.perf_counter() - t0
        logger.info("TTS | done in %.2fs → '%s'", elapsed, out)
        return str(out.resolve())
