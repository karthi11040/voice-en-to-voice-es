"""
Speech-to-Text module.

Transcribes English audio to text using the OpenAI Whisper API.
The audio file is sent directly to the API endpoint, so no local
GPU/model download is required.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import openai

logger = logging.getLogger(__name__)


class SpeechToText:
    """Transcribe audio files to English text via the Whisper API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        language: str = "en",
    ) -> None:
        """
        Initialise the STT module.

        Parameters
        ----------
        api_key:
            OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when not supplied.
        model:
            Whisper model name as accepted by the OpenAI API.
        language:
            BCP-47 language tag of the spoken language in the audio.
        """
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.language = language

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe *audio_path* and return the recognised text.

        Parameters
        ----------
        audio_path:
            Path to a Whisper-supported audio file (mp3, wav, m4a, …).

        Returns
        -------
        str
            Transcribed text, stripped of leading/trailing whitespace.

        Raises
        ------
        FileNotFoundError
            When *audio_path* does not exist.
        openai.OpenAIError
            On API-level failures.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("STT | transcribing '%s' …", path.name)
        t0 = time.perf_counter()

        with path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
            )

        elapsed = time.perf_counter() - t0
        text = response.text.strip()
        logger.info("STT | done in %.2fs → '%s'", elapsed, text[:80])
        return text
