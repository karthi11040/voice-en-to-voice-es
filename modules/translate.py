"""
Translation module.

Translates English text to Spanish using the Google Translate back-end
provided by the ``deep-translator`` library.  The library is pure-Python
and does **not** require an API key for the free tier.
"""

import time
import logging
from typing import Optional

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


class Translator:
    """Translate English text to Spanish."""

    def __init__(
        self,
        source: str = "en",
        target: str = "es",
    ) -> None:
        """
        Initialise the translation module.

        Parameters
        ----------
        source:
            BCP-47 code of the input language (default: ``"en"``).
        target:
            BCP-47 code of the output language (default: ``"es"``).
        """
        self.source = source
        self.target = target
        self._translator = GoogleTranslator(source=source, target=target)

    def translate(self, text: str) -> str:
        """
        Translate *text* and return the Spanish equivalent.

        Parameters
        ----------
        text:
            English text to translate.

        Returns
        -------
        str
            Translated text, or an empty string when *text* is empty.

        Raises
        ------
        deep_translator.exceptions.TranslationNotFound
            When the translation service returns no result.
        """
        if not text.strip():
            return ""

        logger.info("TRANSLATE | '%s' → [%s] …", text[:80], self.target)
        t0 = time.perf_counter()

        translated: Optional[str] = self._translator.translate(text)
        translated = (translated or "").strip()

        elapsed = time.perf_counter() - t0
        logger.info("TRANSLATE | done in %.2fs → '%s'", elapsed, translated[:80])
        return translated
