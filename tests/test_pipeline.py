"""
Unit tests for the three pipeline modules and the orchestrating pipeline.

All external API calls (OpenAI Whisper, Google Translate, gTTS) are mocked
so the test suite runs without network access or API keys.
"""

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _stub_gtts_module():
    """Return a minimal stub for the gtts package."""
    gtts_mod = types.ModuleType("gtts")
    tts_mod = types.ModuleType("gtts.tts")

    class _FakegTTS:
        def __init__(self, text, lang, slow=False):
            self.text = text
            self.lang = lang

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake-mp3-data")

    gtts_mod.gTTS = _FakegTTS
    tts_mod.gTTSError = Exception
    return gtts_mod, tts_mod


def _stub_deep_translator_module():
    """Return a minimal stub for the deep_translator package."""
    dt_mod = types.ModuleType("deep_translator")
    exc_mod = types.ModuleType("deep_translator.exceptions")

    class _FakeGoogleTranslator:
        def __init__(self, source="en", target="es"):
            self._target = target

        def translate(self, text):
            # Simple deterministic stub
            return f"[{self._target}] {text}"

    dt_mod.GoogleTranslator = _FakeGoogleTranslator
    exc_mod.TranslationNotFound = Exception
    return dt_mod, exc_mod


def _stub_openai_module():
    """Return a minimal stub for the openai package."""
    openai_mod = types.ModuleType("openai")

    class _FakeTranscription:
        text = "Hello, world."

    class _FakeAudioTranscriptions:
        def create(self, **kwargs):
            return _FakeTranscription()

    class _FakeAudio:
        transcriptions = _FakeAudioTranscriptions()

    class _FakeOpenAI:
        def __init__(self, api_key=None):
            self.audio = _FakeAudio()

    openai_mod.OpenAI = _FakeOpenAI

    class _FakeOpenAIError(Exception):
        pass

    openai_mod.OpenAIError = _FakeOpenAIError
    return openai_mod


# ---------------------------------------------------------------------------
# Module-level patches so the real packages are never imported
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_external_libs(tmp_path, monkeypatch):
    """Patch all external library imports before each test."""
    gtts_mod, tts_mod = _stub_gtts_module()
    dt_mod, exc_mod = _stub_deep_translator_module()
    openai_mod = _stub_openai_module()

    monkeypatch.setitem(sys.modules, "gtts", gtts_mod)
    monkeypatch.setitem(sys.modules, "gtts.tts", tts_mod)
    monkeypatch.setitem(sys.modules, "deep_translator", dt_mod)
    monkeypatch.setitem(sys.modules, "deep_translator.exceptions", exc_mod)
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    # Ensure our local modules are reloaded with the stubs in place
    for mod_name in list(sys.modules):
        if mod_name.startswith("modules") or mod_name == "pipeline":
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    yield


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------

class TestSpeechToText:
    def test_transcribe_returns_text(self, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        from modules.stt import SpeechToText
        stt = SpeechToText(api_key="fake-key")
        result = stt.transcribe(str(audio))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transcribe_missing_file_raises(self):
        from modules.stt import SpeechToText
        stt = SpeechToText(api_key="fake-key")
        with pytest.raises(FileNotFoundError):
            stt.transcribe("/nonexistent/path/audio.wav")

    def test_transcribe_strips_whitespace(self, tmp_path, monkeypatch):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        # Override stub to return text with extra whitespace
        import sys
        openai_mod = sys.modules["openai"]

        class _PaddedTranscription:
            text = "   Hello, world.   "

        class _PaddedAudioTranscriptions:
            def create(self, **kwargs):
                return _PaddedTranscription()

        class _PaddedAudio:
            transcriptions = _PaddedAudioTranscriptions()

        class _PaddedOpenAI:
            def __init__(self, api_key=None):
                self.audio = _PaddedAudio()

        openai_mod.OpenAI = _PaddedOpenAI
        # Force reimport
        monkeypatch.delitem(sys.modules, "modules.stt", raising=False)

        from modules.stt import SpeechToText
        stt = SpeechToText(api_key="fake-key")
        result = stt.transcribe(str(audio))
        assert result == "Hello, world."


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class TestTranslator:
    def test_translate_returns_string(self):
        from modules.translate import Translator
        t = Translator()
        result = t.translate("Hello, world.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_empty_input_returns_empty(self):
        from modules.translate import Translator
        t = Translator()
        assert t.translate("") == ""
        assert t.translate("   ") == ""

    def test_translate_passes_text_to_backend(self):
        from modules.translate import Translator
        t = Translator(source="en", target="es")
        result = t.translate("Good morning")
        # Stub prefixes with [es]
        assert "[es]" in result
        assert "Good morning" in result


# ---------------------------------------------------------------------------
# TextToSpeech
# ---------------------------------------------------------------------------

class TestTextToSpeech:
    def test_synthesise_creates_file(self, tmp_path):
        from modules.tts import TextToSpeech
        tts = TextToSpeech()
        out = str(tmp_path / "out.mp3")
        returned = tts.synthesise("Hola, mundo.", out)
        assert Path(returned).exists()

    def test_synthesise_empty_raises(self, tmp_path):
        from modules.tts import TextToSpeech
        tts = TextToSpeech()
        with pytest.raises(ValueError):
            tts.synthesise("", str(tmp_path / "out.mp3"))

    def test_synthesise_returns_absolute_path(self, tmp_path):
        from modules.tts import TextToSpeech
        tts = TextToSpeech()
        out = str(tmp_path / "out.mp3")
        returned = tts.synthesise("Hola", out)
        assert Path(returned).is_absolute()


# ---------------------------------------------------------------------------
# VoiceTranslationPipeline
# ---------------------------------------------------------------------------

class TestVoiceTranslationPipeline:
    def test_full_pipeline_success(self, tmp_path):
        audio = tmp_path / "input.wav"
        audio.write_bytes(b"\x00" * 100)

        from pipeline import VoiceTranslationPipeline
        p = VoiceTranslationPipeline(
            openai_api_key="fake-key",
            output_dir=str(tmp_path / "output"),
        )
        result = p.run(str(audio))

        assert result.success
        assert result.english_text == "Hello, world."
        assert len(result.spanish_text) > 0
        assert Path(result.output_audio).exists()
        # All three stage latencies should be non-negative
        assert result.latency_stt >= 0
        assert result.latency_translate >= 0
        assert result.latency_tts >= 0

    def test_pipeline_missing_audio_fails(self, tmp_path):
        from pipeline import VoiceTranslationPipeline
        p = VoiceTranslationPipeline(
            openai_api_key="fake-key",
            output_dir=str(tmp_path / "output"),
        )
        result = p.run("/nonexistent/audio.wav")
        assert not result.success
        assert len(result.errors) == 1
        assert "STT" in result.errors[0]

    def test_pipeline_total_latency(self, tmp_path):
        audio = tmp_path / "input.wav"
        audio.write_bytes(b"\x00" * 100)

        from pipeline import VoiceTranslationPipeline
        p = VoiceTranslationPipeline(
            openai_api_key="fake-key",
            output_dir=str(tmp_path / "output"),
        )
        result = p.run(str(audio))
        assert result.total_latency == pytest.approx(
            result.latency_stt + result.latency_translate + result.latency_tts,
            abs=1e-6,
        )

    def test_pipeline_output_filename_override(self, tmp_path):
        audio = tmp_path / "input.wav"
        audio.write_bytes(b"\x00" * 100)

        from pipeline import VoiceTranslationPipeline
        out_dir = tmp_path / "output"
        p = VoiceTranslationPipeline(
            openai_api_key="fake-key",
            output_dir=str(out_dir),
        )
        result = p.run(str(audio), output_filename="custom_name.mp3")
        assert result.success
        assert "custom_name.mp3" in result.output_audio

    def test_pipeline_result_dataclass_defaults(self):
        from pipeline import PipelineResult
        r = PipelineResult(input_audio="test.wav")
        assert r.english_text == ""
        assert r.spanish_text == ""
        assert r.output_audio == ""
        assert r.errors == []
        assert not r.success
        assert r.total_latency == 0.0


# ---------------------------------------------------------------------------
# app.py CLI
# ---------------------------------------------------------------------------

class TestAppCLI:
    def test_missing_api_key_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        audio = tmp_path / "input.wav"
        audio.write_bytes(b"\x00" * 100)

        from app import main
        code = main(["--input", str(audio)])
        assert code == 1

    def test_file_mode_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        audio = tmp_path / "input.wav"
        audio.write_bytes(b"\x00" * 100)

        from app import main
        code = main([
            "--input", str(audio),
            "--output-dir", str(tmp_path / "output"),
        ])
        assert code == 0

    def test_nonexistent_file_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        from app import main
        code = main([
            "--input", "/nonexistent/audio.wav",
            "--output-dir", str(tmp_path / "output"),
        ])
        assert code == 1
