# voice-en--to-voice-es-

Real-time **English voice → Spanish voice** translator built as a
three-stage pipeline:

```
English audio  ──▶  Speech-to-Text  ──▶  Translation  ──▶  Text-to-Speech  ──▶  Spanish audio
                    (Whisper API)        (Google Translate)      (gTTS)
```

---

## Features

- **Multi-stage pipeline** – each stage (STT → Translate → TTS) is an
  independent, replaceable module.
- **Real-time inference** – per-stage latency is measured and reported so
  bottlenecks are immediately visible.
- **Low-latency data flow** – results flow directly from one stage to the
  next with no intermediate file I/O between stages.
- **File mode** – translate any existing audio file.
- **Microphone mode** – record from the default mic, then translate
  immediately.

---

## Project structure

```
.
├── app.py              # CLI entry point
├── pipeline.py         # Orchestrates the three stages
├── modules/
│   ├── stt.py          # Stage 1 – Speech-to-Text   (OpenAI Whisper API)
│   ├── translate.py    # Stage 2 – Translation      (Google Translate via deep-translator)
│   └── tts.py          # Stage 3 – Text-to-Speech   (gTTS)
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3a. Translate a pre-recorded file

```bash
python app.py --input path/to/english.wav --output-dir output/
```

### 3b. Record from the microphone and translate

```bash
python app.py --record --duration 10 --output-dir output/
```

### Output

```
============================================================
  English  : Hello, how are you today?
  Spanish  : Hola, ¿cómo estás hoy?
  Output   : /absolute/path/output/english_es.mp3
  Latency  : STT=1.23s  Translate=0.45s  TTS=0.38s  Total=2.07s
============================================================
```

---

## Running tests

```bash
pytest tests/ -v
```

All tests mock external API calls so no network access or API key is
required.

---

## Environment variables

| Variable         | Description                              | Required |
|------------------|------------------------------------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for the Whisper STT stage | Yes      |

---

## CLI reference

```
usage: app.py [-h] (--input FILE | --record) [--duration SECONDS]
              [--output-dir DIR] [--output-filename FILENAME]
              [--whisper-model MODEL] [--slow-tts] [--api-key KEY]

options:
  --input FILE, -i FILE       Path to an existing English audio file
  --record, -r                Record audio from the default microphone
  --duration SECONDS, -d N    Recording duration in seconds (default: 10)
  --output-dir DIR, -o DIR    Output directory (default: output)
  --output-filename FILENAME  Name for the output MP3
  --whisper-model MODEL       Whisper model (default: whisper-1)
  --slow-tts                  Generate speech at reduced speed
  --api-key KEY               OpenAI API key (overrides env variable)
```
