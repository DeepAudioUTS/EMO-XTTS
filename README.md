# EMO-XTTS

An emotion-driven story narration system. Given any story text, it detects the emotion of each sentence using a fine-tuned BERT classifier, then generates speech for each sentence using Coqui XTTS v2 with a reference voice that matches that emotion. All sentence audio clips are stitched together into a single narrated audio file.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [How Each File Connects](#how-each-file-connects)
- [Folder Structure](#folder-structure)
- [Emotion Classes](#emotion-classes)
- [Environment Setup](#environment-setup)
- [Training the Emotion Model](#training-the-emotion-model)
- [Running the System](#running-the-system)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)

---

## Pipeline Overview

The system is made up of three independent services. Each runs in its own conda environment and communicates with the others over HTTP.

```
                   +------------------------------------+
                   |        User / curl / client       |
                   |  POST /narrate { "text": "..." }  |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |       Pipeline API  :8002         |
                   |       pipeline_api.py             |
                   |                                   |
                   |  1. Calls Classifier API          |
                   |  2. Maps emotions to WAVs         |
                   |  3. Calls XTTS per sentence       |
                   |  4. Concatenates WAVs             |
                   |  5. Returns story.wav             |
                   +--------+-----------------+--------+
                            |                 |
            POST /classify  |                 |  POST /audio/generate
                            v                 v
          +--------------------+    +---------------------+
          | Classifier  :8001  |    |    XTTS API  :8000  |
          | classifier_api.py  |    |    app.py           |
          |                    |    |                     |
          | chunker.py         |    | Loads XTTS v2       |
          |  splits sentences  |    | Accepts text +      |
          |                    |    | speaker_wav path    |
          | inference.py       |    | Clones voice style  |
          |  BERT predicts     |    | Saves output WAV    |
          |  emotion per       |    |                     |
          |  sentence          |    | EmotionAudio/       |
          |                    |    | *.wav  <- reference |
          | Returns:           |    | voices per emotion  |
          | [{ sentence,       |    |                     |
          |   emotion,         |    |                     |
          |   confidence }]    |    |                     |
          +--------------------+    +---------------------+
```

### Step-by-step walkthrough

**Step 1 — Input arrives at Pipeline API**

The user sends a POST request to `http://localhost:8002/narrate` with the full story text.

**Step 2 — Pipeline calls the Classifier API**

`pipeline_api.py` forwards the text to `http://classifier:8001/classify` (or `localhost:8001` when running locally without Docker).

**Step 3 — Classifier splits and classifies**

`classifier_api.py` receives the text and passes it to `chunker.py`, which splits it into individual sentences using a two-stage approach: first a regex-based splitter with abbreviation protection (handles `Dr.`, `U.S.A.`, decimal numbers etc.), then a spaCy sentencizer fallback if the result looks suspicious.

Each sentence is then passed to `inference.py`, which runs the fine-tuned BERT model and returns the predicted emotion label and confidence score. The classifier responds with:

```json
[
  { "sentence": "She wept softly.", "emotion": "Sad", "confidence": 0.91 },
  { "sentence": "He slammed the door!", "emotion": "Angry", "confidence": 0.84 }
]
```

**Step 4 — Pipeline maps emotions to reference WAVs**

`pipeline_api.py` contains a map of all 8 emotion labels to their corresponding reference WAV files in `EmotionAudio/`:

```
"Sad"   -> EmotionAudio/sad_sample.wav
"Angry" -> EmotionAudio/angry_sample.wav
"Happy" -> EmotionAudio/happy_sample.wav
```

These reference WAVs are short voice recordings that define the emotional tone XTTS will clone when generating speech.

**Step 5 — Pipeline calls XTTS API per sentence**

For each sentence, `pipeline_api.py` sends a POST request to `http://xtts:8000/audio/generate` with the sentence text and the path to the matching emotion WAV. `app.py` loads the reference WAV, clones its voice style using XTTS v2, synthesises the sentence in that emotional tone, and saves it as `output/<uuid>.wav`.

**Step 6 — Concatenation**

Once all sentences are generated, `pipeline_api.py` reads each `output/<uuid>.wav` in order and concatenates them using Python's `wave` module into a single `story_<uuid>.wav`.

**Step 7 — Final audio returned**

The final WAV is streamed back to the caller as `audio/wav`.

---

## How Each File Connects

### emotion_classifier/

```
data_pipeline.py  <-- called by train.py
        |
        v
train.py  <-- run manually
        |
        v
emotion_model/    <-- saved BERT checkpoint (not committed, must run train.py)

chunker.py  <-- called by inference.py
        |
        v
inference.py  <-- called by classifier_api.py
        |
        v
classifier_api.py  <-- FastAPI, port 8001, called by pipeline_api.py via HTTP
```

| File | What it does | Called by |
|------|-------------|-----------|
| `data_pipeline.py` | Downloads GoEmotions, maps 28 labels to 8 classes, splits 80/10/10 | `train.py` |
| `train.py` | Fine-tunes bert-base-uncased, saves best checkpoint, generates plots | Run manually |
| `chunker.py` | Splits text into individual sentences (regex + spaCy fallback) | `inference.py` |
| `inference.py` | Loads model, runs prediction, returns label + confidence | `classifier_api.py` |
| `classifier_api.py` | FastAPI exposing `POST /classify` | `pipeline_api.py` via HTTP |
| `app.py` | Gradio UI for interactive testing | Run manually |

### coqui-xtts/

```
EmotionAudio/*.wav  <-- paths sent by pipeline_api.py to XTTS API
        |
        v
pipeline_api.py  <-- calls Classifier and XTTS, concatenates results
        |
        v
app.py  <-- XTTS API, reads reference WAV, generates speech, saves output
        |
        v
output/<uuid>.wav  <-- collected and concatenated by pipeline_api.py
        |
        v
output/story_<uuid>.wav  <-- returned to caller
```

| File | What it does | Called by |
|------|-------------|-----------|
| `app.py` | XTTS FastAPI service, loads XTTS v2 at startup, generates speech from text + reference WAV | `pipeline_api.py` via HTTP |
| `pipeline_api.py` | Orchestrator, calls Classifier, maps emotions to WAVs, calls XTTS per sentence, concatenates, returns audio | User / curl |
| `pipeline.py` | Standalone CLI, runs the full pipeline locally without separate services | Run manually |
| `EmotionAudio/` | 8 reference WAV files, one per emotion, passed to XTTS to clone emotional tone | `pipeline_api.py` |
| `docker-compose.yml` | Defines all 3 services with GPU access, volumes, and shared network | `docker compose up` |

---

## Folder Structure

```
EMO-XTTS/
|
+-- emotion_classifier/
|   |
|   +-- Dockerfile                  # CUDA 12.1 + PyTorch + classifier deps
|   +-- requirements.txt            # transformers, datasets, spacy, scikit-learn, gradio
|   +-- data_pipeline.py            # GoEmotions download, 28->8 label mapping, 80/10/10 split
|   +-- train.py                    # Fine-tunes bert-base-uncased, saves best model
|   +-- chunker.py                  # Sentence splitting: regex + spaCy fallback
|   +-- inference.py                # Model loader, predict(), attention span highlighting
|   +-- classifier_api.py           # FastAPI service, POST /classify, port 8001
|   +-- app.py                      # Gradio UI for interactive testing
|
+-- coqui-xtts/
    |
    +-- Dockerfile                  # CUDA 12.1 + TTS + fastapi + httpx
    +-- docker-compose.yml          # Starts all 3 services together
    +-- requirements.txt            # TTS, fastapi, uvicorn, httpx, torch
    +-- app.py                      # XTTS FastAPI service, POST /audio/generate, port 8000
    +-- pipeline_api.py             # Pipeline orchestrator, POST /narrate, port 8002
    +-- pipeline.py                 # Standalone CLI (no Docker needed)
    |
    +-- EmotionAudio/
        +-- neutral_sample.wav
        +-- happy_sample.wav
        +-- sad_sample.wav
        +-- tense_fearful_sample.wav
        +-- angry_sample.wav
        +-- surprised_sample.wav
        +-- disgusted_sample.wav
        +-- loving_tender_sample.wav
```

> `emotion_model/` is not committed. Run `python train.py` to generate it at `emotion_classifier/emotion_model/`.

---

## Emotion Classes

| ID | Label | GoEmotions Labels Mapped |
|----|-------|--------------------------|
| 0 | Neutral | neutral |
| 1 | Happy | joy, amusement, excitement, optimism, relief, pride, gratitude |
| 2 | Sad | sadness, grief, disappointment, remorse |
| 3 | Tense/Fearful | fear, nervousness |
| 4 | Angry | anger, annoyance, disapproval |
| 5 | Surprised | surprise, realization |
| 6 | Disgusted | disgust, contempt |
| 7 | Loving/Tender | love, caring, admiration, desire |

---

## Environment Setup

The classifier and XTTS have conflicting dependencies so they run in separate conda environments.

### emotion_classifier env

```bash
conda create -n emotion_classifier python=3.10 -y
conda activate emotion_classifier

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd emotion_classifier
pip install -r requirements.txt
pip install fastapi uvicorn httpx accelerate
python -m spacy download en_core_web_sm

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### xtts env

```bash
conda activate xtts
pip install fastapi uvicorn httpx TTS torch
```

---

## Training the Emotion Model

```bash
conda activate emotion_classifier
cd emotion_classifier
python train.py
```

This will:
1. Download GoEmotions from HuggingFace (~54k samples after filtering)
2. Fine-tune `bert-base-uncased` for up to 4 epochs with early stopping (patience=2)
3. Save the best checkpoint to `./emotion_model/`
4. Print a classification report and save `confusion_matrix.png` and `loss_curve.png`

| Setting | Value |
|---------|-------|
| Base model | bert-base-uncased |
| Epochs | 4 (early stopping patience=2) |
| Batch size | 4 |
| Learning rate | 2e-5 |
| Precision | fp16 |
| Max tokens | 400 |
| Metric | Macro F1 |

---

## Running the System

### Option 1 — Docker (recommended)

```bash
cd coqui-xtts

# One-time: create the shared Docker network
docker network create deep-audio-network

# Build and start all 3 services
docker compose up --build
```

The XTTS model (~2GB) downloads on first run and is cached in a Docker volume.

### Option 2 — Conda (3 terminals)

**Terminal 1 — Classifier API**
```bash
conda activate emotion_classifier
cd emotion_classifier
uvicorn classifier_api:app --port 8001
```

**Terminal 2 — XTTS API**
```bash
conda activate xtts
cd coqui-xtts
uvicorn app:app --port 8000
```

**Terminal 3 — Pipeline API**
```bash
conda activate xtts
cd coqui-xtts
uvicorn pipeline_api:app --port 8002
```

### Generate audio

```bash
curl -X POST http://127.0.0.1:8002/narrate \
  -H "Content-Type: application/json" \
  -d '{"text": "She wept softly. He slammed the door. I love you so much."}' \
  --output story.wav

# Open on WSL
explorer.exe story.wav
```

---

## Docker

`docker-compose.yml` defines three containers on the `deep-audio-network` Docker bridge network so they can reach each other by service name.

| Container | Image | Port | GPU |
|-----------|-------|------|-----|
| `classifier` | `emotion_classifier/Dockerfile` | 8001 | Yes |
| `xtts` | `coqui-xtts/Dockerfile` | 8000 | Yes |
| `pipeline` | `coqui-xtts/Dockerfile` | 8002 | No |

Environment variables used by the pipeline container:

| Variable | Docker value | Local fallback |
|----------|-------------|----------------|
| `CLASSIFIER_URL` | `http://classifier:8001` | `http://127.0.0.1:8001` |
| `XTTS_URL` | `http://xtts:8000` | `http://127.0.0.1:8000` |
| `EMOTION_WAV_BASE` | `/app/EmotionAudio` | `./EmotionAudio` |

Volumes:
- `tts-cache` — persists downloaded XTTS model between restarts
- `./output` — generated WAVs accessible from the host
- `./EmotionAudio` — mounted into the XTTS container so reference WAVs are readable

---

## API Reference

### Classifier API — port 8001

**GET /**
```json
{ "message": "Emotion Classifier API is running" }
```

**POST /classify**

Request:
```json
{ "text": "She wept softly. He slammed the door!" }
```
Response:
```json
{
  "sentences": [
    { "sentence": "She wept softly.", "emotion": "Sad", "confidence": 0.91 },
    { "sentence": "He slammed the door!", "emotion": "Angry", "confidence": 0.84 }
  ]
}
```

---

### XTTS API — port 8000

All endpoints require header: `x-api-key: 0403766117`

**POST /audio/generate**

Request:
```json
{
  "text": "She wept softly.",
  "language": "en",
  "speaker_wav": "/app/EmotionAudio/sad_sample.wav"
}
```
Response:
```json
{ "file_id": "abc123", "audio_url": "/audio/abc123" }
```

**GET /audio/{file_id}** — download generated WAV

**DELETE /audio/{file_id}** — delete generated WAV

---

### Pipeline API — port 8002

**POST /narrate**

Request:
```json
{ "text": "Your full story here.", "language": "en" }
```
Response: Binary `audio/wav` — save with `--output story.wav`.

---
