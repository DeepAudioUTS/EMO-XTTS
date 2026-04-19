"""
pipeline_api.py

Orchestrator API — calls Emotion Classifier API and XTTS API,
concatenates per-sentence audio, and returns the final WAV.

Run (xtts conda env, port 8002):
    conda activate xtts
    uvicorn pipeline_api:app --port 8002

Endpoints:
    GET  /        → health check
    POST /narrate → full story → final audio WAV
"""

from __future__ import annotations

import uuid
import wave
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os

# Use env vars so the same image works both locally and in Docker
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://127.0.0.1:8001")
XTTS_URL       = os.getenv("XTTS_URL",       "http://127.0.0.1:8000")
XTTS_API_KEY   = "0403766117"

PIPELINE_DIR   = Path(__file__).parent
OUTPUT_DIR     = str(PIPELINE_DIR / "output")

# EMOTION_WAV_BASE is the path to EmotionAudio *inside the XTTS container*
EMOTION_WAV_BASE = os.getenv("EMOTION_WAV_BASE", str(PIPELINE_DIR / "EmotionAudio"))
EMOTION_WAV_MAP: dict[str, str] = {
    "Neutral":       f"{EMOTION_WAV_BASE}/neutral_sample.wav",
    "Happy":         f"{EMOTION_WAV_BASE}/happy_sample.wav",
    "Sad":           f"{EMOTION_WAV_BASE}/sad_sample.wav",
    "Tense/Fearful": f"{EMOTION_WAV_BASE}/tense_fearful_sample.wav",
    "Angry":         f"{EMOTION_WAV_BASE}/angry_sample.wav",
    "Surprised":     f"{EMOTION_WAV_BASE}/surprised_sample.wav",
    "Disgusted":     f"{EMOTION_WAV_BASE}/disgusted_sample.wav",
    "Loving/Tender": f"{EMOTION_WAV_BASE}/loving_tender_sample.wav",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Story Narration Pipeline API")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def concatenate_wavs(wav_paths: list[str], output_path: str) -> None:
    with wave.open(wav_paths[0], "rb") as first:
        params = first.getparams()
    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        for path in wav_paths:
            with wave.open(path, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class NarrateRequest(BaseModel):
    text: str
    language: str = "en"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Pipeline API is running"}


@app.post("/narrate")
def narrate(request: NarrateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Step 1 — classify emotions
    try:
        clf_response = httpx.post(
            f"{CLASSIFIER_URL}/classify",
            json={"text": request.text},
            timeout=60,
        )
        clf_response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Classifier API error: {e}")

    sentences = clf_response.json()["sentences"]
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences returned by classifier.")

    # Step 2 — generate audio per sentence
    sentence_wavs: list[str] = []

    for i, item in enumerate(sentences):
        sentence  = item["sentence"]
        emotion   = item["emotion"]
        speaker_wav = EMOTION_WAV_MAP.get(emotion, EMOTION_WAV_MAP["Neutral"])

        print(f"[{i+1}] ({emotion}) {sentence}")

        try:
            tts_response = httpx.post(
                f"{XTTS_URL}/audio/generate",
                json={
                    "text": sentence,
                    "language": request.language,
                    "speaker_wav": speaker_wav,
                },
                headers={"x-api-key": XTTS_API_KEY},
                timeout=120,
            )
            tts_response.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"XTTS API error on sentence {i+1}: {e}")

        file_id = tts_response.json()["file_id"]
        sentence_wavs.append(os.path.join(OUTPUT_DIR, f"{file_id}.wav"))

    # Step 3 — concatenate
    final_id   = str(uuid.uuid4())
    final_path = os.path.join(OUTPUT_DIR, f"story_{final_id}.wav")
    concatenate_wavs(sentence_wavs, final_path)

    return FileResponse(final_path, media_type="audio/wav", filename="story.wav")
