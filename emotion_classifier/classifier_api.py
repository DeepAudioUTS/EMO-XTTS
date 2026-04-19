"""
classifier_api.py

FastAPI service for per-sentence emotion classification.

Run (emotion_classifier conda env, port 8001):
    conda activate emotion_classifier
    uvicorn classifier_api:app --port 8001

Endpoints:
    GET  /          → health check
    POST /classify  → classify emotions per sentence
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import load_model, predict
from chunker import chunk_by_sentence

MODEL_PATH = "./emotion_model"

ml: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"], ml["tokenizer"] = load_model(MODEL_PATH)
    yield
    ml.clear()


app = FastAPI(title="Emotion Classifier API", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    text: str


class SentenceEmotion(BaseModel):
    sentence: str
    emotion: str
    confidence: float


class ClassifyResponse(BaseModel):
    sentences: list[SentenceEmotion]


@app.get("/")
def home():
    return {"message": "Emotion Classifier API is running"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    sentences = chunk_by_sentence(request.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found in input.")

    results = []
    for sentence in sentences:
        result = predict(sentence, ml["model"], ml["tokenizer"])
        results.append(SentenceEmotion(
            sentence=sentence,
            emotion=result["label"],
            confidence=result["confidence"],
        ))

    return ClassifyResponse(sentences=results)
