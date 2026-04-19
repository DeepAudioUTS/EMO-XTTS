from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
import uuid
from typing import Optional

from torch import serialization as torch_serialization

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    torch_serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig,
    ])
except Exception:
    pass

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from TTS.api import TTS

app = FastAPI(title="XTTS API")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

OUTPUT_DIR = "output"
SPEAKER_WAV = "speaker.wav"
API_KEY = "0403766117"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav: Optional[str] = None   # emotion-specific WAV; falls back to SPEAKER_WAV
    file_id: Optional[str] = None

@app.get("/")
def home():
    return {"message": "XTTS API is running", "device": device}

@app.post("/audio/generate")
def generate_tts(request: TTSRequest, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    speaker = request.speaker_wav or SPEAKER_WAV
    if not os.path.exists(speaker):
        raise HTTPException(status_code=500, detail=f"speaker wav not found: {speaker}")

    file_id = request.file_id or str(uuid.uuid4())
    output_file = os.path.join(OUTPUT_DIR, f"{file_id}.wav")

    tts.tts_to_file(
        text=request.text,
        speaker_wav=speaker,
        language=request.language,
        file_path=output_file
    )

    return {"file_id": file_id, "audio_url": f"/audio/{file_id}"}

@app.get("/audio/{file_id}")
def get_audio(file_id: str, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    audio_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{file_id}.wav")

@app.delete("/audio/{file_id}")
def delete_audio(file_id: str, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    audio_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    os.remove(audio_path)
    return {"message": f"{file_id} deleted."}
