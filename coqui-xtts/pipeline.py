"""
pipeline.py

End-to-end story narration pipeline:
  1. Split story text into sentences (emotion_classifier/chunker.py)
  2. Classify emotion per sentence (emotion_classifier/inference.py)
  3. Route each sentence to the matching emotion reference WAV
  4. Generate audio per sentence via XTTS
  5. Concatenate all sentence WAVs into one final output WAV

Usage:
    python pipeline.py --text "She wept softly. He slammed the door!"
    python pipeline.py --file story.txt --output final_story.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Patch torch.load for XTTS compatibility
# ---------------------------------------------------------------------------
from torch import serialization as torch_serialization

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    torch_serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
except Exception:
    pass

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from TTS.api import TTS

# ---------------------------------------------------------------------------
# Add emotion_classifier to path
# ---------------------------------------------------------------------------
EMOTION_CLASSIFIER_DIR = str(Path.home() / "emotion_classifier")
sys.path.insert(0, EMOTION_CLASSIFIER_DIR)

from inference import load_model, predict
from chunker import chunk_by_sentence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTION_MODEL_PATH = str(Path.home() / "emotion_classifier/emotion_model")

PIPELINE_DIR = Path(__file__).parent
EMOTION_WAV_MAP: dict[str, str] = {
    "Neutral":       str(PIPELINE_DIR / "EmotionAudio/neutral_sample.wav"),
    "Happy":         str(PIPELINE_DIR / "EmotionAudio/happy_sample.wav"),
    "Sad":           str(PIPELINE_DIR / "EmotionAudio/sad_sample.wav"),
    "Tense/Fearful": str(PIPELINE_DIR / "EmotionAudio/tense_fearful_sample.wav"),
    "Angry":         str(PIPELINE_DIR / "EmotionAudio/angry_sample.wav"),
    "Surprised":     str(PIPELINE_DIR / "EmotionAudio/surprised_sample.wav"),
    "Disgusted":     str(PIPELINE_DIR / "EmotionAudio/disgusted_sample.wav"),
    "Loving/Tender": str(PIPELINE_DIR / "EmotionAudio/loving_tender_sample.wav"),
}

OUTPUT_DIR = str(PIPELINE_DIR / "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# WAV concatenation
# ---------------------------------------------------------------------------

def concatenate_wavs(wav_paths: list[str], output_path: str) -> None:
    """Concatenate multiple WAV files into one."""
    with wave.open(wav_paths[0], "rb") as first:
        params = first.getparams()

    with wave.open(output_path, "wb") as out_wav:
        out_wav.setparams(params)
        for path in wav_paths:
            with wave.open(path, "rb") as w:
                out_wav.writeframes(w.readframes(w.getnframes()))

    print(f"Final audio saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(text: str, output_path: str = "final_story.wav") -> None:
    # Step 1 — Load emotion classifier
    print("Loading emotion classifier...")
    emotion_model, tokenizer = load_model(EMOTION_MODEL_PATH)

    # Step 2 — Load XTTS
    print("Loading XTTS model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Step 3 — Split into sentences
    sentences = chunk_by_sentence(text)
    if not sentences:
        print("No sentences found in input.")
        return

    print(f"\nFound {len(sentences)} sentence(s). Processing...\n")

    sentence_wavs: list[str] = []

    for i, sentence in enumerate(sentences):
        # Step 4 — Classify emotion
        result = predict(sentence, emotion_model, tokenizer)
        emotion = result["label"]
        confidence = result["confidence"]
        print(f"[{i+1}] ({emotion}, {confidence:.2f}) {sentence}")

        # Step 5 — Route to reference WAV
        speaker_wav = EMOTION_WAV_MAP.get(emotion, EMOTION_WAV_MAP["Neutral"])

        # Step 6 — Generate audio
        out_file = os.path.join(OUTPUT_DIR, f"sentence_{i+1:03d}.wav")
        tts.tts_to_file(
            text=sentence,
            speaker_wav=speaker_wav,
            language="en",
            file_path=out_file,
        )
        sentence_wavs.append(out_file)

    # Step 7 — Concatenate
    print("\nConcatenating audio...")
    concatenate_wavs(sentence_wavs, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion-driven story narration pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Story text string.")
    group.add_argument("--file", type=str, help="Path to a .txt story file.")
    parser.add_argument("--output", type=str, default="final_story.wav", help="Output WAV file path.")
    args = parser.parse_args()

    if args.text:
        story = args.text
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            story = f.read()

    run_pipeline(story, output_path=args.output)


if __name__ == "__main__":
    main()
