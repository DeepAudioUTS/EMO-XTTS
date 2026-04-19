"""
inference.py

Inference utilities for the trained 8-class emotion classifier.

Functions:
  - load_model(model_path)             → (model, tokenizer)
  - highlight_emotion_spans(text, ...) → list of token-highlight dicts
  - predict(text, model, tokenizer)    → single-chunk prediction dict
  - chunk_and_predict(text, ...)       → multi-chunk prediction dict

CLI usage:
    python inference.py --text "She wept when she heard the news"
    python inference.py --file path/to/script.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chunker import chunk_by_sentence
from data_pipeline import EMOTION_LABELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH: str = "./emotion_model"
TOP_K_SPANS: int = 5          # Number of token spans to highlight


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> tuple[Any, Any]:
    """
    Load a fine-tuned emotion classifier from a saved checkpoint directory.

    Args:
        model_path: Path to the directory saved by ``train.py`` (contains
                    ``config.json``, ``model.safetensors``, and tokenizer files).

    Returns:
        Tuple of ``(model, tokenizer)`` ready for inference.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: If the model files cannot be loaded.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist. "
            "Run train.py first to create a checkpoint."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), output_attentions=True
        )
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded from '{model_path}' on {device}.")
        return model, tokenizer
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from '{model_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Attention-based span highlighting
# ---------------------------------------------------------------------------


def highlight_emotion_spans(
    text: str,
    model: Any,
    tokenizer: Any,
    top_k: int = TOP_K_SPANS,
) -> list[dict[str, Any]]:
    """
    Identify the top-K tokens that most influenced the model's prediction
    using the mean attention weight across all heads in the last encoder layer.

    Special tokens ([CLS], [SEP], [PAD]) are excluded from results.

    Args:
        text:      Input text (single chunk, already under MAX_LEN tokens).
        model:     Loaded AutoModelForSequenceClassification with
                   ``output_attentions=True``.
        tokenizer: Matching HuggingFace tokenizer.
        top_k:     Number of top tokens to return.

    Returns:
        List of dicts, each with keys:
            ``"word"`` (str), ``"score"`` (float), ``"emotion_role"`` (str).
        Sorted by score descending.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Predicted label for the emotion_role field
    predicted_class = int(torch.argmax(outputs.logits, dim=-1).item())
    emotion_role = EMOTION_LABELS.get(predicted_class, "Unknown")

    # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
    # Use last layer, mean over heads, CLS row → attention from CLS to each token
    last_layer_attn = outputs.attentions[-1]  # (1, heads, seq, seq)
    cls_attention = last_layer_attn[0, :, 0, :]  # (heads, seq) – CLS row
    mean_attention = cls_attention.mean(dim=0).cpu().numpy()  # (seq,)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }
    input_ids = inputs["input_ids"][0].tolist()

    # Build (token, score) pairs excluding special tokens
    token_scores: list[tuple[str, float]] = []
    for idx, (tok, score) in enumerate(zip(tokens, mean_attention)):
        if input_ids[idx] not in special_ids:
            token_scores.append((tok, float(score)))

    # Sort by attention score descending, take top_k
    token_scores.sort(key=lambda x: x[1], reverse=True)
    top_tokens = token_scores[:top_k]

    # Normalise scores relative to the max so they read as [0, 1]
    max_score = top_tokens[0][1] if top_tokens else 1.0
    return [
        {
            "word": tok.replace("##", ""),  # strip BERT sub-word prefix
            "score": round(score / max_score, 4) if max_score > 0 else 0.0,
            "emotion_role": emotion_role,
        }
        for tok, score in top_tokens
    ]


# ---------------------------------------------------------------------------
# Single-chunk prediction
# ---------------------------------------------------------------------------


def predict(text: str, model: Any, tokenizer: Any) -> dict[str, Any]:
    """
    Run the emotion classifier on a single pre-chunked text segment.

    Args:
        text:      Input text, assumed to be under MAX_LEN tokens.
        model:     Loaded classification model.
        tokenizer: Matching tokenizer.

    Returns:
        Dict::

            {
              "label":         "Sad",
              "confidence":    0.94,
              "emotion_spans": [{"word": "...", "score": 0.92,
                                  "emotion_role": "Sad"}, ...]
            }
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    label = EMOTION_LABELS.get(predicted_class, "Unknown")

    spans = highlight_emotion_spans(text, model, tokenizer)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "emotion_spans": spans,
    }


# ---------------------------------------------------------------------------
# Multi-chunk prediction
# ---------------------------------------------------------------------------


def chunk_and_predict(text: str, model: Any, tokenizer: Any) -> dict[str, Any]:
    """
    Split ``text`` into sentence chunks then run ``predict()`` on each.

    The ``overall_emotion`` is the label whose *average confidence* across
    all chunks is highest (i.e. the dominant emotion of the passage).

    Args:
        text:      Arbitrary-length input text.
        model:     Loaded classification model.
        tokenizer: Matching tokenizer.

    Returns:
        Dict::

            {
              "overall_emotion":    "Angry",
              "overall_confidence": 0.89,
              "chunks": [
                {
                  "chunk":         "She walked in.",
                  "label":         "Neutral",
                  "confidence":    0.91,
                  "emotion_spans": [...]
                },
                ...
              ]
            }
    """
    sentences = chunk_by_sentence(text)
    if not sentences:
        return {
            "overall_emotion": "Neutral",
            "overall_confidence": 0.0,
            "chunks": [],
        }

    chunks = sentences

    chunk_results: list[dict[str, Any]] = []
    # Accumulate average confidence per label index
    label_confidence_sum: dict[int, float] = {i: 0.0 for i in range(len(EMOTION_LABELS))}

    for chunk_text in chunks:
        result = predict(chunk_text, model, tokenizer)
        chunk_results.append({"chunk": chunk_text, **result})

        # Find the class index for the predicted label
        for idx, name in EMOTION_LABELS.items():
            if name == result["label"]:
                label_confidence_sum[idx] += result["confidence"]
                break

    # Overall = label with highest accumulated confidence
    overall_idx = max(label_confidence_sum, key=lambda k: label_confidence_sum[k])
    overall_emotion = EMOTION_LABELS[overall_idx]
    # Average confidence for overall emotion
    overall_conf = label_confidence_sum[overall_idx] / len(chunk_results) if chunk_results else 0.0

    return {
        "overall_emotion": overall_emotion,
        "overall_confidence": round(overall_conf, 4),
        "chunks": chunk_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run emotion inference on text or a text file."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model directory (default: {DEFAULT_MODEL_PATH})",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text string to classify.")
    group.add_argument("--file", type=str, help="Path to a .txt file to classify.")
    return parser


def main() -> None:
    """CLI entry point for inference."""
    parser = _build_parser()
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.text:
        input_text = args.text
    else:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                input_text = fh.read()
        except OSError as exc:
            print(f"Error reading file '{args.file}': {exc}", file=sys.stderr)
            sys.exit(1)

    result = chunk_and_predict(input_text, model, tokenizer)
    for chunk in result["chunks"]:
        label = chunk["label"]
        print(f"<{label}>{chunk['chunk']}</{label}>")


if __name__ == "__main__":
    main()
