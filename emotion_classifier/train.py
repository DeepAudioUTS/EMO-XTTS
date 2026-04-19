"""
train.py

Fine-tunes bert-base-uncased on the 8-class emotion dataset produced by
data_pipeline.py.  After training it saves the best checkpoint and prints
evaluation metrics, a confusion matrix, and a training-loss curve.

Usage:
    python train.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_pipeline import EMOTION_LABELS, NUM_LABELS, load_and_prepare_dataset

# ---------------------------------------------------------------------------
# Hyperparameters – change these without editing training logic
# ---------------------------------------------------------------------------
MODEL_NAME: str = "bert-base-uncased"
OUTPUT_DIR: str = "./emotion_model"
EPOCHS: int = 4
BATCH_SIZE: int = 4
LR: float = 2e-5
WARMUP_RATIO: float = 0.1
WEIGHT_DECAY: float = 0.01
MAX_LEN: int = 400          # Must match chunker.py MAX_TOKENS
LOGGING_STEPS: int = 50
EVAL_STRATEGY: str = "epoch"
SAVE_STRATEGY: str = "epoch"
CONFUSION_MATRIX_PATH: str = "confusion_matrix.png"
LOSS_CURVE_PATH: str = "loss_curve.png"

USE_FP16: bool = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Compute macro-F1 from model logits and true labels.

    Args:
        eval_pred: NamedTuple with fields ``predictions`` (logits) and
                   ``label_ids`` (int array).

    Returns:
        Dict ``{"f1": <macro_f1>}``.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"f1": float(f1)}


def tokenize_dataset(dataset, tokenizer) -> object:
    """
    Tokenize the full DatasetDict in batches.

    Args:
        dataset:   HuggingFace DatasetDict with a ``"text"`` column.
        tokenizer: HuggingFace tokenizer compatible with the model.

    Returns:
        DatasetDict with ``input_ids``, ``attention_mask``, and ``label`` columns.
    """

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    tokenized = dataset.map(_tokenize, batched=True, desc="Tokenizing")
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized


def _save_confusion_matrix(y_true: list[int], y_pred: list[int]) -> None:
    """Save a labelled confusion matrix PNG to CONFUSION_MATRIX_PATH."""
    label_names = [EMOTION_LABELS[i] for i in range(NUM_LABELS)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Emotion Classifier – Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


def _save_loss_curve(trainer: Trainer) -> None:
    """Save training/validation loss curve to LOSS_CURVE_PATH."""
    history = trainer.state.log_history
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Train loss", color="steelblue")
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Val loss", color="tomato", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train() -> None:
    """
    Full training pipeline:
      1. Load and prepare the GoEmotions dataset.
      2. Tokenize.
      3. Fine-tune BERT with HuggingFace Trainer.
      4. Evaluate on test set, print classification report.
      5. Save confusion matrix and loss curve.
    """
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = load_and_prepare_dataset()

    # ------------------------------------------------------------------
    # Tokenizer + Model
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: EMOTION_LABELS[i] for i in range(NUM_LABELS)},
        label2id={v: k for k, v in EMOTION_LABELS.items()},
    )

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    tokenized = tokenize_dataset(dataset, tokenizer)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
        per_device_eval_batch_size=BATCH_SIZE,
            dataloader_num_workers=2,
            dataloader_pin_memory=False,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=USE_FP16,
            torch_compile=False,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=LOGGING_STEPS,
        report_to="none",         # disable W&B / HF Hub reporting
        save_total_limit=2,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== Starting training ===")
    trainer.train()

    # ------------------------------------------------------------------
    # Save best model + tokenizer
    # ------------------------------------------------------------------
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nBest model saved to {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Test-set evaluation
    # ------------------------------------------------------------------
    print("\n=== Test-set evaluation ===")
    preds_output = trainer.predict(tokenized["test"])
    y_pred = np.argmax(preds_output.predictions, axis=-1).tolist()
    y_true = list(tokenized["test"]["label"])

    label_names = [EMOTION_LABELS[i] for i in range(NUM_LABELS)]
    print(
        "\nClassification Report:\n",
        classification_report(y_true, y_pred, target_names=label_names, digits=4),
    )

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    _save_confusion_matrix(y_true, y_pred)
    _save_loss_curve(trainer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
