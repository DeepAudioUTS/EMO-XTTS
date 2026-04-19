"""
data_pipeline.py

Downloads GoEmotions (simplified config) from HuggingFace Datasets,
maps the 28 original labels to 8 target emotion classes, filters
ambiguous samples, and returns train/val/test splits.

Add new emotion classes by editing LABEL_MAPPING and EMOTION_LABELS below.
"""

from __future__ import annotations

from datasets import DatasetDict, concatenate_datasets, load_dataset

# ---------------------------------------------------------------------------
# Constants – edit these to add / rearrange emotion classes
# ---------------------------------------------------------------------------

NUM_LABELS: int = 8

# Maps each GoEmotions label string → integer class index (0-7).
# To add a new class: add a new index, update NUM_LABELS, and map labels here.
LABEL_MAPPING: dict[str, int] = {
    # 0 – Neutral
    "neutral": 0,
    # 1 – Happy
    "joy": 1,
    "amusement": 1,
    "excitement": 1,
    "optimism": 1,
    "relief": 1,
    "pride": 1,
    "gratitude": 1,
    # 2 – Sad
    "sadness": 2,
    "grief": 2,
    "disappointment": 2,
    "remorse": 2,
    # 3 – Tense/Fearful
    "fear": 3,
    "nervousness": 3,
    # 4 – Angry
    "anger": 4,
    "annoyance": 4,
    "disapproval": 4,
    # 5 – Surprised
    "surprise": 5,
    "realization": 5,
    # 6 – Disgusted
    "disgust": 6,
    "contempt": 6,
    # 7 – Loving/Tender
    "love": 7,
    "caring": 7,
    "admiration": 7,
    "desire": 7,
    # GoEmotions labels not mapped to any of our 8 classes are marked -1
    # and will be filtered out.
    "approval": -1,
    "confusion": -1,
    "curiosity": -1,
    "embarrassment": -1,
}

EMOTION_LABELS: dict[int, str] = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Tense/Fearful",
    4: "Angry",
    5: "Surprised",
    6: "Disgusted",
    7: "Loving/Tender",
}

DATASET_NAME: str = "google-research-datasets/go_emotions"
DATASET_CONFIG: str = "simplified"
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
# Test ratio is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.1

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_label_names(dataset_split) -> list[str]:
    """Return the list of GoEmotions label name strings from dataset features."""
    return dataset_split.features["labels"].feature.names


def map_labels(examples: dict, label_names: list[str]) -> dict:
    """
    Map a batch of GoEmotions 28-class label lists to 8-class integer labels.

    Samples with:
    - no labels → filtered out (label = -1)
    - labels that all map to the same 8-class bucket → kept
    - labels mapping to *different* 8-class buckets (ambiguous) → filtered (-1)
    - any label not in LABEL_MAPPING → filtered (-1)

    Args:
        examples: Batch dict from HuggingFace Datasets map().
        label_names: List of GoEmotions label name strings (from dataset features).

    Returns:
        Dict with added key ``"label"`` (int, -1 means discard).
    """
    mapped: list[int] = []
    for label_ids in examples["labels"]:
        if not label_ids:
            mapped.append(-1)
            continue

        target_classes: set[int] = set()
        for lid in label_ids:
            name = label_names[lid]
            cls = LABEL_MAPPING.get(name, -1)
            target_classes.add(cls)

        # Ambiguous: multiple distinct non-negative target classes
        non_neg = target_classes - {-1}
        if len(non_neg) != 1:
            mapped.append(-1)
        else:
            mapped.append(non_neg.pop())

    examples["label"] = mapped
    return examples


def load_and_prepare_dataset() -> DatasetDict:
    """
    Download GoEmotions (simplified), map to 8 classes, filter ambiguous
    samples, and return an 80/10/10 train/val/test DatasetDict.

    Returns:
        DatasetDict with keys ``"train"``, ``"validation"``, ``"test"``.
    """
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    raw: DatasetDict = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Collect all splits into one pool then re-split for clean 80/10/10
    # (GoEmotions simplified ships train/validation/test but proportions differ)
    splits_to_merge = [raw["train"]]
    for split in ("validation", "test"):
        if split in raw:
            splits_to_merge.append(raw[split])
    combined = concatenate_datasets(splits_to_merge)

    label_names: list[str] = combined.features["labels"].feature.names
    print(f"Original label names ({len(label_names)}): {label_names}")

    # Map to 8-class labels
    combined = combined.map(
        lambda batch: map_labels(batch, label_names),
        batched=True,
        desc="Mapping labels",
    )

    # Filter out ambiguous / unmapped samples
    before = len(combined)
    combined = combined.filter(lambda x: x["label"] != -1, desc="Filtering")
    after = len(combined)
    print(f"Kept {after}/{before} samples after filtering ambiguous labels.")

    # Keep only text + label columns
    combined = combined.remove_columns(
        [c for c in combined.column_names if c not in ("text", "label")]
    )

    # 80/10/10 split
    train_val_test = combined.train_test_split(test_size=1 - TRAIN_RATIO, seed=42)
    val_test = train_val_test["test"].train_test_split(
        test_size=VAL_RATIO / (1 - TRAIN_RATIO), seed=42
    )

    dataset = DatasetDict(
        {
            "train": train_val_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    # Print class distribution
    from collections import Counter

    for split_name, split_ds in dataset.items():
        counter = Counter(split_ds["label"])
        dist = {EMOTION_LABELS[k]: v for k, v in sorted(counter.items())}
        print(f"{split_name}: {len(split_ds)} samples | {dist}")

    return dataset


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = load_and_prepare_dataset()
    print("\nDataset ready:")
    print(ds)
