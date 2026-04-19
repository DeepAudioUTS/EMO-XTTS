"""
chunker.py

Sentence boundary detection and token-aware chunk grouping for BERT input.

Two main entry points:
  - chunk_by_sentence(text) → list of clean sentence strings
  - group_into_token_chunks(sentences, tokenizer) → list of grouped strings
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import spacy

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOKENS: int = 400  # Must match train.py MAX_LEN for safety headroom

# Regex abbreviations that should NOT trigger a sentence split
# Order matters: longer patterns first
_ABBREV_PATTERN = re.compile(
    r"""
    (?:
        # Titles / honorifics
        \b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Rev|Gen|Sgt|Cpl|Pvt|Capt|Lt|Col|Maj
            |Gov|Pres|Rep|Sen|St|vs|etc|approx|dept|est|vol|no|pp|ed)\.
        |
        # US state / country abbreviations (all-caps or mixed)
        \b[A-Z]{1,4}\.(?:[A-Z]{1,4}\.)*
        |
        # Decimal numbers
        \d+\.\d+
        |
        # Single letter followed by period (initials)
        \b[A-Za-z]\.
    )
    """,
    re.VERBOSE,
)

# Sentence-ending punctuation followed by whitespace + capital or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')

# ---------------------------------------------------------------------------
# spaCy lazy loader (singleton)
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    """Lazily load the spaCy model (en_core_web_sm)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
            _nlp.add_pipe("sentencizer")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _regex_split(text: str) -> list[str]:
    """
    Split ``text`` on sentence boundaries using regex heuristics.

    Splits on [.!?] followed by whitespace + capital letter, but preserves
    common abbreviations, decimal numbers, and all-caps initialisms.

    Args:
        text: Raw input text.

    Returns:
        List of candidate sentence strings (may still contain abbreviation
        false-positives that spaCy will fix).
    """
    # Protect abbreviations by temporarily replacing their periods
    protected = text
    placeholder_map: dict[str, str] = {}

    for i, match in enumerate(_ABBREV_PATTERN.finditer(text)):
        token = match.group()
        placeholder = f"__ABBREV{i}__"
        placeholder_map[placeholder] = token
        protected = protected.replace(token, placeholder, 1)

    # Split on sentence boundaries
    parts = _SENTENCE_END.split(protected)

    # Restore abbreviations
    restored: list[str] = []
    for part in parts:
        for ph, orig in placeholder_map.items():
            part = part.replace(ph, orig)
        stripped = part.strip()
        if stripped:
            restored.append(stripped)

    return restored


def _spacy_split(text: str) -> list[str]:
    """
    Split ``text`` into sentences using spaCy's sentencizer.

    Used as a fallback / verification step for edge cases.

    Args:
        text: Raw input text.

    Returns:
        List of sentence strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _needs_spacy_fallback(sentences: list[str], original: str) -> bool:
    """
    Heuristic: decide whether regex output looks suspicious enough to
    re-run with spaCy.

    Triggers spaCy if any "sentence" is very short (likely a false split
    on an abbreviation) or if the number of splits looks excessive.

    Args:
        sentences: Candidate list from regex.
        original: Original unsplit text.

    Returns:
        True if spaCy fallback should be applied.
    """
    if not sentences:
        return True
    # Any fragment under 4 characters is suspicious
    if any(len(s) < 4 for s in sentences):
        return True
    # More than one split per 50 characters is suspicious
    if len(sentences) > max(1, len(original) // 50):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_by_sentence(text: str) -> list[str]:
    """
    Split input text into individual sentence strings.

    Strategy:
    1. Apply regex-based splitting with abbreviation protection.
    2. If result looks suspicious (very short fragments, over-splitting),
       fall back to spaCy sentencizer for the full text.
    3. Filter out empty / whitespace-only fragments.

    Args:
        text: Arbitrary input text (may contain multiple sentences).

    Returns:
        List of non-empty sentence strings.
    """
    text = text.strip()
    if not text:
        return []

    sentences = _regex_split(text)

    if _needs_spacy_fallback(sentences, text):
        sentences = _spacy_split(text)

    # Final clean-up
    return [s.strip() for s in sentences if s.strip()]


def group_into_token_chunks(
    sentences: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """
    Greedily combine consecutive sentences into chunks that fit within
    ``max_tokens`` BERT tokens (not counting special tokens [CLS]/[SEP]).

    Args:
        sentences: List of individual sentence strings.
        tokenizer:  HuggingFace tokenizer (used for token counting).
        max_tokens: Maximum token budget per chunk (default: ``MAX_TOKENS``).

    Returns:
        List of grouped text strings, each safely under ``max_tokens`` tokens.
    """
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_token_count: int = 0

    for sentence in sentences:
        # Count tokens for this sentence (exclude special tokens)
        n_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        # If a single sentence exceeds the limit, truncate it alone
        if n_tokens >= max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_token_count = 0
            chunks.append(sentence)
            continue

        if current_token_count + n_tokens > max_tokens:
            # Flush current group
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_token_count = n_tokens
        else:
            current_sentences.append(sentence)
            current_token_count += n_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Dr. Smith walked into the room. He was furious! "
        "The U.S.A. had declared war. 3.14 was mentioned briefly. "
        "She wept silently in the corner. Mr. Johnson didn't notice."
    )
    print("Input:", sample)
    print("\nSentences:")
    for i, s in enumerate(chunk_by_sentence(sample), 1):
        print(f"  {i}: {s}")
