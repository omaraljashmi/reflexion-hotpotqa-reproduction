"""HotPotQA-style answer normalization and scoring."""

from __future__ import annotations

from collections import Counter
import re
import string


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles, and duplicate whitespace."""

    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0

    precision = shared / len(predicted_tokens)
    recall = shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)

