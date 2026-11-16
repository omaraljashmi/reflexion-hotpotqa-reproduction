"""HotPotQA-style answer normalization and scoring."""

from __future__ import annotations

from collections import Counter
import re
import string


# Typographic characters and the keyboard character they stand for. Current
# models often write a non-breaking hyphen or curly quotes where the dataset
# has plain ones. Without this map exact match would mark "Kim Ha‑neul" wrong
# against "Kim Ha-neul", which is a typing difference, not a wrong answer.
_PLAIN = {ord(character): "-" for character in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad"}
_PLAIN.update({ord(character): "'" for character in "\u2018\u2019\u201a\u201b\u2032\u02bc"})
_PLAIN.update({ord(character): '"' for character in "\u201c\u201d\u201e\u201f\u2033"})
_PLAIN.update({ord(character): " " for character in "\u00a0\u2007\u2009\u200a\u202f"})
_PLAIN[0x2026] = "..."


def plain_punctuation(text: str) -> str:
    """Replace typographic hyphens, quotes and spaces with their keyboard forms."""

    return text.translate(_PLAIN)


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles, and duplicate whitespace.

    This is the HotpotQA normalization with one addition: typographic
    punctuation is first mapped to its keyboard form (see ``plain_punctuation``).
    """

    text = plain_punctuation(text).lower()
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

