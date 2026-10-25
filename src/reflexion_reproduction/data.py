"""Loading HotpotQA examples from the mini sample or the 60/20/20 split files."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterator

from .core import Example


SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "hotpotqa_splits"
SPLIT_FILES = {
    "train": (
        "train-00000-of-00002.jsonl.gz",
        "train-00001-of-00002.jsonl.gz",
    ),
    "test": ("test.jsonl.gz",),
    "validation": ("validation_unseen/validation.jsonl.gz",),
}


class SealedSplitError(RuntimeError):
    """Raised when the sealed validation split is requested without permission."""


def _normalize_context(raw: object) -> tuple[dict[str, str], ...]:
    """Accept both the mini-sample format and the official column format."""

    if isinstance(raw, dict) and "title" in raw and "sentences" in raw:
        return tuple(
            {"title": str(title), "text": "".join(sentences)}
            for title, sentences in zip(raw["title"], raw["sentences"])
        )
    if isinstance(raw, (list, tuple)):
        items = []
        for item in raw:
            if isinstance(item, dict):
                items.append({"title": str(item["title"]), "text": str(item["text"])})
            else:  # original HotpotQA list format: [title, [sentences]]
                title, sentences = item
                items.append({"title": str(title), "text": "".join(sentences)})
        return tuple(items)
    raise ValueError("Unrecognized HotpotQA context format")


def row_to_example(row: dict) -> Example:
    return Example(
        id=str(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        context=_normalize_context(row["context"]),
        qtype=str(row.get("type", "")),
        level=str(row.get("level", "")),
    )


def iter_split_rows(split: str, splits_dir: Path = SPLITS_DIR) -> Iterator[dict]:
    for name in SPLIT_FILES[split]:
        with gzip.open(splits_dir / name, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_split(
    split: str,
    n: int,
    seed: int = 0,
    allow_validation: bool = False,
    splits_dir: Path = SPLITS_DIR,
) -> list[Example]:
    """Return a seeded random sample of ``n`` examples from one split.

    Uses reservoir sampling so the 50k-row training split never sits in memory.
    The sealed validation split needs ``allow_validation=True``.
    """

    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLIT_FILES)}")
    if split == "validation" and not allow_validation:
        raise SealedSplitError(
            "The validation split is sealed for the final evaluation. "
            "Pass --final-validation only when development is finished."
        )
    if n < 1:
        raise ValueError("Sample size must be at least 1")

    rng = random.Random(seed)
    reservoir: list[dict] = []
    for index, row in enumerate(iter_split_rows(split, splits_dir)):
        if index < n:
            reservoir.append(row)
        else:
            slot = rng.randint(0, index)
            if slot < n:
                reservoir[slot] = row
    reservoir.sort(key=lambda row: row["id"])  # stable order independent of sampling
    rng.shuffle(reservoir)
    return [row_to_example(row) for row in reservoir]


def load_mini(path: Path, limit: int) -> list[Example]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    return [row_to_example(row) for row in rows[:limit]]
