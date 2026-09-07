#!/usr/bin/env python3
"""Create deterministic 60/20/20 HotpotQA splits without loading all text into memory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DATASET_PAGE = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
SOURCE_URLS = [
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/train-00001-of-00002.parquet",
]
PAPER_URL = (
    "https://proceedings.neurips.cc/paper_files/paper/2023/hash/"
    "1b44b878bb782e6954cd888628510e90-Abstract-Conference.html"
)


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream objects from a top-level JSON array."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False

        while True:
            if position >= len(buffer):
                if eof:
                    return
                buffer = handle.read(chunk_size)
                position = 0
                eof = buffer == ""
                continue

            while position < len(buffer) and buffer[position].isspace():
                position += 1

            if position >= len(buffer):
                continue

            if not started:
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                started = True
                position += 1
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1

            if position >= len(buffer):
                continue
            if buffer[position] == "]":
                return

            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    position = end
                    yield item
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer = buffer[position:] + handle.read(chunk_size)
                    position = 0
                    eof = handle.tell() == path.stat().st_size


def iter_records(path: Path) -> Iterator[dict]:
    """Read the official JSON file or Hugging Face Parquet shard(s)."""
    if path.is_dir():
        parquet_paths = sorted(path.glob("train-*.parquet"))
        if not parquet_paths:
            raise ValueError(f"No train-*.parquet files found in {path}")
        for parquet_path in parquet_paths:
            yield from iter_records(parquet_path)
        return

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("Reading Parquet input requires pyarrow") from error
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
        return

    yield from iter_json_array(path)


def record_id(item: dict) -> str:
    value = item.get("_id", item.get("id"))
    if value is None:
        raise ValueError("A HotpotQA record is missing _id/id")
    return str(value)


def stratum(item: dict) -> str:
    return f"{item.get('type', 'unknown')}|{item.get('level', 'unknown')}"


def stable_key(seed: int, item_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()


def allocate_by_stratum(
    sizes: dict[str, int], target: int, fraction: float, capacities: dict[str, int]
) -> dict[str, int]:
    allocation = {
        name: min(math.floor(size * fraction), capacities[name])
        for name, size in sizes.items()
    }
    remaining = target - sum(allocation.values())
    ranked = sorted(
        sizes,
        key=lambda name: (-(sizes[name] * fraction % 1), name),
    )
    while remaining:
        changed = False
        for name in ranked:
            if allocation[name] < capacities[name]:
                allocation[name] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise RuntimeError("Unable to allocate requested split size")
    return allocation


def file_metadata(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "records": 0,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Official JSON file, Parquet shard, or directory of train-*.parquet shards",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups: dict[str, list[tuple[bytes, str]]] = defaultdict(list)
    seen: set[str] = set()
    total = 0
    for item in iter_records(args.input):
        item_id = record_id(item)
        if item_id in seen:
            raise ValueError(f"Duplicate record id: {item_id}")
        seen.add(item_id)
        groups[stratum(item)].append((stable_key(args.seed, item_id), item_id))
        total += 1

    target_train = round(total * 0.60)
    target_test = round(total * 0.20)
    target_validation = total - target_train - target_test
    sizes = {name: len(values) for name, values in groups.items()}
    train_counts = allocate_by_stratum(sizes, target_train, 0.60, sizes)
    remaining_capacity = {
        name: sizes[name] - train_counts[name] for name in sizes
    }
    test_counts = allocate_by_stratum(
        sizes, target_test, 0.20, remaining_capacity
    )

    assignments: dict[str, str] = {}
    for name, values in groups.items():
        values.sort(key=lambda pair: pair[0])
        train_end = train_counts[name]
        test_end = train_end + test_counts[name]
        for _, item_id in values[:train_end]:
            assignments[item_id] = "train"
        for _, item_id in values[train_end:test_end]:
            assignments[item_id] = "test"
        for _, item_id in values[test_end:]:
            assignments[item_id] = "validation"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = args.output_dir / "validation_unseen"
    validation_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [
        args.output_dir / "train-00000-of-00002.jsonl.gz",
        args.output_dir / "train-00001-of-00002.jsonl.gz",
    ]
    paths = {
        "test": args.output_dir / "test.jsonl.gz",
        "validation": validation_dir / "validation.jsonl.gz",
    }
    counts = {"train": 0, "test": 0, "validation": 0}
    train_shard_counts = [0, 0]
    outputs = {
        "train_0": gzip.open(
            train_paths[0], "wt", encoding="utf-8", compresslevel=6
        ),
        "train_1": gzip.open(
            train_paths[1], "wt", encoding="utf-8", compresslevel=6
        ),
        "test": gzip.open(paths["test"], "wt", encoding="utf-8", compresslevel=6),
        "validation": gzip.open(
            paths["validation"], "wt", encoding="utf-8", compresslevel=6
        ),
    }
    first_train_shard_size = math.ceil(target_train / 2)
    try:
        for item in iter_records(args.input):
            split = assignments[record_id(item)]
            output_name = split
            if split == "train":
                shard_index = 0 if counts["train"] < first_train_shard_size else 1
                output_name = f"train_{shard_index}"
                train_shard_counts[shard_index] += 1
            outputs[output_name].write(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            counts[split] += 1
    finally:
        for output in outputs.values():
            output.close()

    expected = {
        "train": target_train,
        "test": target_test,
        "validation": target_validation,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected counts: {counts}; expected {expected}")

    train_shards = []
    for path, record_count in zip(train_paths, train_shard_counts):
        metadata = file_metadata(path)
        metadata["path"] = str(path.relative_to(args.output_dir))
        metadata["records"] = record_count
        train_shards.append(metadata)

    files = {
        "train": {
            "records": counts["train"],
            "shards": train_shards,
        }
    }
    for name, path in paths.items():
        metadata = file_metadata(path)
        metadata["path"] = str(path.relative_to(args.output_dir))
        metadata["records"] = counts[name]
        files[name] = metadata

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "HotpotQA distractor training partition",
        "dataset_page": DATASET_PAGE,
        "source_urls": SOURCE_URLS,
        "local_source": str(args.input),
        "parent_paper": PAPER_URL,
        "license": "CC BY-SA 4.0",
        "source_records": total,
        "ratios": {"train": 0.60, "test": 0.20, "validation": 0.20},
        "seed": args.seed,
        "stratified_by": ["type", "level"],
        "files": files,
        "validation_policy": (
            "Do not load validation_unseen/validation.jsonl.gz during development. "
            "Use it once after training and test work is complete."
        ),
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    notice = validation_dir / "DO_NOT_OPEN_UNTIL_FINAL_EVALUATION.txt"
    notice.write_text(
        "UNSEEN VALIDATION DATA\n\n"
        "Do not open or load validation.jsonl.gz until development and testing are complete.\n",
        encoding="utf-8",
    )

    print(json.dumps({"total": total, "counts": counts, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
