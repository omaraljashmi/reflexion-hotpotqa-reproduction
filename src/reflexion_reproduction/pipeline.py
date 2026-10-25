"""Experiment runner: conditions, per-attempt CSV logging, and analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .core import Example, Strategy, run_experiment
from .failure_memory import BuildLog, MemoryBank, Retriever, classify_failure


# condition name -> (Reflexion strategy, retrieval mode or None)
CONDITIONS: dict[str, tuple[Strategy, str | None]] = {
    "no_memory": (Strategy.BASELINE, None),
    "reflexion": (Strategy.REFLEXION, None),
    "failure_memory": (Strategy.REFLEXION, "failure_type"),
    "random_memory": (Strategy.REFLEXION, "random"),
    "similarity_memory": (Strategy.REFLEXION, "similarity"),
}

CSV_FIELDS = [
    "phase", "condition", "run_id", "model", "example_id", "qtype", "level", "trial",
    "prediction", "reference", "exact_match", "f1", "failure_type",
    "failure_label_source", "lesson_ids", "n_lessons", "reflection",
    "prompt_tokens", "completion_tokens",
]


def write_rows(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return len(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rows(logs: Sequence[BuildLog], model_name: str) -> list[dict]:
    return [
        {
            "phase": "build",
            "condition": "build",
            "run_id": 0,
            "model": model_name,
            "example_id": log.example.id,
            "qtype": log.example.qtype,
            "level": log.example.level,
            "trial": log.trial,
            "prediction": log.output.answer,
            "reference": log.example.answer,
            "exact_match": int(log.correct),
            "f1": round(log.f1, 4),
            "failure_type": log.failure_type,
            "failure_label_source": log.failure_label_source,
            "lesson_ids": log.lesson_id,
            "n_lessons": 1 if log.trial == "validation" else 0,
            "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens,
        }
        for log in logs
    ]


def run_condition(
    condition: str,
    examples: Sequence[Example],
    model,
    max_trials: int,
    bank: MemoryBank | None = None,
    k: int = 3,
    lesson_content: str = "fix",
    seed: int = 0,
    run_id: int = 0,
    label: str = "llm",
) -> list[dict]:
    """Run one experimental condition and return one CSV row per attempt."""

    strategy, retrieval = CONDITIONS[condition]
    lessons_for = None
    if retrieval:
        if bank is None or not bank.validated():
            raise ValueError(f"{condition} needs a memory bank with validated lessons")
        lessons_for = Retriever(bank, retrieval, k=k, seed=seed + run_id).lessons_for(lesson_content)

    result = run_experiment(examples, model, strategy, max_trials, lessons_for=lessons_for)
    by_id = {example.id: example for example in examples}
    rows = []
    for record in result.records:
        example = by_id[record.example_id]
        failure_type, source = "", ""
        if not record.exact_match and label != "none":
            trace = f"Thought: {record.thought}\nFinish[{record.prediction}]\nObservation: Answer is INCORRECT"
            failure_type, source = classify_failure(
                model, example, record.prediction, trace, use_llm=(label == "llm")
            )
        rows.append(
            {
                "phase": "eval",
                "condition": condition,
                "run_id": run_id,
                "model": model.name,
                "example_id": record.example_id,
                "qtype": example.qtype,
                "level": example.level,
                "trial": record.trial,
                "prediction": record.prediction,
                "reference": record.reference,
                "exact_match": int(record.exact_match),
                "f1": round(record.f1, 4),
                "failure_type": failure_type,
                "failure_label_source": source,
                "lesson_ids": ";".join(record.lesson_ids),
                "n_lessons": len(record.lesson_ids),
                "reflection": record.reflection_added or "",
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
            }
        )
    return rows


# ---------------------------------------------------------------- analysis


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts b and c."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class Outcome:
    first: bool
    final: bool
    first_f1: float
    lesson_ids: tuple[str, ...]
    first_failure: str


def outcomes(rows: Sequence[dict]) -> dict[tuple[str, str, str], Outcome]:
    """Key: (condition, run_id, example_id)."""

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "eval":
            grouped[(row["condition"], str(row["run_id"]), row["example_id"])].append(row)
    result = {}
    for key, attempts in grouped.items():
        attempts.sort(key=lambda row: int(row["trial"]))
        first = attempts[0]
        result[key] = Outcome(
            first=first["exact_match"] in ("1", 1, True),
            final=any(row["exact_match"] in ("1", 1, True) for row in attempts),
            first_f1=float(first["f1"] or 0),
            lesson_ids=tuple(filter(None, str(first["lesson_ids"]).split(";"))),
            first_failure=first["failure_type"],
        )
    return result


def paired(
    data: dict[tuple[str, str, str], Outcome],
    reference: str,
    condition: str,
    field: str,
) -> dict:
    ref = {(run, ex): o for (cond, run, ex), o in data.items() if cond == reference}
    cur = {(run, ex): o for (cond, run, ex), o in data.items() if cond == condition}
    keys = sorted(set(ref) & set(cur))
    b = sum(getattr(ref[k], field) and not getattr(cur[k], field) for k in keys)
    c = sum(not getattr(ref[k], field) and getattr(cur[k], field) for k in keys)
    return {
        "reference": reference,
        "condition": condition,
        "metric": field,
        "pairs": len(keys),
        "fail_to_pass": c,
        "pass_to_fail": b,
        "net_gain": c - b,
        "mcnemar_p": round(mcnemar_exact(b, c), 4),
    }


def analyze(rows: Sequence[dict]) -> dict:
    data = outcomes(rows)
    conditions = sorted({cond for cond, _, _ in data}, key=list(CONDITIONS).index)

    summary = []
    for cond in conditions:
        items = [o for (c, _, _), o in data.items() if c == cond]
        n = len(items)
        tokens = [
            int(row["prompt_tokens"] or 0) + int(row["completion_tokens"] or 0)
            for row in rows
            if row["phase"] == "eval" and row["condition"] == cond
        ]
        summary.append(
            {
                "condition": cond,
                "examples": n,
                "first_attempt_em": round(sum(o.first for o in items) / n, 4) if n else 0,
                "final_em": round(sum(o.final for o in items) / n, 4) if n else 0,
                "first_attempt_f1": round(sum(o.first_f1 for o in items) / n, 4) if n else 0,
                "tokens_per_example": round(sum(tokens) / n, 1) if n else 0,
            }
        )

    comparisons = []
    pairs_to_check = [
        ("no_memory", c) for c in conditions if c != "no_memory"
    ] + [
        ("reflexion", c) for c in conditions if CONDITIONS[c][1]
    ] + [
        ("random_memory", "failure_memory"),
        ("similarity_memory", "failure_memory"),
    ]
    seen = set()
    for ref, cond in pairs_to_check:
        if ref in conditions and cond in conditions and (ref, cond) not in seen:
            seen.add((ref, cond))
            for field in ("first", "final"):
                comparisons.append(paired(data, ref, cond, field))

    failure_counts = {
        cond: dict(Counter(o.first_failure or "UNLABELED" for (c, _, _), o in data.items() if c == cond and not o.first))
        for cond in conditions
    }

    # Lesson-level transfer vs the no-memory first attempt on the same example.
    lesson_effects: dict[str, Counter] = defaultdict(Counter)
    base = {(run, ex): o for (c, run, ex), o in data.items() if c == "no_memory"}
    for (cond, run, ex), o in data.items():
        if CONDITIONS[cond][1] is None or (run, ex) not in base:
            continue
        before, after = base[(run, ex)].first, o.first
        effect = "helped" if (not before and after) else "hurt" if (before and not after) else "neutral"
        for lesson_id in o.lesson_ids:
            lesson_effects[f"{cond}:{lesson_id}"][effect] += 1

    # Repeated mistakes: consecutive failed trials with the same failure type.
    repeats = Counter()
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "eval" and row["exact_match"] in ("0", 0, False) and row["failure_type"]:
            grouped[(row["condition"], str(row["run_id"]), row["example_id"])].append(row)
    for (cond, _, _), failed in grouped.items():
        failed.sort(key=lambda row: int(row["trial"]))
        for prev, nxt in zip(failed, failed[1:]):
            repeats[(cond, "transitions")] += 1
            repeats[(cond, "same_type")] += prev["failure_type"] == nxt["failure_type"]
    repeated = {
        cond: {
            "consecutive_failures": repeats[(cond, "transitions")],
            "same_failure_type": repeats[(cond, "same_type")],
        }
        for cond in conditions
    }

    return {
        "summary": summary,
        "comparisons": comparisons,
        "first_attempt_failure_types": failure_counts,
        "lesson_effects": {key: dict(value) for key, value in sorted(lesson_effects.items())},
        "repeated_mistakes": repeated,
    }


def print_analysis(report: dict, echo: Callable[[str], None] = print) -> None:
    echo("\nCondition            n   first EM   final EM   first F1   tokens/ex")
    for row in report["summary"]:
        echo(
            f"{row['condition']:<18} {row['examples']:>4}   {row['first_attempt_em']:>7.1%}"
            f"   {row['final_em']:>7.1%}   {row['first_attempt_f1']:>7.3f}   {row['tokens_per_example']:>8}"
        )
    echo("\nPaired comparisons (McNemar exact test)")
    for comp in report["comparisons"]:
        echo(
            f"  {comp['condition']:<18} vs {comp['reference']:<18} [{comp['metric']:<5}] "
            f"+{comp['fail_to_pass']} / -{comp['pass_to_fail']}  net {comp['net_gain']:+d}  "
            f"p={comp['mcnemar_p']}"
        )
    echo("\nFirst-attempt failure types")
    for cond, counts in report["first_attempt_failure_types"].items():
        top = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        echo(f"  {cond:<18} {top or '(none)'}")


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
