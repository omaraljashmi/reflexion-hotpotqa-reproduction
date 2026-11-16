"""Experiment runner: conditions, per-attempt CSV logging, and analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable, Sequence

from .core import Example, Strategy, run_experiment
from .failure_memory import BuildLog, MemoryBank, Retriever, classify_failure
from .parallel import parallel_map


@dataclass(frozen=True)
class Condition:
    """One row of the comparison: how the agent uses memory."""

    strategy: Strategy  # BASELINE: fresh prompt every attempt. REFLEXION: reflect and retry.
    retrieval: str | None = None  # how lessons from other questions are chosen, if any
    pool: str = "validated"  # which lessons may be chosen (see MemoryBank.pool)
    label: str = ""  # name used in tables
    lessons_from: int = 1  # first attempt on which the lessons are shown


CONDITIONS: dict[str, Condition] = {
    # The four conditions of the planned experiment.
    "no_memory": Condition(Strategy.BASELINE, label="No memory"),
    "reflexion": Condition(Strategy.REFLEXION, label="Reflexion (parent paper)"),
    "failure_memory": Condition(
        Strategy.REFLEXION, "failure_type", label="Lessons by failure type (proposed)"
    ),
    "random_memory": Condition(Strategy.REFLEXION, "random", label="Random lessons"),
    # Other techniques tried against the proposed one. The three rows above
    # and the next one draw on validated lessons only.
    "similarity_memory": Condition(
        Strategy.REFLEXION, "similarity", label="Lessons from similar questions"
    ),
    "retry_memory": Condition(
        Strategy.REFLEXION, "failure_type", pool="retry", label="Lessons checked by retry only"
    ),
    "unvalidated_memory": Condition(
        Strategy.REFLEXION, "failure_type", pool="unvalidated", label="Unchecked lessons"
    ),
    # The proposed lessons, held back until the agent has failed once.
    "late_memory": Condition(
        Strategy.REFLEXION, "failure_type", label="Lessons after a failed attempt", lessons_from=2
    ),
}

CSV_FIELDS = [
    "phase", "condition", "run_id", "model", "example_id", "qtype", "level", "trial",
    "prediction", "reference", "exact_match", "f1", "failure_type",
    "failure_label_source", "lesson_ids", "n_lessons", "reflection",
    "prompt_tokens", "completion_tokens", "seconds", "reflection_tokens",
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
            "n_lessons": 1 if log.trial in ("validation", "dev_check") else 0,
            "reflection": log.reflection,
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
    workers: int = 1,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[dict]:
    """Run one experimental condition and return one CSV row per attempt."""

    setup = CONDITIONS[condition]
    lessons_for = None
    if setup.retrieval:
        if bank is None or not bank.pool(setup.pool):
            raise ValueError(f"{condition} needs a memory bank with {setup.pool} lessons")
        retriever = Retriever(bank, setup.retrieval, k=k, seed=seed + run_id, pool=setup.pool)
        lessons_for = retriever.lessons_for(lesson_content)

    result = run_experiment(
        examples, model, setup.strategy, max_trials,
        lessons_for=lessons_for, workers=workers, progress=progress,
        lessons_from_trial=setup.lessons_from,
    )
    by_id = {example.id: example for example in examples}

    def failure_label(record) -> tuple[str, str]:
        if record.exact_match or label == "none":
            return "", ""
        trace = f"Thought: {record.thought}\nFinish[{record.prediction}]\nObservation: Answer is INCORRECT"
        return classify_failure(
            model, by_id[record.example_id], record.prediction, trace, use_llm=(label == "llm")
        )

    labels = parallel_map(failure_label, result.records, workers)
    rows = []
    for record, (failure_type, source) in zip(result.records, labels):
        example = by_id[record.example_id]
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
                "seconds": round(record.seconds, 3),
                "reflection_tokens": record.reflection_tokens,
            }
        )
    return rows


# ---------------------------------------------------------------- analysis


def _passed(row: dict) -> bool:
    return row["exact_match"] in ("1", 1, True)


def _number(row: dict, field: str) -> float:
    return float(row.get(field) or 0)


def cumulative_accuracy(
    rows: Sequence[dict], condition: str, max_trials: int, run_id: int = 0
) -> list[float]:
    """Share of examples solved by the end of each attempt, as in Fig. 4 of the paper.

    An example counts as solved at attempt ``t`` if any of its attempts up to
    ``t`` was an exact match. Examples that passed early have no later rows,
    so the curve never decreases.
    """

    solved_at: dict[str, int] = {}
    examples: set[str] = set()
    for row in rows:
        if row["phase"] != "eval" or row["condition"] != condition:
            continue
        if str(row["run_id"]) != str(run_id):
            continue
        example_id = row["example_id"]
        examples.add(example_id)
        if _passed(row):
            trial = int(row["trial"])
            solved_at[example_id] = min(trial, solved_at.get(example_id, trial))
    if not examples:
        return [0.0] * max_trials
    return [
        sum(1 for t in solved_at.values() if t <= trial) / len(examples)
        for trial in range(1, max_trials + 1)
    ]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts b and c."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a share (Wilson score interval)."""

    if n == 0:
        return 0.0, 0.0
    share = successes / n
    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Outcome:
    first: bool
    final: bool
    first_f1: float
    lesson_ids: tuple[str, ...]
    first_failure: str
    qtype: str = ""


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
            first=_passed(first),
            final=any(_passed(row) for row in attempts),
            first_f1=float(first["f1"] or 0),
            lesson_ids=tuple(filter(None, str(first["lesson_ids"]).split(";"))),
            first_failure=first["failure_type"],
            qtype=first.get("qtype", ""),
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
        "mcnemar_p": round(mcnemar_exact(b, c), 6),
    }


def paired_f1(
    data: dict[tuple[str, str, str], Outcome], reference: str, condition: str,
    resamples: int = 2000, seed: int = 0,
) -> dict:
    """Mean change in first-attempt F1 on the same questions, with a bootstrap 95% interval."""

    ref = {(run, ex): o.first_f1 for (cond, run, ex), o in data.items() if cond == reference}
    cur = {(run, ex): o.first_f1 for (cond, run, ex), o in data.items() if cond == condition}
    differences = [cur[key] - ref[key] for key in sorted(set(ref) & set(cur))]
    n = len(differences)
    if n == 0:
        return {"reference": reference, "condition": condition, "pairs": 0,
                "f1_change": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(resamples))
    return {
        "reference": reference,
        "condition": condition,
        "pairs": n,
        "f1_change": round(sum(differences) / n, 4),
        "ci95": [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)],
    }


def _run_order(run_id: str) -> tuple[int, str]:
    return (int(run_id), "") if run_id.lstrip("-").isdigit() else (10**9, run_id)


def analyze(rows: Sequence[dict], bank: MemoryBank | None = None) -> dict:
    """Summarize an attempts table.

    When the table holds repeated runs, the summary and the paired tests use
    the first run, so every question counts once. ``by_run`` lists every run
    to show how stable the numbers are. ``bank`` adds a breakdown by the
    failure type each lesson teaches.
    """

    everything = outcomes(rows)
    runs = sorted({run for _, run, _ in everything}, key=_run_order)
    primary = runs[0] if runs else "0"
    data = {key: value for key, value in everything.items() if key[1] == primary}
    conditions = sorted({cond for cond, _, _ in data}, key=list(CONDITIONS).index)
    eval_rows = [row for row in rows if row["phase"] == "eval" and str(row["run_id"]) == primary]
    max_trial = max((int(row["trial"]) for row in eval_rows), default=1)

    def first_attempt_mean(cond: str, field: str) -> float:
        values = [
            _number(row, field) for row in eval_rows
            if row["condition"] == cond and str(row["trial"]) == "1"
        ]
        return sum(values) / len(values) if values else 0.0

    summary = []
    for cond in conditions:
        items = [o for (c, _, _), o in data.items() if c == cond]
        n = len(items)
        cond_rows = [row for row in eval_rows if row["condition"] == cond]
        tokens = sum(
            _number(row, "prompt_tokens") + _number(row, "completion_tokens")
            + _number(row, "reflection_tokens")
            for row in cond_rows
        )
        first_tokens = first_attempt_mean(cond, "prompt_tokens") + first_attempt_mean(
            cond, "completion_tokens"
        )
        low, high = wilson_interval(sum(o.first for o in items), n)
        summary.append(
            {
                "condition": cond,
                "label": CONDITIONS[cond].label,
                "examples": n,
                "first_attempt_em": round(sum(o.first for o in items) / n, 6) if n else 0,
                "first_attempt_em_ci95": [round(low, 6), round(high, 6)],
                "first_attempt_f1": round(sum(o.first_f1 for o in items) / n, 6) if n else 0,
                "final_em": round(sum(o.final for o in items) / n, 6) if n else 0,
                "attempts": max_trial,
                "lessons_per_question": round(first_attempt_mean(cond, "n_lessons"), 2),
                "lessons_from": CONDITIONS[cond].lessons_from if CONDITIONS[cond].retrieval else None,
                "first_attempt_tokens": round(first_tokens, 3),
                "first_attempt_seconds": round(first_attempt_mean(cond, "seconds"), 6),
                "tokens_per_example": round(tokens / n, 1) if n else 0,
            }
        )
    base_tokens = next(
        (row["first_attempt_tokens"] for row in summary if row["condition"] == "no_memory"), None
    )
    for row in summary:
        row["added_first_attempt_tokens"] = (
            round(row["first_attempt_tokens"] - base_tokens, 3) if base_tokens is not None else None
        )

    comparisons = []
    pairs_to_check = [
        ("no_memory", c) for c in conditions if c != "no_memory"
    ] + [
        ("reflexion", c) for c in conditions if CONDITIONS[c].retrieval
    ] + [
        (c, "failure_memory") for c in conditions
        if CONDITIONS[c].retrieval and c != "failure_memory"
    ]
    seen = set()
    for ref, cond in pairs_to_check:
        if ref in conditions and cond in conditions and (ref, cond) not in seen:
            seen.add((ref, cond))
            for field in ("first", "final"):
                comparisons.append(paired(data, ref, cond, field))

    f1_changes = [paired_f1(data, "no_memory", c) for c in conditions if c != "no_memory"]
    if "no_memory" not in conditions:
        f1_changes = []

    by_run = []
    for run in runs:
        for cond in sorted({c for c, r, _ in everything if r == run}, key=list(CONDITIONS).index):
            items = [o for (c, r, _), o in everything.items() if c == cond and r == run]
            by_run.append(
                {
                    "run_id": run,
                    "condition": cond,
                    "examples": len(items),
                    "first_attempt_em": round(sum(o.first for o in items) / len(items), 6),
                    "final_em": round(sum(o.final for o in items) / len(items), 6),
                }
            )

    failure_counts = {
        cond: dict(Counter(o.first_failure or "UNLABELED" for (c, _, _), o in data.items() if c == cond and not o.first))
        for cond in conditions
    }

    # Transfer on the first attempt, against the no-memory answer to the same question.
    base = {(run, ex): o for (c, run, ex), o in data.items() if c == "no_memory"}

    def effect(run: str, ex: str, outcome: Outcome) -> str | None:
        if (run, ex) not in base:
            return None
        before, after = base[(run, ex)].first, outcome.first
        return "helped" if (not before and after) else "hurt" if (before and not after) else "neutral"

    by_question_type: dict[str, dict] = {}
    for cond in conditions:
        groups: dict[str, Counter] = defaultdict(Counter)
        for (c, run, ex), o in data.items():
            if c != cond:
                continue
            group = groups[o.qtype or "unknown"]
            group["questions"] += 1
            group["correct"] += o.first
            change = effect(run, ex, o)
            group["fixed"] += change == "helped"
            group["broken"] += change == "hurt"
        by_question_type[cond] = {
            qtype: {
                "questions": g["questions"],
                "first_attempt_em": round(g["correct"] / g["questions"], 4),
                "fixed": g["fixed"],
                "broken": g["broken"],
            }
            for qtype, g in sorted(groups.items())
        }

    lesson_effects: dict[str, Counter] = defaultdict(Counter)
    taught: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    lesson_type = {lesson.id: lesson.failure_type for lesson in bank.lessons} if bank else {}
    for (cond, run, ex), o in data.items():
        change = effect(run, ex, o)
        if CONDITIONS[cond].retrieval is None or change is None:
            continue
        for lesson_id in o.lesson_ids:
            lesson_effects[f"{cond}:{lesson_id}"][change] += 1
        for failure_type in sorted({lesson_type[i] for i in o.lesson_ids if i in lesson_type}):
            group = taught[cond][failure_type]
            group["questions"] += 1
            group["fixed"] += change == "helped"
            group["broken"] += change == "hurt"

    # Repeated mistakes: consecutive failed trials with the same failure type.
    repeats = Counter()
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in eval_rows:
        if not _passed(row) and row["failure_type"]:
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

    models = Counter(row.get("model", "") for row in eval_rows)
    return {
        "model": models.most_common(1)[0][0] if models else "",
        "runs": runs,
        "primary_run": primary,
        "summary": summary,
        "comparisons": comparisons,
        "first_attempt_f1_changes": f1_changes,
        "by_run": by_run,
        "by_question_type": by_question_type,
        "by_lesson_failure_type": {
            cond: {name: dict(group) for name, group in sorted(groups.items())}
            for cond, groups in taught.items()
        },
        "first_attempt_failure_types": failure_counts,
        "lesson_effects": {key: dict(value) for key, value in sorted(lesson_effects.items())},
        "repeated_mistakes": repeated,
    }


def _comparison(report: dict, reference: str, condition: str, metric: str = "first") -> dict | None:
    return next(
        (
            comp for comp in report["comparisons"]
            if comp["reference"] == reference and comp["condition"] == condition
            and comp["metric"] == metric
        ),
        None,
    )


def comparison_table(report: dict) -> list[dict]:
    """One row per condition: the comparison table of the experiment.

    "Fixed" and "broken" count first-attempt answers that changed against the
    no-memory answer to the same question. The p-value is the exact McNemar
    test on those two counts.
    """

    table = []
    for row in report["summary"]:
        comp = _comparison(report, "no_memory", row["condition"])
        low, high = row["first_attempt_em_ci95"]
        table.append(
            {
                "Method": row["label"],
                "Lessons": "on retries" if row.get("lessons_from", 1) == 2 else f"{row['lessons_per_question']:g}",
                "First attempt EM": f"{row['first_attempt_em']:.1%}",
                "95% CI": f"{low:.1%} to {high:.1%}",
                "First attempt F1": f"{row['first_attempt_f1']:.3f}",
                "Fixed": str(comp["fail_to_pass"]) if comp else "",
                "Broken": str(comp["pass_to_fail"]) if comp else "",
                "p": ("<0.001" if comp["mcnemar_p"] < 0.001 else f"{comp['mcnemar_p']:.3f}") if comp else "",
                f"EM after {row['attempts']} attempts": f"{row['final_em']:.1%}",
                "Added tokens": (
                    f"{round(row['added_first_attempt_tokens']) + 0:+d}"
                    if row["added_first_attempt_tokens"] is not None else ""
                ),
                "Seconds": f"{row['first_attempt_seconds']:.2f}",
            }
        )
    return table


def markdown_table(table: Sequence[dict]) -> str:
    if not table:
        return ""
    columns = list(table[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in table]
    return "\n".join(lines)


def repeat_table(report: dict) -> list[dict]:
    """First-attempt EM of every run, one column per run."""

    runs = report["runs"]
    table = []
    for row in report["summary"]:
        line = {"Method": row["label"]}
        for index, run in enumerate(runs, start=1):
            match = next(
                (r for r in report["by_run"] if r["run_id"] == run and r["condition"] == row["condition"]),
                None,
            )
            line[f"Run {index}"] = f"{match['first_attempt_em']:.1%}" if match else ""
        table.append(line)
    return table


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
            f"p={comp['mcnemar_p']:.4f}"
        )
    echo("\nFirst-attempt failure types")
    for cond, counts in report["first_attempt_failure_types"].items():
        top = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        echo(f"  {cond:<18} {top or '(none)'}")
    if len(report.get("runs", [])) > 1:
        echo("\nFirst-attempt EM by run")
        echo(markdown_table(repeat_table(report)))
    echo("\nComparison table")
    echo(markdown_table(comparison_table(report)))


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def save_comparison(out_dir: Path, report: dict, title: str = "", suffix: str = "") -> dict[str, Path]:
    """Write the comparison table as Markdown and CSV next to the report."""

    out_dir.mkdir(parents=True, exist_ok=True)
    table = comparison_table(report)
    paths = {
        "table": out_dir / f"comparison_table{suffix}.md",
        "table_csv": out_dir / f"comparison_table{suffix}.csv",
    }
    parts = [f"# {title}" if title else "# Comparison table", "", markdown_table(table), ""]
    parts += [
        "Fixed and broken count first attempt answers that changed against the no memory",
        "answer to the same question. p is the exact McNemar test on those two counts.",
        "Added tokens and seconds are per question on the first attempt.",
        "",
    ]
    if len(report.get("runs", [])) > 1:
        parts += [
            "## First attempt EM by run", "",
            markdown_table(repeat_table(report)), "",
            "The table above uses run 1. Later runs repeat the same questions.", "",
        ]
    paths["table"].write_text("\n".join(parts), encoding="utf-8")
    with paths["table_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["Method"])
        writer.writeheader()
        writer.writerows(table)
    return paths
