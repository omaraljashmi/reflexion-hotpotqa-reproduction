"""Command line for the cross-task failure-memory experiment.

    python run_experiment.py pilot    --provider groq --n 30
    python run_experiment.py build    --provider groq --n 150
    python run_experiment.py evaluate --provider groq --n 100 --trials 2
    python run_experiment.py analyze  --attempts results/eval_attempts.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .data import SealedSplitError, load_split
from .failure_memory import MemoryBank, build_memory
from .pipeline import (
    CONDITIONS,
    analyze,
    build_rows,
    print_analysis,
    read_rows,
    run_condition,
    save_report,
    write_rows,
)
from .providers import GroqProvider, OpenAIProvider, ScriptedProvider


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def make_provider(args: argparse.Namespace):
    if args.provider == "scripted":
        return ScriptedProvider()
    if args.provider == "groq":
        return GroqProvider(
            args.model,
            temperature=args.temperature,
            reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    return OpenAIProvider(args.model)


def add_common(parser: argparse.ArgumentParser, split: str, n: int) -> None:
    parser.add_argument("--provider", choices=("scripted", "groq", "openai"), default="scripted")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("train", "test", "validation"), default=split)
    parser.add_argument("--n", type=int, default=n, help="number of questions to sample")
    parser.add_argument("--seed", type=int, default=0, help="sampling and retrieval seed")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "none"), default="low")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--final-validation",
        action="store_true",
        help="allow the sealed validation split (use once, at the very end)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-task failure memory on HotpotQA.")
    sub = parser.add_subparsers(dest="command", required=True)

    pilot = sub.add_parser("pilot", help="first-attempt accuracy check to pick a model")
    add_common(pilot, "test", 30)

    build = sub.add_parser("build", help="build the lesson memory from training questions")
    add_common(build, "train", 150)
    build.add_argument("--no-validate", action="store_true", help="keep unvalidated lessons")
    build.add_argument("--lesson-content", choices=("fix", "diagnosis", "both"), default="fix")
    build.add_argument("--memory", type=Path, default=None, help="output memory JSON")

    evaluate = sub.add_parser("evaluate", help="compare conditions on unseen questions")
    add_common(evaluate, "test", 100)
    evaluate.add_argument(
        "--conditions",
        default="no_memory,reflexion,failure_memory,random_memory",
        help=f"comma list from: {','.join(CONDITIONS)}",
    )
    evaluate.add_argument("--trials", type=int, default=2)
    evaluate.add_argument("--k", type=int, default=3, help="lessons retrieved per question")
    evaluate.add_argument("--lesson-content", choices=("fix", "diagnosis", "both"), default="fix")
    evaluate.add_argument("--repeats", type=int, default=1, help="repeat runs to measure noise")
    evaluate.add_argument("--memory", type=Path, default=None)
    evaluate.add_argument("--label", choices=("llm", "rules", "none"), default="llm",
                          help="how to label failed attempts for analysis")

    report = sub.add_parser("analyze", help="summarize an attempts CSV")
    report.add_argument("--attempts", type=Path, default=ROOT / "results" / "eval_attempts.csv")
    report.add_argument("--out", type=Path, default=None)
    return parser


def _sample(args: argparse.Namespace):
    return load_split(args.split, args.n, seed=args.seed, allow_validation=args.final_validation)


def cmd_pilot(args: argparse.Namespace) -> int:
    model = make_provider(args)
    examples = _sample(args)
    rows = run_condition("no_memory", examples, model, max_trials=1, label="rules", seed=args.seed)
    for row in rows:
        row["phase"], row["condition"] = "eval", "no_memory"
    path = args.out_dir / "pilot_attempts.csv"
    write_rows(path, rows)
    em = sum(int(row["exact_match"]) for row in rows) / len(rows)
    print(f"Pilot on {len(rows)} {args.split} questions with {model.name}")
    print(f"First-attempt exact match: {em:.1%}")
    for qtype in sorted({row["qtype"] for row in rows}):
        subset = [row for row in rows if row["qtype"] == qtype]
        print(f"  {qtype or 'unknown':<11} {sum(int(r['exact_match']) for r in subset)}/{len(subset)}")
    if em > 0.75:
        print("Very high: little room to improve. Consider a smaller model or harder questions.")
    elif em < 0.40:
        print("Very low: lessons may be poor. Consider a stronger model.")
    else:
        print("In the 40-75% target range: good for the experiment.")
    print(f"Saved {path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    model = make_provider(args)
    examples = _sample(args)
    bank, logs = build_memory(
        examples, model, validate=not args.no_validate,
        lesson_content=args.lesson_content, progress=print,
    )
    memory_path = args.memory or args.out_dir / "memory.json"
    bank.save(memory_path)
    write_rows(args.out_dir / "build_attempts.csv", build_rows(logs, model.name))
    first = [log for log in logs if log.trial == "1"]
    print(f"\nTraining first-attempt EM: {sum(l.correct for l in first)}/{len(first)}")
    print(f"Lessons written: {len(bank.lessons)} | validated: {len(bank.validated())}")
    rejected = {}
    for lesson in bank.lessons:
        if lesson.rejected_reason:
            rejected[lesson.rejected_reason] = rejected.get(lesson.rejected_reason, 0) + 1
    for reason, count in rejected.items():
        print(f"  rejected ({reason}): {count}")
    print(f"Saved {memory_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if args.split == "train":
        raise ValueError("Evaluate on test (or the sealed validation split), not train.")
    model = make_provider(args)
    examples = _sample(args)
    bank = None
    if any(CONDITIONS[c][1] for c in conditions):
        bank = MemoryBank.load(args.memory or args.out_dir / "memory.json")
        print(f"Memory: {len(bank.validated())} validated lessons")

    rows = []
    for run_id in range(args.repeats):
        for condition in conditions:
            print(f"Running {condition} (run {run_id + 1}/{args.repeats}) on {len(examples)} questions")
            rows += run_condition(
                condition, examples, model, max_trials=args.trials, bank=bank, k=args.k,
                lesson_content=args.lesson_content, seed=args.seed, run_id=run_id, label=args.label,
            )
    suffix = "_validation" if args.split == "validation" else ""
    attempts = args.out_dir / f"eval_attempts{suffix}.csv"
    write_rows(attempts, rows)
    report = analyze(rows)
    report["config"] = {k: str(v) for k, v in vars(args).items()}
    save_report(args.out_dir / f"eval_report{suffix}.json", report)
    print_analysis(report)
    print(f"\nSaved {attempts}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    report = analyze(read_rows(args.attempts))
    print_analysis(report)
    out = args.out or args.attempts.with_name(args.attempts.stem + "_report.json")
    save_report(out, report)
    print(f"\nSaved {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {
            "pilot": cmd_pilot,
            "build": cmd_build,
            "evaluate": cmd_evaluate,
            "analyze": cmd_analyze,
        }[args.command](args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError, SealedSplitError) as error:
        print(f"\nERROR: {error}")
        return 1
