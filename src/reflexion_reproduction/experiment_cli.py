"""Command line for the cross-task failure-memory experiment.

    python run_experiment.py reproduce --provider groq --n 100 --trials 4
    python run_experiment.py pilot    --provider groq --n 30
    python run_experiment.py build    --provider groq --n 1000 --dev-n 100
    python run_experiment.py evaluate --provider groq --n 300 --trials 4 --conditions all
    python run_experiment.py analyze  --attempts results/eval_attempts.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

from .data import SealedSplitError, load_split
from .failure_memory import MemoryBank, build_memory
from .figures import plot_all, plot_lesson_checks
from .pipeline import (
    CONDITIONS,
    analyze,
    build_rows,
    print_analysis,
    read_rows,
    run_condition,
    save_comparison,
    save_report,
    write_rows,
)
from .providers import GroqProvider, OpenAIProvider, ScriptedProvider
from .reproduction import format_table, run_reproduction, save_reproduction


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
PLANNED_CONDITIONS = "no_memory,reflexion,failure_memory,random_memory"


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
            tokens_per_minute=args.tpm,
        )
    return OpenAIProvider(args.model)


def add_common(parser: argparse.ArgumentParser, split: str, n: int) -> None:
    parser.add_argument("--provider", choices=("scripted", "groq", "openai"), default="scripted")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("train", "test", "validation"), default=split)
    parser.add_argument("--n", type=int, default=n, help="number of questions to sample")
    parser.add_argument(
        "--level", choices=("all", "easy", "medium", "hard"), default="all",
        help="restrict the sample to one HotpotQA difficulty level",
    )
    parser.add_argument(
        "--context", choices=("gold", "full"), default="full",
        help="gold: only the supporting paragraphs (the paper's CoT (GT) setting); "
        "full: all ten distractor paragraphs",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling and retrieval seed")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "none"), default="low")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="questions answered side by side; changes the run time, not the results",
    )
    parser.add_argument(
        "--tpm", type=int, default=0,
        help="pace Groq calls to this many tokens per minute (0: no pacing); "
        "set it just under your account limit when using several workers",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--final-validation",
        action="store_true",
        help="allow the sealed validation split (use once, at the very end)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-task failure memory on HotpotQA.")
    sub = parser.add_subparsers(dest="command", required=True)

    reproduce = sub.add_parser(
        "reproduce", help="parent paper check: baseline vs Reflexion accuracy over attempts"
    )
    add_common(reproduce, "test", 100)
    reproduce.add_argument("--trials", type=int, default=4, help="attempts per question")
    reproduce.set_defaults(context="gold")  # the paper's CoT (GT) setting

    pilot = sub.add_parser("pilot", help="first-attempt accuracy check to pick a model")
    add_common(pilot, "test", 30)

    build = sub.add_parser("build", help="build the lesson memory from training questions")
    add_common(build, "train", 150)
    build.add_argument(
        "--dev-n", type=int, default=40,
        help="separate development questions each lesson is checked on (0: retry check only)",
    )
    build.add_argument(
        "--min-net", type=int, default=1,
        help="a lesson is kept if it helps at least this many more development answers than it hurts",
    )
    build.add_argument(
        "--trials", type=int, default=3,
        help="Reflexion attempts on each lesson question before the lesson is written",
    )
    build.add_argument("--no-validate", action="store_true", help="keep every usable lesson, unchecked")
    build.add_argument("--lesson-content", choices=("fix", "diagnosis", "both"), default="fix")
    build.add_argument("--memory", type=Path, default=None, help="output memory JSON")

    evaluate = sub.add_parser("evaluate", help="compare conditions on unseen questions")
    add_common(evaluate, "test", 100)
    evaluate.add_argument(
        "--conditions",
        default=PLANNED_CONDITIONS,
        help=f"comma list from: {','.join(CONDITIONS)} (or 'all')",
    )
    evaluate.add_argument("--trials", type=int, default=2)
    evaluate.add_argument("--k", type=int, default=3, help="lessons retrieved per question")
    evaluate.add_argument("--lesson-content", choices=("fix", "diagnosis", "both"), default="fix")
    evaluate.add_argument("--repeats", type=int, default=1, help="repeat runs to measure noise")
    evaluate.add_argument("--memory", type=Path, default=None)
    evaluate.add_argument("--label", choices=("llm", "rules", "none"), default="llm",
                          help="how to label failed attempts for analysis")
    evaluate.add_argument(
        "--resume", action="store_true",
        help="continue an interrupted run: keep the conditions already saved in the attempts file",
    )
    evaluate.add_argument(
        "--exclude-attempts", type=Path, default=None,
        help="attempts CSV of an earlier run: draw only questions that run did not use "
        "(with a new --seed this gives a fresh sample for a replication)",
    )

    report = sub.add_parser("analyze", help="summarize one or more attempts CSVs")
    report.add_argument(
        "--attempts", type=Path, nargs="+", default=[ROOT / "results" / "eval_attempts.csv"]
    )
    report.add_argument("--memory", type=Path, default=None,
                        help="memory JSON, to break results down by the failure type each lesson teaches")
    report.add_argument("--conditions", default="",
                        help="comma list of conditions to keep (default: all in the files)")
    report.add_argument("--out", type=Path, default=None)
    report.add_argument(
        "--layout", choices=("wide", "column"), default="wide",
        help="figure size: wide for a README or slide, column for one column of a two column "
        "paper (those files end in _column.png)",
    )
    return parser


def _sample(args: argparse.Namespace):
    return load_split(
        args.split, args.n, seed=args.seed, allow_validation=args.final_validation,
        level=None if args.level == "all" else args.level,
        gold_only=(args.context == "gold"),
    )


def _dev_sample(args: argparse.Namespace, exclude: set[str]):
    """Development questions for checking lessons, never the questions lessons came from."""

    if args.dev_n < 1:
        return []
    drawn = load_split(
        args.split, args.dev_n + 25, seed=args.seed + 1, allow_validation=args.final_validation,
        level=None if args.level == "all" else args.level,
        gold_only=(args.context == "gold"),
    )
    return [example for example in drawn if example.id not in exclude][: args.dev_n]


def _usage_line(model) -> str:
    usage = getattr(model, "total_usage", None)
    if not usage or not usage.get("calls"):
        return ""
    return (
        f"Model calls: {usage['calls']} | prompt tokens: {usage['prompt_tokens']:,} | "
        f"completion tokens: {usage['completion_tokens']:,} | retries: {usage.get('retries', 0)}"
    )


def cmd_reproduce(args: argparse.Namespace) -> int:
    if args.split == "train":
        raise ValueError("Reproduce on the test split, not train.")
    model = make_provider(args)
    examples = _sample(args)
    rows, summary = run_reproduction(
        examples, model, max_trials=args.trials, seed=args.seed, progress=print, workers=args.workers
    )
    paths = save_reproduction(args.out_dir, rows, summary, vars(args))
    print()
    print(format_table(summary))
    if args.provider == "scripted":
        print("\nScripted provider: this checks the pipeline only. The numbers are not results.")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    model = make_provider(args)
    examples = _sample(args)
    rows = run_condition(
        "no_memory", examples, model, max_trials=1, label="rules", seed=args.seed, workers=args.workers
    )
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
    dev = [] if args.no_validate else _dev_sample(args, {example.id for example in examples})
    print(f"Lesson questions: {len(examples)} | development questions: {len(dev)}")
    bank, logs = build_memory(
        examples, model, validate=not args.no_validate,
        lesson_content=args.lesson_content, progress=print,
        dev_examples=dev, workers=args.workers, min_net=args.min_net, max_trials=args.trials,
    )
    first = [log for log in logs if log.trial == "1"]
    dev_base = [log for log in logs if log.trial == "dev_baseline"]
    funnel = {
        "lesson_questions": len(first),
        "first_attempt_failures": sum(not log.correct for log in first),
        "lessons_written": len(bank.lessons),
        "from_recovered_questions": sum(lesson.recovered for lesson in bank.lessons),
        "usable": len(bank.pool("unvalidated")),
        "passed_retry_on_own_question": len(bank.pool("retry")),
        "validated": len(bank.validated()),
    }
    bank.meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model.name,
        "config": {key: str(value) for key, value in vars(args).items()},
        "funnel": funnel,
        "development_questions": len(dev),
        "development_baseline_correct": sum(log.correct for log in dev_base),
        "validated_by_failure_type": dict(Counter(lesson.failure_type for lesson in bank.validated())),
        "usage": dict(getattr(model, "total_usage", {}) or {}),
    }
    memory_path = args.memory or args.out_dir / "memory.json"
    bank.save(memory_path)
    write_rows(args.out_dir / "build_attempts.csv", build_rows(logs, model.name))
    plot_lesson_checks(bank, memory_path.with_name("lesson_checks.png"))

    print(f"\nTraining first-attempt EM: {sum(l.correct for l in first)}/{len(first)}")
    print(
        f"Lessons written: {len(bank.lessons)} "
        f"({funnel['from_recovered_questions']} from questions Reflexion recovered) | "
        f"usable: {funnel['usable']} | "
        f"passed the retry on their own question: {funnel['passed_retry_on_own_question']} | "
        f"validated: {len(bank.validated())}"
    )
    if dev:
        print(
            f"Development questions: {len(dev)} "
            f"({sum(log.correct for log in dev_base)} right without lessons)"
        )
    rejected = Counter(lesson.rejected_reason for lesson in bank.lessons if lesson.rejected_reason)
    for reason, count in rejected.items():
        print(f"  rejected ({reason}): {count}")
    for failure_type, count in Counter(l.failure_type for l in bank.validated()).most_common():
        print(f"  validated {failure_type}: {count}")
    if _usage_line(model):
        print(_usage_line(model))
    print(f"Saved {memory_path}")
    return 0


def _finish_evaluation(out_dir: Path, rows: list[dict], bank, config: dict, suffix: str = "") -> dict:
    report = analyze(rows, bank)
    report["config"] = config
    save_report(out_dir / f"eval_report{suffix}.json", report)
    paths = save_comparison(out_dir, report, suffix=suffix)
    paths.update(plot_all(report, out_dir, suffix))
    print_analysis(report)
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    return report


def cmd_evaluate(args: argparse.Namespace) -> int:
    names = list(CONDITIONS) if args.conditions.strip() == "all" else args.conditions.split(",")
    conditions = [c.strip() for c in names if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if args.split == "train":
        raise ValueError("Evaluate on test (or the sealed validation split), not train.")
    model = make_provider(args)
    examples = _sample(args)
    if args.exclude_attempts:
        used = {row["example_id"] for row in read_rows(args.exclude_attempts)}
        wanted, args.n = args.n, args.n + 150
        examples = [example for example in _sample(args) if example.id not in used][:wanted]
        args.n = wanted
        print(f"Fresh sample: {len(examples)} questions not in {args.exclude_attempts.name}")
    bank = None
    if any(CONDITIONS[c].retrieval for c in conditions):
        bank = MemoryBank.load(args.memory or args.out_dir / "memory.json")
        print(f"Memory: {len(bank.validated())} validated lessons")

    suffix = "_validation" if args.split == "validation" else ""
    attempts = args.out_dir / f"eval_attempts{suffix}.csv"
    rows: list[dict] = []
    if args.resume and attempts.exists():
        rows = read_rows(attempts)
        if {row["example_id"] for row in rows} - {example.id for example in examples}:
            raise ValueError(f"{attempts} holds other questions; delete it or drop --resume.")
        print(f"Resuming: {len(rows)} saved attempts")
    finished = {(str(row["run_id"]), row["condition"]) for row in rows}

    for run_id in range(args.repeats):
        for condition in conditions:
            if (str(run_id), condition) in finished:
                print(f"Skipping {condition} (run {run_id + 1}/{args.repeats}): already saved")
                continue
            print(f"Running {condition} (run {run_id + 1}/{args.repeats}) on {len(examples)} questions")

            def progress(trial: int, done: int, total: int) -> None:
                if done == total or done % 100 == 0:
                    print(f"  attempt {trial}: {done}/{total}")

            rows += run_condition(
                condition, examples, model, max_trials=args.trials, bank=bank, k=args.k,
                lesson_content=args.lesson_content, seed=args.seed, run_id=run_id, label=args.label,
                workers=args.workers, progress=progress,
            )
            write_rows(attempts, rows)  # saved after every condition so a crash loses little
    config = {k: str(v) for k, v in vars(args).items()}
    _finish_evaluation(args.out_dir, rows, bank, config, suffix)
    if _usage_line(model):
        print(_usage_line(model))
    print(f"\nSaved {attempts}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    rows: list[dict] = []
    for path in args.attempts:
        rows += read_rows(path)
    keep = [name.strip() for name in args.conditions.split(",") if name.strip()]
    if keep:
        rows = [row for row in rows if row["condition"] in keep]
    first = args.attempts[0]
    memory = args.memory or first.with_name("memory.json")
    bank = MemoryBank.load(memory) if memory.exists() else None
    report = analyze(rows, bank)
    print_analysis(report)
    out = args.out or first.with_name(first.stem.replace("_attempts", "") + "_report.json")
    if out.exists():  # keep the settings the evaluation was run with
        config = json.loads(out.read_text(encoding="utf-8")).get("config")
        if config:
            report["config"] = config
    save_report(out, report)
    suffix = "_validation" if first.stem.endswith("_validation") else ""
    paths = save_comparison(out.parent, report, suffix=suffix)
    # The lesson check figure belongs beside the memory file it is drawn from.
    beside_memory = bank is not None and memory.parent.resolve() == out.parent.resolve()
    paths.update(plot_all(report, out.parent, suffix, args.layout, bank if beside_memory else None))
    print(f"\nSaved {out}")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {
            "reproduce": cmd_reproduce,
            "pilot": cmd_pilot,
            "build": cmd_build,
            "evaluate": cmd_evaluate,
            "analyze": cmd_analyze,
        }[args.command](args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError, SealedSplitError) as error:
        print(f"\nERROR: {error}")
        return 1
