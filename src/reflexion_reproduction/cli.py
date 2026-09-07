"""Command-line interface used for both the recording and live experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

from .core import Example, ExperimentResult, Strategy, run_experiment
from .providers import OpenAIProvider, ScriptedProvider


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_examples(path: Path, limit: int) -> list[Example]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["examples"] if isinstance(payload, dict) else payload
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    return [
        Example(
            id=str(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            context=tuple(row["context"]),
        )
        for row in rows[:limit]
    ]


def print_run(result: ExperimentResult) -> None:
    heading = result.strategy.value.upper()
    print(f"\n{'=' * 72}\n{heading}\n{'=' * 72}")
    for record in result.records:
        status = "PASS" if record.exact_match else "FAIL"
        print(f"\n[{record.example_id}] Trial {record.trial}: {status}")
        print(f"  Thought:    {record.thought}")
        print(f"  Prediction: {record.prediction}")
        print(f"  Reference:  {record.reference}")
        print(f"  Feedback:   {record.feedback} (EM={int(record.exact_match)}, F1={record.f1:.2f})")
        if record.reflection_added:
            print(f"  Reflection: {record.reflection_added}")

    print("\nCumulative exact-match accuracy:")
    for trial, accuracy in enumerate(result.accuracy_by_trial(), start=1):
        solved = result.solved_by_trial(trial)
        print(f"  Trial {trial}: {solved}/{result.total_examples} ({accuracy:.1%})")


def save_results(
    path: Path,
    provider: str,
    results: Sequence[ExperimentResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "note": (
            "scripted-offline is a deterministic control-flow smoke test, not a "
            "scientific model result"
            if provider == "scripted-offline"
            else "Live model experiment"
        ),
        "experiments": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the Reflexion actor/evaluator/reflection/memory loop."
    )
    parser.add_argument(
        "--provider",
        choices=("scripted", "openai"),
        default="scripted",
        help="scripted is offline and deterministic; openai runs a live LLM experiment",
    )
    parser.add_argument(
        "--mode",
        choices=("compare", "baseline", "reflexion"),
        default="compare",
        help="experimental condition(s) to run",
    )
    parser.add_argument("--limit", type=int, default=3, help="number of examples")
    parser.add_argument("--trials", type=int, default=2, help="maximum trials per example")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        help="OpenAI model for --provider openai",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root() / "data" / "hotpotqa_mini.json",
        help="HotPotQA-style JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results" / "latest.json",
        help="where to save detailed JSON results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        examples = load_examples(args.data, args.limit)
        provider = ScriptedProvider() if args.provider == "scripted" else OpenAIProvider(args.model)
        strategies = {
            "compare": (Strategy.BASELINE, Strategy.REFLEXION),
            "baseline": (Strategy.BASELINE,),
            "reflexion": (Strategy.REFLEXION,),
        }[args.mode]

        print("REFLEXION HOTPOTQA MINI REPRODUCTION")
        print(f"Provider: {provider.name}")
        print(f"Examples: {len(examples)} | Maximum trials: {args.trials}")
        if args.provider == "scripted":
            print("Mode note: deterministic smoke test; not a scientific LLM result")

        results = [
            run_experiment(examples, provider, strategy, args.trials)
            for strategy in strategies
        ]
        for result in results:
            print_run(result)

        if len(results) == 2:
            baseline, reflexion = results
            change = reflexion.final_accuracy - baseline.final_accuracy
            print(f"\n{'=' * 72}\nCOMPARISON\n{'=' * 72}")
            print(f"Baseline final accuracy:  {baseline.final_accuracy:.1%}")
            print(f"Reflexion final accuracy: {reflexion.final_accuracy:.1%}")
            print(f"Absolute change:           {change:+.1%}")

        save_results(args.output, provider.name, results)
        print(f"\nSUCCESS: run completed without errors")
        print(f"Detailed results: {args.output.resolve()}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"\nERROR: {error}")
        return 1

