"""Reproduce the parent paper's HotPotQA result: baseline vs Reflexion over trials.

The paper (Shinn et al., 2023, Fig. 4b) runs 100 HotPotQA questions with the
supporting context supplied. Without reflection the share of solved questions
stays flat; with reflection it rises over trials. This module runs the same two
conditions on a seeded sample of the test split and prints the two curves next
to the paper's values.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Sequence

from .core import Example
from .pipeline import cumulative_accuracy, run_condition, write_rows


# Approximate values read from Fig. 4(b) of the paper ("CoT (GT)" setting,
# 100 questions). Attempt 1 is the paper's trial 0.
PAPER_REFERENCE = {
    "paper": "Shinn et al. 2023, Fig. 4(b), CoT (GT), 100 HotPotQA questions",
    "baseline": [0.61] * 8,
    "reflexion": [0.61, 0.69, 0.69, 0.70, 0.72, 0.74, 0.75, 0.75],
}

CONDITIONS = ("no_memory", "reflexion")


def run_reproduction(
    examples: Sequence[Example],
    model,
    max_trials: int,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run baseline and Reflexion on the same questions and summarize both curves."""

    rows: list[dict] = []
    for condition in CONDITIONS:
        if progress:
            progress(f"Running {condition} on {len(examples)} questions, up to {max_trials} attempts")
        # label="none": no extra model calls to classify failures.
        rows += run_condition(condition, examples, model, max_trials=max_trials, seed=seed, label="none")

    curves = {cond: cumulative_accuracy(rows, cond, max_trials) for cond in CONDITIONS}
    first = {cond: _first_attempt_failures(rows, cond) for cond in CONDITIONS}
    tokens = {
        cond: sum(
            int(row["prompt_tokens"] or 0) + int(row["completion_tokens"] or 0)
            for row in rows
            if row["condition"] == cond
        )
        for cond in CONDITIONS
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model.name,
        "examples": len(examples),
        "max_trials": max_trials,
        "accuracy_by_attempt": curves,
        "first_attempt_failures": first,
        "recovered_by_reflexion": _recovered(rows, "reflexion"),
        "total_tokens": tokens,
        "paper_reference": PAPER_REFERENCE,
    }
    return rows, summary


def _first_attempt_failures(rows: Sequence[dict], condition: str) -> int:
    return sum(
        1
        for row in rows
        if row["condition"] == condition
        and str(row["trial"]) == "1"
        and row["exact_match"] not in ("1", 1, True)
    )


def _recovered(rows: Sequence[dict], condition: str) -> int:
    """Examples that failed their first attempt and passed a later one."""

    first_failed: set[str] = set()
    later_passed: set[str] = set()
    for row in rows:
        if row["condition"] != condition:
            continue
        passed = row["exact_match"] in ("1", 1, True)
        if str(row["trial"]) == "1" and not passed:
            first_failed.add(row["example_id"])
        elif str(row["trial"]) != "1" and passed:
            later_passed.add(row["example_id"])
    return len(first_failed & later_passed)


def format_table(summary: dict) -> str:
    """Side by side curves: this run vs the paper."""

    curves = summary["accuracy_by_attempt"]
    paper = summary["paper_reference"]
    lines = [
        f"Model: {summary['model']} | questions: {summary['examples']} | attempts: {summary['max_trials']}",
        "",
        "Attempt   Baseline   Reflexion   Paper baseline   Paper Reflexion",
    ]
    for i in range(summary["max_trials"]):
        pb = f"{paper['baseline'][i]:.2f}" if i < len(paper["baseline"]) else "  -"
        pr = f"{paper['reflexion'][i]:.2f}" if i < len(paper["reflexion"]) else "  -"
        lines.append(
            f"{i + 1:>7}   {curves['no_memory'][i]:>8.2f}   {curves['reflexion'][i]:>9.2f}"
            f"   {pb:>14}   {pr:>15}"
        )
    gain = curves["reflexion"][-1] - curves["reflexion"][0]
    lines += [
        "",
        f"Reflexion: {summary['first_attempt_failures']['reflexion']} first-attempt failures, "
        f"{summary['recovered_by_reflexion']} recovered by the last attempt "
        f"(+{gain:.0%} accuracy). Paper: +{paper['reflexion'][-1] - paper['reflexion'][0]:.0%}.",
        f"Baseline change over attempts: "
        f"{curves['no_memory'][-1] - curves['no_memory'][0]:+.0%} (paper: flat).",
        "Paper values are read from the figure and are approximate.",
    ]
    return "\n".join(lines)


def save_reproduction(out_dir: Path, rows: list[dict], summary: dict, config: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "attempts": out_dir / "reproduction_attempts.csv",
        "report": out_dir / "reproduction_report.json",
    }
    write_rows(paths["attempts"], rows)
    payload = dict(summary)
    payload["config"] = {key: str(value) for key, value in config.items()}
    paths["report"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_path = out_dir / "reproduction_curve.png"
    if plot_curves(summary, plot_path):
        paths["plot"] = plot_path
    return paths


def plot_curves(summary: dict, path: Path) -> bool:
    """Write the accuracy curves as a PNG. Returns False if matplotlib is missing."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    curves = summary["accuracy_by_attempt"]
    paper = summary["paper_reference"]
    t = summary["max_trials"]
    x = list(range(1, t + 1))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(x, paper["reflexion"][:t], color="#9A9A9A", linestyle=":", marker="o", markersize=3,
            label="Paper: Reflexion")
    ax.plot(x, paper["baseline"][:t], color="#9A9A9A", linestyle="--", label="Paper: baseline")
    ax.plot(x, curves["reflexion"], color="#111111", marker="o", markersize=4, label="This run: Reflexion")
    ax.plot(x, curves["no_memory"], color="#C8102E", marker="s", markersize=4, label="This run: baseline")
    ax.set_xticks(x)
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Share of questions solved")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{summary['examples']} HotPotQA questions, {summary['model']}", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True
