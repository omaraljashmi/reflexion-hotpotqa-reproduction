"""The comparison figure: accuracy of every condition on the same questions."""

from __future__ import annotations

from pathlib import Path
import textwrap

INK = "#111111"
SECONDARY = "#52514E"
MUTED = "#898781"
ACCENT = "#C8102E"  # the proposed method
GRID = "#E1E0D9"
AXIS = "#C3C2B7"
PROPOSED = "failure_memory"

# Sizes in inches and points. "wide" suits a README or a slide, "column" one
# column of a two column paper, where the figure is printed at this exact width.
LAYOUTS = {
    "wide": dict(width=7.2, row=0.6, left=2.45, right=1.3, top=0.62, bottom=0.78, wrap=24,
                 label=10, number=10, small=9, note=8, dot=10, ring=9, legend_columns=3,
                 first_column=1.15, second_column=1.34),
    "column": dict(width=3.45, row=0.34, left=1.2, right=0.8, top=0.5, bottom=0.58, wrap=19,
                   label=7, number=7, small=6.5, note=5.5, dot=6.5, ring=6, legend_columns=2,
                   first_column=1.21, second_column=1.47),
}


def plot_comparison(report: dict, path: Path, layout: str = "wide") -> bool:
    """Dot plot of first-attempt accuracy and accuracy after the last attempt.

    One row per condition. The filled dot is the first attempt with its 95%
    interval, the open dot is the share solved after all attempts. The thin
    vertical line marks the no-memory first attempt, the level every lesson
    condition has to beat. Returns False if matplotlib is missing.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return False

    rows = report.get("summary") or []
    if not rows:
        return False
    size = LAYOUTS[layout]

    attempts = rows[0]["attempts"]
    first = [100 * row["first_attempt_em"] for row in rows]
    final = [100 * row["final_em"] for row in rows]
    low = [100 * row["first_attempt_em_ci95"][0] for row in rows]
    high = [100 * row["first_attempt_em_ci95"][1] for row in rows]
    labels = ["\n".join(textwrap.wrap(row["label"], size["wrap"])) for row in rows]
    positions = list(range(len(rows)))[::-1]  # first condition on top

    # Fixed margins: condition names on the left, the numbers on the right.
    width = size["width"]
    height = size["top"] + size["bottom"] + size["row"] * len(rows)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([
        size["left"] / width,
        size["bottom"] / height,
        (width - size["left"] - size["right"]) / width,
        size["row"] * len(rows) / height,
    ])
    left = max(0, 5 * int((min(low) - 1.5) // 5))
    right = min(100, 5 * int((max(high + final) + 1.5) // 5 + 1))
    step = 5 if right - left <= 40 else 10
    ax.set_xlim(left, right)
    ax.set_ylim(-0.6, len(rows) - 0.4)

    baseline = next((100 * r["first_attempt_em"] for r in rows if r["condition"] == "no_memory"), None)
    if baseline is not None:
        ax.axvline(baseline, color=AXIS, linewidth=1, zorder=1)

    for y, row, a, b, lo, hi in zip(positions, rows, first, final, low, high):
        color = ACCENT if row["condition"] == PROPOSED else INK
        ax.plot([lo, hi], [y, y], color=color, linewidth=1, solid_capstyle="butt", zorder=2)
        if attempts > 1:
            ax.plot([b], [y], marker="o", markersize=size["ring"], markerfacecolor="white",
                    clip_on=False, markeredgecolor=color, markeredgewidth=1.4, linestyle="none", zorder=3)
        ax.plot([a], [y], marker="o", markersize=size["dot"], markerfacecolor=color, clip_on=False,
                markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=4)

    # The numbers, as a small table beside the plot.
    columns = [("First", first, size["first_column"])]
    if attempts > 1:
        columns.append((f"After {attempts}", final, size["second_column"]))
    for title, values, x in columns:
        ax.text(x, 1.0, title, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=size["small"], color=MUTED)
        for y, row, value in zip(positions, rows, values):
            ax.text(
                x, y, f"{value:.1f}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=size["number"], color=INK,
                fontweight="bold" if row["condition"] == PROPOSED and title == "First" else "normal",
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=size["label"], color=INK)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["condition"] == PROPOSED:
            tick.set_fontweight("bold")
    ax.set_xticks(range(left, right + 1, step))
    ax.tick_params(axis="x", labelsize=size["small"], colors=SECONDARY, length=0)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("Exact match (%)", fontsize=size["small"], color=SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    handles = [
        Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK, markeredgecolor="white",
               markersize=size["dot"] - 1, linewidth=1, label="First attempt, 95% interval"),
    ]
    if attempts > 1:
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK,
                   markeredgewidth=1.4, markersize=size["ring"] - 1, label=f"After {attempts} attempts")
        )
    if baseline is not None:
        handles.append(Line2D([0], [0], color=AXIS, linewidth=1, label="No memory, first attempt"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.995),
               ncol=size["legend_columns"], frameon=False, fontsize=size["small"], handletextpad=0.5,
               columnspacing=1.4, borderaxespad=0.3, labelspacing=0.3)

    note = f"{rows[0]['examples']} HotPotQA questions"
    if report.get("model"):
        note += f", {report['model']}"
    fig.text(0.988, 0.02, note, ha="right", va="bottom", fontsize=size["note"], color=MUTED)
    fig.savefig(path, dpi=300 if layout == "column" else 200, facecolor="white")
    plt.close(fig)
    return True
