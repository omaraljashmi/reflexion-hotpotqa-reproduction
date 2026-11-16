"""Figures of the experiment. Each one is drawn from the analysis report.

comparison      accuracy of every condition, first attempt and after the last attempt
attempts        share of questions solved by the end of each attempt
fixed_broken    first-attempt answers each condition fixed and broke against no memory
gain_bootstrap  how certain the first-attempt gain of the proposed method is
lesson_checks   every lesson on the development questions: answers helped against hurt

One style throughout: black and gray on white, and one accent color that always
marks the proposed method.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

INK = "#111111"
SECONDARY = "#52514E"
MUTED = "#898781"
SOFT = "#CFCEC6"  # marks that only give context
ACCENT = "#C8102E"  # the proposed method
GRID = "#E1E0D9"
AXIS = "#C3C2B7"
PROPOSED = "failure_memory"
MAIN = ("no_memory", "reflexion", PROPOSED)  # the conditions drawn in the attempt curves


def _pyplot():
    """matplotlib.pyplot, or None if matplotlib is not installed."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _frame(plt, width: float, height: float, left: float, right: float, top: float, bottom: float):
    """A figure with one plot area and fixed margins, all in inches."""

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([
        left / width, bottom / height, (width - left - right) / width, (height - top - bottom) / height,
    ])
    return fig, ax


def _plot_size(ax) -> tuple[float, float]:
    """Width and height of the plot area in inches."""

    box = ax.get_position()
    return box.width * ax.figure.get_figwidth(), box.height * ax.figure.get_figheight()


def _per_inch(ax) -> tuple[float, float]:
    """Data units per inch along x and y. Call it after the axis limits are set."""

    width, height = _plot_size(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return abs(x1 - x0) / width, abs(y1 - y0) / height


def _bar(ax, base: float, tip: float, centre: float, thickness: float, color: str,
         radius: float = 0.035, horizontal: bool = True, zorder: float = 2) -> None:
    """A bar that is square at its baseline and rounded at its data end.

    ``base`` and ``tip`` run along the value axis, ``centre`` and ``thickness``
    across it, all in data units. ``radius`` is in inches.
    """

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    if tip == base:
        return
    per_x, per_y = _per_inch(ax)
    along, across = (per_x, per_y) if horizontal else (per_y, per_x)
    ra, rc = radius * along, min(radius * across, thickness / 2)
    if abs(tip - base) < 2 * ra:  # too short to round: a corner larger than the bar looks like a bracket
        ra = rc = 0.0
    sign = 1 if tip > base else -1
    low, high = centre - thickness / 2, centre + thickness / 2
    points = [
        (base, low), (tip - sign * ra, low), (tip, low), (tip, low + rc),
        (tip, high - rc), (tip, high), (tip - sign * ra, high), (base, high), (base, low),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    if not horizontal:
        points = [(y, x) for x, y in points]
    ax.add_patch(PathPatch(MplPath(points, codes), facecolor=color, edgecolor="none", zorder=zorder))


def _note(fig, report: dict, size: dict, lead: str = "") -> None:
    """Small line in the bottom right corner: what the figure is drawn from."""

    rows = report.get("summary") or []
    text = f"{rows[0]['examples']} HotPotQA questions" if rows else ""
    if report.get("model"):
        text += f", {report['model']}"
    if lead:
        text = f"{lead}. {text}" if text else lead
    fig.text(0.988, 0.02, text, ha="right", va="bottom", fontsize=size["note"], color=MUTED)


def _minus(text: str) -> str:
    """Numbers are set with a minus sign, not a hyphen."""

    return text.replace("-", "\u2212")


def _label(report: dict, condition: str) -> str:
    return next((row["label"] for row in report.get("summary", []) if row["condition"] == condition), condition)


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


ATTEMPT_LAYOUTS = {
    "wide": dict(width=7.2, height=3.7, left=0.72, right=0.72, top=0.62, bottom=0.72,
                 label=10, small=9, note=8, dot=9, line=2.0, legend_columns=3),
    "column": dict(width=3.45, height=2.3, left=0.44, right=0.4, top=0.56, bottom=0.56,
                   label=7, small=6.5, note=5.5, dot=6, line=1.5, legend_columns=2),
}


def plot_attempts(report: dict, path: Path, layout: str = "wide") -> bool:
    """Share of questions solved by the end of each attempt, as in Fig. 4 of the parent paper.

    Three lines: no memory, Reflexion and the proposed method. The value after
    the last attempt is written at the end of each line.
    """

    plt = _pyplot()
    curves = report.get("accuracy_by_attempt") or {}
    shown = [cond for cond in MAIN if cond in curves]
    if plt is None or len(shown) < 2 or len(curves[shown[0]]) < 2:
        return False
    from matplotlib.lines import Line2D

    size = ATTEMPT_LAYOUTS[layout]
    attempts = len(curves[shown[0]])
    values = {cond: [100 * share for share in curves[cond]] for cond in shown}
    colors = {"no_memory": MUTED, "reflexion": INK, PROPOSED: ACCENT}

    fig, ax = _frame(plt, size["width"], size["height"], size["left"], size["right"], size["top"], size["bottom"])
    lowest = min(min(v) for v in values.values())
    highest = max(max(v) for v in values.values())
    bottom, top = 5 * int((lowest - 2) // 5), 5 * int((highest + 2) // 5 + 1)
    ax.set_xlim(0.75, attempts + 0.25)
    ax.set_ylim(bottom, top)

    xs = list(range(1, attempts + 1))
    for order, cond in enumerate(shown):
        ax.plot(xs, values[cond], color=colors[cond], linewidth=size["line"], solid_capstyle="round",
                solid_joinstyle="round", marker="o", markersize=size["dot"], markerfacecolor=colors[cond],
                markeredgecolor="white", markeredgewidth=1.4, clip_on=False, zorder=3 + order)

    # The last value beside each line end, moved apart only if two would touch.
    _, per_y = _per_inch(ax)
    gap = 1.25 * size["label"] / 72 * per_y
    ends = sorted(((values[cond][-1], cond) for cond in shown))
    placed: list[list] = []
    for value, cond in ends:
        y = value if not placed else max(value, placed[-1][0] + gap)
        placed.append([y, value, cond])
    for y, value, cond in placed:
        ax.annotate(f"{value:.1f}", xy=(attempts, y), xytext=(size["dot"] * 0.9, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=size["label"], color=INK, annotation_clip=False,
                    fontweight="bold" if cond == PROPOSED else "normal")

    ax.set_xticks(xs)
    ax.set_yticks(range(bottom, top + 1, 5))
    ax.tick_params(axis="both", labelsize=size["small"], colors=SECONDARY, length=0, pad=5)
    ax.set_xlabel("Attempt", fontsize=size["small"], color=SECONDARY)
    ax.set_ylabel("Questions solved (%)", fontsize=size["small"], color=SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    handles = [
        Line2D([0], [0], color=colors[cond], linewidth=size["line"], marker="o", markersize=size["dot"] - 1,
               markerfacecolor=colors[cond], markeredgecolor="white", label=_label(report, cond))
        for cond in reversed(shown)
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.995), ncol=size["legend_columns"],
               frameon=False, fontsize=size["small"], handletextpad=0.5, columnspacing=1.4,
               borderaxespad=0.3, labelspacing=0.3)
    _note(fig, report, size)
    fig.savefig(path, dpi=300 if layout == "column" else 200, facecolor="white")
    plt.close(fig)
    return True


FLIP_LAYOUTS = {
    "wide": dict(width=7.2, row=0.52, left=2.45, right=1.6, top=0.5, bottom=0.78, wrap=24,
                 label=10, number=10, small=9, note=8, bar=0.2, net_column=0.62, p_column=1.38),
    "column": dict(width=3.45, row=0.3, left=1.2, right=0.8, top=0.36, bottom=0.58, wrap=19,
                   label=7, number=7, small=6.5, note=5.5, bar=0.115, net_column=0.3, p_column=0.7),
}


def plot_fixed_broken(report: dict, path: Path, layout: str = "wide") -> bool:
    """First-attempt answers each condition fixed and broke against no memory.

    One row per condition. The bar to the right counts wrong answers that
    became right, the bar to the left right answers that became wrong: the two
    numbers the exact McNemar test compares. Net and p stand beside each row.
    """

    plt = _pyplot()
    rows = []
    for row in report.get("summary") or []:
        comp = next(
            (c for c in report.get("comparisons", []) if c["reference"] == "no_memory"
             and c["condition"] == row["condition"] and c["metric"] == "first"),
            None,
        )
        if comp:
            rows.append((row, comp))
    if plt is None or not rows:
        return False

    size = FLIP_LAYOUTS[layout]
    width = size["width"]
    height = size["top"] + size["bottom"] + size["row"] * len(rows)
    fig, ax = _frame(plt, width, height, size["left"], size["right"], size["top"], size["bottom"])
    largest = max(max(comp["fail_to_pass"], comp["pass_to_fail"]) for _, comp in rows)
    step = 5 if largest <= 12 else 10 if largest <= 45 else 20
    limit = step * int(largest * 1.3 // step + 1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    per_x, per_y = _per_inch(ax)
    positions = list(range(len(rows)))[::-1]  # first condition on top

    for y, (row, comp) in zip(positions, rows):
        color = ACCENT if row["condition"] == PROPOSED else INK
        fixed, broken = comp["fail_to_pass"], comp["pass_to_fail"]
        _bar(ax, 0, fixed, y, size["bar"] * per_y, color)
        _bar(ax, 0, -broken, y, size["bar"] * per_y, color)
        offset = 0.05 * per_x
        weight = "bold" if row["condition"] == PROPOSED else "normal"
        ax.text(fixed + offset, y, str(fixed), ha="left", va="center", fontsize=size["number"],
                color=INK, fontweight=weight)
        ax.text(-broken - offset, y, str(broken), ha="right", va="center", fontsize=size["number"],
                color=INK, fontweight=weight)
        plot_width, _ = _plot_size(ax)
        for x_inches, text in ((size["net_column"], _minus(f"{comp['net_gain']:+d}")),
                               (size["p_column"], "<0.001" if comp["mcnemar_p"] < 0.001 else f"{comp['mcnemar_p']:.3f}")):
            ax.text(1 + x_inches / plot_width, y, text, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=size["number"], color=INK, fontweight=weight)

    # The two bars of a row meet at zero: a white gap, then the axis line.
    ax.axvline(0, color="white", linewidth=2.4, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=4)
    plot_width, _ = _plot_size(ax)
    for x, text, align in ((0.25, "Broken", "center"), (0.75, "Fixed", "center"),
                           (1 + size["net_column"] / plot_width, "Net", "right"),
                           (1 + size["p_column"] / plot_width, "p", "right")):
        ax.text(x, 1.0, text, transform=ax.transAxes, ha=align, va="bottom", fontsize=size["small"], color=MUTED)

    labels = ["\n".join(textwrap.wrap(row["label"], size["wrap"])) for row, _ in rows]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=size["label"], color=INK)
    for tick, (row, _) in zip(ax.get_yticklabels(), rows):
        if row["condition"] == PROPOSED:
            tick.set_fontweight("bold")
    tick_step = step if limit // step <= 3 else 2 * step
    reach = (limit - 1) // tick_step
    ticks = [tick_step * k for k in range(-reach, reach + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(tick)) for tick in ticks])
    ax.tick_params(axis="x", labelsize=size["small"], colors=SECONDARY, length=0)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("First attempt answers changed", fontsize=size["small"], color=SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    _note(fig, report, size)
    fig.savefig(path, dpi=300 if layout == "column" else 200, facecolor="white")
    plt.close(fig)
    return True


BOOT_LAYOUTS = {
    "wide": dict(width=7.2, height=3.3, left=0.4, right=0.4, top=0.62, bottom=0.78,
                 label=10, small=9, note=8, line=2.0, legend_columns=2),
    "column": dict(width=3.45, height=2.0, left=0.2, right=0.2, top=0.42, bottom=0.6,
                   label=7, small=6.5, note=5.5, line=1.5, legend_columns=2),
}


def plot_gain_bootstrap(report: dict, path: Path, layout: str = "wide", condition: str = PROPOSED) -> bool:
    """How certain the first-attempt gain over no memory is.

    A histogram of the gain in exact match over bootstrap resamples of the
    questions. Darker bars lie inside the 95% interval. One line marks zero,
    the other the gain that was measured.
    """

    plt = _pyplot()
    gain = next(
        (g for g in report.get("first_attempt_em_gains") or [] if g["condition"] == condition), None
    )
    if plt is None or not gain or not gain.get("net_counts"):
        return False
    from matplotlib.patches import Patch

    size = BOOT_LAYOUTS[layout]
    n, resamples = gain["pairs"], gain["resamples"]
    counts = {int(net): count for net, count in gain["net_counts"].items()}
    low, high = gain["ci95_points"]

    # Bars hold a whole number of possible values each, so no bar is short by construction.
    span = max(counts) - min(counts) + 1
    per_bin = max(1, round(span / 32))
    bins: dict[int, int] = {}
    for net, count in counts.items():
        index = (net + per_bin // 2) // per_bin
        bins[index] = bins.get(index, 0) + count
    point = 100 / n  # one answer, in percentage points
    centre = {index: point * (index * per_bin - per_bin // 2 + (per_bin - 1) / 2) for index in bins}
    bin_width = point * per_bin
    share = {index: 100 * count / resamples for index, count in bins.items()}

    fig, ax = _frame(plt, size["width"], size["height"], size["left"], size["right"], size["top"], size["bottom"])
    left = min(min(centre.values()) - bin_width, -bin_width)
    right = max(centre.values()) + bin_width
    ax.set_xlim(left, right)
    ax.set_ylim(0, max(share.values()) * 1.24)
    per_x, _ = _per_inch(ax)
    bar_width = bin_width - min(0.02 * per_x, 0.3 * bin_width)  # a thin white gap between bars
    for index in sorted(bins):
        inside = low - 1e-9 <= centre[index] <= high + 1e-9
        _bar(ax, 0, share[index], centre[index], bar_width, MUTED if inside else SOFT,
             radius=0.02, horizontal=False)

    top = ax.get_ylim()[1]
    ax.plot([0, 0], [0, top * 0.93], color=INK, linewidth=0.9, zorder=4)
    ax.plot([gain["gain_points"]] * 2, [0, top * 0.93], color=ACCENT, linewidth=size["line"],
            solid_capstyle="butt", zorder=5)
    pad = 0.04 * per_x
    ax.text(-pad, top * 0.93, "No gain", ha="right", va="top", fontsize=size["small"], color=SECONDARY)
    ax.text(gain["gain_points"] + pad, top * 0.93, _minus(f"Measured gain {gain['gain_points']:.1f}"), ha="left",
            va="top", fontsize=size["small"], color=INK, fontweight="bold")

    width_points = right - left
    step = 1 if width_points <= 7 else 2 if width_points <= 14 else 5
    first = step * int(-(-left // step))
    ticks = [tick for tick in range(first, int(right) + 1, step)]
    ax.set_xticks(ticks)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=size["small"], colors=SECONDARY, length=0, pad=4)
    ax.set_xlabel("Gain in first attempt exact match (percentage points)", fontsize=size["small"], color=SECONDARY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    handles = [
        Patch(facecolor=MUTED, edgecolor="none",
              label=_minus(f"Inside the 95% interval, {low:.1f} to {high:.1f}")),
        Patch(facecolor=SOFT, edgecolor="none", label="Outside"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.995), ncol=size["legend_columns"],
               frameon=False, fontsize=size["small"], handlelength=1.1, handleheight=0.8, handletextpad=0.5,
               columnspacing=1.4, borderaxespad=0.3)
    fig.text(0.988, 0.02,
             f"{_label(report, condition)} against no memory. {resamples:,} resamples of {n} questions",
             ha="right", va="bottom", fontsize=size["note"], color=MUTED)
    fig.savefig(path, dpi=300 if layout == "column" else 200, facecolor="white")
    plt.close(fig)
    return True


CHECK_LAYOUTS = {
    "wide": dict(width=5.0, height=4.45, left=0.75, right=0.4, top=0.9, bottom=0.78,
                 label=10, small=9, note=8, dot=8.5),
    "column": dict(width=3.45, height=2.85, left=0.5, right=0.5, top=0.56, bottom=0.6,
                   label=7, small=6.5, note=5.5, dot=5.5),
}


def plot_lesson_checks(bank, path: Path, layout: str = "wide") -> bool:
    """Every lesson that reached the second check, on the development questions.

    Across: answers the lesson turned from right to wrong. Up: answers it turned
    from wrong to right. A lesson above the diagonal helped more than it hurt
    and was kept. A larger dot is several lessons with the same two counts.
    """

    plt = _pyplot()
    lessons = [
        lesson for lesson in (bank.lessons if bank else [])
        if lesson.dev_questions > 0 and lesson.fixes_source and lesson.usable
    ]
    if plt is None or not lessons:
        return False
    from matplotlib.lines import Line2D

    size = CHECK_LAYOUTS[layout]
    fig, ax = _frame(plt, size["width"], size["height"], size["left"], size["right"], size["top"], size["bottom"])
    most = max(max(lesson.dev_helped, lesson.dev_hurt) for lesson in lessons) + 1
    ax.set_xlim(0, most)
    ax.set_ylim(0, most)
    ax.plot([0, most], [0, most], color=AXIS, linewidth=0.9, zorder=1)

    groups: dict[tuple[int, int, bool], int] = {}
    for lesson in lessons:
        key = (lesson.dev_hurt, lesson.dev_helped, lesson.validated)
        groups[key] = groups.get(key, 0) + 1
    for (hurt, helped, kept), count in sorted(groups.items(), key=lambda item: -item[1]):
        ax.scatter([hurt], [helped], s=size["dot"] ** 2 * count, color=ACCENT if kept else MUTED,
                   edgecolor="white", linewidth=1.2, zorder=3 if kept else 2, clip_on=False)

    step = 1 if most <= 10 else 2 if most <= 20 else 5
    ax.set_xticks(range(0, most + 1, step))
    ax.set_yticks(range(0, most + 1, step))
    ax.tick_params(axis="both", labelsize=size["small"], colors=SECONDARY, length=0, pad=4)
    questions = lessons[0].dev_questions
    ax.set_xlabel(f"Right answers turned wrong, of {questions}", fontsize=size["small"], color=SECONDARY)
    ax.set_ylabel(f"Wrong answers turned right, of {questions}", fontsize=size["small"], color=SECONDARY)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    kept = sum(lesson.validated for lesson in lessons)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=size["dot"], markerfacecolor=ACCENT,
               markeredgecolor="white", label=f"Kept, helped more than it hurt ({kept})"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=size["dot"], markerfacecolor=MUTED,
               markeredgecolor="white", label=f"Dropped ({len(lessons) - kept})"),
        Line2D([0], [0], color=AXIS, linewidth=0.9, label="Helped as often as it hurt"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.995),
               ncol=2, frameon=False, fontsize=size["small"],
               handletextpad=0.4, columnspacing=1.4, borderaxespad=0.3, labelspacing=0.3)
    fig.text(0.988, 0.02, f"{len(lessons)} lessons, each tried alone. A larger dot is several lessons",
             ha="right", va="bottom", fontsize=size["note"], color=MUTED)
    fig.savefig(path, dpi=300 if layout == "column" else 200, facecolor="white")
    plt.close(fig)
    return True


def plot_all(report: dict, out_dir: Path, suffix: str = "", layout: str = "wide", bank=None) -> dict[str, Path]:
    """Draw every figure the report has data for. Returns {name: path} of those written.

    ``bank`` adds the lesson check figure. Files of the column layout end in
    ``_column.png`` so both sizes can sit in one folder.
    """

    tail = f"{suffix}{'_column' if layout == 'column' else ''}.png"
    written: dict[str, Path] = {}
    for name, draw in (
        ("comparison", plot_comparison), ("attempts", plot_attempts),
        ("fixed_broken", plot_fixed_broken), ("gain_bootstrap", plot_gain_bootstrap),
    ):
        target = out_dir / f"{name}{tail}"
        if draw(report, target, layout):
            written[name] = target
    if bank is not None:
        target = out_dir / f"lesson_checks{tail}"
        if plot_lesson_checks(bank, target, layout):
            written["lesson_checks"] = target
    return written
