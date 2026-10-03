"""Figures for the memo.

Three plots, each earning its place by answering a question the numbers alone
answer badly:

- ``options_chart``   -- what shape is each option's tradeoff?
- ``fds_chart``       -- how confidently can each option predict, and where does
  it get shaky?
- ``run_layout_chart`` -- where in the design space do the runs actually land,
  and what corners are never visited?
- ``power_curve_chart`` -- how many runs does it take to see the effect you
  care about, and where do the options sit on that curve?

Matplotlib with the Agg backend so this works headless on a locked-down laptop.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .candidates import ScoredDesign  # noqa: E402
from .designs.properties import PowerCurve  # noqa: E402
from .designs.spec import DesignSpec  # noqa: E402

# Matches the PDF stylesheet in pdstat.report so figures and document read as
# one artefact rather than two.
TEAL = "#0f5e6e"
LIGHT_TEAL = "#7fb8c2"
AMBER = "#d98c00"
GREY = "#8a8a8a"
PALETTE = [TEAL, AMBER, LIGHT_TEAL, GREY]

AXIS_LABELS = {
    "power": "Power",
    "aliasing": "Clarity\n(no aliasing)",
    "prediction": "Prediction\nprecision",
    "robustness": "Robustness\nto losses",
}


def _finish(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def options_chart(options: list[ScoredDesign], path: str | Path) -> Path:
    """Grouped bars: each option's score on each axis.

    The point is the *shape*, not the heights. A design that is tall everywhere
    except aliasing is telling a very different story from one that is merely
    mediocre throughout, and that distinction is invisible in a single total.
    """
    path = Path(path)
    axes_order = [
        a for a in ("power", "aliasing", "prediction", "robustness") if any(a in o.sub_scores for o in options)
    ]
    # Sized and scaled to match fds_chart: both are placed at the same width in
    # the PDF, so anything smaller here renders illegible next to it.
    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    width = 0.8 / max(len(options), 1)
    positions = np.arange(len(axes_order))

    for i, option in enumerate(options):
        values = [option.sub_scores.get(a, 0.0) for a in axes_order]
        offset = (i - (len(options) - 1) / 2) * width
        label = f"{option.design.name} ({option.n_runs} runs)"
        ax.bar(positions + offset, values, width * 0.92, label=label, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(positions)
    ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in axes_order], fontsize=11)
    # Headroom so the in-axes legend never sits on top of a full-height bar.
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("score (higher is better)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.axhline(0.8, color=GREY, linestyle=":", linewidth=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=1)
    ax.set_title("How the options compare on each axis", fontsize=12, color=TEAL, loc="left")
    return _finish(fig, path)


def fds_chart(options: list[ScoredDesign], path: str | Path) -> Path:
    """Fraction-of-design-space curve: sorted prediction variance.

    Read it as "for what fraction of the design space is my prediction
    uncertainty below this level?". A curve that stays low and flat predicts
    evenly everywhere; one that rears up at the right predicts badly somewhere,
    usually near the edges — which is exactly where range-setting questions get
    asked.
    """
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    plotted = 0
    for i, option in enumerate(options):
        curve = option.properties.prediction.fds_curve
        if not curve or not np.all(np.isfinite(curve)):
            continue
        fraction = np.linspace(0, 1, len(curve))
        ax.plot(
            fraction,
            curve,
            color=PALETTE[i % len(PALETTE)],
            linewidth=2,
            label=f"{option.design.name} ({option.n_runs} runs)",
        )
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "no design with finite prediction variance", ha="center", va="center", color=GREY)
        ax.set_axis_off()
        return _finish(fig, path)

    ax.set_xlabel("fraction of the design space", fontsize=9)
    ax.set_ylabel("scaled prediction variance\n(lower is better)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Prediction uncertainty across the design space", fontsize=10.5, color=TEAL, loc="left")
    return _finish(fig, path)


def run_layout_chart(option: ScoredDesign, spec: DesignSpec, path: str | Path, max_pairs: int = 3) -> Path:
    """Where the runs land, projected onto factor pairs, in real units.

    Real units rather than coded, because "pH 6.8 to 7.4" is a thing a scientist
    can sanity-check against their process and "-1 to +1" is not.
    """
    path = Path(path)
    pairs = list(combinations(range(spec.n_factors), 2))[:max_pairs]
    if not pairs:
        pairs = [(0, 0)]

    fig, axes = plt.subplots(1, len(pairs), figsize=(2.6 * len(pairs), 2.8), squeeze=False)
    decoded = option.design.decoded(spec.factors)

    for ax, (i, j) in zip(axes[0], pairs, strict=True):
        x, y = decoded[:, i], decoded[:, j]
        # Jitter-free: overlapping runs are shown by size, so replication and
        # centre points read as "many runs here" rather than as a single point.
        points, counts = np.unique(np.column_stack([x, y]), axis=0, return_counts=True)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=28 + 22 * (counts - 1),
            color=TEAL,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        fi, fj = spec.factors[i], spec.factors[j]
        ax.set_xlabel(f"{fi.name}{f' ({fi.units})' if fi.units else ''}", fontsize=8)
        ax.set_ylabel(f"{fj.name}{f' ({fj.units})' if fj.units else ''}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Where the runs land — {option.design.name}",
        fontsize=10.5,
        color=TEAL,
        x=0.02,
        ha="left",
    )
    return _finish(fig, path)


def power_curve_chart(curve: PowerCurve, options: list[ScoredDesign], spec: DesignSpec, path: str | Path) -> Path:
    """Power against run count, with the options marked on it.

    The line is the best case — an ideal orthogonal design — so an option below
    the line is spending runs on something other than power (centre points,
    replicates), which is often deliberate. The dashed line is the same curve if
    the noise is worse than entered (the stress factor): the gap between them is the price of being
    wrong about the SD.
    """
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    response = spec.primary_response
    units = f" {response.units}" if response and response.units else ""
    label = f"target effect {response.target_effect:g}{units}"
    ax.plot(curve.n_runs, curve.power, color=TEAL, linewidth=2, label=label)
    ax.plot(
        curve.n_runs,
        curve.power_if_noise_high,
        color=TEAL,
        linewidth=1.4,
        linestyle="--",
        label=f"same, if the noise SD is really {curve.noise_stress_factor:.2g}x larger",
    )
    ax.axhline(0.8, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(curve.n_runs[0], 0.815, "80% power", fontsize=8, color=GREY, va="bottom")

    for i, option in enumerate(options):
        power = option.properties.power.min_main_effect_power
        if power is None:
            continue
        ax.scatter(
            [option.n_runs],
            [power],
            s=48,
            color=PALETTE[i % len(PALETTE)],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
            label=f"{option.design.name} ({option.n_runs} runs, {power:.0%})",
        )

    ax.set_xlabel("runs", fontsize=9)
    ax.set_ylabel("power (chance of finding the effect)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(curve.n_runs[0], curve.n_runs[-1])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{v:.0%}" for v in ax.get_yticks()])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    title = "How many runs it takes to see your target effect"
    if curve.runs_for_80 is not None:
        title += f" — about {curve.runs_for_80} for 80%"
    ax.set_title(title, fontsize=10.5, color=TEAL, loc="left")
    return _finish(fig, path)
