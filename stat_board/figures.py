"""Deterministic plots for the PDF report, drawn from the DATA (never authored by
the model). Uses a headless matplotlib Agg backend and the report's teal palette.

Each function saves a PNG into ``assets_dir`` and the module returns a list of
{file, caption} so the report can embed and label them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

TEAL_DARK = "#0f5e6e"
TEAL = "#157d90"
TEAL_LIGHT = "#cbeef0"
ACCENT = "#b45309"
_DPI = 96  # keeps the report light; still crisp at the 460px display width


def _arrays(groups: dict[str, list[float]]) -> tuple[list[str], list[np.ndarray]]:
    names = list(groups.keys())
    return names, [np.asarray(groups[n], float) for n in names]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#333333", labelsize=8)
    ax.title.set_color(TEAL_DARK)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color("#333333")
        lbl.set_fontsize(9)


def _residuals(names, arrays) -> tuple[np.ndarray, np.ndarray]:
    """Pooled one-way residuals (value - group mean) and the fitted values."""
    resid = np.concatenate([a - a.mean() for a in arrays])
    fitted = np.concatenate([np.full(len(a), a.mean()) for a in arrays])
    return resid, fitted


def boxplot(groups, assets_dir: Path) -> dict[str, str]:
    names, arrays = _arrays(groups)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bp = ax.boxplot(arrays, tick_labels=names, patch_artist=True, widths=0.55,
                    medianprops=dict(color=TEAL_DARK, linewidth=1.6))
    for patch in bp["boxes"]:
        patch.set(facecolor=TEAL_LIGHT, edgecolor=TEAL_DARK)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set(color=TEAL_DARK)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arrays, 1):
        ax.scatter(np.full(len(a), i) + rng.uniform(-0.12, 0.12, len(a)), a,
                   s=14, color=TEAL, alpha=0.5, edgecolors="none", zorder=3)
    ax.set_ylabel("Value"); ax.set_title("Distribution by group (box + observations)")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_box.png"; fig.savefig(path, dpi=_DPI); plt.close(fig)
    return {"file": path.name, "caption": "Figure 1. Box plot of each group with individual observations overlaid."}


def means_ci(groups, assets_dir: Path, alpha: float = 0.05) -> dict[str, str]:
    names, arrays = _arrays(groups)
    means = [a.mean() for a in arrays]
    half = []
    for a in arrays:
        n = len(a)
        se = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        half.append(stats.t.ppf(1 - alpha / 2, n - 1) * se if n > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(names))
    ax.errorbar(x, means, yerr=half, fmt="o", color=TEAL_DARK, ecolor=TEAL,
                elinewidth=2, capsize=6, markersize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Mean"); ax.set_title(f"Group means with {int((1-alpha)*100)}% confidence intervals")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_means.png"; fig.savefig(path, dpi=_DPI); plt.close(fig)
    return {"file": path.name, "caption": f"Figure 2. Group means with {int((1-alpha)*100)}% CIs; non-overlapping bars suggest a difference."}


def residual_qq(groups, assets_dir: Path) -> dict[str, str]:
    names, arrays = _arrays(groups)
    resid, _ = _residuals(names, arrays)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=16, color=TEAL, alpha=0.7, edgecolors="none")
    ax.plot(osm, slope * osm + intercept, color=ACCENT, linewidth=1.5)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Ordered residuals")
    ax.set_title("Normal Q-Q plot of residuals")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_qq.png"; fig.savefig(path, dpi=_DPI); plt.close(fig)
    return {"file": path.name, "caption": "Figure 3. Normal Q-Q plot of model residuals; points on the line support the normality assumption."}


def residuals_vs_fitted(groups, assets_dir: Path) -> dict[str, str]:
    names, arrays = _arrays(groups)
    resid, fitted = _residuals(names, arrays)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.scatter(fitted, resid, s=16, color=TEAL, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Fitted value (group mean)"); ax.set_ylabel("Residual")
    ax.set_title("Residuals vs. fitted")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_resid.png"; fig.savefig(path, dpi=_DPI); plt.close(fig)
    return {"file": path.name, "caption": "Figure 4. Residuals vs. fitted values; even vertical spread across groups supports equal variance."}


def generate_all(groups, assets_dir: Path) -> list[dict[str, str]]:
    """Produce the standard figure set. Residual diagnostics need >=2 groups and
    enough points to be meaningful."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    figs = [boxplot(groups, assets_dir), means_ci(groups, assets_dir)]
    total_n = sum(len(v) for v in groups.values())
    if len(groups) >= 2 and total_n >= 6:
        figs.append(residual_qq(groups, assets_dir))
        figs.append(residuals_vs_fitted(groups, assets_dir))
    return figs
