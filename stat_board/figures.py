"""Deterministic plots for the PDF report, drawn from the DATA (never authored by
the model). Uses a headless matplotlib Agg backend and the report's teal palette.

Each function saves a PNG into ``assets_dir`` and the module returns a list of
{file, caption} so the report can embed and label them.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

TEAL_DARK = "#0f5e6e"
TEAL = "#157d90"
TEAL_LIGHT = "#cbeef0"
ACCENT = "#b45309"
_DPI = 96  # keeps the report light; still crisp at the 460px display width


def _save(fig, path: Path) -> None:
    """Write the figure as a flat, opaque DeviceRGB PNG — no alpha/SMask and no
    ICC profile — so it renders in every PDF viewer (Preview, Chrome, PDF.js),
    not only MuPDF. matplotlib's default RGBA + sRGB-ICC PNGs get embedded as
    ICCBased images with a soft mask, which several common viewers draw blank."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.save(path, format="PNG")  # a fresh RGB image carries no ICC profile


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
    path = assets_dir / "fig_box.png"; _save(fig, path)
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
    path = assets_dir / "fig_means.png"; _save(fig, path)
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
    path = assets_dir / "fig_qq.png"; _save(fig, path)
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
    path = assets_dir / "fig_resid.png"; _save(fig, path)
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


# --------------------------------------------------------------------------- #
# Multi-factor / DoE model figures — general fitted-model diagnostics, not tied
# to the one-way `groups` shape above. `predicted`/`observed`/`resid`/`fitted`
# come from `stat_board.engine.analyses.predict_table`; `terms` comes from the
# `anova` list already returned by `regression`/`two_way_anova`/`ancova`.
# --------------------------------------------------------------------------- #

def predicted_vs_observed(rows: list[dict], assets_dir: Path) -> dict[str, str]:
    observed = np.array([r["observed"] for r in rows], float)
    predicted = np.array([r["predicted"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    lo, hi = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
    pad = (hi - lo) * 0.05 or 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999", linewidth=1, linestyle="--")
    ax.scatter(observed, predicted, s=16, color=TEAL, alpha=0.7, edgecolors="none")
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs. observed")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_pred_obs.png"; _save(fig, path)
    return {"file": path.name,
            "caption": "Figure. Predicted vs. observed; points on the dashed line indicate a perfect fit."}


def pareto_effects(terms: list[dict], assets_dir: Path) -> dict[str, str]:
    """Bar chart of each model term's partial eta^2, sorted largest first — the
    classic DoE 'Pareto of effects' view of which terms move the response most."""
    ordered = sorted(terms, key=lambda t: t.get("partial_eta_sq") or 0, reverse=True)
    names = [t["term"] for t in ordered]
    magnitudes = [t.get("partial_eta_sq") or 0 for t in ordered]
    colors = [ACCENT if t.get("significant") else TEAL_LIGHT for t in ordered]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(names))
    ax.barh(y, magnitudes, color=colors, edgecolor=TEAL_DARK)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Partial eta^2"); ax.set_title("Pareto of model term effects")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_pareto.png"; _save(fig, path)
    return {"file": path.name,
            "caption": "Figure. Model terms ranked by effect size (partial eta^2); "
                      "solid bars are significant at the report's alpha."}


def regression_qq(resid: np.ndarray, assets_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=16, color=TEAL, alpha=0.7, edgecolors="none")
    ax.plot(osm, slope * osm + intercept, color=ACCENT, linewidth=1.5)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Ordered residuals")
    ax.set_title("Normal Q-Q plot of residuals")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_qq.png"; _save(fig, path)
    return {"file": path.name,
            "caption": "Figure. Normal Q-Q plot of model residuals; points on the line "
                      "support the normality assumption."}


def regression_resid_vs_fitted(resid: np.ndarray, fitted: np.ndarray, assets_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.scatter(fitted, resid, s=16, color=TEAL, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Fitted value"); ax.set_ylabel("Residual")
    ax.set_title("Residuals vs. fitted")
    _style(ax); fig.tight_layout()
    path = assets_dir / "fig_resid.png"; _save(fig, path)
    return {"file": path.name,
            "caption": "Figure. Residuals vs. fitted values; even vertical spread with no "
                      "trend supports the equal-variance assumption."}


def contour(path, formula: str, factor_x: str, factor_y: str, assets_dir: Path,
           n_grid: int = 40) -> dict[str, str]:
    """Filled contour of predicted response over factor_x/factor_y's tested
    range, with every other model term held at its mean (numeric) or mode
    (categorical) — the classic DoE response-surface visual. Caller is
    responsible for confirming factor_x/factor_y are the only two continuous
    factors before calling this (see `appendix.build_multifactor`)."""
    import pandas as pd
    import statsmodels.formula.api as smf

    from .engine.data import load_dataframe

    df = load_dataframe(path)
    model = smf.ols(formula, data=df).fit()

    ref = {}
    for c in df.columns:
        if c in (factor_x, factor_y):
            continue
        ref[c] = df[c].mean() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]

    xs = np.linspace(df[factor_x].min(), df[factor_x].max(), n_grid)
    ys = np.linspace(df[factor_y].min(), df[factor_y].max(), n_grid)
    grid = pd.DataFrame([{**ref, factor_x: x, factor_y: y} for y in ys for x in xs])
    z = model.predict(grid).to_numpy().reshape(len(ys), len(xs))

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cs = ax.contourf(xs, ys, z, levels=14, cmap="viridis")
    ax.scatter(df[factor_x], df[factor_y], s=14, color="white", edgecolors=TEAL_DARK, zorder=3)
    fig.colorbar(cs, ax=ax, label="Predicted response")
    ax.set_xlabel(factor_x); ax.set_ylabel(factor_y)
    ax.set_title(f"Predicted response over {factor_x} x {factor_y}")
    ax.tick_params(colors="#333333", labelsize=8)
    fig.tight_layout()
    path_out = assets_dir / "fig_contour.png"; _save(fig, path_out)
    return {"file": path_out.name,
            "caption": f"Figure. Predicted response surface over {factor_x} and {factor_y} "
                      f"(other model terms held at their mean/mode); white points are actual runs."}


def generate_all_multifactor(rows: list[dict], terms: list[dict], resid: np.ndarray,
                             fitted: np.ndarray, assets_dir: Path) -> list[dict[str, str]]:
    """Standard multi-factor/DoE figure set (contour is added separately by
    the caller when exactly 2 continuous factors are present — see
    `appendix.build_multifactor`)."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    return [
        predicted_vs_observed(rows, assets_dir),
        regression_resid_vs_fitted(resid, fitted, assets_dir),
        regression_qq(resid, assets_dir),
        pareto_effects(terms, assets_dir),
    ]
