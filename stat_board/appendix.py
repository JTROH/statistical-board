"""Build the report's deterministic appendix — figures + detailed statistics
tables — straight from the dataset. Everything here is computed by the engine, so
the plots and every cell are grounded in the data, not written by the model.

`build()` returns (markdown, assets_dir): the Markdown to append to the report,
and the directory holding the figure PNGs (used as the PDF renderer's image root).
"""

from __future__ import annotations

from pathlib import Path

from . import figures
from .engine import analyses
from .engine.data import DataError, load_groups


def _f(x, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, bool):
            return "yes" if x else "no"
        return f"{x:.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}\n"


def build(data_path: str, *, group_col: str | None = None, value_col: str | None = None,
          alpha: float = 0.05, assets_dir: Path | None = None) -> tuple[str, Path] | None:
    """Return (markdown, assets_dir) for the appendix, or None if the data can't
    be read as grouped numeric data (in which case no appendix is added)."""
    try:
        groups = load_groups(data_path, group_col=group_col, value_col=value_col)
    except (DataError, ValueError, OSError):
        return None
    if not groups:
        return None

    assets_dir = assets_dir or Path(data_path).resolve().parent / "_assets"
    figs = figures.generate_all(groups, assets_dir)

    # Figures, two per page. Story shrinks an image to fit leftover space, so we
    # force page breaks (page-break-before is honored; break-inside:avoid is not).
    _BREAK = '<div style="page-break-before: always"></div>'
    parts: list[str] = [_BREAK, "## Appendix A — Figures"]
    for i, fig in enumerate(figs):
        if i > 0 and i % 2 == 0:
            parts.append(_BREAK)
        parts.append(
            f'<figure style="text-align:center;margin:6pt 0">'
            f'<img src="{fig["file"]}" width="460">'
            f'<figcaption><em>{fig["caption"]}</em></figcaption></figure>')
    parts.append(_BREAK)

    parts.append("\n## Appendix B — Detailed statistics\n")

    # Descriptives
    desc = analyses.describe(groups, alpha)["groups"]
    parts.append("**Descriptive statistics**\n")
    parts.append(_table(
        ["Group", "n", "Mean", "SD", "SEM", "Median", f"{int((1-alpha)*100)}% CI (mean)"],
        [[g, _f(s["n"]), _f(s["mean"]), _f(s["sd"]), _f(s["sem"]), _f(s["median"]),
          f"[{_f(s['ci95_mean'][0])}, {_f(s['ci95_mean'][1])}]"] for g, s in desc.items()]))

    if len(groups) >= 2:
        # ANOVA source table + goodness of fit
        at = analyses.anova_table(groups, alpha=alpha)
        parts.append("\n**One-way ANOVA source table**\n")
        parts.append(_table(
            ["Source", "SS", "df", "MS", "F", "p"],
            [[r["source"], _f(r["SS"]), _f(r["df"]), _f(r["MS"]), _f(r["F"]), _f(r["p"])]
             for r in at["rows"]]))
        g = at["goodness_of_fit"]
        parts.append("\n**Goodness of fit**\n")
        parts.append(_table(
            ["R²", "η²", "ω²", "ε²", "Root MSE"],
            [[_f(g["r_squared"]), _f(g["eta_squared"]), _f(g["omega_squared"]),
              _f(g["epsilon_squared"]), _f(g["root_mse"])]]))

        # Residual diagnostics
        rd = analyses.residual_diagnostics(groups, alpha=alpha)
        parts.append("\n**Residual diagnostics** (value − group mean)\n")
        parts.append(_table(
            ["n", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"],
            [[_f(rd["n"]), _f(rd["mean"]), _f(rd["sd"]), _f(rd["min"]), _f(rd["q1"]),
              _f(rd["median"]), _f(rd["q3"]), _f(rd["max"])]]))
        checks = []
        if "normality" in rd:
            n = rd["normality"]
            checks.append(["Normality of residuals (Shapiro–Wilk)",
                           f"W={_f(n['shapiro_W'])}, p={_f(n['shapiro_p'])}",
                           "OK" if n["normal_at_alpha"] else "violated"])
        if "homoscedasticity" in rd:
            h = rd["homoscedasticity"]
            checks.append(["Equal variance (Levene)",
                           f"W={_f(h['levene_W'])}, p={_f(h['levene_p'])}",
                           "OK" if h["equal_variance_at_alpha"] else "violated"])
        if checks:
            parts.append("\n" + _table(["Assumption", "Statistic", "Verdict"], checks))

        # Scaled pairwise effect sizes
        pe = analyses.pairwise_effects(groups, alpha=alpha)["pairs"]
        parts.append("\n**Scaled effect sizes (pairwise)**\n")
        parts.append(_table(
            ["Pair", "Mean diff", f"{int((1-alpha)*100)}% CI", "Cohen's d", "Hedges' g", "Magnitude"],
            [[p["pair"], _f(p["mean_diff"]), f"[{_f(p['ci'][0])}, {_f(p['ci'][1])}]",
              _f(p["cohens_d"]), _f(p["hedges_g"]), p["magnitude"]] for p in pe]))

    return "\n\n".join(parts) + "\n", assets_dir
