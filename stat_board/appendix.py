"""Build the report's deterministic appendix — figures + detailed statistics
tables — straight from the dataset. Everything here is computed by the engine, so
the plots and every cell are grounded in the data, not written by the model.

`build()` returns (markdown, assets_dir): the Markdown to append to the report,
and the directory holding the figure PNGs (used as the PDF renderer's image root).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .engine import analyses
from .engine.data import DataError, load_dataframe, load_groups


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


def _formula_lhs(formula: str) -> str:
    """The raw (unwrapped) outcome column name from a patsy formula's left side."""
    lhs = formula.split("~", 1)[0].strip()
    m = re.match(r"^Q\((['\"])(.*)\1\)$", lhs)
    return m.group(2) if m else lhs


def build_multifactor(data_path: str, formula: str, *, factors: list[str] | None = None,
                      typ: int = 2, alpha: float = 0.05,
                      assets_dir: Path | None = None) -> tuple[str, Path] | None:
    """Multi-factor/DoE counterpart to `build()`. `regression`/`two_way_anova`/
    `ancova` all reduce to one fitted patsy-formula model, so this only needs the
    formula itself (not which of the three commands produced it) plus, optionally,
    the DoE factor columns (enables the design-coverage/curvature/optimum-ranking
    sections — the fact base a "Recommended Next Experiments" section is written
    from). Returns None if the model can't be fit (mirrors `build()`'s contract).
    """
    value = _formula_lhs(formula)
    try:
        df = load_dataframe(data_path)
        fit = analyses.regression(data_path, formula, typ=typ, alpha=alpha)
        pred = analyses.predict_table(data_path, formula, alpha=alpha)
        vif = analyses.vif_table(data_path, formula)
    except Exception:  # noqa: BLE001 -- patsy/statsmodels raise many error types
        # for a malformed formula or unfittable model; any of them means "no
        # appendix", never a crashed report.
        return None

    assets_dir = assets_dir or Path(data_path).resolve().parent / "_assets"
    resid = np.array([r["residual"] for r in pred["rows"]])
    fitted = np.array([r["predicted"] for r in pred["rows"]])
    figs = figures.generate_all_multifactor(pred["rows"], fit["anova"], resid, fitted, assets_dir)

    factors = factors or []
    continuous = [f for f in factors if f in df.columns
                 and pd.api.types.is_numeric_dtype(df[f]) and df[f].nunique() > 2]
    if len(continuous) == 2:
        figs.append(figures.contour(data_path, formula, continuous[0], continuous[1], assets_dir))

    _BREAK = '<div style="page-break-before: always"></div>'
    parts: list[str] = [_BREAK, "## Appendix A — Figures"]
    for i, fg in enumerate(figs):
        if i > 0 and i % 2 == 0:
            parts.append(_BREAK)
        parts.append(
            f'<figure style="text-align:center;margin:6pt 0">'
            f'<img src="{fg["file"]}" width="460">'
            f'<figcaption><em>{fg["caption"]}</em></figcaption></figure>')
    parts.append(_BREAK)

    parts.append("\n## Appendix B — Model tables\n")
    parts.append("**Coefficients**\n")
    parts.append(_table(
        ["Term", "Coefficient", f"{int((1 - alpha) * 100)}% CI", "p"],
        [[name, _f(c["coef"]), f"[{_f(c['ci'][0])}, {_f(c['ci'][1])}]", _f(c["p"])]
         for name, c in fit["coefficients"].items()]))
    parts.append(f"\n**ANOVA term table (Type {typ} SS)**\n")
    parts.append(_table(
        ["Term", "SS", "df", "F", "p", "Partial η²", "Significant"],
        [[t["term"], _f(t["sum_sq"]), _f(t["df"]), _f(t["F"]), _f(t["p"]),
          _f(t["partial_eta_sq"]), "yes" if t["significant"] else "no"] for t in fit["anova"]]))
    parts.append("\n**Goodness of fit**\n")
    parts.append(_table(
        ["R²", "Adj R²", "AIC", "BIC"],
        [[_f(fit["r_squared"]), _f(fit["adj_r_squared"]), _f(fit["aic"]), _f(fit["bic"])]]))

    rd = fit["residual_diagnostics"]
    checks = [["Autocorrelation (Durbin–Watson)", _f(rd["durbin_watson"]), ""]]
    if "resid_shapiro_W" in rd:
        checks.append(["Normality of residuals (Shapiro–Wilk)",
                       f"W={_f(rd['resid_shapiro_W'])}, p={_f(rd['resid_shapiro_p'])}",
                       "OK" if rd["resid_normal_at_alpha"] else "violated"])
    parts.append("\n**Residual diagnostics**\n")
    parts.append(_table(["Check", "Statistic", "Verdict"], checks))

    parts.append("\n## Appendix C — Design diagnostics\n")
    parts.append(f"**Per-run predictions** ({pred['n_flagged']} of {pred['n']} run(s) flagged as "
                 f"high-leverage or influential — leverage > {_f(pred['leverage_threshold'])}, "
                 f"Cook's D > {_f(pred['cooks_d_threshold'])})\n")
    parts.append(_table(
        ["Row", "Observed", "Predicted", "Residual", "Leverage", "Cook's D", "Flag"],
        [[str(r["row"]), _f(r["observed"]), _f(r["predicted"]), _f(r["residual"]),
          _f(r["leverage"]), _f(r["cooks_d"]),
          "influential" if r["influential"] else ("high-leverage" if r["high_leverage"] else "")]
         for r in pred["rows"]]))
    parts.append("\n**Variance inflation factors**\n")
    parts.append(_table(
        ["Term", "VIF", "Concern"],
        [[t["term"], _f(t["vif"]), "yes" if t["collinearity_concern"] else "no"]
         for t in vif["terms"]]))

    if factors:
        try:
            cov = analyses.design_coverage(data_path, factors, value=value)
            opt = analyses.doe_optimum(data_path, formula, factors, value)
        except Exception:  # noqa: BLE001 -- same rationale as the outer fit above
            cov = opt = None

        if cov is not None:
            parts.append("\n**Design coverage**\n")
            parts.append(_table(
                ["Factor", "Levels tested"],
                [[f, ", ".join(str(v) for v in cov["levels"][f])] for f in factors]))
            parts.append(f"\n{cov['n_tested_combinations']} of {cov['n_possible_combinations']} "
                         f"possible factor-level combination(s) tested "
                         f"({_f(cov['coverage_fraction'] * 100)}% coverage).\n")
            if cov["missing_combinations"]:
                shown = "; ".join(", ".join(f"{k}={v}" for k, v in m.items())
                                 for m in cov["missing_combinations"][:20])
                parts.append(f"\nUntested combination(s): {shown}\n")
            if cov["curvature"]:
                c = cov["curvature"]
                if c["p"] is None:
                    verdict = "untestable (fewer than 2 center-point runs)"
                else:
                    verdict = ("significant curvature detected" if c["p"] < alpha
                              else "no significant curvature detected")
                power_note = (" Note: very few center-point degrees of freedom — low power "
                             "to detect real curvature." if c["low_power_warning"] else "")
                p_part = f", p={_f(c['p'])}" if c["p"] is not None else ""
                parts.append(f"\n**Curvature check** ({cov['n_center_points']} center-point run(s)): "
                             f"corner mean={_f(c['corner_mean'])}, center mean={_f(c['center_mean'])}"
                             f"{p_part} — {verdict}.{power_note}\n")

        if opt is not None:
            parts.append("\n**Tested combinations ranked by predicted response**\n")
            parts.append(_table(
                [*factors, "Predicted", "Observed mean", "Tested"],
                [[_f(r[f]) for f in factors] + [_f(r["predicted"]), _f(r.get("observed_mean")),
                  "yes" if r["tested"] else "no"] for r in opt["rows"]]))
            parts.append("\n**Boundary check** (is the best-tested setting at the edge of the "
                         "tested range?)\n")
            parts.append(_table(
                ["Factor", "Best setting", "Position"],
                [[f, _f(opt["best"][f]), opt["boundary_flags"].get(f, "")] for f in factors]))

    return "\n\n".join(parts) + "\n", assets_dir
