"""The analysis functions. Each returns a plain, JSON-serializable dict so an
agent can quote exact numbers and so results survive a round-trip to disk.

Design rules that keep the board honest:
  * Every test reports its inputs (n, alpha, which variant) alongside outputs.
  * Comparisons carry an effect size, not just a p-value.
  * A short ``conclusion`` states the decision at the given alpha in words, so a
    critic can check that the prose matches the number.
Nothing here decides *whether* a test was appropriate — that's the assumptions
skeptic's job, fed by ``check_assumptions``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from statsmodels.stats.weightstats import ttost_ind as _sm_tost

from . import bayes, effects
from .data import Groups, require_two

_Z = 1.959963984540054  # 97.5th percentile of the standard normal


def _ci_mean(x: np.ndarray, alpha: float = 0.05) -> list[float]:
    n = len(x)
    if n < 2:
        return [float("nan"), float("nan")]
    se = x.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    m = x.mean()
    return [float(m - crit * se), float(m + crit * se)]


def describe(groups: Groups, alpha: float = 0.05) -> dict[str, Any]:
    out = {}
    for name, vals in groups.items():
        x = np.asarray(vals, float)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        out[name] = {
            "n": int(len(x)),
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"),
            "median": float(med),
            "iqr": [float(q1), float(q3)],
            "min": float(x.min()),
            "max": float(x.max()),
            "ci95_mean": _ci_mean(x, alpha),
        }
    return {"analysis": "describe", "alpha": alpha, "groups": out}


def check_assumptions(groups: Groups, alpha: float = 0.05) -> dict[str, Any]:
    """Normality (Shapiro-Wilk) per group, homogeneity of variance (Levene, and
    Bartlett which assumes normality), and IQR-rule outlier counts. These are the
    facts the skeptic uses to decide whether a parametric test is even valid."""
    normality = {}
    for name, vals in groups.items():
        x = np.asarray(vals, float)
        if len(x) < 3:
            normality[name] = {"n": int(len(x)), "note": "n<3, Shapiro not computed"}
            continue
        w, p = stats.shapiro(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        normality[name] = {
            "n": int(len(x)),
            "shapiro_W": float(w),
            "shapiro_p": float(p),
            "normal_at_alpha": bool(p > alpha),
            "outliers_iqr": n_out,
        }

    arrays = [np.asarray(v, float) for v in groups.values() if len(v) >= 2]
    homogeneity: dict[str, Any] = {}
    if len(arrays) >= 2:
        lev_w, lev_p = stats.levene(*arrays, center="median")
        homogeneity["levene"] = {
            "W": float(lev_w), "p": float(lev_p),
            "equal_variance_at_alpha": bool(lev_p > alpha),
        }
        if all(len(a) >= 3 for a in arrays):
            bar_t, bar_p = stats.bartlett(*arrays)
            homogeneity["bartlett"] = {
                "T": float(bar_t), "p": float(bar_p),
                "equal_variance_at_alpha": bool(bar_p > alpha),
                "note": "assumes normality; prefer Levene if in doubt",
            }

    return {
        "analysis": "assumptions",
        "alpha": alpha,
        "normality": normality,
        "homogeneity": homogeneity,
    }


def ttest(
    groups: Groups, *, paired: bool = False, equal_var: bool = False, alpha: float = 0.05
) -> dict[str, Any]:
    """Two-sample t-test. Default is Welch (equal_var=False) — the safer choice
    when variances may differ. Reports Cohen's d and Hedges' g."""
    n1, v1, n2, v2 = require_two(groups)
    a, b = np.asarray(v1, float), np.asarray(v2, float)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired t-test needs equal-length groups")
        t, p = stats.ttest_rel(a, b)
        variant = "paired"
    else:
        t, p = stats.ttest_ind(a, b, equal_var=equal_var)
        variant = "Student (pooled)" if equal_var else "Welch"
    d = effects.cohens_d(v1, v2)
    g = effects.hedges_g(v1, v2)
    diff = float(a.mean() - b.mean())
    return {
        "analysis": "ttest",
        "variant": variant,
        "groups": [n1, n2],
        "n": [int(len(a)), int(len(b))],
        "mean_diff": diff,
        "t": float(t),
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "cohens_d": d,
        "hedges_g": g,
        "effect_magnitude": effects.interpret_d(d),
        "conclusion": (
            f"{'Significant' if p < alpha else 'No significant'} difference between "
            f"{n1} and {n2} (p={p:.4g}); effect size d={d:.3f} ({effects.interpret_d(d)})."
        ),
    }


def mann_whitney(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Non-parametric two-sample test — the fallback when normality fails."""
    n1, v1, n2, v2 = require_two(groups)
    a, b = np.asarray(v1, float), np.asarray(v2, float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial effect size
    rbc = 1 - (2 * u) / (len(a) * len(b))
    return {
        "analysis": "mann_whitney",
        "groups": [n1, n2],
        "n": [int(len(a)), int(len(b))],
        "U": float(u),
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "rank_biserial": float(rbc),
        "conclusion": (
            f"{'Significant' if p < alpha else 'No significant'} difference in "
            f"distributions of {n1} vs {n2} (Mann-Whitney U, p={p:.4g})."
        ),
    }


def _ss_decomposition(groups: Groups) -> dict[str, float]:
    arrays = [np.asarray(v, float) for v in groups.values()]
    allv = np.concatenate(arrays)
    grand = allv.mean()
    ss_total = float(((allv - grand) ** 2).sum())
    ss_between = float(sum(len(a) * (a.mean() - grand) ** 2 for a in arrays))
    ss_within = ss_total - ss_between
    k = len(arrays)
    N = len(allv)
    df_b, df_w = k - 1, N - k
    return {
        "ss_total": ss_total, "ss_between": ss_between, "ss_within": ss_within,
        "df_between": df_b, "df_within": df_w,
        "ms_between": ss_between / df_b if df_b else float("nan"),
        "ms_within": ss_within / df_w if df_w else float("nan"),
    }


def anova(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Classic one-way ANOVA with eta-squared and omega-squared effect sizes."""
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least 2 groups")
    arrays = [np.asarray(v, float) for v in groups.values()]
    f, p = stats.f_oneway(*arrays)
    ss = _ss_decomposition(groups)
    eta = effects.eta_squared(ss["ss_between"], ss["ss_total"])
    omega = effects.omega_squared(
        ss["ss_between"], ss["df_between"], ss["ms_within"], ss["ss_total"]
    )
    return {
        "analysis": "anova",
        "variant": "one-way (assumes equal variance)",
        "groups": list(groups.keys()),
        "F": float(f),
        "df": [ss["df_between"], ss["df_within"]],
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "eta_squared": eta,
        "omega_squared": omega,
        "effect_magnitude": effects.interpret_eta(eta),
        "conclusion": (
            f"{'At least one group differs' if p < alpha else 'No detectable difference'} "
            f"(F={f:.4g}, p={p:.4g}); eta^2={eta:.3f} ({effects.interpret_eta(eta)})."
        ),
    }


def welch_anova(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Welch's ANOVA — robust to unequal variances (does not assume homogeneity)."""
    names = list(groups.keys())
    arrays = [np.asarray(v, float) for v in groups.values()]
    k = len(arrays)
    if k < 2:
        raise ValueError("Welch ANOVA needs at least 2 groups")
    n = np.array([len(a) for a in arrays], float)
    means = np.array([a.mean() for a in arrays])
    var = np.array([a.var(ddof=1) for a in arrays])
    w = n / var
    sw = w.sum()
    xbar = (w * means).sum() / sw
    numer = (w * (means - xbar) ** 2).sum() / (k - 1)
    lam = ((1 - w / sw) ** 2 / (n - 1)).sum()
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * lam
    f = numer / denom
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * lam)
    p = stats.f.sf(f, df1, df2)
    return {
        "analysis": "welch_anova",
        "variant": "Welch (unequal variances OK)",
        "groups": names,
        "F": float(f),
        "df": [float(df1), float(df2)],
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "conclusion": (
            f"{'At least one group differs' if p < alpha else 'No detectable difference'} "
            f"(Welch F={f:.4g}, p={p:.4g})."
        ),
    }


def kruskal(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Kruskal-Wallis — the non-parametric one-way test."""
    arrays = [np.asarray(v, float) for v in groups.values()]
    if len(arrays) < 2:
        raise ValueError("Kruskal-Wallis needs at least 2 groups")
    h, p = stats.kruskal(*arrays)
    return {
        "analysis": "kruskal",
        "groups": list(groups.keys()),
        "H": float(h),
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "conclusion": (
            f"{'At least one group differs' if p < alpha else 'No detectable difference'} "
            f"(Kruskal-Wallis H={h:.4g}, p={p:.4g})."
        ),
    }


def tukey_posthoc(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Tukey HSD — which pairs differ, with family-wise error controlled."""
    labels, values = [], []
    for name, vals in groups.items():
        labels += [name] * len(vals)
        values += list(vals)
    res = pairwise_tukeyhsd(np.asarray(values, float), np.asarray(labels), alpha=alpha)
    pairs = []
    for row in res.summary().data[1:]:
        g1, g2, meandiff, p_adj, lower, upper, reject = row
        pairs.append({
            "group1": g1, "group2": g2,
            "mean_diff": float(meandiff), "p_adj": float(p_adj),
            "ci95": [float(lower), float(upper)], "reject_null": bool(reject),
        })
    return {"analysis": "tukey_posthoc", "alpha": alpha, "pairs": pairs}


def tost(
    groups: Groups,
    *,
    low: float | None = None,
    high: float | None = None,
    margin_pct: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two one-sided tests for *equivalence*. Bounds are on the mean difference
    (group1 - group2). Give absolute bounds via ``low``/``high``, or a symmetric
    relative margin via ``margin_pct`` (percent of |mean of group1|)."""
    n1, v1, n2, v2 = require_two(groups)
    a, b = np.asarray(v1, float), np.asarray(v2, float)
    if margin_pct is not None:
        bound = margin_pct / 100.0 * abs(a.mean())
        low, high = -bound, bound
    if low is None or high is None:
        raise ValueError("provide --low and --high, or --margin-pct")
    p, lower_res, upper_res = _sm_tost(a, b, low, high)
    equivalent = p < alpha
    return {
        "analysis": "tost",
        "groups": [n1, n2],
        "bounds": [float(low), float(high)],
        "mean_diff": float(a.mean() - b.mean()),
        "p": float(p),
        "p_lower": float(lower_res[1]),
        "p_upper": float(upper_res[1]),
        "alpha": alpha,
        "equivalent": bool(equivalent),
        "conclusion": (
            f"Means are {'equivalent' if equivalent else 'NOT shown to be equivalent'} "
            f"within [{low:.4g}, {high:.4g}] (TOST p={p:.4g})."
        ),
    }


def bayes_ttest(
    groups: Groups, *, paired: bool = False, r: float = 0.707, alpha: float = 0.05
) -> dict[str, Any]:
    """JZS Bayes factor for the two-group comparison — evidence *for* the null is
    reportable here, unlike a non-significant p-value."""
    n1, v1, n2, v2 = require_two(groups)
    a, b = np.asarray(v1, float), np.asarray(v2, float)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired test needs equal-length groups")
        t, p = stats.ttest_rel(a, b)
        bf10 = bayes.bayesfactor_ttest(t, len(a), paired=True, r=r)
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        bf10 = bayes.bayesfactor_ttest(t, len(a), len(b), r=r)
    return {
        "analysis": "bayes_ttest",
        "groups": [n1, n2],
        "t": float(t),
        "frequentist_p": float(p),
        "prior_scale_r": r,
        "BF10": float(bf10),
        "BF01": float(1 / bf10) if bf10 else float("nan"),
        "interpretation": bayes.interpret_bf(bf10),
        "conclusion": (
            f"Bayes factor BF10={bf10:.3g}: {bayes.interpret_bf(bf10)} "
            f"(prior scale r={r})."
        ),
    }


def correlation(groups: Groups, *, method: str = "pearson", alpha: float = 0.05) -> dict[str, Any]:
    """Correlation between the first two groups, treated as paired x, y."""
    n1, v1, n2, v2 = require_two(groups)
    a, b = np.asarray(v1, float), np.asarray(v2, float)
    if len(a) != len(b):
        raise ValueError("correlation needs equal-length paired columns")
    if method == "spearman":
        r, p = stats.spearmanr(a, b)
    elif method == "kendall":
        r, p = stats.kendalltau(a, b)
    else:
        r, p = stats.pearsonr(a, b)
    return {
        "analysis": "correlation",
        "method": method,
        "variables": [n1, n2],
        "n": int(len(a)),
        "r": float(r),
        "r_squared": float(r**2),
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "conclusion": (
            f"{method.title()} r={r:.3f} (p={p:.4g}); "
            f"{'significant' if p < alpha else 'not significant'}."
        ),
    }


def _safe(name: str) -> str:
    """Wrap a column name for patsy: bare if it's a valid identifier, else Q('..')."""
    import re
    return name if re.match(r"^[A-Za-z_]\w*$", name) else f"Q('{name}')"


def _fit_and_anova(path: str, formula: str, *, typ: int = 2, alpha: float = 0.05) -> dict[str, Any]:
    """Fit an OLS model from a patsy formula and return its coefficient table, the
    ANOVA table (Type II/III SS, with partial eta^2 per term), overall fit, and
    residual diagnostics. Shared core for regression / two-way ANOVA / ANCOVA."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.stattools import durbin_watson

    from .data import load_dataframe

    df = load_dataframe(path)
    model = smf.ols(formula, data=df).fit()
    ci = model.conf_int(alpha)
    coefs = {
        name: {
            "coef": float(model.params[name]), "std_err": float(model.bse[name]),
            "t": float(model.tvalues[name]), "p": float(model.pvalues[name]),
            "ci": [float(ci.loc[name][0]), float(ci.loc[name][1])],
        }
        for name in model.params.index
    }
    aov = anova_lm(model, typ=typ)
    ss_resid = float(aov["sum_sq"].get("Residual", float("nan")))
    terms = []
    for name, row in aov.iterrows():
        if name == "Residual":
            continue
        ss = float(row["sum_sq"])
        peta = ss / (ss + ss_resid) if (ss + ss_resid) else float("nan")
        terms.append({
            "term": name, "sum_sq": ss, "df": float(row["df"]), "F": float(row["F"]),
            "p": float(row["PR(>F)"]), "partial_eta_sq": peta,
            "magnitude": effects.interpret_eta(peta),
            "significant": bool(row["PR(>F)"] < alpha),
        })
    resid = np.asarray(model.resid, float)
    diagnostics: dict[str, Any] = {"durbin_watson": float(durbin_watson(resid))}
    if len(resid) >= 3:
        w, p = stats.shapiro(resid)
        diagnostics.update(resid_shapiro_W=float(w), resid_shapiro_p=float(p),
                           resid_normal_at_alpha=bool(p > alpha))
    return {
        "formula": formula, "typ": typ, "n": int(model.nobs),
        "r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj),
        "f_pvalue": float(model.f_pvalue), "aic": float(model.aic), "bic": float(model.bic),
        "anova": terms, "coefficients": coefs, "residual_diagnostics": diagnostics,
    }


def regression(path: str, formula: str, *, typ: int = 2, alpha: float = 0.05) -> dict[str, Any]:
    """Multiple/linear regression from a patsy formula (e.g. 'y ~ x1 + x2 + C(g)').
    Returns coefficients, an ANOVA term table, overall fit (R²/AIC/BIC), and
    residual diagnostics."""
    out = _fit_and_anova(path, formula, typ=typ, alpha=alpha)
    out["analysis"] = "regression"
    return out


def formula_for(value: str, factors: list[str], covariates: list[str] | None = None) -> str:
    """Build a patsy formula for a multi-factor design: with no covariates, the
    factors are joined with '*' (all interactions, what `two_way_anova` needs);
    with covariates, factors and covariates are joined additively with '+'
    (main-effects ANCOVA style, what `ancova` needs)."""
    factor_terms = [f"C({_safe(f)})" for f in factors]
    rhs = " + ".join(factor_terms + [_safe(c) for c in covariates]) if covariates \
        else " * ".join(factor_terms)
    return f"{_safe(value)} ~ {rhs}"


def two_way_anova(path: str, *, value: str, factors: list[str], typ: int = 2,
                  alpha: float = 0.05) -> dict[str, Any]:
    """Factorial ANOVA: main effects + all interactions of 2+ categorical factors
    (e.g. score ~ treatment * sex). Reports per-term F, p, and partial eta²."""
    if len(factors) < 2:
        raise ValueError("two-way/factorial ANOVA needs at least 2 factors")
    formula = formula_for(value, factors)
    out = _fit_and_anova(path, formula, typ=typ, alpha=alpha)
    out["analysis"] = "two_way_anova"
    out["value"], out["factors"] = value, factors
    sig = [t["term"] for t in out["anova"] if t["significant"]]
    out["conclusion"] = (f"Significant effects (Type {typ} SS, alpha={alpha:g}): "
                         f"{', '.join(sig) if sig else 'none'}.")
    return out


def ancova(path: str, *, value: str, factors: list[str], covariates: list[str],
           typ: int = 2, alpha: float = 0.05) -> dict[str, Any]:
    """ANCOVA: categorical factor(s) adjusted for numeric covariate(s)
    (e.g. score ~ C(treatment) + age)."""
    if not factors or not covariates:
        raise ValueError("ANCOVA needs at least one factor and one covariate")
    formula = formula_for(value, factors, covariates)
    out = _fit_and_anova(path, formula, typ=typ, alpha=alpha)
    out["analysis"] = "ancova"
    out["value"], out["factors"], out["covariates"] = value, factors, covariates
    sig = [t["term"] for t in out["anova"] if t["significant"]]
    out["conclusion"] = (f"Adjusting for {', '.join(covariates)} (alpha={alpha:g}), "
                         f"significant effects: {', '.join(sig) if sig else 'none'}.")
    return out


def predict_table(path: str, formula: str, *, alpha: float = 0.05) -> dict[str, Any]:
    """Per-row observed/predicted/residual/leverage/Cook's distance for a fitted
    OLS model — the basis for a predicted-vs-observed figure and for flagging
    which specific runs are influential enough to warrant a confirmation rerun.
    Thresholds are the conventional ones: leverage > 2p/n, Cook's D > 4/n."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import OLSInfluence

    from .data import load_dataframe

    df = load_dataframe(path)
    model = smf.ols(formula, data=df).fit()
    infl = OLSInfluence(model)
    n, p = int(model.nobs), int(len(model.params))
    leverage_cut, cooks_cut = 2 * p / n, 4 / n

    endog = np.asarray(model.model.endog, float)
    fitted = np.asarray(model.fittedvalues, float)
    resid = np.asarray(model.resid, float)
    leverage = np.asarray(infl.hat_matrix_diag, float)
    cooks_d = np.asarray(infl.cooks_distance[0], float)
    stud_resid = np.asarray(infl.resid_studentized_internal, float)
    orig = model.model.data.frame

    rows = []
    for i in range(n):
        high_leverage = bool(leverage[i] > leverage_cut)
        influential = bool(cooks_d[i] > cooks_cut)
        rows.append({
            "row": int(i), "data": {k: (v.item() if hasattr(v, "item") else v)
                                    for k, v in orig.iloc[i].to_dict().items()},
            "observed": float(endog[i]), "predicted": float(fitted[i]),
            "residual": float(resid[i]), "standardized_residual": float(stud_resid[i]),
            "leverage": float(leverage[i]), "cooks_d": float(cooks_d[i]),
            "high_leverage": high_leverage, "influential": influential,
        })
    flagged = [r["row"] for r in rows if r["influential"] or r["high_leverage"]]
    return {
        "analysis": "predict_table", "formula": formula, "n": n, "alpha": alpha,
        "leverage_threshold": float(leverage_cut), "cooks_d_threshold": float(cooks_cut),
        "rows": rows, "flagged_rows": flagged, "n_flagged": len(flagged),
    }


def vif_table(path: str, formula: str) -> dict[str, Any]:
    """Variance Inflation Factor per model term — flags multicollinearity /
    confounding between predictors. VIF > 5 is a common concern threshold."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    from .data import load_dataframe

    df = load_dataframe(path)
    model = smf.ols(formula, data=df).fit()
    exog = model.model.exog
    terms = []
    for i, name in enumerate(model.model.exog_names):
        if name == "Intercept":
            continue
        vif = float(variance_inflation_factor(exog, i))
        terms.append({"term": name, "vif": vif, "collinearity_concern": bool(vif > 5)})
    return {"analysis": "vif", "formula": formula, "terms": terms}


def design_coverage(path: str, factors: list[str], *, value: str | None = None) -> dict[str, Any]:
    """Design-level diagnostics for a multi-factor/DoE dataset: how many of the
    factors' possible level combinations were actually run, replicate counts per
    combination tested, and — when center-point runs are present alongside corner
    runs — a curvature contrast (do center points sit off the corners' trend?)."""
    import itertools

    from .data import load_dataframe

    df = load_dataframe(path)
    levels = {f: sorted(df[f].dropna().unique().tolist()) for f in factors}
    numeric = {f: bool(pd.api.types.is_numeric_dtype(df[f])) for f in factors}

    n_possible = 1
    for f in factors:
        n_possible *= max(len(levels[f]), 1)

    keys = df[factors].astype(str).agg("|".join, axis=1)
    counts = keys.value_counts()
    replicate_counts = {k: int(v) for k, v in counts.items()}
    tested = set(counts.index)

    missing: list[dict[str, Any]] = []
    full_grid_enumerated = n_possible <= 64
    if full_grid_enumerated:
        for combo in itertools.product(*(levels[f] for f in factors)):
            key = "|".join(str(v) for v in combo)
            if key not in tested:
                missing.append(dict(zip(factors, combo, strict=True)))

    # A "center point" only makes sense for continuous factors with a real
    # min/max range, and only if every OTHER (non-continuous) factor is constant.
    rangey = [f for f in factors if numeric[f] and len(levels[f]) >= 2]
    has_varying_categorical = any(not numeric[f] and len(levels[f]) > 1 for f in factors)
    center_mask = pd.Series(False, index=df.index)
    curvature = None
    if rangey and not has_varying_categorical:
        mask = pd.Series(True, index=df.index)
        for f in rangey:
            lo, hi = levels[f][0], levels[f][-1]
            mid = (lo + hi) / 2
            tol = max((hi - lo) * 0.05, 1e-9)
            mask &= (df[f] - mid).abs() <= tol
        center_mask = mask

        n_center = int(center_mask.sum())
        if value and 0 < n_center < len(df):
            corner_vals = df.loc[~center_mask, value].astype(float)
            center_vals = df.loc[center_mask, value].astype(float)
            result: dict[str, Any] = {
                "corner_mean": float(corner_vals.mean()),
                "center_mean": float(center_vals.mean()),
                "difference": float(corner_vals.mean() - center_vals.mean()),
            }
            if len(center_vals) >= 2 and len(corner_vals) >= 2:
                t, p = stats.ttest_ind(corner_vals, center_vals, equal_var=False)
                result.update(t=float(t), p=float(p))
            else:
                result.update(t=None, p=None)
            result["center_point_df"] = int(max(len(center_vals) - 1, 0))
            result["low_power_warning"] = result["center_point_df"] < 3
            curvature = result

    return {
        "analysis": "design_coverage", "factors": factors, "levels": levels,
        "n_possible_combinations": int(n_possible),
        "n_tested_combinations": int(len(tested)),
        "coverage_fraction": float(len(tested) / n_possible) if n_possible else float("nan"),
        "replicate_counts": replicate_counts,
        "missing_combinations": missing, "n_missing_shown": len(missing),
        "full_grid_enumerated": full_grid_enumerated,
        "n_center_points": int(center_mask.sum()),
        "curvature": curvature,
    }


def doe_optimum(path: str, formula: str, factors: list[str], value: str) -> dict[str, Any]:
    """Rank every ACTUALLY TESTED combination of the given factors' observed
    levels by the fitted model's predicted response (generalizes "which of the
    N tested corners is best" — never extrapolates to an untested/interior
    point). Flags, per factor, whether the best setting sits at that factor's
    tested min/max ("boundary" — a real optimum may lie beyond the tested
    range) or strictly inside it ("interior")."""
    import itertools

    import statsmodels.formula.api as smf

    from .data import load_dataframe

    df = load_dataframe(path)
    model = smf.ols(formula, data=df).fit()
    levels = {f: sorted(df[f].dropna().unique().tolist()) for f in factors}

    # Reference value for any model term not among `factors` (e.g. a covariate),
    # held at its mean (numeric) or mode (categorical), so predict() has every
    # column the formula needs.
    ref = {}
    for c in df.columns:
        if c in factors:
            continue
        ref[c] = df[c].mean() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]

    observed_key = df[factors].astype(str).agg("|".join, axis=1)
    rows = []
    for combo in itertools.product(*(levels[f] for f in factors)):
        point = dict(zip(factors, combo, strict=True))
        predicted = float(model.predict(pd.DataFrame([{**ref, **point}])).iloc[0])
        key = "|".join(str(v) for v in combo)
        mask = observed_key == key
        tested = bool(mask.any())
        entry: dict[str, Any] = {**point, "predicted": predicted, "tested": tested}
        if tested:
            entry["observed_mean"] = float(df.loc[mask, value].astype(float).mean())
        rows.append(entry)
    rows.sort(key=lambda r: r["predicted"], reverse=True)

    best = rows[0] if rows else None
    boundary_flags: dict[str, str] = {}
    if best:
        for f in factors:
            lv = levels[f]
            if pd.api.types.is_numeric_dtype(df[f]) and len(lv) > 1:
                boundary_flags[f] = "boundary" if best[f] in (lv[0], lv[-1]) else "interior"
            else:
                boundary_flags[f] = "categorical"

    return {
        "analysis": "doe_optimum", "formula": formula, "factors": factors,
        "rows": rows, "best": best, "boundary_flags": boundary_flags,
        "note": "Predictions are evaluated only at each factor's ACTUALLY TESTED "
                "levels -- this never claims an interior/untested optimum.",
    }


def _count_coefs(model, alpha: float, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Coefficient table for a count model, reported as incidence-rate ratios
    (IRR = exp(coef): the multiplicative change in the event rate per unit)."""
    ci = model.conf_int(alpha)
    out = {}
    for name in model.params.index:
        if name in skip:
            continue
        out[name] = {
            "coef": float(model.params[name]),
            "irr": float(np.exp(model.params[name])),
            "irr_ci": [float(np.exp(ci.loc[name][0])), float(np.exp(ci.loc[name][1]))],
            "p": float(model.pvalues[name]),
            "significant": bool(model.pvalues[name] < alpha),
        }
    return out


def poisson_regression(path: str, formula: str, *, exposure: str | None = None,
                       alpha: float = 0.05) -> dict[str, Any]:
    """Poisson rate regression (log link) for COUNT data, e.g. events per period:
    `count ~ year`. Coefficients are reported as incidence-rate ratios (IRR).
    Includes an overdispersion check (Pearson χ²/df); if it's >> 1 the counts are
    overdispersed and `negbin` should be used instead. Pass ``exposure`` (a column)
    to model a *rate* when periods differ in length/exposure (adds a log offset)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from .data import load_dataframe
    df = load_dataframe(path)
    offset = np.log(df[exposure].astype(float)) if exposure else None
    model = smf.glm(formula, data=df, family=sm.families.Poisson(), offset=offset).fit()
    disp = float(model.pearson_chi2 / model.df_resid)
    out = {
        "analysis": "poisson_regression",
        "formula": formula,
        "n": int(model.nobs),
        "coefficients": _count_coefs(model, alpha),
        "aic": float(model.aic),
        "deviance": float(model.deviance),
        "df_resid": int(model.df_resid),
        "overdispersion": {
            "dispersion": disp,
            "pearson_chi2": float(model.pearson_chi2),
            "overdispersed": bool(disp > 1.5),
            "recommendation": ("dispersion >> 1 → refit with negative binomial (negbin)"
                               if disp > 1.5 else "dispersion ≈ 1 → Poisson is adequate"),
        },
    }
    if exposure:
        out["exposure"] = exposure
    return out


def negbin_regression(path: str, formula: str, *, exposure: str | None = None,
                      alpha: float = 0.05) -> dict[str, Any]:
    """Negative-binomial rate regression for OVERDISPERSED count data (variance >
    mean). Like `poisson` but estimates an extra dispersion parameter (alpha); use
    it when the Poisson overdispersion check flags a problem. IRR-reported."""
    import warnings

    import statsmodels.formula.api as smf

    from .data import load_dataframe
    df = load_dataframe(path)
    kwargs = {"exposure": df[exposure].astype(float)} if exposure else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.negativebinomial(formula, data=df, **kwargs).fit(disp=0)
    converged = bool(getattr(model, "mle_retvals", {}).get("converged", True))
    bse_ok = bool(np.isfinite(np.asarray(model.bse, float)).all())
    out = {
        "analysis": "negbin_regression",
        "formula": formula,
        "n": int(model.nobs),
        "coefficients": _count_coefs(model, alpha, skip=("alpha",)),
        "dispersion_alpha": float(model.params.get("alpha", float("nan"))),
        "converged": converged and bse_ok,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "llf": float(model.llf),
    }
    if not (converged and bse_ok):
        out["warning"] = (
            "negative-binomial MLE did not converge / has no standard errors — most "
            "often because the counts are NOT overdispersed (dispersion alpha → 0, so "
            "Poisson is the correct model), or a large-magnitude predictor needs "
            "centering (e.g. use I(year - min)). Prefer the Poisson fit here.")
    if exposure:
        out["exposure"] = exposure
    return out


def chi_square(table: list[list[float]], *, alpha: float = 0.05) -> dict[str, Any]:
    """Chi-square test of independence on a contingency table."""
    arr = np.asarray(table, float)
    chi2, p, dof, expected = stats.chi2_contingency(arr)
    n = arr.sum()
    min_dim = min(arr.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else float("nan")
    return {
        "analysis": "chi_square",
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "cramers_v": cramers_v,
        "min_expected": float(expected.min()),
        "low_expected_warning": bool(expected.min() < 5),
        "conclusion": (
            f"Variables are {'associated' if p < alpha else 'not detectably associated'} "
            f"(chi2={chi2:.4g}, p={p:.4g}, Cramer's V={cramers_v:.3f})."
        ),
    }


def power_analysis(
    *,
    test: str = "ttest",
    effect_size: float,
    n: float | None = None,
    power: float | None = None,
    alpha: float = 0.05,
    k_groups: int | None = None,
) -> dict[str, Any]:
    """Solve for whichever of {n, power} is left as None. For a t-test, n is the
    sample size **per group** (statsmodels nobs1); for ANOVA, n is the **total**
    sample size across all groups (statsmodels nobs) — the output is labelled
    accordingly so the number is never misread."""
    if (n is None) == (power is None):
        raise ValueError("leave exactly one of --n / --power unset to solve for it")
    if test == "anova":
        if k_groups is None:
            raise ValueError("--test anova requires --k-groups (number of groups)")
        analysis = FTestAnovaPower()
        solved = analysis.solve_power(
            effect_size=effect_size, nobs=n, alpha=alpha, power=power, k_groups=k_groups
        )
    else:
        analysis = TTestIndPower()
        solved = analysis.solve_power(
            effect_size=effect_size, nobs1=n, alpha=alpha, power=power
        )
    n_label = "total_n" if test == "anova" else "n_per_group"
    solved_label = f"required_{n_label}" if n is None else "achieved_power"
    out: dict[str, Any] = {
        "analysis": "power",
        "test": test,
        "effect_size": effect_size,
        "alpha": alpha,
        solved_label: float(solved),
        "given": {n_label: n, "power": power},
    }
    if test == "anova":
        out["k_groups"] = k_groups
    return out


def anova_table(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Full one-way ANOVA source table + goodness-of-fit measures. Feeds the
    report's detail tables (SS/df/MS/F/p, plus R^2, eta^2, omega^2, epsilon^2)."""
    ss = _ss_decomposition(groups)
    f = ss["ms_between"] / ss["ms_within"] if ss["ms_within"] else float("nan")
    p = float(stats.f.sf(f, ss["df_between"], ss["df_within"]))
    eta = effects.eta_squared(ss["ss_between"], ss["ss_total"])
    omega = effects.omega_squared(ss["ss_between"], ss["df_between"], ss["ms_within"], ss["ss_total"])
    epsilon = ((ss["ss_between"] - ss["df_between"] * ss["ms_within"]) / ss["ss_total"]
               if ss["ss_total"] else float("nan"))
    return {
        "analysis": "anova_table",
        "rows": [
            {"source": "Between groups", "SS": ss["ss_between"], "df": ss["df_between"],
             "MS": ss["ms_between"], "F": float(f), "p": p},
            {"source": "Within groups", "SS": ss["ss_within"], "df": ss["df_within"],
             "MS": ss["ms_within"], "F": None, "p": None},
            {"source": "Total", "SS": ss["ss_total"], "df": ss["df_between"] + ss["df_within"],
             "MS": None, "F": None, "p": None},
        ],
        "goodness_of_fit": {
            "r_squared": eta,  # for one-way ANOVA R^2 == eta^2
            "eta_squared": eta,
            "omega_squared": omega,
            "epsilon_squared": float(epsilon),
            "root_mse": float(ss["ms_within"] ** 0.5) if ss["ms_within"] >= 0 else float("nan"),
        },
    }


def residual_diagnostics(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Summary and normality/homogeneity checks on the one-way model residuals
    (value - group mean) — the 'residuals' block of the report."""
    arrays = [np.asarray(v, float) for v in groups.values()]
    resid = np.concatenate([a - a.mean() for a in arrays])
    q1, med, q3 = np.percentile(resid, [25, 50, 75])
    out: dict[str, Any] = {
        "analysis": "residual_diagnostics",
        "n": int(len(resid)),
        "mean": float(resid.mean()),
        "sd": float(resid.std(ddof=1)) if len(resid) > 1 else float("nan"),
        "min": float(resid.min()), "q1": float(q1), "median": float(med),
        "q3": float(q3), "max": float(resid.max()),
    }
    if len(resid) >= 3:
        w, p = stats.shapiro(resid)
        out["normality"] = {"shapiro_W": float(w), "shapiro_p": float(p),
                            "normal_at_alpha": bool(p > alpha)}
    if len(arrays) >= 2 and all(len(a) >= 2 for a in arrays):
        lw, lp = stats.levene(*arrays, center="median")
        out["homoscedasticity"] = {"levene_W": float(lw), "levene_p": float(lp),
                                   "equal_variance_at_alpha": bool(lp > alpha)}
    return out


def pairwise_effects(groups: Groups, *, alpha: float = 0.05) -> dict[str, Any]:
    """Scaled effect sizes for every pair: mean difference (+ Welch CI), Cohen's d,
    Hedges' g, and a plain magnitude label."""
    names = list(groups.keys())
    arrays = {n: np.asarray(v, float) for n, v in groups.items()}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            a, b = arrays[n1], arrays[n2]
            diff = float(a.mean() - b.mean())
            se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            df = ((a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2) / (
                (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
            crit = float(stats.t.ppf(1 - alpha / 2, df))
            d = effects.cohens_d(a.tolist(), b.tolist())
            pairs.append({
                "pair": f"{n1} vs {n2}",
                "mean_diff": diff,
                "ci": [diff - crit * se, diff + crit * se],
                "cohens_d": d,
                "hedges_g": effects.hedges_g(a.tolist(), b.tolist()),
                "magnitude": effects.interpret_d(d),
            })
    return {"analysis": "pairwise_effects", "alpha": alpha, "pairs": pairs}


def correct_pvalues(pvalues: list[float], *, method: str = "fdr_bh", alpha: float = 0.05) -> dict[str, Any]:
    """Apply a multiple-comparison correction to a set of p-values."""
    reject, p_adj, _, _ = multipletests(pvalues, alpha=alpha, method=method)
    return {
        "analysis": "multiple_comparison",
        "method": method,
        "alpha": alpha,
        "raw_p": [float(p) for p in pvalues],
        "adjusted_p": [float(p) for p in p_adj],
        "reject_null": [bool(r) for r in reject],
        "n_significant": int(reject.sum()),
    }
