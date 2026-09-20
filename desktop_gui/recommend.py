"""Pick the most appropriate group-comparison test for a set of datasets, run
it with the shared ``stat_board`` engine, and explain the choice in plain
language.

This codifies the same reasoning the ``/stat-advisor`` skill uses (check
normality and variance homogeneity first, then choose the test those checks
support) as deterministic Python, since this desktop tool has no LLM to reason
with. It decides *which* test to run; the engine itself still owns every
number and every verdict — ``plain_summary`` leans on each analysis
function's own ``conclusion`` string rather than re-deriving language here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for flat `_bootstrap` import
from _bootstrap import ensure_stat_board_importable  # noqa: E402

if not ensure_stat_board_importable():
    raise ModuleNotFoundError(
        "stat_board engine not found; set STAT_BOARD_HOME to the repo path"
    )

from stat_board.engine import analyses  # noqa: E402
from stat_board.engine.data import Groups  # noqa: E402

_TUKEY_ELIGIBLE = {"anova", "welch_anova"}


def _normality_verdict(assumptions: dict[str, Any], alpha: float) -> tuple[bool, list[str]]:
    caveats = []
    normal = True
    for name, entry in assumptions["normality"].items():
        if "shapiro_p" not in entry:
            normal = False
            caveats.append(
                f'"{name}" has only {entry["n"]} observation(s) — too few to test '
                "normality, so a non-parametric test was used to be safe."
            )
        elif not entry["normal_at_alpha"]:
            normal = False
    return normal, caveats


def _homogeneity_verdict(assumptions: dict[str, Any]) -> tuple[bool, list[str]]:
    lev = assumptions["homogeneity"].get("levene")
    if lev is None:
        return False, [
            "Variance homogeneity could not be tested (a dataset has fewer than "
            "2 observations) — the more conservative unequal-variance test was used."
        ]
    return bool(lev["equal_variance_at_alpha"]), []


def analyze(groups: Groups, alpha: float = 0.05) -> dict[str, Any]:
    """Inspect ``groups``, choose the best-fitting comparison test, run it (plus
    any supplementary tests), and return the full reasoning trail."""
    if len(groups) < 2:
        raise ValueError("need at least 2 datasets to recommend a test")

    desc = analyses.describe(groups, alpha=alpha)
    assumptions = analyses.check_assumptions(groups, alpha=alpha)

    normal, normality_caveats = _normality_verdict(assumptions, alpha)
    equal_var, homogeneity_caveats = _homogeneity_verdict(assumptions)
    caveats = normality_caveats + homogeneity_caveats

    for name, entry in assumptions["normality"].items():
        if entry.get("outliers_iqr"):
            caveats.append(
                f'"{name}" has {entry["outliers_iqr"]} possible outlier(s) '
                "(outside 1.5x the interquartile range)."
            )

    rationale: list[str] = []
    if normal:
        min_p = min(
            e["shapiro_p"] for e in assumptions["normality"].values() if "shapiro_p" in e
        )
        rationale.append(
            f"Shapiro-Wilk did not reject normality for any dataset "
            f"(smallest p={min_p:.3g} > alpha={alpha:g})."
        )
    else:
        rationale.append(
            "At least one dataset failed the normality check (or was too small to "
            "check), so a non-parametric test was chosen instead of an assumption "
            "that may not hold."
        )
    lev = assumptions["homogeneity"].get("levene")
    if lev is not None:
        rationale.append(
            f"Levene's test {'found no significant difference' if equal_var else 'found a significant difference'} "
            f"in variances across datasets (p={lev['p']:.3g})."
        )

    supplementary: dict[str, Any] = {}
    two_groups = len(groups) == 2

    if two_groups:
        if not normal:
            result = analyses.mann_whitney(groups, alpha=alpha)
        elif equal_var:
            result = analyses.ttest(groups, equal_var=True, alpha=alpha)
        else:
            result = analyses.ttest(groups, equal_var=False, alpha=alpha)
        supplementary["bayes_ttest"] = analyses.bayes_ttest(groups, alpha=alpha)
    else:
        if not normal:
            result = analyses.kruskal(groups, alpha=alpha)
        elif equal_var:
            result = analyses.anova(groups, alpha=alpha)
        else:
            result = analyses.welch_anova(groups, alpha=alpha)
        if result["analysis"] in _TUKEY_ELIGIBLE:
            supplementary["tukey_posthoc"] = analyses.tukey_posthoc(groups, alpha=alpha)

    return {
        "chosen_test": result["analysis"],
        "alpha": alpha,
        "rationale": rationale,
        "caveats": caveats,
        "describe": desc,
        "assumptions": assumptions,
        "result": result,
        "supplementary": supplementary,
    }


def _tukey_summary(tukey: dict[str, Any]) -> str:
    diffs = [p for p in tukey["pairs"] if p["reject_null"]]
    if not diffs:
        return "A Tukey HSD follow-up found no pair of datasets that differ significantly."
    pairs = ", ".join(f'{p["group1"]} vs {p["group2"]}' for p in diffs)
    return f"A Tukey HSD follow-up found significant differences between: {pairs}."


def plain_summary(analysis: dict[str, Any]) -> str:
    """Turn an ``analyze()`` result into a short paragraph a non-statistician
    can read."""
    desc = analysis["describe"]["groups"]
    names = list(desc.keys())
    means = ", ".join(f'{n} (n={desc[n]["n"]}, mean={desc[n]["mean"]:.4g})' for n in names)

    result = analysis["result"]
    sentences = [f"Datasets compared: {means}."]
    sentences.append(analysis["rationale"][0])
    # The engine's own conclusion already states significance and, for
    # ttest/anova, the effect size and its magnitude — no need to restate it.
    sentences.append(result["conclusion"])

    bayes = analysis["supplementary"].get("bayes_ttest")
    if bayes is not None:
        sentences.append(
            f"For additional context, the Bayesian angle gives BF10={bayes['BF10']:.3g}: "
            f"{bayes['interpretation']}."
        )

    tukey = analysis["supplementary"].get("tukey_posthoc")
    if tukey is not None:
        sentences.append(_tukey_summary(tukey))

    if analysis["caveats"]:
        sentences.append("Note: " + " ".join(analysis["caveats"]))

    return " ".join(sentences)
