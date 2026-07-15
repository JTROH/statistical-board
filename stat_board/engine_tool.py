"""Exposes the statistics engine as a single Anthropic tool, `run_stat`, that the
Analyst and Verifier call to ground every claim in a real computation.

`make_executor` binds the tool to one run's dataset (path + how to load it), so
the model only has to name a command and its parameters — never the data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .engine import analyses
from .engine.data import DataError, load_groups

# Commands that operate on the run's grouped data (loaded from the bound path).
_GROUP_COMMANDS = {
    "describe", "assumptions", "ttest", "mannwhitney", "anova", "welch-anova",
    "kruskal", "tukey", "tost", "bayes-ttest", "correlation",
}
# Commands that run on the whole table by column name (multi-factor / standalone).
_STANDALONE = {"regression", "two-way-anova", "ancova", "poisson", "negbin",
               "chisquare", "power", "correct",
               "predict", "vif", "design-coverage", "doe-optimum"}

TOOL_NAME = "run_stat"


def build_tool() -> dict[str, Any]:
    """The Anthropic tool spec for the engine."""
    return {
        "name": TOOL_NAME,
        "description": (
            "Run one statistical computation on the current dataset and return its "
            "exact JSON result. This is the ONLY way to obtain real numbers — never "
            "estimate a statistic yourself. Group-based commands (describe, "
            "assumptions, ttest, mannwhitney, anova, welch-anova, kruskal, tukey, "
            "tost, bayes-ttest, correlation) compare one numeric outcome across one "
            "grouping factor. MULTI-FACTOR commands work by column name on the whole "
            "table: two-way-anova (2+ categorical factors + interactions), ancova "
            "(factor(s) + numeric covariate(s)), and regression (a patsy formula). "
            "For COUNT/frequency data (events per period), poisson and negbin are "
            "rate regressions (report incidence-rate ratios). chisquare/power/correct "
            "carry their data in the parameters. For a MULTI-FACTOR/DoE analysis, also "
            "run: predict (per-row predicted/residual/leverage/Cook's D — flags which "
            "runs are influential enough to warrant a confirmation rerun), vif "
            "(variance inflation factor per term — flags confounding/multicollinearity), "
            "design-coverage (how many of the factors' possible level combinations were "
            "actually run, replicate counts, and a curvature contrast if center-point "
            "runs are present), and doe-optimum (ranks every ACTUALLY TESTED factor "
            "combination by predicted response, and flags whether the best setting sits "
            "at a factor's tested boundary — a real optimum may lie beyond the tested "
            "range). These four are the grounding for any 'recommended next experiment' "
            "claim — never recommend a follow-up experiment without one of them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": sorted(_GROUP_COMMANDS | _STANDALONE),
                    "description": "Which analysis to run.",
                },
                "alpha": {"type": "number", "description": "Significance level (default 0.05)."},
                "paired": {"type": "boolean", "description": "ttest/bayes-ttest: paired design."},
                "equal_var": {"type": "boolean", "description": "ttest: Student's pooled t (default Welch)."},
                "margin_pct": {"type": "number", "description": "tost: symmetric bound as %% of |mean of group1|."},
                "low": {"type": "number", "description": "tost: absolute lower equivalence bound."},
                "high": {"type": "number", "description": "tost: absolute upper equivalence bound."},
                "method": {"type": "string", "description": "correlation: pearson|spearman|kendall. correct: fdr_bh|bonferroni|holm."},
                "r": {"type": "number", "description": "bayes-ttest: Cauchy prior scale (default 0.707)."},
                "formula": {"type": "string", "description": "regression/predict/vif/doe-optimum: "
                                                            "patsy formula, e.g. 'y ~ x1 + x2 + C(g)'."},
                "value": {"type": "string", "description": "two-way-anova/ancova/design-coverage/"
                                                           "doe-optimum: numeric outcome column."},
                "factors": {"type": "array", "items": {"type": "string"},
                            "description": "two-way-anova/ancova/design-coverage/doe-optimum: "
                                          "factor columns (categorical for two-way-anova/ancova; "
                                          "the DoE factor columns for design-coverage/doe-optimum)."},
                "covariates": {"type": "array", "items": {"type": "string"},
                               "description": "ancova: numeric covariate columns."},
                "typ": {"type": "integer", "description": "regression/two-way-anova/ancova: ANOVA SS type 1|2|3 (default 2)."},
                "exposure": {"type": "string", "description": "poisson/negbin: column of per-row exposure (adds a log offset for rates)."},
                "table": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                          "description": "chisquare: 2D contingency table."},
                "pvalues": {"type": "array", "items": {"type": "number"}, "description": "correct: raw p-values."},
                "test": {"type": "string", "description": "power: ttest|anova."},
                "effect_size": {"type": "number", "description": "power: Cohen's d (ttest) or f (anova)."},
                "n": {"type": "number", "description": "power: n per group (omit to solve for it)."},
                "power": {"type": "number", "description": "power: target power (omit to solve for it)."},
                "k_groups": {"type": "integer", "description": "power+anova: number of groups (required)."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    }


def make_executor(
    data_path: str, *, group_col: str | None = None, value_col: str | None = None
) -> Callable[[dict[str, Any]], str]:
    """Return a function that executes one `run_stat` tool input and returns a
    JSON string (the engine's result, or an {"error": ...} object)."""
    _cache: dict[str, Any] = {}

    def _groups() -> Any:
        if "g" not in _cache:
            _cache["g"] = load_groups(data_path, group_col=group_col, value_col=value_col)
        return _cache["g"]

    def execute(inp: dict[str, Any]) -> str:
        cmd = inp.get("command")
        a = inp.get("alpha", 0.05)
        try:
            if cmd in _GROUP_COMMANDS:
                g = _groups()
                if cmd == "describe":
                    res = analyses.describe(g, a)
                elif cmd == "assumptions":
                    res = analyses.check_assumptions(g, a)
                elif cmd == "ttest":
                    res = analyses.ttest(g, paired=inp.get("paired", False),
                                         equal_var=inp.get("equal_var", False), alpha=a)
                elif cmd == "mannwhitney":
                    res = analyses.mann_whitney(g, alpha=a)
                elif cmd == "anova":
                    res = analyses.anova(g, alpha=a)
                elif cmd == "welch-anova":
                    res = analyses.welch_anova(g, alpha=a)
                elif cmd == "kruskal":
                    res = analyses.kruskal(g, alpha=a)
                elif cmd == "tukey":
                    res = analyses.tukey_posthoc(g, alpha=a)
                elif cmd == "tost":
                    res = analyses.tost(g, low=inp.get("low"), high=inp.get("high"),
                                        margin_pct=inp.get("margin_pct"), alpha=a)
                elif cmd == "bayes-ttest":
                    res = analyses.bayes_ttest(g, paired=inp.get("paired", False),
                                               r=inp.get("r", 0.707), alpha=a)
                elif cmd == "correlation":
                    res = analyses.correlation(g, method=inp.get("method", "pearson"), alpha=a)
            elif cmd == "regression":
                res = analyses.regression(data_path, inp["formula"], typ=inp.get("typ", 2), alpha=a)
            elif cmd == "two-way-anova":
                res = analyses.two_way_anova(data_path, value=inp["value"], factors=inp["factors"],
                                             typ=inp.get("typ", 2), alpha=a)
            elif cmd == "ancova":
                res = analyses.ancova(data_path, value=inp["value"], factors=inp["factors"],
                                      covariates=inp["covariates"], typ=inp.get("typ", 2), alpha=a)
            elif cmd == "poisson":
                res = analyses.poisson_regression(data_path, inp["formula"],
                                                  exposure=inp.get("exposure"), alpha=a)
            elif cmd == "negbin":
                res = analyses.negbin_regression(data_path, inp["formula"],
                                                 exposure=inp.get("exposure"), alpha=a)
            elif cmd == "chisquare":
                res = analyses.chi_square(inp["table"], alpha=a)
            elif cmd == "power":
                res = analyses.power_analysis(
                    test=inp.get("test", "ttest"), effect_size=inp["effect_size"],
                    n=inp.get("n"), power=inp.get("power"), alpha=a,
                    k_groups=inp.get("k_groups"))
            elif cmd == "correct":
                res = analyses.correct_pvalues(inp["pvalues"], method=inp.get("method", "fdr_bh"), alpha=a)
            elif cmd == "predict":
                res = analyses.predict_table(data_path, inp["formula"], alpha=a)
            elif cmd == "vif":
                res = analyses.vif_table(data_path, inp["formula"])
            elif cmd == "design-coverage":
                res = analyses.design_coverage(data_path, inp["factors"], value=inp.get("value"))
            elif cmd == "doe-optimum":
                res = analyses.doe_optimum(data_path, inp["formula"], inp["factors"], inp["value"])
            else:
                return json.dumps({"error": "unknown_command", "command": cmd})
        except (DataError, ValueError, KeyError, TypeError) as exc:
            return json.dumps({"error": type(exc).__name__, "message": str(exc), "command": cmd})
        return json.dumps(res)

    return execute
