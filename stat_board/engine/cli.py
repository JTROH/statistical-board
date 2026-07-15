"""Command-line front door to the engine. One subcommand per analysis; every
run prints a single JSON object to stdout (errors go to stderr as JSON too, with
a non-zero exit) so an agent can parse the result unambiguously.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import analyses
from .data import DataError, load_groups


def _add_data_args(p: argparse.ArgumentParser, *, needs_groups: bool = True) -> None:
    if needs_groups:
        p.add_argument("--data", required=True, help="CSV or JSON file of grouped values")
        p.add_argument("--group-col", help="long-format: column holding group labels")
        p.add_argument("--value-col", help="long-format: column holding values")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stat_board.engine",
        description="Statistics engine — JSON in, JSON out.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["describe", "assumptions", "mannwhitney", "anova",
                 "welch-anova", "kruskal", "tukey"]:
        _add_data_args(sub.add_parser(name))

    t = sub.add_parser("ttest"); _add_data_args(t)
    t.add_argument("--paired", action="store_true")
    t.add_argument("--equal-var", action="store_true", help="Student's t (default is Welch)")

    b = sub.add_parser("bayes-ttest"); _add_data_args(b)
    b.add_argument("--paired", action="store_true")
    b.add_argument("--r", type=float, default=0.707, help="Cauchy prior scale")

    to = sub.add_parser("tost"); _add_data_args(to)
    to.add_argument("--low", type=float)
    to.add_argument("--high", type=float)
    to.add_argument("--margin-pct", type=float, help="symmetric bound as %% of |mean of group1|")

    c = sub.add_parser("correlation"); _add_data_args(c)
    c.add_argument("--method", choices=["pearson", "spearman", "kendall"], default="pearson")

    r = sub.add_parser("regression")
    r.add_argument("--data", required=True)
    r.add_argument("--formula", required=True, help="patsy formula, e.g. 'y ~ x1 + x2 + C(g)'")
    r.add_argument("--type", type=int, choices=[1, 2, 3], default=2, dest="typ",
                   help="ANOVA sum-of-squares type (default 2)")
    r.add_argument("--alpha", type=float, default=0.05)

    tw = sub.add_parser("two-way-anova", help="factorial ANOVA (2+ categorical factors + interactions)")
    tw.add_argument("--data", required=True)
    tw.add_argument("--value", required=True, help="numeric outcome column")
    tw.add_argument("--factor", action="append", required=True, dest="factors",
                    help="categorical factor column (repeat --factor for each)")
    tw.add_argument("--type", type=int, choices=[1, 2, 3], default=2, dest="typ")
    tw.add_argument("--alpha", type=float, default=0.05)

    an = sub.add_parser("ancova", help="factor(s) adjusted for numeric covariate(s)")
    an.add_argument("--data", required=True)
    an.add_argument("--value", required=True, help="numeric outcome column")
    an.add_argument("--factor", action="append", required=True, dest="factors")
    an.add_argument("--covariate", action="append", required=True, dest="covariates")
    an.add_argument("--type", type=int, choices=[1, 2, 3], default=2, dest="typ")
    an.add_argument("--alpha", type=float, default=0.05)

    for cnt in ("poisson", "negbin"):
        cp = sub.add_parser(cnt, help=f"{cnt} rate regression for count data (IRR-reported)")
        cp.add_argument("--data", required=True)
        cp.add_argument("--formula", required=True, help="e.g. 'n ~ year' (count column ~ predictors)")
        cp.add_argument("--exposure", help="column giving per-row exposure (adds a log offset for rates)")
        cp.add_argument("--alpha", type=float, default=0.05)

    ch = sub.add_parser("chisquare")
    ch.add_argument("--table", required=True, help="JSON 2D array, e.g. '[[10,20],[30,40]]'")
    ch.add_argument("--alpha", type=float, default=0.05)

    pw = sub.add_parser("power")
    pw.add_argument("--test", choices=["ttest", "anova"], default="ttest")
    pw.add_argument("--effect-size", type=float, required=True)
    pw.add_argument("--n", type=float, help="n per group (omit to solve for it)")
    pw.add_argument("--power", type=float, help="target power (omit to solve for it)")
    pw.add_argument("--k-groups", type=int, default=None,
                    help="number of groups (REQUIRED for --test anova)")
    pw.add_argument("--alpha", type=float, default=0.05)

    mc = sub.add_parser("correct")
    mc.add_argument("--pvalues", required=True, help="JSON array, e.g. '[0.01,0.04,0.2]'")
    mc.add_argument("--method", default="fdr_bh",
                    help="fdr_bh, bonferroni, holm, ... (statsmodels names)")
    mc.add_argument("--alpha", type=float, default=0.05)

    pt = sub.add_parser("predict", help="per-row predicted/residual/leverage/Cook's D for a fitted model")
    pt.add_argument("--data", required=True)
    pt.add_argument("--formula", required=True)
    pt.add_argument("--alpha", type=float, default=0.05)

    vi = sub.add_parser("vif", help="variance inflation factor per model term (multicollinearity)")
    vi.add_argument("--data", required=True)
    vi.add_argument("--formula", required=True)

    dcov = sub.add_parser("design-coverage", help="DoE design coverage, replicates, and curvature check")
    dcov.add_argument("--data", required=True)
    dcov.add_argument("--factor", action="append", required=True, dest="factors")
    dcov.add_argument("--value", help="numeric outcome column (enables the curvature contrast)")

    opt = sub.add_parser("doe-optimum", help="rank actually-tested factor combinations by predicted response")
    opt.add_argument("--data", required=True)
    opt.add_argument("--formula", required=True)
    opt.add_argument("--factor", action="append", required=True, dest="factors")
    opt.add_argument("--value", required=True, help="numeric outcome column")

    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    cmd = args.command

    if cmd in {"describe", "assumptions", "ttest", "mannwhitney", "anova",
               "welch-anova", "kruskal", "tukey", "bayes-ttest", "tost", "correlation"}:
        groups = load_groups(
            args.data,
            group_col=getattr(args, "group_col", None),
            value_col=getattr(args, "value_col", None),
        )

    if cmd == "describe":
        return analyses.describe(groups, args.alpha)
    if cmd == "assumptions":
        return analyses.check_assumptions(groups, args.alpha)
    if cmd == "ttest":
        return analyses.ttest(groups, paired=args.paired, equal_var=args.equal_var, alpha=args.alpha)
    if cmd == "mannwhitney":
        return analyses.mann_whitney(groups, alpha=args.alpha)
    if cmd == "anova":
        return analyses.anova(groups, alpha=args.alpha)
    if cmd == "welch-anova":
        return analyses.welch_anova(groups, alpha=args.alpha)
    if cmd == "kruskal":
        return analyses.kruskal(groups, alpha=args.alpha)
    if cmd == "tukey":
        return analyses.tukey_posthoc(groups, alpha=args.alpha)
    if cmd == "bayes-ttest":
        return analyses.bayes_ttest(groups, paired=args.paired, r=args.r, alpha=args.alpha)
    if cmd == "tost":
        return analyses.tost(groups, low=args.low, high=args.high,
                             margin_pct=args.margin_pct, alpha=args.alpha)
    if cmd == "correlation":
        return analyses.correlation(groups, method=args.method, alpha=args.alpha)
    if cmd == "regression":
        return analyses.regression(args.data, args.formula, typ=args.typ, alpha=args.alpha)
    if cmd == "two-way-anova":
        return analyses.two_way_anova(args.data, value=args.value, factors=args.factors,
                                      typ=args.typ, alpha=args.alpha)
    if cmd == "ancova":
        return analyses.ancova(args.data, value=args.value, factors=args.factors,
                               covariates=args.covariates, typ=args.typ, alpha=args.alpha)
    if cmd == "poisson":
        return analyses.poisson_regression(args.data, args.formula,
                                           exposure=args.exposure, alpha=args.alpha)
    if cmd == "negbin":
        return analyses.negbin_regression(args.data, args.formula,
                                          exposure=args.exposure, alpha=args.alpha)
    if cmd == "chisquare":
        return analyses.chi_square(json.loads(args.table), alpha=args.alpha)
    if cmd == "power":
        return analyses.power_analysis(
            test=args.test, effect_size=args.effect_size, n=args.n,
            power=args.power, alpha=args.alpha, k_groups=args.k_groups,
        )
    if cmd == "correct":
        return analyses.correct_pvalues(json.loads(args.pvalues), method=args.method, alpha=args.alpha)
    if cmd == "predict":
        return analyses.predict_table(args.data, args.formula, alpha=args.alpha)
    if cmd == "vif":
        return analyses.vif_table(args.data, args.formula)
    if cmd == "design-coverage":
        return analyses.design_coverage(args.data, args.factors, value=args.value)
    if cmd == "doe-optimum":
        return analyses.doe_optimum(args.data, args.formula, args.factors, args.value)
    raise ValueError(f"unknown command: {cmd}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (DataError, ValueError) as exc:
        json.dump({"error": type(exc).__name__, "message": str(exc)}, sys.stderr)
        sys.stderr.write("\n")
        return 2
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0
