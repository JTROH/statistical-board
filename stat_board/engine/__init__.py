"""The shared statistics engine — pure functions, JSON-serializable results.

Import the analysis functions directly, or call the CLI:

    python -m stat_board.engine <command> --data data.csv [options]

Every command prints one JSON object to stdout. The board's Analyst agent runs
these so its claims are grounded in a real computation, not a guess.
"""

from . import analyses, bayes, data, effects
from .analyses import (
    anova,
    anova_table,
    bayes_ttest,
    check_assumptions,
    chi_square,
    correct_pvalues,
    correlation,
    describe,
    kruskal,
    mann_whitney,
    pairwise_effects,
    power_analysis,
    regression,
    residual_diagnostics,
    tost,
    ttest,
    tukey_posthoc,
    welch_anova,
)

__all__ = [
    "analyses", "bayes", "data", "effects",
    "describe", "check_assumptions", "ttest", "mann_whitney", "anova",
    "welch_anova", "kruskal", "tukey_posthoc", "tost", "bayes_ttest",
    "correlation", "regression", "chi_square", "power_analysis", "correct_pvalues",
    "anova_table", "residual_diagnostics", "pairwise_effects",
]
