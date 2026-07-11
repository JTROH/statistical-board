"""Bayesian evidence for the Bayesian critic.

Implements the JZS (Jeffreys-Zellner-Siow) default Bayes factor for the t-test
(Rouder, Speckman, Sun, Morey & Iverson, 2009), computed by numerically
integrating over the scaled prior on the standardized effect. This is the same
quantity the R ``BayesFactor`` package and pingouin report, done here with only
scipy so the engine takes no extra dependencies.

BF10 is the ratio of the marginal likelihood under the alternative to that under
the null: BF10 = 5 means the data are 5x more likely under 'there is an effect'
than under 'there is none'.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def bayesfactor_ttest(
    t: float, nx: int, ny: int | None = None, *, paired: bool = False, r: float = 0.707
) -> float:
    """JZS BF10 from a t statistic and the sample size(s).

    ``r`` is the Cauchy prior scale on effect size (0.707 = the 'medium' default).
    For a one-sample or paired test pass only ``nx``; for two independent samples
    pass both ``nx`` and ``ny``.
    """
    if ny is None or ny == 1 or paired:
        n = nx
        df = nx - 1
    else:
        n = nx * ny / (nx + ny)
        df = nx + ny - 2

    def integrand(g: float) -> float:
        return (
            (1 + n * g * r**2) ** -0.5
            * (1 + t**2 / ((1 + n * g * r**2) * df)) ** (-(df + 1) / 2)
            * (2 * np.pi) ** -0.5
            * g**-1.5
            * np.exp(-1 / (2 * g))
        )

    integral, _ = quad(integrand, 0, np.inf)
    null = (1 + t**2 / df) ** (-(df + 1) / 2)
    return float(integral / null)


def interpret_bf(bf10: float) -> str:
    """Jeffreys' evidence categories, phrased for the direction the data favor."""
    if np.isnan(bf10) or bf10 <= 0:
        return "undefined"
    bf, favors = (bf10, "H1 (an effect)") if bf10 >= 1 else (1 / bf10, "H0 (no effect)")
    if bf < 1:
        band = "no"
    elif bf < 3:
        band = "anecdotal"
    elif bf < 10:
        band = "moderate"
    elif bf < 30:
        band = "strong"
    elif bf < 100:
        band = "very strong"
    else:
        band = "extreme"
    return f"{band} evidence for {favors}"
