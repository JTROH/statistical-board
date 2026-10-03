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


# Default prior scale for regression coefficients: BayesFactor's "medium"
# (sqrt(2)/4), so the numbers match what an R user would get from lmBF.
REGRESSION_PRIOR_SCALE = float(np.sqrt(2.0) / 4.0)


def bayesfactor_r2(n: int, p: int, r2: float, *, r: float = REGRESSION_PRIOR_SCALE) -> float:
    """JZS BF10 of a linear model with ``p`` predictors against intercept-only,
    from its R² (Liang et al. 2008; Rouder & Morey 2012).

    BF10 = integral of (1+g)^((n-p-1)/2) * (1 + g(1-R²))^(-(n-1)/2) over the
    Zellner-Siow prior g ~ InvGamma(1/2, r² n / 2). All predictors share one g,
    as in BayesFactor's ``regressionBF`` / ``linearReg.R2stat``. Integrated in
    log g with the peak shifted to zero, so large n does not overflow.
    """
    if p < 1 or p >= n - 1:
        raise ValueError("need 1 <= predictors < n - 1")
    if not 0.0 <= r2 < 1.0:
        raise ValueError("R² must be in [0, 1)")
    b = r**2 * n / 2.0

    def log_integrand(u: float) -> float:
        # In u = log g: InvGamma(1/2, b) prior (Gamma(1/2) = sqrt(pi)), the
        # likelihood ratio, and the Jacobian e^u.
        log_prior = 0.5 * np.log(b) - 0.5 * np.log(np.pi) - 1.5 * u - b * np.exp(-u)
        log_lik = 0.5 * (n - p - 1) * np.logaddexp(0.0, u) - 0.5 * (n - 1) * np.logaddexp(0.0, u + np.log1p(-r2))
        return log_lik + log_prior + u

    # Locate the peak on a coarse grid, then integrate a window around it in
    # which the integrand falls by many orders of magnitude on both sides.
    grid = np.linspace(-30.0, 40.0, 3501)
    values = np.array([log_integrand(u) for u in grid])
    peak, centre = float(values.max()), float(grid[int(values.argmax())])
    f = lambda u: np.exp(log_integrand(u) - peak)  # noqa: E731
    left, _ = quad(f, centre - 40.0, centre, limit=200)
    right, _ = quad(f, centre, centre + 40.0, limit=200)
    return float(np.exp(np.log(left + right) + peak))


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
