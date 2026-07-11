"""Effect sizes and their plain-language interpretation.

Effect sizes stay in the engine (not as a standalone critic persona) so every
comparison reports magnitude alongside its p-value — the frequentist critic and
the judge lean on these to separate 'statistically detectable' from 'actually
big enough to matter'.
"""

from __future__ import annotations

import numpy as np


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d for two independent samples, using the pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def hedges_g(a: list[float], b: list[float]) -> float:
    """Bias-corrected standardized mean difference (small-sample safe)."""
    d = cohens_d(a, b)
    n = len(a) + len(b)
    correction = 1 - 3 / (4 * n - 9)
    return float(d * correction)


def eta_squared(ss_between: float, ss_total: float) -> float:
    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


def omega_squared(ss_between: float, df_between: int, ms_within: float, ss_total: float) -> float:
    denom = ss_total + ms_within
    if denom == 0:
        return float("nan")
    return float((ss_between - df_between * ms_within) / denom)


def interpret_d(d: float) -> str:
    """Cohen's rough benchmarks — a communication aid, not a decision rule."""
    ad = abs(d)
    if np.isnan(ad):
        return "undefined"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def interpret_eta(eta: float) -> str:
    if np.isnan(eta):
        return "undefined"
    if eta < 0.01:
        return "negligible"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"
