"""Scoring a design on the four axes the options table compares.

1. **Power** — will this design actually see the effect you care about?
2. **Aliasing** — what will it refuse to tell you apart? (in ``alias.py``)
3. **Prediction precision** — how confidently can it predict anywhere in the
   space, including near the edges? This is the axis that decides whether the
   resulting data can support a NOR/PAR claim later, and it is also the
   machinery Phase 3's I-optimal search needs.
4. **Robustness** — if bioreactors are lost, does the study still answer the
   question?

Run count is reported alongside but is not scored: it is the price, not a
virtue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
from scipy import stats
from scipy.stats import qmc

from .alias import AliasReport, alias_report
from .model import (
    Levels,
    effect_codes,
    information,
    is_estimable,
    model_matrix,
    model_terms,
    n_columns,
    residual_df,
    stratum_df,
    term_label,
    term_slices,
)
from .spec import Design, DesignSpec

# The "what if the noise is worse than you said" check. With a guessed noise SD
# the tool uses a flat 1.5x. With a measured one it uses the upper confidence
# bound of that SD given its degrees of freedom — about 1.5x at 8 df, about 1.7x
# at 3 df — so the stress test reflects how much the estimate can be trusted.
DEFAULT_NOISE_STRESS = 1.5
NOISE_UPPER_CONFIDENCE = 0.80

# Points used to characterise prediction variance across the design space.
FDS_SAMPLE_SIZE = 2048
# Cap on how many run-loss scenarios we enumerate before switching to sampling.
MAX_LOSS_SCENARIOS = 400


# --------------------------------------------------------------------------
# Optimality criteria (shared with the Phase 3 optimal-design search)
# --------------------------------------------------------------------------


def d_efficiency(
    matrix: np.ndarray,
    terms: list,
    blocks: np.ndarray | None = None,
    levels: Levels = None,
    whole_plots: np.ndarray | None = None,
    eta: float = 1.0,
) -> float:
    """det(X'X / n)^(1/p). Higher is better. Measures how precisely the
    *coefficients* are estimated — the classic criterion, and the wrong one to
    optimise if your endgame is predicting across a range. For a split-plot
    design, X' V^-1 X replaces X'X (see ``model.information``)."""
    x = model_matrix(matrix, terms, blocks, levels)
    n, p = x.shape
    # errstate: some numpy builds (Accelerate on macOS) emit spurious divide/
    # overflow warnings from slogdet on matrices wider than 32 while returning
    # the right answer; a genuinely singular matrix still comes back sign <= 0.
    with np.errstate(all="ignore"):
        sign, logdet = np.linalg.slogdet(information(x, whole_plots, eta) / n)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p))


def a_efficiency(
    matrix: np.ndarray,
    terms: list,
    blocks: np.ndarray | None = None,
    levels: Levels = None,
    whole_plots: np.ndarray | None = None,
    eta: float = 1.0,
) -> float:
    """p / trace(n (X'X)^-1). Higher is better. Average coefficient variance."""
    x = model_matrix(matrix, terms, blocks, levels)
    n, p = x.shape
    try:
        inv = np.linalg.inv(information(x, whole_plots, eta))
    except np.linalg.LinAlgError:
        return 0.0
    tr = float(np.trace(inv) * n)
    return float(p / tr) if tr > 0 else 0.0


def scaled_prediction_variance(
    matrix: np.ndarray,
    terms: list,
    points: np.ndarray,
    blocks: np.ndarray | None = None,
    levels: Levels = None,
    whole_plots: np.ndarray | None = None,
    eta: float = 1.0,
) -> np.ndarray:
    """SPV(x) = n * x_m' (X'X)^-1 x_m at each point.

    Scaling by n makes designs of different sizes comparable: it answers "how
    much uncertainty per run am I buying?". With blocks, the prediction is for
    the average block (block columns at zero under effect coding).
    """
    x = model_matrix(matrix, terms, blocks, levels)
    n = x.shape[0]
    try:
        inv = np.linalg.inv(information(x, whole_plots, eta))
    except np.linalg.LinAlgError:
        return np.full(points.shape[0], np.inf)
    xm = model_matrix(points, terms, levels=levels)
    if x.shape[1] > xm.shape[1]:
        xm = np.hstack([xm, np.zeros((xm.shape[0], x.shape[1] - xm.shape[1]))])
    return n * np.einsum("ij,jk,ik->i", xm, inv, xm)


def design_space_sample(
    n_factors: int, n_points: int = FDS_SAMPLE_SIZE, seed: int = 0, levels: Levels = None
) -> np.ndarray:
    """A low-discrepancy *uniform* sample of the region the scientist declared.

    Used for the averaged quantities — the I-value, the median, the FDS curve —
    which are integrals over the region and so need points spread evenly, with
    no extra weight at the edges. A categorical factor is sampled with every
    level equally likely.

    Sobol rather than uniform random so coverage is even and the numbers are
    stable run to run: two designs must be comparable, which means the sample
    must not wobble underneath them.
    """
    m = max(1, int(np.ceil(np.log2(max(n_points, 2)))))
    sampler = qmc.Sobol(d=n_factors, scramble=True, seed=seed)
    unit = sampler.random_base2(m)
    points = 2.0 * unit - 1.0
    if levels is not None:
        for j, k in enumerate(levels):
            if k:
                points[:, j] = np.minimum(np.floor(unit[:, j] * k), k - 1)
    return points


def _vertices(levels: list[int]) -> np.ndarray:
    """Every corner of the region: +/-1 per continuous factor, every level per categorical one."""
    axes = [list(range(k)) if k else [-1.0, 1.0] for k in levels]
    return np.array(list(product(*axes)), dtype=float)


def extreme_point_sample(
    n_factors: int, n_points: int = FDS_SAMPLE_SIZE, seed: int = 0, levels: Levels = None
) -> np.ndarray:
    """Sample aimed at finding the *worst* prediction variance in the region.

    Prediction variance peaks at the boundary — almost always at a vertex — and
    a uniform interior sample will sail straight past it. Missing that peak
    inflates G-efficiency above 1.0, which is impossible (the Kiefer-Wolfowitz
    equivalence theorem puts max SPV at or above the number of model terms), so
    the vertices are included explicitly rather than hoped for. With
    categorical factors, a vertex is every combination of levels.
    """
    lv = list(levels) if levels is not None else [0] * n_factors
    n_vertices = int(np.prod([k if k else 2 for k in lv], dtype=float))
    rng = np.random.default_rng(seed)
    if n_vertices <= 4096:
        vertices = _vertices(lv)
    elif not any(lv):  # keep the evaluation bounded for wide designs
        vertices = _vertices(lv)
        vertices = vertices[rng.choice(vertices.shape[0], 4096, replace=False)]
    else:
        vertices = np.column_stack(
            [rng.integers(0, k, 4096) if k else rng.choice([-1.0, 1.0], 4096) for k in lv]
        ).astype(float)
    cont = [j for j, k in enumerate(lv) if not k]
    edges = np.zeros((2 * len(cont), n_factors))
    for r, j in enumerate(cont):
        edges[r, j], edges[len(cont) + r, j] = 1.0, -1.0
    return np.vstack([vertices, edges, design_space_sample(n_factors, n_points, seed, levels)])


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def noise_stress_factor(response) -> float:
    """How much larger than entered the noise SD is assumed in the stress test."""
    df = getattr(response, "noise_df", None) if response is not None else None
    if not df or df <= 0:
        return DEFAULT_NOISE_STRESS
    return float(np.sqrt(df / stats.chi2.ppf(1.0 - NOISE_UPPER_CONFIDENCE, df)))


def _power_for_ncp(ncp: float, df: int, alpha: float) -> float:
    """Two-sided t-test power from a non-centrality parameter."""
    if df <= 0:
        return float("nan")
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def _solve_ncp_for_power(target: float, df: int, alpha: float) -> float:
    """Smallest non-centrality parameter reaching ``target`` power.

    Bisection rather than a closed form because power in the non-central t is
    monotone in the non-centrality parameter but not analytically invertible.
    """
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _power_for_ncp(mid, df, alpha) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# How ``target_effect`` maps onto each kind of coded coefficient. Printed in the
# memo, because the curvature rule is a choice and power for a squared term
# moves by a factor of several depending on it.
#
# - main effect b*x: the change from low to high is 2b, so b = target / 2.
# - interaction c*x1*x2: the classical interaction effect is 2c, so c = target / 2.
# - curvature q*x^2: the rise or dip from the centre to either edge of the
#   declared range is q, so q = target.
CURVATURE_RULE = (
    "For a curvature (squared) term, the target effect is read as the rise or dip from the "
    "centre of a factor's range to its edge, with the other factors at their centre."
)
# The same choice for categorical factors, printed in the memo whenever one is
# in the model. A k-level main effect has k-1 coefficients, so "the effect" has
# to be pinned to something a scientist would act on: the gap between two
# levels. Power is quoted for the hardest pair to tell apart.
#
# - categorical main effect: the difference between two levels' means.
# - categorical x anything: half the difference, between two levels, of the
#   other factor's effect — the classical interaction convention, which makes a
#   two-level categorical behave exactly like a continuous factor.
CATEGORICAL_RULE = (
    "For a categorical factor, the target effect is read as the difference between two of its levels, "
    "and power is quoted for the pair of levels that is hardest to tell apart."
)


def term_kind(term: tuple) -> str:
    """``main``, ``interaction`` or ``curvature``."""
    if len(term) == 1:
        return "main"
    if len(term) == 2 and term[0] == term[1]:
        return "curvature"
    return "interaction"


def _coefficient_per_target(term: tuple) -> float:
    """Coded coefficient produced by a target effect of 1, for this kind of term."""
    return 1.0 if term_kind(term) == "curvature" else 0.5


def _has_categorical(term: tuple, levels: Levels) -> bool:
    return levels is not None and any(levels[i] for i in term)


def term_contrasts(term: tuple, levels: Levels = None) -> list[np.ndarray]:
    """Contrast vectors ``v`` (over the term's columns) with ``v'beta`` = the term's effect.

    A target effect of ``delta`` is detected when ``v'beta = delta``; the
    contrast's standard error is ``sigma * sqrt(v'(X'X)^-1 v)``. For a
    continuous factor the low-to-high difference of its coded column is 2; for
    a categorical factor it is the difference of two levels' coded rows, one
    contrast per pair of levels. An interaction's contrast is the product of
    its factors', halved once per extra factor (the classical convention).
    """
    if term_kind(term) == "curvature":
        return [np.array([1.0])]
    per_factor: list[list[np.ndarray]] = []
    for i in term:
        k = 0 if levels is None else int(levels[i])
        if k:
            codes = effect_codes(k)
            per_factor.append([codes[b] - codes[a] for a, b in combinations(range(k), 2)])
        else:
            per_factor.append([np.array([2.0])])
    scale = 0.5 ** (len(term) - 1)
    out = []
    for parts in product(*per_factor):
        v = np.array([1.0])
        for part in parts:
            v = np.kron(v, part)
        out.append(scale * v)
    return out


def _worst_se(cov: np.ndarray, term: tuple, levels: Levels = None) -> float:
    """Largest contrast SE (in units of sigma) for a term, from its block of (X'X)^-1.

    For a one-column continuous term this is ``sqrt(c) / coefficient-per-target``,
    computed exactly that way so continuous designs keep their published numbers.
    """
    if not _has_categorical(term, levels):
        return float(np.sqrt(cov[0, 0])) / _coefficient_per_target(term)
    return max(float(np.sqrt(v @ cov @ v)) for v in term_contrasts(term, levels))


def _term_ses(inv: np.ndarray, terms: list, levels: Levels = None) -> list[tuple[tuple, float]]:
    """(term, worst contrast SE) for every non-intercept term."""
    slices = term_slices(terms, levels)
    return [(t, _worst_se(inv[s, s], t, levels)) for t, s in zip(terms, slices, strict=True) if t]


@dataclass
class PowerReport:
    """Power to detect the target effect, for every term in the model.

    ``per_term`` holds the main effects; ``interaction_terms`` and
    ``curvature_terms`` hold the rest. The headline (``min_power``) is the
    weakest term the scientist asked the model to estimate — a quadratic model
    for finding an optimum is only as good as its curvature estimates.

    ``None`` values mean the scientist did not supply a target effect and a
    noise estimate, so power is genuinely unknown — the tool says so rather
    than inventing a default.
    """

    per_term: dict[str, float] = field(default_factory=dict)
    interaction_terms: dict[str, float] = field(default_factory=dict)
    curvature_terms: dict[str, float] = field(default_factory=dict)
    residual_df: int = 0
    saturated: bool = False
    standardised_effect: float | None = None
    # Effect (in SDs) detectable at 80% on the *weakest* model term, so it agrees
    # with ``min_power`` about which side of 80% the target falls.
    detectable_effect_sd: float | None = None
    # The same figure in the response's own units — only when a noise SD was given.
    detectable_effect_units: float | None = None
    # Power for the weakest model term if the true noise SD is
    # ``noise_stress_factor`` times what was entered. The noise SD is the input
    # scientists are least sure of, and this is what it costs to be wrong.
    min_power_if_noise_high: float | None = None
    noise_stress_factor: float = DEFAULT_NOISE_STRESS
    # Split-plot only: degrees of freedom of the two error strata. Whole-plot
    # terms are tested on ``whole_plot_df``, everything else on ``subplot_df``.
    whole_plot_df: int | None = None
    subplot_df: int | None = None

    @staticmethod
    def _min(values: dict[str, float]) -> float | None:
        finite = [v for v in values.values() if not np.isnan(v)]
        return min(finite) if finite else None

    @property
    def min_main_effect_power(self) -> float | None:
        return self._min(self.per_term)

    @property
    def min_interaction_power(self) -> float | None:
        return self._min(self.interaction_terms)

    @property
    def min_curvature_power(self) -> float | None:
        return self._min(self.curvature_terms)

    @property
    def min_power(self) -> float | None:
        """The weakest term in the model: the headline power figure."""
        return self._min({**self.per_term, **self.interaction_terms, **self.curvature_terms})


@dataclass
class PowerCurve:
    """Power against run count, for the scientist's target effect.

    The curve is for an *ideal* design: orthogonal, balanced, no centre points,
    every run spent on the factorial — two-level when every factor is
    continuous, and balanced across the levels of the factor with the most
    levels otherwise. Real options sit on or below it — centre points and
    replicates add runs without adding power to a main effect — so it is a best
    case and a floor on the runs needed, which is exactly what "how many more
    runs do I need?" is asking.
    """

    n_runs: list[int] = field(default_factory=list)
    power: list[float] = field(default_factory=list)
    power_if_noise_high: list[float] = field(default_factory=list)
    runs_for_80: int | None = None  # smallest run count reaching 80% power
    runs_for_80_if_noise_high: int | None = None
    noise_stress_factor: float = DEFAULT_NOISE_STRESS


# Beyond this many runs the search for 80% power gives up; the answer is
# "more than the study can afford" either way.
POWER_CURVE_SEARCH_CAP = 2000


def _ideal_power(std_effect: float, n: int, n_terms: int, alpha: float, max_levels: int = 2) -> float:
    """Power of the weakest main effect in an ideal orthogonal design of ``n`` runs.

    Continuous: coefficient SE is sigma / sqrt(n) and the coded coefficient is
    half the across-range target effect, as in :func:`power_report`. A k-level
    categorical factor in a balanced design has n/k runs per level, so the
    difference between two levels has SE sigma * sqrt(2k/n); at k = 2 that is
    the continuous case.
    """
    if max_levels <= 2:
        return _power_for_ncp((std_effect / 2.0) * np.sqrt(n), n - n_terms, alpha)
    return _power_for_ncp(std_effect * np.sqrt(n / (2.0 * max_levels)), n - n_terms, alpha)


def power_curve(spec: DesignSpec, up_to: int, target: float = 0.80) -> PowerCurve | None:
    """Power vs run count for the model and target effect in ``spec``.

    Returns ``None`` when the target effect or noise SD is unknown, because
    there is then no honest curve to draw.
    """
    response = spec.primary_response
    std_effect = response.standardised_effect if response else None
    if std_effect is None:
        return None

    levels = spec.levels
    n_terms = n_columns(list(fitted_model(spec).terms), levels)
    max_levels = max([2, *levels])
    start = n_terms + 2  # at least two residual df, or the t-test is meaningless
    stress = noise_stress_factor(response)
    curve = PowerCurve(noise_stress_factor=stress)
    for n in range(start, max(up_to, start) + 1):
        curve.n_runs.append(n)
        curve.power.append(_ideal_power(std_effect, n, n_terms, spec.alpha, max_levels))
        curve.power_if_noise_high.append(_ideal_power(std_effect / stress, n, n_terms, spec.alpha, max_levels))

    def first_reaching(effect: float) -> int | None:
        for n in range(start, POWER_CURVE_SEARCH_CAP + 1):
            if _ideal_power(effect, n, n_terms, spec.alpha, max_levels) >= target:
                return n
        return None

    curve.runs_for_80 = first_reaching(std_effect)
    curve.runs_for_80_if_noise_high = first_reaching(std_effect / stress)
    return curve


def _term_dfs(
    x: np.ndarray, terms: list, df: int, whole_plots: np.ndarray | None, levels: Levels
) -> tuple[list[int], int | None, int | None]:
    """Test df for every non-intercept term, plus the two stratum dfs (split-plot only)."""
    model_terms_only = [t for t in terms if t]
    if whole_plots is None:
        return [df] * len(model_terms_only), None, None
    df_wp, df_sp, flags = stratum_df(x, terms, whole_plots, levels)
    return [df_wp if f else df_sp for t, f in zip(terms, flags, strict=True) if t], df_wp, df_sp


def _min_term_power(
    matrix: np.ndarray,
    terms: list,
    std_effect: float,
    alpha: float,
    blocks: np.ndarray | None = None,
    levels: Levels = None,
    whole_plots: np.ndarray | None = None,
    eta: float = 1.0,
) -> float | None:
    """Weakest-term power, without the rest of :func:`power_report`.

    The robustness check calls this once per loss scenario, so it skips the
    detectable-effect bisection that ``power_report`` does.
    """
    df = residual_df(matrix, terms, blocks, levels)
    if df <= 0:
        return None
    x = model_matrix(matrix, terms, blocks, levels)
    inv = np.linalg.inv(information(x, whole_plots, eta))
    dfs, df_wp, df_sp = _term_dfs(x, terms, df, whole_plots, levels)
    if whole_plots is not None and min(df_wp, df_sp) <= 0:
        return None
    # Power rises with the non-centrality parameter, so within each df the
    # weakest term is the one with the smallest — one t call per distinct df.
    weakest: dict[int, float] = {}
    for (_, se), d in zip(_term_ses(inv, terms, levels), dfs, strict=True):
        weakest[d] = min(weakest.get(d, np.inf), std_effect / se)
    return min(_power_for_ncp(ncp, d, alpha) for d, ncp in weakest.items())


def power_report(design: Design, spec: DesignSpec) -> PowerReport:
    """Power for every model term at the scientist's stated target effect.

    Convention: ``target_effect`` is the change in the response across a
    factor's full low-to-high range, so a main-effect coded coefficient is half
    of it. Interactions and curvature follow :data:`CURVATURE_RULE`, and
    categorical factors :data:`CATEGORICAL_RULE`.
    """
    levels = design.levels
    terms = list(fitted_model(spec).terms)
    df = residual_df(design.matrix, terms, design.blocks, levels)
    response = spec.primary_response
    report = PowerReport(residual_df=df, saturated=df <= 0, noise_stress_factor=noise_stress_factor(response))

    if not is_estimable(design.matrix, terms, blocks=design.blocks, levels=levels) or df <= 0:
        return report

    x = model_matrix(design.matrix, terms, design.blocks, levels)
    inv = np.linalg.inv(information(x, design.whole_plots, spec.wp_variance_ratio))
    dfs, report.whole_plot_df, report.subplot_df = _term_dfs(x, terms, df, design.whole_plots, levels)
    if design.whole_plots is not None and min(report.whole_plot_df, report.subplot_df) <= 0:
        report.saturated = True
        return report

    std_effect = response.standardised_effect if response else None
    report.standardised_effect = std_effect

    # Effect detectable at 80% power on the weakest term. Reported in standard
    # deviations, so it stays meaningful even when the scientist has not yet
    # committed to a target effect size. (A split-plot design has two test df.)
    model = _term_ses(inv, terms, levels)
    ncp_80 = {d: _solve_ncp_for_power(0.80, d, spec.alpha) for d in set(dfs)}
    report.detectable_effect_sd = float(max(ncp_80[d] * se for (_, se), d in zip(model, dfs, strict=True)))
    if response is not None and response.noise_sd:
        report.detectable_effect_units = report.detectable_effect_sd * abs(response.noise_sd)

    if std_effect is not None:
        buckets = {
            "main": report.per_term,
            "interaction": report.interaction_terms,
            "curvature": report.curvature_terms,
        }
        pessimistic: list[float] = []
        for (term, se), d in zip(model, dfs, strict=True):
            ncp = std_effect / se
            label = term_label(term, spec.factor_names)
            buckets[term_kind(term)][label] = _power_for_ncp(ncp, d, spec.alpha)
            pessimistic.append(_power_for_ncp(ncp / report.noise_stress_factor, d, spec.alpha))
        report.min_power_if_noise_high = min(pessimistic)

    return report


# --------------------------------------------------------------------------
# Prediction precision
# --------------------------------------------------------------------------


@dataclass
class PredictionReport:
    """How well the design predicts across the declared design space."""

    i_value: float = float("inf")  # mean SPV — the I-optimality criterion
    median_spv: float = float("inf")
    max_spv: float = float("inf")
    g_efficiency: float = 0.0
    fds_curve: list[float] = field(default_factory=list)  # sorted SPV, for the Phase 2 plot
    axial_points_outside_range: bool = False


def region_samples(spec: DesignSpec, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(uniform, extreme) samples of the region runs may go in, in coded units.

    The region is the declared ranges and levels, cut down by the spec's
    constraints: a prediction where no run may ever go is not one anyone will
    make. Both the prediction score and the I-optimal search use this, so the
    search optimises exactly the number the options table reports. If the
    constraints leave no sampled point, the whole box is used rather than
    nothing.
    """
    levels = spec.levels if spec.has_categorical else None
    uniform = design_space_sample(spec.n_factors, seed=seed, levels=levels)
    extreme = extreme_point_sample(spec.n_factors, seed=seed, levels=levels)
    if spec.is_constrained:
        ok_u, ok_e = spec.feasible(uniform), spec.feasible(extreme)
        if ok_u.any():
            uniform = uniform[ok_u]
        if ok_e.any():
            extreme = extreme[ok_e]
    return uniform, extreme


@dataclass(frozen=True)
class FittedModel:
    """The terms the scientist's model will actually fit.

    Normally every term of the chosen model order. A disallowed combination
    can make a term impossible to estimate by *any* design — never running
    media C with clone 2 leaves part of the media x cell line interaction
    unseen — and such a term is dropped here, once, for every design alike,
    rather than making every candidate look singular. Dropped terms are still
    checked as potential aliases, because if they are active they bias what
    is fitted.
    """

    terms: tuple
    dropped: tuple = ()


_FITTED_CACHE: dict[tuple, FittedModel] = {}


def _spec_key(spec: DesignSpec) -> tuple:
    return (
        tuple(spec.factors),
        spec.model_order,
        tuple(c.describe() for c in spec.constraints),
        tuple(d.describe() for d in spec.disallowed),
    )


def fitted_model(spec: DesignSpec) -> FittedModel:
    """The model terms to fit, with any that the constraints make inestimable dropped.

    A term is kept when adding it to the terms before it (in order: intercept,
    main effects, interactions, curvature) raises the rank of the model matrix
    over the feasible region by its full width, so lower-order terms are
    always preferred.
    """
    levels = spec.levels if spec.has_categorical else None
    terms = model_terms(spec.n_factors, spec.model_order, levels)
    if not spec.is_constrained:
        return FittedModel(terms=tuple(terms))
    key = _spec_key(spec)
    if key in _FITTED_CACHE:
        return _FITTED_CACHE[key]
    uniform, extreme = region_samples(spec)
    points = np.vstack([uniform, extreme])
    kept: list = []
    dropped: list = []
    for term in terms:
        trial = [*kept, term]
        x = model_matrix(points, trial, levels=levels)
        if np.linalg.matrix_rank(x, tol=1e-8 * max(1.0, float(np.abs(x).max()))) == x.shape[1]:
            kept.append(term)
        else:
            dropped.append(term)
    result = FittedModel(terms=tuple(kept), dropped=tuple(dropped))
    if len(_FITTED_CACHE) > 128:
        _FITTED_CACHE.clear()
    _FITTED_CACHE[key] = result
    return result


def prediction_report(design: Design, spec: DesignSpec, seed: int = 0) -> PredictionReport:
    levels = design.levels
    terms = list(fitted_model(spec).terms)
    if not is_estimable(design.matrix, terms, blocks=design.blocks, levels=levels):
        return PredictionReport()

    # Averaged quantities come from the uniform sample; the maximum comes from
    # a sample that deliberately includes the boundary. Using one set for both
    # gets one of the two answers wrong. (With linear constraints the corners
    # of the feasible region are not sampled exactly, so G-efficiency is then
    # approximate.)
    uniform, extreme = region_samples(spec, seed=seed)
    wp, eta = design.whole_plots, spec.wp_variance_ratio
    spv = scaled_prediction_variance(design.matrix, terms, uniform, design.blocks, levels, wp, eta)
    spv_extreme = scaled_prediction_variance(design.matrix, terms, extreme, design.blocks, levels, wp, eta)
    spv_sorted = np.sort(spv)
    p = n_columns(terms, levels)
    max_spv = float(max(np.max(spv_extreme), np.max(spv)))

    # Report the FDS curve thinned to 101 quantiles — enough to draw, small
    # enough to ship inside a JSON response.
    idx = np.linspace(0, len(spv_sorted) - 1, 101).astype(int)
    cont = design.continuous_columns

    return PredictionReport(
        i_value=float(np.mean(spv)),
        median_spv=float(np.median(spv)),
        max_spv=max_spv,
        g_efficiency=float(p / max_spv) if max_spv > 0 else 0.0,
        fds_curve=[float(v) for v in spv_sorted[idx]],
        axial_points_outside_range=bool(cont.size and np.abs(design.matrix[:, cont]).max() > 1.0 + 1e-9),
    )


# --------------------------------------------------------------------------
# Robustness to lost runs
# --------------------------------------------------------------------------


@dataclass
class RobustnessReport:
    """What survives when bioreactors fail.

    ``fraction_estimable`` is the headline: the share of plausible run-loss
    scenarios in which the model can still be fitted at all.
    """

    n_losses: int = 0
    n_scenarios: int = 0
    fraction_estimable: float = 1.0
    worst_d_ratio: float = 1.0  # worst D-efficiency relative to the intact design
    # Weakest-term power in the worst loss scenario that can still be fitted.
    # "Still estimable" is not "still answers the question": losing one run
    # from a 17-run CCD can keep every term while cutting power by a quarter.
    # ``None`` when no target effect was given or no scenario survives.
    worst_power_after_loss: float | None = None
    exhaustive: bool = True
    # False when the question does not apply: the intact design already cannot
    # fit the model, or the scientist expects to lose no runs. Reporting "100%
    # robust" for a design that never worked would be actively misleading.
    applicable: bool = True

    @property
    def is_fragile(self) -> bool:
        return self.applicable and self.fraction_estimable < 1.0


def _relabel(blocks: np.ndarray) -> np.ndarray:
    """Block labels renumbered 0..b'-1, for when a loss empties a whole block."""
    _, labels = np.unique(blocks, return_inverse=True)
    return labels.ravel()


def robustness_report(design: Design, spec: DesignSpec, seed: int = 0) -> RobustnessReport:
    r = max(int(spec.expected_run_losses), 0)
    levels = design.levels
    terms = list(fitted_model(spec).terms)
    blocks = design.blocks
    if r == 0 or not is_estimable(design.matrix, terms, blocks=blocks, levels=levels):
        return RobustnessReport(n_losses=r, n_scenarios=0, fraction_estimable=0.0, applicable=False)

    wp, eta = design.whole_plots, spec.wp_variance_ratio
    base_d = d_efficiency(design.matrix, terms, blocks, levels, wp, eta)
    response = spec.primary_response
    std_effect = response.standardised_effect if response else None
    n = design.n_runs
    all_idx = np.arange(n)

    total = 1
    for i in range(r):
        total = total * (n - i) // (i + 1)

    rng = np.random.default_rng(seed)
    if total <= MAX_LOSS_SCENARIOS:
        scenarios = [np.array(c) for c in combinations(range(n), r)]
        exhaustive = True
    else:
        seen: set[tuple[int, ...]] = set()
        while len(seen) < MAX_LOSS_SCENARIOS:
            seen.add(tuple(sorted(rng.choice(n, size=r, replace=False).tolist())))
        scenarios = [np.array(s) for s in seen]
        exhaustive = False

    n_ok = 0
    worst_ratio = 1.0
    powers: list[float] = []
    for lost in scenarios:
        keep = np.setdiff1d(all_idx, lost)
        kept = design.matrix[keep]
        kept_blocks = None if blocks is None else _relabel(blocks[keep])
        kept_wp = None if wp is None else _relabel(wp[keep])
        if is_estimable(kept, terms, blocks=kept_blocks, levels=levels):
            n_ok += 1
            if base_d > 0:
                worst_ratio = min(worst_ratio, d_efficiency(kept, terms, kept_blocks, levels, kept_wp, eta) / base_d)
            if std_effect is not None:
                power = _min_term_power(kept, terms, std_effect, spec.alpha, kept_blocks, levels, kept_wp, eta)
                if power is not None:
                    powers.append(power)
        else:
            worst_ratio = 0.0

    return RobustnessReport(
        n_losses=r,
        n_scenarios=len(scenarios),
        fraction_estimable=n_ok / len(scenarios),
        worst_d_ratio=float(worst_ratio),
        exhaustive=exhaustive,
        worst_power_after_loss=min(powers) if powers else None,
    )


# --------------------------------------------------------------------------
# The whole picture
# --------------------------------------------------------------------------


@dataclass
class DesignProperties:
    """Everything the options table, the explorer and the memo need."""

    design: Design
    estimable: bool
    n_runs: int
    n_model_terms: int
    residual_df: int
    d_efficiency: float
    a_efficiency: float
    power: PowerReport
    prediction: PredictionReport
    robustness: RobustnessReport
    aliasing: AliasReport
    # Runs that break one of the spec's constraints. Only a classical design,
    # which knows nothing of constraints, can have any.
    n_infeasible_runs: int = 0
    # Coefficients the model estimates (block offsets excluded). Equal to
    # ``n_model_terms`` unless a categorical factor's term spans several.
    n_coefficients: int = 0


def evaluate(design: Design, spec: DesignSpec, seed: int = 0) -> DesignProperties:
    """Score one design against one spec. Pure and deterministic."""
    levels = design.levels
    terms = list(fitted_model(spec).terms)
    estimable = is_estimable(design.matrix, terms, blocks=design.blocks, levels=levels)
    feasible = spec.feasible(design.matrix) if spec.is_constrained else np.ones(design.n_runs, dtype=bool)
    wp, eta = design.whole_plots, spec.wp_variance_ratio
    return DesignProperties(
        design=design,
        estimable=estimable,
        n_runs=design.n_runs,
        n_model_terms=len(terms),
        residual_df=residual_df(design.matrix, terms, design.blocks, levels),
        d_efficiency=d_efficiency(design.matrix, terms, design.blocks, levels, wp, eta) if estimable else 0.0,
        a_efficiency=a_efficiency(design.matrix, terms, design.blocks, levels, wp, eta) if estimable else 0.0,
        power=power_report(design, spec),
        prediction=prediction_report(design, spec, seed=seed),
        robustness=robustness_report(design, spec, seed=seed),
        aliasing=alias_report(
            design,
            spec.model_order,
            spec.factor_names,
            terms=terms,
            extra_potential=list(fitted_model(spec).dropped),
            eta=eta,
        ),
        n_infeasible_runs=int((~feasible).sum()),
        n_coefficients=n_columns(terms, levels),
    )
