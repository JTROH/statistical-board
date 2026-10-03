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
from .model import is_estimable, model_matrix, model_terms, residual_df, term_label
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


def d_efficiency(matrix: np.ndarray, terms: list, blocks: np.ndarray | None = None) -> float:
    """det(X'X / n)^(1/p). Higher is better. Measures how precisely the
    *coefficients* are estimated — the classic criterion, and the wrong one to
    optimise if your endgame is predicting across a range."""
    x = model_matrix(matrix, terms, blocks)
    n, p = x.shape
    sign, logdet = np.linalg.slogdet(x.T @ x / n)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p))


def a_efficiency(matrix: np.ndarray, terms: list, blocks: np.ndarray | None = None) -> float:
    """p / trace(n (X'X)^-1). Higher is better. Average coefficient variance."""
    x = model_matrix(matrix, terms, blocks)
    n, p = x.shape
    try:
        inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return 0.0
    tr = float(np.trace(inv) * n)
    return float(p / tr) if tr > 0 else 0.0


def scaled_prediction_variance(
    matrix: np.ndarray, terms: list, points: np.ndarray, blocks: np.ndarray | None = None
) -> np.ndarray:
    """SPV(x) = n * x_m' (X'X)^-1 x_m at each point.

    Scaling by n makes designs of different sizes comparable: it answers "how
    much uncertainty per run am I buying?". With blocks, the prediction is for
    the average block (block columns at zero under effect coding).
    """
    x = model_matrix(matrix, terms, blocks)
    n = x.shape[0]
    try:
        inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return np.full(points.shape[0], np.inf)
    xm = model_matrix(points, terms)
    if x.shape[1] > xm.shape[1]:
        xm = np.hstack([xm, np.zeros((xm.shape[0], x.shape[1] - xm.shape[1]))])
    return n * np.einsum("ij,jk,ik->i", xm, inv, xm)


def design_space_sample(n_factors: int, n_points: int = FDS_SAMPLE_SIZE, seed: int = 0) -> np.ndarray:
    """A low-discrepancy *uniform* sample of the cube the scientist declared.

    Used for the averaged quantities — the I-value, the median, the FDS curve —
    which are integrals over the region and so need points spread evenly, with
    no extra weight at the edges.

    Sobol rather than uniform random so coverage is even and the numbers are
    stable run to run: two designs must be comparable, which means the sample
    must not wobble underneath them.
    """
    m = max(1, int(np.ceil(np.log2(max(n_points, 2)))))
    sampler = qmc.Sobol(d=n_factors, scramble=True, seed=seed)
    unit = sampler.random_base2(m)
    return 2.0 * unit - 1.0


def extreme_point_sample(n_factors: int, n_points: int = FDS_SAMPLE_SIZE, seed: int = 0) -> np.ndarray:
    """Sample aimed at finding the *worst* prediction variance in the region.

    Prediction variance peaks at the boundary — almost always at a vertex — and
    a uniform interior sample will sail straight past it. Missing that peak
    inflates G-efficiency above 1.0, which is impossible (the Kiefer-Wolfowitz
    equivalence theorem puts max SPV at or above the number of model terms), so
    the vertices are included explicitly rather than hoped for.
    """
    vertices = np.array(list(product([-1.0, 1.0], repeat=n_factors)), dtype=float)
    if vertices.shape[0] > 4096:  # keep the evaluation bounded for wide designs
        rng = np.random.default_rng(seed)
        vertices = vertices[rng.choice(vertices.shape[0], 4096, replace=False)]
    edges = np.vstack([np.eye(n_factors), -np.eye(n_factors)])
    return np.vstack([vertices, edges, design_space_sample(n_factors, n_points, seed)])


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

    The curve is for an *ideal* two-level design: orthogonal, balanced, no
    centre points, every run spent on the factorial. Real options sit on or
    below it — centre points and replicates add runs without adding power to a
    main effect — so it is a best case and a floor on the runs needed, which is
    exactly what "how many more runs do I need?" is asking.
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


def _ideal_power(std_effect: float, n: int, n_terms: int, alpha: float) -> float:
    """Power of one main effect in an ideal orthogonal design of ``n`` runs.

    Coefficient SE is sigma / sqrt(n) and the coded coefficient is half the
    across-range target effect, as in :func:`power_report`.
    """
    return _power_for_ncp((std_effect / 2.0) * np.sqrt(n), n - n_terms, alpha)


def power_curve(spec: DesignSpec, up_to: int, target: float = 0.80) -> PowerCurve | None:
    """Power vs run count for the model and target effect in ``spec``.

    Returns ``None`` when the target effect or noise SD is unknown, because
    there is then no honest curve to draw.
    """
    response = spec.primary_response
    std_effect = response.standardised_effect if response else None
    if std_effect is None:
        return None

    n_terms = len(model_terms(spec.n_factors, spec.model_order))
    start = n_terms + 2  # at least two residual df, or the t-test is meaningless
    stress = noise_stress_factor(response)
    curve = PowerCurve(noise_stress_factor=stress)
    for n in range(start, max(up_to, start) + 1):
        curve.n_runs.append(n)
        curve.power.append(_ideal_power(std_effect, n, n_terms, spec.alpha))
        curve.power_if_noise_high.append(_ideal_power(std_effect / stress, n, n_terms, spec.alpha))

    def first_reaching(effect: float) -> int | None:
        for n in range(start, POWER_CURVE_SEARCH_CAP + 1):
            if _ideal_power(effect, n, n_terms, spec.alpha) >= target:
                return n
        return None

    curve.runs_for_80 = first_reaching(std_effect)
    curve.runs_for_80_if_noise_high = first_reaching(std_effect / stress)
    return curve


def _min_term_power(
    matrix: np.ndarray, terms: list, std_effect: float, alpha: float, blocks: np.ndarray | None = None
) -> float | None:
    """Weakest-term power, without the rest of :func:`power_report`.

    The robustness check calls this once per loss scenario, so it skips the
    detectable-effect bisection that ``power_report`` does.
    """
    df = residual_df(matrix, terms, blocks)
    if df <= 0:
        return None
    x = model_matrix(matrix, terms, blocks)
    diag = np.diag(np.linalg.inv(x.T @ x))
    # Power rises with the non-centrality parameter, so the weakest term is the
    # one with the smallest — one t-distribution call instead of one per term.
    ncp = min(std_effect * _coefficient_per_target(t) / float(np.sqrt(diag[j])) for j, t in enumerate(terms) if t)
    return _power_for_ncp(ncp, df, alpha)


def power_report(design: Design, spec: DesignSpec) -> PowerReport:
    """Power for every model term at the scientist's stated target effect.

    Convention: ``target_effect`` is the change in the response across a
    factor's full low-to-high range, so a main-effect coded coefficient is half
    of it. Interactions and curvature follow :data:`CURVATURE_RULE`.
    """
    terms = model_terms(design.n_factors, spec.model_order)
    df = residual_df(design.matrix, terms, design.blocks)
    response = spec.primary_response
    report = PowerReport(residual_df=df, saturated=df <= 0, noise_stress_factor=noise_stress_factor(response))

    if not is_estimable(design.matrix, terms, blocks=design.blocks) or df <= 0:
        return report

    x = model_matrix(design.matrix, terms, design.blocks)
    inv = np.linalg.inv(x.T @ x)

    std_effect = response.standardised_effect if response else None
    report.standardised_effect = std_effect

    # Effect detectable at 80% power on the weakest term. Reported in standard
    # deviations, so it stays meaningful even when the scientist has not yet
    # committed to a target effect size.
    model = [(j, t) for j, t in enumerate(terms) if t]
    ncp_80 = _solve_ncp_for_power(0.80, df, spec.alpha)
    report.detectable_effect_sd = float(
        max(ncp_80 * float(np.sqrt(inv[j, j])) / _coefficient_per_target(t) for j, t in model)
    )
    if response is not None and response.noise_sd:
        report.detectable_effect_units = report.detectable_effect_sd * abs(response.noise_sd)

    if std_effect is not None:
        buckets = {
            "main": report.per_term,
            "interaction": report.interaction_terms,
            "curvature": report.curvature_terms,
        }
        pessimistic: list[float] = []
        for j, term in model:
            ncp = std_effect * _coefficient_per_target(term) / float(np.sqrt(inv[j, j]))
            label = term_label(term, spec.factor_names)
            buckets[term_kind(term)][label] = _power_for_ncp(ncp, df, spec.alpha)
            pessimistic.append(_power_for_ncp(ncp / report.noise_stress_factor, df, spec.alpha))
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


def prediction_report(design: Design, spec: DesignSpec, seed: int = 0) -> PredictionReport:
    terms = model_terms(design.n_factors, spec.model_order)
    if not is_estimable(design.matrix, terms, blocks=design.blocks):
        return PredictionReport()

    # Averaged quantities come from the uniform sample; the maximum comes from
    # a sample that deliberately includes the boundary. Using one set for both
    # gets one of the two answers wrong.
    uniform = design_space_sample(design.n_factors, seed=seed)
    extreme = extreme_point_sample(design.n_factors, seed=seed)
    spv = scaled_prediction_variance(design.matrix, terms, uniform, design.blocks)
    spv_extreme = scaled_prediction_variance(design.matrix, terms, extreme, design.blocks)
    spv_sorted = np.sort(spv)
    p = len(terms)
    max_spv = float(np.max(spv_extreme))

    # Report the FDS curve thinned to 101 quantiles — enough to draw, small
    # enough to ship inside a JSON response.
    idx = np.linspace(0, len(spv_sorted) - 1, 101).astype(int)

    return PredictionReport(
        i_value=float(np.mean(spv)),
        median_spv=float(np.median(spv)),
        max_spv=max_spv,
        g_efficiency=float(p / max_spv) if max_spv > 0 else 0.0,
        fds_curve=[float(v) for v in spv_sorted[idx]],
        axial_points_outside_range=bool(np.abs(design.matrix).max() > 1.0 + 1e-9),
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
    terms = model_terms(design.n_factors, spec.model_order)
    blocks = design.blocks
    if r == 0 or not is_estimable(design.matrix, terms, blocks=blocks):
        return RobustnessReport(n_losses=r, n_scenarios=0, fraction_estimable=0.0, applicable=False)

    base_d = d_efficiency(design.matrix, terms, blocks)
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
        if is_estimable(kept, terms, blocks=kept_blocks):
            n_ok += 1
            if base_d > 0:
                worst_ratio = min(worst_ratio, d_efficiency(kept, terms, kept_blocks) / base_d)
            if std_effect is not None:
                power = _min_term_power(kept, terms, std_effect, spec.alpha, kept_blocks)
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


def evaluate(design: Design, spec: DesignSpec, seed: int = 0) -> DesignProperties:
    """Score one design against one spec. Pure and deterministic."""
    terms = model_terms(design.n_factors, spec.model_order)
    estimable = is_estimable(design.matrix, terms, blocks=design.blocks)
    return DesignProperties(
        design=design,
        estimable=estimable,
        n_runs=design.n_runs,
        n_model_terms=len(terms),
        residual_df=residual_df(design.matrix, terms, design.blocks),
        d_efficiency=d_efficiency(design.matrix, terms, design.blocks) if estimable else 0.0,
        a_efficiency=a_efficiency(design.matrix, terms, design.blocks) if estimable else 0.0,
        power=power_report(design, spec),
        prediction=prediction_report(design, spec, seed=seed),
        robustness=robustness_report(design, spec, seed=seed),
        aliasing=alias_report(design, spec.model_order, spec.factor_names),
    )
