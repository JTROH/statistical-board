"""Custom (optimal) designs by coordinate exchange.

The classical families in ``classical.py`` come in fixed sizes and assume every
factor is continuous and every corner of the box can be run. Real studies break
all three: 14 bioreactors, not 16; a media factor with three suppliers; a feed
rate that cannot be high when the glucose bolus is. This is the generator for
those cases — the same idea as JMP's Custom Design.

**Algorithm.** Meyer & Nachtsheim's coordinate exchange (Technometrics, 1995):
start from random feasible runs; for each run and each factor in turn, try
every candidate setting of that one coordinate and keep the one that most
improves the criterion; repeat until a whole pass changes nothing; keep the
best of several random starts. Each trial is scored with rank-one updates of
``(X'X)^-1`` rather than a fresh inverse.

**Criteria.**

- ``D``: maximise det(X'X) — precise *coefficients*. The default for screening
  and interaction models, where the question is "which factors matter?".
- ``I``: minimise the average prediction variance over the region — precise
  *predictions*. The default for response-surface models, where the question is
  "what happens at this setting?". The region is the one
  :func:`~doe_advisor.designs.properties.region_samples` defines, so the search
  minimises exactly the I-value the options table reports.

**Candidate settings.** A continuous factor is searched over a grid: two levels
(-1, +1) for main-effect and interaction models, three (-1, 0, +1) for a
quadratic, where the optimum is known to lie on that grid for a box region.
With linear constraints the corners of the feasible region are generally off
the grid, so a finer one is used. A categorical factor is searched over its
levels.

Like the rest of ``designs/``, this module holds no opinions about which design
to show; ``candidates.py`` decides that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .model import Levels, Term, expansion, model_matrix, n_columns
from .properties import fitted_model, region_samples
from .spec import Design, DesignSpec, ModelOrder

# Random starts per design. Coordinate exchange finds a local optimum; the best
# of several starts is reliably at or very near the global one for the design
# sizes this tool handles. Large searches (many runs x many factors) get fewer,
# down to MIN_STARTS, so the web app stays responsive; see :func:`n_starts_for`.
N_STARTS = 12
MIN_STARTS = 4
STARTS_WORK = 4800  # runs x factors at which the count starts to fall
# Passes over the whole design before giving up on convergence.
MAX_PASSES = 60
# Ridge added to X'X so a singular random start can still be improved. It is
# tiny relative to X'X, so it steers the search but does not change the answer.
RIDGE = 1e-8
# A trial must beat the current setting by this relative margin to be taken,
# so floating-point noise cannot make the search cycle.
IMPROVEMENT = 1e-9
# Continuous grid used when linear constraints are present.
FINE_GRID = tuple(np.round(np.linspace(-1.0, 1.0, 9), 10))
# Attempts to draw a random feasible run before declaring the region empty.
MAX_DRAWS = 5000

Feasible = Callable[[np.ndarray], np.ndarray]


def default_grid(order: ModelOrder, constrained: bool = False) -> tuple[float, ...]:
    """Candidate coded settings for a continuous factor."""
    if constrained:
        return FINE_GRID
    return (-1.0, 0.0, 1.0) if order is ModelOrder.QUADRATIC else (-1.0, 1.0)


def _candidates(levels: Sequence[int], grid: Sequence[float]) -> list[np.ndarray]:
    return [np.arange(k, dtype=float) if k else np.asarray(grid, dtype=float) for k in levels]


def _random_run(
    rng: np.random.Generator,
    candidates: list[np.ndarray],
    feasible: Feasible | None,
    template: np.ndarray,
    free: np.ndarray,
) -> np.ndarray:
    """A random run, drawing only the free coordinates, that satisfies the constraints."""
    for _ in range(MAX_DRAWS):
        run = template.copy()
        for j in np.flatnonzero(free):
            run[j] = rng.choice(candidates[j])
        if feasible is None or feasible(run[None, :])[0]:
            return run
    raise ValueError("no feasible run found: the constraints may leave nothing to run")


def _criterion_value(x: np.ndarray, criterion: str, moments: np.ndarray | None) -> float:
    """The true criterion (no ridge), oriented so larger is better."""
    xtx = x.T @ x
    # A singular start is expected and scored -inf, not warned about (and see
    # properties.d_efficiency for the spurious warnings some numpy builds emit).
    with np.errstate(all="ignore"):
        sign, logdet = np.linalg.slogdet(xtx)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    if criterion == "D":
        return float(logdet)
    inv = np.linalg.inv(xtx)
    return -float(np.trace(inv @ moments))


def coordinate_exchange(
    n_runs: int,
    levels: Sequence[int],
    terms: list[Term],
    *,
    criterion: str = "D",
    grid: Sequence[float] = (-1.0, 1.0),
    feasible: Feasible | None = None,
    region: np.ndarray | None = None,
    fixed_runs: np.ndarray | None = None,
    partial_runs: np.ndarray | None = None,
    n_starts: int = N_STARTS,
    seed: int = 0,
) -> np.ndarray:
    """An optimal run matrix (n_runs, k), coded, with any fixed runs first.

    ``fixed_runs`` are included as given and never changed — runs already done,
    for augmenting a study. ``partial_runs`` are runs whose continuous
    coordinates are fixed (centre points, say) but whose categorical ones are
    free to be chosen; they count towards ``n_runs``. ``region`` is the sample
    of the region the I-criterion averages over (required for ``criterion="I"``).

    Raises ``ValueError`` if the model cannot be estimated from ``n_runs`` runs
    or the constraints leave no run to make.
    """
    if criterion not in ("D", "I"):
        raise ValueError(f"criterion must be 'D' or 'I', got {criterion!r}")
    levels = [int(k) for k in levels]
    k = len(levels)
    lv: Levels = levels if any(levels) else None
    p = n_columns(terms, lv)
    if n_runs < p:
        raise ValueError(f"{n_runs} runs cannot estimate a model with {p} coefficients")
    if criterion == "I" and region is None:
        raise ValueError("the I-criterion needs a region sample to average over")

    fixed = np.zeros((0, k)) if fixed_runs is None else np.atleast_2d(np.asarray(fixed_runs, dtype=float))
    partial = np.zeros((0, k)) if partial_runs is None else np.atleast_2d(np.asarray(partial_runs, dtype=float))
    n_new = n_runs - fixed.shape[0]
    if n_new < partial.shape[0]:
        raise ValueError("more fixed and partly fixed runs than the run budget")

    candidates = _candidates(levels, grid)
    categorical = np.array([bool(kk) for kk in levels])
    moments = None
    if criterion == "I":
        f_region = model_matrix(region, terms, levels=lv)
        moments = f_region.T @ f_region / f_region.shape[0]

    rng = np.random.default_rng(seed)
    best_runs, best_value = None, -np.inf
    for _ in range(n_starts):
        # Rows: fixed (frozen), partial (categorical free), new (all free).
        free = np.zeros((n_runs, k), dtype=bool)
        free[fixed.shape[0]:fixed.shape[0] + partial.shape[0]] = categorical
        free[fixed.shape[0] + partial.shape[0]:] = True
        runs = np.vstack([fixed, partial, np.zeros((n_new - partial.shape[0], k))])
        for i in range(fixed.shape[0], n_runs):
            runs[i] = _random_run(rng, candidates, feasible, runs[i], free[i])
        runs = _exchange(runs, free, terms, lv, candidates, criterion, moments, feasible)
        value = _criterion_value(model_matrix(runs, terms, levels=lv), criterion, moments)
        if value > best_value:
            best_runs, best_value = runs, value

    if best_runs is None or not np.isfinite(best_value):
        raise ValueError(f"no estimable design with {n_runs} runs was found for this model")
    return best_runs


def _exchange(
    runs: np.ndarray,
    free: np.ndarray,
    terms: list[Term],
    levels: Levels,
    candidates: list[np.ndarray],
    criterion: str,
    moments: np.ndarray | None,
    feasible: Feasible | None,
) -> np.ndarray:
    """Coordinate-exchange passes from one start until nothing improves."""
    runs = runs.copy()
    n, k = runs.shape
    ex = expansion(tuple(terms), tuple(int(v) for v in levels) if levels is not None else (0,) * k)
    base = ex.base(runs)
    x = ex.rows(base)
    # Each factor's base columns at each of its candidate settings, computed once.
    cand_base = [ex.factor_base(j, candidates[j]) for j in range(k)]

    def inverse() -> np.ndarray:
        xtx = x.T @ x
        ridge = RIDGE * max(float(np.trace(xtx)) / xtx.shape[0], 1.0)
        return np.linalg.inv(xtx + ridge * np.eye(xtx.shape[0]))

    a = inverse()
    for _ in range(MAX_PASSES):
        changed = False
        for i in range(n):
            for j in np.flatnonzero(free[i]):
                keep = np.abs(candidates[j] - runs[i, j]) > 1e-12
                if feasible is not None:
                    trials = np.repeat(runs[i][None, :], len(candidates[j]), axis=0)
                    trials[:, j] = candidates[j]
                    keep &= feasible(trials)
                if not keep.any():
                    continue
                x_old = x[i]
                ax = a @ x_old
                leverage = float(x_old @ ax)
                if leverage >= 1.0 - 1e-12:
                    continue  # removing this run would make X'X singular; leave it
                # (X'X - x x')^-1, by Sherman-Morrison.
                a1 = a + np.outer(ax, ax) / (1.0 - leverage)
                trial_base = np.repeat(base[i][None, :], int(keep.sum()), axis=0)
                trial_base[:, ex.slices[j]] = cand_base[j][keep]
                y = ex.rows(trial_base)
                q = y @ a1
                denom = 1.0 + np.einsum("ij,ij->i", q, y)
                q_old = a1 @ x_old
                denom_old = 1.0 + float(x_old @ q_old)
                if criterion == "D":
                    gain, current = denom, denom_old
                else:
                    gain = np.einsum("ij,jk,ik->i", q, moments, q) / denom
                    current = float(q_old @ moments @ q_old) / denom_old
                best = int(np.argmax(gain))
                if gain[best] > current * (1.0 + IMPROVEMENT) + 1e-15:
                    runs[i, j] = candidates[j][keep][best]
                    base[i] = trial_base[best]
                    x[i] = y[best]
                    a = a1 - np.outer(q[best], q[best]) / denom[best]
                    changed = True
        a = inverse()  # wash out the drift of many rank-one updates
        if not changed:
            break
    return runs


# --------------------------------------------------------------------------
# A whole design, from a spec
# --------------------------------------------------------------------------


def n_starts_for(n_runs: int, n_factors: int) -> int:
    """Random starts for a search of this size: N_STARTS for small ones, fewer for large."""
    return int(np.clip(round(N_STARTS * STARTS_WORK / max(n_runs * n_factors, 1) / 10), MIN_STARTS, N_STARTS))


def default_criterion(order: ModelOrder) -> str:
    """D for screening and interaction models, I for response surfaces (JMP's defaults)."""
    return "I" if order is ModelOrder.QUADRATIC else "D"


def custom_design(
    spec: DesignSpec,
    n_runs: int,
    *,
    criterion: str | None = None,
    n_center: int = 0,
    fixed_runs: np.ndarray | None = None,
    seed: int = 0,
    n_starts: int | None = None,
) -> Design:
    """An optimal design of exactly ``n_runs`` runs for ``spec``'s model and constraints.

    ``n_center`` runs are placed at the centre of every continuous factor (a
    pure-error and curvature check), with their categorical settings chosen by
    the search. They are skipped when the spec has no continuous factor or the
    centre breaks a constraint. ``fixed_runs`` (coded) are runs already made;
    they come first and count towards ``n_runs``.
    """
    levels = spec.levels
    terms = list(fitted_model(spec).terms)
    criterion = criterion or default_criterion(spec.model_order)
    n_starts = n_starts or n_starts_for(n_runs, spec.n_factors)
    region = region_samples(spec, seed=seed)[0] if criterion == "I" else None
    feasible = spec.feasible if spec.is_constrained else None

    has_continuous = any(k == 0 for k in levels)
    partial = None
    if n_center > 0 and has_continuous:
        centre = np.zeros((1, spec.n_factors))
        cats = [j for j, k in enumerate(levels) if k]
        if not cats and (feasible is None or feasible(centre)[0]):
            partial = np.repeat(centre, n_center, axis=0)
        elif cats:
            # Spread the centre runs over categorical levels to start with; the
            # search may move their categorical settings.
            rows = np.repeat(centre, n_center, axis=0)
            for r in range(n_center):
                for j in cats:
                    rows[r, j] = r % levels[j]
            if feasible is None or feasible(rows).any():
                partial = rows if feasible is None else rows[feasible(rows)]
    n_center_used = 0 if partial is None else partial.shape[0]

    matrix = coordinate_exchange(
        n_runs,
        levels,
        terms,
        criterion=criterion,
        grid=default_grid(spec.model_order, constrained=bool(spec.constraints)),
        feasible=feasible,
        region=region,
        fixed_runs=fixed_runs,
        partial_runs=partial,
        n_starts=n_starts,
        seed=seed,
    )
    n_fixed = 0 if fixed_runs is None else int(np.atleast_2d(fixed_runs).shape[0])
    label = f"{criterion}-optimal custom design"
    name = f"{label} ({n_runs} runs" + (f", {n_center_used} centre" if n_center_used else "") + ")"
    return Design(
        name=name,
        family="custom",
        matrix=matrix,
        factor_names=spec.factor_names,
        detail={
            "criterion": criterion,
            "algorithm": "coordinate exchange",
            "n_starts": n_starts,
            "grid": list(default_grid(spec.model_order, constrained=bool(spec.constraints))),
            "n_center_requested": n_center,
            "n_fixed_runs": n_fixed,
            "constrained": spec.is_constrained,
        },
        levels=levels if any(levels) else None,
    )
