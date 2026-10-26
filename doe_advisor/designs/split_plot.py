"""Split-plot designs: hard-to-change factors set once per whole plot.

Some factors cannot be reset between every run: the temperature of an
incubator or a bank of microbioreactors, a media batch, a plate-level setting.
Completely randomising them is impractical, and running them in groups while
analysing as if randomised is wrong — it understates the noise on exactly
those factors. A split-plot design makes the grouping explicit: the
hard-to-change (whole-plot) factors are set once per *whole plot*, and the
easy-to-change (subplot) factors vary among the runs inside it.

**Model.** The runs of a whole plot share a random effect, so
``Var(y) = sigma^2 (I + eta Z Z')``, with ``eta`` the whole-plot-to-run
variance ratio. Designs are scored on the generalised least-squares
information ``X' V^-1 X`` (``model.information``). Whole-plot terms are tested
on whole-plot degrees of freedom, everything else on subplot ones
(``model.stratum_df``).

**Algorithm.** Jones & Goos's coordinate exchange for split-plot designs
(Journal of Quality Technology, 2007): whole plots of fixed sizes; a pass first
tries every setting of each whole-plot factor for each whole plot (changing all
its runs together), then every setting of each subplot factor for each run.
Subplot moves change ``X' V^-1 X`` by a rank-4 matrix (the run's row, and its
whole plot's column sums), so they are scored and applied with the matrix
determinant lemma and Woodbury's identity; whole-plot moves are rarer and are
recomputed directly.

Like the rest of ``designs/`` this holds no opinions; ``candidates.py`` chooses
run counts and whole-plot counts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .blocking import block_sizes
from .model import Levels, Term, expansion, information, model_matrix, n_columns
from .optimal import (
    IMPROVEMENT,
    MAX_DRAWS,
    MAX_PASSES,
    RIDGE,
    Feasible,
    _candidates,
    _random_run,
    default_criterion,
    default_grid,
    n_starts_for,
)
from .properties import fitted_model, region_samples
from .spec import Design, DesignSpec


def _criterion(m: np.ndarray, criterion: str, moments: np.ndarray | None) -> float:
    """Larger is better; -inf for a singular information matrix."""
    with np.errstate(all="ignore"):  # see properties.d_efficiency on spurious slogdet warnings
        sign, logdet = np.linalg.slogdet(m)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    if criterion == "D":
        return float(logdet)
    return -float(np.trace(np.linalg.solve(m, moments)))


def split_plot_exchange(
    plot_sizes: Sequence[int],
    levels: Sequence[int],
    terms: list[Term],
    whole_plot_factors: Sequence[int],
    *,
    eta: float = 1.0,
    criterion: str = "D",
    grid: Sequence[float] = (-1.0, 1.0),
    feasible: Feasible | None = None,
    region: np.ndarray | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(coded run matrix, whole-plot label per run), runs grouped by whole plot.

    Raises ``ValueError`` when the model cannot be estimated from this many
    runs, or the constraints leave nothing to run.
    """
    if criterion not in ("D", "I"):
        raise ValueError(f"criterion must be 'D' or 'I', got {criterion!r}")
    if criterion == "I" and region is None:
        raise ValueError("the I-criterion needs a region sample to average over")
    levels = [int(k) for k in levels]
    k = len(levels)
    lv: Levels = levels if any(levels) else None
    sizes = np.asarray(plot_sizes, dtype=int)
    if sizes.min() < 1:
        raise ValueError("every whole plot needs at least one run")
    n = int(sizes.sum())
    if n < n_columns(terms, lv):
        raise ValueError(f"{n} runs cannot estimate a model with {n_columns(terms, lv)} coefficients")
    labels = np.repeat(np.arange(sizes.size), sizes)
    is_wp = np.zeros(k, dtype=bool)
    is_wp[list(whole_plot_factors)] = True
    if is_wp.all() or not is_wp.any():
        raise ValueError("a split-plot design needs both hard-to-change and easy-to-change factors")

    candidates = _candidates(levels, grid)
    moments = None
    if criterion == "I":
        f_region = model_matrix(region, terms, levels=lv)
        moments = f_region.T @ f_region / f_region.shape[0]

    rng = np.random.default_rng(seed)
    best, best_value = None, -np.inf
    for _ in range(n_starts):
        runs = _random_start(rng, sizes, labels, candidates, is_wp, feasible, k)
        runs = _exchange(runs, labels, sizes, is_wp, terms, lv, candidates, criterion, moments, feasible, eta)
        value = _criterion(information(model_matrix(runs, terms, levels=lv), labels, eta), criterion, moments)
        if value > best_value:
            best, best_value = runs, value
    if best is None or not np.isfinite(best_value):
        raise ValueError(f"no estimable split-plot design with {n} runs in {sizes.size} whole plots was found")
    return best, labels


def _random_start(
    rng: np.random.Generator,
    sizes: np.ndarray,
    labels: np.ndarray,
    candidates: list[np.ndarray],
    is_wp: np.ndarray,
    feasible: Feasible | None,
    k: int,
) -> np.ndarray:
    """Random feasible runs: whole-plot settings drawn per plot, subplot settings per run."""
    runs = np.zeros((labels.size, k))
    for g in range(sizes.size):
        rows = np.flatnonzero(labels == g)
        for _ in range(MAX_DRAWS):
            template = np.zeros(k)
            for j in np.flatnonzero(is_wp):
                template[j] = rng.choice(candidates[j])
            try:
                runs[rows] = [_random_run(rng, candidates, feasible, template, ~is_wp) for _ in rows]
                break
            except ValueError:
                continue  # these whole-plot settings leave no feasible subplot run; draw again
        else:
            raise ValueError("no feasible run found: the constraints may leave nothing to run")
    return runs


def _exchange(
    runs: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
    is_wp: np.ndarray,
    terms: list[Term],
    levels: Levels,
    candidates: list[np.ndarray],
    criterion: str,
    moments: np.ndarray | None,
    feasible: Feasible | None,
    eta: float,
) -> np.ndarray:
    runs = runs.copy()
    n, k = runs.shape
    ex = expansion(tuple(terms), tuple(levels) if levels is not None else (0,) * k)
    base = ex.base(runs)
    x = ex.rows(base)
    cand_base = [ex.factor_base(j, candidates[j]) for j in range(k)]
    c = eta / (1.0 + sizes * eta)
    plot_rows = [np.flatnonzero(labels == g) for g in range(sizes.size)]

    def plot_info(rows_x: np.ndarray, g: int) -> np.ndarray:
        s = rows_x.sum(axis=0)
        return rows_x.T @ rows_x - c[g] * np.outer(s, s)

    def ridged(m: np.ndarray) -> np.ndarray:
        return m + RIDGE * max(float(np.trace(m)) / m.shape[0], 1.0) * np.eye(m.shape[0])

    for _ in range(MAX_PASSES):
        changed = False
        m = information(x, labels, eta)

        # Whole-plot moves: one hard-to-change setting for a whole plot at a time.
        for g, rows in enumerate(plot_rows):
            current = _criterion(ridged(m), criterion, moments)
            old = plot_info(x[rows], g)
            for j in np.flatnonzero(is_wp):
                for value, vb in zip(candidates[j], cand_base[j], strict=True):
                    if abs(value - runs[rows[0], j]) <= 1e-12:
                        continue
                    trial = runs[rows].copy()
                    trial[:, j] = value
                    if feasible is not None and not feasible(trial).all():
                        continue
                    tb = base[rows].copy()
                    tb[:, ex.slices[j]] = vb
                    ty = ex.rows(tb)
                    m_new = m - old + plot_info(ty, g)
                    score = _criterion(ridged(m_new), criterion, moments)
                    if score > current + IMPROVEMENT * max(1.0, abs(current)):
                        runs[rows], base[rows], x[rows] = trial, tb, ty
                        m, old, current, changed = m_new, plot_info(ty, g), score, True

        # Subplot moves: one easy-to-change setting for one run, by rank-4 updates.
        a = np.linalg.inv(ridged(m))
        sums = np.array([x[rows].sum(axis=0) for rows in plot_rows])
        for i in range(n):
            g = labels[i]
            d = np.array([1.0, -1.0, -c[g], c[g]])
            for j in np.flatnonzero(~is_wp):
                keep = np.abs(candidates[j] - runs[i, j]) > 1e-12
                if feasible is not None:
                    trials = np.repeat(runs[i][None, :], len(candidates[j]), axis=0)
                    trials[:, j] = candidates[j]
                    keep &= feasible(trials)
                if not keep.any():
                    continue
                tb = np.repeat(base[i][None, :], int(keep.sum()), axis=0)
                tb[:, ex.slices[j]] = cand_base[j][keep]
                y = ex.rows(tb)
                s_new = sums[g] - x[i] + y
                # X'V^-1X changes by U D U' with U = [y, x, s_new, s] and
                # D = diag(1, -1, -c, c): the run's own row in, old row out, and
                # the whole plot's column sums replaced in the -c s s' term.
                u = np.stack(
                    [y, np.broadcast_to(x[i], y.shape), s_new, np.broadcast_to(sums[g], y.shape)], axis=2
                )
                au = np.einsum("pq,cqr->cpr", a, u)
                gram = np.einsum("cpr,cps->crs", u, au)
                kmat = np.eye(4) + d[None, :, None] * gram
                if criterion == "D":
                    with np.errstate(all="ignore"):
                        gain = np.linalg.det(kmat)  # det ratio new/old
                    threshold = 1.0 + IMPROVEMENT
                else:
                    h = np.einsum("cpr,pq,cqs->crs", au, moments, au)
                    gain = np.trace(np.linalg.solve(kmat, d[None, :, None] * h), axis1=1, axis2=2)
                    threshold = IMPROVEMENT * max(1.0, abs(float(np.trace(a @ moments))))
                best = int(np.argmax(gain))
                if not np.isfinite(gain[best]) or gain[best] <= threshold:
                    continue
                # Woodbury: (M + U D U')^-1 = A - A U (I + D U'AU)^-1 D U'A.
                au_b = au[best]
                a = a - au_b @ np.linalg.solve(kmat[best], d[:, None] * au_b.T)
                runs[i, j] = candidates[j][keep][best]
                base[i], x[i] = tb[best], y[best]
                sums[g] = s_new[best]
                changed = True
        if not changed:
            break
    return runs


def plot_sizes_for(n_runs: int, n_whole_plots: int) -> list[int]:
    """Whole-plot sizes as equal as possible (earlier plots take the remainder)."""
    if not 2 <= n_whole_plots <= n_runs:
        raise ValueError(f"{n_runs} runs cannot be split into {n_whole_plots} whole plots")
    return block_sizes(n_runs, n_whole_plots)


def split_plot_design(
    spec: DesignSpec,
    n_runs: int,
    n_whole_plots: int,
    *,
    criterion: str | None = None,
    seed: int = 0,
    n_starts: int | None = None,
) -> Design:
    """An optimal split-plot design for ``spec``: ``n_runs`` runs in ``n_whole_plots`` whole plots."""
    if not spec.is_split_plot:
        raise ValueError("no factor is marked hard to change, so this is not a split-plot problem")
    levels = spec.levels
    terms = list(fitted_model(spec).terms)
    criterion = criterion or default_criterion(spec.model_order)
    n_starts = n_starts or max(4, n_starts_for(n_runs, spec.n_factors) * 2 // 3)
    grid = default_grid(spec.model_order, constrained=bool(spec.constraints))
    matrix, labels = split_plot_exchange(
        plot_sizes_for(n_runs, n_whole_plots),
        levels,
        terms,
        spec.whole_plot_factors,
        eta=spec.wp_variance_ratio,
        criterion=criterion,
        grid=grid,
        feasible=spec.feasible if spec.is_constrained else None,
        region=region_samples(spec, seed=seed)[0] if criterion == "I" else None,
        n_starts=n_starts,
        seed=seed,
    )
    return Design(
        name=f"{criterion}-optimal split-plot design ({n_runs} runs, {n_whole_plots} whole plots)",
        family="split_plot",
        matrix=matrix,
        factor_names=spec.factor_names,
        detail={
            "criterion": criterion,
            "algorithm": "split-plot coordinate exchange",
            "n_starts": n_starts,
            "grid": list(grid),
            "n_whole_plots": n_whole_plots,
            "whole_plot_factors": [spec.factor_names[j] for j in spec.whole_plot_factors],
            "wp_variance_ratio": spec.wp_variance_ratio,
            "constrained": spec.is_constrained,
        },
        levels=levels if any(levels) else None,
        whole_plots=labels,
    )
