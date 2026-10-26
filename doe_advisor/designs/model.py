"""Model terms and model matrices.

A *term* is a tuple of factor indices, with repetition meaning a power:

    ()      intercept
    (0,)    main effect of factor 0
    (0, 1)  interaction of factors 0 and 1
    (0, 0)  quadratic in factor 0

Everything that needs an ``X`` matrix — power, prediction variance, aliasing,
optimality criteria — builds it here, so there is exactly one definition of
what "the model" means.

**Categorical factors.** ``levels`` gives, per factor, its number of levels
(0 for a continuous factor). A categorical factor's column in the run matrix
holds a level index, and it enters ``X`` as k-1 *effect-coded* columns:
column j (for level j+1) is +1 at level j+1, -1 at level 0 (the reference) and
0 elsewhere. A term then spans the product of its factors' columns, so a
categorical main effect has k-1 columns and a categorical x continuous
interaction k-1 more. A two-level categorical is coded -1 / +1, exactly like a
continuous factor at its low / high, so the two give identical numbers.

A categorical factor has no curvature: ``(i, i)`` is never a term for it.

Terms keep their meaning (and their labels) at the level of factors; use
:func:`term_slices` to find which columns of ``X`` belong to which term.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from .spec import ModelOrder

Term = tuple[int, ...]
Levels = Sequence[int] | None


def _level(levels: Levels, i: int) -> int:
    return 0 if levels is None else int(levels[i])


def _quadratic_ok(levels: Levels, i: int) -> bool:
    return _level(levels, i) == 0


def model_terms(n_factors: int, order: ModelOrder, levels: Levels = None) -> list[Term]:
    """Terms the scientist wants to be able to estimate, intercept first."""
    terms: list[Term] = [()]
    terms += [(i,) for i in range(n_factors)]
    if order in (ModelOrder.INTERACTION, ModelOrder.QUADRATIC):
        terms += [tuple(c) for c in combinations(range(n_factors), 2)]
    if order is ModelOrder.QUADRATIC:
        terms += [(i, i) for i in range(n_factors) if _quadratic_ok(levels, i)]
    return terms


def potential_terms(n_factors: int, order: ModelOrder, levels: Levels = None) -> list[Term]:
    """Terms *not* in the model that could still be active, and so could bias it.

    This is the set the alias matrix is computed against. The convention matches
    JMP's: look one order up from what you are fitting.

    - fitting main effects only -> two-factor interactions could bias you
    - fitting 2FI (or quadratic) -> three-factor interactions could bias you,
      and for a 2FI model the pure quadratics could too
    """
    if order is ModelOrder.MAIN:
        return [tuple(c) for c in combinations(range(n_factors), 2)]

    extra: list[Term] = [tuple(c) for c in combinations(range(n_factors), 3)]
    if order is ModelOrder.INTERACTION:
        extra += [(i, i) for i in range(n_factors) if _quadratic_ok(levels, i)]
    return extra


def term_label(term: Term, names: list[str]) -> str:
    """Human-readable name, e.g. ``pH x temperature`` or ``pH^2``."""
    if not term:
        return "intercept"
    counts: dict[int, int] = {}
    for i in term:
        counts[i] = counts.get(i, 0) + 1
    parts = [names[i] if p == 1 else f"{names[i]}^{p}" for i, p in sorted(counts.items())]
    return " x ".join(parts)


def term_width(term: Term, levels: Levels = None) -> int:
    """How many columns of ``X`` a term occupies."""
    width = 1
    for i in set(term):
        k = _level(levels, i)
        if k:
            width *= k - 1
    return width


def term_slices(terms: list[Term], levels: Levels = None) -> list[slice]:
    """The columns of ``X`` that belong to each term, in order."""
    out, start = [], 0
    for term in terms:
        width = term_width(term, levels)
        out.append(slice(start, start + width))
        start += width
    return out


def n_columns(terms: list[Term], levels: Levels = None) -> int:
    """Columns of ``X`` for these terms (blocks not included)."""
    return sum(term_width(t, levels) for t in terms)


def effect_codes(k: int) -> np.ndarray:
    """(k, k-1) effect-coding matrix: row L is the coded row for level L."""
    codes = np.zeros((k, k - 1), dtype=float)
    codes[0, :] = -1.0
    codes[np.arange(1, k), np.arange(k - 1)] = 1.0
    return codes


def categorical_columns(index_column: np.ndarray, k: int) -> np.ndarray:
    """Effect-coded columns (n, k-1) for a column of level indices."""
    idx = np.rint(np.asarray(index_column, dtype=float)).astype(int)
    if idx.size and (idx.min() < 0 or idx.max() >= k):
        raise ValueError(f"level index out of range for a {k}-level factor")
    return effect_codes(k)[idx]


def block_columns(blocks: np.ndarray | None) -> np.ndarray:
    """Effect-coded block columns, (n_runs, b-1); empty when unblocked.

    Effect (sum-to-zero) coding means a prediction with every block column at
    zero is the average over blocks — which is what a prediction for "the
    process", rather than for one particular day, should be.
    """
    if blocks is None:
        return np.zeros((0, 0))
    blocks = np.asarray(blocks, dtype=int)
    b = int(blocks.max()) + 1
    cols = np.zeros((blocks.size, b - 1), dtype=float)
    for j in range(b - 1):
        cols[blocks == j, j] = 1.0
        cols[blocks == b - 1, j] = -1.0
    return cols


class Expansion:
    """How to build X by one vectorised product, for one model and one set of factors.

    The run matrix is first widened into *base columns* — each continuous
    factor as is, each categorical one as its k-1 effect codes — plus a final
    column of ones. Every column of X is a product of base columns; ``index``
    holds their positions, padded with the ones column to a common length, so
    ``X = prod(base[:, index], axis=2)``. Columns come out in term order and,
    within a term, in the order of :func:`itertools.product` over its factors'
    columns.

    The optimal-design search uses the pieces directly, so a trial change to
    one factor only rewrites that factor's base columns.
    """

    def __init__(self, terms: tuple[Term, ...], levels: tuple[int, ...]):
        self.levels = levels
        self.slices: list[slice] = []
        position = 0
        for k in levels:
            width = k - 1 if k else 1
            self.slices.append(slice(position, position + width))
            position += width
        self.n_base = position + 1  # + the ones column
        rows: list[list[int]] = []
        for term in terms:
            options = [range(self.slices[i].start, self.slices[i].stop) for i in term]
            rows += [list(combo) for combo in product(*options)] if term else [[]]
        depth = max((len(r) for r in rows), default=0) or 1
        self.index = np.full((len(rows), depth), position, dtype=int)
        for r, row in enumerate(rows):
            self.index[r, : len(row)] = row
        self._codes = {i: effect_codes(k) for i, k in enumerate(levels) if k}

    def base(self, matrix: np.ndarray) -> np.ndarray:
        """(n, n_base) base columns of a coded run matrix."""
        out = np.ones((matrix.shape[0], self.n_base))
        for i, k in enumerate(self.levels):
            out[:, self.slices[i]] = categorical_columns(matrix[:, i], k) if k else matrix[:, [i]]
        return out

    def factor_base(self, i: int, values: np.ndarray) -> np.ndarray:
        """(len(values), width) base columns of factor ``i`` at these settings."""
        values = np.asarray(values, dtype=float)
        if i in self._codes:
            return self._codes[i][np.rint(values).astype(int)]
        return values[:, None]

    def rows(self, base: np.ndarray) -> np.ndarray:
        """Model rows from base columns."""
        if not self.index.size:
            return np.zeros((base.shape[0], 0))
        return np.prod(base[:, self.index], axis=2)


@lru_cache(maxsize=256)
def expansion(terms: tuple[Term, ...], levels: tuple[int, ...]) -> Expansion:
    """The cached :class:`Expansion` for a model."""
    return Expansion(terms, levels)


def model_matrix(
    matrix: np.ndarray, terms: list[Term], blocks: np.ndarray | None = None, levels: Levels = None
) -> np.ndarray:
    """Build X (n_runs, n_columns [+ b-1]) from a coded run matrix.

    Block columns, when given, go *after* the model terms, so a term's columns
    are the same blocked or not. Without categorical factors there is one
    column per term, in term order.
    """
    matrix = np.asarray(matrix, dtype=float)
    lv = tuple(int(k) for k in levels) if levels is not None else (0,) * matrix.shape[1]
    ex = expansion(tuple(terms), lv)
    x = ex.rows(ex.base(matrix))
    if blocks is not None:
        x = np.hstack([x, block_columns(blocks)])
    return x


def n_block_params(blocks: np.ndarray | None) -> int:
    return 0 if blocks is None else int(np.asarray(blocks).max())


def n_params(terms: list[Term], blocks: np.ndarray | None = None, levels: Levels = None) -> int:
    """Coefficients the model estimates, block offsets included."""
    return n_columns(terms, levels) + n_block_params(blocks)


def is_estimable(
    matrix: np.ndarray, terms: list[Term], tol: float = 1e-8, blocks: np.ndarray | None = None, levels: Levels = None
) -> bool:
    """Can this design fit this model at all?

    False means the model is *singular* — there are not enough independent runs,
    so some coefficients cannot be separated even in principle. A design that
    fails this is not "weak", it is unusable for the stated model. With blocks,
    a term confounded with a block counts as not estimable.
    """
    p = n_params(terms, blocks, levels)
    if matrix.shape[0] < p:
        return False
    x = model_matrix(matrix, terms, blocks, levels)
    return bool(np.linalg.matrix_rank(x, tol=tol) == p)


def xtx_inv(
    matrix: np.ndarray, terms: list[Term], blocks: np.ndarray | None = None, levels: Levels = None
) -> np.ndarray:
    """(X'X)^-1, the covariance structure of the coefficient estimates.

    Raises ``np.linalg.LinAlgError`` when the model is not estimable; callers
    are expected to gate on :func:`is_estimable` first.
    """
    x = model_matrix(matrix, terms, blocks, levels)
    return np.linalg.inv(x.T @ x)


def residual_df(
    matrix: np.ndarray, terms: list[Term], blocks: np.ndarray | None = None, levels: Levels = None
) -> int:
    """Degrees of freedom left over to estimate noise with.

    Zero means the model is saturated: it will fit the data perfectly and tell
    you nothing about uncertainty. This is the trap behind "just run the
    minimum number of runs". Each extra block costs one, and each categorical
    factor costs one per level beyond the first.
    """
    return int(matrix.shape[0] - n_params(terms, blocks, levels))


# --------------------------------------------------------------------------
# Split-plot: correlated errors within whole plots
# --------------------------------------------------------------------------
#
# In a split-plot design the runs of one whole plot share a random whole-plot
# effect, so Var(y) = sigma^2 * V with V = I + eta * Z Z', where Z maps runs to
# whole plots and eta = sigma_wp^2 / sigma^2. Every precision-based number
# (D, I, prediction variance, power, aliasing) must then use the generalised
# least-squares information X' V^-1 X instead of X'X. V is block diagonal, and
# for a whole plot of m runs its inverse is I - c J with c = eta / (1 + m eta),
# so X' V^-1 X = X'X - sum over whole plots of c * s s', s the plot's column sums.


def _plot_weights(whole_plots: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """(label per run as 0..g-1, c per whole plot)."""
    _, labels, sizes = np.unique(np.asarray(whole_plots), return_inverse=True, return_counts=True)
    return labels.ravel(), eta / (1.0 + sizes * eta)


def _plot_sums(x: np.ndarray, labels: np.ndarray, n_plots: int) -> np.ndarray:
    sums = np.zeros((n_plots, x.shape[1]))
    np.add.at(sums, labels, x)
    return sums


def information(x: np.ndarray, whole_plots: np.ndarray | None = None, eta: float = 1.0) -> np.ndarray:
    """X' V^-1 X (in units of 1/sigma^2); plain X'X when there are no whole plots."""
    xtx = x.T @ x
    if whole_plots is None or eta == 0:
        return xtx
    labels, c = _plot_weights(whole_plots, eta)
    sums = _plot_sums(x, labels, c.size)
    return xtx - (sums * c[:, None]).T @ sums


def cross_information(
    x1: np.ndarray, x2: np.ndarray, whole_plots: np.ndarray | None = None, eta: float = 1.0
) -> np.ndarray:
    """X1' V^-1 X2, for the GLS alias matrix."""
    cross = x1.T @ x2
    if whole_plots is None or eta == 0:
        return cross
    labels, c = _plot_weights(whole_plots, eta)
    s1, s2 = _plot_sums(x1, labels, c.size), _plot_sums(x2, labels, c.size)
    return cross - (s1 * c[:, None]).T @ s2


def whole_plot_term_flags(
    x: np.ndarray, terms: list[Term], whole_plots: np.ndarray, levels: Levels = None
) -> list[bool]:
    """Per term: is it a whole-plot term (constant within every whole plot)?

    Read off the design itself rather than from which factors were marked hard
    to change, so it stays right for a design with runs lost.
    """
    labels, _ = _plot_weights(whole_plots, 0.0)
    flags = []
    for slc in term_slices(terms, levels):
        cols = x[:, slc]
        constant = True
        for g in range(int(labels.max()) + 1):
            block = cols[labels == g]
            if block.shape[0] > 1 and np.ptp(block, axis=0).max() > 1e-9:
                constant = False
                break
        flags.append(constant)
    return flags


def stratum_df(
    x: np.ndarray, terms: list[Term], whole_plots: np.ndarray, levels: Levels = None
) -> tuple[int, int, list[bool]]:
    """(whole-plot df, subplot df, whole-plot flag per term) — the containment rule.

    Whole-plot terms (the intercept, hard-to-change factors and their
    interactions) are estimated from whole-plot totals, so they are tested on
    g - p_wp degrees of freedom; every other term on n - g - p_sp.
    """
    flags = whole_plot_term_flags(x, terms, whole_plots, levels)
    widths = [term_width(t, levels) for t in terms]
    g = int(np.unique(whole_plots).size)
    p_wp = sum(w for w, f in zip(widths, flags, strict=True) if f)
    p_sp = sum(w for w, f in zip(widths, flags, strict=True) if not f)
    return g - p_wp, x.shape[0] - g - p_sp, flags
