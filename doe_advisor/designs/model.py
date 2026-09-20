"""Model terms and model matrices.

A *term* is a tuple of factor indices, with repetition meaning a power:

    ()      intercept
    (0,)    main effect of factor 0
    (0, 1)  interaction of factors 0 and 1
    (0, 0)  quadratic in factor 0

Everything that needs an ``X`` matrix — power, prediction variance, aliasing,
optimality criteria — builds it here, so there is exactly one definition of
what "the model" means.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .spec import ModelOrder

Term = tuple[int, ...]


def model_terms(n_factors: int, order: ModelOrder) -> list[Term]:
    """Terms the scientist wants to be able to estimate, intercept first."""
    terms: list[Term] = [()]
    terms += [(i,) for i in range(n_factors)]
    if order in (ModelOrder.INTERACTION, ModelOrder.QUADRATIC):
        terms += [tuple(c) for c in combinations(range(n_factors), 2)]
    if order is ModelOrder.QUADRATIC:
        terms += [(i, i) for i in range(n_factors)]
    return terms


def potential_terms(n_factors: int, order: ModelOrder) -> list[Term]:
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
        extra += [(i, i) for i in range(n_factors)]
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


def model_matrix(matrix: np.ndarray, terms: list[Term]) -> np.ndarray:
    """Build X (n_runs, n_terms) from a coded run matrix."""
    matrix = np.asarray(matrix, dtype=float)
    n_runs = matrix.shape[0]
    x = np.empty((n_runs, len(terms)), dtype=float)
    for j, term in enumerate(terms):
        col = np.ones(n_runs, dtype=float)
        for i in term:
            col = col * matrix[:, i]
        x[:, j] = col
    return x


def is_estimable(matrix: np.ndarray, terms: list[Term], tol: float = 1e-8) -> bool:
    """Can this design fit this model at all?

    False means the model is *singular* — there are not enough independent runs,
    so some coefficients cannot be separated even in principle. A design that
    fails this is not "weak", it is unusable for the stated model.
    """
    if matrix.shape[0] < len(terms):
        return False
    x = model_matrix(matrix, terms)
    return bool(np.linalg.matrix_rank(x, tol=tol) == len(terms))


def xtx_inv(matrix: np.ndarray, terms: list[Term]) -> np.ndarray:
    """(X'X)^-1, the covariance structure of the coefficient estimates.

    Raises ``np.linalg.LinAlgError`` when the model is not estimable; callers
    are expected to gate on :func:`is_estimable` first.
    """
    x = model_matrix(matrix, terms)
    return np.linalg.inv(x.T @ x)


def residual_df(matrix: np.ndarray, terms: list[Term]) -> int:
    """Degrees of freedom left over to estimate noise with.

    Zero means the model is saturated: it will fit the data perfectly and tell
    you nothing about uncertainty. This is the trap behind "just run the
    minimum number of runs".
    """
    return int(matrix.shape[0] - len(terms))
