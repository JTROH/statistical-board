"""Split a design's runs into blocks (days, bioreactor batches, virus lots).

A block shares something that may shift the response as a whole: the same
seed train, the same day's media, the same operator. Fitting one offset per
block removes that shift — but only if no model term is entangled with the
block split. Which runs go in which block is therefore a design decision.

The assignment here is a D-optimal exchange: start from a balanced random
split, swap pairs of runs between blocks while that improves det(X'X) of the
model *with* its block columns, and keep the best of several starts. This
recovers the textbook answers where they exist — a 2^3 factorial in two blocks
puts the blocks on the three-factor interaction, which the tests check — and
gives a sensible answer for designs with no textbook blocking. Two tie-breaks,
in order: keep the blocks off the low-order terms the model leaves out (so a
2^4 in two blocks uses ABCD, not a three-factor interaction — the minimum-
aberration choice), then spread centre points across blocks, so each block
keeps a check on curvature and drift.

No opinions about *whether* to block live here; that is the scientist's call.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .model import Term, block_columns, model_matrix

# Random starts for the exchange. More is slower and rarely better for the
# design sizes this tool handles (tens of runs).
N_STARTS = 6
# Textbook starts carried into the swap search, after ranking them all.
N_REFINED = 3


def block_sizes(n_runs: int, n_blocks: int) -> list[int]:
    """As equal as possible; earlier blocks take the remainder."""
    base, extra = divmod(n_runs, n_blocks)
    return [base + 1] * extra + [base] * (n_blocks - extra)


def _criterion(
    x_model: np.ndarray, x_potential: np.ndarray, weights: np.ndarray, labels: np.ndarray, centre: np.ndarray
) -> tuple[float, float, int]:
    x = np.hstack([x_model, block_columns(labels)])
    xtx = x.T @ x
    sign, logdet = np.linalg.slogdet(xtx)
    if sign <= 0:
        return (-np.inf, -np.inf, 0)
    # How much of each left-out term the block columns soak up, weighted so a
    # lower-order term costs more than a higher-order one.
    if x_potential.shape[1]:
        a = np.linalg.solve(xtx, x.T @ x_potential)[x_model.shape[1]:]
        load = float((a**2).sum(axis=0) @ weights)
    else:
        load = 0.0
    n_blocks = int(labels.max()) + 1
    counts = np.bincount(labels[centre], minlength=n_blocks) if centre.any() else np.zeros(n_blocks, int)
    return (round(float(logdet), 9), -round(load, 9), -int(counts.max() - counts.min()))


def _structured_starts(matrix: np.ndarray, potential: list[Term], n_blocks: int, sizes: list[int]) -> list[np.ndarray]:
    """Textbook starting splits: blocks defined by the signs of high-order
    interaction columns (one column for 2 blocks, two for 4, three for 8).

    Pairwise swaps from a random start can stall on a split that is just as
    D-efficient but sits on a lower-order term; starting from the classical
    split avoids that. Runs where the column is zero (centre, axial) are dealt
    round-robin. Starts that do not hit the required block sizes are dropped.
    """
    r = {2: 1, 4: 2, 8: 3}.get(n_blocks)
    interactions = [t for t in potential if len(set(t)) == len(t) and len(t) >= 2]
    if r is None or not interactions:
        return []
    # Highest order first; cap the pool so the number of combinations stays small.
    top = sorted(interactions, key=len, reverse=True)[: {1: 128, 2: 48, 3: 20}[r]]
    cols = model_matrix(matrix, top)
    starts = []
    for combo in combinations(range(len(top)), r):
        signs = cols[:, list(combo)]
        zero = np.any(np.isclose(signs, 0.0), axis=1)
        labels = np.zeros(matrix.shape[0], dtype=int)
        labels[~zero] = (signs[~zero] > 0).astype(int) @ (1 << np.arange(r))
        counts = np.bincount(labels[~zero], minlength=n_blocks)
        for i in np.flatnonzero(zero):
            labels[i] = int(np.argmin(counts))
            counts[labels[i]] += 1
        if sorted(np.bincount(labels, minlength=n_blocks).tolist()) == sorted(sizes):
            starts.append(labels)
    return starts


def assign_blocks(
    matrix: np.ndarray, terms: list[Term], n_blocks: int, seed: int = 0, potential: list[Term] | None = None
) -> np.ndarray:
    """Block label (0..n_blocks-1) for each run.

    Raises ``ValueError`` when the runs cannot be split that finely: every
    block needs at least two runs, or its offset is just that run's value.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if n_blocks < 2:
        return np.zeros(n, dtype=int)
    if n < 2 * n_blocks:
        raise ValueError(f"{n} runs cannot be split into {n_blocks} blocks of at least 2")

    x_model = model_matrix(matrix, terms)
    if potential is None:
        # Every left-out interaction of any order, plus left-out squares: the
        # weights then push the blocks onto the highest-order one available.
        k = matrix.shape[1]
        fitted = set(terms)
        potential = [c for r in range(2, k + 1) for c in combinations(range(k), r) if c not in fitted]
        potential += [(i, i) for i in range(k) if (i, i) not in fitted]
    x_potential = model_matrix(matrix, potential)
    weights = np.array([10.0 ** -len(t) for t in potential])
    centre = np.all(np.isclose(matrix, 0.0), axis=1)
    sizes = block_sizes(n, n_blocks)
    rng = np.random.default_rng(seed)

    # Score every textbook start (cheap), refine only the best few (the swap
    # search is the expensive part), plus a few random starts for designs with
    # no textbook blocking.
    structured = _structured_starts(matrix, potential, n_blocks, sizes)
    structured.sort(key=lambda lab: _criterion(x_model, x_potential, weights, lab, centre), reverse=True)
    starts = structured[:N_REFINED]
    starts += [np.repeat(np.arange(n_blocks), sizes)[rng.permutation(n)] for _ in range(N_STARTS)]

    best_labels, best_score = None, (-np.inf, -np.inf, -n)
    for labels in starts:
        score = _criterion(x_model, x_potential, weights, labels, centre)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j] or np.array_equal(matrix[i], matrix[j]):
                        continue
                    labels[i], labels[j] = labels[j], labels[i]
                    trial = _criterion(x_model, x_potential, weights, labels, centre)
                    if trial > score:
                        score, improved = trial, True
                    else:
                        labels[i], labels[j] = labels[j], labels[i]
        if score > best_score:
            best_labels, best_score = labels.copy(), score
    return best_labels
