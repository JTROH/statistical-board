"""Classical design generators.

Everything here is built in-house on numpy so that the arithmetic is auditable
and so that the JMP/R validation suite has something of our own to check.

Families implemented:

- ``full_factorial``    -- 2^k, every corner of the design space
- ``fractional_factorial`` -- 2^(k-p) from published minimum-aberration generators
- ``definitive_screening`` -- Jones & Nachtsheim (2011) DSD, via Paley conference matrices
- ``central_composite`` -- factorial core + axial points, for curvature
- ``box_behnken``       -- three-level design that never visits a corner

Resolution is *computed* from the defining relation rather than hardcoded, so a
bad generator-table entry shows up as a failing test instead of a quiet lie in
the memo.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from .spec import Design

# Factor letters, conventionally skipping "I" because it denotes the identity
# in the defining relation.
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

# Published minimum-aberration generators, keyed by (n_factors, n_generators).
# Each entry defines an added factor as a product of the base factors.
# Sources: Box, Hunter & Hunter, *Statistics for Experimenters*, 2nd ed., Table
# 6.17; NIST/SEMATECH e-Handbook of Statistical Methods, section 5.3.3.4.4.
GENERATORS: dict[tuple[int, int], list[str]] = {
    (3, 1): ["C=AB"],
    (4, 1): ["D=ABC"],
    (5, 1): ["E=ABCD"],
    (5, 2): ["D=AB", "E=AC"],
    (6, 1): ["F=ABCDE"],
    (6, 2): ["E=ABC", "F=BCD"],
    (6, 3): ["D=AB", "E=AC", "F=BC"],
    (7, 1): ["G=ABCDEF"],
    (7, 2): ["F=ABCD", "G=ABDE"],
    (7, 3): ["E=ABC", "F=BCD", "G=ACD"],
    (7, 4): ["D=AB", "E=AC", "F=BC", "G=ABC"],
    (8, 2): ["G=ABCD", "H=ABEF"],
    (8, 3): ["F=ABC", "G=ABD", "H=BCDE"],
    (8, 4): ["E=BCD", "F=ACD", "G=ABC", "H=ABD"],
    (9, 4): ["F=BCDE", "G=ACDE", "H=ABDE", "J=ABCE"],
    (9, 5): ["E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD"],
    (10, 5): ["F=ABCD", "G=ABCE", "H=ABDE", "J=ACDE", "K=BCDE"],
    (10, 6): ["E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD", "K=AB"],
}


def _letter_index(letter: str) -> int:
    idx = LETTERS.find(letter.upper())
    if idx < 0:
        raise ValueError(f"unknown factor letter {letter!r}")
    return idx


def _center_block(n_factors: int, n_center: int) -> np.ndarray:
    return np.zeros((max(n_center, 0), n_factors), dtype=float)


def _named(base: str, n_center: int) -> str:
    """Design names must distinguish variants, because they are used as labels.

    Two full factorials differing only in centre points are genuinely different
    designs with different power and different degrees of freedom. Giving them
    the same name lets a comparison table show two identical-looking rows, and
    lets anything keyed on the name silently merge them.
    """
    if n_center <= 0:
        return base
    return f"{base} +{n_center} centre"


def replicate(design: Design, times: int) -> Design:
    """Run the whole design ``times`` over.

    Replication buys precision and a clean estimate of pure error, but it never
    unties an alias: running a resolution-III screen twice gives you the same
    confounding, measured better. Worth offering as an option and worth being
    explicit about, because "just repeat it" is a common instinct.
    """
    if times < 2:
        raise ValueError("replication needs times >= 2")
    detail = dict(design.detail)
    detail["replicates"] = times
    detail["replicated_from"] = design.name
    return Design(
        name=f"{design.name} x{times} replicated",
        family=design.family,
        matrix=np.vstack([design.matrix] * times),
        factor_names=list(design.factor_names),
        detail=detail,
    )


# --------------------------------------------------------------------------
# Full factorial
# --------------------------------------------------------------------------


def full_factorial(n_factors: int, n_center: int = 0) -> Design:
    """2^k: every combination of low and high.

    The gold standard — nothing is confounded with anything — and the reason
    nobody runs it past about five factors, because the run count doubles with
    each factor added.
    """
    if n_factors < 2:
        raise ValueError("full factorial needs at least 2 factors")
    corners = np.array(list(product([-1.0, 1.0], repeat=n_factors)), dtype=float)
    matrix = np.vstack([corners, _center_block(n_factors, n_center)])
    return Design(
        name=_named(f"Full factorial 2^{n_factors}", n_center),
        family="full_factorial",
        matrix=matrix,
        factor_names=[LETTERS[i] for i in range(n_factors)],
        detail={
            "fraction": "full",
            "resolution": None,  # nothing is aliased
            "n_factorial_points": int(corners.shape[0]),
            "n_center": int(max(n_center, 0)),
            "defining_relation": [],
        },
    )


# --------------------------------------------------------------------------
# Fractional factorial
# --------------------------------------------------------------------------


def _defining_relation(generator_specs: list[str]) -> list[frozenset[int]]:
    """Full defining relation from the generator words.

    Each generator ``D=ABC`` contributes the word ``ABCD``. The complete
    relation is the group those words generate under symmetric difference
    (multiplication mod 2), which has 2^p - 1 non-identity members.
    """
    base_words: list[frozenset[int]] = []
    for spec in generator_specs:
        added, _, source = spec.partition("=")
        word = {_letter_index(added.strip())}
        for ch in source.strip():
            word ^= {_letter_index(ch)}
        base_words.append(frozenset(word))

    words: set[frozenset[int]] = set()
    for r in range(1, len(base_words) + 1):
        for combo in combinations(base_words, r):
            product_word: set[int] = set()
            for w in combo:
                product_word ^= set(w)
            if product_word:
                words.add(frozenset(product_word))
    return sorted(words, key=lambda w: (len(w), sorted(w)))


def resolution_of(generator_specs: list[str]) -> int | None:
    """Shortest word in the defining relation.

    Resolution III means a main effect is tangled with a two-factor
    interaction; IV means main effects are clear of 2FIs but 2FIs are tangled
    with each other; V means 2FIs are clear too. ``None`` means nothing is
    aliased at all (a full factorial).
    """
    words = _defining_relation(generator_specs)
    if not words:
        return None
    return min(len(w) for w in words)


def word_length_pattern(generator_specs: list[str], max_length: int | None = None) -> list[int]:
    """Counts of defining words of length 3, 4, ... up to ``max_length``.

    The word-length pattern is a design's aliasing fingerprint: two 2^(k-p)
    designs with the same pattern confound things in the same way. Resolution is
    only its first non-zero entry, so comparing patterns is a far stricter
    cross-check against R's FrF2 than comparing resolutions — and it is what the
    validation suite uses.

    ``max_length`` defaults to whichever is larger: 7, so that short patterns
    still line up with FrF2's fixed-width vector, or the longest word actually
    present. A fixed ceiling of 7 silently dropped the length-8 word in
    2^(8-4), which made the counts fail to add up to the 2^p - 1 words the
    design must have.
    """
    words = _defining_relation(generator_specs)
    longest = max((len(w) for w in words), default=3)
    ceiling = max(7, longest) if max_length is None else max_length
    return [sum(1 for w in words if len(w) == length) for length in range(3, ceiling + 1)]


def fractional_factorial(n_factors: int, n_generators: int, n_center: int = 0) -> Design:
    """2^(k-p) from the published generator table.

    Raises ``KeyError`` when the (k, p) combination is not tabulated; callers
    that are enumerating candidates should catch this and move on.
    """
    key = (n_factors, n_generators)
    if key not in GENERATORS:
        raise KeyError(f"no tabulated generators for 2^({n_factors}-{n_generators})")

    specs = GENERATORS[key]
    n_base = n_factors - n_generators
    base = np.array(list(product([-1.0, 1.0], repeat=n_base)), dtype=float)

    matrix = np.zeros((base.shape[0], n_factors), dtype=float)
    matrix[:, :n_base] = base
    for spec in specs:
        added, _, source = spec.partition("=")
        col = np.ones(base.shape[0], dtype=float)
        for ch in source.strip():
            col = col * matrix[:, _letter_index(ch)]
        matrix[:, _letter_index(added.strip())] = col

    matrix = np.vstack([matrix, _center_block(n_factors, n_center)])
    res = resolution_of(specs)
    roman = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}.get(res or 0, str(res))
    return Design(
        name=_named(f"Fractional factorial 2^({n_factors}-{n_generators}) res {roman}", n_center),
        family="fractional_factorial",
        matrix=matrix,
        factor_names=[LETTERS[i] for i in range(n_factors)],
        detail={
            "fraction": f"1/{2 ** n_generators}",
            "resolution": res,
            "generators": specs,
            "n_factorial_points": int(base.shape[0]),
            "n_center": int(max(n_center, 0)),
            "defining_relation": ["".join(LETTERS[i] for i in sorted(w)) for w in _defining_relation(specs)],
        },
    )


# --------------------------------------------------------------------------
# Definitive screening designs
# --------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _paley_conference(order: int) -> np.ndarray | None:
    """Conference matrix of the given order via the Paley construction.

    A conference matrix C is square with a zero diagonal, +/-1 elsewhere, and
    orthogonal rows (C'C = (n-1)I). Paley's construction builds one of order
    q+1 whenever q is an odd prime. Supporting primes only (not general prime
    powers) covers orders 4, 6, 8, 12, 14, 18, 20, 24 -- comfortably more
    factors than any realistic bioprocess screen.
    """
    q = order - 1
    if q < 3 or not _is_prime(q):
        return None

    residues = {(i * i) % q for i in range(1, q)}

    def chi(a: int) -> int:
        a %= q
        if a == 0:
            return 0
        return 1 if a in residues else -1

    jacobsthal = np.array([[chi(j - i) for j in range(q)] for i in range(q)], dtype=float)

    c = np.zeros((order, order), dtype=float)
    if q % 4 == 3:
        # Paley type I -> skew-symmetric conference matrix
        c[0, 1:] = 1.0
        c[1:, 0] = -1.0
        c[1:, 1:] = jacobsthal
    else:
        # Paley type II -> symmetric conference matrix
        c[0, 1:] = 1.0
        c[1:, 0] = 1.0
        c[1:, 1:] = jacobsthal
    return c


def _conference_matrix(min_order: int) -> np.ndarray | None:
    """Smallest available conference matrix of order >= ``min_order``."""
    for order in range(max(min_order, 4), max(min_order, 4) + 24):
        c = _paley_conference(order)
        if c is not None:
            return c
    return None


def definitive_screening(n_factors: int, n_center: int = 1) -> Design:
    """Definitive Screening Design (Jones & Nachtsheim, 2011).

    Its selling point for process development: in about 2k+1 runs it keeps main
    effects clear of *all* two-factor interactions and can detect curvature,
    which a resolution-III fraction of similar size cannot. Extra columns from
    an oversized conference matrix are simply dropped, which stays valid.
    """
    if n_factors < 3:
        raise ValueError("a definitive screening design needs at least 3 factors")
    c = _conference_matrix(n_factors)
    if c is None:
        raise ValueError(f"no conference matrix available for {n_factors} factors")

    c = c[:, :n_factors]
    matrix = np.vstack([c, -c, _center_block(n_factors, n_center)])
    return Design(
        name=_named(f"Definitive screening design ({n_factors} factors)", n_center),
        family="definitive_screening",
        matrix=matrix,
        factor_names=[LETTERS[i] for i in range(n_factors)],
        detail={
            "conference_order": int(c.shape[0]),
            "resolution": None,  # main effects are orthogonal to all 2FIs by construction
            "n_center": int(max(n_center, 0)),
            "three_level": True,
        },
    )


# --------------------------------------------------------------------------
# Central composite
# --------------------------------------------------------------------------


def central_composite(
    n_factors: int,
    alpha: float | str = "rotatable",
    n_center: int = 4,
    core_generators: int = 0,
) -> Design:
    """Factorial core + star points: the workhorse for fitting curvature.

    ``alpha`` accepts a number, or:

    - ``"rotatable"``  -- prediction variance depends only on distance from the
      centre, so the design is equally informative in every direction
    - ``"face"``       -- axial points sit on the faces (alpha = 1), so no run
      exceeds the ranges the scientist declared safe
    - ``"spherical"``  -- all non-centre points equidistant from the centre
    """
    if n_factors < 2:
        raise ValueError("a central composite design needs at least 2 factors")

    if core_generators > 0:
        core = fractional_factorial(n_factors, core_generators, n_center=0)
        core_matrix = core.matrix
        core_label = core.detail["fraction"]
    else:
        core_matrix = np.array(list(product([-1.0, 1.0], repeat=n_factors)), dtype=float)
        core_label = "full"

    n_f = core_matrix.shape[0]
    if isinstance(alpha, str):
        if alpha == "rotatable":
            alpha_value = float(n_f) ** 0.25
        elif alpha == "face":
            alpha_value = 1.0
        elif alpha == "spherical":
            alpha_value = float(n_factors) ** 0.5
        else:
            raise ValueError(f"unknown alpha rule {alpha!r}")
        alpha_rule = alpha
    else:
        alpha_value = float(alpha)
        alpha_rule = "custom"

    axial = np.zeros((2 * n_factors, n_factors), dtype=float)
    for i in range(n_factors):
        axial[2 * i, i] = -alpha_value
        axial[2 * i + 1, i] = alpha_value

    matrix = np.vstack([core_matrix, axial, _center_block(n_factors, n_center)])
    label = "face-centred" if alpha_rule == "face" else alpha_rule
    return Design(
        name=_named(f"Central composite ({label}, {n_factors} factors)", n_center),
        family="central_composite",
        matrix=matrix,
        factor_names=[LETTERS[i] for i in range(n_factors)],
        detail={
            "alpha": alpha_value,
            "alpha_rule": alpha_rule,
            "core": core_label,
            "n_factorial_points": int(n_f),
            "n_axial_points": int(2 * n_factors),
            "n_center": int(max(n_center, 0)),
            "exceeds_declared_range": bool(alpha_value > 1.0 + 1e-9),
            "three_level": True,
        },
    )


# --------------------------------------------------------------------------
# Box-Behnken
# --------------------------------------------------------------------------

# Published Box-Behnken block structures. For k = 3..5 the blocks are every
# pair of factors; for k = 6, 7 they are the classic triples.
_BB_BLOCKS: dict[int, list[tuple[int, ...]]] = {
    6: [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 0), (4, 5, 1), (5, 0, 2)],
    7: [(3, 4, 5), (0, 5, 6), (1, 4, 6), (0, 2, 4), (1, 2, 5), (0, 1, 3), (2, 3, 6)],
}


def box_behnken(n_factors: int, n_center: int = 3) -> Design:
    """Box-Behnken: three levels, and never a corner.

    Its practical virtue in bioprocess work is that it never combines every
    factor at its extreme simultaneously — so you avoid the run where high
    temperature meets high pH meets high feed rate and the culture simply dies.
    """
    if not 3 <= n_factors <= 7:
        raise ValueError("Box-Behnken designs are defined here for 3 to 7 factors")

    blocks = _BB_BLOCKS.get(n_factors) or list(combinations(range(n_factors), 2))
    rows: list[np.ndarray] = []
    for block in blocks:
        for combo in product([-1.0, 1.0], repeat=len(block)):
            row = np.zeros(n_factors, dtype=float)
            for idx, value in zip(block, combo, strict=True):
                row[idx] = value
            rows.append(row)

    matrix = np.vstack([np.array(rows, dtype=float), _center_block(n_factors, n_center)])
    return Design(
        name=_named(f"Box-Behnken ({n_factors} factors)", n_center),
        family="box_behnken",
        matrix=matrix,
        factor_names=[LETTERS[i] for i in range(n_factors)],
        detail={
            "n_blocks": len(blocks),
            "block_size": len(blocks[0]),
            "n_center": int(max(n_center, 0)),
            "exceeds_declared_range": False,
            "visits_corners": False,
            "three_level": True,
        },
    )
