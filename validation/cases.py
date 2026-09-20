"""Frozen benchmark cases for the cross-validation suite.

These are fixed on purpose. Adding a case is fine; changing one silently is not,
because the whole point is that today's answers can be compared with the answers
from before a refactor, and with R's and JMP's answers for the same input.
"""

from __future__ import annotations

# Fractional factorials: (n_factors, n_generators). Compared against R's FrF2 on
# run count, resolution and word-length pattern.
FRACTIONAL_CASES: list[tuple[int, int]] = [
    (3, 1), (4, 1), (5, 1), (5, 2),
    (6, 1), (6, 2), (6, 3),
    (7, 1), (7, 2), (7, 3), (7, 4),
    (8, 2), (8, 3), (8, 4),
    (9, 4), (9, 5),
    (10, 5), (10, 6),
]

# Box-Behnken designs: n_factors. Compared against R's rsm::bbd.
BOX_BEHNKEN_CASES: list[int] = [3, 4, 5, 6, 7]

# Central composite designs: (n_factors, alpha_rule). Compared against rsm::ccd.
CENTRAL_COMPOSITE_CASES: list[tuple[int, str]] = [
    (2, "rotatable"), (3, "rotatable"), (4, "rotatable"), (5, "rotatable"),
    (2, "face"), (3, "face"), (4, "face"),
]

# Definitive screening designs: n_factors. No R reference package is installed,
# so these are checked against the properties Jones & Nachtsheim (2011) claim:
# 2m+1 runs, three levels, and main effects orthogonal to every two-factor
# interaction.
DSD_CASES: list[int] = [4, 5, 6, 7, 8]
