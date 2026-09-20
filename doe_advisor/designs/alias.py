"""Aliasing: what this design will not let you tell apart.

This is the trap a bench scientist most often walks into unaided. A 2^(7-4)
screen in 8 runs looks wonderfully cheap right up to the moment you find a big
effect and discover the design cannot tell you whether it was temperature or
the pH-by-feed-rate interaction.

The alias matrix is ``A = (X1'X1)^-1 X1' X2``, where ``X1`` holds the terms you
intend to fit and ``X2`` the terms you are not fitting but which could still be
active. Entry ``A[i, j]`` says how much of potential term ``j`` gets absorbed
into your estimate of term ``i``. An entry of 1.0 means completely
indistinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Term, is_estimable, model_matrix, model_terms, potential_terms, term_label
from .spec import Design, ModelOrder

# Below this, an alias coefficient is numerical dust rather than a real concern.
ALIAS_TOL = 1e-8
# At or above this, treat the two terms as completely indistinguishable.
FULL_ALIAS = 1.0 - 1e-8


@dataclass
class AliasEntry:
    """One model term and everything tangled up in its estimate."""

    term: Term
    label: str
    partners: list[tuple[str, float]] = field(default_factory=list)

    @property
    def is_clear(self) -> bool:
        return not self.partners

    @property
    def worst(self) -> float:
        return max((abs(c) for _, c in self.partners), default=0.0)

    @property
    def is_fully_aliased(self) -> bool:
        return self.worst >= FULL_ALIAS


@dataclass
class AliasReport:
    """The whole aliasing picture, plus the numbers the ranker needs."""

    entries: list[AliasEntry]
    estimable: bool
    n_model_terms: int
    n_potential_terms: int

    @property
    def main_effect_entries(self) -> list[AliasEntry]:
        return [e for e in self.entries if len(e.term) == 1]

    @property
    def worst_main_effect_alias(self) -> float:
        """The ranking number: how badly is the *most* compromised main effect
        contaminated? Main effects are what a screening study is for, so this is
        the figure that matters most."""
        return max((e.worst for e in self.main_effect_entries), default=0.0)

    @property
    def n_fully_aliased(self) -> int:
        return sum(1 for e in self.entries if e.is_fully_aliased)

    @property
    def all_clear(self) -> bool:
        return all(e.is_clear for e in self.entries)

    def statements(self, limit: int = 6) -> list[str]:
        """Plain-English sentences a non-statistician can act on.

        Ordered worst-first and truncated, because a full alias listing for a
        resolution-III design is a wall of text nobody reads.

        Factor names are already baked into the labels by :func:`alias_report`,
        so callers do not pass them again — an earlier signature accepted and
        silently discarded them, which invited callers to pass anything.
        """
        if not self.estimable:
            return ["This design cannot fit the model you asked for at all — there are not enough independent runs."]
        if self.all_clear:
            return ["Nothing is tangled: every term in your model can be estimated on its own."]

        ranked = sorted(
            (e for e in self.entries if not e.is_clear),
            key=lambda e: (-e.worst, e.label),
        )
        out: list[str] = []
        for entry in ranked[:limit]:
            names = ", ".join(name for name, _ in sorted(entry.partners, key=lambda p: -abs(p[1])))
            if entry.is_fully_aliased:
                out.append(
                    f"'{entry.label}' cannot be told apart from {names}. If you see an effect here, "
                    f"you will not know which of them caused it."
                )
            else:
                out.append(
                    f"'{entry.label}' is partly contaminated by {names} (up to "
                    f"{entry.worst:.2f} of it leaks in), so its estimate will be biased if those are active."
                )
        remaining = len(ranked) - len(out)
        if remaining > 0:
            out.append(f"...and {remaining} further term{'s' if remaining != 1 else ''} with the same problem.")
        return out


def alias_report(design: Design, order: ModelOrder, factor_names: list[str] | None = None) -> AliasReport:
    """Compute the alias structure of ``design`` under the given model order."""
    names = factor_names or design.factor_names
    fitted = model_terms(design.n_factors, order)
    potential = potential_terms(design.n_factors, order)

    if not is_estimable(design.matrix, fitted):
        return AliasReport(entries=[], estimable=False, n_model_terms=len(fitted), n_potential_terms=len(potential))

    x1 = model_matrix(design.matrix, fitted)
    x2 = model_matrix(design.matrix, potential) if potential else np.zeros((design.n_runs, 0))
    a = np.linalg.solve(x1.T @ x1, x1.T @ x2) if potential else np.zeros((len(fitted), 0))

    entries: list[AliasEntry] = []
    for i, term in enumerate(fitted):
        if not term:  # the intercept is always contaminated and never interesting
            continue
        partners = [
            (term_label(potential[j], names), float(a[i, j]))
            for j in range(a.shape[1])
            if abs(a[i, j]) > ALIAS_TOL
        ]
        entries.append(AliasEntry(term=term, label=term_label(term, names), partners=partners))

    return AliasReport(
        entries=entries,
        estimable=True,
        n_model_terms=len(fitted),
        n_potential_terms=len(potential),
    )
