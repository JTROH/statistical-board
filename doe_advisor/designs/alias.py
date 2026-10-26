"""Aliasing: what this design will not let you tell apart.

This is the trap a bench scientist most often walks into unaided. A 2^(7-4)
screen in 8 runs looks wonderfully cheap right up to the moment you find a big
effect and discover the design cannot tell you whether it was temperature or
the pH-by-feed-rate interaction.

The alias matrix is ``A = (X1'X1)^-1 X1' X2``, where ``X1`` holds the terms you
intend to fit and ``X2`` the terms you are not fitting but which could still be
active. Entry ``A[i, j]`` says how much of potential term ``j`` gets absorbed
into your estimate of term ``i``. In an orthogonal two-level design an entry
of 1.0 means completely indistinguishable. In a non-orthogonal design (a
constrained or optimal one) entries can exceed 1 without anything being
indistinguishable, so "completely" is decided separately: a potential term is
completely confounded when its column lies inside the span of the fitted
columns, i.e. adding it to the model would make the model singular.

A term with a categorical factor in it spans several columns. The report is
kept at the level of terms — "media is tangled with pH x feed" — by taking the
largest entry across the two terms' columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    Term,
    cross_information,
    information,
    is_estimable,
    model_matrix,
    model_terms,
    potential_terms,
    term_label,
    term_slices,
)
from .spec import Design, ModelOrder

# Below this, an alias coefficient is numerical dust rather than a real concern.
ALIAS_TOL = 1e-8
# At or above this, treat the two terms as completely indistinguishable.
FULL_ALIAS = 1.0 - 1e-8
# A potential term whose column keeps less than this share of its length after
# projecting out the fitted columns is completely confounded with them.
FULL_ALIAS_RESIDUAL = 1e-8
# Partners named in a sentence. An optimal or constrained design leaves many
# small, non-zero alias entries; naming all of them buries the ones that
# matter, so smaller ones are counted rather than named.
MENTION_ALIAS = 0.1
MAX_NAMED_PARTNERS = 4


@dataclass
class AliasEntry:
    """One model term and everything tangled up in its estimate."""

    term: Term
    label: str
    partners: list[tuple[str, float]] = field(default_factory=list)
    # Partners this term cannot be told apart from at all (see the module notes).
    full_partners: list[str] = field(default_factory=list)

    @property
    def is_clear(self) -> bool:
        return not self.partners

    @property
    def worst(self) -> float:
        return max((abs(c) for _, c in self.partners), default=0.0)

    @property
    def is_fully_aliased(self) -> bool:
        return bool(self.full_partners)


@dataclass
class AliasReport:
    """The whole aliasing picture, plus the numbers the ranker needs."""

    entries: list[AliasEntry]
    estimable: bool
    n_model_terms: int
    n_potential_terms: int
    # Potential (unmodelled) terms that the block effects absorb, e.g. the
    # three-factor interaction in a 2^3 run in two blocks. Usually harmless and
    # usually deliberate, but worth stating.
    block_partners: list[str] = field(default_factory=list)

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
    def main_effect_fully_aliased(self) -> bool:
        """Is any main effect completely indistinguishable from something left out?"""
        return any(e.is_fully_aliased for e in self.main_effect_entries)

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
            ordered = sorted(entry.partners, key=lambda p: (-abs(p[1]), p[0]))
            if entry.is_fully_aliased:
                named = [name for name, _ in ordered if name in entry.full_partners]
            else:
                named = [name for name, c in ordered if abs(c) >= MENTION_ALIAS][:MAX_NAMED_PARTNERS] or [ordered[0][0]]
            rest = len(ordered) - len(named)
            names = ", ".join(named)
            more = f" It is also partly mixed up with {rest} other term{'s' if rest != 1 else ''}." if rest else ""
            if entry.is_fully_aliased:
                out.append(
                    f"'{entry.label}' cannot be told apart from {names}. If you see an effect here, "
                    f"you will not know which of them caused it.{more}"
                )
            else:
                out.append(
                    f"'{entry.label}' is partly contaminated by {names} (up to "
                    f"{entry.worst:.2f} of it leaks in), so its estimate will be biased if those are active.{more}"
                )
        remaining = len(ranked) - len(out)
        if remaining > 0:
            out.append(f"...and {remaining} further term{'s' if remaining != 1 else ''} with the same problem.")
        return out


def alias_report(
    design: Design,
    order: ModelOrder,
    factor_names: list[str] | None = None,
    terms: list[Term] | None = None,
    extra_potential: list[Term] | None = None,
    eta: float = 1.0,
) -> AliasReport:
    """Compute the alias structure of ``design`` under the given model order.

    ``terms`` overrides the fitted terms (when constraints made some of them
    inestimable); ``extra_potential`` adds terms to check as aliases — the
    ones that were dropped, which still bias the fit if they are active.
    For a split-plot design (``design.whole_plots``) the alias matrix is the
    GLS one, ``(X1'V^-1 X1)^-1 X1'V^-1 X2``, with whole-plot variance ratio ``eta``.
    """
    names = factor_names or design.factor_names
    levels = design.levels
    fitted = list(terms) if terms is not None else model_terms(design.n_factors, order, levels)
    potential = potential_terms(design.n_factors, order, levels)
    potential += [t for t in extra_potential or [] if t not in potential]

    if not is_estimable(design.matrix, fitted, blocks=design.blocks, levels=levels):
        return AliasReport(entries=[], estimable=False, n_model_terms=len(fitted), n_potential_terms=len(potential))

    # Blocks sit in X1 alongside the model: they are fitted, so they cannot
    # bias it, but they can absorb a potential term — reported separately.
    x1 = model_matrix(design.matrix, fitted, design.blocks, levels)
    x2 = model_matrix(design.matrix, potential, levels=levels) if potential else np.zeros((design.n_runs, 0))
    wp = design.whole_plots
    a = (
        np.linalg.solve(information(x1, wp, eta), cross_information(x1, x2, wp, eta))
        if potential
        else np.zeros((x1.shape[1], 0))
    )
    fitted_cols = term_slices(fitted, levels)
    potential_cols = term_slices(potential, levels)
    n_model_cols = fitted_cols[-1].stop if fitted_cols else 0

    def coefficient(rows: slice, j: int) -> float:
        """Largest alias entry between a set of X1 columns and potential term j, sign kept."""
        block = a[rows, potential_cols[j]]
        flat = block.ravel()
        return float(flat[np.argmax(np.abs(flat))]) if flat.size else 0.0

    # Which potential terms lie entirely inside the fitted columns' span.
    residual = x2 - x1 @ a if potential else x2
    lengths = np.einsum("ij,ij->j", x2, x2)
    leftover = np.einsum("ij,ij->j", residual, residual)
    full_cols = leftover <= FULL_ALIAS_RESIDUAL * np.maximum(lengths, 1e-300)
    full_terms = {j for j in range(len(potential)) if full_cols[potential_cols[j]].all()}

    block_rows = slice(n_model_cols, x1.shape[1])
    block_partners = sorted(
        {term_label(potential[j], names) for j in range(len(potential))
         if x1.shape[1] > n_model_cols and abs(coefficient(block_rows, j)) > ALIAS_TOL}
    )

    entries: list[AliasEntry] = []
    for i, term in enumerate(fitted):
        if not term:  # the intercept is always contaminated and never interesting
            continue
        partners = [
            (term_label(potential[j], names), c)
            for j in range(len(potential))
            if abs(c := coefficient(fitted_cols[i], j)) > ALIAS_TOL
        ]
        full = [
            term_label(potential[j], names)
            for j in sorted(full_terms)
            if abs(coefficient(fitted_cols[i], j)) > ALIAS_TOL
        ]
        entries.append(AliasEntry(term=term, label=term_label(term, names), partners=partners, full_partners=full))

    return AliasReport(
        entries=entries,
        estimable=True,
        n_model_terms=len(fitted),
        n_potential_terms=len(potential),
        block_partners=block_partners,
    )
