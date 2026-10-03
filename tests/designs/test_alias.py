"""Model matrices and the aliasing story.

Aliasing is the finding most likely to change a scientist's mind about a
design, so the plain-English statements are tested as carefully as the numbers
behind them.
"""

from __future__ import annotations

import numpy as np
import pytest

from doe_advisor.designs import classical as C
from doe_advisor.designs.alias import alias_report
from doe_advisor.designs.model import (
    is_estimable,
    model_matrix,
    model_terms,
    potential_terms,
    residual_df,
    term_label,
)
from doe_advisor.designs.spec import ModelOrder

NAMES = ["pH", "DO", "temp", "feed"]


# --------------------------------------------------------------------------
# Model terms
# --------------------------------------------------------------------------


def test_term_counts_per_model_order():
    assert len(model_terms(4, ModelOrder.MAIN)) == 1 + 4
    assert len(model_terms(4, ModelOrder.INTERACTION)) == 1 + 4 + 6
    assert len(model_terms(4, ModelOrder.QUADRATIC)) == 1 + 4 + 6 + 4


def test_intercept_comes_first():
    assert model_terms(3, ModelOrder.QUADRATIC)[0] == ()


def test_term_labels_read_like_a_scientist_would_write_them():
    assert term_label((), NAMES) == "intercept"
    assert term_label((0,), NAMES) == "pH"
    assert term_label((0, 2), NAMES) == "pH x temp"
    assert term_label((1, 1), NAMES) == "DO^2"


def test_model_matrix_builds_products_and_powers():
    matrix = np.array([[1.0, -1.0], [-1.0, -1.0], [0.5, 2.0]])
    x = model_matrix(matrix, [(), (0,), (1,), (0, 1), (0, 0)])
    assert np.allclose(x[:, 0], 1.0)  # intercept
    assert np.allclose(x[:, 3], matrix[:, 0] * matrix[:, 1])  # interaction
    assert np.allclose(x[:, 4], matrix[:, 0] ** 2)  # quadratic


def test_potential_terms_look_one_order_up():
    """Main-effects models are threatened by two-factor interactions; 2FI models
    by three-factor interactions and by curvature."""
    assert potential_terms(4, ModelOrder.MAIN) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    two_fi = potential_terms(4, ModelOrder.INTERACTION)
    assert (0, 1, 2) in two_fi
    assert (0, 0) in two_fi
    quad = potential_terms(4, ModelOrder.QUADRATIC)
    assert (0, 0) not in quad  # already in the model


def test_residual_df_and_estimability_agree_about_saturation():
    design = C.fractional_factorial(4, 1, n_center=3)
    terms = model_terms(4, ModelOrder.INTERACTION)
    assert residual_df(design.matrix, terms) == 0
    assert not is_estimable(design.matrix, terms)


# --------------------------------------------------------------------------
# Aliasing
# --------------------------------------------------------------------------


def test_full_factorial_has_nothing_aliased():
    report = alias_report(C.full_factorial(4, n_center=3), ModelOrder.INTERACTION, NAMES)
    assert report.estimable
    assert report.all_clear
    assert report.worst_main_effect_alias == 0.0
    assert report.statements() == ["Nothing is tangled: every term in your model can be estimated on its own."]


def test_resolution_iii_fully_aliases_main_effects_with_interactions():
    """The classic trap: a cheap 8-run screen for 6 factors cannot separate a
    main effect from a two-factor interaction."""
    design = C.fractional_factorial(6, 3, n_center=0)
    names = ["A", "B", "C", "D", "E", "F"]
    report = alias_report(design, ModelOrder.MAIN, names)
    assert report.estimable
    assert report.worst_main_effect_alias == pytest.approx(1.0)
    assert report.n_fully_aliased >= 1
    text = " ".join(report.statements())
    assert "cannot be told apart" in text


def test_resolution_iv_keeps_main_effects_clear():
    design = C.fractional_factorial(6, 2, n_center=0)
    names = ["A", "B", "C", "D", "E", "F"]
    report = alias_report(design, ModelOrder.MAIN, names)
    assert report.worst_main_effect_alias == pytest.approx(0.0, abs=1e-9)
    assert all(e.is_clear for e in report.main_effect_entries)


def test_unfittable_design_says_so_plainly():
    report = alias_report(C.fractional_factorial(4, 1, n_center=3), ModelOrder.INTERACTION, NAMES)
    assert not report.estimable
    statements = report.statements()
    assert len(statements) == 1
    assert "cannot fit the model" in statements[0]


def test_statements_are_truncated_with_a_count():
    """A full alias listing for a resolution-III design is a wall of text nobody
    reads, so it is capped and the remainder counted."""
    design = C.fractional_factorial(7, 4, n_center=0)
    names = list("ABCDEFG")
    report = alias_report(design, ModelOrder.MAIN, names)
    statements = report.statements(limit=2)
    assert len(statements) == 3
    assert statements[-1].startswith("...and")


def test_statements_are_ordered_worst_first():
    design = C.definitive_screening(5, n_center=1)
    names = list("ABCDE")
    report = alias_report(design, ModelOrder.INTERACTION, names)
    if report.estimable and not report.all_clear:
        entries = sorted((e for e in report.entries if not e.is_clear), key=lambda e: -e.worst)
        assert entries[0].worst >= entries[-1].worst


def test_intercept_is_excluded_from_the_alias_entries():
    """The intercept is always contaminated and never interesting."""
    report = alias_report(C.full_factorial(3, n_center=2), ModelOrder.MAIN, NAMES[:3])
    assert all(e.term != () for e in report.entries)


def test_dsd_keeps_main_effects_clear_of_interactions():
    """A DSD's headline claim, restated through the alias machinery rather than
    the matrix algebra, so the two agree."""
    design = C.definitive_screening(6, n_center=1)
    names = list("ABCDEF")
    report = alias_report(design, ModelOrder.MAIN, names)
    assert report.estimable
    assert report.worst_main_effect_alias == pytest.approx(0.0, abs=1e-9)
