"""Categorical factors in the maths layer.

The invariant everything else leans on: a two-level categorical factor is
coded -1 / +1, so it must give *exactly* the numbers a continuous factor at
its low / high gives — power, aliasing, prediction. If those ever disagree,
the categorical coding is wrong, not the continuous one.

Power for a k-level factor is checked against an independent closed form: in
a balanced one-factor design the difference between two levels' means has
standard error sigma * sqrt(2 / n_per_level), and scipy is asked directly.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy import stats

from doe_advisor.designs import classical as C
from doe_advisor.designs.model import (
    effect_codes,
    model_matrix,
    model_terms,
    n_columns,
    potential_terms,
    term_slices,
    term_width,
)
from doe_advisor.designs.properties import evaluate, fitted_model, power_report, term_contrasts
from doe_advisor.designs.spec import (
    Design,
    DesignSpec,
    DisallowedCombination,
    Factor,
    LinearConstraint,
    ModelOrder,
    Response,
)

RESPONSE = Response("titer", target_effect=1.0, noise_sd=0.5)


# --------------------------------------------------------------------------
# Factor and spec
# --------------------------------------------------------------------------


def test_a_categorical_factor_has_levels_and_no_range():
    f = Factor("media", levels=("A", "B", "C"))
    assert f.is_categorical and f.n_levels == 3
    assert f.setting(2) == "C"
    assert f.level_index("B") == 1
    with pytest.raises(ValueError, match="no level"):
        f.level_index("D")


@pytest.mark.parametrize(
    "levels, message",
    [(("A",), "at least 2"), (("A", "A"), "unique"), (("A", " "), "blank")],
)
def test_bad_levels_are_refused(levels, message):
    with pytest.raises(ValueError, match=message):
        Factor("media", levels=levels)


def test_spec_reports_levels_per_factor():
    spec = DesignSpec([Factor("pH", 6.8, 7.2), Factor("media", levels=("A", "B", "C"))])
    assert spec.levels == (0, 3)
    assert spec.has_categorical


def test_a_disallowed_combination_on_one_factor_alone_is_refused():
    with pytest.raises(ValueError, match="remove A from the factor"):
        DesignSpec(
            [Factor("media", levels=("A", "B")), Factor("line", levels=("L1", "L2"))],
            disallowed=[DisallowedCombination({"media": ("A",)})],
        )


def test_a_linear_constraint_on_a_categorical_factor_is_refused():
    with pytest.raises(ValueError, match="only continuous"):
        DesignSpec(
            [Factor("pH", 6.8, 7.2), Factor("media", levels=("A", "B"))],
            constraints=[LinearConstraint({"media": 1.0}, 1.0)],
        )


def test_feasible_applies_constraints_in_natural_units_and_disallowed_levels():
    spec = DesignSpec(
        [Factor("feed", 2, 6), Factor("bolus", 0, 4), Factor("media", levels=("A", "B")),
         Factor("line", levels=("L1", "L2"))],
        constraints=[LinearConstraint({"feed": 1.0, "bolus": 1.0}, 8.0)],
        disallowed=[DisallowedCombination({"media": ("B",), "line": ("L2",)})],
    )
    runs = np.array([
        [1, 1, 0, 0],    # feed 6 + bolus 4 = 10 > 8
        [1, 0, 0, 0],    # 6 + 2 = 8, on the boundary: allowed
        [-1, -1, 1, 1],  # media B with line L2: disallowed
        [-1, -1, 1, 0],  # media B with line L1: fine
    ], dtype=float)
    assert spec.feasible(runs).tolist() == [False, True, False, True]


# --------------------------------------------------------------------------
# Model layer
# --------------------------------------------------------------------------


def test_effect_codes_use_the_first_level_as_the_reference():
    codes = effect_codes(3)
    assert codes.tolist() == [[-1, -1], [1, 0], [0, 1]]
    assert effect_codes(2).tolist() == [[-1], [1]]


def test_a_categorical_factor_has_no_curvature_term():
    terms = model_terms(3, ModelOrder.QUADRATIC, levels=(0, 3, 0))
    assert (1, 1) not in terms
    assert (0, 0) in terms and (2, 2) in terms
    assert (1, 1) not in potential_terms(3, ModelOrder.INTERACTION, levels=(0, 3, 0))


def test_term_width_is_the_product_of_level_counts_minus_one():
    levels = (0, 3, 4)
    assert term_width((0,), levels) == 1
    assert term_width((1,), levels) == 2
    assert term_width((0, 1), levels) == 2
    assert term_width((1, 2), levels) == 6
    terms = model_terms(3, ModelOrder.INTERACTION, levels)
    slices = term_slices(terms, levels)
    assert slices[-1].stop == n_columns(terms, levels) == 1 + 1 + 2 + 3 + 2 + 3 + 6


def test_model_matrix_matches_a_naive_product_for_every_term_order():
    rng = np.random.default_rng(3)
    matrix = rng.uniform(-1, 1, size=(11, 4))
    terms = model_terms(4, ModelOrder.QUADRATIC) + potential_terms(4, ModelOrder.QUADRATIC)
    naive = np.column_stack([np.prod(matrix[:, list(t)], axis=1) if t else np.ones(11) for t in terms])
    assert np.array_equal(model_matrix(matrix, terms), naive)


def test_categorical_columns_are_effect_coded_and_interactions_are_products():
    matrix = np.array([[-1.0, 0], [1.0, 1], [-1.0, 2]])
    x = model_matrix(matrix, [(), (0,), (1,), (0, 1)], levels=(0, 3))
    expected = np.array([
        [1, -1, -1, -1, 1, 1],
        [1, 1, 1, 0, 1, 0],
        [1, -1, 0, 1, 0, -1],
    ], dtype=float)
    assert np.array_equal(x, expected)


# --------------------------------------------------------------------------
# A two-level categorical is a continuous factor in disguise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [ModelOrder.MAIN, ModelOrder.INTERACTION])
def test_a_two_level_categorical_scores_exactly_like_a_continuous_factor(order):
    base = C.full_factorial(3, n_center=0)
    continuous = DesignSpec([Factor(n, -1, 1) for n in "ABC"], responses=[RESPONSE], model_order=order)
    categorical = DesignSpec([Factor("A", -1, 1), Factor("B", -1, 1), Factor("C", levels=("lo", "hi"))],
                             responses=[RESPONSE], model_order=order)
    as_levels = base.matrix.copy()
    as_levels[:, 2] = (as_levels[:, 2] + 1) / 2  # -1 -> level 0, +1 -> level 1
    cat_design = Design("cat", "test", as_levels, ["A", "B", "C"], levels=(0, 0, 2))

    a, b = evaluate(base, continuous), evaluate(cat_design, categorical)
    assert a.power.per_term == pytest.approx(b.power.per_term, rel=1e-12)
    assert a.power.interaction_terms == pytest.approx(b.power.interaction_terms, rel=1e-12)
    assert a.power.detectable_effect_sd == pytest.approx(b.power.detectable_effect_sd, rel=1e-12)
    assert a.d_efficiency == pytest.approx(b.d_efficiency, rel=1e-12)
    assert a.aliasing.worst_main_effect_alias == pytest.approx(b.aliasing.worst_main_effect_alias, abs=1e-12)
    assert a.robustness.fraction_estimable == b.robustness.fraction_estimable
    # The worst prediction is at a corner either way...
    assert a.prediction.g_efficiency == pytest.approx(b.prediction.g_efficiency, rel=1e-9)
    # ...but the average is not the same quantity: a continuous factor is also
    # predicted between its edges, where the variance is lower; a categorical
    # one only ever at its two levels.
    assert b.prediction.i_value > a.prediction.i_value


# --------------------------------------------------------------------------
# Power for a k-level factor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k, per_level", [(3, 4), (4, 3), (5, 6)])
def test_categorical_power_matches_the_two_sample_closed_form(k, per_level):
    # One categorical factor (k levels, balanced) and a dummy two-level one,
    # crossed, so the categorical main effect is orthogonal to everything.
    rows = [(x, lvl) for x, lvl in product([-1.0, 1.0], range(k)) for _ in range(per_level)]
    design = Design("bal", "test", np.array(rows, dtype=float), ["x", "media"], levels=(0, k))
    spec = DesignSpec([Factor("x", -1, 1), Factor("media", levels=tuple("ABCDE"[:k]))],
                      responses=[RESPONSE], model_order=ModelOrder.MAIN)
    report = power_report(design, spec)

    n = len(rows)
    df = n - (1 + 1 + (k - 1))
    n_per = n / k
    ncp = (RESPONSE.target_effect / RESPONSE.noise_sd) / np.sqrt(2.0 / n_per)
    crit = stats.t.ppf(0.975, df)
    expected = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    assert report.per_term["media"] == pytest.approx(expected, rel=1e-9)


def test_power_is_quoted_for_the_hardest_pair_of_levels():
    # Level C gets fewer runs than A and B, so A-vs-C and B-vs-C are the weak pairs.
    rows = [[0], [0], [0], [0], [1], [1], [1], [1], [2], [2]]
    design = Design("unbal", "test", np.array(rows, dtype=float), ["media"], levels=(3,))
    terms = [(), (0,)]
    x = model_matrix(design.matrix, terms, levels=(3,))
    inv = np.linalg.inv(x.T @ x)
    ses = [float(np.sqrt(v @ inv[1:, 1:] @ v)) for v in term_contrasts((0,), (3,))]
    # Pairs in order (A,B), (A,C), (B,C): SE = sqrt(1/n_a + 1/n_b).
    assert ses == pytest.approx([np.sqrt(1 / 4 + 1 / 4), np.sqrt(1 / 4 + 1 / 2), np.sqrt(1 / 4 + 1 / 2)])


# --------------------------------------------------------------------------
# Terms the constraints make inestimable
# --------------------------------------------------------------------------


def test_a_disallowed_cell_drops_the_interaction_it_hides_and_nothing_else():
    spec = DesignSpec(
        [Factor("pH", 6.8, 7.2), Factor("media", levels=("A", "B", "C")), Factor("line", levels=("L1", "L2"))],
        model_order=ModelOrder.INTERACTION,
        disallowed=[DisallowedCombination({"media": ("C",), "line": ("L2",)})],
    )
    model = fitted_model(spec)
    assert model.dropped == ((1, 2),)
    assert (1,) in model.terms and (2,) in model.terms and (0, 1) in model.terms


def test_an_unconstrained_spec_fits_the_full_model():
    spec = DesignSpec([Factor("pH", 6.8, 7.2), Factor("media", levels=("A", "B", "C"))],
                      model_order=ModelOrder.INTERACTION)
    assert fitted_model(spec).terms == tuple(model_terms(2, ModelOrder.INTERACTION, (0, 3)))
    assert fitted_model(spec).dropped == ()
