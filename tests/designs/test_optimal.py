"""The custom (coordinate-exchange) design generator.

A search cannot be checked by re-deriving its answer, so these tests pin it to
cases where the optimum is known: the 2^3 factorial is D-optimal for a 2FI
model in 8 runs (D-efficiency exactly 1); the 3^2 factorial is D-optimal for a
two-factor quadratic in 9 runs; and a crossed full factorial is the D-optimal
12-run design for 2 x 2 x 3 factors. Against classical designs of the same size
it must never do worse on the criterion it optimises. And it must never
propose a run the constraints forbid.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from doe_advisor.designs import classical as C
from doe_advisor.designs.model import model_matrix, model_terms
from doe_advisor.designs.optimal import coordinate_exchange, custom_design, default_criterion, n_starts_for
from doe_advisor.designs.properties import d_efficiency, evaluate
from doe_advisor.designs.spec import (
    DesignSpec,
    DisallowedCombination,
    Factor,
    LinearConstraint,
    ModelOrder,
)


def cube(k: int, order: ModelOrder, **kwargs) -> DesignSpec:
    return DesignSpec([Factor(f"x{i}", -1, 1) for i in range(k)], model_order=order, **kwargs)


# --------------------------------------------------------------------------
# Known optima
# --------------------------------------------------------------------------


def test_finds_the_full_factorial_for_eight_runs_and_a_2fi_model():
    design = custom_design(cube(3, ModelOrder.INTERACTION), 8)
    assert evaluate(design, cube(3, ModelOrder.INTERACTION)).d_efficiency == pytest.approx(1.0)
    assert sorted(map(tuple, design.matrix)) == sorted(product([-1.0, 1.0], repeat=3))


def test_finds_the_three_level_factorial_for_a_two_factor_quadratic():
    spec = cube(2, ModelOrder.QUADRATIC)
    design = custom_design(spec, 9, criterion="D")
    terms = model_terms(2, ModelOrder.QUADRATIC)
    reference = np.array(list(product([-1.0, 0.0, 1.0], repeat=2)))
    assert d_efficiency(design.matrix, terms) == pytest.approx(d_efficiency(reference, terms), rel=1e-9)


def test_finds_the_crossed_factorial_for_mixed_categorical_factors():
    spec = DesignSpec(
        [Factor("pH", 6.8, 7.2), Factor("temp", 35, 37), Factor("media", levels=("A", "B", "C"))],
        model_order=ModelOrder.INTERACTION,
    )
    design = custom_design(spec, 12)
    assert design.levels == (0, 0, 3)
    runs = sorted(map(tuple, design.matrix))
    assert runs == sorted(product([-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0, 2.0]))


def test_never_worse_than_a_face_centred_ccd_of_the_same_size_on_its_own_criterion():
    spec = cube(3, ModelOrder.QUADRATIC)
    ccd = C.central_composite(3, alpha="face", n_center=1)
    terms = model_terms(3, ModelOrder.QUADRATIC)
    d_opt = custom_design(spec, ccd.n_runs, criterion="D")
    assert d_efficiency(d_opt.matrix, terms) >= d_efficiency(ccd.matrix, terms) - 1e-9
    i_opt = custom_design(spec, ccd.n_runs, criterion="I")
    assert evaluate(i_opt, spec).prediction.i_value <= evaluate(ccd, spec).prediction.i_value + 1e-9


def test_quadratic_models_default_to_i_and_screening_to_d():
    assert default_criterion(ModelOrder.QUADRATIC) == "I"
    assert default_criterion(ModelOrder.MAIN) == default_criterion(ModelOrder.INTERACTION) == "D"


# --------------------------------------------------------------------------
# Odd budgets, constraints, fixed runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [7, 11, 13])
def test_any_run_count_at_or_above_the_parameter_count(n_runs):
    spec = cube(4, ModelOrder.MAIN)
    design = custom_design(spec, n_runs)
    assert design.n_runs == n_runs
    assert evaluate(design, spec).estimable


def test_too_few_runs_is_an_error_not_a_singular_design():
    with pytest.raises(ValueError, match="cannot estimate"):
        custom_design(cube(4, ModelOrder.INTERACTION), 8)


def test_every_run_respects_linear_constraints_and_disallowed_combinations():
    spec = DesignSpec(
        [Factor("feed", 2, 6), Factor("bolus", 0, 4), Factor("media", levels=("A", "B", "C")),
         Factor("line", levels=("L1", "L2"))],
        model_order=ModelOrder.INTERACTION,
        constraints=[LinearConstraint({"feed": 1.0, "bolus": 1.0}, 8.0)],
        disallowed=[DisallowedCombination({"media": ("C",), "line": ("L2",)})],
    )
    design = custom_design(spec, 20)
    assert spec.feasible(design.matrix).all()
    # The search goes right up to the constraint rather than retreating from it.
    decoded = design.decoded(spec.factors)
    assert np.isclose(decoded[:, 0] + decoded[:, 1], 8.0).any()


def test_a_region_with_nothing_feasible_is_an_error():
    spec = cube(2, ModelOrder.MAIN, constraints=[LinearConstraint({"x0": 1.0, "x1": 1.0}, -5.0)])
    with pytest.raises(ValueError, match="no feasible run"):
        custom_design(spec, 5)


def test_fixed_runs_are_kept_first_and_unchanged():
    spec = cube(3, ModelOrder.INTERACTION)
    done = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
    design = custom_design(spec, 12, fixed_runs=done)
    assert np.array_equal(design.matrix[:3], done)
    assert design.detail["n_fixed_runs"] == 3


def test_centre_runs_are_included_when_asked_for():
    spec = DesignSpec([Factor("pH", 6.8, 7.2), Factor("temp", 35, 37), Factor("media", levels=("A", "B"))],
                      model_order=ModelOrder.INTERACTION)
    design = custom_design(spec, 12, n_center=3)
    assert design.n_center_points >= 3


def test_the_search_is_deterministic_for_a_seed():
    spec = cube(4, ModelOrder.INTERACTION)
    a, b = custom_design(spec, 14, seed=3), custom_design(spec, 14, seed=3)
    assert np.array_equal(a.matrix, b.matrix)


def test_i_criterion_needs_a_region():
    with pytest.raises(ValueError, match="region"):
        coordinate_exchange(9, [0, 0], model_terms(2, ModelOrder.QUADRATIC), criterion="I")


def test_large_searches_use_fewer_starts():
    assert n_starts_for(8, 3) > n_starts_for(70, 10) >= 4


def test_the_result_matrix_is_estimable_for_categorical_interactions():
    spec = DesignSpec([Factor("media", levels=("A", "B", "C")), Factor("line", levels=("L1", "L2", "L3"))],
                      model_order=ModelOrder.INTERACTION)
    design = custom_design(spec, 12)
    x = model_matrix(design.matrix, model_terms(2, ModelOrder.INTERACTION, (3, 3)), levels=(3, 3))
    assert np.linalg.matrix_rank(x) == x.shape[1] == 9
