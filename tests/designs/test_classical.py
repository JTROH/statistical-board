"""Generators must produce the designs they claim to.

The generator table is copied from published sources, so the tests check
*derived* properties (resolution computed from the defining relation, run
counts, orthogonality) rather than restating the table. A typo in a generator
word shows up as a resolution mismatch.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from doe_advisor.designs import classical as C
from doe_advisor.designs.model import model_matrix
from doe_advisor.designs.spec import ModelOrder

# Published resolutions for the tabulated minimum-aberration designs.
EXPECTED_RESOLUTION = {
    (3, 1): 3, (4, 1): 4, (5, 1): 5, (5, 2): 3,
    (6, 1): 6, (6, 2): 4, (6, 3): 3,
    (7, 1): 7, (7, 2): 4, (7, 3): 4, (7, 4): 3,
    (8, 2): 5, (8, 3): 4, (8, 4): 4,
    (9, 4): 4, (9, 5): 3,
    (10, 5): 4, (10, 6): 3,
}


@pytest.mark.parametrize(("key", "expected"), sorted(EXPECTED_RESOLUTION.items()))
def test_generator_table_has_published_resolution(key, expected):
    assert C.resolution_of(C.GENERATORS[key]) == expected


def test_generator_table_is_complete():
    assert set(C.GENERATORS) == set(EXPECTED_RESOLUTION)


# --------------------------------------------------------------------------
# Full factorial
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_full_factorial_run_count(k):
    d = C.full_factorial(k, n_center=0)
    assert d.n_runs == 2**k
    assert set(np.unique(d.matrix)) == {-1.0, 1.0}


def test_full_factorial_is_orthogonal_for_interaction_model():
    d = C.full_factorial(4)
    from doe_advisor.designs.model import model_terms

    x = model_matrix(d.matrix, model_terms(4, ModelOrder.INTERACTION))
    xtx = x.T @ x
    assert np.allclose(xtx, np.diag(np.diag(xtx)))


def test_center_points_are_counted():
    d = C.full_factorial(3, n_center=4)
    assert d.n_runs == 12
    assert d.n_center_points == 4


# --------------------------------------------------------------------------
# Fractional factorial
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "p"), sorted(EXPECTED_RESOLUTION))
def test_fractional_factorial_shape_and_levels(k, p):
    d = C.fractional_factorial(k, p, n_center=0)
    assert d.n_runs == 2 ** (k - p)
    assert d.n_factors == k
    assert set(np.unique(d.matrix)) == {-1.0, 1.0}


@pytest.mark.parametrize(("k", "p"), sorted(EXPECTED_RESOLUTION))
def test_fractional_factorial_columns_are_balanced(k, p):
    """Every factor must be run at low and high equally often, or its effect is
    entangled with the overall mean."""
    d = C.fractional_factorial(k, p, n_center=0)
    assert np.allclose(d.matrix.sum(axis=0), 0.0)


@pytest.mark.parametrize(("k", "p"), sorted(EXPECTED_RESOLUTION))
def test_fractional_factorial_main_effects_are_orthogonal(k, p):
    d = C.fractional_factorial(k, p, n_center=0)
    gram = d.matrix.T @ d.matrix
    assert np.allclose(gram, np.diag(np.diag(gram)))


def test_resolution_iv_keeps_main_effects_clear_of_two_factor_interactions():
    """The defining property of resolution IV, checked directly on the matrix."""
    d = C.fractional_factorial(6, 2, n_center=0)
    main = d.matrix
    twofi = model_matrix(d.matrix, [(i, j) for i, j in combinations(range(6), 2)])
    assert np.abs(main.T @ twofi).max() < 1e-9


def test_resolution_iii_does_not_keep_main_effects_clear():
    d = C.fractional_factorial(6, 3, n_center=0)
    twofi = model_matrix(d.matrix, [(i, j) for i, j in combinations(range(6), 2)])
    assert np.abs(d.matrix.T @ twofi).max() > 1e-9


def test_untabulated_fraction_raises():
    with pytest.raises(KeyError):
        C.fractional_factorial(4, 3)


# --------------------------------------------------------------------------
# Definitive screening designs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [4, 6, 8, 12, 14, 18, 20])
def test_paley_conference_matrix_properties(order):
    c = C._paley_conference(order)
    assert c is not None
    assert np.allclose(np.diag(c), 0.0)
    assert set(np.unique(np.abs(c))) <= {0.0, 1.0}
    assert np.allclose(c.T @ c, (order - 1) * np.eye(order))


def test_conference_matrix_falls_back_to_a_larger_order():
    """Order 10 needs GF(9); the search should step up to 12 rather than fail."""
    assert C._paley_conference(10) is None
    c = C._conference_matrix(10)
    assert c is not None and c.shape[0] >= 10


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8])
def test_dsd_main_effects_are_orthogonal_to_every_two_factor_interaction(k):
    """This is the whole point of a DSD, and the reason to prefer it over a
    resolution-III fraction of similar size."""
    d = C.definitive_screening(k, n_center=1)
    main = model_matrix(d.matrix, [(i,) for i in range(k)])
    twofi = model_matrix(d.matrix, [(i, j) for i, j in combinations(range(k), 2)])
    assert np.abs(main.T @ twofi).max() < 1e-9


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8])
def test_dsd_has_three_levels_and_a_center_point(k):
    d = C.definitive_screening(k, n_center=1)
    assert set(np.unique(d.matrix)) == {-1.0, 0.0, 1.0}
    assert d.n_center_points >= 1


def test_dsd_requires_three_factors():
    with pytest.raises(ValueError):
        C.definitive_screening(2)


# --------------------------------------------------------------------------
# Central composite
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_ccd_run_count(k):
    d = C.central_composite(k, n_center=4)
    assert d.n_runs == 2**k + 2 * k + 4


def test_ccd_rotatable_alpha():
    d = C.central_composite(3, alpha="rotatable", n_center=4)
    assert d.detail["alpha"] == pytest.approx(8**0.25)


def test_ccd_face_centred_stays_inside_declared_ranges():
    """The practical reason to pick face-centred: no run asks the scientist to
    go outside the range they said was safe."""
    d = C.central_composite(3, alpha="face", n_center=4)
    assert np.abs(d.matrix).max() == pytest.approx(1.0)
    assert d.detail["exceeds_declared_range"] is False


def test_ccd_rotatable_leaves_declared_ranges():
    d = C.central_composite(3, alpha="rotatable", n_center=4)
    assert d.detail["exceeds_declared_range"] is True


def test_ccd_with_fractional_core_is_smaller():
    full = C.central_composite(5, n_center=4)
    frac = C.central_composite(5, n_center=4, core_generators=1)
    assert frac.n_runs < full.n_runs
    assert frac.detail["core"] == "1/2"


# --------------------------------------------------------------------------
# Box-Behnken
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "factorial_runs"), [(3, 12), (4, 24), (5, 40), (6, 48), (7, 56)])
def test_box_behnken_run_counts_match_published_designs(k, factorial_runs):
    d = C.box_behnken(k, n_center=3)
    assert d.n_runs == factorial_runs + 3


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
def test_box_behnken_never_visits_a_corner(k):
    """Its selling point in bioprocess work: it never combines every factor at
    an extreme at once, so it avoids the run where the culture simply dies."""
    d = C.box_behnken(k, n_center=3)
    assert not np.any(np.all(np.abs(d.matrix) == 1.0, axis=1))
    assert np.abs(d.matrix).max() == pytest.approx(1.0)


@pytest.mark.parametrize("k", [2, 8])
def test_box_behnken_rejects_unsupported_factor_counts(k):
    with pytest.raises(ValueError):
        C.box_behnken(k)
