"""Scoring must be right, and must be right for reasons we can restate.

The power test is deliberately an *independent* computation: for an orthogonal
two-level design the standard error of a coded coefficient is sigma/sqrt(n_f),
so the non-centrality parameter has a closed form and scipy can be asked
directly. If the engine and the closed form disagree, the engine is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from doe_advisor.designs import classical as C
from doe_advisor.designs.model import model_terms
from doe_advisor.designs.properties import (
    a_efficiency,
    d_efficiency,
    evaluate,
    extreme_point_sample,
    power_curve,
    power_report,
    prediction_report,
    robustness_report,
    scaled_prediction_variance,
)
from doe_advisor.designs.spec import Design, DesignSpec, Factor, ModelOrder, Response


def make_spec(k=4, order=ModelOrder.INTERACTION, target=0.5, noise=0.25, losses=2):
    factors = [Factor(f"x{i}", -1.0, 1.0) for i in range(k)]
    responses = [Response("titer", "g/L", target_effect=target, noise_sd=noise)]
    return DesignSpec(factors=factors, responses=responses, model_order=order, expected_run_losses=losses)


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def test_power_matches_closed_form_for_an_orthogonal_design():
    """2^4 with 3 centre points, fitting main effects + 2FI.

    Centre points carry no information about a main effect (their coded value
    is zero), so the effective replication is the 16 factorial points, while
    the degrees of freedom come from all 19 runs.
    """
    spec = make_spec(k=4, target=0.5, noise=0.25)  # standardised effect = 2.0 SD
    design = C.full_factorial(4, n_center=3)
    report = power_report(design, spec)

    n_factorial, n_terms = 16, 11
    df = 19 - n_terms
    ncp = (2.0 / 2.0) * np.sqrt(n_factorial)  # (delta/2) / (sigma/sqrt(n_f))
    crit = stats.t.ppf(0.975, df)
    expected = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)

    assert report.residual_df == df
    for value in report.per_term.values():
        assert value == pytest.approx(expected, rel=1e-10)


def test_power_is_unavailable_without_an_effect_size():
    """No target effect means power is genuinely unknown. The tool must say so
    rather than substitute a default."""
    spec = make_spec()
    spec.responses = [Response("titer", "g/L")]
    report = power_report(C.full_factorial(4, n_center=3), spec)
    assert report.per_term == {}
    assert report.min_main_effect_power is None
    assert report.detectable_effect_sd is not None  # still computable


def test_detectable_effect_is_also_reported_in_the_response_units():
    spec = make_spec(noise=0.25)
    report = power_report(C.full_factorial(4, n_center=3), spec)
    assert report.detectable_effect_units == pytest.approx(report.detectable_effect_sd * 0.25)


def test_detectable_effect_in_units_needs_a_noise_sd():
    spec = make_spec()
    spec.responses = [Response("titer", "g/L", target_effect=0.5)]
    report = power_report(C.full_factorial(4, n_center=3), spec)
    assert report.detectable_effect_sd is not None
    assert report.detectable_effect_units is None


def test_detectable_effect_and_power_agree_on_which_side_of_80_percent():
    """If power at the target is at least 80%, the target must be at least the
    smallest detectable effect, and vice versa."""
    for target in (0.2, 0.4, 0.6, 1.0):
        report = power_report(C.full_factorial(4, n_center=3), make_spec(target=target, noise=0.25))
        assert (report.min_main_effect_power >= 0.80) == (target >= report.detectable_effect_units - 1e-9)


def test_power_with_larger_noise_is_lower():
    report = power_report(C.full_factorial(4, n_center=3), make_spec())
    assert report.min_power_if_noise_high < report.min_main_effect_power
    # and it equals power at the same target with the SD actually 1.5x larger
    worse = power_report(C.full_factorial(4, n_center=3), make_spec(noise=0.25 * 1.5))
    assert report.min_power_if_noise_high == pytest.approx(worse.min_main_effect_power)


def test_power_curve_matches_a_real_orthogonal_design():
    """A full factorial with no centre points *is* the ideal design, so the
    curve must pass through its power exactly."""
    spec = make_spec(k=4)
    actual = power_report(C.full_factorial(4, n_center=0), spec).min_main_effect_power
    curve = power_curve(spec, up_to=32)
    assert curve.power[curve.n_runs.index(16)] == pytest.approx(actual, abs=1e-9)


def test_power_curve_rises_with_runs_and_reports_the_80_percent_crossing():
    curve = power_curve(make_spec(k=4), up_to=40)
    assert all(b >= a for a, b in zip(curve.power, curve.power[1:], strict=False))
    assert curve.runs_for_80 in curve.n_runs
    assert curve.power[curve.n_runs.index(curve.runs_for_80)] >= 0.80
    assert curve.power[curve.n_runs.index(curve.runs_for_80) - 1] < 0.80
    assert curve.runs_for_80_if_noise_high > curve.runs_for_80


def test_power_curve_needs_a_target_effect_and_noise():
    spec = make_spec()
    spec.responses = [Response("titer", "g/L")]
    assert power_curve(spec, up_to=40) is None


def test_power_curve_gives_up_on_a_hopeless_effect():
    curve = power_curve(make_spec(target=0.001, noise=1.0), up_to=40)
    assert curve.runs_for_80 is None


def test_saturated_design_reports_no_power():
    """Zero residual degrees of freedom means the model fits perfectly and tells
    you nothing about uncertainty."""
    spec = make_spec(k=4)
    report = power_report(C.fractional_factorial(4, 1, n_center=3), spec)
    assert report.saturated
    assert report.min_main_effect_power is None


def test_more_replication_raises_power():
    spec = make_spec(k=3, order=ModelOrder.MAIN)
    small = power_report(C.full_factorial(3, n_center=2), spec).min_main_effect_power
    large = power_report(C.full_factorial(3, n_center=10), spec).min_main_effect_power
    assert large > small


def test_detectable_effect_shrinks_as_runs_grow():
    spec = make_spec(k=3, order=ModelOrder.MAIN)
    small = power_report(C.full_factorial(3, n_center=2), spec).detectable_effect_sd
    large = power_report(C.full_factorial(3, n_center=12), spec).detectable_effect_sd
    assert large < small


# --------------------------------------------------------------------------
# Prediction precision
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "design_factory",
    [
        lambda: C.full_factorial(3, n_center=3),
        lambda: C.full_factorial(4, n_center=3),
        lambda: C.central_composite(3, n_center=4),
        lambda: C.central_composite(4, n_center=4),
        lambda: C.box_behnken(4, n_center=3),
        lambda: C.box_behnken(3, n_center=3),
    ],
)
def test_g_efficiency_never_exceeds_one(design_factory):
    """The Kiefer-Wolfowitz equivalence theorem puts the maximum scaled
    prediction variance at or above the number of model terms, so G-efficiency
    is capped at 1.0. Exceeding it means the sample missed the true maximum.
    """
    design = design_factory()
    spec = make_spec(k=design.n_factors)
    report = prediction_report(design, spec)
    assert report.g_efficiency <= 1.0 + 1e-9


def test_max_spv_is_at_least_the_number_of_model_terms():
    design = C.full_factorial(4, n_center=3)
    spec = make_spec(k=4)
    terms = model_terms(4, ModelOrder.INTERACTION)
    report = prediction_report(design, spec)
    assert report.max_spv >= len(terms) - 1e-9


def test_extreme_sample_finds_a_higher_maximum_than_the_uniform_one():
    """Guards the bug this sampling split exists to fix: a uniform interior
    sample sails past the corner where prediction variance peaks."""
    from doe_advisor.designs.properties import design_space_sample

    design = C.full_factorial(4, n_center=3)
    terms = model_terms(4, ModelOrder.INTERACTION)
    uniform_max = scaled_prediction_variance(design.matrix, terms, design_space_sample(4)).max()
    extreme_max = scaled_prediction_variance(design.matrix, terms, extreme_point_sample(4)).max()
    assert extreme_max > uniform_max


def test_i_value_is_finite_and_ordered_sensibly():
    """A full factorial predicts an interaction model better than a Box-Behnken,
    which spends its runs on curvature instead."""
    spec = make_spec(k=4)
    ff = prediction_report(C.full_factorial(4, n_center=3), spec)
    bb = prediction_report(C.box_behnken(4, n_center=3), spec)
    assert np.isfinite(ff.i_value)
    assert ff.i_value < bb.i_value


def test_unfittable_design_reports_infinite_prediction_variance():
    spec = make_spec(k=4)
    report = prediction_report(C.fractional_factorial(4, 1, n_center=3), spec)
    assert not np.isfinite(report.i_value)
    assert report.g_efficiency == 0.0


def test_fds_curve_is_sorted_and_thinned():
    spec = make_spec(k=3)
    report = prediction_report(C.full_factorial(3, n_center=3), spec)
    assert len(report.fds_curve) == 101
    assert report.fds_curve == sorted(report.fds_curve)


def test_rotatable_ccd_is_flagged_as_leaving_the_declared_range():
    spec = make_spec(k=3)
    rotatable = prediction_report(C.central_composite(3, alpha="rotatable", n_center=4), spec)
    faced = prediction_report(C.central_composite(3, alpha="face", n_center=4), spec)
    assert rotatable.axial_points_outside_range
    assert not faced.axial_points_outside_range


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_robustness_is_not_applicable_when_the_design_never_worked():
    """Reporting '100% robust' for a design that cannot be fitted at all would
    be actively misleading."""
    spec = make_spec(k=4, losses=2)
    report = robustness_report(C.fractional_factorial(4, 1, n_center=3), spec)
    assert report.applicable is False
    assert report.is_fragile is False
    assert report.n_scenarios == 0


def test_robustness_is_not_applicable_when_no_losses_expected():
    spec = make_spec(k=4, losses=0)
    assert robustness_report(C.full_factorial(4, n_center=3), spec).applicable is False


def test_generous_design_survives_run_losses():
    spec = make_spec(k=3, order=ModelOrder.MAIN, losses=2)
    report = robustness_report(C.full_factorial(3, n_center=4), spec)
    assert report.applicable
    assert report.fraction_estimable == 1.0
    assert 0.0 < report.worst_d_ratio <= 1.0


def test_tight_design_is_fragile():
    """A Box-Behnken with a single centre point can fit a quadratic model with
    three degrees of freedom to spare — and still fails in roughly a quarter of
    two-reactor-loss scenarios. This is the 'we lost 3 reactors, is this study
    still valid?' question, answered before the campaign rather than after."""
    spec = make_spec(k=3, order=ModelOrder.QUADRATIC, losses=2)
    design = C.box_behnken(3, n_center=1)
    assert evaluate(design, spec).estimable  # it works intact
    report = robustness_report(design, spec)
    assert report.applicable
    assert 0.0 < report.fraction_estimable < 1.0
    assert report.is_fragile


def test_barely_saturated_design_collapses_on_any_loss():
    """2^(5-1) with one centre point fits a 16-term interaction model using 17
    runs — one spare. Lose any two runs and nothing is estimable at all."""
    spec = make_spec(k=5, order=ModelOrder.INTERACTION, losses=2)
    report = robustness_report(C.fractional_factorial(5, 1, n_center=1), spec)
    assert report.applicable
    assert report.fraction_estimable == 0.0
    assert report.worst_d_ratio == 0.0


def test_robustness_samples_when_scenarios_explode():
    spec = make_spec(k=5, order=ModelOrder.MAIN, losses=3)
    report = robustness_report(C.full_factorial(5, n_center=4), spec)
    assert not report.exhaustive
    assert report.n_scenarios <= 400


# --------------------------------------------------------------------------
# Optimality criteria and the combined view
# --------------------------------------------------------------------------


def test_d_efficiency_is_one_for_a_saturated_orthogonal_design():
    """For an orthogonal two-level design fitting all its terms, X'X/n is the
    identity, so det(...)^(1/p) is exactly 1."""
    design = C.full_factorial(3, n_center=0)
    terms = model_terms(3, ModelOrder.INTERACTION)
    assert d_efficiency(design.matrix, terms) == pytest.approx(1.0)
    assert a_efficiency(design.matrix, terms) == pytest.approx(1.0)


def test_d_efficiency_is_zero_for_a_singular_design():
    design = C.fractional_factorial(4, 1, n_center=3)
    assert d_efficiency(design.matrix, model_terms(4, ModelOrder.INTERACTION)) == 0.0


def test_evaluate_is_deterministic():
    spec = make_spec(k=4)
    design = C.full_factorial(4, n_center=3)
    first, second = evaluate(design, spec), evaluate(design, spec)
    assert first.prediction.i_value == second.prediction.i_value
    assert first.robustness.fraction_estimable == second.robustness.fraction_estimable


def test_evaluate_flags_an_unfittable_design_consistently():
    spec = make_spec(k=4)
    props = evaluate(C.definitive_screening(4, n_center=3), spec)
    assert props.estimable is False
    assert props.d_efficiency == 0.0
    assert props.aliasing.estimable is False
    assert props.robustness.applicable is False


# --------------------------------------------------------------------------
# Power for every model term, not just the main effects
# --------------------------------------------------------------------------


def _ccd_quadratic_spec():
    """Face-centred CCD, 3 factors, 3 centre points (17 runs), target = 2 SD."""
    factors = [Factor(f"x{i}", -1.0, 1.0) for i in range(3)]
    responses = [Response("titer", target_effect=2.0, noise_sd=1.0)]
    return DesignSpec(factors=factors, responses=responses, model_order=ModelOrder.QUADRATIC, expected_run_losses=1)


def _independent_power(design, spec, column, coefficient):
    """Power for one coded coefficient, via statsmodels-free plain algebra."""
    x0 = design.matrix
    cols = [np.ones(len(x0))] + [x0[:, i] for i in range(3)]
    cols += [x0[:, 0] * x0[:, 1], x0[:, 0] * x0[:, 2], x0[:, 1] * x0[:, 2]]
    cols += [x0[:, i] ** 2 for i in range(3)]
    x = np.column_stack(cols)
    se = np.sqrt(np.linalg.inv(x.T @ x)[column, column])
    df = len(x0) - x.shape[1]
    crit = stats.t.ppf(0.975, df)
    ncp = coefficient / se
    return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)


def test_power_is_reported_for_interactions_and_curvature():
    spec = _ccd_quadratic_spec()
    design = C.central_composite(3, alpha="face", n_center=3)
    report = power_report(design, spec)

    assert len(report.per_term) == 3
    assert len(report.interaction_terms) == 3
    assert len(report.curvature_terms) == 3
    # Main and interaction coefficients are half the target; curvature is the
    # whole target (centre-to-edge rise), see CURVATURE_RULE.
    assert report.min_main_effect_power == pytest.approx(_independent_power(design, spec, 1, 1.0), abs=1e-6)
    assert report.min_interaction_power == pytest.approx(_independent_power(design, spec, 4, 1.0), abs=1e-6)
    assert report.min_curvature_power == pytest.approx(_independent_power(design, spec, 7, 2.0), abs=1e-6)


def test_headline_power_is_the_weakest_term():
    """The 17-run face-centred CCD: main effects look fine at 77%, but the
    interactions are at 68% — and the headline must say 68%."""
    report = power_report(C.central_composite(3, alpha="face", n_center=3), _ccd_quadratic_spec())
    assert report.min_main_effect_power == pytest.approx(0.774, abs=1e-3)
    assert report.min_interaction_power == pytest.approx(0.681, abs=1e-3)
    assert report.min_power == pytest.approx(report.min_interaction_power)


def test_detectable_effect_agrees_with_the_weakest_term():
    """Unequal SEs across terms: the detectable effect must be the one the
    weakest term needs, or it will claim margin the headline power denies."""
    spec = _ccd_quadratic_spec()
    design = C.central_composite(3, alpha="face", n_center=3)
    report = power_report(design, spec)
    assert (report.min_power >= 0.80) == (2.0 >= report.detectable_effect_sd)
    at_detectable = DesignSpec(
        factors=spec.factors,
        responses=[Response("titer", target_effect=report.detectable_effect_sd, noise_sd=1.0)],
        model_order=spec.model_order,
    )
    assert power_report(design, at_detectable).min_power == pytest.approx(0.80, abs=1e-3)


def test_main_effects_model_has_no_interaction_or_curvature_power():
    report = power_report(C.full_factorial(3, n_center=3), make_spec(k=3, order=ModelOrder.MAIN))
    assert report.interaction_terms == {}
    assert report.curvature_terms == {}
    assert report.min_power == report.min_main_effect_power


# --------------------------------------------------------------------------
# Robustness: still estimable is not still powered
# --------------------------------------------------------------------------


def test_robustness_reports_power_after_losing_a_run():
    spec = _ccd_quadratic_spec()
    design = C.central_composite(3, alpha="face", n_center=3)
    intact = power_report(design, spec).min_power
    report = robustness_report(design, spec)
    assert report.fraction_estimable == 1.0
    assert report.worst_power_after_loss is not None
    assert report.worst_power_after_loss < intact
    # Brute force: drop each run, take the weakest term, take the worst case.
    worst = 1.0
    for i in range(design.n_runs):
        kept = Design(design.name, design.family, np.delete(design.matrix, i, axis=0), design.factor_names)
        worst = min(worst, power_report(kept, spec).min_power)
    assert report.worst_power_after_loss == pytest.approx(worst, abs=1e-9)


def test_robustness_power_after_loss_is_none_without_a_target_effect():
    spec = make_spec(k=3, target=None, losses=1)
    assert robustness_report(C.full_factorial(3, n_center=3), spec).worst_power_after_loss is None


# --------------------------------------------------------------------------
# Noise stress: a guess gets 1.5x, a measurement gets its confidence bound
# --------------------------------------------------------------------------


def test_guessed_noise_is_stressed_by_1_5():
    from doe_advisor.designs.properties import noise_stress_factor

    assert noise_stress_factor(Response("y", noise_sd=1.0)) == 1.5
    assert noise_stress_factor(None) == 1.5


@pytest.mark.parametrize("df", [2, 3, 8, 30])
def test_measured_noise_is_stressed_by_its_upper_80_percent_bound(df):
    """Upper one-sided 80% bound of sigma from s with df: s * sqrt(df / chi2_0.20(df))."""
    from doe_advisor.designs.properties import noise_stress_factor

    expected = np.sqrt(df / stats.chi2.ppf(0.20, df))
    assert noise_stress_factor(Response("y", noise_sd=1.0, noise_df=df)) == pytest.approx(expected)


def test_power_report_uses_the_measured_stress_factor():
    design = C.full_factorial(3, n_center=3)
    factors = [Factor(f"x{i}", -1.0, 1.0) for i in range(3)]
    measured = DesignSpec(factors=factors, responses=[Response("y", target_effect=2.0, noise_sd=1.0, noise_df=3)])
    report = power_report(design, measured)
    k = report.noise_stress_factor
    assert k == pytest.approx(np.sqrt(3 / stats.chi2.ppf(0.20, 3)))
    worse = DesignSpec(factors=factors, responses=[Response("y", target_effect=2.0, noise_sd=k)])
    assert report.min_power_if_noise_high == pytest.approx(power_report(design, worse).min_power)
