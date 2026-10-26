"""Split-plot designs: generation, GLS scoring and the two error strata.

Each piece is checked against something computed independently:

- the shortcut for ``X' V^-1 X`` against the dense ``V = I + eta Z Z'``;
- the whole-plot factor's power against the closed form for a balanced
  split-plot, where its coefficient variance is ``sigma^2 (1 + m eta) / n``
  and it is tested on ``g - 2`` degrees of freedom;
- the search against the textbook design (the whole-plot factor balanced over
  the plots, a full factorial of the subplot factors inside each);
- and the search's own answer by brute force: no single change of one
  whole-plot setting or one subplot setting may improve it.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy import stats

from doe_advisor.designs.model import (
    cross_information,
    information,
    model_matrix,
    model_terms,
    stratum_df,
)
from doe_advisor.designs.optimal import default_grid
from doe_advisor.designs.properties import evaluate, power_report, region_samples
from doe_advisor.designs.spec import Design, DesignSpec, Factor, LinearConstraint, ModelOrder, Response
from doe_advisor.designs.split_plot import plot_sizes_for, split_plot_design

RESPONSE = Response("titer", target_effect=1.0, noise_sd=0.5)


def textbook_spec(eta: float = 1.0, order: ModelOrder = ModelOrder.INTERACTION) -> DesignSpec:
    return DesignSpec(
        [Factor("temp", 35, 37, hard_to_change=True), Factor("pH", 6.8, 7.2), Factor("feed", 2, 6)],
        responses=[RESPONSE], model_order=order, wp_variance_ratio=eta,
    )


def textbook_design() -> Design:
    """4 whole plots of 4: temp -, +, -, +; a 2^2 of pH x feed inside each."""
    rows, labels = [], []
    for g, temp in enumerate([-1.0, 1.0, -1.0, 1.0]):
        for ph, feed in product([-1.0, 1.0], repeat=2):
            rows.append([temp, ph, feed])
            labels.append(g)
    return Design("textbook", "test", np.array(rows), ["temp", "pH", "feed"], whole_plots=np.array(labels))


# --------------------------------------------------------------------------
# GLS information and the strata
# --------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0, 4.0])
def test_information_matches_the_dense_covariance(eta):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(11, 4))
    x2 = rng.normal(size=(11, 3))
    plots = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    z = (plots[:, None] == np.arange(4)[None, :]).astype(float)
    v_inv = np.linalg.inv(np.eye(11) + eta * z @ z.T)
    assert np.allclose(information(x, plots, eta), x.T @ v_inv @ x)
    assert np.allclose(cross_information(x, x2, plots, eta), x.T @ v_inv @ x2)


def test_stratum_df_follow_the_containment_rule():
    design = textbook_design()
    terms = model_terms(3, ModelOrder.INTERACTION)
    df_wp, df_sp, flags = stratum_df(model_matrix(design.matrix, terms), terms, design.whole_plots)
    # whole plot: intercept + temp in 4 plots; subplot: pH, feed and the three 2FIs in 16 - 4 runs.
    assert (df_wp, df_sp) == (2, 16 - 4 - 5)
    assert flags == [True, True, False, False, False, False, False]


def test_whole_plot_power_matches_the_balanced_closed_form():
    spec, design = textbook_spec(eta=1.0), textbook_design()
    report = power_report(design, spec)
    assert (report.whole_plot_df, report.subplot_df) == (2, 7)

    def power(var_coef, df):
        ncp = (RESPONSE.target_effect / 2) / (RESPONSE.noise_sd * np.sqrt(var_coef))
        crit = stats.t.ppf(0.975, df)
        return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)

    n, m, eta = 16, 4, 1.0
    assert report.per_term["temp"] == pytest.approx(power((1 + m * eta) / n, 2), rel=1e-9)
    # A subplot factor is orthogonal to the whole plots, so it keeps the CRD variance 1/n.
    assert report.per_term["pH"] == pytest.approx(power(1 / n, 7), rel=1e-9)


def test_without_whole_plot_variance_scores_like_a_randomised_design():
    design = textbook_design()
    flat = Design("flat", "test", design.matrix, design.factor_names)
    a = evaluate(design, textbook_spec(eta=0.0))
    b = evaluate(flat, DesignSpec([Factor(n, -1, 1) for n in ("temp", "pH", "feed")], responses=[RESPONSE]))
    assert a.d_efficiency == pytest.approx(b.d_efficiency)
    assert a.prediction.i_value == pytest.approx(b.prediction.i_value)


def test_whole_plot_variance_costs_precision_only_on_whole_plot_terms():
    design = textbook_design()
    low, high = evaluate(design, textbook_spec(eta=0.1)), evaluate(design, textbook_spec(eta=4.0))
    assert high.power.per_term["temp"] < low.power.per_term["temp"]
    assert high.power.per_term["pH"] == pytest.approx(low.power.per_term["pH"])


def test_randomised_order_keeps_whole_plots_together():
    design = textbook_design()
    order = design.randomised_order(seed=5)
    plots = design.whole_plots[order]
    changes = int((np.diff(plots) != 0).sum())
    assert changes == design.n_whole_plots - 1


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def test_finds_the_textbook_split_plot():
    spec = textbook_spec()
    design = split_plot_design(spec, 16, 4)
    assert design.family == "split_plot" and design.n_whole_plots == 4
    for g in range(4):
        rows = design.matrix[design.whole_plots == g]
        assert np.unique(rows[:, 0]).size == 1  # temp constant within the plot
        assert sorted(map(tuple, rows[:, 1:])) == sorted(product([-1.0, 1.0], repeat=2))
    temps = [design.matrix[design.whole_plots == g][0, 0] for g in range(4)]
    assert sorted(temps) == [-1.0, -1.0, 1.0, 1.0]


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_no_single_change_improves_the_answer(criterion):
    spec = DesignSpec(
        [Factor("temp", 35, 37, hard_to_change=True), Factor("pH", 6.8, 7.2), Factor("feed", 2, 6),
         Factor("DO", 30, 60)],
        responses=[RESPONSE], model_order=ModelOrder.QUADRATIC, wp_variance_ratio=1.0,
    )
    design = split_plot_design(spec, 24, 6, criterion=criterion, seed=1)
    terms = model_terms(4, ModelOrder.QUADRATIC)
    region = region_samples(spec)[0]
    f_region = model_matrix(region, terms)
    moments = f_region.T @ f_region / f_region.shape[0]

    def score(matrix):
        m = information(model_matrix(matrix, terms), design.whole_plots, 1.0)
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0:
            return -np.inf
        return logdet if criterion == "D" else -np.trace(np.linalg.solve(m, moments))

    best = score(design.matrix)
    grid = default_grid(ModelOrder.QUADRATIC)
    for g in range(design.n_whole_plots):  # whole-plot moves
        for v in grid:
            trial = design.matrix.copy()
            trial[design.whole_plots == g, 0] = v
            assert score(trial) <= best + 1e-7 * abs(best)
    for i, j in product(range(design.n_runs), [1, 2, 3]):  # subplot moves
        for v in grid:
            trial = design.matrix.copy()
            trial[i, j] = v
            assert score(trial) <= best + 1e-7 * abs(best)


def test_categorical_hard_to_change_factor_and_constraints_are_respected():
    spec = DesignSpec(
        [Factor("media", levels=("A", "B", "C"), hard_to_change=True), Factor("feed", 2, 6),
         Factor("bolus", 0, 4)],
        responses=[RESPONSE], model_order=ModelOrder.INTERACTION,
        constraints=[LinearConstraint({"feed": 1.0, "bolus": 1.0}, 8.0)],
    )
    design = split_plot_design(spec, 24, 6)
    assert spec.feasible(design.matrix).all()
    for g in range(6):
        assert np.unique(design.matrix[design.whole_plots == g, 0]).size == 1
    props = evaluate(design, spec)
    assert props.estimable and props.power.whole_plot_df == 6 - 3  # intercept + 2 media columns


def test_robustness_and_aliasing_run_on_a_split_plot():
    props = evaluate(split_plot_design(textbook_spec(), 16, 4), textbook_spec())
    assert props.robustness.applicable and 0 < props.robustness.fraction_estimable <= 1
    assert props.aliasing.estimable


def test_the_search_is_deterministic_for_a_seed():
    a = split_plot_design(textbook_spec(), 20, 5, seed=2)
    b = split_plot_design(textbook_spec(), 20, 5, seed=2)
    assert np.array_equal(a.matrix, b.matrix) and np.array_equal(a.whole_plots, b.whole_plots)


# --------------------------------------------------------------------------
# What is refused
# --------------------------------------------------------------------------


def test_every_factor_hard_to_change_is_refused():
    with pytest.raises(ValueError, match="at least one factor as easy to change"):
        DesignSpec([Factor("a", 0, 1, hard_to_change=True), Factor("b", 0, 1, hard_to_change=True)])


def test_blocks_and_whole_plots_together_are_refused():
    with pytest.raises(ValueError, match="whole plots instead of blocks"):
        DesignSpec([Factor("a", 0, 1, hard_to_change=True), Factor("b", 0, 1)], n_blocks=2)


def test_plot_sizes_are_balanced_and_bounded():
    assert plot_sizes_for(14, 4) == [4, 4, 3, 3]
    with pytest.raises(ValueError):
        plot_sizes_for(5, 6)
