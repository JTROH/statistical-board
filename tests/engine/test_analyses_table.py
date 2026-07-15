from __future__ import annotations

import pytest

from stat_board.engine import analyses


def test_regression_on_strongly_correlated_columns(reg_csv):
    out = analyses.regression(str(reg_csv), "y ~ x")
    assert out["analysis"] == "regression"
    assert out["r_squared"] > 0.5
    assert "x" in out["coefficients"]
    assert out["coefficients"]["x"]["p"] < 0.001
    assert "durbin_watson" in out["residual_diagnostics"]


def test_two_way_anova_on_multifactor_data(multifactor_csv):
    out = analyses.two_way_anova(str(multifactor_csv), value="score", factors=["treatment", "sex"])
    assert out["analysis"] == "two_way_anova"
    terms = {t["term"] for t in out["anova"]}
    assert "C(treatment)" in terms
    assert "C(sex)" in terms
    assert "C(treatment):C(sex)" in terms
    assert "conclusion" in out


def test_two_way_anova_requires_at_least_two_factors(multifactor_csv):
    with pytest.raises(ValueError, match="at least 2 factors"):
        analyses.two_way_anova(str(multifactor_csv), value="score", factors=["treatment"])


def test_ancova_adjusts_for_covariate(multifactor_csv):
    out = analyses.ancova(
        str(multifactor_csv), value="score", factors=["treatment"], covariates=["age"]
    )
    assert out["analysis"] == "ancova"
    terms = {t["term"] for t in out["anova"]}
    assert "C(treatment)" in terms
    assert "age" in terms
    assert out["covariates"] == ["age"]


def test_ancova_requires_factors_and_covariates(multifactor_csv):
    with pytest.raises(ValueError, match="factor and one covariate"):
        analyses.ancova(str(multifactor_csv), value="score", factors=[], covariates=["age"])
    with pytest.raises(ValueError, match="factor and one covariate"):
        analyses.ancova(str(multifactor_csv), value="score", factors=["treatment"], covariates=[])


@pytest.fixture
def count_csv(tmp_path):
    """Synthetic count data with a clear, non-overdispersed Poisson relationship
    (rate roughly doubles from period 0 to period 1)."""
    p = tmp_path / "counts.csv"
    rows = ["period,count"]
    counts_period0 = [4, 5, 3, 6, 4, 5, 4, 6, 3, 5]
    counts_period1 = [9, 10, 8, 11, 9, 10, 9, 11, 8, 10]
    rows += [f"0,{c}" for c in counts_period0]
    rows += [f"1,{c}" for c in counts_period1]
    p.write_text("\n".join(rows) + "\n")
    return p


def test_poisson_regression_detects_the_rate_increase(count_csv):
    out = analyses.poisson_regression(str(count_csv), "count ~ period")
    assert out["analysis"] == "poisson_regression"
    assert out["coefficients"]["period"]["irr"] > 1.5
    assert out["coefficients"]["period"]["significant"] is True
    assert "overdispersion" in out


def test_negbin_regression_runs_on_the_same_data(count_csv):
    out = analyses.negbin_regression(str(count_csv), "count ~ period")
    assert out["analysis"] == "negbin_regression"
    assert "period" in out["coefficients"]


# --------------------------------------------------------------------------- #
# formula_for
# --------------------------------------------------------------------------- #

def test_formula_for_without_covariates_joins_factors_with_star():
    assert analyses.formula_for("score", ["treatment", "sex"]) == "score ~ C(treatment) * C(sex)"


def test_formula_for_with_covariates_joins_additively():
    assert (analyses.formula_for("score", ["treatment"], ["age"])
            == "score ~ C(treatment) + age")


def test_formula_for_matches_what_two_way_anova_and_ancova_actually_fit(multifactor_csv):
    tw = analyses.two_way_anova(str(multifactor_csv), value="score", factors=["treatment", "sex"])
    an = analyses.ancova(str(multifactor_csv), value="score", factors=["treatment"], covariates=["age"])
    assert tw["anova"] == analyses.regression(
        str(multifactor_csv), analyses.formula_for("score", ["treatment", "sex"]))["anova"]
    assert an["anova"] == analyses.regression(
        str(multifactor_csv), analyses.formula_for("score", ["treatment"], ["age"]))["anova"]


# --------------------------------------------------------------------------- #
# predict_table / vif_table
# --------------------------------------------------------------------------- #

def test_predict_table_reports_per_row_diagnostics(reg_csv):
    out = analyses.predict_table(str(reg_csv), "y ~ x")
    assert out["analysis"] == "predict_table"
    assert out["n"] == len(out["rows"])
    assert out["leverage_threshold"] == pytest.approx(2 * 2 / out["n"])
    assert out["cooks_d_threshold"] == pytest.approx(4 / out["n"])
    row = out["rows"][0]
    assert set(["observed", "predicted", "residual", "standardized_residual",
               "leverage", "cooks_d", "high_leverage", "influential", "data"]) <= row.keys()
    assert row["residual"] == pytest.approx(row["observed"] - row["predicted"])
    assert out["n_flagged"] == len(out["flagged_rows"])


def test_vif_table_reports_low_vif_for_a_single_predictor(reg_csv):
    out = analyses.vif_table(str(reg_csv), "y ~ x")
    assert out["analysis"] == "vif"
    assert len(out["terms"]) == 1
    assert out["terms"][0]["term"] == "x"
    assert out["terms"][0]["vif"] == pytest.approx(1.0, abs=0.05)
    assert out["terms"][0]["collinearity_concern"] is False


def test_vif_table_flags_collinear_terms(tmp_path):
    # x2 is a near-exact linear function of x1 -> severe collinearity.
    p = tmp_path / "collinear.csv"
    rows = ["x1,x2,y"]
    for i in range(20):
        x1 = i
        x2 = 2 * i + 0.001 * (i % 3)  # tiny perturbation so it's not perfectly singular
        y = x1 + x2 + (i % 5)
        rows.append(f"{x1},{x2},{y}")
    p.write_text("\n".join(rows) + "\n")
    out = analyses.vif_table(str(p), "y ~ x1 + x2")
    assert any(t["collinearity_concern"] for t in out["terms"])


# --------------------------------------------------------------------------- #
# design_coverage / doe_optimum -- deterministic 2-factor-with-center-points DoE
# --------------------------------------------------------------------------- #

@pytest.fixture
def doe_csv(tmp_path):
    """A noiseless 2^2 factorial (each corner replicated twice) plus 3 center
    points with a deliberate +0.5 curvature bump -- every statistic below is
    hand-computable, so assertions can be exact rather than approximate."""
    p = tmp_path / "doe.csv"
    corners = [(10, 100), (10, 200), (20, 100), (20, 200)]
    rows = ["x1,x2,y"]
    for x1, x2 in corners:
        y = 5 + 0.3 * x1 + 0.01 * x2
        rows += [f"{x1},{x2},{y}"] * 2
    center_y = 5 + 0.3 * 15 + 0.01 * 150 + 0.5
    rows += [f"15,150,{center_y}"] * 3
    p.write_text("\n".join(rows) + "\n")
    return p


def test_design_coverage_counts_combinations_and_replicates(doe_csv):
    out = analyses.design_coverage(str(doe_csv), ["x1", "x2"], value="y")
    assert out["analysis"] == "design_coverage"
    assert out["n_possible_combinations"] == 9  # 3 levels x 3 levels
    assert out["n_tested_combinations"] == 5  # 4 corners + 1 center combo
    assert out["coverage_fraction"] == pytest.approx(5 / 9)
    assert out["full_grid_enumerated"] is True
    assert len(out["missing_combinations"]) == 4
    assert out["replicate_counts"]["10|100"] == 2
    assert out["replicate_counts"]["15|150"] == 3


def test_design_coverage_detects_center_points_and_curvature(doe_csv):
    out = analyses.design_coverage(str(doe_csv), ["x1", "x2"], value="y")
    assert out["n_center_points"] == 3
    curvature = out["curvature"]
    assert curvature is not None
    assert curvature["corner_mean"] == pytest.approx(11.0)
    assert curvature["center_mean"] == pytest.approx(11.5)
    assert curvature["difference"] == pytest.approx(-0.5)
    assert curvature["center_point_df"] == 2
    assert curvature["low_power_warning"] is True  # only 2 df of pure error


def test_design_coverage_without_value_skips_curvature(doe_csv):
    out = analyses.design_coverage(str(doe_csv), ["x1", "x2"])
    assert out["n_center_points"] == 3
    assert out["curvature"] is None


def test_design_coverage_categorical_factors_have_no_center_points(multifactor_csv):
    out = analyses.design_coverage(str(multifactor_csv), ["treatment", "sex"], value="score")
    assert out["n_center_points"] == 0
    assert out["curvature"] is None


def test_doe_optimum_ranks_by_predicted_response_and_flags_boundary(doe_csv):
    out = analyses.doe_optimum(str(doe_csv), "y ~ x1 + x2", ["x1", "x2"], "y")
    assert out["analysis"] == "doe_optimum"
    assert len(out["rows"]) == 9  # full grid of 3x3 observed levels
    best = out["best"]
    assert best["x1"] == 20 and best["x2"] == 200  # response increases in both
    assert best["tested"] is True
    assert out["rows"] == sorted(out["rows"], key=lambda r: r["predicted"], reverse=True)
    assert out["boundary_flags"] == {"x1": "boundary", "x2": "boundary"}


def test_doe_optimum_flags_interior_when_best_is_the_center_point(doe_csv):
    # With the curvature bump, the CENTER point (15, 150) is the single highest
    # OBSERVED value -- but the fitted linear model still predicts the corner
    # (20, 200) highest, since the model has no quadratic term to capture the
    # bump. This exercises the "tested but not the model's pick" distinction.
    out = analyses.doe_optimum(str(doe_csv), "y ~ x1 + x2", ["x1", "x2"], "y")
    center_row = next(r for r in out["rows"] if r["x1"] == 15 and r["x2"] == 150)
    assert center_row["tested"] is True
    assert center_row["observed_mean"] == pytest.approx(11.5)
