"""Tests for the group-based engine analyses, using small synthetic fixtures with
known statistical properties (identical groups -> null result; clearly separated
groups -> significant result with a large effect size) rather than relying on
exact reference numbers from a stats package."""

from __future__ import annotations

import pytest

from stat_board.engine import analyses

IDENTICAL = {
    "A": [10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0, 10.0, 12.0],
    "B": [10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0, 10.0, 12.0],
}

SEPARATED = {
    "A": [10.0, 11.0, 12.0, 13.0, 10.0, 11.0, 12.0, 13.0, 10.0, 11.0],
    "B": [30.0, 31.0, 32.0, 33.0, 30.0, 31.0, 32.0, 33.0, 30.0, 31.0],
}

THREE_GROUP = {
    "A": [10.0, 11.0, 12.0, 13.0, 10.0, 11.0, 12.0, 13.0],
    "B": [11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0],
    "C": [30.0, 31.0, 32.0, 33.0, 30.0, 31.0, 32.0, 33.0],
}


def test_describe_reports_expected_shape_and_values():
    out = analyses.describe(IDENTICAL, alpha=0.05)
    assert out["analysis"] == "describe"
    a = out["groups"]["A"]
    assert a["n"] == 10
    assert a["mean"] == pytest.approx(11.4)
    assert a["min"] == 10.0
    assert a["max"] == 13.0
    assert len(a["ci95_mean"]) == 2


def test_check_assumptions_reports_normality_and_homogeneity():
    out = analyses.check_assumptions(SEPARATED, alpha=0.05)
    assert set(out["normality"]) == {"A", "B"}
    assert "levene" in out["homogeneity"]
    # identical shapes/variances -> Levene should not reject equal variance
    assert out["homogeneity"]["levene"]["equal_variance_at_alpha"] is True


def test_check_assumptions_small_group_skips_shapiro():
    out = analyses.check_assumptions({"A": [1.0, 2.0], "B": [3.0, 4.0, 5.0]})
    assert "note" in out["normality"]["A"]
    assert "shapiro_W" not in out["normality"]["A"]


@pytest.mark.parametrize("equal_var", [False, True])
def test_ttest_identical_groups_is_not_significant(equal_var):
    out = analyses.ttest(IDENTICAL, equal_var=equal_var, alpha=0.05)
    assert out["p"] == pytest.approx(1.0, abs=1e-9)
    assert out["significant"] is False
    assert out["mean_diff"] == pytest.approx(0.0)


def test_ttest_separated_groups_is_significant_with_large_effect():
    out = analyses.ttest(SEPARATED, alpha=0.05)
    assert out["significant"] is True
    assert out["p"] < 0.001
    assert abs(out["cohens_d"]) > 2
    assert out["effect_magnitude"] == "large"


def test_ttest_paired_requires_equal_length_groups():
    with pytest.raises(ValueError, match="paired"):
        analyses.ttest({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0]}, paired=True)


def test_ttest_paired_identical_groups():
    out = analyses.ttest(IDENTICAL, paired=True)
    assert out["variant"] == "paired"
    assert out["significant"] is False


def test_mann_whitney_separated_groups_is_significant():
    out = analyses.mann_whitney(SEPARATED)
    assert out["significant"] is True
    assert out["p"] < 0.01


def test_mann_whitney_identical_groups_is_not_significant():
    out = analyses.mann_whitney(IDENTICAL)
    assert out["significant"] is False


def test_anova_three_groups_detects_the_outlier_group():
    out = analyses.anova(THREE_GROUP)
    assert out["significant"] is True
    assert out["eta_squared"] > 0.5


def test_anova_requires_at_least_two_groups():
    with pytest.raises(ValueError, match="at least 2 groups"):
        analyses.anova({"A": [1.0, 2.0, 3.0]})


def test_welch_anova_three_groups_detects_the_outlier_group():
    out = analyses.welch_anova(THREE_GROUP)
    assert out["significant"] is True
    assert out["variant"].startswith("Welch")


def test_kruskal_three_groups_detects_the_outlier_group():
    out = analyses.kruskal(THREE_GROUP)
    assert out["significant"] is True


def test_tukey_posthoc_flags_the_outlier_pairs():
    out = analyses.tukey_posthoc(THREE_GROUP)
    pairs = {(p["group1"], p["group2"]): p for p in out["pairs"]}
    assert pairs[("A", "C")]["reject_null"] is True
    assert pairs[("A", "B")]["reject_null"] is False


def test_tost_identical_groups_are_equivalent_within_a_wide_margin():
    out = analyses.tost(IDENTICAL, margin_pct=50, alpha=0.05)
    assert out["equivalent"] is True


def test_tost_separated_groups_are_not_equivalent_within_a_narrow_margin():
    out = analyses.tost(SEPARATED, low=-1.0, high=1.0, alpha=0.05)
    assert out["equivalent"] is False


def test_tost_requires_bounds():
    with pytest.raises(ValueError, match="--low and --high"):
        analyses.tost(IDENTICAL)


def test_bayes_ttest_identical_groups_favors_the_null():
    out = analyses.bayes_ttest(IDENTICAL)
    assert out["BF10"] < 1


def test_bayes_ttest_separated_groups_favors_the_alternative():
    out = analyses.bayes_ttest(SEPARATED)
    assert out["BF10"] > 10


def test_correlation_pearson_perfect_linear_relationship():
    groups = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]}
    out = analyses.correlation(groups, method="pearson")
    assert out["r"] == pytest.approx(1.0)
    assert out["significant"] is True


def test_correlation_spearman_monotonic_relationship():
    groups = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 8.0, 9.0, 20.0, 21.0]}
    out = analyses.correlation(groups, method="spearman")
    assert out["r"] == pytest.approx(1.0)


def test_correlation_requires_equal_length_columns():
    with pytest.raises(ValueError, match="equal-length"):
        analyses.correlation({"x": [1.0, 2.0], "y": [1.0, 2.0, 3.0]})


def test_anova_table_matches_ss_decomposition_of_anova():
    table = analyses.anova_table(THREE_GROUP)
    aov = analyses.anova(THREE_GROUP)
    between_row = table["rows"][0]
    assert between_row["F"] == pytest.approx(aov["F"])
    assert table["goodness_of_fit"]["eta_squared"] == pytest.approx(aov["eta_squared"])


def test_residual_diagnostics_reports_centered_residuals():
    out = analyses.residual_diagnostics(THREE_GROUP)
    assert out["mean"] == pytest.approx(0.0, abs=1e-9)
    assert "normality" in out
    assert "homoscedasticity" in out


def test_pairwise_effects_covers_every_pair():
    out = analyses.pairwise_effects(THREE_GROUP)
    pairs = {p["pair"] for p in out["pairs"]}
    assert pairs == {"A vs B", "A vs C", "B vs C"}
