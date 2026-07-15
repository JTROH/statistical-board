from __future__ import annotations

import pytest

from stat_board.engine import analyses


def test_chi_square_independent_table_is_not_significant():
    # perfectly proportional rows -> independence
    table = [[10, 10], [10, 10]]
    out = analyses.chi_square(table)
    assert out["significant"] is False
    assert out["p"] == pytest.approx(1.0)


def test_chi_square_associated_table_is_significant():
    table = [[50, 5], [5, 50]]
    out = analyses.chi_square(table)
    assert out["significant"] is True
    assert out["cramers_v"] > 0.5


def test_power_analysis_solves_for_n_given_power():
    out = analyses.power_analysis(test="ttest", effect_size=0.8, power=0.8, alpha=0.05)
    assert "required_n_per_group" in out
    assert out["required_n_per_group"] > 0


def test_power_analysis_solves_for_power_given_n():
    out = analyses.power_analysis(test="ttest", effect_size=0.8, n=26, alpha=0.05)
    assert "achieved_power" in out
    assert 0 < out["achieved_power"] <= 1


def test_power_analysis_anova_requires_k_groups():
    with pytest.raises(ValueError, match="k-groups"):
        analyses.power_analysis(test="anova", effect_size=0.25, power=0.8)


def test_power_analysis_anova_solves_for_total_n():
    out = analyses.power_analysis(test="anova", effect_size=0.25, power=0.8, k_groups=3)
    assert "required_total_n" in out
    assert out["k_groups"] == 3


def test_power_analysis_requires_exactly_one_of_n_or_power_unset():
    with pytest.raises(ValueError, match="exactly one"):
        analyses.power_analysis(effect_size=0.8, n=20, power=0.8)
    with pytest.raises(ValueError, match="exactly one"):
        analyses.power_analysis(effect_size=0.8)


@pytest.mark.parametrize("method", ["fdr_bh", "bonferroni", "holm"])
def test_correct_pvalues_flags_the_clearly_significant_one(method):
    out = analyses.correct_pvalues([0.001, 0.4, 0.9], method=method)
    assert out["method"] == method
    assert out["reject_null"][0] is True
    assert out["n_significant"] >= 1
    assert len(out["adjusted_p"]) == 3
