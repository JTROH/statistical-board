"""Tests for the desktop GUI's test-recommendation logic, using handcrafted
fixtures with known assumption-check outcomes (mirrors the style of
tests/engine/test_analyses_groups.py) rather than relying on randomness."""

from __future__ import annotations

import pytest

from desktop_gui import recommend

# Tight, mildly-jittered spread -> Shapiro does not reject normality.
A = [10.1, 9.8, 10.3, 9.9, 10.2, 9.7, 10.4, 9.6, 10.0, 10.1]
B = [12.0, 11.7, 12.2, 11.8, 12.1, 11.6, 12.3, 11.5, 11.9, 12.0]
E = [14.0, 13.7, 14.2, 13.8, 14.1, 13.6, 14.3, 13.5, 13.9, 14.0]
# Same shape as A/B/E but spread much wider -> normal, but Levene rejects
# equal variance against A/B/E.
C = [5.0, 15.0, 8.0, 18.0, 6.0, 16.0, 9.0, 17.0, 7.0, 14.0]
# One extreme outlier -> Shapiro strongly rejects normality.
D = [1.0, 1.1, 1.2, 1.0, 1.1, 1.3, 1.0, 1.2, 1.1, 50.0]


def test_two_group_normal_equal_variance_picks_student_t():
    out = recommend.analyze({"A": A, "B": B})
    assert out["chosen_test"] == "ttest"
    assert out["result"]["variant"] == "Student (pooled)"
    assert "bayes_ttest" in out["supplementary"]
    assert "tukey_posthoc" not in out["supplementary"]


def test_two_group_normal_unequal_variance_picks_welch_t():
    out = recommend.analyze({"A": A, "C": C})
    assert out["chosen_test"] == "ttest"
    assert out["result"]["variant"] == "Welch"


def test_two_group_non_normal_picks_mann_whitney():
    out = recommend.analyze({"A": A, "D": D})
    assert out["chosen_test"] == "mann_whitney"
    assert out["caveats"] == [] or all("outlier" in c for c in out["caveats"])


def test_three_group_normal_equal_variance_picks_anova_with_tukey():
    out = recommend.analyze({"A": A, "B": B, "E": E})
    assert out["chosen_test"] == "anova"
    assert "tukey_posthoc" in out["supplementary"]
    assert "bayes_ttest" not in out["supplementary"]


def test_three_group_normal_unequal_variance_picks_welch_anova_with_tukey():
    out = recommend.analyze({"A": A, "B": B, "C": C})
    assert out["chosen_test"] == "welch_anova"
    assert "tukey_posthoc" in out["supplementary"]


def test_three_group_non_normal_picks_kruskal_without_tukey():
    out = recommend.analyze({"A": A, "B": B, "D": D})
    assert out["chosen_test"] == "kruskal"
    assert "tukey_posthoc" not in out["supplementary"]


def test_small_group_falls_back_to_nonparametric_with_caveat():
    out = recommend.analyze({"A": [1.0, 2.0], "B": [3.0, 4.0, 5.0, 3.5, 4.5]})
    assert out["chosen_test"] == "mann_whitney"
    assert any("too few" in c for c in out["caveats"])


def test_analyze_requires_at_least_two_datasets():
    with pytest.raises(ValueError):
        recommend.analyze({"A": A})


@pytest.mark.parametrize(
    "groups",
    [
        {"A": A, "B": B},
        {"A": A, "C": C},
        {"A": A, "D": D},
        {"A": A, "B": B, "E": E},
        {"A": A, "B": B, "C": C},
        {"A": A, "B": B, "D": D},
    ],
)
def test_plain_summary_mentions_the_chosen_tests_conclusion(groups):
    out = recommend.analyze(groups)
    summary = recommend.plain_summary(out)
    assert out["result"]["conclusion"] in summary
    assert isinstance(summary, str) and len(summary) > 0
