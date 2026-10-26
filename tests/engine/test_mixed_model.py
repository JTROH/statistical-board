"""Mixed-model (split-plot) analysis, with Kenward-Roger tests.

Two independent checks:

- The classical split-plot ANOVA. In a balanced split-plot the exact F test
  for the whole-plot factor is the ordinary F test on the whole-plot means,
  and the within-plot tests are those of an OLS fit with the whole plot as a
  fixed factor. Kenward-Roger must reproduce both exactly — F, denominator df
  and no scaling — including for a multi-df (categorical) whole-plot factor.
- Satterthwaite. For a single-df contrast the Kenward-Roger df reduces
  algebraically to the Satterthwaite df (A1 = A2 = A, so m = 2/A). The two are
  coded separately here, so on unbalanced data they must agree to rounding.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from stat_board.engine import analyses
from stat_board.engine.cli import main
from stat_board.engine_tool import build_tool, make_executor


def _split_plot(tmp_path, wp_sd=1.0, noise=0.5, seed=1, n_plots=8):
    """Whole-plot factor A (alternating by plot), subplot factor B (2 x 2 per plot)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_plots):
        a = 1 if g % 2 else -1
        offset = rng.normal(0, wp_sd)
        for b in (-1, -1, 1, 1):
            y = 10 + 1.0 * a + 0.8 * b + 0.5 * a * b + offset + rng.normal(0, noise)
            rows.append({"whole_plot": g + 1, "A": a, "B": b, "y": y})
    path = tmp_path / "sp.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, pd.DataFrame(rows)


def test_f_tests_match_the_classical_split_plot_anova(tmp_path):
    path, df = _split_plot(tmp_path)
    out = analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")
    terms = {t["term"]: t for t in out["terms"]}

    means = df.groupby("whole_plot").agg(y=("y", "mean"), A=("A", "first"))
    f_a = anova_lm(smf.ols("y ~ A", means).fit()).loc["A", "F"]
    within = anova_lm(smf.ols("y ~ C(whole_plot) + B + A:B", df).fit(), typ=2)

    assert out["df_method"] == "Kenward-Roger"
    assert terms["A"]["F"] == pytest.approx(f_a, rel=1e-3)
    assert terms["A"]["den_df"] == pytest.approx(8 - 2, rel=1e-6) and terms["A"]["stratum"] == "whole plot"
    assert terms["B"]["F"] == pytest.approx(within.loc["B", "F"], rel=1e-3)
    assert terms["A:B"]["F"] == pytest.approx(within.loc["A:B", "F"], rel=1e-3)
    assert terms["B"]["den_df"] == pytest.approx(32 - 8 - 2, rel=1e-6) and terms["B"]["stratum"] == "within plot"
    # Balanced: Kenward-Roger needs no scaling and agrees with containment.
    assert terms["B"]["kr_scale"] == pytest.approx(1.0, abs=1e-9)
    assert terms["A"]["containment"]["den_df"] == 6 and terms["B"]["containment"]["den_df"] == 22


def test_kenward_roger_is_exact_for_a_multi_df_whole_plot_factor(tmp_path):
    """A three-level categorical hard-to-change factor: a 2-df whole-plot term."""
    rng = np.random.default_rng(11)
    rows = []
    for g in range(9):
        media = "ABC"[g % 3]
        offset = rng.normal(0, 1.0)
        for b in (-1, -1, 1, 1):
            y = 5 + {"A": 0.0, "B": 1.0, "C": -0.5}[media] + 0.7 * b + offset + rng.normal(0, 0.4)
            rows.append({"whole_plot": g, "media": media, "B": b, "y": y})
    df = pd.DataFrame(rows)
    path = tmp_path / "cat.csv"
    df.to_csv(path, index=False)
    terms = {t["term"]: t for t in analyses.mixed_model(str(path), "y ~ C(media) + B", "whole_plot")["terms"]}

    means = df.groupby("whole_plot").agg(y=("y", "mean"), media=("media", "first"))
    f_media = anova_lm(smf.ols("y ~ C(media)", means).fit()).loc["C(media)", "F"]
    assert terms["C(media)"]["num_df"] == 2
    assert terms["C(media)"]["F"] == pytest.approx(f_media, rel=1e-3)
    assert terms["C(media)"]["den_df"] == pytest.approx(9 - 3, rel=1e-6)


def test_single_df_kenward_roger_equals_satterthwaite_on_unbalanced_data(tmp_path):
    rng = np.random.default_rng(5)
    for trial in range(8):
        rows = []
        for g in range(int(rng.integers(4, 9))):
            a = int(rng.choice([-1, 1]))
            offset = rng.normal(0, rng.uniform(0.2, 2.0))
            for _ in range(int(rng.integers(2, 6))):  # unequal whole-plot sizes
                b = rng.uniform(-1, 1)
                rows.append({"whole_plot": g, "A": a, "B": b, "y": 1 + a + b + offset + rng.normal(0, 0.5)})
        path = tmp_path / f"u{trial}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        out = analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")
        assert out["df_method"] == "Kenward-Roger"
        for c in out["coefficients"].values():
            assert c["df"] == pytest.approx(c["df_satterthwaite"], rel=1e-9)


def test_unbalanced_data_get_their_own_df_with_containment_alongside(tmp_path):
    path, df = _split_plot(tmp_path, n_plots=6)
    df = df.drop(index=[0, 13, 18])  # three lost runs
    df.to_csv(path, index=False)
    out = analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")
    assert not out["balanced"]
    b = next(t for t in out["terms"] if t["term"] == "B")
    assert b["den_df"] != pytest.approx(b["containment"]["den_df"], abs=1e-6)
    assert b["kr_scale"] is not None and b["containment"]["p"] > 0


def test_variance_components_are_recovered(tmp_path):
    path, _ = _split_plot(tmp_path, wp_sd=1.5, noise=0.4, seed=7, n_plots=40)
    vc = analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")["variance_components"]
    assert vc["group_sd"] == pytest.approx(1.5, rel=0.3)
    assert vc["residual_sd"] == pytest.approx(0.4, rel=0.2)
    assert 0 < vc["intraclass_correlation"] < 1


def test_ols_would_overstate_the_whole_plot_factor(tmp_path):
    """The reason the command exists: a regression ignoring whole plots gives A far too small a p-value."""
    path, df = _split_plot(tmp_path, wp_sd=2.0, seed=4)
    mixed = {t["term"]: t for t in analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")["terms"]}
    ols = smf.ols("y ~ A * B", df).fit()
    assert mixed["A"]["p"] > 5 * float(ols.pvalues["A"])


def test_no_whole_plot_variance_is_reported_not_hidden(tmp_path):
    path, _ = _split_plot(tmp_path, wp_sd=0.0, seed=2)
    out = analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")
    a = next(t for t in out["terms"] if t["term"] == "A")
    if out["variance_at_boundary"]:
        assert any("estimated at zero" in n for n in out["notes"])
    # In a balanced split-plot the whole-plot-means test is exact whatever the
    # variance estimate, and Kenward-Roger reproduces it: A stays on 6 df even
    # when the data show next to no whole-plot variance.
    assert a["den_df"] == pytest.approx(6, rel=1e-6)
    assert out["df_whole_plot"] == 6


def test_kenward_roger_df_track_how_much_whole_plot_variance_there_is(tmp_path):
    """Unequal plot sizes and a continuous subplot factor: with little whole-plot
    variance a subplot term also borrows between-plot information (more df than
    containment); with a lot, both strata settle on their containment df."""
    def fit(wp_sd):
        rng = np.random.default_rng(1)
        rows = []
        for g in range(8):
            a = 1 if g % 2 else -1
            offset = rng.normal(0, wp_sd)
            for _ in range(2 + g % 4):
                b = rng.uniform(-1, 1)
                rows.append({"whole_plot": g, "A": a, "B": b, "y": a + b + offset + rng.normal(0, 0.5)})
        path = tmp_path / f"wp{wp_sd}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return {t["term"]: t for t in analyses.mixed_model(str(path), "y ~ A * B", "whole_plot")["terms"]}

    small, large = fit(0.3), fit(3.0)
    assert small["B"]["den_df"] > small["B"]["containment"]["den_df"] + 2
    assert large["B"]["den_df"] == pytest.approx(large["B"]["containment"]["den_df"], abs=1.0)
    assert large["A"]["den_df"] == pytest.approx(large["A"]["containment"]["den_df"], abs=0.2)


def test_a_missing_group_column_is_a_clear_error(tmp_path):
    path, _ = _split_plot(tmp_path)
    with pytest.raises(ValueError, match="no column called 'plot'"):
        analyses.mixed_model(str(path), "y ~ A * B", "plot")


def test_runs_from_the_cli(tmp_path, capsys):
    path, _ = _split_plot(tmp_path)
    assert main(["mixed-model", "--data", str(path), "--formula", "y ~ A * B", "--groups", "whole_plot"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "mixed_model" and out["n_groups"] == 8


def test_the_board_can_call_it(tmp_path):
    path, _ = _split_plot(tmp_path)
    assert "mixed-model" in build_tool()["input_schema"]["properties"]["command"]["enum"]
    out = json.loads(make_executor(str(path))({"command": "mixed-model", "formula": "y ~ A * B",
                                               "groups": "whole_plot"}))
    assert out["df_whole_plot"] == 6
