"""Lack of fit, constant variance and Box-Cox — each checked against an
independent route to the same number."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stat_board.engine import analyses
from stat_board.engine.cli import main


@pytest.fixture
def curved_ccd_csv(tmp_path):
    """A 2-factor face-centred CCD with 4 centre points and real curvature in x1."""
    rng = np.random.default_rng(7)
    pts = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)] + [(0, 0)] * 4
    x = np.array(pts, float)
    y = 50 + 4 * x[:, 0] + 2 * x[:, 1] - 8 * x[:, 0] ** 2 + rng.normal(0, 1.0, len(x))
    path = tmp_path / "ccd.csv"
    pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y}).to_csv(path, index=False)
    return path


def test_lack_of_fit_matches_the_nested_model_comparison(curved_ccd_csv):
    """Lack of fit is the F test of the linear model against the cell-means
    model (one mean per distinct setting). statsmodels' nested ANOVA gives that
    directly."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    out = analyses.regression(str(curved_ccd_csv), "y ~ x1 + x2")
    lof = out["lack_of_fit"]

    df = pd.read_csv(curved_ccd_csv)
    df["setting"] = df["x1"].astype(str) + "|" + df["x2"].astype(str)
    reduced = smf.ols("y ~ x1 + x2", data=df).fit()
    full = smf.ols("y ~ C(setting)", data=df).fit()
    ref = anova_lm(reduced, full)

    assert lof["df_pure_error"] == 3
    assert lof["df_lack_of_fit"] == 6  # 9 distinct settings - 3 parameters
    assert lof["F"] == pytest.approx(float(ref["F"].iloc[1]), rel=1e-9)
    assert lof["p"] == pytest.approx(float(ref["Pr(>F)"].iloc[1]), rel=1e-9)
    assert lof["lack_of_fit_at_alpha"] is True  # the missing x1^2 must show


def test_lack_of_fit_clears_once_the_curvature_is_in_the_model(curved_ccd_csv):
    out = analyses.regression(str(curved_ccd_csv), "y ~ x1 + x2 + I(x1**2)")
    assert out["lack_of_fit"]["lack_of_fit_at_alpha"] is False


def test_lack_of_fit_is_none_without_replicates(reg_csv):
    assert analyses.regression(str(reg_csv), "y ~ x")["lack_of_fit"] is None


def test_breusch_pagan_flags_noise_that_grows_with_x(tmp_path):
    rng = np.random.default_rng(3)
    x = np.linspace(1, 10, 80)
    y = 3 * x + rng.normal(0, 1, x.size) * x  # SD proportional to x
    path = tmp_path / "het.csv"
    pd.DataFrame({"x": x, "y": y}).to_csv(path, index=False)
    diag = analyses.regression(str(path), "y ~ x")["residual_diagnostics"]
    assert diag["constant_variance_at_alpha"] is False


def test_box_cox_matches_scipy_for_an_intercept_only_model(tmp_path):
    rng = np.random.default_rng(11)
    y = np.exp(rng.normal(3, 0.5, 60))  # lognormal: the right answer is near 0
    path = tmp_path / "pos.csv"
    pd.DataFrame({"y": y}).to_csv(path, index=False)
    out = analyses.box_cox(str(path), "y ~ 1")
    _, lam = stats.boxcox(y)
    assert out["lambda"] == pytest.approx(lam, abs=0.01)
    assert out["includes_0"] is True
    assert out["includes_1"] is False
    assert "log" in out["advice"]


def test_box_cox_refuses_a_non_positive_response(curved_ccd_csv, tmp_path):
    df = pd.read_csv(curved_ccd_csv)
    df.loc[0, "y"] = -1.0
    path = tmp_path / "neg.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly positive"):
        analyses.box_cox(str(path), "y ~ x1 + x2")


def test_box_cox_runs_from_the_cli(curved_ccd_csv, capsys):
    assert main(["box-cox", "--data", str(curved_ccd_csv), "--formula", "y ~ x1 + x2 + I(x1**2)"]) == 0
    assert '"analysis": "box_cox"' in capsys.readouterr().out
