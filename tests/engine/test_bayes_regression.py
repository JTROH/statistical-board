"""JZS Bayes factors for regression terms.

Reference values are R's BayesFactor 0.9.12-4.x ``linearReg.R2stat`` with
rscale="medium" (sqrt(2)/4), recorded here so the test needs no R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_board.engine import analyses
from stat_board.engine.bayes import bayesfactor_r2
from stat_board.engine.cli import main


@pytest.mark.parametrize(
    ("n", "p", "r2", "expected"),
    [
        (20, 3, 0.5, 6.3924974),
        (17, 9, 0.8, 1.5901582),
        (50, 2, 0.05, 0.29599919),
        (11, 7, 0.95, 2.6702397),
        (200, 4, 0.3, 321629044429.0),
        (12, 1, 0.01, 0.48270824),
    ],
)
def test_r2_bayes_factor_matches_r_bayesfactor(n, p, r2, expected):
    assert bayesfactor_r2(n, p, r2) == pytest.approx(expected, rel=1e-6)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        bayesfactor_r2(10, 0, 0.5)
    with pytest.raises(ValueError):
        bayesfactor_r2(10, 9, 0.5)
    with pytest.raises(ValueError):
        bayesfactor_r2(10, 2, 1.0)


@pytest.fixture
def two_factor_csv(tmp_path):
    rng = np.random.default_rng(8)
    a = np.tile([-1.0, 1.0, -1.0, 1.0], 6)
    b = np.repeat([-1.0, 1.0], 12)
    y = 10 + 3 * a + 0 * b + rng.normal(0, 1, a.size)  # a matters, b does not
    path = tmp_path / "ab.csv"
    pd.DataFrame({"a": a, "b": b, "y": y}).to_csv(path, index=False)
    return path


def test_per_term_bayes_factor_is_full_over_reduced(two_factor_csv):
    out = analyses.bayes_regression(str(two_factor_csv), "y ~ a + b")
    df = pd.read_csv(two_factor_csv)
    n = len(df)

    def r2(cols):
        x = np.column_stack([np.ones(n)] + [df[c] for c in cols])
        beta, *_ = np.linalg.lstsq(x, df["y"], rcond=None)
        resid = df["y"] - x @ beta
        return 1 - (resid**2).sum() / ((df["y"] - df["y"].mean()) ** 2).sum()

    full = bayesfactor_r2(n, 2, r2(["a", "b"]))
    terms = {t["term"]: t for t in out["terms"]}
    assert terms["a"]["bf10"] == pytest.approx(full / bayesfactor_r2(n, 1, r2(["b"])), rel=1e-9)
    assert terms["b"]["bf10"] == pytest.approx(full / bayesfactor_r2(n, 1, r2(["a"])), rel=1e-9)


def test_a_real_effect_and_a_null_effect_read_the_right_way(two_factor_csv):
    terms = {t["term"]: t for t in analyses.bayes_regression(str(two_factor_csv), "y ~ a + b")["terms"]}
    assert terms["a"]["bf10"] > 100
    assert terms["b"]["bf01"] > 3  # positive evidence that b does nothing
    assert "H0" in terms["b"]["interpretation"]


def test_runs_from_the_cli(two_factor_csv, capsys):
    assert main(["bayes-regression", "--data", str(two_factor_csv), "--formula", "y ~ a + b", "--r", "1"]) == 0
    assert '"prior_scale": 1.0' in capsys.readouterr().out
