"""Canonical analysis: the stationary point of a fitted quadratic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_board.engine import analyses
from stat_board.engine.cli import main

FORMULA = "y ~ hpi * cells + I(hpi**2) + I(cells**2)"


def _ccd(tmp_path, peak=(62.0, 2.6), noise=0.5, seed=3):
    """Face-centred CCD over harvest 48-72 h and density 2-4, true maximum at ``peak``."""
    rng = np.random.default_rng(seed)
    coded = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)] * 2 + [(0, 0)] * 5
    c = np.array(coded, float)
    hpi, cells = 60 + 12 * c[:, 0], 3 + 1 * c[:, 1]
    y = 100 - 0.08 * (hpi - peak[0]) ** 2 - 9 * (cells - peak[1]) ** 2 + 0.2 * (hpi - 60) * (cells - 3)
    y = y + rng.normal(0, noise, len(y))
    path = tmp_path / "ccd.csv"
    pd.DataFrame({"hpi": hpi, "cells": cells, "y": y}).to_csv(path, index=False)
    return path


def test_location_matches_solving_the_fitted_equations(tmp_path):
    path = _ccd(tmp_path)
    out = analyses.stationary_point(str(path), FORMULA, ["hpi", "cells"])

    # Independent route: fit the same model by least squares in natural units
    # and solve grad = 0 for y = b0 + b1 h + b2 c + b3 h c + b4 h^2 + b5 c^2.
    df = pd.read_csv(path)
    h, c = df["hpi"].to_numpy(), df["cells"].to_numpy()
    X = np.column_stack([np.ones_like(h), h, c, h * c, h**2, c**2])
    b, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    A = np.array([[2 * b[4], b[3]], [b[3], 2 * b[5]]])
    expected = np.linalg.solve(A, -b[1:3])

    assert out["kind"] == "maximum"
    assert out["inside_tested_region"] is True
    assert out["location"]["hpi"]["value"] == pytest.approx(expected[0], rel=1e-8)
    assert out["location"]["cells"]["value"] == pytest.approx(expected[1], rel=1e-8)


def test_confidence_interval_matches_simulation(tmp_path):
    """Delta-method SE vs the spread of stationary points under the coefficients' sampling distribution."""
    import statsmodels.formula.api as smf

    path = _ccd(tmp_path, noise=1.0)
    out = analyses.stationary_point(str(path), FORMULA, ["hpi", "cells"])
    fit = smf.ols(FORMULA, data=pd.read_csv(path)).fit()
    names = list(fit.params.index)
    draws = np.random.default_rng(0).multivariate_normal(fit.params.to_numpy(), fit.cov_params().to_numpy(), 20000)
    pts = []
    for beta in draws:
        p = dict(zip(names, beta, strict=True))
        A = np.array([[2 * p["I(hpi ** 2)"], p["hpi:cells"]], [p["hpi:cells"], 2 * p["I(cells ** 2)"]]])
        pts.append(np.linalg.solve(A, -np.array([p["hpi"], p["cells"]])))
    sd = np.std(np.array(pts), axis=0)
    assert out["location"]["hpi"]["std_err"] == pytest.approx(sd[0], rel=0.15)
    assert out["location"]["cells"]["std_err"] == pytest.approx(sd[1], rel=0.15)


def test_an_optimum_beyond_the_tested_range_is_flagged(tmp_path):
    path = _ccd(tmp_path, peak=(80.0, 2.6))
    out = analyses.stationary_point(str(path), FORMULA, ["hpi", "cells"])
    assert out["location"]["hpi"]["inside_tested_range"] is False
    assert out["inside_tested_region"] is False
    assert "outside" in out["verdict"]


def test_a_saddle_is_called_a_saddle(tmp_path):
    rng = np.random.default_rng(5)
    coded = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)] * 2 + [(0, 0)] * 4, float)
    y = 50 + 6 * coded[:, 0] ** 2 - 6 * coded[:, 1] ** 2 + rng.normal(0, 0.3, len(coded))
    path = tmp_path / "saddle.csv"
    pd.DataFrame({"a": coded[:, 0], "b": coded[:, 1], "y": y}).to_csv(path, index=False)
    out = analyses.stationary_point(str(path), "y ~ a * b + I(a**2) + I(b**2)", ["a", "b"])
    assert out["kind"] == "saddle"


def test_a_model_without_curvature_is_refused(tmp_path):
    path = _ccd(tmp_path)
    with pytest.raises(ValueError, match="no curvature|singular"):
        analyses.stationary_point(str(path), "y ~ hpi * cells", ["hpi", "cells"])


def test_a_cubic_model_is_refused(tmp_path):
    path = _ccd(tmp_path)
    with pytest.raises(ValueError, match="not quadratic"):
        analyses.stationary_point(str(path), FORMULA + " + I(hpi**3)", ["hpi", "cells"])


def test_runs_from_the_cli(tmp_path, capsys):
    path = _ccd(tmp_path)
    argv = ["stationary-point", "--data", str(path), "--formula", FORMULA, "--factor", "hpi", "--factor", "cells"]
    assert main(argv) == 0
    assert '"kind": "maximum"' in capsys.readouterr().out
