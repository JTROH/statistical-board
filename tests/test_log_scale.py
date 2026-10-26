"""Noise stated as a CV, effect as a fold change: powered on log10.

Titre noise grows with the titre. On a log scale it is constant, which is what
the power maths assumes. These tests pin the conversion and check that the
whole path — form, power, memo, run sheet, analysis formula — stays on it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from doe_advisor.candidates import top_options
from doe_advisor.export import analysis_hint, run_sheet_csv
from doe_advisor.intake import IntakeError, cv_to_log10_sd, spec_from_dict, spec_to_dict
from doe_advisor.memo import build_memo
from stat_board.engine import analyses

FORM = {
    "factors": [
        {"name": "MOI", "low": -0.3, "high": 0.48, "units": "log10 pfu/cell"},
        {"name": "harvest", "low": 48, "high": 72, "units": "h"},
        {"name": "density", "low": 2, "high": 4, "units": "e6/mL"},
    ],
    "responses": [
        {"name": "titre", "units": "mg/L", "goal": "maximize",
         "noise_model": "cv", "noise_cv_pct": 10, "target_fold": 1.5},
    ],
    "model_order": "interaction",
    "max_runs": 12,
}


def test_cv_converts_to_the_exact_lognormal_sd():
    # sigma_ln = sqrt(ln(1 + 0.1^2)) = 0.099751; / ln 10
    assert cv_to_log10_sd(10) == pytest.approx(0.0433208, abs=1e-6)
    # For a lognormal, simulated log10 SD at that CV matches.
    rng = np.random.default_rng(0)
    sigma_ln = math.sqrt(math.log1p(0.01))
    x = rng.lognormal(0, sigma_ln, 200_000)
    assert np.std(x) / np.mean(x) == pytest.approx(0.10, abs=0.002)
    assert np.std(np.log10(x)) == pytest.approx(cv_to_log10_sd(10), abs=3e-4)


def test_form_lands_on_the_log10_scale():
    r = spec_from_dict(FORM).primary_response
    assert r.is_log and r.display_units == "log10 mg/L"
    assert r.target_effect == pytest.approx(math.log10(1.5))
    assert r.noise_sd == pytest.approx(cv_to_log10_sd(10))
    assert r.standardised_effect == pytest.approx(math.log10(1.5) / cv_to_log10_sd(10))


def test_a_fall_is_the_same_size_as_the_matching_rise():
    form = {**FORM, "responses": [{**FORM["responses"][0], "target_fold": 1 / 1.5}]}
    assert spec_from_dict(form).primary_response.target_effect == pytest.approx(math.log10(1.5))


def test_cv_form_round_trips():
    again = spec_from_dict(spec_to_dict(spec_from_dict(FORM))).primary_response
    assert again.noise_cv_pct == 10 and again.target_fold == 1.5 and again.is_log


@pytest.mark.parametrize("bad", [{"noise_cv_pct": 0}, {"target_fold": 1}, {"target_fold": -2}, {"noise_model": "x"}])
def test_bad_cv_inputs_are_refused(bad):
    form = {**FORM, "responses": [{**FORM["responses"][0], **bad}]}
    with pytest.raises(IntakeError):
        spec_from_dict(form)


def test_memo_speaks_in_folds_and_cv():
    spec = spec_from_dict(FORM)
    text = build_memo(spec, prefer_llm=False).markdown
    assert "10% CV" in text
    assert "1.5-fold" in text
    assert "log10" in text


def test_the_analysis_hint_formula_runs_on_raw_bench_values(tmp_path):
    spec = spec_from_dict(FORM)
    design = next(o for o in top_options(spec) if "recommended" in o.roles).design
    path = run_sheet_csv(design, spec, tmp_path / "runs.csv")
    hint = analysis_hint(spec)
    assert hint["response_scale"] == "log10"
    assert hint["formula"].startswith("np.log10(titre_mg_l) ~")

    # Fill raw titres with 10% CV noise around a true 1.5-fold MOI effect.
    import pandas as pd

    df = pd.read_csv(path)
    rng = np.random.default_rng(1)
    moi = df["moi_coded"].astype(float)
    df["titre_mg_l"] = 100 * 1.5 ** (moi / 2) * rng.lognormal(0, math.sqrt(math.log1p(0.01)), len(df))
    df.to_csv(path, index=False)
    fit = analyses.regression(str(path), hint["formula"])
    assert any("moi" in name for name in fit["coefficients"])
