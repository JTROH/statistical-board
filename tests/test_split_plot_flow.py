"""Split-plot designs from the form to the run sheet, memo and analysis hint.

The maths is tested in ``tests/designs/test_split_plot.py`` and the analysis in
``tests/engine/test_mixed_model.py``. These protect the path around them: a
hard-to-change factor turns the options into split-plot designs, the run sheet
keeps each whole plot together and says which is which, and the hand-off to
the analysis names the mixed model rather than a regression that would
overstate the evidence.
"""

from __future__ import annotations

import csv

import pytest
from fastapi.testclient import TestClient

from doe_advisor import export
from doe_advisor.candidates import generate_candidates, score_candidates, top_options
from doe_advisor.intake import IntakeError, spec_from_dict, spec_to_dict
from doe_advisor.memo import build_memo
from doe_advisor.server import app

FORM = {
    "factors": [
        {"name": "temperature", "low": 35, "high": 37, "units": "C", "hard_to_change": True},
        {"name": "pH", "low": 6.8, "high": 7.2},
        {"name": "feed", "low": 2, "high": 6},
    ],
    "responses": [{"name": "titer", "units": "g/L", "target_effect": 0.6, "noise_sd": 0.25}],
    "model_order": "interaction",
}


def test_intake_reads_hard_to_change_and_the_whole_plot_settings():
    spec = spec_from_dict({**FORM, "n_whole_plots": 6, "wp_variance_ratio": 2})
    assert spec.is_split_plot and spec.whole_plot_factors == [0]
    assert (spec.n_whole_plots, spec.wp_variance_ratio) == (6, 2.0)
    again = spec_from_dict(spec_to_dict(spec))
    assert again.factors == spec.factors and again.n_whole_plots == 6


@pytest.mark.parametrize(
    "change, message",
    [
        ({"n_blocks": 2}, "whole plots or blocks, not both"),
        ({"n_whole_plots": 1}, "2 or more"),
        ({"wp_variance_ratio": -1}, "cannot be negative"),
    ],
)
def test_split_plot_mistakes_are_refused_in_plain_words(change, message):
    with pytest.raises(IntakeError, match=message):
        spec_from_dict({**FORM, **change})


def test_every_factor_hard_to_change_is_refused_at_intake():
    form = {**FORM, "factors": [{**f, "hard_to_change": True} for f in FORM["factors"]]}
    with pytest.raises(IntakeError, match="at least one factor as easy to change"):
        spec_from_dict(form)


def test_a_hard_to_change_factor_gets_only_split_plot_designs_with_a_whole_plot_tradeoff():
    spec = spec_from_dict(FORM)
    designs = generate_candidates(spec)
    assert designs and {d.family for d in designs} == {"split_plot"}
    assert len({d.n_whole_plots for d in designs}) > 1  # more whole plots is offered as an option
    assert any("recommended" in o.roles for o in top_options(spec))


def test_too_few_whole_plots_is_shown_and_explained_not_dropped():
    scored = score_candidates(spec_from_dict({**FORM, "n_whole_plots": 2}))
    assert scored and all(s.disqualified for s in scored)
    assert any("too few whole plots" in s.disqualified for s in scored)


def test_run_sheet_numbers_whole_plots_in_run_order_and_keeps_them_together(tmp_path):
    spec = spec_from_dict(FORM)
    option = next(o for o in top_options(spec) if "recommended" in o.roles)
    rows = list(csv.DictReader(export.run_sheet_csv(option.design, spec, tmp_path / "runs.csv").open()))
    plots = [int(r["whole_plot"]) for r in rows]
    assert plots == sorted(plots) and plots[0] == 1
    for g in set(plots):
        assert len({r["temperature_c"] for r in rows if int(r["whole_plot"]) == g}) == 1


def test_analysis_hint_points_at_the_mixed_model():
    hint = export.analysis_hint(spec_from_dict(FORM))
    assert hint["groups"] == "whole_plot"
    assert "mixed-model" in hint["mixed_model"] and "--groups whole_plot" in hint["mixed_model"]
    assert "overstates the evidence for temperature" in hint["note"]


def test_memo_explains_the_split_plot_and_the_run_order():
    text = build_memo(spec_from_dict(FORM), prefer_llm=False).markdown
    assert "**split-plot** design" in text
    assert "| Run | Whole plot |" in text
    assert "set the hard-to-change factors once for each of the" in text
    assert "mixed model with the whole plot as a random effect" in text
    assert "whole plots; hard-to-change terms tested on" in text


def test_design_endpoint_returns_split_plot_options():
    res = TestClient(app).post("/api/design", json={**FORM, "n_whole_plots": 6})
    assert res.status_code == 200, res.text
    options = res.json()["options"]
    assert options and all(o["n_whole_plots"] == 6 and o["whole_plot_df"] is not None for o in options)
