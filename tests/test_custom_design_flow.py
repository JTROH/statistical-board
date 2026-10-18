"""Categorical factors and custom designs, from the form to the run sheet.

The maths is tested in ``tests/designs``. These tests protect the path around
it: what the scientist types is read correctly (or refused in words they can
act on), the options table offers a custom design exactly when the classical
families cannot serve, and the run sheet and memo say what will actually be
run.
"""

from __future__ import annotations

import csv

import pytest
from fastapi.testclient import TestClient

from doe_advisor import export
from doe_advisor.candidates import custom_run_budgets, generate_candidates, score_candidates, top_options
from doe_advisor.intake import IntakeError, parse_constraint_text, spec_from_dict, spec_to_dict
from doe_advisor.memo import build_memo
from doe_advisor.server import app

FORM = {
    "factors": [
        {"name": "pH", "low": 6.8, "high": 7.2},
        {"name": "feed rate", "low": 2, "high": 6, "units": "% v/v/day"},
        {"name": "media", "type": "categorical", "levels": ["CD-A", "CD-B", "CD-C"]},
    ],
    "responses": [{"name": "titer", "units": "g/L", "target_effect": 0.6, "noise_sd": 0.25}],
    "model_order": "interaction",
    "n_center_points": 2,
    "expected_run_losses": 1,
}


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------


def test_categorical_levels_can_be_a_list_or_a_comma_separated_string():
    as_list = spec_from_dict(FORM)
    typed = {"name": "media", "levels": "CD-A, CD-B,CD-C"}  # no "type": levels alone make it categorical
    as_text = spec_from_dict({**FORM, "factors": [*FORM["factors"][:2], typed]})
    assert as_list.factors[2].levels == as_text.factors[2].levels == ("CD-A", "CD-B", "CD-C")


@pytest.mark.parametrize(
    "levels, message",
    [(["CD-A"], "at least two levels"), (["A", "A"], "unique"), ([str(i) for i in range(13)], "at most 12")],
)
def test_bad_categorical_levels_are_refused_in_plain_words(levels, message):
    form = {**FORM, "factors": [*FORM["factors"][:2], {"name": "media", "type": "categorical", "levels": levels}]}
    with pytest.raises(IntakeError, match=message):
        spec_from_dict(form)


def test_the_run_budget_floor_counts_categorical_levels():
    # 1 intercept + pH + feed + 2 media coefficients = 5.
    with pytest.raises(IntakeError, match="at least 5"):
        spec_from_dict({**FORM, "max_runs": 4})


def test_structured_and_typed_constraints_are_both_read():
    spec = spec_from_dict({
        **FORM,
        "factors": [*FORM["factors"], {"name": "line", "levels": ["L1", "L2"]}],
        "constraints": [{"coefficients": {"pH": 1, "feed rate": 1}, "op": "<=", "bound": 12}],
        "constraints_text": "2*feed rate - pH >= -4\nnever media = CD-C with line = L2",
    })
    assert [c.describe() for c in spec.constraints] == ["pH + feed rate <= 12", "2*feed rate - pH >= -4"]
    assert spec.disallowed[0].levels == {"media": ("CD-C",), "line": ("L2",)}


@pytest.mark.parametrize(
    "text, message",
    [
        ("glucose <= 3", "no continuous factor name"),
        ("media <= 1", "no continuous factor name"),
        ("never pH = 7 with media = CD-A", "continuous; limit it with a linear constraint"),
        ("never media = CD-D with pH = 7", "has levels"),
        ("never media = CD-A", "remove it from media's levels"),
        ("pH is small", "expected"),
    ],
)
def test_constraint_mistakes_name_the_line_and_the_fix(text, message):
    spec = spec_from_dict(FORM)
    with pytest.raises(IntakeError, match=message):
        parse_constraint_text(text, spec.factors)


def test_spec_round_trips_through_the_form_with_categoricals_and_constraints():
    form = {**FORM, "factors": [*FORM["factors"], {"name": "line", "levels": ["L1", "L2"]}],
            "constraints_text": "pH + feed rate <= 12\nnever media = CD-C with line = L2"}
    spec = spec_from_dict(form)
    again = spec_from_dict(spec_to_dict(spec))
    assert again.factors == spec.factors
    assert [c.describe() for c in again.constraints] == [c.describe() for c in spec.constraints]
    assert [d.levels for d in again.disallowed] == [d.levels for d in spec.disallowed]


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


def test_a_categorical_factor_gets_only_custom_designs():
    spec = spec_from_dict(FORM)
    designs = generate_candidates(spec)
    assert designs and {d.family for d in designs} == {"custom"}
    options = top_options(spec)
    assert any("recommended" in o.roles for o in options)


def test_constraints_disqualify_classical_designs_that_break_them_and_add_custom_ones():
    form = {**FORM, "factors": FORM["factors"][:2] + [{"name": "temp", "low": 35, "high": 37}],
            "constraints_text": "feed rate + temp <= 41"}
    scored = score_candidates(spec_from_dict(form))
    broken = [s for s in scored if s.properties.n_infeasible_runs and s.properties.estimable]
    assert broken and all(s.design.family != "custom" for s in broken)
    assert all("break the constraints" in s.disqualified for s in broken)
    assert any(s.design.family == "custom" and s.is_viable for s in scored)


def test_an_odd_budget_no_classical_design_fits_gets_a_custom_design_of_exactly_that_size():
    form = {**FORM, "factors": FORM["factors"][:2] + [{"name": "temp", "low": 35, "high": 37}],
            "model_order": "quadratic", "max_runs": 14}
    options = top_options(spec_from_dict(form))
    assert any(o.design.family == "custom" and o.n_runs == 14 for o in options)
    assert all(o.n_runs <= 14 for o in options)


def test_budgets_without_a_limit_follow_the_power_curve():
    spec = spec_from_dict(FORM)
    budgets = custom_run_budgets(spec)
    # smallest sensible, 80% power, 80% power even if the noise is worse
    assert len(budgets) == 3 and budgets == sorted(set(budgets))


def test_blocked_custom_designs_keep_their_levels():
    spec = spec_from_dict({**FORM, "n_blocks": 2})
    designs = generate_candidates(spec)
    assert designs and all(d.blocks is not None and d.levels == (0, 0, 3) for d in designs)


# --------------------------------------------------------------------------
# Run sheet and memo
# --------------------------------------------------------------------------


def test_run_sheet_writes_level_names_and_no_coded_column_for_a_categorical(tmp_path):
    spec = spec_from_dict(FORM)
    option = next(o for o in top_options(spec) if "recommended" in o.roles)
    path = export.run_sheet_csv(option.design, spec, tmp_path / "runs.csv")
    rows = list(csv.DictReader(path.open()))
    assert "media_coded" not in rows[0]
    assert {r["media"] for r in rows} <= {"CD-A", "CD-B", "CD-C"}
    assert len(rows) == option.n_runs


def test_analysis_hint_marks_categorical_columns_as_categorical():
    hint = export.analysis_hint(spec_from_dict(FORM))
    assert "C(media)" in hint["formula"]
    assert hint["categorical_columns"] == ["media"]


def test_run_type_ignores_categorical_columns():
    assert export.run_type([0.0, 0.0, 2.0], levels=(0, 0, 3)) == "center"
    assert export.run_type([1.0, -1.0, 0.0], levels=(0, 0, 3)) == "factorial"
    assert export.run_type([0.0, 1.0], levels=(3, 2)) == "factorial"


def test_memo_lists_levels_constraints_and_what_cannot_be_estimated():
    form = {**FORM, "factors": [*FORM["factors"], {"name": "line", "levels": ["L1", "L2"]}],
            "constraints_text": "never media = CD-C with line = L2"}
    memo = build_memo(spec_from_dict(form), prefer_llm=False)
    text = memo.markdown
    assert "CD-A, CD-B, CD-C" in text
    assert "never media = CD-C with line = L2" in text
    assert "no design can estimate **media x line**" in text
    assert "difference between two of its levels" in text
    assert "coordinate exchange" in text


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def test_design_endpoint_accepts_categorical_factors_and_typed_constraints():
    client = TestClient(app)
    form = {**FORM, "constraints_text": "pH + feed rate <= 12"}
    res = client.post("/api/design", json=form)
    assert res.status_code == 200, res.text
    options = res.json()["options"]
    assert options and all(o["family"] == "custom" for o in options)
    assert all(o["n_infeasible_runs"] == 0 for o in options)


def test_design_endpoint_explains_a_bad_constraint():
    client = TestClient(app)
    res = client.post("/api/design", json={**FORM, "constraints_text": "glucose <= 3"})
    assert res.status_code == 400
    assert "no continuous factor name" in res.json()["error"]
