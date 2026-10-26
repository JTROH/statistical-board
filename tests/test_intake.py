"""Intake validation.

Error messages are part of the product here: the person who sees them is a
bench scientist filling in a form, not a developer reading a stack trace. So the
tests assert on what the messages say, not only that something was raised.
"""

from __future__ import annotations

import json

import pytest

from doe_advisor.designs.spec import DesignSpec, Factor, ModelOrder, Response, ResponseGoal
from doe_advisor.intake import (
    IntakeError,
    extract_form_from_text,
    list_presets,
    llm_available,
    load_preset,
    spec_from_dict,
    spec_to_dict,
)

GOOD = {
    "factors": [
        {"name": "pH", "low": 6.8, "high": 7.2, "units": ""},
        {"name": "DO", "low": 30, "high": 60, "units": "%"},
        {"name": "temp", "low": 35, "high": 37, "units": "C"},
    ],
    "responses": [{"name": "titer", "units": "g/L", "target_effect": 0.5, "noise_sd": 0.25}],
    "model_order": "main",
    "max_runs": 24,
    "n_center_points": 3,
    "expected_run_losses": 1,
}


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------


def test_presets_load_and_blank_comes_first():
    presets = list_presets()
    assert len(presets) >= 3
    assert presets[0]["id"] == "blank"


def test_every_preset_has_the_required_shape():
    for preset in list_presets():
        assert preset["id"] and preset["label"]
        assert isinstance(preset["factors"], list)
        assert isinstance(preset["responses"], list)


def test_insect_preset_is_the_first_real_preset():
    """Blank stays first; the Sf9 preset is the first domain preset and so the default."""
    ids = [p["id"] for p in list_presets()]
    assert ids[0] == "blank"
    assert ids[1] == "upstream_insect"


def test_domain_presets_produce_a_valid_spec():
    """A preset the scientist has not edited must already be runnable."""
    for preset_id in ("upstream_insect", "upstream_mammalian", "upstream_microbial"):
        preset = load_preset(preset_id)
        spec = spec_from_dict(
            {
                "factors": preset["factors"],
                "responses": preset["responses"][:1],
                **preset["defaults"],
            }
        )
        assert spec.n_factors >= 4
        assert spec.primary_response.standardised_effect is not None


def test_unknown_preset_raises():
    with pytest.raises(IntakeError):
        load_preset("does-not-exist")


def test_malformed_preset_is_skipped_not_fatal(tmp_path):
    """One bad JSON file must not take the whole app down."""
    (tmp_path / "good.json").write_text(json.dumps({"id": "x", "label": "X", "factors": [], "responses": []}))
    (tmp_path / "bad.json").write_text("{not json")
    assert [p["id"] for p in list_presets(tmp_path)] == ["x"]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_valid_payload_round_trips():
    spec = spec_from_dict(GOOD)
    assert spec.n_factors == 3
    assert spec.model_order is ModelOrder.MAIN
    assert spec.max_runs == 24
    assert spec_from_dict(spec_to_dict(spec)).factor_names == spec.factor_names


def test_blank_factor_rows_are_ignored():
    """The form ships with empty rows; they are not errors."""
    payload = {**GOOD, "factors": GOOD["factors"] + [{"name": "", "low": None, "high": None}]}
    assert spec_from_dict(payload).n_factors == 3


def test_too_few_factors():
    with pytest.raises(IntakeError, match="at least two factors"):
        spec_from_dict({**GOOD, "factors": GOOD["factors"][:1]})


def test_too_many_factors():
    factors = [{"name": f"x{i}", "low": 0, "high": 1} for i in range(11)]
    with pytest.raises(IntakeError, match="At most"):
        spec_from_dict({**GOOD, "factors": factors})


def test_identical_low_and_high():
    with pytest.raises(IntakeError, match="not a factor"):
        spec_from_dict({**GOOD, "factors": [{"name": "pH", "low": 7, "high": 7}, *GOOD["factors"][1:]]})


def test_transposed_range_is_corrected_silently():
    """A high below a low is a typo, not a reason to stop."""
    spec = spec_from_dict({**GOOD, "factors": [{"name": "pH", "low": 7.2, "high": 6.8}, *GOOD["factors"][1:]]})
    assert spec.factors[0].low == 6.8
    assert spec.factors[0].high == 7.2


def test_duplicate_factor_names():
    duplicated = [GOOD["factors"][0], {**GOOD["factors"][1], "name": "pH"}]
    with pytest.raises(IntakeError, match="unique"):
        spec_from_dict({**GOOD, "factors": duplicated})


def test_non_numeric_range():
    with pytest.raises(IntakeError, match="must be a number"):
        spec_from_dict({**GOOD, "factors": [{"name": "pH", "low": "acidic", "high": 7.2}, *GOOD["factors"][1:]]})


def test_budget_smaller_than_the_factor_count_explains_the_minimum():
    with pytest.raises(IntakeError, match="at least 4"):
        spec_from_dict({**GOOD, "max_runs": 2})


def test_absurd_budget_is_rejected():
    with pytest.raises(IntakeError, match="implausibly large"):
        spec_from_dict({**GOOD, "max_runs": 99999})


def test_unknown_model_order_lists_the_valid_ones():
    with pytest.raises(IntakeError, match="main, interaction, quadratic"):
        spec_from_dict({**GOOD, "model_order": "cubic"})


def test_zero_noise_is_rejected():
    bad = [{"name": "titer", "units": "g/L", "target_effect": 0.5, "noise_sd": 0}]
    with pytest.raises(IntakeError, match="greater than zero"):
        spec_from_dict({**GOOD, "responses": bad})


def test_responses_are_optional():
    spec = spec_from_dict({**GOOD, "responses": []})
    assert spec.responses == []


def test_blank_response_rows_are_ignored():
    spec = spec_from_dict({**GOOD, "responses": [{"name": "", "units": "g/L"}]})
    assert spec.responses == []


def test_missing_optional_numbers_fall_back_to_defaults():
    spec = spec_from_dict(
        {"factors": GOOD["factors"], "responses": [], "model_order": "main"}
    )
    assert spec.max_runs is None
    assert spec.n_center_points == 3
    assert spec.expected_run_losses == 1


def test_negative_centre_points_are_clamped():
    assert spec_from_dict({**GOOD, "n_center_points": -5}).n_center_points == 0


def test_non_dict_payload():
    with pytest.raises(IntakeError):
        spec_from_dict(["not", "a", "form"])


def test_response_goal_defaults_to_screen():
    spec = spec_from_dict(GOOD)
    assert spec.primary_response.goal is ResponseGoal.SCREEN
    assert spec.primary_response.target_value is None


def test_response_goal_and_target_value_round_trip():
    payload = {
        **GOOD,
        "responses": [{"name": "pH", "units": "", "goal": "target", "target_value": 7.0}],
    }
    spec = spec_from_dict(payload)
    assert spec.primary_response.goal is ResponseGoal.TARGET
    assert spec.primary_response.target_value == 7.0
    assert spec.primary_response.goal_statement == "bring pH to 7"
    assert spec_from_dict(spec_to_dict(spec)).primary_response == spec.primary_response


def test_target_value_is_dropped_unless_goal_is_target():
    payload = {**GOOD, "responses": [{"name": "titer", "goal": "maximize", "target_value": 3.0}]}
    assert spec_from_dict(payload).primary_response.target_value is None


def test_unknown_goal_is_rejected_with_the_choices():
    payload = {**GOOD, "responses": [{"name": "titer", "goal": "optimise"}]}
    with pytest.raises(IntakeError, match="maximize"):
        spec_from_dict(payload)


def test_spec_to_dict_preserves_unknown_power_inputs():
    spec = DesignSpec(
        factors=[Factor("a", 0, 1), Factor("b", 0, 1)],
        responses=[Response("y", "g/L")],
    )
    payload = spec_to_dict(spec)
    assert payload["responses"][0]["target_effect"] is None
    assert spec_from_dict(payload).primary_response.standardised_effect is None


# --------------------------------------------------------------------------
# Conversational intake
# --------------------------------------------------------------------------


def test_extraction_without_a_key_explains_the_alternative(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(IntakeError, match="Fill the form in directly"):
        extract_form_from_text("screen pH and DO", api_key=None)


def test_extraction_rejects_empty_text():
    with pytest.raises(IntakeError, match="Describe your experiment"):
        extract_form_from_text("   ", api_key="sk-not-used")


def test_llm_availability_follows_the_environment(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_available() is False
