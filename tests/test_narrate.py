"""Narration must be honest before it is eloquent.

The load-bearing test here is the numeric guard: a narration layer that can
quietly invent a power figure is worse than no narration layer at all, so the
guard is tested directly rather than only through the happy path.
"""

from __future__ import annotations

import json

import pytest

from doe_advisor.candidates import top_options
from doe_advisor.designs.spec import DesignSpec, Factor, ModelOrder, Response, ResponseGoal
from doe_advisor.narrate import (
    ClaudeNarrator,
    Narration,
    TemplateNarrator,
    _allowed_numbers,
    _extract_json,
    _merge,
    _unsupported_numbers,
    get_narrator,
    option_facts,
)


def make_spec(order=ModelOrder.MAIN, max_runs=24, target=0.5, noise=0.25):
    return DesignSpec(
        factors=[
            Factor("pH", 6.8, 7.4),
            Factor("DO", 30, 60, "%"),
            Factor("temp", 35, 37, "C"),
            Factor("feed", 2, 6, "mL/day"),
        ],
        responses=[Response("titer", "g/L", target_effect=target, noise_sd=noise)],
        model_order=order,
        max_runs=max_runs,
        n_center_points=3,
        expected_run_losses=2,
    )


@pytest.fixture
def options():
    return top_options(make_spec())


# --------------------------------------------------------------------------
# The numeric guard
# --------------------------------------------------------------------------


def test_guard_accepts_numbers_drawn_from_the_facts():
    allowed = _allowed_numbers('{"runs": 16, "min_power_pct": 95}')
    assert _unsupported_numbers("16 runs give 95% power", allowed) == set()


def test_guard_catches_an_invented_statistic():
    """The failure this exists to prevent: plausible prose, fabricated number."""
    allowed = _allowed_numbers('{"runs": 16, "min_power_pct": 95}')
    assert _unsupported_numbers("this design has 87% power", allowed) == {"87"}


def test_guard_tolerates_small_counting_numbers():
    """Single digits appear in ordinary prose far more often than they smuggle a
    statistic, and every figure that matters here has more than one digit."""
    allowed = _allowed_numbers('{"runs": 16}')
    assert _unsupported_numbers("one of the 3 options uses 16 runs", allowed) == set()


def test_guard_catches_decimals():
    allowed = _allowed_numbers('{"i_value": 2.33}')
    assert "0.91" in _unsupported_numbers("efficiency of 0.91 here", allowed)


# --------------------------------------------------------------------------
# Template narrator
# --------------------------------------------------------------------------


def test_template_narrator_covers_every_option(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    assert narration.source == "template"
    assert len(narration.options) == len(options)
    assert narration.headline


def test_template_narrator_is_deterministic(options):
    spec = make_spec()
    first = TemplateNarrator().narrate(spec, options)
    second = TemplateNarrator().narrate(spec, options)
    assert first.headline == second.headline
    assert [o.pros for o in first.options] == [o.pros for o in second.options]


def test_every_option_gets_a_unique_id(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    ids = [o.option_id for o in narration.options]
    assert len(set(ids)) == len(ids)


def test_underpowered_design_is_called_out_as_a_con(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    weak = [
        n
        for n, o in zip(narration.options, options, strict=True)
        if (o.properties.power.min_main_effect_power or 1.0) < 0.8
    ]
    assert weak, "expected at least one underpowered option in this scenario"
    assert any("Underpowered" in c for c in weak[0].cons)


def test_confounded_design_is_called_out_as_a_con():
    """The 7-factor case, where a cheap screen is completely confounded."""
    spec = DesignSpec(
        factors=[Factor(f"x{i}", 0, 1) for i in range(7)],
        responses=[Response("titer", "g/L", target_effect=0.6, noise_sd=0.3)],
        model_order=ModelOrder.MAIN,
        max_runs=24,
        n_center_points=3,
        expected_run_losses=1,
    )
    opts = top_options(spec)
    narration = TemplateNarrator().narrate(spec, opts)
    confounded = [
        n
        for n, o in zip(narration.options, opts, strict=True)
        if o.properties.aliasing.worst_main_effect_alias >= 0.99
    ]
    if confounded:
        assert any("confounded" in c.lower() for c in confounded[0].cons)


def test_missing_centre_points_are_flagged(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    without = [
        n for n, o in zip(narration.options, options, strict=True) if o.design.n_center_points == 0
    ]
    if without:
        assert any("centre points" in c for c in without[0].cons)


def test_no_options_produces_an_explanation_not_a_crash():
    narration = TemplateNarrator().narrate(make_spec(), [])
    assert narration.options == []
    assert "budget" in narration.headline or "simplified" in narration.headline


def test_caveats_warn_when_power_could_not_be_computed():
    spec = make_spec()
    spec.responses = [Response("titer", "g/L")]
    narration = TemplateNarrator().narrate(spec, top_options(spec))
    assert any("power could not be computed" in c for c in narration.caveats)


def test_caveats_state_the_assumed_noise(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    assert any("standard deviation" in c for c in narration.caveats)


def test_caveats_mention_the_phase_one_limits(options):
    """The memo must not imply capabilities the tool does not yet have."""
    narration = TemplateNarrator().narrate(make_spec(), options)
    joined = " ".join(narration.caveats)
    assert "Constrained design spaces" in joined
    assert "Randomise the run order" in joined


# --------------------------------------------------------------------------
# Claude narrator plumbing
# --------------------------------------------------------------------------


def test_claude_narrator_is_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ClaudeNarrator(api_key=None).available is False


def test_claude_narrator_falls_back_to_templates_without_a_key(monkeypatch, options):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    narration = ClaudeNarrator(api_key=None).narrate(make_spec(), options)
    assert narration.source == "template"
    assert narration.fallback_reason
    assert len(narration.options) == len(options)


def test_get_narrator_returns_templates_when_llm_not_wanted():
    assert isinstance(get_narrator(prefer_llm=False), TemplateNarrator)


def test_extract_json_handles_a_fenced_reply():
    assert json.loads(_extract_json('```json\n{"a": 1}\n```')) == {"a": 1}


def test_extract_json_handles_a_prefaced_reply():
    assert json.loads(_extract_json('Sure, here you go:\n{"a": 1}\nHope that helps.')) == {"a": 1}


def test_merge_keeps_engine_facts_and_takes_only_prose(options):
    """Claude may rewrite the words; it may not change the run count."""
    baseline = TemplateNarrator().narrate(make_spec(), options)
    parsed = {
        "headline": "rewritten",
        "options": [
            {"id": o.option_id, "summary": "new summary", "pros": ["nice"], "cons": ["meh"]}
            for o in baseline.options
        ],
    }
    merged = _merge(parsed, baseline)
    for original, new in zip(baseline.options, merged, strict=True):
        assert new.n_runs == original.n_runs
        assert new.option_name == original.option_name
        assert new.role == original.role
        assert new.summary == "new summary"


def test_merge_falls_back_per_option_when_an_id_is_missing(options):
    baseline = TemplateNarrator().narrate(make_spec(), options)
    merged = _merge({"headline": "x", "options": []}, baseline)
    assert [o.summary for o in merged] == [o.summary for o in baseline.options]


def test_option_facts_carry_no_unformatted_floats(options):
    """Everything Claude sees is pre-rounded, so it cannot echo back a
    seventeen-decimal number that looks computed but is not."""
    spec = make_spec()
    facts = option_facts(options[0], spec, min(o.n_runs for o in options), max(o.n_runs for o in options))
    for key, value in facts.items():
        if isinstance(value, float):
            assert len(str(value).split(".")[-1]) <= 3, f"{key} is not rounded"


def test_narration_dataclass_defaults():
    narration = Narration(source="template", headline="hi")
    assert narration.options == []
    assert narration.caveats == []
    assert narration.fallback_reason is None


# --------------------------------------------------------------------------
# Power explained in the scientist's units, and the goal caveat
# --------------------------------------------------------------------------


def test_template_states_the_smallest_detectable_change_in_units(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    text = " ".join(line for o in narration.options for line in o.pros + o.cons)
    assert "Smallest change it can reliably see" in text or "smallest change it can reliably see" in text
    assert "g/L" in text
    assert "1.5 times" in text


def test_template_explains_power_as_a_chance(options):
    narration = TemplateNarrator().narrate(make_spec(), options)
    text = " ".join(line for o in narration.options for line in o.pros + o.cons)
    assert "in 100" in text


def test_goal_caveat_when_an_optimum_is_wanted_without_curvature():
    spec = make_spec(order=ModelOrder.INTERACTION)
    spec.responses = [Response("titer", "g/L", 0.5, 0.25, goal=ResponseGoal.MAXIMIZE)]
    narration = TemplateNarrator().narrate(spec, top_options(spec))
    assert any("make titer as high as possible" in c and "curvature" in c for c in narration.caveats)


def test_no_goal_caveat_when_curvature_model_is_requested():
    spec = make_spec(order=ModelOrder.QUADRATIC, max_runs=40)
    spec.responses = [Response("titer", "g/L", 0.5, 0.25, goal=ResponseGoal.MAXIMIZE)]
    narration = TemplateNarrator().narrate(spec, top_options(spec))
    assert not any("Your goal is to" in c for c in narration.caveats)


def test_no_goal_caveat_when_screening():
    narration = TemplateNarrator().narrate(make_spec(), top_options(make_spec()))
    assert not any("Your goal is to" in c for c in narration.caveats)
