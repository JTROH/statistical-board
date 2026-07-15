from __future__ import annotations

import asyncio
import json

import pytest

from stat_board.llm import call_agent, parse_json


def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_fenced_code_block():
    assert parse_json("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json("not json")


def test_parse_json_fixes_stray_unicode_escape_text():
    # The model sometimes writes the literal 6 characters '–' as text
    # instead of an actual em dash -- json.dumps below reproduces that exactly
    # (a JSON string whose *content*, after real JSON decoding, still contains
    # a backslash-u-escape-looking substring).
    raw = json.dumps({"draft": "A\\u2013B and \\u03bc/\\u03b7\\u00b2"})
    out = parse_json(raw)
    assert out["draft"] == "A–B and μ/η²"


def test_parse_json_leaves_real_json_escaped_unicode_untouched():
    raw = json.dumps({"draft": "Already real: – μ"}, ensure_ascii=True)
    out = parse_json(raw)
    assert out["draft"] == "Already real: – μ"


def test_parse_json_fixes_stray_escapes_inside_nested_lists_and_dicts():
    raw = json.dumps({"key_questions": ["A\\u2013B?"], "nested": {"x": "\\u03bc"}})
    out = parse_json(raw)
    assert out["key_questions"] == ["A–B?"]
    assert out["nested"]["x"] == "μ"


def test_call_agent_dry_run_plain_call_needs_no_client():
    reply = asyncio.run(call_agent(
        None, system="YOUR ROLE: Analyst.", user="anything", dry_run=True))
    assert reply.stop_reason == "end_turn"
    assert "Analyst" in reply.text
    assert reply.tool_calls == []


def test_call_agent_dry_run_with_local_tools_returns_a_stub_tool_call():
    reply = asyncio.run(call_agent(
        None, system="YOUR ROLE: Analyst.", user="anything",
        local_tools={"run_stat": lambda inp: "{}"}, dry_run=True))
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].command == "describe"


def test_call_agent_dry_run_plan_schema_returns_hypothesis_plan_and_key_questions():
    schema = {"properties": {"hypothesis": {}, "plan": {}, "key_questions": {}}}
    reply = asyncio.run(call_agent(
        None, system="YOUR ROLE: Judge.", user="plan", output_schema=schema, dry_run=True))
    data = json.loads(reply.text)
    assert "hypothesis" in data
    assert isinstance(data["key_questions"], list)


def test_call_agent_dry_run_decision_schema_returns_iterate_or_finalize():
    schema = {"properties": {"decision": {}, "rationale": {}, "next_instructions": {}, "draft": {}}}
    reply = asyncio.run(call_agent(
        None, system="YOUR ROLE: Judge.", user="decide", output_schema=schema, dry_run=True))
    data = json.loads(reply.text)
    assert data["decision"] in {"iterate", "finalize"}


def test_call_agent_dry_run_decision_schema_defaults_model_fields_to_not_applicable():
    # model_formula/model_factors/model_typ are only meaningful for a
    # multi-factor/DoE round; dry-run never fits a real model, so these must
    # come back as empty/zero sentinels rather than fabricated values.
    schema = {"properties": {"decision": {}, "rationale": {}, "next_instructions": {},
                             "draft": {}, "model_formula": {}, "model_factors": {},
                             "model_typ": {}}}
    reply = asyncio.run(call_agent(
        None, system="YOUR ROLE: Judge.", user="decide", output_schema=schema, dry_run=True))
    data = json.loads(reply.text)
    assert data["model_formula"] == ""
    assert data["model_factors"] == []
    assert data["model_typ"] == 0


def test_call_agent_without_dry_run_requires_a_client():
    with pytest.raises(AssertionError, match="requires a client"):
        asyncio.run(call_agent(None, system="s", user="u", dry_run=False))
