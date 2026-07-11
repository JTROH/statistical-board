"""Async wrapper around the Anthropic SDK for the statistical board.

The notable feature is a *local* tool-execution loop: the Analyst and Verifier
are given the `run_stat` engine tool, and when the model calls it we execute the
computation in-process and feed the JSON back — so every claim is grounded in a
real calculation. Every call streams, uses adaptive thinking + the configured
effort level, and resumes paused turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from anthropic import AsyncAnthropic

from . import config


@dataclass
class ToolCall:
    command: str
    input: dict[str, Any]
    result: str  # JSON string returned by the engine


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    refused: bool = False


def _extract_text(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _role_label(system: str) -> str:
    for line in system.splitlines():
        s = line.strip()
        if s.startswith("YOUR ROLE:"):
            return s[len("YOUR ROLE:"):].strip().rstrip(".")
    return "agent"


def _stub_reply(
    system: str,
    local_tools: dict[str, Callable[[dict[str, Any]], str]] | None,
    output_schema: dict[str, Any] | None,
) -> AgentReply:
    """Canned reply for --dry-run: zero API calls, zero spend. Structured-output
    calls return schema-conforming JSON so the judge's parsing works unchanged."""
    if output_schema is not None:
        props = output_schema.get("properties", {})
        if "key_questions" in props:  # judge plan
            data: dict[str, Any] = {
                "hypothesis": "[dry-run] H0: group means are equal, two-sided, alpha=0.05.",
                "plan": "[dry-run] Path: describe -> assumptions -> (Welch ANOVA | Kruskal) "
                        "-> pairwise + effect sizes + Bayes factors.",
                "key_questions": ["[dry-run] Do variances differ (Levene)?",
                                  "[dry-run] Is n adequate for the effect of interest?"],
            }
        else:  # judge decision
            data = {
                "decision": "finalize",
                "rationale": "[dry-run] Stubbed rationale.",
                "next_instructions": "",
                "draft": "# [dry-run] Statistical report\n\nStubbed draft (no API calls).",
            }
        return AgentReply(text=json.dumps(data), stop_reason="end_turn")

    tool_calls = (
        [ToolCall(command="describe", input={"command": "describe"},
                  result='{"analysis": "describe", "note": "[dry-run] stubbed"}')]
        if local_tools else []
    )
    return AgentReply(
        text=f"[dry-run] {_role_label(system)} output (stubbed, no API call).",
        tool_calls=tool_calls, stop_reason="end_turn",
    )


async def call_agent(
    client: AsyncAnthropic | None,
    *,
    system: str,
    user: str,
    local_tools: dict[str, Callable[[dict[str, Any]], str]] | None = None,
    tool_specs: list[dict[str, Any]] | None = None,
    max_tokens: int = config.MAX_TOKENS,
    output_schema: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> AgentReply:
    """Run a single agent turn. If `local_tools`/`tool_specs` are given, execute
    engine tool calls locally and loop until the model stops."""
    if dry_run:
        return _stub_reply(system, local_tools, output_schema)

    assert client is not None, "call_agent requires a client unless dry_run=True"
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    output_config: dict[str, Any] = {"effort": config.EFFORT}
    if output_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}

    tool_calls: list[ToolCall] = []

    while True:
        params: dict[str, Any] = {
            "model": config.MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
        }
        if tool_specs:
            params["tools"] = tool_specs

        async with client.messages.stream(**params) as stream:
            message = await stream.get_final_message()

        if message.stop_reason == "tool_use" and local_tools:
            results = []
            for block in message.content:
                if getattr(block, "type", None) == "tool_use":
                    executor = local_tools.get(block.name)
                    out = (executor(block.input) if executor
                           else json.dumps({"error": "no_such_tool", "name": block.name}))
                    tool_calls.append(ToolCall(
                        command=block.input.get("command", block.name),
                        input=block.input, result=out))
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            messages = messages + [
                {"role": "assistant", "content": message.content},
                {"role": "user", "content": results},
            ]
            if len(tool_calls) >= config.MAX_TOOL_CALLS:
                # Safety valve: ask for the final answer with no more tools.
                messages.append({"role": "user", "content":
                                 "Tool-call budget reached. Now give your final answer using "
                                 "the results already gathered; do not call the tool again."})
                tool_specs = None
            continue

        if message.stop_reason == "pause_turn":
            messages = messages + [{"role": "assistant", "content": message.content}]
            continue

        return AgentReply(
            text=_extract_text(message.content),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason,
            refused=message.stop_reason == "refusal",
        )


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)
