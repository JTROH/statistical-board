"""The judge-led round loop that drives the six roles of the statistical board.

Each round: the Analyst runs the `run_stat` engine tool (executed locally) to
gather real numbers; three critics (frequentist / assumptions / bayesian) attack
the analysis in parallel; the Verifier independently reproduces every statistic;
and the Judge adjudicates, composes the draft, and decides iterate vs. finalize.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Callable

from anthropic import AsyncAnthropic

from . import config, engine_tool, prompts
from .engine import analyses
from .engine.data import DataError, load_groups
from .llm import call_agent, parse_json
from .transcript import RoundRecord, Transcript

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "plan": {"type": "string"},
        "key_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hypothesis", "plan", "key_questions"],
    "additionalProperties": False,
}

_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["iterate", "finalize"]},
        "rationale": {"type": "string"},
        "next_instructions": {"type": "string"},
        "draft": {"type": "string"},
    },
    "required": ["decision", "rationale", "next_instructions", "draft"],
    "additionalProperties": False,
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _describe_data(data_path: str, group_col: str | None, value_col: str | None) -> str:
    """A compact, deterministic description of the dataset for the judge's plan."""
    try:
        groups = load_groups(data_path, group_col=group_col, value_col=value_col)
    except DataError as exc:
        return f"(could not load as grouped data: {exc})"
    d = analyses.describe(groups)["groups"]
    lines = [f"- {name}: n={s['n']}, mean={s['mean']:.4g}, sd={s['sd']:.4g}"
             for name, s in d.items()]
    return f"{len(groups)} group(s):\n" + "\n".join(lines)


def _describe_table(data_path: str, value: str, factors: list[str],
                    covariates: list[str] | None) -> str:
    """Column-by-column description for a multi-factor design."""
    import pandas as pd

    from .engine.data import load_dataframe
    try:
        df = load_dataframe(data_path)
    except Exception as exc:  # noqa: BLE001
        return f"(could not load table: {exc})"
    lines = [f"Design — outcome: {value!r}; factors: {factors}; "
             f"covariates: {covariates or 'none'}.", f"{len(df)} rows, columns:"]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            lines.append(f"- {c}: numeric (mean={df[c].mean():.4g}, sd={df[c].std():.4g})")
        else:
            levels = list(map(str, df[c].dropna().unique()))[:8]
            lines.append(f"- {c}: categorical, {df[c].nunique()} level(s) {levels}")
    return "\n".join(lines)


async def run(
    question: str,
    data_path: str,
    *,
    group_col: str | None = None,
    value_col: str | None = None,
    factors: list[str] | None = None,
    covariates: list[str] | None = None,
    paired: bool = False,
    alpha: float = 0.05,
    max_rounds: int | None = None,
    dry_run: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Transcript:
    max_rounds = max_rounds or config.MAX_ROUNDS
    client = None if dry_run else AsyncAnthropic()
    transcript = Transcript(
        question=question, model="dry-run" if dry_run else config.MODEL,
        data_path=data_path, group_col=group_col, value_col=value_col, alpha=alpha)

    def emit(kind: str, **data: Any) -> None:
        if on_event:
            on_event({"type": kind, **data})

    # Engine tool, bound to this run's dataset.
    executor = engine_tool.make_executor(data_path, group_col=group_col, value_col=value_col)
    local_tools = {engine_tool.TOOL_NAME: executor}
    tool_specs = [engine_tool.build_tool()]

    multifactor = bool(factors) or bool(covariates)
    if multifactor:
        data_desc = _describe_table(data_path, value_col or "?", factors or [], covariates)
        design_note = (
            "MULTI-FACTOR design. Analyze the whole table by column name with the "
            "run_stat multi-factor commands: `two-way-anova` (main effects + "
            "interactions of the factors), `ancova` (factors adjusted for the "
            "covariates), and/or `regression` (a patsy formula). The one-factor "
            "group commands (anova/ttest/tukey) do NOT apply to this design.")
        instructions = ("First round: run two-way-anova over the factors (and ancova "
                        "if covariates are given), report every term's F, p and "
                        "partial eta^2, and check the residual diagnostics.")
    else:
        load_hint = (f"Load with group column {group_col!r}, value column {value_col!r}."
                     if group_col else "Loaded automatically (wide CSV or JSON groups).")
        design_note = ("PAIRED/repeated-measures design." if paired
                       else "Independent-samples, single-factor design (assumed).")
        design_note += " " + load_hint
        data_desc = _describe_data(data_path, group_col, value_col)
        instructions = "First round: run describe + assumptions, then the tests the plan calls for."
    context = (f"QUESTION:\n{question}\n\nDATA: `{data_path}`\n{design_note}\n"
               f"alpha = {alpha}\n\nDATA SUMMARY:\n{data_desc}")

    if dry_run:
        _log("*** DRY RUN — stubbed responses, no API calls, zero spend ***")

    # ---- Planning ---------------------------------------------------------
    _log(f"\n=== Planning: {question} ===")
    plan_reply = await call_agent(
        client, system=prompts.JUDGE_PLAN,
        user=context + "\n\nProduce the hypothesis, analysis plan, and key open questions.",
        output_schema=_PLAN_SCHEMA, dry_run=dry_run,
    )
    plan = parse_json(plan_reply.text)
    transcript.hypothesis = plan["hypothesis"]
    transcript.key_questions = plan.get("key_questions", [])
    plan_text = plan["plan"]
    open_qs = "\n".join(f"- {q}" for q in transcript.key_questions)
    # `instructions` for round 1 was set with the design context above.
    _log(f"Plan set. Hypothesis: {transcript.hypothesis[:80]}...")
    emit("plan", hypothesis=transcript.hypothesis, plan=plan_text,
         key_questions=transcript.key_questions)

    draft = ""

    for rnd in range(1, max_rounds + 1):
        force_finalize = rnd == max_rounds
        _log(f"\n=== Round {rnd}/{max_rounds} ===")
        emit("round_start", round=rnd, total=max_rounds)

        # ---- Analyst (engine tool) ---------------------------------------
        _log("  [analyst] running the engine on the data...")
        emit("agent_start", round=rnd, agent="analyst")
        analyst = await call_agent(
            client, system=prompts.ANALYST,
            user=(f"{context}\n\nHYPOTHESIS:\n{transcript.hypothesis}\n\nPLAN:\n{plan_text}\n\n"
                  f"OPEN QUESTIONS:\n{open_qs}\n\nJUDGE'S INSTRUCTIONS THIS ROUND:\n{instructions}\n\n"
                  f"CURRENT DRAFT (may be empty):\n{draft or '(none yet)'}\n\n"
                  "Run the needed computations with run_stat and report the exact results."),
            local_tools=local_tools, tool_specs=tool_specs, dry_run=dry_run,
        )
        tool_log = [{"command": tc.command, "input": tc.input, "result": tc.result}
                    for tc in analyst.tool_calls]
        emit("agent_output", round=rnd, agent="analyst", text=analyst.text, tool_calls=tool_log)

        material = (f"{context}\n\nHYPOTHESIS:\n{transcript.hypothesis}\n\n"
                    f"ANALYST RESULTS:\n{analyst.text}\n\n"
                    f"CURRENT DRAFT (may be empty):\n{draft or '(none yet)'}")

        # ---- Critics in parallel -----------------------------------------
        _log("  [frequentist/assumptions/bayesian] critiquing in parallel...")
        for a in ("frequentist", "assumptions", "bayesian"):
            emit("agent_start", round=rnd, agent=a)
        freq, assum, bayes = await asyncio.gather(
            call_agent(client, system=prompts.FREQUENTIST, dry_run=dry_run,
                       user=material + "\n\nProduce your prioritized objections."),
            call_agent(client, system=prompts.ASSUMPTIONS, dry_run=dry_run,
                       user=material + "\n\nProduce your prioritized assumption risks."),
            call_agent(client, system=prompts.BAYESIAN, dry_run=dry_run,
                       user=material + "\n\nProduce your prioritized Bayesian reading."),
        )
        emit("agent_output", round=rnd, agent="frequentist", text=freq.text)
        emit("agent_output", round=rnd, agent="assumptions", text=assum.text)
        emit("agent_output", round=rnd, agent="bayesian", text=bayes.text)

        # ---- Verifier (engine tool) --------------------------------------
        _log("  [verifier] independently reproducing the numbers...")
        emit("agent_start", round=rnd, agent="verifier")
        verification = await call_agent(
            client, system=prompts.VERIFIER,
            user=material + "\n\nIndependently reproduce every statistic with run_stat and "
                            "issue per-claim verdicts. List unsupported/contradicted first.",
            local_tools=local_tools, tool_specs=tool_specs, dry_run=dry_run,
        )
        emit("agent_output", round=rnd, agent="verifier", text=verification.text)

        # ---- Judge: synthesize + decide ----------------------------------
        _log("  [judge] adjudicating and composing the draft...")
        emit("agent_start", round=rnd, agent="judge")
        finalize_note = ("\n\nThis is the FINAL allowed round — you MUST finalize. Produce the "
                         "best honest report the evidence supports and flag residual uncertainty."
                         if force_finalize else "")
        decision = await call_agent(
            client, system=prompts.JUDGE_DECIDE,
            user=(f"{material}\n\nFREQUENTIST CRITIQUE:\n{freq.text}\n\n"
                  f"ASSUMPTIONS CRITIQUE:\n{assum.text}\n\nBAYESIAN CRITIQUE:\n{bayes.text}\n\n"
                  f"VERIFIER REPRODUCTION:\n{verification.text}\n\n"
                  f"Adjudicate, compose the updated Markdown report draft, and decide iterate "
                  f"or finalize.{finalize_note}"),
            max_tokens=config.JUDGE_MAX_TOKENS, output_schema=_DECISION_SCHEMA, dry_run=dry_run,
        )
        dec = parse_json(decision.text)
        draft = dec["draft"]

        transcript.rounds.append(RoundRecord(
            number=rnd, analyst_notes=analyst.text, analyst_tool_calls=tool_log,
            frequentist=freq.text, assumptions=assum.text, bayesian=bayes.text,
            verification=verification.text, judge_rationale=dec["rationale"],
            decision=dec["decision"], next_instructions=dec.get("next_instructions", ""),
            draft=draft,
        ))
        _log(f"  [judge] decision: {dec['decision'].upper()}")
        emit("decision", round=rnd, decision=dec["decision"], rationale=dec["rationale"],
             next_instructions=dec.get("next_instructions", ""), draft=draft)

        if dec["decision"] == "finalize" or force_finalize:
            break
        instructions = dec.get("next_instructions", "") or instructions
        open_qs = instructions

    transcript.final_report = draft
    return transcript
