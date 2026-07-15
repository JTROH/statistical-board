"""End-to-end test of the statistical board's judge-led round loop using
--dry-run: every LLM call is stubbed (stat_board/llm.py `_stub_reply`), but the
Analyst/Verifier's engine tool is still bound to the real dataset, so this
exercises the full orchestration flow with zero network calls."""

from __future__ import annotations

import asyncio

from stat_board.orchestrator import run
from stat_board.transcript import Transcript


def test_dry_run_single_round_produces_a_complete_transcript(long_csv):
    transcript = asyncio.run(run(
        "Do the groups differ?", str(long_csv),
        group_col="group", value_col="value", max_rounds=1, dry_run=True))

    assert isinstance(transcript, Transcript)
    assert transcript.hypothesis
    assert transcript.plan
    assert len(transcript.key_questions) > 0
    assert len(transcript.rounds) == 1

    rnd = transcript.rounds[0]
    assert rnd.analyst_notes
    assert rnd.frequentist
    assert rnd.assumptions
    assert rnd.bayesian
    assert rnd.verification
    assert rnd.decision in {"iterate", "finalize"}
    assert transcript.final_report


def test_dry_run_stops_early_when_the_judge_finalizes(long_csv):
    # The stat_board dry-run stub always "decides" finalize (unlike storm's,
    # which always iterates) -> the loop must stop after round 1 even though
    # up to 3 rounds were allowed.
    transcript = asyncio.run(run(
        "Do the groups differ?", str(long_csv),
        group_col="group", value_col="value", max_rounds=3, dry_run=True))
    assert len(transcript.rounds) == 1
    assert transcript.rounds[0].decision == "finalize"


def test_dry_run_works_on_wide_csv_with_no_group_or_value_col(wide_csv):
    transcript = asyncio.run(run(
        "Do A, B, and C differ?", str(wide_csv), max_rounds=1, dry_run=True))
    assert transcript.final_report


def test_dry_run_multifactor_leaves_formula_empty(multifactor_csv):
    # Dry-run never fits a real model (the stub's model_formula is always ""),
    # so a multi-factor run must stay on the one-way appendix path in
    # transcript.save() rather than claiming a model that was never fit.
    transcript = asyncio.run(run(
        "Does treatment x sex affect score?", str(multifactor_csv),
        factors=["treatment", "sex"], value_col="score", max_rounds=1, dry_run=True))
    assert transcript.formula == ""
    assert transcript.factors == []
    assert transcript.final_report
