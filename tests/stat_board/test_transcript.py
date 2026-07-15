from __future__ import annotations

import json

from stat_board.transcript import RoundRecord, Transcript, _render_transcript_markdown, _slug, save


def test_slug_lowercases_and_hyphenates():
    assert _slug("Do The Three Groups Differ?") == "do-the-three-groups-differ"


def test_slug_falls_back_to_report_for_empty_input():
    assert _slug("???") == "report"


def _sample_transcript():
    t = Transcript(
        question="Do the groups differ?",
        model="dry-run",
        hypothesis="H0: means are equal.",
        plan="Path: describe -> assumptions -> anova.",
        key_questions=["Are variances equal?", "Is n adequate?"],
        final_report="# Do the groups differ?\n\nBody text.",
    )
    t.rounds.append(RoundRecord(
        number=1, analyst_notes="Ran describe and anova.",
        analyst_tool_calls=[{"command": "describe", "input": {}, "result": "{}"},
                           {"command": "anova", "input": {}, "result": "{}"}],
        frequentist="Looks fine.", assumptions="Normality OK.", bayesian="BF10=3.2.",
        verification="Reproduced.", judge_rationale="All checks out.",
        decision="finalize", next_instructions="", draft=t.final_report,
    ))
    return t


def test_save_writes_report_and_transcript_files(tmp_path, long_csv):
    t = _sample_transcript()
    t.data_path, t.group_col, t.value_col = str(long_csv), "group", "value"

    paths = save(t, tmp_path)

    assert set(paths) == {"pdf", "md", "json", "transcript_md", "transcript_pdf"}
    for key in paths:
        assert paths[key].exists()
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"
    assert paths["transcript_pdf"].read_bytes()[:4] == b"%PDF"

    md_text = paths["md"].read_text()
    assert "Do the groups differ?" in md_text
    assert "1 round(s)" in md_text

    data = json.loads(paths["json"].read_text())
    assert data["question"] == "Do the groups differ?"
    assert len(data["rounds"]) == 1
    assert data["plan"] == "Path: describe -> assumptions -> anova."

    transcript_md = paths["transcript_md"].read_text()
    assert "# Transcript — Do the groups differ?" in transcript_md
    assert "Ran describe and anova." in transcript_md
    assert "Frequentist critic" in transcript_md
    assert "Judge — FINALIZE" in transcript_md


def test_render_transcript_markdown_includes_every_role_each_round():
    t = _sample_transcript()
    md = _render_transcript_markdown(t)

    assert "Hypothesis:" in md
    assert "- Are variances equal?" in md
    assert "- Is n adequate?" in md
    assert "Path: describe -> assumptions -> anova." in md
    assert "Ran 2 engine call(s): `describe`, `anova`" in md
    for section in ("Analyst", "Frequentist critic", "Assumptions skeptic",
                    "Bayesian critic", "Verifier", "Judge — FINALIZE"):
        assert section in md


def test_render_transcript_markdown_handles_zero_rounds():
    t = Transcript(question="No rounds yet", model="dry-run")
    md = _render_transcript_markdown(t)
    assert "# Transcript — No rounds yet" in md
    assert "0 round(s)" in md


def test_save_without_data_path_still_renders_a_plain_pdf(tmp_path):
    t = Transcript(question="No dataset", model="dry-run", final_report="# Report\n")
    paths = save(t, tmp_path)
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"


def test_save_with_multifactor_formula_renders_the_design_diagnostics_appendix(
        tmp_path, multifactor_csv):
    import fitz

    t = Transcript(
        question="Does treatment x sex affect score?", model="dry-run",
        final_report="# Does treatment x sex affect score?\n\nBody text.",
        data_path=str(multifactor_csv),
        formula="score ~ C(treatment) * C(sex)", factors=["treatment", "sex"], typ=2,
    )
    paths = save(t, tmp_path)
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"

    assets_dir = paths["pdf"].parent / f"{paths['pdf'].stem}_assets"
    assert (assets_dir / "fig_pareto.png").exists()
    # The .md source never carries the appendix (only the rendered PDF does --
    # see appendix.build_multifactor's own docstring/contract).
    assert "Appendix C" not in paths["md"].read_text()

    pdf_text = "\n".join(page.get_text() for page in fitz.open(paths["pdf"]))
    assert "Appendix C" in pdf_text
    assert "Design coverage" in pdf_text
    assert "Tested combinations ranked by predicted response" in pdf_text
