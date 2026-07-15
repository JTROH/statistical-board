from __future__ import annotations

import json

from stat_board.transcript import RoundRecord, Transcript, _slug, save


def test_slug_lowercases_and_hyphenates():
    assert _slug("Do The Three Groups Differ?") == "do-the-three-groups-differ"


def test_slug_falls_back_to_report_for_empty_input():
    assert _slug("???") == "report"


def test_save_writes_md_pdf_and_json_transcript(tmp_path, long_csv):
    t = Transcript(
        question="Do the groups differ?",
        model="dry-run",
        hypothesis="H0: means are equal.",
        key_questions=["Are variances equal?"],
        final_report="# Do the groups differ?\n\nBody text.",
        data_path=str(long_csv), group_col="group", value_col="value",
    )
    t.rounds.append(RoundRecord(
        number=1, analyst_notes="notes", analyst_tool_calls=[],
        frequentist="f", assumptions="a", bayesian="b", verification="v",
        judge_rationale="r", decision="finalize", next_instructions="", draft=t.final_report,
    ))

    paths = save(t, tmp_path)

    assert set(paths) == {"pdf", "md", "json"}
    assert paths["md"].exists()
    assert paths["pdf"].exists()
    assert paths["json"].exists()
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"

    md_text = paths["md"].read_text()
    assert "Do the groups differ?" in md_text
    assert "1 round(s)" in md_text

    data = json.loads(paths["json"].read_text())
    assert data["question"] == "Do the groups differ?"
    assert len(data["rounds"]) == 1


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
