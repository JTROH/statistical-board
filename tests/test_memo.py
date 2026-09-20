"""The memo is the artefact that leaves the tool, so it is tested for internal
consistency above all else.

A memo whose prose and whose numbers disagree is worse than no memo: it will be
read in a meeting by someone who was not in the room and cannot check it.
"""

from __future__ import annotations

import re

import pytest

from doe_advisor.designs.spec import DesignSpec, Factor, ModelOrder, Response
from doe_advisor.memo import build_memo


def make_spec(order=ModelOrder.MAIN, max_runs=24, k=4, target=0.5, noise=0.25):
    factors = [
        Factor("pH", 6.8, 7.4),
        Factor("DO", 30, 60, "%"),
        Factor("temp", 35, 37, "C"),
        Factor("feed", 2, 6, "mL/day"),
    ][:k]
    return DesignSpec(
        factors=factors,
        responses=[Response("titer", "g/L", target_effect=target, noise_sd=noise)],
        model_order=order,
        max_runs=max_runs,
        n_center_points=3,
        expected_run_losses=2,
    )


@pytest.fixture
def memo():
    return build_memo(make_spec(), prefer_llm=False)


# --------------------------------------------------------------------------
# Internal consistency
# --------------------------------------------------------------------------


def test_each_section_states_one_run_count(memo):
    """Regression: two designs once shared the display name 'Full factorial
    2^4', so the narrative lookup collided and a section's prose claimed 19 runs
    while its fact line said 16.
    """
    sections = memo.markdown.split("### ")[1:]
    assert sections
    for section in sections:
        if not section.strip():
            continue
        prose_runs = re.search(r"\*(\d+) runs, fitting", section)
        fact_runs = re.search(r"<small>(\d+) runs", section)
        if prose_runs and fact_runs:
            assert prose_runs.group(1) == fact_runs.group(1), f"run counts disagree in: {section[:80]}"


def test_option_names_are_unique(memo):
    names = [o.design.name for o in memo.options]
    assert len(set(names)) == len(names)


def test_comparison_table_row_count_matches_the_options(memo):
    table = memo.markdown.split("## The options")[1].split("##")[0]
    rows = [line for line in table.splitlines() if line.startswith("| **")]
    assert len(rows) == len(memo.options)


def test_every_option_gets_a_detail_section(memo):
    for option in memo.options:
        assert option.design.name in memo.markdown


def test_run_sheet_lists_every_run_of_the_recommended_design(memo):
    recommended = next((o for o in memo.options if "recommended" in o.roles), memo.options[0])
    sheet = memo.markdown.split("## Run sheet")[1]
    rows = [line for line in sheet.splitlines() if re.match(r"^\| \d+ \|", line)]
    assert len(rows) == recommended.n_runs


def test_run_sheet_is_in_real_units_not_coded(memo):
    """'pH 6.8' is checkable against a process; '-1' is not."""
    sheet = memo.markdown.split("## Run sheet")[1]
    assert "6.8" in sheet or "7.4" in sheet
    assert "| -1 |" not in sheet


def test_run_sheet_columns_match_the_factors(memo):
    spec = make_spec()
    sheet = memo.markdown.split("## Run sheet")[1]
    header = next(line for line in sheet.splitlines() if line.startswith("| Run |"))
    for factor in spec.factors:
        assert factor.name in header


# --------------------------------------------------------------------------
# Required content
# --------------------------------------------------------------------------


def test_memo_states_the_question(memo):
    assert "## The question" in memo.markdown
    for name in ("pH", "DO", "temp", "feed"):
        assert name in memo.markdown


def test_memo_states_the_recommendation(memo):
    assert "## Recommendation" in memo.markdown
    assert sum("recommended" in o.roles for o in memo.options) == 1


def test_memo_records_its_own_provenance(memo):
    """It must be possible to tell later how a memo was produced."""
    assert "## How this was produced" in memo.markdown
    assert "doe-advisor" in memo.markdown
    assert "Scoring weights" in memo.markdown
    assert "Random seed" in memo.markdown


def test_memo_names_the_narration_source(memo):
    assert "**template**" in memo.markdown or "**claude**" in memo.markdown


def test_memo_carries_caveats(memo):
    assert "### Caveats" in memo.markdown
    assert "Randomise the run order" in memo.markdown


def test_aliasing_section_appears_only_when_something_is_confounded():
    clean = build_memo(make_spec(), prefer_llm=False)
    if all(o.properties.aliasing.worst_main_effect_alias == 0 for o in clean.options):
        assert "## What you will not be able to tell apart" not in clean.markdown

    seven = DesignSpec(
        factors=[Factor(f"x{i}", 0, 1) for i in range(7)],
        responses=[Response("titer", "g/L", target_effect=0.6, noise_sd=0.3)],
        model_order=ModelOrder.MAIN,
        max_runs=24,
        n_center_points=3,
        expected_run_losses=1,
    )
    confounded = build_memo(seven, prefer_llm=False)
    if any(o.properties.aliasing.worst_main_effect_alias > 0 for o in confounded.options):
        assert "## What you will not be able to tell apart" in confounded.markdown


# --------------------------------------------------------------------------
# Edge cases and output
# --------------------------------------------------------------------------


def test_impossible_budget_yields_an_explanation_not_a_bad_recommendation():
    memo = build_memo(make_spec(order=ModelOrder.QUADRATIC, max_runs=6), prefer_llm=False)
    assert memo.options == []
    assert "## No viable design" in memo.markdown
    assert "## Run sheet" not in memo.markdown


def test_memo_without_an_effect_size_still_builds():
    spec = make_spec()
    spec.responses = [Response("titer", "g/L")]
    memo = build_memo(spec, prefer_llm=False)
    assert memo.options
    assert "power could not be computed" in memo.markdown


def test_memo_is_deterministic():
    first = build_memo(make_spec(), prefer_llm=False).markdown
    second = build_memo(make_spec(), prefer_llm=False).markdown
    strip = lambda text: re.sub(r"on \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", "", text)  # noqa: E731
    assert strip(first) == strip(second)


def test_memo_writes_markdown(tmp_path, memo):
    path = memo.write(tmp_path / "memo.md")
    assert path.read_text(encoding="utf-8") == memo.markdown


def test_memo_renders_a_pdf_with_figures(tmp_path):
    memo = build_memo(make_spec(), figure_dir=tmp_path / "figs", prefer_llm=False)
    for name in ("options.png", "fds.png", "layout.png", "power.png"):
        assert (tmp_path / "figs" / name).exists()
    pdf = memo.to_pdf(tmp_path / "memo.pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 10_000
    assert pdf.read_bytes()[:5] == b"%PDF-"


def test_memo_renders_a_pdf_without_figures(tmp_path, memo):
    """No figure directory means no image tags, and the render must still work."""
    assert "<img" not in memo.markdown
    assert memo.to_pdf(tmp_path / "plain.pdf").exists()


def test_memo_has_a_power_primer_with_units():
    spec = DesignSpec(
        factors=[Factor("pH", 6.8, 7.4), Factor("DO", 30, 60, "%"), Factor("temp", 35, 37, "C")],
        responses=[Response("titer", "g/L", target_effect=0.6, noise_sd=0.3)],
        max_runs=24,
    )
    md = build_memo(spec, prefer_llm=False).markdown
    assert "## How to read the power figures" in md
    assert "Smallest change seen at 80% power" in md
    assert "Goal: **find out which factors move titer**" in md
    assert "reaches 80% power at your target effect with about **" in md


def test_memo_has_no_power_primer_without_a_noise_sd():
    spec = DesignSpec(
        factors=[Factor("pH", 6.8, 7.4), Factor("DO", 30, 60, "%"), Factor("temp", 35, 37, "C")],
        responses=[Response("titer", "g/L")],
        max_runs=24,
    )
    assert "## How to read the power figures" not in build_memo(spec, prefer_llm=False).markdown
