"""Wire the cross-validation suite into pytest.

The R leg is skipped when R or its packages are missing, so the suite stays
runnable on a machine that only has Python — but when R *is* present, a
disagreement fails the build rather than waiting to be noticed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from validation.cases import BOX_BEHNKEN_CASES, CENTRAL_COMPOSITE_CASES, DSD_CASES, FRACTIONAL_CASES
from validation.compare import (
    Comparison,
    check_dsd,
    compare_box_behnken,
    compare_central_composite,
    compare_fractional,
)
from validation.run_tool import (
    box_behnken_results,
    central_composite_results,
    dsd_results,
    fractional_results,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = 'cat(all(sapply(c("FrF2","rsm","jsonlite"), requireNamespace, quietly=TRUE)))'
    try:
        result = subprocess.run(
            ["Rscript", "-e", probe], capture_output=True, text=True, timeout=120, cwd=REPO_ROOT
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "TRUE" in result.stdout


requires_r = pytest.mark.skipif(not _r_available(), reason="R with FrF2/rsm/jsonlite not available")


@pytest.fixture(scope="module")
def r_results(tmp_path_factory):
    """Run the R reference script once and return its parsed output."""
    subprocess.run(
        ["Rscript", "validation/validate_r.R"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=True,
    )
    return json.loads((REPO_ROOT / "validation" / "results" / "r.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The tool's own side
# --------------------------------------------------------------------------


def test_every_frozen_case_produces_a_result():
    assert len(fractional_results()) == len(FRACTIONAL_CASES)
    assert len(box_behnken_results()) == len(BOX_BEHNKEN_CASES)
    assert len(central_composite_results()) == len(CENTRAL_COMPOSITE_CASES)
    assert len(dsd_results()) == len(DSD_CASES)


def test_word_length_patterns_account_for_every_defining_word():
    """A 2^(k-p) design has exactly 2^p - 1 defining words; the pattern must add
    up to that, or words are being lost or double-counted."""
    for entry in fractional_results():
        assert sum(entry["word_length_pattern"]) == 2 ** entry["n_generators"] - 1


def test_resolution_is_the_first_non_zero_entry_of_the_pattern():
    for entry in fractional_results():
        pattern = entry["word_length_pattern"]
        first = next(i for i, count in enumerate(pattern) if count > 0)
        assert entry["resolution"] == first + 3


def test_definitive_screening_designs_meet_their_published_properties():
    """Runs without R: the DSD claims are checkable from the matrix alone."""
    comparison = Comparison()
    check_dsd(dsd_results(), comparison)
    assert comparison.failures == [], comparison.failures


# --------------------------------------------------------------------------
# Against R
# --------------------------------------------------------------------------


@requires_r
def test_fractional_factorials_agree_with_frf2(r_results):
    """The strict one: word-length patterns, not just resolutions."""
    comparison = Comparison()
    compare_fractional(fractional_results(), r_results["fractional"], comparison)
    assert comparison.rows, "no cases were compared"
    assert comparison.failures == [], comparison.failures


@requires_r
def test_box_behnken_designs_agree_with_rsm(r_results):
    comparison = Comparison()
    compare_box_behnken(box_behnken_results(), r_results["box_behnken"], comparison)
    assert comparison.rows
    assert comparison.failures == [], comparison.failures


@requires_r
def test_central_composite_designs_agree_with_rsm(r_results):
    comparison = Comparison()
    compare_central_composite(central_composite_results(), r_results["central_composite"], comparison)
    assert comparison.rows
    assert comparison.failures == [], comparison.failures


@requires_r
def test_no_centre_points_means_no_centre_points(r_results):
    """Regression: the generators once forced a centre point even when asked for
    zero, so every run count came out one too high against rsm."""
    ours = {e["case"]: e for e in box_behnken_results()}
    theirs = {e["case"]: e for e in r_results["box_behnken"]}
    for case in sorted(set(ours) & set(theirs)):
        assert ours[case]["n_runs"] == theirs[case]["n_runs"]
