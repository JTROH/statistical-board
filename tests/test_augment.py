"""Analysis diagnostics -> the next design.

The fixtures below are the *real* engine output for a hand-built dataset, not
hand-written JSON, so these tests fail if ``stat_board``'s diagnostic output
shape ever drifts away from what ``augment`` reads.
"""

from __future__ import annotations

import pytest

from doe_advisor import augment
from doe_advisor.candidates import top_options
from doe_advisor.intake import spec_from_dict
from stat_board.engine import analyses

FORMULA = "y ~ x1 * x2"
FACTORS = ["x1", "x2"]


@pytest.fixture
def curved_csv(tmp_path):
    """A 2^2 factorial (each corner twice) plus 3 centre points with a deliberate
    +6 bump at the centre.

    The bump is far larger than the corner spread, so the curvature contrast is
    significant by construction, and the response rises with both factors, so the
    best tested corner is always the (high, high) one -- both optima land on a
    boundary. Every trigger below is therefore guaranteed, not incidental.
    """
    rows = ["x1,x2,y"]
    for x1, x2, y in [(-1, -1, 10), (1, -1, 14), (-1, 1, 16), (1, 1, 20)]:
        rows += [f"{x1},{x2},{y}", f"{x1},{x2},{y + 0.4}"]
    rows += ["0,0,21", "0,0,21.4", "0,0,21.2"]
    p = tmp_path / "curved.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def diagnostics(curved_csv):
    return {
        "coverage": analyses.design_coverage(curved_csv, FACTORS, value="y"),
        "optimum": analyses.doe_optimum(curved_csv, FORMULA, FACTORS, "y"),
        "vif": analyses.vif_table(curved_csv, FORMULA),
        "predict": analyses.predict_table(curved_csv, FORMULA),
    }


# ---- findings ---- #

def test_boundary_optimum_is_detected(diagnostics):
    fired = augment.findings(**diagnostics)
    boundary = next(f for f in fired if f["trigger"] == "optimum_at_boundary")
    assert boundary["source"] == "doe-optimum"
    assert set(boundary["factors"]) == {"x1", "x2"}


def test_curvature_is_detected(diagnostics):
    fired = augment.findings(**diagnostics)
    assert any(f["trigger"] == "curvature_detected" for f in fired)


def test_no_center_points_reads_as_untested_rather_than_absent(curved_csv, tmp_path):
    """A design with no centre runs has not *disproved* curvature -- it never
    looked. That must not be silently treated as 'no curvature'."""
    corners = tmp_path / "corners.csv"
    corners.write_text("x1,x2,y\n-1,-1,10\n1,-1,14\n-1,1,16\n1,1,20\n", encoding="utf-8")
    coverage = analyses.design_coverage(str(corners), FACTORS, value="y")
    fired = augment.findings(coverage=coverage)
    assert any(f["trigger"] == "curvature_untested" for f in fired)


def test_clean_diagnostics_fire_nothing():
    assert augment.findings() == []


# ---- the proposed form ---- #

def test_ranges_are_widened_only_past_the_edge_that_won(diagnostics):
    out = augment.form_from_diagnostics(**diagnostics)
    by_name = {f["name"]: f for f in out["form"]["factors"]}
    # Tested levels were -1, 0, +1; the optimum sat high, so only the top moves.
    assert by_name["x1"]["low"] == pytest.approx(-1.0)
    assert by_name["x1"]["high"] == pytest.approx(1.0 + 2.0 * augment.RANGE_EXTENSION)


def test_detected_curvature_upgrades_the_model_to_quadratic(diagnostics):
    out = augment.form_from_diagnostics(**diagnostics)
    assert out["form"]["model_order"] == "quadratic"


def test_units_are_carried_through_when_supplied(diagnostics):
    out = augment.form_from_diagnostics(**diagnostics, factor_units={"x1": "g/L"})
    by_name = {f["name"]: f for f in out["form"]["factors"]}
    assert by_name["x1"]["units"] == "g/L"
    assert by_name["x2"]["units"] == ""


def test_every_rationale_line_traces_to_a_finding(diagnostics):
    out = augment.form_from_diagnostics(**diagnostics)
    assert out["rationale"] == [f["detail"] for f in out["findings"]]


def test_coverage_is_required_because_it_carries_the_tested_levels():
    with pytest.raises(ValueError, match="design-coverage output is required"):
        augment.form_from_diagnostics(coverage=None)


def test_categorical_factors_are_skipped_not_widened(tmp_path):
    p = tmp_path / "cat.csv"
    p.write_text("x1,grp,y\n-1,a,10\n1,a,14\n-1,b,16\n1,b,20\n", encoding="utf-8")
    coverage = analyses.design_coverage(str(p), ["x1", "grp"], value="y")
    out = augment.form_from_diagnostics(coverage=coverage)
    assert out["skipped_factors"] == ["grp"]
    assert [f["name"] for f in out["form"]["factors"]] == ["x1"]


# ---- the round trip ---- #

def test_proposed_form_is_accepted_by_intake_and_yields_a_curvature_design(diagnostics):
    """The whole point: the diagnostics must produce a design, not just advice."""
    out = augment.form_from_diagnostics(
        **diagnostics,
        response={"name": "y", "target_effect": 4.0, "noise_sd": 1.0, "goal": "maximize"},
        max_runs=30)
    spec = spec_from_dict(out["form"])
    assert spec.model_order.value == "quadratic"

    options = top_options(spec, limit=3)
    assert options, "a quadratic spec must yield at least one design"
    # Curvature needs a design that can fit it -- CCD or Box-Behnken, never a
    # plain two-level factorial.
    families = {o.design.family for o in options}
    assert families & {"central_composite", "box_behnken"}
