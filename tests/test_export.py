"""Run-sheet CSV export — the handoff from the design half to the analysis half."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from doe_advisor import export
from doe_advisor.designs.classical import central_composite, full_factorial
from doe_advisor.designs.spec import DesignSpec, Factor, Response


@pytest.fixture
def spec() -> DesignSpec:
    """Two factors with deliberately un-CSV-safe names and units, one response.

    Ranges are symmetric about a round centre (5 and 4), so every decoded value
    below is exact rather than approximate.
    """
    return DesignSpec(
        factors=[
            Factor(name="Glucose", low=2.0, high=8.0, units="g/L"),
            Factor(name="Glutamine", low=2.0, high=6.0, units="mM"),
        ],
        responses=[Response(name="Titre", units="g/L", target_effect=2.0, noise_sd=1.0)],
    )


def test_column_names_are_csv_safe_and_carry_units(spec):
    design = full_factorial(2, n_center=0)
    header, _ = export.run_sheet_rows(design, spec, seed=0)
    assert header == [
        "run_order", "run_type",
        "glucose_coded", "glucose_g_l",
        "glutamine_coded", "glutamine_mm",
        "titre_g_l",
    ]


def test_response_columns_are_left_empty_for_the_bench(spec):
    design = full_factorial(2, n_center=0)
    _, rows = export.run_sheet_rows(design, spec, seed=0)
    assert all(row[-1] == "" for row in rows)


def test_every_run_appears_exactly_once_in_randomised_order(spec):
    design = full_factorial(2, n_center=2)
    _, rows = export.run_sheet_rows(design, spec, seed=7)
    assert [r[0] for r in rows] == list(range(1, design.n_runs + 1))
    # The four corners of a 2^2, each exactly once, regardless of the shuffle.
    corners = sorted((r[3], r[5]) for r in rows if r[1] == "factorial")
    assert corners == [("2", "2"), ("2", "6"), ("8", "2"), ("8", "6")]


def test_run_type_is_derived_from_the_coded_row():
    assert export.run_type(np.array([0.0, 0.0])) == "center"
    assert export.run_type(np.array([-1.0, 1.0])) == "factorial"
    assert export.run_type(np.array([1.682, 0.0])) == "axial"


def test_central_composite_export_labels_all_three_run_types(spec):
    design = central_composite(2, n_center=3)
    _, rows = export.run_sheet_rows(design, spec, seed=0)
    kinds = {r[1] for r in rows}
    assert kinds == {"factorial", "axial", "center"}
    assert sum(1 for r in rows if r[1] == "center") == 3


def test_csv_round_trips_through_the_csv_module(spec, tmp_path):
    design = central_composite(2, n_center=3)
    out = export.run_sheet_csv(design, spec, tmp_path / "nested" / "runs.csv", seed=0)
    assert out.exists()
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0] == "run_order"
    assert len(rows) == design.n_runs + 1


def test_export_is_deterministic_for_a_given_seed(spec, tmp_path):
    design = central_composite(2, n_center=3)
    a = export.run_sheet_csv(design, spec, tmp_path / "a.csv", seed=3).read_text()
    b = export.run_sheet_csv(design, spec, tmp_path / "b.csv", seed=3).read_text()
    c = export.run_sheet_csv(design, spec, tmp_path / "c.csv", seed=4).read_text()
    assert a == b
    assert a != c


def test_analysis_hint_names_the_natural_unit_columns(spec):
    hint = export.analysis_hint(spec)
    assert hint["factor_columns"] == ["glucose_g_l", "glutamine_mm"]
    assert hint["response_columns"] == ["titre_g_l"]
    assert hint["formula"] == "titre_g_l ~ glucose_g_l * glutamine_mm"
    # The quoted commands must name the same columns the CSV actually carries.
    header, _ = export.run_sheet_rows(full_factorial(2), spec, seed=0)
    for column in hint["factor_columns"] + hint["response_columns"]:
        assert column in header
        assert column in hint["cli"]
