"""The design engine's CLI — same JSON-in/JSON-out contract as stat_board.engine."""

from __future__ import annotations

import json

import pytest

from doe_advisor.cli import main


@pytest.fixture
def spec_file(tmp_path):
    """A three-factor study with a budget that admits more than one design.

    target_effect / noise_sd = 2.0 standardised, comfortably detectable, so the
    options list is never empty and power assertions are stable.
    """
    payload = {
        "factors": [
            {"name": "glucose", "low": 2, "high": 8, "units": "g/L"},
            {"name": "glutamine", "low": 2, "high": 6, "units": "mM"},
            {"name": "serum", "low": 2, "high": 10, "units": "%"},
        ],
        "responses": [
            {"name": "titre", "units": "g/L", "target_effect": 8, "noise_sd": 4, "goal": "maximize"}
        ],
        "model_order": "interaction",
        "max_runs": 24,
        "n_center_points": 3,
        "expected_run_losses": 1,
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---- metadata commands (no spec needed) ---- #

def test_presets_command(capsys):
    assert main(["presets"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "presets"
    assert out["presets"][0]["id"] == "blank"  # blank template is always first


def test_capabilities_command(capsys):
    assert main(["capabilities"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "capabilities"
    assert set(out["axis_weights"]) == {"power", "aliasing", "prediction", "robustness"}


# ---- design commands ---- #

def test_options_command(spec_file, capsys):
    assert main(["options", "--spec", str(spec_file)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "options"
    assert out["n_factors"] == 3
    assert 1 <= len(out["options"]) <= 3
    assert any("recommended" in o["roles"] for o in out["options"])


def test_options_respects_limit(spec_file, capsys):
    assert main(["options", "--spec", str(spec_file), "--limit", "1"]) == 0
    assert len(json.loads(capsys.readouterr().out)["options"]) == 1


def test_candidates_command_reports_rejects_too(spec_file, capsys):
    assert main(["candidates", "--spec", str(spec_file)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "candidates"
    assert out["n_candidates"] == len(out["candidates"]) + len(out["rejected"])


def test_properties_command_defaults_to_the_recommended_design(spec_file, capsys):
    assert main(["properties", "--spec", str(spec_file)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "properties"
    assert "recommended" in out["roles"]


def test_properties_selects_by_rank(spec_file, capsys):
    assert main(["properties", "--spec", str(spec_file), "--design", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["analysis"] == "properties"


def test_runsheet_writes_csv_and_names_the_next_step(spec_file, tmp_path, capsys):
    out_csv = tmp_path / "runs.csv"
    assert main(["runsheet", "--spec", str(spec_file), "--out", str(out_csv)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "runsheet"
    assert out_csv.exists()

    header = out_csv.read_text().splitlines()[0].split(",")
    assert header == out["columns"]
    assert header[:2] == ["run_order", "run_type"]
    # The quoted handoff command must name columns the CSV really has.
    for column in out["next_step"]["factor_columns"]:
        assert column in header
    assert "stat_board" in out["next_step"]["cli"]


# ---- the error contract ---- #

def test_missing_spec_file_exits_two_with_json_error(tmp_path, capsys):
    assert main(["options", "--spec", str(tmp_path / "nope.json")]) == 2
    err = json.loads(capsys.readouterr().err)
    assert err["error"] == "IntakeError"


def test_malformed_spec_exits_two(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["options", "--spec", str(bad)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "IntakeError"


def test_one_factor_spec_is_rejected_with_a_readable_message(tmp_path, capsys):
    thin = tmp_path / "thin.json"
    thin.write_text(json.dumps({"factors": [{"name": "pH", "low": 6, "high": 8}], "responses": []}),
                    encoding="utf-8")
    assert main(["options", "--spec", str(thin)]) == 2
    assert "at least two factors" in json.loads(capsys.readouterr().err)["message"]


def test_unknown_option_selector_exits_two(spec_file, tmp_path, capsys):
    assert main(["runsheet", "--spec", str(spec_file), "--option", "does-not-exist",
                 "--out", str(tmp_path / "x.csv")]) == 2
    assert "no option matching" in json.loads(capsys.readouterr().err)["message"]
