from __future__ import annotations

import json

from stat_board.engine.cli import main


def test_describe_prints_json_to_stdout_and_exits_zero(long_csv, capsys):
    code = main(["describe", "--data", str(long_csv), "--group-col", "group", "--value-col", "value"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "describe"
    assert set(out["groups"]) == {"A", "B", "C"}


def test_ttest_wide_csv_needs_no_group_or_value_col(wide_csv, capsys):
    code = main(["ttest", "--data", str(wide_csv)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "ttest"


def test_regression_command(reg_csv, capsys):
    code = main(["regression", "--data", str(reg_csv), "--formula", "y ~ x"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "regression"


def test_bad_path_exits_two_with_json_error_on_stderr(tmp_path, capsys):
    code = main(["describe", "--data", str(tmp_path / "nope.csv"),
                 "--group-col", "g", "--value-col", "v"])
    assert code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["error"] == "DataError"


def test_predict_command(reg_csv, capsys):
    code = main(["predict", "--data", str(reg_csv), "--formula", "y ~ x"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "predict_table"


def test_vif_command(reg_csv, capsys):
    code = main(["vif", "--data", str(reg_csv), "--formula", "y ~ x"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "vif"


def test_design_coverage_command(multifactor_csv, capsys):
    code = main(["design-coverage", "--data", str(multifactor_csv),
                 "--factor", "treatment", "--factor", "sex", "--value", "score"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "design_coverage"


def test_doe_optimum_command(multifactor_csv, capsys):
    code = main(["doe-optimum", "--data", str(multifactor_csv),
                 "--formula", "score ~ C(treatment) * C(sex)",
                 "--factor", "treatment", "--factor", "sex", "--value", "score"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "doe_optimum"
