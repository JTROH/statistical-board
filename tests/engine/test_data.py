from __future__ import annotations

import json

import pytest

from stat_board.engine.data import DataError, load_dataframe, load_groups, require_two


def test_load_groups_wide_csv(wide_csv):
    groups = load_groups(wide_csv)
    assert set(groups) == {"A", "B", "C"}
    assert all(len(v) > 0 for v in groups.values())


def test_load_groups_long_csv(long_csv):
    groups = load_groups(long_csv, group_col="group", value_col="value")
    assert set(groups) == {"A", "B", "C"}


def test_load_groups_long_csv_infers_group_column_from_two_column_table(tmp_path):
    p = tmp_path / "long_infer.csv"
    p.write_text("group,value\nA,1\nA,2\nB,3\nB,4\n")
    groups = load_groups(p)
    assert groups == {"A": [1.0, 2.0], "B": [3.0, 4.0]}


def test_load_groups_json_plain_dict(two_json):
    groups = load_groups(two_json)
    assert set(groups) == {"treatment", "control"}
    assert len(groups["treatment"]) == 30


def test_load_groups_json_nested_groups_key(tmp_path):
    p = tmp_path / "nested.json"
    p.write_text(json.dumps({"groups": {"A": [1, 2, 3], "B": [4, 5, 6]}}))
    groups = load_groups(p)
    assert groups == {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}


def test_load_groups_missing_file_raises(tmp_path):
    with pytest.raises(DataError, match="file not found"):
        load_groups(tmp_path / "nope.csv")


def test_load_groups_ambiguous_table_raises(multifactor_csv):
    with pytest.raises(DataError, match="ambiguous"):
        load_groups(multifactor_csv)


def test_load_groups_group_col_without_value_col_raises(long_csv):
    with pytest.raises(DataError, match="provide both"):
        load_groups(long_csv, group_col="group")


def test_load_groups_unknown_column_raises(long_csv):
    with pytest.raises(DataError, match="not in"):
        load_groups(long_csv, group_col="nope", value_col="value")


def test_load_groups_all_numeric_wide_with_no_numeric_columns_raises(tmp_path):
    p = tmp_path / "text_only.csv"
    p.write_text("a,b\nfoo,bar\nbaz,qux\n")
    with pytest.raises(DataError):
        load_groups(p)


def test_load_dataframe_csv(reg_csv):
    df = load_dataframe(reg_csv)
    assert list(df.columns) == ["x", "y"]
    assert len(df) > 0


def test_load_dataframe_missing_file_raises(tmp_path):
    with pytest.raises(DataError, match="file not found"):
        load_dataframe(tmp_path / "nope.csv")


def test_require_two_returns_first_two_groups():
    groups = {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}
    n1, v1, n2, v2 = require_two(groups)
    assert (n1, v1, n2, v2) == ("A", [1.0, 2.0], "B", [3.0, 4.0])


def test_require_two_raises_with_fewer_than_two_groups():
    with pytest.raises(DataError, match="need at least 2 groups"):
        require_two({"A": [1.0, 2.0]})
