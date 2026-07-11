"""Data loading and shaping.

The engine works on *named groups* of numeric values (the common shape for
comparison tests) and, where a test needs the raw table (regression,
contingency), on a pandas DataFrame. This module turns the several shapes a user
might hand us — long CSV, wide CSV, JSON — into those two forms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

Groups = dict[str, list[float]]


class DataError(ValueError):
    """Raised when input data can't be parsed into the requested shape."""


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame, unchanged."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"file not found: {p}")
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    return pd.read_csv(p)


def _clean(values: Any) -> list[float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr.tolist()


def load_groups(
    path: str | Path,
    *,
    group_col: str | None = None,
    value_col: str | None = None,
) -> Groups:
    """Return an ordered dict of {group name: [values]}.

    Accepts:
      * JSON — either ``{"A": [...], "B": [...]}`` or ``{"groups": {...}}``.
      * Long CSV — pass ``group_col`` and ``value_col`` (one row per observation).
        If a two-column CSV has exactly one non-numeric column, it's inferred as
        the group column automatically.
      * Wide CSV — every numeric column becomes a group (header = group name).
    """
    p = Path(path)
    if not p.exists():
        raise DataError(f"file not found: {p}")

    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text())
        if isinstance(raw, dict) and "groups" in raw and isinstance(raw["groups"], dict):
            raw = raw["groups"]
        if not isinstance(raw, dict):
            raise DataError("JSON must be an object mapping group names to arrays")
        groups = {str(k): _clean(v) for k, v in raw.items()}
        return _validate(groups)

    df = pd.read_csv(p)

    if group_col or value_col:
        if not (group_col and value_col):
            raise DataError("provide both --group-col and --value-col, or neither")
        if group_col not in df.columns or value_col not in df.columns:
            raise DataError(
                f"columns {group_col!r}/{value_col!r} not in {list(df.columns)}"
            )
        groups = {
            str(name): _clean(sub[value_col])
            for name, sub in df.groupby(group_col, sort=False)
        }
        return _validate(groups)

    # No columns specified: infer shape.
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(df.columns) == 2 and len(non_numeric) == 1:
        gcol = non_numeric[0]
        vcol = [c for c in df.columns if c != gcol][0]
        groups = {
            str(name): _clean(sub[vcol])
            for name, sub in df.groupby(gcol, sort=False)
        }
        return _validate(groups)

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # A tidy table that mixes categorical and numeric columns is ambiguous for
    # one-factor loading — we can't guess which column is the factor and which is
    # the outcome. Fail loudly with guidance instead of silently treating every
    # numeric column (IDs, covariates, ...) as a comparison group.
    if non_numeric:
        raise DataError(
            f"ambiguous multi-column table (categorical columns {non_numeric}, "
            f"numeric columns {numeric}). For a ONE-factor comparison, pass "
            f"--group-col <factor> --value-col <outcome>. For a MULTI-factor design, "
            f"use the two-way-anova / ancova / regression commands (they take "
            f"--value / --factor / --covariate)."
        )

    # Wide: every column is numeric → each is a group.
    if not numeric:
        raise DataError("no numeric columns found; is this a long table? pass --group-col/--value-col")
    groups = {str(c): _clean(df[c]) for c in numeric}
    return _validate(groups)


def _validate(groups: Groups) -> Groups:
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    if len(groups) == 0:
        raise DataError("no non-empty groups found after cleaning")
    return groups


def require_two(groups: Groups) -> tuple[str, list[float], str, list[float]]:
    """Return the first two groups, or raise if fewer than two exist."""
    if len(groups) < 2:
        raise DataError(f"need at least 2 groups, got {len(groups)}")
    (n1, v1), (n2, v2) = list(groups.items())[:2]
    return n1, v1, n2, v2
