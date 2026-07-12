"""Data preparation: profile a messy file and apply a reproducible cleaning recipe.

The rest of stat_board *reads* a table; this module *fixes* one first — the step
that was previously done by hand (skip preamble rows, parse dates, log-transform
skewed columns, bin into decades, …). Two deterministic entry points:

  * ``profile(path)``  → a JSON report of the file's structure, column types,
    missingness, skew, likely roles, and problems worth fixing.
  * ``apply(path, recipe, out)`` → run an ordered list of cleaning ops and write a
    clean CSV, returning a step-by-step report.

``suggest_recipe(profile)`` turns a profile into a proposed recipe (with a reason
on every step) — the basis for the /stat-prep skill's "propose & confirm" flow and
the ``--auto`` CLI mode. Nothing here is irreversible: it always writes a *new*
file and records exactly what it did.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

Recipe = dict[str, Any]


# --------------------------------------------------------------------------- #
# Reading / structure detection
# --------------------------------------------------------------------------- #

def _detect_read_opts(path: Path) -> dict[str, Any]:
    """Sniff encoding, delimiter, and how many preamble rows precede the header."""
    raw = None
    encoding = "utf-8"
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = path.read_text(encoding=enc)
            encoding = "utf-8" if enc == "utf-8-sig" else enc
            break
        except (UnicodeDecodeError, OSError):
            continue
    if raw is None:
        raise ValueError(f"could not read {path}")

    sample = "\n".join(raw.splitlines()[:50])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        delimiter = ","

    rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))
    rows = [r for r in rows if r != []][:50]
    nonempty = [sum(1 for c in r if str(c).strip() != "") for r in rows]
    if not nonempty:
        return {"skiprows": 0, "delimiter": delimiter, "encoding": encoding}
    width = max(set(nonempty), key=nonempty.count)  # modal populated-field count
    # The header is the first row that is fully populated (== modal width); rows
    # before it with fewer populated fields are a preamble/banner.
    skiprows = 0
    for i, ne in enumerate(nonempty):
        if ne >= width:
            skiprows = i
            break
    return {"skiprows": int(skiprows), "delimiter": delimiter, "encoding": encoding}


def _read(path: Path, read_opts: dict[str, Any]) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path, skiprows=read_opts.get("skiprows", 0),
                       sep=read_opts.get("delimiter", ","),
                       encoding=read_opts.get("encoding", "utf-8"))


# --------------------------------------------------------------------------- #
# Column heuristics
# --------------------------------------------------------------------------- #

_DATEISH = re.compile(r"date|time|year|day|month|dob|timestamp", re.I)


def _looks_yyyymmdd(s: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(s):
        return False
    v = s.dropna()
    return len(v) > 0 and bool(((v >= 19000101) & (v <= 21001231)).mean() > 0.9) \
        and bool((v.astype(str).str.len() == 8).mean() > 0.9)


def _looks_datestr(s: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() > 0.8)


def _numeric_as_text(s: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    cleaned = sample.str.replace(r"[,$%\s]", "", regex=True)
    return bool(pd.to_numeric(cleaned, errors="coerce").notna().mean() > 0.8)


def _role(name: str, s: pd.Series, nrows: int) -> str:
    nunique = s.nunique(dropna=True)
    if _DATEISH.search(name) and (_looks_yyyymmdd(s) or _looks_datestr(s)):
        return "date"
    if _looks_yyyymmdd(s) or _looks_datestr(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        if nunique >= max(0.95 * nrows, nrows - 1) and re.search(r"id|index|no\.?$", name, re.I):
            return "id"
        return "numeric"
    # object / categorical
    if nunique >= 0.9 * nrows:
        return "id_or_label"          # e.g. a Name column
    if nunique <= 20:
        return "factor"
    return "high_cardinality"


def _column_profile(name: str, s: pd.Series, nrows: int) -> dict[str, Any]:
    role = _role(name, s, nrows)
    prof: dict[str, Any] = {
        "dtype": str(s.dtype),
        "role_guess": role,
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean() * 100), 1),
        "n_unique": int(s.nunique(dropna=True)),
        "flags": [],
    }
    if pd.api.types.is_numeric_dtype(s) and role not in ("date", "id"):
        v = s.dropna().astype(float)
        if len(v) > 0:
            sk = float(stats.skew(v)) if len(v) > 2 else 0.0
            prof.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                        median=float(v.median()), skew=round(sk, 2),
                        n_zero=int((v == 0).sum()), n_negative=int((v < 0).sum()))
            if abs(sk) > 1:
                prof["flags"].append("high_skew:log-transform"
                                     if (v > 0).all() else "high_skew:log1p")
    else:
        prof["sample_values"] = [str(x) for x in s.dropna().unique()[:8]]
        if role == "date":
            prof["flags"].append("date:parse->year/decade")
        if _numeric_as_text(s):
            prof["flags"].append("numeric_as_text:coerce")
    if prof["pct_missing"] > 20:
        prof["flags"].append("high_missing")
    if prof["n_unique"] <= 1:
        prof["flags"].append("constant")
    return prof


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #

def profile(path: str | Path) -> dict[str, Any]:
    """Inspect a file and return a structural + per-column profile with flags."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    read_opts = ({"skiprows": 0} if path.suffix.lower() == ".json"
                 else _detect_read_opts(path))
    df = _read(path, read_opts)
    nrows = len(df)
    columns = {str(c): _column_profile(str(c), df[c], nrows) for c in df.columns}
    return {
        "path": str(path),
        "read": read_opts,
        "n_rows": int(nrows),
        "n_cols": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
        "notes": _profile_notes(read_opts, columns),
    }


def _profile_notes(read_opts: dict[str, Any], columns: dict[str, Any]) -> list[str]:
    notes = []
    if read_opts.get("skiprows", 0):
        notes.append(f"{read_opts['skiprows']} preamble row(s) detected before the header.")
    skewed = [c for c, p in columns.items() if any(f.startswith("high_skew") for f in p["flags"])]
    if skewed:
        notes.append(f"Right-skewed numeric column(s) → consider log-transform: {skewed}.")
    dates = [c for c, p in columns.items() if any(f.startswith("date") for f in p["flags"])]
    if dates:
        notes.append(f"Date-like column(s) → parse & derive year/decade: {dates}.")
    text_num = [c for c, p in columns.items() if any(f.startswith("numeric_as_text") for f in p["flags"])]
    if text_num:
        notes.append(f"Numeric-looking text (has $/commas) → coerce: {text_num}.")
    return notes


# --------------------------------------------------------------------------- #
# Suggest a recipe from a profile
# --------------------------------------------------------------------------- #

def _snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", s) or name


def suggest_recipe(prof: dict[str, Any]) -> Recipe:
    """Turn a profile into a proposed recipe. Every op carries a `reason`."""
    ops: list[dict[str, Any]] = []
    cols = prof["columns"]

    # rename messy headers (spaces / punctuation) to snake_case
    renames = {c: _snake(c) for c in cols if _snake(c) != c}
    if renames:
        ops.append({"op": "rename", "map": renames,
                    "reason": "normalize column names to snake_case (safe for formulas)"})

    def cur(c: str) -> str:
        return renames.get(c, c)

    for c, p in cols.items():
        for flag in p["flags"]:
            if flag.startswith("numeric_as_text"):
                ops.append({"op": "coerce", "column": cur(c), "to": "numeric",
                            "reason": f"'{c}' looks numeric but is stored as text ($/commas)"})
            elif flag.startswith("date:parse"):
                ops.append({"op": "parse_date", "column": cur(c),
                            "derive": ["year", "decade"],
                            "reason": f"'{c}' is a date → derive year and decade"})
    for c, p in cols.items():
        for flag in p["flags"]:
            if flag == "high_skew:log-transform":
                ops.append({"op": "transform", "column": cur(c), "method": "log10",
                            "name": f"log_{cur(c)}",
                            "reason": f"'{c}' is strongly right-skewed (skew={p.get('skew')}) → log10"})
            elif flag == "high_skew:log1p":
                ops.append({"op": "transform", "column": cur(c), "method": "log1p",
                            "name": f"log_{cur(c)}",
                            "reason": f"'{c}' is skewed but has zeros/negatives → log1p"})

    if prof.get("n_duplicate_rows", 0) > 0:
        ops.append({"op": "dedupe",
                    "reason": f"{prof['n_duplicate_rows']} exact duplicate row(s)"})

    return {"read": prof["read"], "ops": ops}


# --------------------------------------------------------------------------- #
# Apply a recipe
# --------------------------------------------------------------------------- #

class RecipeError(ValueError):
    pass


def _op_rename(df, op, rep):
    df = df.rename(columns=op["map"])
    rep["detail"] = op["map"]
    return df


def _op_drop(df, op, rep):
    cols = [c for c in op["columns"] if c in df.columns]
    rep["detail"] = {"dropped": cols}
    return df.drop(columns=cols)


def _op_select(df, op, rep):
    cols = [c for c in op["columns"] if c in df.columns]
    rep["detail"] = {"kept": cols}
    return df[cols]


def _op_coerce(df, op, rep):
    c, to = op["column"], op["to"]
    if to in ("numeric", "integer"):
        s = df[c].astype(str).str.replace(r"[,$%\s]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")
        if to == "integer":
            df[c] = df[c].astype("Int64")
    elif to == "category":
        df[c] = df[c].astype("category")
    elif to == "string":
        df[c] = df[c].astype(str)
    else:
        raise RecipeError(f"coerce: unknown target type {to!r}")
    rep["detail"] = {"column": c, "to": to, "n_null_after": int(df[c].isna().sum())}
    return df


def _op_parse_date(df, op, rep):
    c = op["column"]
    fmt = op.get("format")
    if pd.api.types.is_integer_dtype(df[c]) and not fmt:
        fmt = "%Y%m%d"
    dt = pd.to_datetime(df[c].astype(str) if fmt else df[c], format=fmt, errors="coerce")
    derived = []
    for part in op.get("derive", ["year"]):
        col = op.get("prefix", "") + part
        if part == "year":
            df[col] = dt.dt.year
        elif part == "month":
            df[col] = dt.dt.month
        elif part == "day":
            df[col] = dt.dt.day
        elif part == "quarter":
            df[col] = dt.dt.quarter
        elif part == "decade":
            df[col] = (dt.dt.year // 10 * 10).astype("Int64").astype(str) + "s"
        else:
            raise RecipeError(f"parse_date: unknown derive part {part!r}")
        derived.append(col)
    rep["detail"] = {"column": c, "derived": derived, "n_unparsed": int(dt.isna().sum())}
    return df


def _op_date_diff(df, op, rep):
    fmt = op.get("format")
    a = pd.to_datetime(df[op["start"]].astype(str), format=fmt, errors="coerce")
    b = pd.to_datetime(df[op["end"]].astype(str), format=fmt, errors="coerce")
    name = op.get("name", "duration")
    df[name] = (b - a).dt.days
    if op.get("clip_negative", True):
        df[name] = df[name].clip(lower=0)
    rep["detail"] = {"name": name, "unit": "days"}
    return df


def _op_transform(df, op, rep):
    c, method = op["column"], op["method"]
    name = op.get("name", c)
    v = pd.to_numeric(df[c], errors="coerce")
    if method == "log10":
        if (v <= 0).any():
            raise RecipeError(f"transform log10: '{c}' has non-positive values; use log1p")
        df[name] = np.log10(v)
    elif method == "log1p":
        df[name] = np.log1p(v)
    elif method == "sqrt":
        df[name] = np.sqrt(v.clip(lower=0))
    elif method == "zscore":
        df[name] = (v - v.mean()) / v.std(ddof=1)
    elif method == "boxcox":
        if (v <= 0).any():
            raise RecipeError(f"transform boxcox: '{c}' must be strictly positive")
        transformed, lam = stats.boxcox(v.dropna())
        out = pd.Series(index=v.index, dtype=float)
        out[v.notna()] = transformed
        df[name] = out
        rep.setdefault("detail", {})["boxcox_lambda"] = float(lam)
    else:
        raise RecipeError(f"transform: unknown method {method!r}")
    rep.setdefault("detail", {}).update({"column": c, "method": method, "new_column": name})
    return df


def _op_bin(df, op, rep):
    c, method = op["column"], op.get("method", "cut")
    name = op.get("name", c + "_bin")
    v = pd.to_numeric(df[c], errors="coerce")
    if method == "decade":
        df[name] = (v // 10 * 10).astype("Int64").astype(str) + "s"
    elif method == "median_split":
        med = v.median()
        df[name] = np.where(v <= med, f"<= {med:g}", f"> {med:g}")
    elif method == "quantile":
        df[name] = pd.qcut(v, q=op.get("q", 4), duplicates="drop").astype(str)
    elif method == "cut":
        df[name] = pd.cut(v, bins=op["bins"]).astype(str)
    else:
        raise RecipeError(f"bin: unknown method {method!r}")
    rep["detail"] = {"column": c, "method": method, "new_column": name}
    return df


def _op_filter(df, op, rep):
    c, o, val = op["column"], op["cmp"], op["value"]
    s = df[c]
    ops = {"<": s < val, "<=": s <= val, ">": s > val, ">=": s >= val,
           "==": s == val, "!=": s != val,
           "in": s.isin(val if isinstance(val, list) else [val]),
           "notin": ~s.isin(val if isinstance(val, list) else [val])}
    if o not in ops:
        raise RecipeError(f"filter: unknown op {o!r}")
    before = len(df)
    df = df[ops[o]]
    rep["detail"] = {"kept": len(df), "removed": before - len(df)}
    return df


def _op_dropna(df, op, rep):
    cols = op.get("columns")
    before = len(df)
    df = df.dropna(subset=cols) if isinstance(cols, list) else df.dropna(how=op.get("how", "any"))
    rep["detail"] = {"removed": before - len(df)}
    return df


def _op_fillna(df, op, rep):
    c = op["column"]
    if "value" in op:
        fill = op["value"]
    else:
        m = op.get("method", "median")
        fill = {"mean": df[c].mean, "median": df[c].median,
                "mode": lambda: df[c].mode().iloc[0], "zero": lambda: 0}[m]()
    df[c] = df[c].fillna(fill)
    rep["detail"] = {"column": c, "fill": str(fill)}
    return df


def _op_dedupe(df, op, rep):
    before = len(df)
    df = df.drop_duplicates(subset=op.get("subset"))
    rep["detail"] = {"removed": before - len(df)}
    return df


def _op_aggregate(df, op, rep):
    """Collapse rows to counts per group — turns an event log into a counts table
    for rate/Poisson modeling. `complete` fills a gap-free integer range with 0s."""
    by = op["group_by"] if isinstance(op["group_by"], list) else [op["group_by"]]
    name = op.get("count_name", "n")
    g = df.groupby(by, dropna=False).size().reset_index(name=name)
    comp = op.get("complete")
    if comp:
        col = comp["column"]
        lo = int(comp.get("from", g[col].min()))
        hi = int(comp.get("to", g[col].max()))
        full = pd.DataFrame({col: range(lo, hi + 1)})
        keep = [c for c in g.columns if c != name]
        g = full.merge(g, on=col, how="left") if keep == [col] else \
            full.merge(g, on=col, how="right")
        g[name] = g[name].fillna(0).astype(int)
        g = g.sort_values(col).reset_index(drop=True)
    rep["detail"] = {"group_by": by, "count_name": name, "rows": int(len(g))}
    return g


def _op_winsorize(df, op, rep):
    c = op["column"]
    lo, hi = op.get("limits", [0.01, 0.01])
    v = pd.to_numeric(df[c], errors="coerce")
    low, high = v.quantile(lo), v.quantile(1 - hi)
    df[c] = v.clip(low, high)
    rep["detail"] = {"column": c, "clipped_to": [float(low), float(high)]}
    return df


_OPS = {
    "rename": _op_rename, "drop": _op_drop, "select": _op_select, "coerce": _op_coerce,
    "parse_date": _op_parse_date, "date_diff": _op_date_diff, "transform": _op_transform,
    "bin": _op_bin, "filter": _op_filter, "dropna": _op_dropna, "fillna": _op_fillna,
    "dedupe": _op_dedupe, "winsorize": _op_winsorize, "aggregate": _op_aggregate,
}


def apply(path: str | Path, recipe: Recipe, out: str | Path | None = None) -> dict[str, Any]:
    """Apply a recipe to a file and (optionally) write the cleaned CSV. Returns a
    step-by-step report. Never modifies the input file."""
    path = Path(path)
    read_opts = recipe.get("read") or ({"skiprows": 0} if path.suffix.lower() == ".json"
                                       else _detect_read_opts(path))
    df = _read(path, read_opts)
    n_in = len(df)
    steps = []
    for op in recipe.get("ops", []):
        name = op.get("op")
        if name not in _OPS:
            raise RecipeError(f"unknown op {name!r}")
        rep: dict[str, Any] = {"op": name}
        if "reason" in op:
            rep["reason"] = op["reason"]
        before = len(df)
        df = _OPS[name](df, op, rep)
        rep["rows"] = [before, len(df)]
        steps.append(rep)

    report = {
        "analysis": "prep", "input": str(path), "rows_in": n_in,
        "rows_out": int(len(df)), "columns_out": list(map(str, df.columns)),
        "steps": steps,
    }
    if out is not None:
        out = Path(out)
        df.to_csv(out, index=False)
        report["output"] = str(out)
    return report


# --------------------------------------------------------------------------- #
# CLI:  python -m stat_board.prep {profile|suggest|apply|auto}
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m stat_board.prep",
        description="Profile a data file and apply a reproducible cleaning recipe.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="inspect a file's structure and problems")
    p.add_argument("--data", required=True)

    s = sub.add_parser("suggest", help="print a proposed cleaning recipe (JSON)")
    s.add_argument("--data", required=True)

    a = sub.add_parser("apply", help="apply a recipe file and write a clean CSV")
    a.add_argument("--data", required=True)
    a.add_argument("--recipe", required=True, help="recipe JSON file")
    a.add_argument("--out", required=True)

    au = sub.add_parser("auto", help="suggest + apply in one shot (writes recipe too)")
    au.add_argument("--data", required=True)
    au.add_argument("--out", required=True)

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "profile":
            result = profile(args.data)
        elif args.command == "suggest":
            result = suggest_recipe(profile(args.data))
        elif args.command == "apply":
            recipe = json.loads(Path(args.recipe).read_text())
            result = apply(args.data, recipe, args.out)
        elif args.command == "auto":
            recipe = suggest_recipe(profile(args.data))
            result = apply(args.data, recipe, args.out)
            recipe_path = Path(args.out).with_suffix(".recipe.json")
            recipe_path.write_text(json.dumps(recipe, indent=2))
            result["recipe_saved"] = str(recipe_path)
    except (RecipeError, ValueError, KeyError, OSError) as exc:
        json.dump({"error": type(exc).__name__, "message": str(exc)}, sys.stderr)
        sys.stderr.write("\n")
        return 2
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
