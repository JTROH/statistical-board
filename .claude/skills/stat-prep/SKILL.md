---
name: stat-prep
description: Profile a messy data file and clean/transform it into an analysis-ready CSV using a reproducible recipe — skip preamble rows, coerce types, parse dates into year/decade, log-transform skewed columns, bin/derive columns, handle missing values. Proposes a recipe with reasoning and confirms before writing. Use when the user invokes /stat-prep, when a dataset isn't ready for analysis (preamble rows, dates-as-text, extreme skew, "ambiguous table" load errors), or as the prep step before /stat-board or /stat-advisor.
---

# Statistical Data Prep — profile, clean, transform

You turn a raw/messy data file into an analysis-ready CSV, **reproducibly** and
**without ever overwriting the source**. The heavy lifting is done by the
deterministic engine `stat_board.prep`; your job is to read its profile, propose a
sensible cleaning **recipe**, get the user's OK, apply it, and hand off to analysis.

Argument: `data=<path>` (required). Optional `out=<path>` (default: the source path
with a `.clean.csv` suffix, written next to the source).

## Phase 1 — Profile

Run and read the profile:
```
python3 -m stat_board.prep profile --data <path>
```
It reports: detected preamble rows / delimiter / encoding, and per column the
dtype, missingness, cardinality, skew, a **role guess** (id / factor / numeric
outcome / date / high-cardinality label), and **flags** (high-skew→log, date→parse,
numeric-as-text→coerce, high-missing, constant). Summarize the notable findings to
the user in plain language.

## Phase 2 — Propose a recipe (and confirm — do NOT skip this)

Start from the engine's suggestion:
```
python3 -m stat_board.prep suggest --data <path>
```
Then **improve it with judgment the auto-suggester can't supply**, and present the
full proposed recipe to the user as a numbered list, each step with a one-line
reason. Consider adding, where the data warrants:
- **`bin`** a numeric into a factor — e.g. `year` → `decade` (method `decade`),
  quantiles (`quantile`), a `median_split`, or explicit `cut` points, when the
  user wants a grouping factor.
- **`date_diff`** to derive a duration (e.g. `duration_days` from two date columns).
- **`drop`** an ID or free-text label column that isn't part of the analysis.
- **`filter` / `dropna` / `fillna` / `dedupe` / `winsorize`** only when clearly
  justified — and call out that these change the sample.

**State the interpretation cost of any transform** (e.g. "log10 means results are on
a log/multiplicative scale"). Then ask the user to confirm or adjust before writing
anything. This propose-and-confirm gate is mandatory; do not write the cleaned file
until the user approves (unless they explicitly asked for automatic cleaning, in
which case use `python3 -m stat_board.prep auto --data <path> --out <out>`).

## Phase 3 — Apply

Write the approved recipe to `<out>.recipe.json`, then:
```
python3 -m stat_board.prep apply --data <path> --recipe <out>.recipe.json --out <out>
```
Report back: rows in → out, columns added/renamed/dropped, and the two paths (the
clean CSV and the recipe JSON, so the exact prep is reproducible and auditable).

## Recipe reference (ordered list of ops; `read` sets skiprows/delimiter/encoding)

`rename` {map} · `drop` {columns} · `select` {columns} ·
`coerce` {column, to: numeric|integer|category|string} (strips $/commas) ·
`parse_date` {column, format?, derive:[year,month,day,quarter,decade]} ·
`date_diff` {start, end, name, format?} ·
`transform` {column, method: log10|log1p|sqrt|zscore|boxcox, name?} ·
`bin` {column, method: decade|median_split|quantile(q)|cut(bins), name} ·
`filter` {column, cmp: <|<=|>|>=|==|!=|in|notin, value} ·
`dropna` {columns?|how} · `fillna` {column, value|method} ·
`dedupe` {subset?} · `winsorize` {column, limits:[lo,hi]}

## Hand-off

After writing the clean file, offer the natural next step and a **suggested design**
from the column roles — e.g. "outcome = `log_cost`, factor = `disaster`, covariate =
`year`; run `/stat-board data=<out> ...`" (or `/stat-advisor` for interactive help).

## Rules

- **Never modify the input file.** Always write a new CSV; keep the recipe JSON.
- **Show the recipe and get approval before applying** (unless auto mode was asked).
- Every transform carries a stated reason and, if it changes scale/interpretation,
  a note.
- Log-transform needs strictly positive values; use `log1p` when a column has zeros.
- Don't silently drop or impute rows — surface it and confirm; report the row delta.
