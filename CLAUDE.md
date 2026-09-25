# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope note

This is a **pure Python 3.13 project**. The Bun/TypeScript guidance in the
parent `~/CLAUDE.md` does not apply here — there is no `package.json`, no
JS/TS source, and no bundler. (`*/static/` is hand-written vanilla JS with no
build step.)

## What this repository is

Two halves of one job, in one repo:

| | `doe_advisor/` | `stat_board/` |
|---|---|---|
| **When** | **Before** the experiment | **After** the experiment |
| **Does** | Generates and scores classical designs, writes a run sheet | Fits the model, vets it adversarially, writes a report |
| **Sees** | Only the design matrix — never a response value | Only results — cannot generate a design |
| **Maths** | numpy + scipy, all in-house | statsmodels + scipy |

The loop they form is the point of the repo:

```
/doe-plan  ->  runs.csv  ->  [the experiment]  ->  /stat-board  ->  Recommended
    ^                                                                Next Experiments
    |                                                                      |
    +------------------  /doe-plan augment=<transcript>  <-----------------+
```

`doe_advisor.augment.form_from_diagnostics` is what closes it: it reads
`stat_board`'s own `design-coverage` / `doe-optimum` / `vif` / `predict` output
and proposes the next design. `doe_advisor.export.run_sheet_csv` is what opens
it: it writes a design as a CSV in the exact shape `stat_board` analyses.

## Commands

```bash
pip install -r requirements.txt       # runtime deps for everything
pip install -r requirements-dev.txt   # adds pytest, ruff, httpx (for TestClient)

python3 -m pytest                     # whole suite (testpaths=tests, addopts=-q)
python3 -m pytest tests/engine/test_analyses_groups.py               # one file
python3 -m pytest tests/engine/test_cli.py::test_regression_command  # one test
python3 -m pytest -k tost                                            # by name

python3 -m ruff check .               # lint (line-length 120, py313, E/F/I/UP/B)
python3 -m ruff check . --fix
```

`desktop_gui/statistical_analysis.py` is excluded from ruff (tkinter widget
code). `E702` is ignored repo-wide — the argparse one-liners in
`stat_board/engine/cli.py` and `doe_advisor/cli.py` are deliberate.

### Cross-validation against R

```bash
python3 -m validation.run_tool     # -> validation/results/tool.json
Rscript validation/validate_r.R    # -> validation/results/r.json
python3 -m validation.compare      # non-zero exit on any disagreement
```

`tests/test_validation.py` runs `validation.compare` inside pytest, so a design-engine
change that breaks agreement with R fails the normal test run. The R leg auto-skips
if `Rscript` is missing; the JMP leg is manual (`validation/JMP_INSTRUCTIONS.md`).

### Running each front-end

```bash
# --- Both halves in one local app, as tabs (key needed only for live board runs) ---
python3 run_app.py --open                              # pdstat/webapp.py, 127.0.0.1:8700

# --- Design side (no API key) ---
python3 -m doe_advisor options --spec spec.json        # JSON in / JSON out
python3 -m doe_advisor runsheet --spec spec.json --out runs.csv
python3 run_doe.py --open                              # web app, 127.0.0.1:8711

# --- Analysis side (no API key) ---
python3 -m stat_board.engine describe --data sample_data/long.csv --group-col group --value-col value
python3 desktop_gui/statistical_analysis.py            # tkinter GUI

# --- Analysis side, LLM board (needs ANTHROPIC_API_KEY) ---
python3 -m stat_board "Do the groups differ?" --data sample_data/long.csv \
    --group-col group --value-col value
python3 -m stat_board "Any question" --data sample_data/wide.csv --dry-run  # no key, no spend
python3 -m stat_board.webapp                           # web app, 127.0.0.1:8643

# --- Claude Code skills (no key, runs on the current session) ---
#   /doe-plan · /stat-prep · /stat-board · /stat-advisor   (need a FRESH session to load)

# --- Data prep and standalone report rendering ---
python3 -m stat_board.prep auto --data raw.csv --out clean.csv
python3 -m stat_board.report report.md out.pdf --data sample_data/long.csv \
    --group-col group --value-col value
```

`--dry-run` is the fastest way to exercise the full orchestrator flow end to
end without API calls: `llm._stub_reply` returns schema-conforming JSON so the
judge's parsing, transcript writing, and PDF rendering all run for real.

---

## Architecture: `stat_board/` (analysis)

### One engine, four front-ends

`stat_board/engine/` is the **single source of numeric truth** for analysis. It is
pure Python — no LLM, no I/O beyond reading the data file — and every consumer
goes through it. The invariant to protect: *no front-end may compute a statistic
itself*. If the GUI, the board, and the skills ever disagree on a number, that
invariant has been broken.

```
stat_board/engine/  data.py (load) · analyses.py (all tests) · effects.py · bayes.py · cli.py
        ▲                ▲                    ▲                      ▲
        │                │                    │                      │
  engine CLI      engine_tool.py        appendix.py        desktop_gui/recommend.py
  (humans)        (run_stat tool)       (report tables/    (deterministic test
                   ▲                     figures)           chooser, no LLM)
                   │
            orchestrator.py ── llm.py ── Anthropic API
                   ▲
          __main__.py (CLI) · webapp.py (SSE)
```

Engine functions take either **`Groups`** (a `{name: [values]}` dict from
`data.load_groups`, for one-factor commands) or a **file path + formula/column
names** (for multi-factor / DoE / count commands). This split runs all the way
up: `engine_tool._GROUP_COMMANDS` vs `_STANDALONE`, and the `--group-col/
--value-col` vs `--formula/--factor/--covariate` CLI flags.

Data shapes accepted by `load_groups`: wide CSV (one column per group), long CSV
(`--group-col`/`--value-col`), or JSON `{"A": [...], "B": [...]}`.

### The board loop (`orchestrator.run`)

One judge-led loop drives six roles. Claude Code plays the Judge when running
via the skills; in the standalone path the Judge is another API call.

```
plan (judge, JSON schema)
 └─ per round:
      analyst   ─ run_stat tool ─┐
      frequentist ┐              │
      assumptions ├─ asyncio.gather (parallel critics)
      bayesian    ┘              │
      verifier  ─ run_stat tool ─┘   (independently re-runs the numbers)
      judge     ─ JSON schema ─→ decision: iterate | finalize + draft
```

Two structured-output schemas in `orchestrator.py` are the contract:
`_PLAN_SCHEMA` and `_DECISION_SCHEMA`. The decision schema carries
`model_formula` / `model_factors` / `model_typ` — the judge **restates** the
model its own draft was built on, so `appendix.build_multifactor` re-fits
exactly that model rather than guessing one. Changing these fields means
changing the judge prompt in `prompts.py` too.

The orchestrator branches early on `multifactor` (any `--factor`, `--covariate`,
or a comma-separated multi-outcome `--value-col`). That branch picks a different
data summary, design note, and first-round instruction — and switches the whole
report to the DoE appendix.

### Report pipeline

`transcript.save()` is the single exit point for a finished run. It writes five
files to `reports/<slug>-<stamp>.*`: `.md`, `.pdf`, `.transcript.md`,
`.transcript.pdf`, `.transcript.json`.

`appendix.py` appends a deterministic appendix computed from the data, never
written by the model. Two variants, picked by which args arrive:
- `build()` — one-factor: box plot, means ± CI, residual Q-Q, residuals-vs-fitted;
  descriptives, ANOVA source table, goodness of fit, pairwise effect sizes.
- `build_multifactor()` — DoE: predicted-vs-observed, Pareto of effects, contour
  (only with exactly 2 continuous factors), plus design diagnostics
  (leverage/Cook's D, VIF, design coverage, curvature, ranked optimum). These
  four diagnostics are the required grounding for the report's
  "Recommended Next Experiments" section.

---

## Architecture: `doe_advisor/` (design)

The pipeline is one direction, and each stage is independently testable:

```
form dict → intake.spec_from_dict → DesignSpec
          → candidates.generate_candidates → [Design]
          → designs.properties.evaluate → DesignProperties (per design)
          → candidates.score_candidates / top_options → [ScoredDesign] with roles
          → narrate.get_narrator → Narration (Claude or template)
          → memo.build_memo + figures → Markdown → pdstat.report.markdown_to_pdf
          → export.run_sheet_csv → runs.csv  (the handoff to stat_board)
```

`doe_advisor/designs/` is the maths layer and holds no opinions:

- `spec.py` — `Factor`, `Response`, `ModelOrder`, `DesignSpec`, `Design`. Deliberately
  free of statistics so the same `Design` can be re-scored under new assumptions
  without being regenerated.
- `model.py` — the single definition of "the model". A term is a tuple of factor
  indices; repetition means a power (`(0,0)` is a quadratic). Everything needing an
  `X` matrix builds it here. `potential_terms` follows JMP's convention: look one
  order up from what you fit.
- `classical.py` — generators, all in-house numpy. Fractional-factorial generators
  come from a published minimum-aberration table; **resolution is computed from the
  defining relation, never hardcoded**, so a bad table entry fails a test instead of
  lying in the memo.
- `alias.py` — `A = (X1'X1)^-1 X1' X2`.
- `properties.py` — power, prediction (I/D/A/G, FDS), robustness to run loss.

`candidates.py` is where the tool's opinion lives. `AXIS_WEIGHTS` and
`DEFAULT_MIN_POWER` are module-level constants because they are printed to the user
and are the first thing a statistician will argue with — keep them visible and keep
sub-scores attached to each `ScoredDesign`. Policy decisions (e.g. the centre-point
floor) belong here, not in the generators.

`intake.py`: the structured form is the source of truth. `spec_from_dict` never needs
Claude; `extract_form_from_text` is optional sugar that produces the same dict for the
scientist to check. Do not add a path from free text straight to a computation.

`cli.py` / `__main__.py` and `server.py` are two front ends over that one pipeline.
Both serialise through `serialise.py` so they cannot drift on what "the numbers" are.
`server.py` is loopback-only and stateless — every request carries the whole form,
nothing is persisted. Routes: `/api/capabilities`, `/api/presets`,
`/api/intake/extract`, `/api/design`, `/api/memo` (PDF), `/api/memo/markdown`.

`presets/` holds swappable domain presets. The dropdown order is blank first, then the
preset's `order` key (Sf9 = 10, CHO = 20, E. coli = 30); the first non-blank preset is the
default. Each response carries a `goal` (`screen`/`maximize`/`minimize`/`target`, see
`ResponseGoal` in `spec.py`) — it changes the advice (a curvature caveat), never the maths.
Company-specific ranges, platform history, and CQA targets belong here and nowhere
else; `presets/local_*.json` is gitignored.

---

## `pdstat/` — the shared layer

`pdstat/report/markdown_pdf.py` holds the **one** Markdown → PDF renderer, used by
both halves (`stat_board/report.py` and `doe_advisor/memo.py`) so every document
this repo produces is styled identically. Markdown → HTML (python-markdown) → PDF
(PyMuPDF's Story engine). No LaTeX, no headless browser. `_CSS` there is a
hand-tuned subset Story understands — not general CSS — and `_strip_top_bleed` /
`_shrink` encode real, previously-debugged rendering failures. Do not "clean them up".

`stat_board/report.py` keeps only what is specific to a statistical report
(`convert_file`, the appendix wiring, the CLI) and re-exports `markdown_to_pdf`
so existing callers keep working.

`pdstat/webapp.py` mounts both FastAPI apps unchanged (`/design/`, `/analyse/`) behind a tab
shell. It works only because both front ends call `api/...` **relative** URLs — an absolute
`/api/...` would hit the wrong app. `tests/webapp/test_workbench.py` guards that.

`pdstat/compute/` is an empty stub, reserved for hoisting `stat_board/engine` up
when a third tool needs it. **That refactor is deliberately not done yet** — do not
start it without a second consumer that actually needs it.

---

## Agents and skills live in the repo

`.claude/agents/stat-*.md` (six subagents) and `.claude/skills/*/SKILL.md` (four
skills) are checked in — they are project source, not personal config.

- `/doe-plan` + `stat-design` — the pre-experiment half.
- `/stat-board` + `stat-analyst`, `stat-frequentist`, `stat-assumptions`,
  `stat-bayesian`, `stat-verifier` — the post-experiment board.
- `/stat-advisor`, `/stat-prep` — interactive analysis and data cleaning.

The skill prompts and `stat_board/prompts.py` encode the *same* six roles for the
two execution paths, so a change to a role's stance usually belongs in both.

---

## Rules that break silently if violated

- **`stat_board/engine` is the single source of numeric truth.** No front-end
  computes a statistic itself.
- **Everything inside `doe_advisor/designs/` is in coded units** (-1 low, +1 high).
  Decoding to real units happens only at the edges (`Factor.decode`, memo export,
  `export.run_sheet_csv`).
- **`target_effect` is the change across the full low-to-high range**, so the coded
  coefficient is half of it. Someone entering a half-range effect silently gets a
  badly underpowered design.
- **Run count is never scored.** It is the price, not a virtue; it is reported
  alongside the four axes.
- **`narrate.py` owns no numbers.** `ClaudeNarrator` is handed pre-formatted facts,
  and `_unsupported_numbers` re-reads the generated text and rejects it — falling
  back to `TemplateNarrator` — if any numeric token is absent from those facts.
  Never relax that check, and never add a code path where prose can reach the memo
  without passing it. The template path must always work with no API key.
- **`validation/cases.py` benchmark cases may be added to, never silently changed.**
  The point is that today's answers can be compared with yesterday's. Fractional
  factorials are compared on their full **word-length pattern**, not just
  resolution — resolution is only the pattern's first non-zero entry, and a
  resolution-only check misses real errors.
- **`desktop_gui/_bootstrap.py` probes specifically for `stat_board/engine`.** A
  top-level `doe_advisor/` sibling is *not* auto-discovered by it. Harmless today
  (the GUI needs no DoE), but do not assume the GUI can import `doe_advisor`.
- **`figures.py` filename collision:** the one-factor and multifactor appendix paths
  both write `fig_qq.png` / `fig_resid.png`. They must never share an assets dir.

## Configuration

Environment, all optional. The engines, the GUI, the report renderer, and the
Claude Code skills need none of it.

| Variable | Used by | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | standalone board; DoE chat intake + narration | — |
| `STAT_MODEL` | standalone board | `claude-opus-4-8` |
| `STAT_EFFORT`, `STAT_MAX_ROUNDS`, `STAT_MAX_TOKENS`, `STAT_JUDGE_MAX_TOKENS`, `STAT_MAX_TOOL_CALLS` | standalone board | see `stat_board/config.py` |
| `DOE_ADVISOR_MODEL` | DoE narration | `claude-sonnet-5` |
| `STAT_BOARD_HOME` | desktop GUI, when run from elsewhere | — |

`stat_board/config.py` does a best-effort `.env` load; real environment variables
always win.

`reports/`, `uploads/`, `studies/`, `*_assets/`, `validation/results/`,
`presets/local_*.json` and `.env` are gitignored — generated or private output,
never commit it.
