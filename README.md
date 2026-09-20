# Statistical Board

**Design the experiment, then defend the answer.** Two halves of one job in one
package: a design-of-experiments advisor that chooses the runs *before* you
commit reactor time, and an adversarial six-role statistical board that vets the
results *after*.

Inspired by Stanford's STORM, but with one key twist: instead of grounding in web
search, both halves ground in **computation**. An Analyst runs the actual tests; a
Verifier re-runs them; critics attack from different angles; and nothing ships
that a real engine did not produce.

## The loop

```
  /doe-plan  ──▶  runs.csv  ──▶  [ the experiment ]  ──▶  /stat-board
      ▲                                                        │
      │                                                        ▼
      └────────  /doe-plan augment=<transcript>  ◀──  Recommended Next Experiments
```

The return arrow is the part most tools skip. A report that says *"augment to a
central composite design"* is only advice; here the same four diagnostics that
produced that sentence (`design-coverage`, `doe-optimum`, `vif`, `predict`) are
read back by `doe_advisor.augment` and turned into an actual design — ranges
widened where an optimum sat at a boundary, curvature fitted where centre runs
showed a bend.

| | Before the experiment | After the experiment |
|---|---|---|
| **Skill** | `/doe-plan` | `/stat-board` · `/stat-advisor` |
| **Package** | `doe_advisor/` | `stat_board/` |
| **Web app** | `python3 run_doe.py --open` → :8711 | `python3 -m stat_board.webapp` → :8643 |
| **Does** | Generates and scores designs, writes a randomised run sheet | Fits the model, vets every number, writes a PDF report |
| **Sees** | Only the design matrix — never a result | Only results — cannot generate a design |

<p align="center"><em>See <a href="examples/sample_report.pdf">examples/sample_report.pdf</a> for a full generated report (figures + detailed tables included).</em></p>

## Planning an experiment (`doe_advisor`)

Declare your factors and ranges, what you measure, and your run budget. The
engine enumerates every classical design that could answer the question and
returns 2–3 genuinely different options tagged **Cheaper / Recommended / More
thorough**, scored on four axes — and **run count is deliberately not one of
them**; it is the price, not a virtue.

- **Power** — will it see the effect you care about, even at 1.5× the assumed noise?
- **Aliasing** — what will it refuse to tell you apart?
- **Prediction** — I/D/A/G efficiency and FDS: can this support a range claim later?
- **Robustness** — if you lose two bioreactors, does the study still fit the model?

```bash
python3 -m doe_advisor presets                              # domain presets
python3 -m doe_advisor options   --spec spec.json           # the ranked shortlist
python3 -m doe_advisor properties --spec spec.json          # full report for one design
python3 -m doe_advisor runsheet  --spec spec.json --out runs.csv
python3 run_doe.py --open                                   # web app on :8711
```

Designs it builds, all in-house numpy: full factorial, fractional factorial
(minimum-aberration, **resolution computed from the defining relation**, never
hardcoded), definitive screening (Jones & Nachtsheim, via Paley conference
matrices), central composite (rotatable / face-centred / spherical), and
Box-Behnken. Cross-validated against R's `FrF2` and `rsm` on frozen benchmark
cases — run inside the normal test suite, so a design-engine change that breaks
agreement with R fails `pytest`.

The run sheet CSV carries `run_order`, `run_type`, coded *and* natural units per
factor, and an empty column per response — the exact shape `stat_board` analyses.

## The board (six roles)

| Role | Job |
|------|-----|
| **Judge** | Plans the analysis, adjudicates the critics, writes the report, decides iterate vs. finalize. |
| **Analyst** | Runs the engine and reports exact numbers — descriptives, assumptions, tests, effect sizes, Bayes factors. |
| **Frequentist critic** | Right test? Significance vs. importance? Multiplicity? Does the prose match the numbers? |
| **Assumptions skeptic** | Normality, equal variance, independence, power, outliers — is the test even valid? |
| **Bayesian critic** | Bayes-factor reading; where p-values mislead; positive evidence *for* the null. |
| **Verifier** | Independently reproduces every statistic and issues per-claim verdicts. |

## Two layers — and what needs an API key

**Layer 1 — the statistics engine is 100% free, offline, no key, no account.**
The engine, its CLI, the PDF renderer, and the desktop GUI need only `pip install`.

**Layer 2 — the multi-agent board needs an LLM** (the critiquing is done by Claude).
You can run it two ways:

| Front-end | API key? | Needs |
|-----------|----------|-------|
| Engine CLI · desktop GUI · PDF renderer | ❌ none | just `pip install` |
| Claude Code skills (`/stat-board`, `/stat-advisor`) | ❌ none | a Claude Code subscription |
| Standalone CLI + web UI (`python -m stat_board`) | ✅ yes | `ANTHROPIC_API_KEY` |

## Install

```bash
pip install -r requirements.txt
```

## Preparing a messy file first (optional pre-step)

Real files often aren't analysis-ready — a title banner before the header, dates
stored as text, a strongly skewed column that wants a log. `stat_board.prep`
profiles a file and applies a **reproducible cleaning recipe** (it never touches
the original):

```bash
python3 -m stat_board.prep profile --data raw.csv     # inspect: preamble, types, skew, roles, flags
python3 -m stat_board.prep suggest --data raw.csv     # a proposed recipe (JSON) with reasons
python3 -m stat_board.prep auto    --data raw.csv --out clean.csv   # apply the suggestion + save the recipe
```

Recipe ops: `skip_rows` (preamble), `rename`, `drop`/`select`, `coerce` (strip
$/commas), `parse_date` (→ year/decade), `date_diff`, `transform`
(log10/log1p/sqrt/z-score/Box-Cox), `bin` (decade/quantile/median-split),
`aggregate` (events → counts per period, for rate models), `filter`,
`dropna`/`fillna`, `dedupe`, `winsorize`. The `/stat-prep` skill does
this interactively — profile, **propose a recipe, confirm, apply**, then hand off
to `/stat-board`.

## The four ways to use it

### 1. The engine directly (no key)
```bash
python3 -m stat_board.engine describe    --data sample_data/long.csv --group-col group --value-col value
python3 -m stat_board.engine assumptions --data sample_data/long.csv --group-col group --value-col value
python3 -m stat_board.engine welch-anova --data sample_data/wide.csv
```
One-factor commands: `describe`, `assumptions`, `ttest`, `mannwhitney`, `anova`,
`welch-anova`, `kruskal`, `tukey`, `tost`, `bayes-ttest`, `correlation`,
`chisquare`, `power`, `correct`. **Multi-factor** (whole table, by column name):
`two-way-anova`, `ancova`, `regression`, plus the DoE diagnostics `predict`,
`vif`, `design-coverage`, `doe-optimum`. **Count/rate** (events per period):
`poisson`, `negbin`. JSON in, JSON out.

```bash
# multi-factor examples
python3 -m stat_board.engine two-way-anova --data data.csv --value score --factor treatment --factor sex
python3 -m stat_board.engine ancova       --data data.csv --value score --factor treatment --covariate age
python3 -m stat_board.engine regression   --data data.csv --formula "score ~ treatment * sex + age"

# DoE diagnostics: per-run leverage/Cook's D, collinearity (VIF), design coverage +
# curvature, and ranking every ACTUALLY TESTED combination by predicted response
python3 -m stat_board.engine predict         --data data.csv --formula "score ~ treatment * sex"
python3 -m stat_board.engine vif             --data data.csv --formula "score ~ treatment * sex"
python3 -m stat_board.engine design-coverage --data data.csv --factor treatment --factor sex --value score
python3 -m stat_board.engine doe-optimum     --data data.csv --formula "score ~ treatment * sex" --factor treatment --factor sex --value score

# count/rate: how many events per period? (incidence-rate ratios + overdispersion check)
python3 -m stat_board.engine poisson --data counts_per_year.csv --formula "n ~ I(year - 1980)"
```

### 2. Desktop GUI (no key)
```bash
python3 desktop_gui/statistical_analysis.py
```
A tkinter app that picks the right test for you: paste/import your data, click
**Recommend & Run Best Test**, get a plain-language summary and an optional PDF
report — plus manual access to the full test menu (t-test variants, ANOVA
family, TOST, Bayesian t-test, correlation), all backed by the same engine.

### 3. Standalone board: CLI + web UI (needs a key)
```bash
export ANTHROPIC_API_KEY=sk-ant-...        # or copy .env.example -> .env

python3 -m stat_board "Do the groups differ?" \
    --data sample_data/long.csv --group-col group --value-col value

# multi-factor: name the outcome, factor(s) and covariate(s)
python3 -m stat_board "Does treatment affect score, adjusting for age?" \
    --data data.csv --value-col score --factor treatment --factor sex --covariate age

python3 -m stat_board.webapp               # → http://127.0.0.1:8643 (upload data, pick columns, watch live)
```
Try it with **no key** using `--dry-run` (stubbed agents, real report structure):
```bash
python3 -m stat_board "Any question" --data sample_data/wide.csv --dry-run
```

### 4. Claude Code skills (no key, runs on your Claude Code session)
Open this folder in Claude Code (a fresh session) and run:
```
/doe-plan                # design the experiment BEFORE you run it
/stat-prep data=raw.csv  # profile, propose a cleaning recipe, confirm, apply
/stat-board data=sample_data/long.csv question="Do the three groups differ?"
/stat-advisor            # interactive statistical consultant
```

## The report

Always a **PDF**, rendered in pure Python (no LaTeX, no headless browser). With a
dataset it appends a deterministic appendix built entirely from the data:

- **Figures** — box plot with observations, group means ± CI, residual Normal Q-Q,
  residuals-vs-fitted.
- **Tables** — descriptives, the ANOVA source table, goodness of fit
  (R² / η² / ω² / ε² / Root MSE), residual diagnostics, and scaled pairwise
  effect sizes (Cohen's d, Hedges' g, CIs).

For a multi-factor/DoE analysis, the appendix instead has predicted-vs-observed,
Pareto-of-effects, and (for exactly 2 continuous factors) a contour plot, plus a
design-diagnostics appendix (leverage/Cook's D, VIF, design coverage, curvature,
ranked-optimum table) — the grounding for the report's **Recommended Next
Experiments** section.

Render any Markdown report yourself:
```bash
python3 -m stat_board.report examples/sample_report.md out.pdf \
    --data sample_data/long.csv --group-col group --value-col value

# multi-factor/DoE variant
python3 -m stat_board.report reports/doe_report.md out.pdf \
    --data data.csv --formula "score ~ treatment * sex" --factor treatment --factor sex
```

## Layout

```
doe_advisor/           BEFORE the experiment: design generation and scoring
  designs/             the maths layer, coded units, numpy+scipy only (no opinions)
  candidates.py        where the tool's opinion lives: AXIS_WEIGHTS, scoring, roles
  export.py            run sheet -> CSV  (the handoff to stat_board)
  augment.py           stat_board diagnostics -> the next design  (the return path)
stat_board/            AFTER the experiment: analysis, vetting, reporting
  engine/              pure-Python statistics core (no LLM)
  prep.py              data profiling + reproducible cleaning recipes
pdstat/                shared layer: the one Markdown -> PDF renderer
presets/               swappable domain presets (Sf9, CHO, E. coli)
validation/            cross-checks the design engine against R (FrF2, rsm) and JMP
.claude/               Claude Code agents (stat-*) and skills (doe-plan, stat-*)
desktop_gui/           tkinter GUI, backed by the engine
sample_data/           synthetic example datasets
examples/              a full generated sample report (PDF + Markdown + transcript)
```

## Configuration

Environment variables (all optional): `DOE_ADVISOR_MODEL` (default
`claude-sonnet-5`), `STAT_MODEL` (default `claude-opus-4-8`),
`STAT_EFFORT`, `STAT_MAX_ROUNDS`, `STAT_MAX_TOKENS`, `STAT_JUDGE_MAX_TOKENS`,
`STAT_MAX_TOOL_CALLS`. See `.env.example`.

## License

MIT — see [LICENSE](LICENSE).
