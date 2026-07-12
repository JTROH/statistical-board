# Statistical Board

**An adversarial, STORM-inspired multi-agent system for statistical analysis.**
A judge-led board of six roles turns a dataset and a question into a vetted PDF
report — where every number is computed by a real statistics engine and
independently reproduced before anything ships.

Inspired by Stanford's STORM, but with one key twist: instead of grounding in web
search, the board grounds in **computation**. An Analyst runs the actual tests; a
Verifier re-runs them; three critics attack the analysis from different angles;
and a Judge only finalizes when every conclusion is backed by a valid, reproduced
test.

<p align="center"><em>See <a href="examples/sample_report.pdf">examples/sample_report.pdf</a> for a full generated report (figures + detailed tables included).</em></p>

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
`two-way-anova`, `ancova`, `regression`. **Count/rate** (events per period):
`poisson`, `negbin`. JSON in, JSON out.

```bash
# multi-factor examples
python3 -m stat_board.engine two-way-anova --data data.csv --value score --factor treatment --factor sex
python3 -m stat_board.engine ancova       --data data.csv --value score --factor treatment --covariate age
python3 -m stat_board.engine regression   --data data.csv --formula "score ~ treatment * sex + age"

# count/rate: how many events per period? (incidence-rate ratios + overdispersion check)
python3 -m stat_board.engine poisson --data counts_per_year.csv --formula "n ~ I(year - 1980)"
```

### 2. Desktop GUI (no key)
```bash
python3 desktop_gui/statistical_analysis.py
```
A small tkinter app for ANOVA / mean comparison / TOST — backed by the same engine.

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

Render any Markdown report yourself:
```bash
python3 -m stat_board.report examples/sample_report.md out.pdf \
    --data sample_data/long.csv --group-col group --value-col value
```

## Layout

```
stat_board/            the package: engine/ + prep, orchestrator, prompts, report, web UI
  engine/              pure-Python statistics core (no LLM)
  prep.py              data profiling + reproducible cleaning recipes
.claude/               Claude Code agents (stat-*) and skills (stat-prep, stat-board, stat-advisor)
desktop_gui/           tkinter GUI, backed by the engine
sample_data/           synthetic example datasets
examples/              a full generated sample report (PDF + Markdown + transcript)
```

## Configuration

Environment variables (all optional): `STAT_MODEL` (default `claude-opus-4-8`),
`STAT_EFFORT`, `STAT_MAX_ROUNDS`, `STAT_MAX_TOKENS`, `STAT_JUDGE_MAX_TOKENS`,
`STAT_MAX_TOOL_CALLS`. See `.env.example`.

## License

MIT — see [LICENSE](LICENSE). (Fill in your name in the copyright line.)
