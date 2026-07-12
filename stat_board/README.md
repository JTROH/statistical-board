# Statistical Board — STORM-like adversarial statistical analysis

A judge-led board that must defend its claims before anything ships, applied to
**data analysis**. Inspired by STORM, with a twist that makes it work for
statistics: the "researcher" role is an **Analyst** that grounds every claim in a
real computation on your data instead of a web search, and a **Verifier**
independently **re-runs** the math rather than re-reading it.

Two front-ends sit over one shared engine:

| | `/stat-board` (report mode) | `/stat-advisor` (advisory mode) |
|---|---|---|
| Shape | Batch → **PDF report** + transcript | Interactive, turn-by-turn |
| Use when | You want a vetted answer to a specific question | You're deciding how to analyze, or want a result explained |
| Mechanics | Judge fans out 5 subagents over rounds | One consultant, calls the engine on demand |

## The board (six roles)

| Role | Job |
|---|---|
| **Judge** | Plans the analysis, adjudicates the critics, writes the report, decides iterate vs. finalize. |
| **Analyst** | Runs the engine and reports exact numbers — descriptives, assumptions, tests, effect sizes, Bayes factors. The only source of numbers. |
| **Frequentist critic** | Right test? Significance vs. importance? Multiplicity? Prose vs. numbers? |
| **Assumptions skeptic** | Normality, equal variance, independence, power, outliers — is the test even valid? |
| **Bayesian critic** | Bayes factor reading; where p-values mislead; positive evidence for the null. |
| **Verifier** | Independently reproduces every statistic; per-claim supported/contradicted verdicts. |

Claude Code plays the judge; the other five are subagents in `.claude/agents/stat-*.md`.

## The engine (`stat_board.engine`)

Pure Python, JSON in / JSON out, no LLM — the shared source of numeric truth. It's
a clean superset of the original `statistical_analysis_app` (ANOVA, t-test, TOST),
expanded to a broad toolkit.

```bash
python3 -m stat_board.engine <command> --data <file.csv|.json> [options]
```

| Command | What it does |
|---|---|
| `describe` | n, mean, sd, sem, median, IQR, 95% CI per group |
| `assumptions` | Shapiro-Wilk normality, Levene + Bartlett homogeneity, IQR outliers |
| `ttest` `[--paired] [--equal-var]` | Two-sample (Welch by default); Cohen's d, Hedges' g |
| `mannwhitney` | Non-parametric two-sample |
| `anova` / `welch-anova` / `kruskal` | One-way, unequal-variance, non-parametric |
| `tukey` | Post-hoc pairwise (family-wise error controlled) |
| `tost` `(--margin-pct P \| --low L --high H)` | Equivalence testing (TOST) |
| `bayes-ttest` `[--paired] [--r 0.707]` | JZS Bayes factor (BF10) |
| `correlation --method pearson\|spearman\|kendall` | Correlation of two columns |
| `chisquare --table "[[..],[..]]"` | Independence + Cramér's V |
| `power --test ttest\|anova --effect-size D (--n N \| --power P)` | Solve for n or power |
| `correct --pvalues "[..]" --method fdr_bh\|bonferroni\|holm` | Multiple-comparison correction |
| **`two-way-anova --value V --factor F1 --factor F2`** | Factorial ANOVA (main effects + interactions), partial η² |
| **`ancova --value V --factor F --covariate C`** | Factor(s) adjusted for numeric covariate(s) |
| **`regression --formula "y ~ x1 + x2 + C(g)"`** | OLS + ANOVA term table + residual diagnostics |
| **`poisson --formula "n ~ year" [--exposure col]`** | Poisson rate regression (IRR) + overdispersion check |
| **`negbin --formula "n ~ year"`** | Negative-binomial rate regression for overdispersed counts |

Data shapes: for one-factor comparisons — **wide** CSV (one column per group),
**long** CSV (`--group-col`/`--value-col`), or **JSON** `{"A": [...], "B": [...]}`.
For **multi-factor**, a tidy CSV with one column per variable (outcome, factors,
covariates) analyzed by column name.

## Usage (Claude Code)

Open Claude Code in this folder (a **new** session — agents/skills load at start):

```
/stat-board data=sample_data/long.csv question="Do the three groups differ?" rounds=2
/stat-advisor data=sample_data/two.json
```

No `ANTHROPIC_API_KEY` needed — it runs on your Claude Code session.

The `/stat-board` report is always delivered as a **PDF** (`reports/<slug>.pdf`),
rendered from the Markdown source alongside it. The renderer is pure Python (no
LaTeX or headless browser). Pass `--data` and it appends a deterministic
**appendix** — figures (box plot, means ± CI, residual Q-Q, residuals-vs-fitted)
and detailed tables (descriptives, ANOVA source table, goodness of fit
R²/η²/ω²/ε², residual diagnostics, scaled pairwise effect sizes), all computed
from the data, not written by the model:

```bash
python3 -m stat_board.report reports/<slug>.md \
    --data sample_data/long.csv --group-col group --value-col value
```

## Standalone (no Claude Code)

The board also runs as a self-contained Python program driven by the Anthropic
API. Here the Analyst and Verifier reach the engine through a local `run_stat`
tool that executes computations in-process and feeds the results back.

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # or `ant auth login`, or a .env file

# CLI — writes reports/<slug>.pdf (+ .md source + .transcript.json)
python3 -m stat_board "Do the groups differ?" \
    --data sample_data/long.csv --group-col group --value-col value --rounds 2

# Dry run — exercises the full flow with stubbed agents, no API calls, zero spend
python3 -m stat_board "Any question" --data sample_data/wide.csv --dry-run

# Web UI — start runs in the browser, upload a dataset, watch the debate live (SSE)
python3 -m stat_board.webapp            # → http://127.0.0.1:8643
```

Configuration via environment (all optional): `STAT_MODEL` (default
`claude-opus-4-8`), `STAT_EFFORT` (`high`), `STAT_MAX_ROUNDS` (`2`),
`STAT_MAX_TOKENS`, `STAT_JUDGE_MAX_TOKENS`, `STAT_MAX_TOOL_CALLS`.

## The desktop GUI

The original `statistical_analysis_app/` tkinter GUI now calls this same engine
instead of its own inline math (which also fixes its broken TOST import on
modern statsmodels). If the engine isn't importable, set `STAT_BOARD_HOME` to
this project's directory. So the GUI, the CLI/web board, and the Claude Code
skills all share one source of numeric truth.

## Setup

```bash
pip install -r ../requirements.txt   # engine + report + orchestrator/web deps
```
