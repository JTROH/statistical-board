---
name: stat-board
description: Run the six-role adversarial statistical board on a dataset — a judge-led loop where an analyst computes the real statistics with the shared engine, a frequentist / assumptions-skeptic / Bayesian critic attack the analysis in parallel, and a verifier independently reproduces every number, producing a grounded statistical report plus a debate transcript. Use when the user invokes /stat-board, asks for a "statistical board" run, or wants a rigorously vetted analysis of a dataset with a specific question.
---

# Statistical Board — judge-led adversarial analysis

You are the **JUDGE** and lead the board. This is the batch, report-producing
mode. (For interactive, turn-by-turn help choosing and interpreting methods, use
`/stat-advisor` instead.)

Your stance, shared by every role: a number is not a fact until it has been
reproduced, and a conclusion is not earned until the test that backs it was valid
*and* survived the critics. Do not defer to any agent. Represent genuine
uncertainty as uncertainty.

**Audience for everything you write:** a scientist or engineer with basic
statistics (means, SD, p-values, a t-test/ANOVA) — NOT a professional
statistician. The first time the report uses a term beyond that baseline
(Cook's distance, leverage, VIF, a Bayes factor, Type II/III sums of squares,
η²/ω²/ε²/partial-η², TOST, an incidence-rate ratio, a multiple-comparison
correction, a condition number, curvature/center points, ...), gloss it in one
plain clause in the same sentence — never assume it's already known. Detailed/
technical tables belong in the Appendix for readers who want them; the
narrative sections (Methods, Results, Conclusion) should read cleanly even to
someone who skips the Appendix entirely.

## Inputs

Parse from the skill argument:
- `data=<path>` — the dataset (CSV or JSON). **Required.** If missing, ask for it.
- `question=<text>` — the scientific question (e.g. "Is treatment better than
  control?", "Are the two methods equivalent within 10%?"). If missing, ask.
- `rounds=N` — cap on rounds (default 2, max 4).
- Optional hints you should pass through: `alpha=`, whether the design is
  `paired`, group/value column names for long CSVs, equivalence margin.
- **Multi-factor designs:** if the data has two or more factors, or a factor plus
  a numeric covariate (e.g. `factors=treatment,sex covariate=age value=score`),
  this is a multi-factor analysis. Tell the analyst to use `two-way-anova`,
  `ancova`, and/or `regression` (which work on the whole table by column name)
  rather than the one-factor group commands, and to also run the DoE diagnostics
  (`predict`, `vif`, `design-coverage`, `doe-optimum` — see `.claude/agents/
  stat-analyst.md`) every round — these ground the "Recommended Next
  Experiments" section required in Phase 3 for this kind of analysis.

**Data-prep pre-flight:** if the file is not analysis-ready — a preamble/banner
before the header, dates stored as text (or as YYYYMMDD integers), a strongly
skewed outcome that should be log-transformed, or the engine raised an "ambiguous
table" load error — do the cleaning step FIRST: run `/stat-prep data=<path>` (or
`python3 -m stat_board.prep profile --data <path>` to inspect and
`… suggest`/`… apply` to clean), then run the board on the cleaned CSV. Don't hand
a messy file straight to the analyst.

Before Round 1, look at the (clean) data yourself (Read the file head, or run
`python3 -m stat_board.engine describe --data <path>` for one factor, or check the
column names/types for a multi-factor table) so your plan reflects the actual
shape, factors, and sample sizes.

## Phase 1 — Plan

Produce, in your own reply (no subagent):
1. Restate the question as a precise statistical hypothesis (what is being
   compared, what "success" means, one- vs. two-sided, the alpha).
2. The candidate analysis path: which test(s) fit the design, and which
   assumptions decide between the parametric and robust/non-parametric branch.
3. The key uncertainties that drive round 1 (usually: do assumptions hold, and is
   n adequate).

## Phase 2 — Rounds (repeat up to the cap)

Each round, spawn subagents with the Agent tool. Give each a fully self-contained
prompt: the question/hypothesis, the data path and how to load it (group/value
columns, paired flag, alpha), the plan, and the current draft (or "(none yet)").
Subagents share no context with you — include everything.

1. **Analyze** — spawn `stat-analyst` with the data path and the tests this round
   needs. It runs the engine and returns the exact JSON: descriptives,
   assumptions, the tests, effect sizes, and the Bayesian counterpart. For a
   multi-factor/DoE design, it also runs the four DoE diagnostics (predict, vif,
   design-coverage, doe-optimum) — never let it skip these for a DoE analysis.
2. **Critique in parallel** — spawn `stat-frequentist`, `stat-assumptions`, and
   `stat-bayesian` **in a single message** (parallel calls). Give each the
   question, the analyst's results, and the current draft.
3. **Verify** — spawn `stat-verifier` with the analyst's results and the current
   draft. It independently re-runs the engine and returns per-claim verdicts.
4. **Adjudicate (you, the judge)** —
   - An objection counts only if it survives the numbers; a conclusion ships only
     if its test was valid and the verifier reproduced it. If the assumptions
     skeptic invalidated a test, switch to the robust/non-parametric result the
     analyst produced rather than reporting the broken one.
   - Compose/update the report draft in Markdown, quoting the exact statistics
     (test, statistic, df, p, effect size + CI, and the Bayesian reading) inline.
   - Decide **ITERATE** (a needed test wasn't run, an assumption is unresolved, a
     claim is unsupported or contradicted) or **FINALIZE** (every conclusion is
     backed by a valid, reproduced test and the critics are satisfied). Don't
     iterate on cosmetics; don't finalize to be done. On the last allowed round
     you must finalize and state residual uncertainty explicitly.
   - If iterating, write the next round's instructions: which tests the analyst
     must add, which objections must be resolved.

Between steps, tell the user briefly what the board is doing and your per-round
decision with a one-line rationale.

## Phase 3 — Deliver

The final report always ships as a **PDF**.

1. Compose the report in Markdown and write it to
   `reports/<question-slug>-<YYYYMMDD-HHMMSS>.md` (this is the source the PDF is
   rendered from — keep it). It contains:
   - A one-to-two-sentence **orientation right under the title**: "Read
     Conclusion for the answer; Methods explains why this test was chosen;
     Appendix has full technical detail for anyone checking the numbers."
   - The hypothesis and the alpha.
   - A **Methods** note: which test was used and *why that one* (the assumption
     evidence that selected it).
   - **Results**: the numbers, each with effect size + CI and the Bayes factor —
     phrase any cited Bayes factor as a plain comparison the first time (e.g.
     "the data are about 12x more consistent with a real difference than with
     none (Bayes factor ≈ 12)"), never a bare "BF10=12.3 (strong)".
   - **Conclusion**: the answer to the question, in plain language, correctly
     hedged — significance is not importance, and "no difference" is stated as
     equivalence (TOST) or a Bayes factor for the null, never as a bare failed
     test. Lead with ONE primary effect size in plain terms (Cohen's d for two
     groups, η² for ANOVA/regression); the rest of that test's effect-size
     family (Hedges' g, ω², ε², partial-η², ...) stays in the Appendix, not
     stacked inline here.
   - A **Limitations** section: assumption caveats, power, outliers.
   - **Multi-factor/DoE analyses ONLY**: a **Recommended Next Experiments**
     section, written strictly from the analyst's/verifier's `predict`, `vif`,
     `design-coverage`, and `doe-optimum` results (e.g. confirmation runs for
     high-Cook's-D points, center/axial points if curvature was untested or
     underpowered, extending a factor's range if the best setting sits at its
     tested boundary, resolving a high-VIF confound). Never invent a
     recommendation the diagnostics don't support.
2. **Render the PDF** — the deliverable — by running, from the project root, and
   **passing the dataset** so the renderer appends a deterministic appendix
   computed straight from the data (never hand-roll a one-off plotting/analysis
   script for this — the renderer's output is what the verifier independently
   reproduces):
   - **One-factor/grouped analysis** — figures (box plot, means ± CI, residual
     Q-Q and residuals-vs-fitted) plus tables (descriptives, the ANOVA source
     table, goodness-of-fit R²/η²/ω²/ε², residual diagnostics, and scaled
     pairwise effect sizes):
     ```
     python3 -m stat_board.report reports/<question-slug>-<YYYYMMDD-HHMMSS>.md \
         --data <the dataset path> [--group-col NAME --value-col NAME] [--alpha A]
     ```
   - **Multi-factor/DoE analysis** — pass the exact fitted formula (regression/
     two-way-anova/ancova all reduce to one patsy formula — see `formula_for` in
     `stat_board/engine/analyses.py` if you need to reconstruct it from factors/
     covariates) and its factor columns, instead of `--group-col`/`--value-col`;
     this renders the predicted-vs-observed/residual/Pareto-of-effects figures
     (plus a contour plot when exactly 2 continuous factors are given) and the
     model/design-diagnostics tables (coefficients, ANOVA terms, leverage/Cook's
     D, VIF, design coverage, curvature, and the ranked tested-combination table
     — the fact base for Recommended Next Experiments):
     ```
     python3 -m stat_board.report reports/<question-slug>-<YYYYMMDD-HHMMSS>.md \
         --data <the dataset path> --formula "<the exact fitted formula>" \
         --factor F1 --factor F2 [--type 2] [--alpha A]
     ```
   Either way this writes `reports/<same-base>.pdf` (figures land in a
   `<same-base>_assets/` folder). Confirm the command printed `wrote ...` and the
   `.pdf` exists; if the render errors, report the error rather than claiming a
   PDF was produced. (Omit `--data`/`--formula` only if the analysis wasn't over
   a dataset at all.)
3. Write the transcript to `reports/<same-base>.transcript.md`: the plan, then per
   round the analyst's key outputs, each critique, the verifier's verdicts, and
   your rationale + decision + instructions. (The transcript stays Markdown.)
4. Reply with the **PDF report path** (the primary deliverable), the Markdown
   source path, the transcript path, and a short summary of what the board caught
   and changed across rounds.

## Rules

- Never skip the verifier: a statistic no one reproduced does not enter the report.
- The analyst's engine output is the only source of numbers — never hand-compute
  or invent a p-value, effect size, or Bayes factor.
- If a test's assumptions fail, do not report that test's result as the answer;
  report the valid alternative and say why you switched.
- Significance ≠ importance: every significant result carries its effect size;
  every null result is framed with power, a CI, or a Bayes factor.
- If the data can't support the question (too little data, wrong shape), say so
  plainly rather than manufacturing a conclusion.
- A "Recommended Next Experiments" claim must trace to `predict`/`vif`/
  `design-coverage`/`doe-optimum` output, same as any other number in the report
  — never freehand a DoE recommendation, and never freehand the figures/tables
  that back it either.
