# Desktop GUI

A tkinter app for quick group comparisons that figures out the right test for
you: paste or import your data, click **Recommend & Run Best Test**, and get a
plain-language summary plus an optional PDF report — all backed by the same
`stat_board` engine the statistical board's agents use, so the GUI can never
disagree with the board's numbers.

## Run

From the repository root:

```bash
python3 desktop_gui/statistical_analysis.py
```

The engine is located automatically (the script walks up to the repo root). If
you move this file elsewhere, set `STAT_BOARD_HOME` to the repo path:

```bash
STAT_BOARD_HOME=/path/to/statistical-board python3 statistical_analysis.py
```

## Use

**1. Data tab**

- Set the significance level (alpha) once at the top — it's used everywhere
  else in the app (assumption checks, the recommended test, the PDF report).
- Add a dataset by typing a name and pasting its numbers (comma, space, or
  newline separated) into the box, then **Add Dataset**. Non-numeric tokens
  are skipped and counted rather than blocking entry.
- Or **Import CSV/JSON...** — handles a wide CSV (one column per group), a
  long CSV (pick the group/value columns if the file is ambiguous), or JSON
  (`{"A": [...], "B": [...]}`).
- Double-click a dataset in the list to edit it; **Remove Selected** /
  **Clear All** to delete. A box-plot preview updates as you go.
- Once at least 2 datasets each have 2+ values, **Recommend & Run Best Test**
  checks normality (Shapiro-Wilk) and variance homogeneity (Levene) and picks
  the matching test — Student's/Welch's t-test or Mann-Whitney for 2 groups;
  one-way/Welch ANOVA or Kruskal-Wallis (+ Tukey HSD follow-up when
  applicable) for 3+ — the same reasoning the `/stat-advisor` skill uses,
  encoded deterministically since this GUI has no LLM to reason with.

**2. Results & Report tab**

- A plain-language summary and the reasoning behind the chosen test.
- **Advanced**: run a specific test yourself (t-test variants, Mann-Whitney,
  TOST equivalence with an editable margin or explicit bounds, a Bayesian
  t-test, correlation, or the ANOVA family) — two-dataset tests get an
  explicit dataset-pair selector rather than looping over every pair at once.
- **Export PDF Report...** renders a polished PDF (reusing the same
  figures/tables appendix pipeline as `stat_board.report`) next to a
  `<name>_assets/` folder of the source figures.

Notes: correlation treats the two selected datasets as paired (x, y) of equal
length — it's kept separate from auto-recommend since it answers a different
question than a group comparison.
