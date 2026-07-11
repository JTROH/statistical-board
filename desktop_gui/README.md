# Desktop GUI

A small tkinter app for quick group comparisons — **ANOVA**, **mean comparison**
(Student's t with Cohen's d), and **TOST equivalence** (±20% of the first
dataset's mean).

All statistics come from the shared `stat_board` engine, so the GUI can never
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

1. Type a number, **Add Value**; repeat to build a dataset.
2. **New Dataset** to start the next group.
3. Run **ANOVA**, **Compare Means**, or **TOST** across all entered datasets.

Notes: at least 2 datasets are required; TOST bounds are ±20% of the first
dataset's mean; results append to the bottom pane.
