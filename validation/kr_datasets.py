"""Frozen datasets for the Kenward-Roger cross-check against R's pbkrtest.

    python3 -m validation.kr_datasets      # (re)writes validation/datasets/kr_*.csv

The CSVs are committed and are the source of truth: both this tool and
``validate_kr.R`` read the *same file*, so a mismatch can only come from the
analysis, never from two random generators drawing different numbers. This
script records how they were made; rerunning it is only needed to add a case,
and an existing file must not be silently changed (see ``cases.py``).

Each dataset is a split-plot experiment with a different stress on the
small-sample machinery: balance, lost runs, unequal whole plots with a
continuous subplot factor, a multi-df categorical whole-plot factor, almost no
whole-plot variance, and a design built by this tool's own split-plot search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def _classic(seed: int, n_plots: int, wp_sd: float, drop: tuple[int, ...] = ()) -> pd.DataFrame:
    """Whole-plot factor A alternating by plot; subplot factor B as a 2 x 2 in each plot."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_plots):
        a = 1 if g % 2 else -1
        offset = rng.normal(0, wp_sd)
        for b in (-1, -1, 1, 1):
            y = 10 + 1.0 * a + 0.8 * b + 0.5 * a * b + offset + rng.normal(0, 0.5)
            rows.append({"whole_plot": g + 1, "A": a, "B": b, "y": round(y, 6)})
    return pd.DataFrame(rows).drop(index=list(drop)).reset_index(drop=True)


def _unequal(seed: int, wp_sd: float) -> pd.DataFrame:
    """Unequal whole plots (2-5 runs) and a continuous subplot factor."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(8):
        a = 1 if g % 2 else -1
        offset = rng.normal(0, wp_sd)
        for _ in range(2 + g % 4):
            b = round(float(rng.uniform(-1, 1)), 4)
            y = 1 + a + b + 0.4 * a * b + offset + rng.normal(0, 0.5)
            rows.append({"whole_plot": g + 1, "A": a, "B": b, "y": round(y, 6)})
    return pd.DataFrame(rows)


def _categorical(seed: int) -> pd.DataFrame:
    """A three-level categorical whole-plot factor (media), unequal plots."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(9):
        media = ("CDA", "CDB", "CDC")[g % 3]
        offset = rng.normal(0, 0.8)
        for k in range(3 + g % 2):
            b = (-1, 1, -1, 1)[k]
            y = 5 + {"CDA": 0.0, "CDB": 1.0, "CDC": -0.5}[media] + 0.7 * b + offset + rng.normal(0, 0.4)
            rows.append({"whole_plot": g + 1, "media": media, "B": b, "y": round(y, 6)})
    return pd.DataFrame(rows)


def _from_design(seed: int) -> pd.DataFrame:
    """A design made by this tool's own split-plot search, with a simulated response."""
    from doe_advisor.designs.spec import DesignSpec, Factor, ModelOrder
    from doe_advisor.designs.split_plot import split_plot_design

    spec = DesignSpec(
        [Factor("temp", -1, 1, hard_to_change=True), Factor("pH", -1, 1), Factor("feed", -1, 1)],
        model_order=ModelOrder.INTERACTION,
    )
    design = split_plot_design(spec, 20, 6, seed=0)
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, 1.0, design.n_whole_plots)
    x = design.matrix
    y = 3 + 0.8 * x[:, 0] + 0.6 * x[:, 1] - 0.4 * x[:, 2] + 0.3 * x[:, 0] * x[:, 1]
    y = y + offsets[design.whole_plots] + rng.normal(0, 0.5, design.n_runs)
    return pd.DataFrame({
        "whole_plot": design.whole_plots + 1, "temp": x[:, 0], "pH": x[:, 1], "feed": x[:, 2],
        "y": np.round(y, 6),
    })


BUILDERS = {
    "kr_balanced.csv": lambda: _classic(1, 8, 1.0),
    "kr_lost_runs.csv": lambda: _classic(2, 6, 1.0, drop=(0, 13, 18)),
    "kr_unequal.csv": lambda: _unequal(3, 0.6),
    "kr_categorical.csv": lambda: _categorical(4),
    "kr_low_wp_variance.csv": lambda: _unequal(5, 0.1),
    "kr_tool_design.csv": lambda: _from_design(6),
}


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in BUILDERS.items():
        path = DATASET_DIR / name
        build().to_csv(path, index=False)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
