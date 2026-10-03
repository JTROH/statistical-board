"""Write a design out as a CSV the analysis half of this package can read back.

The memo's run sheet (:func:`doe_advisor.memo._run_sheet`) is a Markdown table
for a human to print. This module writes the same runs as a CSV for a machine:
one row per run in execution order, with an **empty column per response** for
the scientist to fill in at the bench.

The column layout deliberately matches ``sample_data/DOE_sample_dataset.csv``,
which is the shape ``stat_board`` already analyses::

    run_order,run_type,glucose_coded,glucose_g_l,...,titre_g_l

Coded and natural units are both written. The coded columns are what the design
was built in; the natural ones are what the scientist sets on the equipment, and
what ``stat_board``'s ``--factor`` flags should name.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .designs.spec import Design, DesignSpec, Factor


def _slug(text: str) -> str:
    """A column-safe version of a factor or response name."""
    out = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return out or "col"


def natural_column(factor: Factor) -> str:
    """The natural-units column name for a factor, units folded in when given."""
    return _slug(f"{factor.name}_{factor.units}") if factor.units else _slug(factor.name)


def response_column(name: str, units: str = "") -> str:
    return _slug(f"{name}_{units}") if units else _slug(name)


def run_type(coded_row: np.ndarray) -> str:
    """Classify one run from its coded coordinates alone.

    Derived, not stored: the generators do not tag individual runs, and inventing
    per-run metadata just for the export would be a second source of truth about
    what a run is.
    """
    if np.all(np.isclose(coded_row, 0.0)):
        return "center"
    if np.any(np.abs(coded_row) > 1.0 + 1e-9):
        return "axial"
    return "factorial"


def run_sheet_rows(design: Design, spec: DesignSpec, *, seed: int = 0) -> tuple[list[str], list[list]]:
    """(header, rows) for the run sheet, in randomised execution order."""
    decoded = design.decoded(spec.factors)
    order = design.randomised_order(seed=seed)

    blocked = design.blocks is not None
    header = ["run_order", "block", "run_type"] if blocked else ["run_order", "run_type"]
    for factor in spec.factors:
        header += [f"{_slug(factor.name)}_coded", natural_column(factor)]
    header += [response_column(r.name, r.units) for r in spec.responses]

    rows: list[list] = []
    for position, run in enumerate(order):
        coded_row = design.matrix[run]
        row: list = [position + 1]
        if blocked:
            row.append(int(design.blocks[run]) + 1)  # 1-based on the bench sheet
        row.append(run_type(coded_row))
        for j in range(len(spec.factors)):
            row += [f"{coded_row[j]:g}", f"{decoded[run, j]:g}"]
        row += [""] * len(spec.responses)  # the scientist fills these in
        rows.append(row)
    return header, rows


def run_sheet_csv(design: Design, spec: DesignSpec, out_path: str | Path, *, seed: int = 0) -> Path:
    """Write the run sheet to ``out_path`` as CSV and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header, rows = run_sheet_rows(design, spec, seed=seed)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return out_path


def analysis_hint(spec: DesignSpec) -> dict:
    """How to hand this run sheet to ``stat_board`` once the responses are in.

    Returned as data rather than printed prose so the CLI, the web app and the
    ``/doe-plan`` skill all quote the *same* command instead of each composing
    its own.
    """
    factors = [natural_column(f) for f in spec.factors]
    responses = [response_column(r.name, r.units) for r in spec.responses]
    value = responses[0] if responses else "<response>"
    log = bool(spec.responses) and spec.responses[0].is_log
    lhs = f"np.log10({value})" if log else value
    formula = f"{lhs} ~ " + " * ".join(factors) if factors else ""
    if formula and spec.n_blocks > 1:
        # The block offset is fitted, never interpreted: it is there to remove
        # day-to-day shifts, not to be tested as a finding.
        formula += " + C(block)"
    return {
        "factor_columns": factors,
        "response_columns": responses,
        "formula": formula,
        "response_scale": "log10" if log else "raw",
        "note": (
            f"The design was powered for log10 {spec.responses[0].name}. Record raw values; analyse the "
            f"log10 (the formula above does). Ask the board to analyse on the log10 scale."
            if log
            else None
        ),
        "skill": "/stat-board data=<run sheet>.csv question=\"...\" "
                 + f"value={value} " + " ".join(f"factor={f}" for f in factors),
        "cli": "python3 -m stat_board \"<your question>\" --data <run sheet>.csv "
               + f"--value-col {value} " + " ".join(f"--factor {f}" for f in factors),
    }
