"""Compute this tool's answers for the frozen benchmark cases.

    python3 -m validation.run_tool

Writes ``validation/results/tool.json``. Deliberately dumb: it computes and
records, it does not compare. Comparison lives in ``compare.py`` so that the
tool's own output can be produced without R installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from doe_advisor.designs import classical as C
from doe_advisor.designs.model import model_matrix, model_terms
from doe_advisor.designs.properties import d_efficiency, power_report
from doe_advisor.designs.spec import DesignSpec, Factor, ModelOrder, Response

from .cases import (
    BLOCKING_CASES,
    BOX_BEHNKEN_CASES,
    CENTRAL_COMPOSITE_CASES,
    DSD_CASES,
    FRACTIONAL_CASES,
    POWER_CASES,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _sorted_matrix(matrix: np.ndarray) -> list[list[float]]:
    """Rows in a canonical order, so run order never causes a false mismatch."""
    rows = [tuple(round(v, 9) + 0.0 for v in row) for row in matrix]
    return [list(row) for row in sorted(rows)]


def fractional_results() -> list[dict]:
    out = []
    for k, p in FRACTIONAL_CASES:
        specs = C.GENERATORS[(k, p)]
        design = C.fractional_factorial(k, p, n_center=0)
        out.append(
            {
                "case": f"2^({k}-{p})",
                "n_factors": k,
                "n_generators": p,
                "n_runs": design.n_runs,
                "resolution": C.resolution_of(specs),
                "word_length_pattern": C.word_length_pattern(specs),
                "generators": specs,
                "d_efficiency_main": round(
                    d_efficiency(design.matrix, model_terms(k, ModelOrder.MAIN)), 9
                ),
            }
        )
    return out


def box_behnken_results() -> list[dict]:
    out = []
    for k in BOX_BEHNKEN_CASES:
        design = C.box_behnken(k, n_center=0)
        out.append(
            {
                "case": f"bbd-{k}",
                "n_factors": k,
                "n_runs": design.n_runs,
                "matrix": _sorted_matrix(design.matrix),
            }
        )
    return out


def central_composite_results() -> list[dict]:
    out = []
    for k, rule in CENTRAL_COMPOSITE_CASES:
        design = C.central_composite(k, alpha=rule, n_center=0)
        out.append(
            {
                "case": f"ccd-{k}-{rule}",
                "n_factors": k,
                "alpha_rule": rule,
                "alpha": round(design.detail["alpha"], 9),
                "n_runs": design.n_runs,
                "n_factorial_points": design.detail["n_factorial_points"],
                "n_axial_points": design.detail["n_axial_points"],
            }
        )
    return out


def dsd_results() -> list[dict]:
    """Checked against the design's published defining properties rather than a
    reference implementation, since no DSD package is installed."""
    out = []
    for k in DSD_CASES:
        design = C.definitive_screening(k, n_center=1)
        main = model_matrix(design.matrix, [(i,) for i in range(k)])
        two_fi = model_matrix(
            design.matrix, [(i, j) for i in range(k) for j in range(i + 1, k)]
        )
        out.append(
            {
                "case": f"dsd-{k}",
                "n_factors": k,
                "n_runs": design.n_runs,
                "conference_order": design.detail["conference_order"],
                "levels": sorted(float(v) for v in np.unique(design.matrix)),
                "max_main_by_2fi_inner_product": round(float(np.abs(main.T @ two_fi).max()), 12),
            }
        )
    return out


def power_results() -> list[dict]:
    out = []
    for family, k, nc, order, effect in POWER_CASES:
        if family == "full":
            design = C.full_factorial(k, n_center=nc)
        elif family == "ccd-face":
            design = C.central_composite(k, alpha="face", n_center=nc)
        else:
            design = C.box_behnken(k, n_center=nc)
        spec = DesignSpec(
            factors=[Factor(f"x{i + 1}", -1.0, 1.0) for i in range(k)],
            responses=[Response("y", target_effect=effect, noise_sd=1.0)],
            model_order=ModelOrder(order),
        )
        report = power_report(design, spec)
        out.append(
            {
                "case": f"power-{family}-{k}-c{nc}-{order}",
                "n_runs": design.n_runs,
                "residual_df": report.residual_df,
                "main": report.min_main_effect_power,
                "interaction": report.min_interaction_power,
                "curvature": report.min_curvature_power,
            }
        )
    return out


def blocking_results() -> list[dict]:
    """Which factorial words the block contrasts are confounded with."""
    from itertools import combinations

    from doe_advisor.designs.blocking import assign_blocks
    from doe_advisor.designs.model import block_columns

    out = []
    for k, b in BLOCKING_CASES:
        design = C.full_factorial(k)
        labels = assign_blocks(design.matrix, model_terms(k, ModelOrder.INTERACTION), b)
        blocks = block_columns(labels)
        lengths = []
        for r in range(1, k + 1):
            for word in combinations(range(k), r):
                col = np.prod(design.matrix[:, list(word)], axis=1)
                coef, *_ = np.linalg.lstsq(blocks, col, rcond=None)
                if np.allclose(blocks @ coef, col):  # the word lies in the block space
                    lengths.append(r)
        out.append({"case": f"2^{k} in {b} blocks", "block_word_lengths": sorted(lengths)})
    return out


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "doe-advisor",
        "fractional": fractional_results(),
        "box_behnken": box_behnken_results(),
        "central_composite": central_composite_results(),
        "dsd": dsd_results(),
        "power": power_results(),
        "blocking": blocking_results(),
    }
    path = RESULTS_DIR / "tool.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    total = sum(len(v) for k, v in payload.items() if isinstance(v, list))
    print(f"wrote {path} ({total} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
