"""Blocking: which runs go on which day, and what that costs.

The textbook answers are the benchmark: a full 2^k factorial in two blocks
confounds the blocks with the k-factor interaction (the highest order, so the
minimum-aberration choice), and in four blocks the block contrasts avoid every
main effect and two-factor interaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_advisor.candidates import generate_candidates, score_candidates, top_options
from doe_advisor.designs import classical as C
from doe_advisor.designs.blocking import assign_blocks, block_sizes
from doe_advisor.designs.model import block_columns, model_terms, residual_df
from doe_advisor.designs.properties import evaluate, power_report
from doe_advisor.designs.spec import Design, DesignSpec, Factor, ModelOrder, Response
from doe_advisor.export import analysis_hint, run_sheet_csv
from doe_advisor.intake import IntakeError, spec_from_dict, spec_to_dict
from stat_board.engine import analyses


def _spec(k=3, blocks=2, order=ModelOrder.INTERACTION, centre=3):
    return DesignSpec(
        factors=[Factor(f"x{i}", -1.0, 1.0) for i in range(k)],
        responses=[Response("y", target_effect=2.0, noise_sd=1.0)],
        model_order=order,
        n_center_points=centre,
        n_blocks=blocks,
    )


@pytest.mark.parametrize("k", [3, 4, 5])
@pytest.mark.parametrize("order", [ModelOrder.MAIN, ModelOrder.INTERACTION])
def test_two_blocks_sit_on_the_highest_order_interaction(k, order):
    d = C.full_factorial(k)
    labels = assign_blocks(d.matrix, model_terms(k, order), 2)
    top = np.prod(d.matrix, axis=1)
    assert abs(np.corrcoef(labels, top)[0, 1]) == pytest.approx(1.0)


def test_four_blocks_stay_clear_of_main_effects_and_two_factor_interactions():
    k = 5
    d = C.full_factorial(k)
    terms = model_terms(k, ModelOrder.INTERACTION)
    labels = assign_blocks(d.matrix, terms, 4)
    assert np.bincount(labels).tolist() == [8, 8, 8, 8]
    b = block_columns(labels)
    x = np.column_stack([np.prod(d.matrix[:, list(t)], axis=1) for t in terms if t])
    assert np.allclose(x.T @ b, 0.0)


def test_centre_points_are_spread_across_blocks():
    d = C.full_factorial(3, n_center=4)
    labels = assign_blocks(d.matrix, model_terms(3, ModelOrder.INTERACTION), 2)
    centre = np.all(d.matrix == 0, axis=1)
    assert np.bincount(labels[centre]).tolist() == [2, 2]


def test_block_sizes_are_balanced():
    assert block_sizes(11, 2) == [6, 5]
    assert block_sizes(12, 3) == [4, 4, 4]
    with pytest.raises(ValueError):
        assign_blocks(np.zeros((5, 2)), [()], 3)


def test_each_block_costs_one_degree_of_freedom():
    d = C.full_factorial(3, n_center=3)
    terms = model_terms(3, ModelOrder.INTERACTION)
    labels = assign_blocks(d.matrix, terms, 2)
    assert residual_df(d.matrix, terms, labels) == residual_df(d.matrix, terms) - 1


def test_orthogonal_blocking_keeps_coefficient_precision_and_loses_only_df():
    """Blocks orthogonal to the model leave every SE unchanged; power drops only
    through the lost degree of freedom."""
    spec = _spec(blocks=2)
    d = C.full_factorial(3, n_center=3)
    blocked = Design(d.name, d.family, d.matrix, d.factor_names, blocks=assign_blocks(
        d.matrix, model_terms(3, spec.model_order), 2))
    plain = power_report(d, _spec(blocks=1))
    with_blocks = power_report(blocked, spec)
    assert with_blocks.residual_df == plain.residual_df - 1
    assert with_blocks.min_power < plain.min_power
    assert with_blocks.min_power > plain.min_power - 0.1


def test_blocks_report_the_term_they_absorb():
    d = C.full_factorial(3)
    spec = _spec(blocks=2, centre=0)
    blocked = Design(d.name, d.family, d.matrix, d.factor_names, blocks=assign_blocks(
        d.matrix, model_terms(3, spec.model_order), 2))
    assert evaluate(blocked, spec).aliasing.block_partners == ["x0 x x1 x x2"]


def test_candidates_are_all_blocked_and_named_so():
    spec = _spec(blocks=2)
    designs = generate_candidates(spec)
    assert designs and all(d.n_blocks == 2 and d.name.endswith("2 blocks") for d in designs)
    assert any(s.is_viable for s in score_candidates(spec))


def test_blocked_run_sheet_runs_block_by_block():
    spec = _spec(blocks=2)
    option = next(o for o in top_options(spec) if "recommended" in o.roles)
    order = option.design.randomised_order(seed=0)
    blocks_in_order = option.design.blocks[order]
    assert (np.diff(blocks_in_order) >= 0).all()


def test_blocks_round_trip_through_the_form():
    spec = spec_from_dict({**spec_to_dict(_spec()), "n_blocks": 3})
    assert spec.n_blocks == 3
    with pytest.raises(IntakeError, match="whole number"):
        spec_from_dict({**spec_to_dict(_spec()), "n_blocks": 1.5})


def test_a_day_shift_is_removed_by_the_block_term(tmp_path):
    """End to end: blocked run sheet -> results with a big day-2 shift -> the
    analysis hint's formula fits C(block) and recovers the true x0 effect."""
    spec = _spec(blocks=2)
    spec.factors[0] = Factor("x0", 10, 20)
    option = next(o for o in top_options(spec) if "recommended" in o.roles)
    path = run_sheet_csv(option.design, spec, tmp_path / "runs.csv")
    df = pd.read_csv(path)
    rng = np.random.default_rng(4)
    df["y"] = 50 + 3 * df["x0_coded"] + 15 * (df["block"] == 2) + rng.normal(0, 0.5, len(df))
    df.to_csv(path, index=False)
    hint = analysis_hint(spec)
    assert hint["formula"].endswith("+ C(block)")
    fit = analyses.regression(str(path), hint["formula"])
    # x0 is in natural units (10..20), so its slope is 3 per 5 units.
    assert fit["coefficients"]["x0"]["coef"] == pytest.approx(0.6, abs=0.1)
