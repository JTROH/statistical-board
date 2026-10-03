"""Candidate generation and ranking.

The ranking is the tool's opinion, so these tests pin down the things that
opinion must never get wrong: it must not recommend a design that cannot fit
the model, it must not hide a cheap design's confounding, and it must be able
to explain its own arithmetic.
"""

from __future__ import annotations

import pytest

from doe_advisor.candidates import (
    AXIS_WEIGHTS,
    generate_candidates,
    score_candidates,
    top_options,
)
from doe_advisor.designs.spec import DesignSpec, Factor, ModelOrder, Response

UPSTREAM_FACTORS = [
    Factor("pH", 6.8, 7.4),
    Factor("DO", 30, 60, "%"),
    Factor("temp", 35, 37, "C"),
    Factor("feed", 2, 6, "mL/day"),
]


def make_spec(order=ModelOrder.INTERACTION, max_runs=40, k=4, losses=2, target=0.5, noise=0.25):
    factors = UPSTREAM_FACTORS[:k] if k <= 4 else [Factor(f"x{i}", 0, 1) for i in range(k)]
    return DesignSpec(
        factors=factors,
        responses=[Response("titer", "g/L", target_effect=target, noise_sd=noise)],
        model_order=order,
        max_runs=max_runs,
        n_center_points=3,
        expected_run_losses=losses,
    )


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_generation_offers_several_families():
    designs = generate_candidates(make_spec())
    families = {d.family for d in designs}
    assert {"full_factorial", "fractional_factorial", "definitive_screening"} <= families


def test_generation_offers_response_surface_designs_at_every_model_order():
    """'For nine more runs you would also learn whether the response curves' is
    a tradeoff worth showing even when the scientist only asked for a 2FI
    model."""
    families = {d.family for d in generate_candidates(make_spec(order=ModelOrder.MAIN))}
    assert "central_composite" in families
    assert "box_behnken" in families


def test_generation_offers_replicates():
    designs = generate_candidates(make_spec())
    assert any(d.detail.get("replicates") == 2 for d in designs)


def test_generation_deduplicates_identical_matrices():
    designs = generate_candidates(make_spec())
    seen = {(d.matrix.shape, d.matrix.tobytes()) for d in designs}
    assert len(seen) == len(designs)


def test_generation_survives_families_that_cannot_serve_the_factor_count():
    """Box-Behnken stops at 7 factors; generation must skip it, not crash."""
    designs = generate_candidates(make_spec(k=9, max_runs=None))
    assert designs
    assert all(d.family != "box_behnken" for d in designs)


# --------------------------------------------------------------------------
# Disqualification
# --------------------------------------------------------------------------


def test_design_that_cannot_fit_the_model_is_disqualified_not_hidden():
    """Showing the rejected design is the point: 'the cheap 8-run screen you
    were about to run cannot fit the model you asked for' is the most valuable
    thing this tool can say."""
    scored = score_candidates(make_spec(order=ModelOrder.INTERACTION))
    dead = [s for s in scored if not s.is_viable]
    assert dead
    assert any("not enough independent information" in (s.disqualified or "") for s in dead)


def test_over_budget_designs_are_disqualified_with_the_number():
    scored = score_candidates(make_spec(order=ModelOrder.QUADRATIC, max_runs=20))
    over = [s for s in scored if s.disqualified and "budget" in s.disqualified]
    assert over
    assert "20" in over[0].disqualified


def test_disqualified_designs_sort_last():
    scored = score_candidates(make_spec())
    viability = [s.is_viable for s in scored]
    assert viability == sorted(viability, reverse=True)


def test_disqualified_designs_are_never_recommended():
    scored = score_candidates(make_spec(order=ModelOrder.INTERACTION))
    assert all(not s.roles for s in scored if not s.is_viable)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_scores_are_bounded():
    for s in score_candidates(make_spec()):
        if s.is_viable:
            assert 0.0 <= s.score <= 1.0 + 1e-9


def test_weights_are_renormalised_when_an_axis_is_unavailable():
    """No stated effect size means power is unknown. The remaining axes must
    absorb its weight rather than the design being punished with a zero."""
    spec = make_spec()
    spec.responses = [Response("titer", "g/L")]  # no target effect
    viable = [s for s in score_candidates(spec) if s.is_viable]
    assert viable
    for s in viable:
        assert "power" not in s.sub_scores
        assert sum(s.weights.values()) == pytest.approx(1.0)


def test_all_axes_are_used_when_everything_is_known():
    viable = [s for s in score_candidates(make_spec()) if s.is_viable]
    assert set(viable[0].sub_scores) == set(AXIS_WEIGHTS)
    assert sum(viable[0].weights.values()) == pytest.approx(1.0)


def test_score_explanation_reproduces_the_total():
    """The scoring rule must be checkable by hand — no hidden weighting."""
    top = next(s for s in score_candidates(make_spec()) if s.is_viable)
    lines = top.score_explanation()
    assert lines[-1].startswith("total =")
    recomputed = sum(top.sub_scores[a] * top.weights[a] for a in top.sub_scores)
    assert recomputed == pytest.approx(top.score)


def test_confounded_design_scores_below_a_clean_one_of_similar_power():
    """The 7-factor case: a cheap replicated resolution-III screen can have
    excellent power and still be nearly useless, because every main effect is
    completely tangled with an interaction."""
    scored = score_candidates(make_spec(order=ModelOrder.MAIN, k=7, max_runs=24, losses=1))
    confounded = [s for s in scored if s.is_viable and s.properties.aliasing.worst_main_effect_alias > 0.9]
    clean = [s for s in scored if s.is_viable and s.properties.aliasing.worst_main_effect_alias < 1e-6]
    assert confounded and clean
    assert max(c.score for c in clean) > max(c.score for c in confounded)


def test_ranking_is_deterministic():
    first = [s.design.name for s in score_candidates(make_spec())]
    second = [s.design.name for s in score_candidates(make_spec())]
    assert first == second


# --------------------------------------------------------------------------
# Roles and the options table
# --------------------------------------------------------------------------


def test_exactly_one_design_is_recommended():
    scored = score_candidates(make_spec())
    assert sum("recommended" in s.roles for s in scored) == 1


def test_roles_land_on_distinct_designs():
    """A table where all three labels sit on one row is not an options table."""
    scored = score_candidates(make_spec(order=ModelOrder.MAIN))
    tagged = [s for s in scored if s.roles]
    assert len({id(s) for s in tagged}) == len(tagged)
    assert all(len(s.roles) == 1 for s in tagged)


def test_economical_is_cheaper_and_thorough_is_dearer_than_the_recommendation():
    scored = score_candidates(make_spec(order=ModelOrder.MAIN))
    rec = next(s for s in scored if "recommended" in s.roles)
    for s in scored:
        if "economical" in s.roles:
            assert s.n_runs < rec.n_runs
        if "thorough" in s.roles:
            assert s.n_runs > rec.n_runs


def test_recommendation_meets_the_thresholds_when_anything_does():
    scored = score_candidates(make_spec(order=ModelOrder.MAIN))
    rec = next(s for s in scored if "recommended" in s.roles)
    if any(s.meets_thresholds for s in scored):
        assert rec.meets_thresholds


def test_options_table_is_capped_and_ordered_by_run_count():
    options = top_options(make_spec(order=ModelOrder.MAIN), limit=3)
    assert 1 <= len(options) <= 3
    assert [o.n_runs for o in options] == sorted(o.n_runs for o in options)


def test_options_table_shows_distinct_variants_not_near_duplicates():
    """Rotatable and face-centred central composites are the same family but a
    different decision, so both may appear; three identical full factorials
    differing only in centre points may not."""
    options = top_options(make_spec(order=ModelOrder.QUADRATIC, max_runs=60), limit=3)
    keys = [
        (o.design.family, o.design.detail.get("alpha_rule"), o.design.detail.get("fraction"))
        for o in options
    ]
    assert len(set(keys)) == len(keys)


def test_options_are_all_viable():
    for order in (ModelOrder.MAIN, ModelOrder.INTERACTION, ModelOrder.QUADRATIC):
        for option in top_options(make_spec(order=order, max_runs=60)):
            assert option.is_viable
            assert option.properties.estimable


def test_impossible_budget_yields_no_options_rather_than_a_bad_one():
    options = top_options(make_spec(order=ModelOrder.QUADRATIC, max_runs=6))
    assert options == []


def test_hard_ranges_disqualify_designs_with_runs_outside_them():
    """Rotatable CCD axial points leave the declared cube. With hard limits on,
    that design must not be offered, let alone recommended."""
    spec = make_spec(order=ModelOrder.QUADRATIC, k=3, max_runs=20, losses=1)
    soft = {s.design.name: s for s in score_candidates(spec)}
    rotatable = [n for n in soft if "rotatable" in n]
    assert rotatable and all(soft[n].is_viable for n in rotatable)

    spec.hard_ranges = True
    hard = {s.design.name: s for s in score_candidates(spec)}
    for name in rotatable:
        assert not hard[name].is_viable
        assert "hard limits" in hard[name].disqualified
    assert all("rotatable" not in s.design.name for s in top_options(spec))


def test_hard_ranges_round_trip_through_the_form():
    from doe_advisor.intake import spec_from_dict, spec_to_dict

    spec = make_spec(k=3)
    spec.hard_ranges = True
    assert spec_from_dict(spec_to_dict(spec)).hard_ranges is True
    assert spec_from_dict({**spec_to_dict(spec), "hard_ranges": False}).hard_ranges is False
