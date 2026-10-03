"""Turn a scored design into plain JSON-safe data.

Shared by the web app (:mod:`doe_advisor.server`) and the CLI
(:mod:`doe_advisor.cli`) so both front ends report the *same* fields with the
same rounding. A second copy of this would be a second place for the two to
drift apart on what "the numbers" are.
"""

from __future__ import annotations


def serialise_option(option, response=None) -> dict:
    """One :class:`~doe_advisor.candidates.ScoredDesign` as a JSON-safe dict.

    ``response`` (the spec's primary response) only adds ``detectable_fold``
    when the design was powered on the log10 scale.
    """
    props = option.properties
    power = props.power.min_power
    detectable = props.power.detectable_effect_units
    fold = 10**detectable if response is not None and response.is_log and detectable is not None else None
    return {
        "detectable_fold": None if fold is None else round(fold, 4),
        "name": option.design.name,
        "family": option.design.family,
        "roles": option.roles,
        "n_runs": props.n_runs,
        "n_center_points": option.design.n_center_points,
        "n_model_terms": props.n_model_terms,
        "residual_df": props.residual_df,
        "score": round(option.score, 4),
        "sub_scores": {k: round(v, 4) for k, v in option.sub_scores.items()},
        "weights": {k: round(v, 4) for k, v in option.weights.items()},
        "score_explanation": option.score_explanation(),
        "power": None if power is None else round(power, 4),
        "power_by_kind": {
            kind: None if value is None else round(value, 4)
            for kind, value in (
                ("main", props.power.min_main_effect_power),
                ("interaction", props.power.min_interaction_power),
                ("curvature", props.power.min_curvature_power),
            )
        },
        "detectable_effect_sd": props.power.detectable_effect_sd,
        "detectable_effect_units": props.power.detectable_effect_units,
        "power_if_noise_high": (
            None if props.power.min_power_if_noise_high is None else round(props.power.min_power_if_noise_high, 4)
        ),
        "noise_stress_factor": round(props.power.noise_stress_factor, 4),
        "worst_alias": round(props.aliasing.worst_main_effect_alias, 4),
        "alias_statements": props.aliasing.statements(limit=4),
        "i_value": props.prediction.i_value if props.prediction.i_value != float("inf") else None,
        "g_efficiency": round(props.prediction.g_efficiency, 4),
        "d_efficiency": round(props.d_efficiency, 4),
        "fds_curve": props.prediction.fds_curve,
        "robustness_applicable": props.robustness.applicable,
        "robustness": round(props.robustness.fraction_estimable, 4),
        "worst_power_after_loss": (
            None
            if props.robustness.worst_power_after_loss is None
            else round(props.robustness.worst_power_after_loss, 4)
        ),
        "exceeds_declared_range": props.prediction.axial_points_outside_range,
        "detail": {k: v for k, v in option.design.detail.items() if k != "defining_relation"},
    }


def curve_extent(options) -> int:
    """How far to draw the power curve: past the dearest option, with room to
    show where 80% is reached if it is not reached yet."""
    return max(int(1.5 * max(o.n_runs for o in options)), 24)
