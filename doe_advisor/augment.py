"""Turn a finished analysis into the *next* design.

``stat_board`` already computes everything needed to say what to run next; today
those findings only ever become prose in a report's "Recommended Next
Experiments" section. This module reads the same JSON verbatim and proposes a
follow-up study.

What each diagnostic triggers:

===========================  ==================================  =========================
Engine output                Trigger                             Proposal
===========================  ==================================  =========================
``doe-optimum``              ``boundary_flags[f] == "boundary"``  widen that factor's range
``design-coverage``          curvature significant, or untested   fit curvature (quadratic)
``design-coverage``          ``low_power_warning``                more centre points
``vif``                      ``collinearity_concern``             break the confound
``predict``                  ``influential`` rows                 confirmation runs
``regression``               noise SD measured                    carry it into power
``stationary-point``         max/min outside the tested range     widen toward it
``stationary-point``         max/min inside the tested range      confirmation run there
===========================  ==================================  =========================

It returns a **form dict**, not a :class:`~doe_advisor.designs.spec.DesignSpec`,
so the result flows through :func:`doe_advisor.intake.spec_from_dict` like any
other input. There is deliberately no second path from evidence to a design.

No statistics are computed here. Every number comes from the engine output that
was passed in.
"""

from __future__ import annotations

from typing import Any

# How far past the tested edge to reach when an optimum sits on a boundary,
# as a fraction of the range already explored. Module-level because it is a
# policy choice a statistician will want to argue with, not a constant of nature.
RANGE_EXTENSION = 0.5

# Largest extension multiple we will ever propose in one step. Doubling a range
# on the strength of one boundary hit is a bigger extrapolation than the data
# supports.
MAX_EXTENSION = 1.0


def _numeric_levels(levels: Any) -> list[float] | None:
    """The tested levels of one factor as sorted numbers, or None if categorical."""
    try:
        values = sorted(float(v) for v in levels)
    except (TypeError, ValueError):
        return None
    return values or None


def _widen(low: float, high: float, side: str, fraction: float = RANGE_EXTENSION) -> tuple[float, float]:
    """Extend a range past the edge the optimum landed on.

    ``side`` is ``"low"``, ``"high"``, or ``"both"`` when which edge won cannot
    be told from the evidence.
    """
    fraction = min(fraction, MAX_EXTENSION)
    step = (high - low) * fraction
    if side == "low":
        return low - step, high
    if side == "high":
        return low, high + step
    return low - step, high + step


def _boundary_side(factor: str, best: dict, levels: list[float]) -> str:
    """Which edge the predicted optimum sits on, from the ranked-optimum output."""
    try:
        value = float(best[factor])
    except (KeyError, TypeError, ValueError):
        return "both"
    if abs(value - levels[0]) <= abs(value - levels[-1]):
        return "low"
    return "high"


def measured_noise(fit: dict | None) -> dict | None:
    """The run-to-run noise a previous fit measured, with its degrees of freedom.

    Pure error (scatter between true replicates, from the ``lack_of_fit``
    block) is preferred: it does not depend on the model being right. The
    model's residual SD is the fallback, and is labelled as such, because it
    also absorbs any lack of fit.
    """
    if not fit:
        return None
    lof = fit.get("lack_of_fit") or {}
    if lof.get("pure_error_sd") and lof.get("df_pure_error"):
        return {"sd": float(lof["pure_error_sd"]), "df": int(lof["df_pure_error"]), "source": "pure error"}
    if fit.get("residual_sd") and fit.get("residual_df"):
        return {"sd": float(fit["residual_sd"]), "df": int(fit["residual_df"]), "source": "model residual"}
    return None


def findings(coverage: dict | None = None, optimum: dict | None = None,
             vif: dict | None = None, predict: dict | None = None,
             fit: dict | None = None, stationary: dict | None = None) -> list[dict]:
    """The diagnostic triggers that fired, each with the evidence behind it.

    Returned separately from :func:`form_from_diagnostics` so a caller can show
    *why* the next design looks the way it does, and so a claim can always be
    traced back to the engine output that produced it.
    """
    out: list[dict] = []

    if optimum:
        flags = optimum.get("boundary_flags") or {}
        at_edge = [f for f, flag in flags.items() if flag == "boundary"]
        if at_edge:
            out.append({
                "trigger": "optimum_at_boundary",
                "source": "doe-optimum",
                "factors": at_edge,
                "detail": f"The best tested combination sits at the edge of the range for "
                          f"{', '.join(at_edge)}. The true optimum may lie beyond it.",
                "action": "extend_range",
            })

    if coverage:
        curvature = coverage.get("curvature")
        if curvature is None:
            if coverage.get("n_center_points", 0) == 0:
                out.append({
                    "trigger": "curvature_untested",
                    "source": "design-coverage",
                    "detail": "No centre points were run, so curvature -- a bend the tested "
                              "corners alone cannot reveal -- was never tested.",
                    "action": "fit_curvature",
                })
        else:
            p = curvature.get("p")
            if p is not None and p < 0.05:
                out.append({
                    "trigger": "curvature_detected",
                    "source": "design-coverage",
                    "detail": f"Centre runs differ from the corner average (p = {p:.4g}), so the "
                              f"response bends; a straight-line model will misstate the optimum.",
                    "action": "fit_curvature",
                })
            elif curvature.get("low_power_warning"):
                out.append({
                    "trigger": "curvature_underpowered",
                    "source": "design-coverage",
                    "detail": f"Only {curvature.get('center_point_df', 0) + 1} centre run(s) -- too few to "
                              f"call curvature either way (p = {p:.4g} proves nothing here).",
                    "action": "more_center_points",
                })
        fraction = coverage.get("coverage_fraction")
        if fraction is not None and fraction < 1.0:
            out.append({
                "trigger": "incomplete_coverage",
                "source": "design-coverage",
                "detail": f"Only {fraction:.0%} of the factor-level combinations were run "
                          f"({coverage.get('n_tested_combinations')} of "
                          f"{coverage.get('n_possible_combinations')}).",
                "action": "note_only",
            })

    if vif:
        confounded = [t for t in (vif.get("terms") or []) if t.get("collinearity_concern")]
        if confounded:
            worst = max(confounded, key=lambda t: t.get("vif") or 0)
            out.append({
                "trigger": "collinearity",
                "source": "vif",
                "terms": [t.get("term") for t in confounded],
                "detail": f"{len(confounded)} term(s) are entangled -- worst is "
                          f"{worst.get('term')} at VIF {worst.get('vif')}. Their effects cannot be "
                          f"cleanly separated by this design.",
                "action": "break_confound",
            })

    if predict:
        n_flagged = predict.get("n_flagged") or 0
        if n_flagged:
            rows = [r.get("row") for r in (predict.get("rows") or []) if r.get("influential")]
            out.append({
                "trigger": "influential_runs",
                "source": "predict",
                "rows": rows,
                "detail": f"{n_flagged} run(s) pull the fit around more than their share. "
                          f"They are candidates for a confirmation run, not for deletion.",
                "action": "confirmation_runs",
            })

    if stationary and stationary.get("kind") in ("maximum", "minimum"):
        location = stationary.get("location") or {}
        beyond = {f: v for f, v in location.items() if not v.get("inside_tested_range")}
        if beyond:
            sides = {f: ("low" if v["value"] < v["tested_range"][0] else "high") for f, v in beyond.items()}
            out.append({
                "trigger": "stationary_point_outside",
                "source": "stationary-point",
                "factors": list(beyond),
                "sides": sides,
                "detail": f"The fitted {stationary['kind']} lies outside the tested range for "
                          f"{', '.join(beyond)}. The model cannot vouch for it there; the next design "
                          f"should move toward it.",
                "action": "extend_range",
            })
        else:
            point = {f: v.get("value") for f, v in location.items()}
            out.append({
                "trigger": "stationary_point_inside",
                "source": "stationary-point",
                "point": point,
                "detail": f"The fitted {stationary['kind']} lies inside the tested region, at "
                          + ", ".join(f"{f} = {v:.4g}" for f, v in point.items())
                          + ". Confirmation runs there test the model directly.",
                "action": "confirmation_runs",
            })

    noise = measured_noise(fit)
    if noise:
        out.append({
            "trigger": "noise_measured",
            "source": "regression",
            "noise": noise,
            "detail": f"The previous study measured the run-to-run noise: SD {noise['sd']:.4g} from "
                      f"{noise['df']} degree(s) of freedom ({noise['source']}). It is in the units the "
                      f"model was fitted in -- if that was a log scale, the next design must be too.",
            "action": "set_noise",
        })

    return out


def form_from_diagnostics(coverage: dict | None = None, optimum: dict | None = None,
                          vif: dict | None = None, predict: dict | None = None,
                          fit: dict | None = None, stationary: dict | None = None,
                          *, factor_units: dict[str, str] | None = None,
                          response: dict | None = None,
                          max_runs: int | None = None) -> dict:
    """Propose the follow-up study as a form dict for ``intake.spec_from_dict``.

    ``coverage`` (the output of ``design-coverage``) is what supplies the factor
    ranges: its ``levels`` are the settings actually run, so the previous study's
    range is recovered from evidence rather than re-entered by hand. Categorical
    factors are skipped -- widening a range means nothing for them.

    ``response`` is an optional ``{"name", "units", "target_effect", "noise_sd",
    "goal"}`` dict carried over from the previous study.

    ``fit`` is the previous study's ``regression`` output. When given, its
    measured noise replaces a ``noise_sd`` the scientist left blank, and its
    degrees of freedom go along with it so the power stress test can use a real
    confidence bound instead of a guess. A ``noise_sd`` the scientist did enter
    is kept; the measured value is still reported in the findings.
    """
    if not coverage or not coverage.get("levels"):
        raise ValueError("design-coverage output is required: it carries the tested factor levels")

    fired = findings(coverage, optimum, vif, predict, fit, stationary)
    actions = {f["action"] for f in fired}
    units = factor_units or {}
    best = (optimum or {}).get("best") or {}
    at_edge = set(next((f["factors"] for f in fired if f["trigger"] == "optimum_at_boundary"), []))
    # The stationary point, when it lies outside, says which way to go more
    # directly than a boundary hit does, so it wins where both fire.
    toward = next((f["sides"] for f in fired if f["trigger"] == "stationary_point_outside"), {})

    factors: list[dict] = []
    skipped: list[str] = []
    for name, raw_levels in coverage["levels"].items():
        levels = _numeric_levels(raw_levels)
        if levels is None or len(levels) < 2:
            skipped.append(name)
            continue
        low, high = levels[0], levels[-1]
        if name in toward:
            low, high = _widen(low, high, toward[name])
        elif name in at_edge:
            low, high = _widen(low, high, _boundary_side(name, best, levels))
        factors.append({"name": name, "low": low, "high": high, "units": units.get(name, "")})

    # Curvature, once suspected or wanted, is the thing the next design must be
    # able to fit -- that is what turns a factorial into a central composite.
    model_order = "quadratic" if {"fit_curvature", "more_center_points"} & actions else "interaction"

    n_center = 3
    if "more_center_points" in actions:
        n_center = max(n_center, 2 * int(coverage.get("n_center_points") or 1) + 1)

    noise = measured_noise(fit)
    if response is not None and noise is not None and response.get("noise_sd") in (None, ""):
        response = {**response, "noise_sd": noise["sd"], "noise_df": noise["df"]}

    form: dict[str, Any] = {
        "factors": factors,
        "responses": [response] if response else [],
        "model_order": model_order,
        "n_center_points": n_center,
        "expected_run_losses": 1,
    }
    if max_runs is not None:
        form["max_runs"] = max_runs

    return {
        "form": form,
        "findings": fired,
        "skipped_factors": skipped,
        "rationale": [f["detail"] for f in fired] or
                     ["No diagnostic flagged a problem. A confirmation run at the predicted "
                      "optimum is the honest next step, not a bigger design."],
    }
