"""Generate candidate designs, score them, and rank them into options.

This is the module that turns "here are some designs" into "here is a
decision". Two commitments shape it:

**The scoring rule is visible.** Every option carries its sub-scores and the
weights that produced its total, so a scientist (or their statistician) can see
exactly why one design outranked another and disagree with the weighting if
they want to. There is no hidden judgement.

**Options come with roles.** A ranked list of eight designs is not a decision
aid. What helps is the shape a good statistician offers: the thorough option,
the economical one that still clears the bar, and a recommendation between
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .designs import classical as C
from .designs.properties import DesignProperties, evaluate
from .designs.spec import Design, DesignSpec

# Defaults for "is this design good enough to be worth considering".
DEFAULT_MIN_POWER = 0.80
# A main effect contaminated by more than this is treated as compromised.
MAX_ACCEPTABLE_ALIAS = 0.5

# Weights for the overall quality score. Deliberately module-level constants
# rather than magic numbers inside a function, because they are reported to the
# user and are the first thing a statistician will want to argue with.
AXIS_WEIGHTS: dict[str, float] = {
    "power": 0.35,
    "aliasing": 0.25,
    "prediction": 0.25,
    "robustness": 0.15,
}


@dataclass
class ScoredDesign:
    """One candidate, its properties, and why it ranked where it did."""

    design: Design
    properties: DesignProperties
    sub_scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    roles: list[str] = field(default_factory=list)
    disqualified: str | None = None

    @property
    def n_runs(self) -> int:
        return self.properties.n_runs

    @property
    def is_viable(self) -> bool:
        return self.disqualified is None

    @property
    def meets_thresholds(self) -> bool:
        if not self.is_viable:
            return False
        power = self.properties.power.min_main_effect_power
        if power is not None and power < DEFAULT_MIN_POWER:
            return False
        return self.properties.aliasing.worst_main_effect_alias <= MAX_ACCEPTABLE_ALIAS

    def score_explanation(self) -> list[str]:
        """The arithmetic, spelled out, so the ranking can be checked by hand."""
        lines = []
        for axis, value in self.sub_scores.items():
            weight = self.weights.get(axis, 0.0)
            lines.append(f"{axis}: {value:.3f} x {weight:.2f} = {value * weight:.3f}")
        lines.append(f"total = {self.score:.3f}")
        return lines


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------


def generate_candidates(spec: DesignSpec) -> list[Design]:
    """Enumerate every design this spec could plausibly be run with.

    Generous rather than clever: generate widely, then let scoring and the
    budget filter decide. A design the scientist can see and reject is more
    useful than one silently withheld.
    """
    k = spec.n_factors
    nc = max(spec.n_center_points, 0)
    designs: list[Design] = []

    def add(factory, *args, **kwargs) -> None:
        try:
            designs.append(factory(*args, **kwargs))
        except (ValueError, KeyError):
            pass  # this family cannot serve this many factors; that is fine

    # Centre-point count is itself a real choice: more centre points buy a
    # curvature check and a pure-error estimate without touching the corners.
    center_options = sorted({0, nc, nc + 2}) if nc else sorted({0, 2, 4})

    for centers in center_options:
        add(C.full_factorial, k, n_center=centers)

    for p in range(1, k - 1):
        add(C.fractional_factorial, k, p, n_center=nc)
        # "Just run it twice" is a common instinct, so show what it does and
        # does not buy rather than leaving the scientist to assume.
        try:
            base = C.fractional_factorial(k, p, n_center=0)
            designs.append(C.replicate(base, 2))
        except (ValueError, KeyError):
            pass

    add(C.definitive_screening, k, n_center=max(nc, 1))

    # Response-surface designs are offered at every model order, not just the
    # quadratic one: "for nine more runs you would also learn whether the
    # response curves" is a tradeoff worth putting in front of the scientist.
    for alpha in ("rotatable", "face"):
        add(C.central_composite, k, alpha=alpha, n_center=max(nc, 1))
        add(C.central_composite, k, alpha=alpha, n_center=max(nc, 1), core_generators=1)
    add(C.box_behnken, k, n_center=max(nc, 1))

    # Deduplicate on the run matrix: several routes can land on the same design.
    unique: list[Design] = []
    seen: list[np.ndarray] = []
    for design in designs:
        if any(m.shape == design.matrix.shape and np.allclose(m, design.matrix) for m in seen):
            continue
        seen.append(design.matrix)
        unique.append(design)
    return unique


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _sub_scores(props: DesignProperties, best_i_value: float) -> tuple[dict[str, float], dict[str, float]]:
    """Normalise each axis onto 0..1, and return the weights actually used.

    Axes that cannot be computed — power with no stated effect size, robustness
    when no losses are expected — are dropped and the remaining weights
    renormalised, rather than being scored as zero. Scoring an unknown as zero
    would quietly punish designs for a gap in the scientist's inputs.
    """
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}

    power = props.power.min_main_effect_power
    if power is not None:
        scores["power"] = float(power)
        weights["power"] = AXIS_WEIGHTS["power"]

    scores["aliasing"] = float(max(0.0, 1.0 - props.aliasing.worst_main_effect_alias))
    weights["aliasing"] = AXIS_WEIGHTS["aliasing"]

    if np.isfinite(props.prediction.i_value) and props.prediction.i_value > 0:
        # Relative I-efficiency: the best design in this candidate set scores 1.
        scores["prediction"] = float(best_i_value / props.prediction.i_value)
        weights["prediction"] = AXIS_WEIGHTS["prediction"]

    if props.robustness.applicable:
        scores["robustness"] = float(props.robustness.fraction_estimable)
        weights["robustness"] = AXIS_WEIGHTS["robustness"]

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    return scores, weights


def score_candidates(spec: DesignSpec, designs: list[Design] | None = None, seed: int = 0) -> list[ScoredDesign]:
    """Evaluate and rank. Best first; disqualified designs last, but retained.

    Disqualified candidates are kept deliberately: "the cheap 8-run screen you
    were about to run cannot fit the model you asked for" is one of the most
    valuable things this tool can say, and it can only say it by showing the
    design it rejected.
    """
    designs = designs if designs is not None else generate_candidates(spec)
    scored: list[ScoredDesign] = []

    evaluated: list[tuple[Design, DesignProperties]] = [(d, evaluate(d, spec, seed=seed)) for d in designs]

    finite_i = [
        p.prediction.i_value
        for _, p in evaluated
        if np.isfinite(p.prediction.i_value) and p.prediction.i_value > 0
    ]
    best_i = min(finite_i) if finite_i else 1.0

    for design, props in evaluated:
        entry = ScoredDesign(design=design, properties=props)

        if not props.estimable:
            entry.disqualified = (
                f"cannot fit a model with {props.n_model_terms} terms — "
                f"{props.n_runs} runs is not enough independent information"
            )
        elif props.residual_df <= 0:
            entry.disqualified = "saturated: no degrees of freedom left to estimate uncertainty"
        elif spec.max_runs is not None and props.n_runs > spec.max_runs:
            entry.disqualified = f"needs {props.n_runs} runs, over your budget of {spec.max_runs}"

        if entry.is_viable:
            entry.sub_scores, entry.weights = _sub_scores(props, best_i)
            entry.score = sum(entry.sub_scores[a] * entry.weights.get(a, 0.0) for a in entry.sub_scores)
        scored.append(entry)

    # Best score first; among near-equals prefer fewer runs, because runs are
    # the thing the scientist is actually spending.
    scored.sort(key=lambda s: (not s.is_viable, -round(s.score, 3), s.n_runs))
    _assign_roles(scored)
    return scored


def _assign_roles(scored: list[ScoredDesign]) -> None:
    """Tag up to three *genuinely different* options.

    The shape a good statistician offers is not a ranked list, it is a spread:
    something cheaper, something recommended, something more thorough — so the
    scientist can see what more runs would buy and what fewer would cost. The
    roles are therefore defined relative to the recommendation and are only
    awarded when a distinct design actually occupies that position.

    ``economical`` and ``thorough`` are labels on *options presented*, not
    endorsements. Only ``recommended`` is an endorsement; the others exist so
    the tradeoff is visible, and narration states their downsides plainly.
    """
    viable = [s for s in scored if s.is_viable]
    if not viable:
        return

    passing = [s for s in viable if s.meets_thresholds]
    pool = passing or viable
    recommended = max(pool, key=lambda s: (round(s.score, 3), -s.n_runs))
    recommended.roles.append("recommended")

    cheaper = [s for s in viable if s.n_runs < recommended.n_runs]
    if cheaper:
        max(cheaper, key=lambda s: (round(s.score, 3), -s.n_runs)).roles.append("economical")

    richer = [s for s in viable if s.n_runs > recommended.n_runs]
    if richer:
        max(richer, key=lambda s: (round(s.score, 3), -s.n_runs)).roles.append("thorough")


def _variant_key(design: Design) -> tuple:
    """What counts as a genuinely different option.

    Finer than the family name, because choices *within* a family are often the
    ones that matter to a bench scientist. A face-centred central composite and
    a rotatable one are the same family but a different decision: only one of
    them keeps every run inside the ranges the scientist declared safe.
    """
    return (
        design.family,
        design.detail.get("alpha_rule"),
        design.detail.get("fraction"),
        design.detail.get("replicates"),
    )


def top_options(spec: DesignSpec, limit: int = 3, seed: int = 0) -> list[ScoredDesign]:
    """The options table, ordered cheapest to most thorough.

    Capped because a decision aid that presents eight choices has handed the
    problem back to the scientist rather than helping with it. Any spare slots
    are filled with designs from *families* not already shown, so the table
    offers real alternatives rather than three near-identical full factorials.
    """
    scored = score_candidates(spec, seed=seed)
    chosen = [s for s in scored if s.roles]

    if len(chosen) < limit:
        shown = {_variant_key(s.design) for s in chosen}
        for s in scored:
            if len(chosen) >= limit:
                break
            key = _variant_key(s.design)
            if s.is_viable and not s.roles and key not in shown:
                chosen.append(s)
                shown.add(key)

    # Reading order: cheapest first, so the run-count column tells a story.
    return sorted(chosen, key=lambda s: (s.n_runs, -s.score))[:limit]
