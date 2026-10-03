"""Turn computed scores into the pros-and-cons prose a scientist reads.

Two paths, one interface, so the memo builder does not care which ran:

- :class:`TemplateNarrator` — deterministic, rule-based, no API key, always
  available. This is the floor: the tool must never become unusable because IT
  has not approved an API key.
- :class:`ClaudeNarrator` — better prose, via the Anthropic SDK.

**The engine owns every number.** Claude is handed pre-formatted facts and is
forbidden from introducing figures of its own. That is enforced, not merely
requested: :func:`_unsupported_numbers` re-reads the generated text and rejects
it if any numeric token is absent from the facts it was given, falling back to
the template. A narration layer that can quietly invent a power figure is worse
than no narration layer at all.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .candidates import DEFAULT_MIN_POWER, ScoredDesign
from .designs.properties import noise_stress_factor
from .designs.spec import DesignSpec, ModelOrder

DEFAULT_MODEL = os.environ.get("DOE_ADVISOR_MODEL", "claude-sonnet-5")

ROLE_HEADINGS = {
    "recommended": "Recommended",
    "economical": "Cheaper option",
    "thorough": "More thorough option",
}


@dataclass
class OptionNarrative:
    option_id: str
    option_name: str
    role: str
    n_runs: int
    summary: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


@dataclass
class Narration:
    source: str  # "claude" or "template"
    headline: str
    options: list[OptionNarrative] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    fallback_reason: str | None = None


# --------------------------------------------------------------------------
# Facts: the only numbers anyone downstream is allowed to use
# --------------------------------------------------------------------------


def option_facts(option: ScoredDesign, spec: DesignSpec, cheapest: int, dearest: int, index: int = 0) -> dict:
    """Everything true about one option, pre-formatted for presentation.

    ``id`` is what everything downstream keys on. Display names are for humans
    and can collide; the id cannot, so a narrative can never be attached to the
    wrong design.
    """
    props = option.properties
    power = props.power.min_power
    detail = option.design.detail

    facts: dict = {
        "id": f"option-{index + 1}",
        "name": option.design.name,
        "role": option.roles[0] if option.roles else "alternative",
        "runs": props.n_runs,
        "centre_points": option.design.n_center_points,
        "model_terms": props.n_model_terms,
        "residual_df": props.residual_df,
        "score": round(option.score, 3),
        "min_power_pct": None if power is None else round(power * 100),
        "main_power_pct": _pct(props.power.min_main_effect_power),
        "interaction_power_pct": _pct(props.power.min_interaction_power),
        "curvature_power_pct": _pct(props.power.min_curvature_power),
        "worst_alias": round(props.aliasing.worst_main_effect_alias, 2),
        "prediction_i_value": round(props.prediction.i_value, 2),
        "g_efficiency_pct": round(props.prediction.g_efficiency * 100),
        "detectable_effect_sd": (
            None if props.power.detectable_effect_sd is None else round(props.power.detectable_effect_sd, 2)
        ),
        "detectable_effect_units": (
            None if props.power.detectable_effect_units is None else _sig(props.power.detectable_effect_units)
        ),
        "min_power_pct_if_noise_high": (
            None if props.power.min_power_if_noise_high is None else round(props.power.min_power_if_noise_high * 100)
        ),
        "noise_stress_factor": round(props.power.noise_stress_factor, 2),
        "response_units": spec.primary_response.display_units if spec.primary_response else "",
        "log_scale": bool(spec.primary_response and spec.primary_response.is_log),
        "detectable_fold": (
            _sig(10 ** props.power.detectable_effect_units)
            if spec.primary_response and spec.primary_response.is_log and props.power.detectable_effect_units
            else None
        ),
        "target_fold": spec.primary_response.target_fold if spec.primary_response else None,
        "noise_cv_pct": spec.primary_response.noise_cv_pct if spec.primary_response else None,
        "target_effect": spec.primary_response.target_effect if spec.primary_response else None,
        "is_cheapest": props.n_runs == cheapest,
        "is_dearest": props.n_runs == dearest,
        "three_level": bool(detail.get("three_level", False)),
        "replicates": detail.get("replicates"),
        "resolution": detail.get("resolution"),
        "fraction": detail.get("fraction"),
        "exceeds_declared_range": bool(detail.get("exceeds_declared_range", False)),
        "alias_statements": props.aliasing.statements(limit=3),
    }
    if props.robustness.applicable:
        facts["robust_pct"] = round(props.robustness.fraction_estimable * 100)
        facts["expected_losses"] = props.robustness.n_losses
        facts["worst_power_pct_after_loss"] = _pct(props.robustness.worst_power_after_loss)
    return facts


def _pct(value: float | None) -> int | None:
    return None if value is None else round(value * 100)


def _sig(value: float, digits: int = 3) -> float:
    """Round to a few significant figures, for numbers in the scientist's units."""
    if value == 0:
        return 0.0
    return float(f"{value:.{digits}g}")


# --------------------------------------------------------------------------
# Template narrator — the floor, always available
# --------------------------------------------------------------------------


class TemplateNarrator:
    """Deterministic narration built from the computed facts.

    Plainer than the Claude path, but it never fails, never invents, and its
    output is byte-identical for identical inputs — which is what makes it a
    safe default under a 'traceability designed in' posture.
    """

    source = "template"

    def narrate(self, spec: DesignSpec, options: list[ScoredDesign]) -> Narration:
        if not options:
            return Narration(
                source=self.source,
                headline=(
                    "No design in the classical families can meet this specification. "
                    "Either the run budget is too small for the model you asked for, or the model "
                    "needs to be simplified."
                ),
                caveats=[_budget_caveat(spec)],
            )

        cheapest = min(o.n_runs for o in options)
        dearest = max(o.n_runs for o in options)
        recommended = next((o for o in options if "recommended" in o.roles), options[0])

        narratives = [
            self._one(option_facts(o, spec, cheapest, dearest, i), spec) for i, o in enumerate(options)
        ]
        return Narration(
            source=self.source,
            headline=self._headline(recommended, spec),
            options=narratives,
            caveats=_caveats(spec, options),
        )

    def _headline(self, recommended: ScoredDesign, spec: DesignSpec) -> str:
        power = recommended.properties.power.min_power
        clause = (
            f" and would detect the effect you care about about {round(power * 100)}% of the time"
            if power is not None
            else ""
        )
        return (
            f"{recommended.design.name} in {recommended.n_runs} runs is the best balance here: "
            f"it can fit {spec.model_order.label}{clause}."
        )

    def _one(self, f: dict, spec: DesignSpec) -> OptionNarrative:
        pros, cons = [], []

        units = f" {f['response_units']}" if f["response_units"] else ""
        if f["min_power_pct"] is not None:
            pct = f["min_power_pct"]
            if pct >= 90:
                pros.append(
                    f"Strong power: if the effect you care about is real, this design would find it "
                    f"about {pct} times in 100."
                )
            elif pct >= DEFAULT_MIN_POWER * 100:
                pros.append(
                    f"Adequate power at about {pct}%, just above the usual 80% bar — about {100 - pct} "
                    f"campaigns in 100 would still miss a real effect."
                )
            else:
                cons.append(
                    f"Underpowered at about {pct}% — if the effect is real, about {100 - pct} campaigns in 100 "
                    f"would miss it. There is a real chance of running everything and concluding nothing."
                )
            # Say which kind of term is the weak one. "Main effects are fine but
            # the curvature is not" is a different problem from "everything is
            # weak", and it is the one an optimisation study usually has.
            main_pct = f["main_power_pct"]
            for kind, words in (("curvature", "curvature (squared) terms"), ("interaction", "interactions")):
                kind_pct = f[f"{kind}_power_pct"]
                if (
                    kind_pct is not None
                    and kind_pct < DEFAULT_MIN_POWER * 100
                    and main_pct is not None
                    and kind_pct < main_pct
                ):
                    cons.append(
                        f"The {words} are the weak point: about {kind_pct}% power, against about {main_pct}% "
                        f"for the main effects."
                    )
            # The same fact in the scientist's units: what is the smallest
            # change this design can reliably see, and how does it compare with
            # the change they asked for?
            if f["detectable_effect_units"] is not None and f["target_effect"] is not None:
                if f["log_scale"] and f["detectable_fold"] is not None and f["target_fold"] is not None:
                    seen_text, asked_text = f"a {f['detectable_fold']}-fold change", f"{f['target_fold']}-fold"
                else:
                    seen_text = f"{f['detectable_effect_units']}{units}"
                    asked_text = f"{f['target_effect']}{units}"
                if pct >= DEFAULT_MIN_POWER * 100:
                    pros.append(
                        f"The smallest change it can reliably see (at 80% power) is about {seen_text}; "
                        f"you asked to see {asked_text}, which is larger, so you have margin."
                    )
                else:
                    cons.append(
                        f"The smallest change it can reliably see (at 80% power) is about {seen_text}; "
                        f"the {asked_text} you asked for is smaller than that, so it will often go unnoticed."
                    )
            if f["min_power_pct_if_noise_high"] is not None and pct >= DEFAULT_MIN_POWER * 100:
                worse = f["min_power_pct_if_noise_high"]
                if worse < DEFAULT_MIN_POWER * 100:
                    cons.append(
                        f"Sensitive to your noise estimate: if the real run-to-run SD is "
                        f"{f['noise_stress_factor']} times what you "
                        f"entered, power falls to about {worse}%."
                    )
                else:
                    pros.append(
                        f"Forgiving of a bad noise estimate: even if the real run-to-run SD is "
                        f"{f['noise_stress_factor']} times "
                        f"what you entered, power stays at about {worse}%."
                    )
        elif f["detectable_effect_sd"] is not None:
            in_units = (
                f" — about {f['detectable_effect_units']}{units}" if f["detectable_effect_units"] is not None else ""
            )
            pros.append(
                f"Without a stated target effect, power cannot be computed. What can be said: at 80% power "
                f"this design can reliably see a change of about {f['detectable_effect_sd']} times your "
                f"run-to-run noise{in_units}. Smaller changes would often go unnoticed."
            )

        if f["worst_alias"] == 0:
            pros.append("Every term in your model is cleanly separated — nothing is confounded.")
        elif f["worst_alias"] >= 0.99:
            cons.append(f"Completely confounded: {f['alias_statements'][0]}")
        else:
            cons.append(f"Partly confounded: {f['alias_statements'][0]}")

        if f["is_cheapest"]:
            pros.append(f"Cheapest option on the table at {f['runs']} runs.")
        if f["is_dearest"]:
            cons.append(f"Most expensive option at {f['runs']} runs.")

        if f["three_level"]:
            pros.append("Uses three levels per factor, so it can tell you whether the response curves.")

        # Centre points are not scored on any of the four axes, so their value
        # would otherwise be invisible — and their absence is easy to miss in a
        # design that scores well precisely because it spent nothing on them.
        if f["centre_points"] >= 2 and not f["three_level"]:
            cons.append(
                f"Its {f['centre_points']} centre points cost runs that add no power, but they are what "
                f"lets you check whether the response is curved before you commit to a straight-line model."
            )
        elif f["centre_points"] == 0:
            cons.append(
                "No centre points, so this design cannot tell you whether the response is curved. "
                "If curvature is plausible, add two or three."
            )

        if f["replicates"]:
            cons.append(
                f"Replication ({f['replicates']}x) buys precision and a clean estimate of pure error, "
                f"but it does not untangle anything that was confounded to begin with."
            )

        if f["exceeds_declared_range"]:
            cons.append(
                "Its star points sit outside the ranges you declared, so some runs ask the process "
                "to go further than you said was safe. Check they are actually runnable."
            )

        if f["residual_df"] <= 2:
            cons.append(
                f"Only {f['residual_df']} degrees of freedom left to estimate noise, so the error bars "
                f"on everything will be wide."
            )

        if "robust_pct" in f:
            after = f.get("worst_power_pct_after_loss")
            if f["robust_pct"] == 100 and after is not None and after < DEFAULT_MIN_POWER * 100:
                cons.append(
                    f"Losing {f['expected_losses']} run(s) still leaves a model that can be fitted, but in the "
                    f"worst case power falls to about {after}%."
                )
            elif f["robust_pct"] == 100:
                pros.append(f"Survives losing any {f['expected_losses']} runs and still answers the question.")
            else:
                cons.append(
                    f"Fragile: in about {100 - f['robust_pct']}% of scenarios where {f['expected_losses']} "
                    f"runs are lost, the model can no longer be fitted at all."
                )

        summary = (
            f"{f['runs']} runs, fitting {f['model_terms']} terms with {f['residual_df']} degrees of "
            f"freedom left over for error."
        )
        return OptionNarrative(
            option_id=f["id"],
            option_name=f["name"],
            role=f["role"],
            n_runs=f["runs"],
            summary=summary,
            pros=pros,
            cons=cons,
        )


def _budget_caveat(spec: DesignSpec) -> str:
    if spec.max_runs is None:
        return "No run budget was set, so nothing was excluded on cost."
    return f"Everything here respects your budget of {spec.max_runs} runs."


def _caveats(spec: DesignSpec, options: list[ScoredDesign]) -> list[str]:
    """Things true of the whole comparison, not of any one option."""
    notes = [_budget_caveat(spec)]

    response = spec.primary_response
    if response is None or response.standardised_effect is None:
        notes.append(
            "No target effect and noise estimate were given, so power could not be computed and the "
            "ranking leans entirely on the other axes. Supplying both would sharpen this considerably."
        )
    else:
        units = f" {response.display_units}" if response.display_units else ""
        stated = (
            f"Power assumes run-to-run noise of {response.noise_cv_pct:g}% CV and a target "
            f"{response.target_fold:g}-fold change, both taken to the log10 scale — a signal"
            if response.is_log and response.noise_cv_pct and response.target_fold
            else f"Power assumes a run-to-run standard deviation of {response.noise_sd}{units} and a target "
            f"effect of {response.target_effect}{units} — a signal"
        )
        notes.append(
            f"{stated} {response.standardised_effect:.2f} times "
            f"the size of the noise. The noise figure is the one to be sure of: if the real variability is "
            f"larger than you entered, every power figure here is optimistic, and each option shows how "
            f"much it would lose if the noise were {round(noise_stress_factor(response), 2)} times larger."
        )

    if response is not None and response.goal.wants_optimum and spec.model_order is not ModelOrder.QUADRATIC:
        notes.append(
            f"Your goal is to {response.goal_statement}. Finding a best setting or hitting a target usually "
            f"needs a model that can describe curvature — the designs here fit {spec.model_order.label}, which "
            f"is enough to find which factors matter and roughly which direction to move, but not to locate "
            f"an optimum. Plan a follow-up with the curvature model once the important factors are known."
        )

    if any(o.properties.prediction.axial_points_outside_range for o in options):
        notes.append(
            "At least one option places runs outside the factor ranges you declared. Confirm those "
            "settings are physically runnable before committing."
        )

    if spec.n_blocks > 1:
        partners = sorted({p for o in options for p in o.properties.aliasing.block_partners})
        notes.append(
            f"The runs are split into {spec.n_blocks} blocks (days or batches). Each block gets its own "
            "offset in the model, which costs degrees of freedom but stops a block-to-block shift from "
            "posing as a factor effect. Analyse with the block in the model; the analysis hint does this."
            + (
                f" The block split absorbs terms the model leaves out: {', '.join(partners[:4])}."
                if partners
                else ""
            )
        )
    notes.append(
        "These are classical designs. Constrained design spaces and must-include historical runs "
        "are not yet supported."
    )
    notes.append("Randomise the run order before executing — the run sheet gives one such order.")
    return notes


# --------------------------------------------------------------------------
# Claude narrator
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

SYSTEM_PROMPT = """You write short, plain-English design-of-experiments briefings for bench scientists \
in upstream bioprocess development. They know basic statistics but are not statisticians, and they are \
the ones who must make the final call.

You will be given a JSON object of computed facts about candidate experimental designs.

Absolute rules:
1. Use ONLY numbers that appear in the JSON you are given. Never calculate, round, estimate, or infer a \
new number. If you want to state a figure that is not in the JSON, leave it out instead.
2. Never contradict the facts. If a design is flagged as underpowered or confounded, say so plainly.
3. Do not recommend a different design from the one marked "recommended".
4. No hedging padding, no "it is important to note", no restating the question. Lead with what matters.
5. Write for someone deciding, not someone studying. Say what each option buys and what it costs.
6. Readers often do not know what "power" or "standard deviations" mean. When you mention power, say it \
as a chance of finding a real effect; when the facts give "detectable_effect_units", prefer that figure in \
the response's own units over anything expressed in standard deviations. "min_power_pct" is the weakest term in the model; "main_power_pct", "interaction_power_pct" and "curvature_power_pct" break it down. When the curvature or interaction figure is the low one, say so by name.

Reply with JSON only, matching this shape exactly:
{"headline": "one sentence naming the recommendation and why",
 "options": [{"id": "...", "summary": "one sentence", "pros": ["..."], "cons": ["..."]}]}
Every option in the input must appear in the output, echoing its "id" field exactly."""


def _allowed_numbers(payload: str) -> set[str]:
    return set(_NUMBER_RE.findall(payload))


def _unsupported_numbers(text: str, allowed: set[str]) -> set[str]:
    """Numeric tokens in ``text`` that were not present in the facts.

    Single digits 0-9 are tolerated: they appear in ordinary prose ("one of the
    two options", "3 factors") far more often than they constitute a smuggled
    statistic, and every figure that actually matters here has more than one
    digit or a decimal point.
    """
    found = set(_NUMBER_RE.findall(text))
    return {n for n in found - allowed if not (len(n) == 1 and n.isdigit())}


class ClaudeNarrator:
    """Narration via the Anthropic SDK, with the template as a hard fallback."""

    source = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._fallback = TemplateNarrator()

    @property
    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def narrate(self, spec: DesignSpec, options: list[ScoredDesign]) -> Narration:
        baseline = self._fallback.narrate(spec, options)
        if not options or not self.available:
            baseline.fallback_reason = "no API key or SDK available"
            return baseline

        cheapest = min(o.n_runs for o in options)
        dearest = max(o.n_runs for o in options)
        payload = json.dumps(
            {
                "question": {
                    "factors": [
                        {"name": f.name, "low": f.low, "high": f.high, "units": f.units} for f in spec.factors
                    ],
                    "model": spec.model_order.label,
                    "budget_runs": spec.max_runs,
                    "response": None
                    if spec.primary_response is None
                    else {
                        "name": spec.primary_response.name,
                        "units": spec.primary_response.display_units,
                        "scale": spec.primary_response.scale,
                        "noise_cv_pct": spec.primary_response.noise_cv_pct,
                        "target_fold": spec.primary_response.target_fold,
                        "goal": spec.primary_response.goal_statement,
                        "target_effect": spec.primary_response.target_effect,
                        "noise_sd": spec.primary_response.noise_sd,
                    },
                },
                "options": [option_facts(o, spec, cheapest, dearest, i) for i, o in enumerate(options)],
            },
            indent=2,
        )

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": payload}],
            )
            text = "".join(block.text for block in message.content if block.type == "text")
            parsed = json.loads(_extract_json(text))
        except Exception as exc:  # network, auth, malformed JSON — all fall back
            baseline.fallback_reason = f"narration call failed ({type(exc).__name__}); used templates"
            return baseline

        # The guardrail: reject prose containing figures the engine never produced.
        allowed = _allowed_numbers(payload)
        prose = json.dumps(parsed)
        smuggled = _unsupported_numbers(prose, allowed)
        if smuggled:
            baseline.fallback_reason = (
                f"narration introduced numbers not produced by the engine ({', '.join(sorted(smuggled))}); "
                f"used templates instead"
            )
            return baseline

        try:
            narratives = _merge(parsed, baseline)
        except (KeyError, TypeError):
            baseline.fallback_reason = "narration did not match the expected shape; used templates"
            return baseline

        return Narration(
            source=self.source,
            headline=str(parsed["headline"]),
            options=narratives,
            caveats=baseline.caveats,
        )


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be fenced or prefaced."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _merge(parsed: dict, baseline: Narration) -> list[OptionNarrative]:
    """Keep the engine's structural facts, take only the prose from Claude.

    Keyed on the id rather than the name: attaching a narrative to the wrong
    design would produce a memo whose prose and figures disagree, which is
    worse than a memo with no prose at all.
    """
    by_id = {o["id"]: o for o in parsed["options"]}
    merged = []
    for base in baseline.options:
        written = by_id.get(base.option_id)
        if written is None:
            merged.append(base)
            continue
        merged.append(
            OptionNarrative(
                option_id=base.option_id,
                option_name=base.option_name,
                role=base.role,
                n_runs=base.n_runs,
                summary=str(written.get("summary") or base.summary),
                pros=[str(p) for p in written.get("pros") or base.pros],
                cons=[str(c) for c in written.get("cons") or base.cons],
            )
        )
    return merged


def get_narrator(prefer_llm: bool = True) -> TemplateNarrator | ClaudeNarrator:
    """Best narrator available. Falls back silently to templates."""
    if prefer_llm:
        claude = ClaudeNarrator()
        if claude.available:
            return claude
    return TemplateNarrator()
