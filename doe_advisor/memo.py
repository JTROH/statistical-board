"""Assemble the decision memo: Markdown source, rendered to PDF.

Markdown is the source of truth — editable, diffable, reviewable in a pull
request — and the PDF is a render of it. That ordering matters under a
'traceability designed in' posture: the thing of record should be readable
without the tool that made it.

The memo is written to be read by someone who was not in the room: it states
the question, the options, the recommendation, the reasoning, and what would
change the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__, figures
from .candidates import AXIS_WEIGHTS, DEFAULT_MIN_POWER, ScoredDesign, top_options
from .designs.properties import CURVATURE_RULE, NOISE_UPPER_CONFIDENCE, PowerCurve, noise_stress_factor, power_curve
from .designs.spec import DesignSpec
from .narrate import Narration, get_narrator

ROLE_LABEL = {
    "recommended": "Recommended",
    "economical": "Cheaper",
    "thorough": "More thorough",
}


@dataclass
class Memo:
    markdown: str
    options: list[ScoredDesign]
    narration: Narration
    figure_dir: Path | None = None

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.markdown, encoding="utf-8")
        return path

    def to_pdf(self, path: str | Path, title: str = "Experimental design memo") -> Path:
        from pdstat.report import markdown_to_pdf

        return markdown_to_pdf(self.markdown, path, title=title, image_root=self.figure_dir)


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".2f", dash: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return format(value, spec)


def _question_section(spec: DesignSpec) -> str:
    rows = [
        f"| {f.name} | {f.low:g} | {f.high:g} | {f.units or '—'} |" for f in spec.factors
    ]
    lines = [
        "## The question",
        "",
        f"Fitting **{spec.model_order.label}** across {spec.n_factors} factors.",
        "",
        "| Factor | Low | High | Units |",
        "|---|---|---|---|",
        *rows,
        "",
    ]

    response = spec.primary_response
    if response is not None:
        units = f" {response.units}" if response.units else ""
        lines += [f"Response: **{response.name}**{f' ({response.units})' if response.units else ''}. "
                  f"Goal: **{response.goal_statement}**.", ""]
        if response.standardised_effect is not None and response.is_log:
            lines += [
                f"Smallest change worth detecting: a **{response.target_fold:g}-fold** change, against run-to-run "
                f"noise of **{response.noise_cv_pct:g}% CV** — a signal **{response.standardised_effect:.2f} "
                f"times** the size of the noise on the log10 scale.",
                "",
            ]
        elif response.standardised_effect is not None:
            lines += [
                f"Smallest change worth detecting: **{response.target_effect}{units}**, against a run-to-run "
                f"standard deviation of **{response.noise_sd}{units}** — a signal "
                f"**{response.standardised_effect:.2f} times** the size of the noise.",
                "",
            ]
        else:
            lines += [
                "No target effect or noise estimate was supplied, so power could not be computed.",
                "",
            ]

    budget = f"{spec.max_runs} runs" if spec.max_runs is not None else "not set"
    lines += [
        f"Run budget: **{budget}**. Centre points requested: **{spec.n_center_points}**. "
        f"Runs expected to be lost: **{spec.expected_run_losses}**.",
        "",
    ]
    return "\n".join(lines)


def _comparison_table(options: list[ScoredDesign]) -> str:
    header = (
        "| Option | Runs | Power | Confounding | Prediction (I) | Robustness | Score |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for option in options:
        props = option.properties
        power = props.power.min_power
        role = ROLE_LABEL.get(option.roles[0] if option.roles else "", "Alternative")
        alias = props.aliasing.worst_main_effect_alias
        alias_text = "none" if alias == 0 else ("complete" if alias >= 0.99 else f"partial ({alias:.2f})")
        robust = (
            f"{props.robustness.fraction_estimable:.0%}" if props.robustness.applicable else "—"
        )
        if props.robustness.worst_power_after_loss is not None:
            robust += f" (power then ≥ {props.robustness.worst_power_after_loss:.0%})"
        rows.append(
            f"| **{role}** — {option.design.name} | {option.n_runs} | "
            f"{'—' if power is None else f'{power:.0%}'} | {alias_text} | "
            f"{_fmt(props.prediction.i_value)} | {robust} | {option.score:.3f} |"
        )
    footnote = "*Lower I is better; every other column is better higher.*"
    return "\n".join(["## The options", "", header, *rows, "", footnote, ""])


def _stress(response) -> str:
    """The noise stress factor as printed, e.g. ``1.5`` or ``1.73``."""
    return f"{noise_stress_factor(response):.3g}"


def _power_primer(
    spec: DesignSpec, options: list[ScoredDesign], curve: PowerCurve | None, with_figure: bool
) -> str:
    """Plain-language explanation of the power column, for readers who do not
    live in statistics. Every number here comes from the engine."""
    response = spec.primary_response
    if response is None or response.noise_sd is None:
        return ""
    units = f" {response.display_units}" if response.display_units else ""
    lines = [
        "## How to read the power figures",
        "",
        f"**Noise SD** ({response.noise_sd:.4g}{units}) is how much {response.name} scatters when you run the "
        f"*same* settings twice. It is the size of a change that could be nothing but chance.",
        "",
    ]
    if response.is_log:
        lines += [
            f"**Log scale.** You gave the noise as a CV of {response.noise_cv_pct:g}%"
            + (f" and the target as a {response.target_fold:g}-fold change" if response.target_fold else "")
            + f". Noise that grows with the level is constant on a log scale, so every power figure here is "
            f"for **log10 {response.name}**: the CV becomes an SD of {response.noise_sd:.4g} log10 units"
            + (f" and the fold change a difference of {response.target_effect:.4g}" if response.target_fold else "")
            + ". Record raw values at the bench, and analyse their log10 afterwards — the run sheet's "
            "analysis hint already does.",
            "",
        ]
    if response.target_effect is not None:
        lines += [
            f"**Target effect** ({response.target_effect}{units}) is the smallest change you would act on. "
            f"Divided by the noise it gives the signal-to-noise ratio: "
            f"**{response.standardised_effect:.2f}**. Above about 2 the signal is clear; between 1 and 2 it takes "
            f"a careful design; below 1 the change is smaller than the scatter and needs many runs.",
            "",
            "**Power** is the chance the study finds that change *if it is really there*. 80% power means a "
            "1-in-5 chance of running everything and seeing nothing, even though the effect exists. Power is "
            "not a pass/fail mark; it is the odds you are accepting.",
            "",
        ]
    lines += [
        "**Smallest detectable change** is the same idea the other way round: the smallest real change each "
        "design would find at 80% power. If it is bigger than the change you care about, the design is too "
        "small for your question.",
        "",
        "| Option | Runs | Smallest change seen at 80% power | Main effects | Interactions | Curvature "
        f"| Weakest term | Weakest if noise is {_stress(response)}× |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value:.0%}"

    for option in options:
        pw = option.properties.power
        if pw.detectable_effect_units is None:
            seen = "—"
        elif response.is_log:
            seen = f"{10 ** pw.detectable_effect_units:.3g}-fold"
        else:
            seen = f"{pw.detectable_effect_units:.3g}{units}"
        lines.append(
            f"| {option.design.name} | {option.n_runs} | {seen} | {pct(pw.min_main_effect_power)} | "
            f"{pct(pw.min_interaction_power)} | {pct(pw.min_curvature_power)} | {pct(pw.min_power)} | "
            f"{pct(pw.min_power_if_noise_high)} |"
        )
    lines += [
        "",
        "*Power is shown for every kind of term in the model; the score uses the weakest. "
        f"{CURVATURE_RULE} A blank (—) means the model has no terms of that kind.*",
        "",
        (
            f"*The last column is the cost of the noise being worse than measured. Your noise SD comes from "
            f"{response.noise_df} degrees of freedom, so its {NOISE_UPPER_CONFIDENCE:.0%} upper confidence bound "
            f"is {_stress(response)}× the value entered; that is the stress used.*"
            if response.noise_df
            else "*The last column is the cost of guessing the noise too low. A noise SD taken from two or three "
            "repeats is easily off by half; one from centre points of a past study, or from many batches at the "
            "same set-point, is much safer.*"
        ),
        "",
    ]
    if curve is not None:
        if curve.runs_for_80 is not None:
            need = (
                f"**How many runs it takes.** An ideal two-level design reaches 80% power at your target "
                f"effect with about **{curve.runs_for_80} runs**"
            )
            if curve.runs_for_80_if_noise_high is not None:
                need += (
                    f", or about **{curve.runs_for_80_if_noise_high}** if the noise is really "
                    f"{curve.noise_stress_factor:.2g}× larger"
                )
            need += (
                ". Real designs sit on or below that curve: centre points and replicates add runs without "
                "adding power to a main effect."
            )
        else:
            need = (
                "**How many runs it takes.** No practical run count reaches 80% power at this target effect "
                "and noise; either the change you want to see is too small, or the noise estimate is too "
                "large, for a single study."
            )
        lines += [need, ""]
        if with_figure:
            lines += ['<img src="power.png" width="640">', ""]
    return "\n".join(lines)


def _option_detail(option: ScoredDesign, narrative) -> str:
    props = option.properties
    role = ROLE_LABEL.get(option.roles[0] if option.roles else "", "Alternative")
    lines = [
        f"### {role}: {option.design.name}",
        "",
        f"*{narrative.summary}*",
        "",
    ]
    if narrative.pros:
        lines += ["**In its favour**", "", *[f"- {p}" for p in narrative.pros], ""]
    if narrative.cons:
        lines += ["**Against it**", "", *[f"- {c}" for c in narrative.cons], ""]

    detail = option.design.detail
    facts = [f"{props.n_runs} runs ({option.design.n_center_points} centre points)"]
    if detail.get("resolution"):
        facts.append(f"resolution {detail['resolution']}")
    if detail.get("fraction") and detail["fraction"] != "full":
        facts.append(f"{detail['fraction']} fraction")
    if detail.get("alpha_rule"):
        facts.append(f"{detail['alpha_rule']} star points (alpha = {detail['alpha']:.3f})")
    facts.append(f"{props.n_model_terms} model terms, {props.residual_df} residual df")
    facts.append(f"D-efficiency {props.d_efficiency:.3f}, G-efficiency {props.prediction.g_efficiency:.0%}")
    lines += [f"<small>{'; '.join(facts)}.</small>", ""]
    return "\n".join(lines)


def _aliasing_section(options: list[ScoredDesign], spec: DesignSpec) -> str:
    entangled = [o for o in options if o.properties.aliasing.worst_main_effect_alias > 0]
    if not entangled:
        return ""
    lines = ["## What you will not be able to tell apart", ""]
    for option in entangled:
        lines.append(f"**{option.design.name}**")
        lines.append("")
        for statement in option.properties.aliasing.statements(limit=4):
            lines.append(f"- {statement}")
        lines.append("")
    return "\n".join(lines)


def _run_sheet(option: ScoredDesign, spec: DesignSpec, seed: int = 0) -> str:
    decoded = option.design.decoded(spec.factors)
    order = option.design.randomised_order(seed=seed)
    header = "| Run | " + " | ".join(f"{f.name}{f' ({f.units})' if f.units else ''}" for f in spec.factors) + " |"
    divider = "|---" * (spec.n_factors + 1) + "|"
    rows = [
        f"| {position + 1} | " + " | ".join(f"{decoded[run, j]:g}" for j in range(spec.n_factors)) + " |"
        for position, run in enumerate(order)
    ]
    return "\n".join(
        [
            f"## Run sheet — {option.design.name}",
            "",
            "Execute in this order. The randomisation is not cosmetic: it is what stops a drifting "
            "bioreactor or a warming incubator from masquerading as a factor effect.",
            "",
            header,
            divider,
            *rows,
            "",
        ]
    )


def _provenance(spec: DesignSpec, narration: Narration, seed: int) -> str:
    weights = ", ".join(f"{k} {v:.2f}" for k, v in AXIS_WEIGHTS.items())
    lines = [
        "## How this was produced",
        "",
        f"- Generated by doe-advisor {__version__} on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.",
        f"- Every number was computed by the design engine. Narrative prose: **{narration.source}**.",
        f"- Scoring weights (before renormalising for unavailable axes): {weights}.",
        f"- A design is treated as clearing the bar at power >= {DEFAULT_MIN_POWER:.0%}.",
        f"- Random seed {seed}; scoring is deterministic and re-runnable.",
    ]
    if narration.fallback_reason:
        lines.append(f"- Narration note: {narration.fallback_reason}.")
    lines.append("")
    lines += ["### Caveats", ""] + [f"- {c}" for c in narration.caveats] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_memo(
    spec: DesignSpec,
    title: str = "Experimental design memo",
    limit: int = 3,
    seed: int = 0,
    figure_dir: str | Path | None = None,
    prefer_llm: bool = True,
) -> Memo:
    """Build the full memo. Figures are written only if ``figure_dir`` is given."""
    options = top_options(spec, limit=limit, seed=seed)
    narration = get_narrator(prefer_llm=prefer_llm).narrate(spec, options)

    parts = [f"# {title}", "", _question_section(spec)]

    if not options:
        parts += ["## No viable design", "", narration.headline, "", _provenance(spec, narration, seed)]
        return Memo(markdown="\n".join(parts), options=[], narration=narration)

    parts += ["## Recommendation", "", f"> {narration.headline}", "", _comparison_table(options)]
    figure_path = Path(figure_dir) if figure_dir else None
    curve = power_curve(spec, up_to=max(int(1.5 * max(o.n_runs for o in options)), 24))
    if figure_path is not None and curve is not None:
        figures.power_curve_chart(curve, options, spec, figure_path / "power.png")
    primer = _power_primer(spec, options, curve, with_figure=figure_path is not None and curve is not None)
    if primer:
        parts.append(primer)

    if figure_path is not None:
        figures.options_chart(options, figure_path / "options.png")
        figures.fds_chart(options, figure_path / "fds.png")
        recommended = next((o for o in options if "recommended" in o.roles), options[0])
        figures.run_layout_chart(recommended, spec, figure_path / "layout.png")
        parts += ['<img src="options.png" width="640">', "", '<img src="fds.png" width="640">', ""]

    parts += ["## The options in detail", ""]
    # Keyed on the narration's stable id, not the display name: a narrative
    # attached to the wrong design would make the prose and the numbers in the
    # same section disagree, which is worse than having no prose at all.
    by_id = {n.option_id: n for n in narration.options}
    for index, option in enumerate(options):
        narrative = by_id.get(f"option-{index + 1}")
        if narrative is not None:
            parts.append(_option_detail(option, narrative))

    aliasing = _aliasing_section(options, spec)
    if aliasing:
        parts.append(aliasing)

    recommended = next((o for o in options if "recommended" in o.roles), options[0])
    if figure_path is not None:
        parts += ['<img src="layout.png" width="640">', ""]
    parts.append(_run_sheet(recommended, spec, seed=seed))
    parts.append(_provenance(spec, narration, seed))

    return Memo(
        markdown="\n".join(parts),
        options=options,
        narration=narration,
        figure_dir=figure_path,
    )
