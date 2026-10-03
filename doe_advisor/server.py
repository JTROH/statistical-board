"""Local FastAPI app. Loopback only — nothing here is meant to face a network.

Deliberately stateless: every request carries the whole form, and nothing is
persisted server-side. That keeps Phase 1 simple and honest about what it is
(a calculator with a good memo writer). Phase 4's lock-and-verify is what
introduces stored records, and it will want a real store rather than an
accidental one grown out of session handling.

    python3 -m doe_advisor.server        # or: python3 run.py --open
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, figures
from .candidates import AXIS_WEIGHTS, DEFAULT_MIN_POWER, score_candidates, top_options
from .designs.properties import power_curve
from .designs.spec import ModelOrder, ResponseGoal
from .intake import IntakeError, extract_form_from_text, list_presets, llm_available, spec_from_dict
from .memo import build_memo
from .serialise import curve_extent as _curve_extent
from .serialise import serialise_option as _serialise

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="doe-advisor", version=__version__)


class FormPayload(BaseModel):
    factors: list[dict] = []
    responses: list[dict] = []
    model_order: str = "interaction"
    max_runs: float | None = None
    n_center_points: float | None = 3
    expected_run_losses: float | None = 1
    hard_ranges: bool = False
    title: str | None = None


class DescriptionPayload(BaseModel):
    text: str = ""


def _bad_request(exc: IntakeError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@app.get("/api/capabilities")
def capabilities() -> dict:
    """What this installation can do.

    The front end uses ``llm`` to decide whether to show the chat box at all —
    an intake shortcut that errors on click is worse than one that is absent.
    """
    return {
        "version": __version__,
        "llm": llm_available(),
        "model_orders": [{"value": o.value, "label": o.label} for o in ModelOrder],
        "response_goals": [{"value": g.value, "label": g.label} for g in ResponseGoal],
        "axis_weights": AXIS_WEIGHTS,
        "min_power": DEFAULT_MIN_POWER,
    }


@app.get("/api/presets")
def presets() -> dict:
    return {"presets": list_presets()}


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------


@app.post("/api/intake/extract")
def intake_extract(payload: DescriptionPayload):
    """Populate the form from free text. The scientist still confirms it."""
    try:
        return {"form": extract_form_from_text(payload.text)}
    except IntakeError as exc:
        return _bad_request(exc)


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------


@app.post("/api/design")
def design(payload: FormPayload):
    """Score every candidate and return the options table."""
    try:
        spec = spec_from_dict(payload.model_dump())
    except IntakeError as exc:
        return _bad_request(exc)

    options = top_options(spec, limit=3)
    all_scored = score_candidates(spec)

    charts: dict[str, str] = {}
    curve = None
    if options:
        curve = power_curve(spec, up_to=_curve_extent(options))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            figures.options_chart(options, tmp_path / "options.png")
            figures.fds_chart(options, tmp_path / "fds.png")
            recommended = next((o for o in options if "recommended" in o.roles), options[0])
            figures.run_layout_chart(recommended, spec, tmp_path / "layout.png")
            keys = ["options", "fds", "layout"]
            if curve is not None:
                figures.power_curve_chart(curve, options, spec, tmp_path / "power.png")
                keys.append("power")
            for key in keys:
                data = (tmp_path / f"{key}.png").read_bytes()
                charts[key] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    response = spec.primary_response
    return {
        "options": [_serialise(o) for o in options],
        # What every power figure above was computed from, so the front end can
        # say "you asked for X; this design sees Y" in the scientist's units.
        "power_basis": None
        if response is None
        else {
            "name": response.name,
            "units": response.units,
            "goal": response.goal.value,
            "goal_statement": response.goal_statement,
            "target_value": response.target_value,
            "target_effect": response.target_effect,
            "noise_sd": response.noise_sd,
            "standardised_effect": response.standardised_effect,
            "runs_for_80_power": None if curve is None else curve.runs_for_80,
            "runs_for_80_power_if_noise_high": None if curve is None else curve.runs_for_80_if_noise_high,
            "noise_df": response.noise_df,
            "noise_stress_factor": None if curve is None else round(curve.noise_stress_factor, 4),
        },
        "rejected": [
            {"name": s.design.name, "n_runs": s.n_runs, "reason": s.disqualified}
            for s in all_scored
            if not s.is_viable
        ],
        "charts": charts,
        "n_candidates": len(all_scored),
    }


@app.post("/api/memo")
def memo(payload: FormPayload):
    """Render the decision memo and hand back the PDF."""
    try:
        spec = spec_from_dict(payload.model_dump())
    except IntakeError as exc:
        return _bad_request(exc)

    tmp_dir = Path(tempfile.mkdtemp(prefix="doe-advisor-memo-"))
    built = build_memo(spec, title=payload.title or "Experimental design memo", figure_dir=tmp_dir)
    built.write(tmp_dir / "memo.md")
    pdf_path = built.to_pdf(tmp_dir / "memo.pdf")
    return FileResponse(pdf_path, media_type="application/pdf", filename="design-memo.pdf")


@app.post("/api/memo/markdown")
def memo_markdown(payload: FormPayload):
    """The Markdown source, for review or version control."""
    try:
        spec = spec_from_dict(payload.model_dump())
    except IntakeError as exc:
        return _bad_request(exc)

    built = build_memo(spec, title=payload.title or "Experimental design memo")
    return {"markdown": built.markdown, "narration_source": built.narration.source}


# Mounted last so the API routes above take precedence.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8711)


if __name__ == "__main__":
    main()
