"""Command-line front door to the design engine. One subcommand per question;
every run prints a single JSON object to stdout (errors go to stderr as JSON
too, with a non-zero exit) so an agent can parse the result unambiguously.

Deliberately the same contract as ``python3 -m stat_board.engine``: JSON in,
JSON out, exit 2 on a bad request. That is what lets a Claude Code subagent
drive the design half of this package over Bash the same way ``stat-analyst``
drives the analysis half.

    python3 -m doe_advisor presets
    python3 -m doe_advisor options    --spec spec.json
    python3 -m doe_advisor candidates --spec spec.json
    python3 -m doe_advisor properties --spec spec.json --design "Full factorial"
    python3 -m doe_advisor runsheet   --spec spec.json --option 1 --out runs.csv

``--spec`` takes the same form object the web app posts, so
:func:`doe_advisor.intake.spec_from_dict` stays the single path from input to a
computation. Use ``-`` to read the spec from stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, export
from .candidates import AXIS_WEIGHTS, DEFAULT_MIN_POWER, score_candidates, top_options
from .designs.properties import power_curve
from .designs.spec import ModelOrder, ResponseGoal
from .intake import IntakeError, list_presets, llm_available, spec_from_dict
from .serialise import curve_extent, serialise_option


def _load_spec(path: str):
    """Read a form-shaped JSON file (or stdin) and validate it into a DesignSpec."""
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise IntakeError(f"cannot read spec {path!r}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IntakeError(f"spec {path!r} is not valid JSON: {exc}") from exc
    # A preset file nests the form under "form"; accept either shape.
    if isinstance(payload, dict) and "form" in payload and "factors" not in payload:
        payload = payload["form"]
    return spec_from_dict(payload)


def _pick_option(options: list, selector: str | int | None):
    """Choose one scored option by 1-based rank, role, or design name."""
    if not options:
        raise IntakeError("no viable design was found for this spec")
    if selector is None:
        return next((o for o in options if "recommended" in o.roles), options[0])
    text = str(selector).strip()
    if text.isdigit():
        index = int(text)
        if not 1 <= index <= len(options):
            raise IntakeError(f"option {index} is out of range (1..{len(options)})")
        return options[index - 1]
    for option in options:
        if text.lower() in (option.design.name.lower(), *(r.lower() for r in option.roles)):
            return option
    names = ", ".join(o.design.name for o in options)
    raise IntakeError(f"no option matching {text!r}; available: {names}")


def _add_spec_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", required=True, help="form-shaped JSON file describing the study ('-' for stdin)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m doe_advisor",
        description="Design-of-experiments engine: generate, score and export classical designs. "
                    "JSON in, JSON out.")
    parser.add_argument("--version", action="version", version=f"doe-advisor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("presets", help="list the available domain presets")
    sub.add_parser("capabilities", help="what this installation can do (model orders, goals, weights)")

    opts = sub.add_parser("options", help="the top scored design options (Cheaper / Recommended / More thorough)")
    _add_spec_arg(opts); opts.add_argument("--limit", type=int, default=3, help="how many options (default 3)")
    opts.add_argument("--seed", type=int, default=0, help="scoring seed (default 0)")

    cand = sub.add_parser("candidates", help="every candidate design considered, scored, including rejects")
    _add_spec_arg(cand); cand.add_argument("--seed", type=int, default=0)

    props = sub.add_parser("properties", help="the full property report for one design")
    _add_spec_arg(props)
    props.add_argument("--design", help="design name, role, or 1-based rank (default: the recommended one)")
    props.add_argument("--seed", type=int, default=0)

    sheet = sub.add_parser("runsheet", help="write the randomised run sheet as a CSV ready for stat_board")
    _add_spec_arg(sheet)
    sheet.add_argument("--option", help="design name, role, or 1-based rank (default: the recommended one)")
    sheet.add_argument("--out", required=True, help="output CSV path")
    sheet.add_argument("--seed", type=int, default=0, help="randomisation seed (default 0)")

    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    cmd = args.command

    if cmd == "presets":
        return {"analysis": "presets", "presets": list_presets()}
    if cmd == "capabilities":
        return {
            "analysis": "capabilities",
            "version": __version__,
            "llm": llm_available(),
            "model_orders": [{"value": o.value, "label": o.label} for o in ModelOrder],
            "response_goals": [{"value": g.value, "label": g.label} for g in ResponseGoal],
            "axis_weights": AXIS_WEIGHTS,
            "min_power": DEFAULT_MIN_POWER,
        }

    spec = _load_spec(args.spec)
    seed = getattr(args, "seed", 0)

    if cmd == "options":
        options = top_options(spec, limit=args.limit, seed=seed)
        curve = power_curve(spec, up_to=curve_extent(options)) if options else None
        response = spec.primary_response
        return {
            "analysis": "options",
            "n_factors": spec.n_factors,
            "options": [serialise_option(o) for o in options],
            "power_basis": None if response is None else {
                "name": response.name,
                "units": response.units,
                "goal": response.goal.value,
                "goal_statement": response.goal_statement,
                "target_effect": response.target_effect,
                "noise_sd": response.noise_sd,
                "standardised_effect": response.standardised_effect,
                "runs_for_80_power": None if curve is None else curve.runs_for_80,
                "runs_for_80_power_if_noise_1_5x": None if curve is None else curve.runs_for_80_if_noise_1_5x,
            },
        }
    if cmd == "candidates":
        scored = score_candidates(spec, seed=seed)
        return {
            "analysis": "candidates",
            "n_candidates": len(scored),
            "candidates": [serialise_option(s) for s in scored if s.is_viable],
            "rejected": [{"name": s.design.name, "n_runs": s.n_runs, "reason": s.disqualified}
                         for s in scored if not s.is_viable],
        }
    if cmd == "properties":
        option = _pick_option(top_options(spec, limit=99, seed=seed), args.design)
        return {"analysis": "properties", **serialise_option(option)}
    if cmd == "runsheet":
        option = _pick_option(top_options(spec, limit=99, seed=seed), args.option)
        out = export.run_sheet_csv(option.design, spec, args.out, seed=seed)
        header, rows = export.run_sheet_rows(option.design, spec, seed=seed)
        return {
            "analysis": "runsheet",
            "design": option.design.name,
            "n_runs": len(rows),
            "path": str(out),
            "columns": header,
            "seed": seed,
            "next_step": export.analysis_hint(spec),
        }
    raise ValueError(f"unknown command: {cmd}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (IntakeError, ValueError, KeyError) as exc:
        json.dump({"error": type(exc).__name__, "message": str(exc)}, sys.stderr)
        sys.stderr.write("\n")
        return 2
    json.dump(result, sys.stdout, indent=2, default=float)
    sys.stdout.write("\n")
    return 0
