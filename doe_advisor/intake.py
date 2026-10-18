"""Turning what the scientist typed into a validated :class:`DesignSpec`.

The structured form is the source of truth. The chat box on top of it is a
typing shortcut: it asks Claude to *populate the form*, which the scientist then
sees and corrects. That ordering is deliberate — an intake that goes straight
from free text to a computation is unauditable, and it stops working entirely
on a laptop with no API key.

So: :func:`spec_from_dict` is the real entry point and never needs Claude.
:func:`extract_form_from_text` is optional sugar that produces the same dict.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .designs.spec import (
    DesignSpec,
    DisallowedCombination,
    Factor,
    LinearConstraint,
    ModelOrder,
    Response,
    ResponseGoal,
)

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"

MAX_FACTORS = 10
MAX_LEVELS = 12
MAX_RUNS_CEILING = 10_000


class IntakeError(ValueError):
    """A problem with what the scientist supplied, phrased for the scientist."""


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------


def list_presets(preset_dir: Path | None = None) -> list[dict]:
    """Available domain presets, blank template first."""
    directory = preset_dir or PRESET_DIR
    presets = []
    for path in sorted(directory.glob("*.json")):
        try:
            presets.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue  # a malformed preset must not take the whole app down
    # Blank template first; then by the preset's own "order" key, then label.
    presets.sort(key=lambda p: (p.get("id") != "blank", p.get("order", 100), p.get("label", "")))
    return presets


def load_preset(preset_id: str, preset_dir: Path | None = None) -> dict:
    for preset in list_presets(preset_dir):
        if preset.get("id") == preset_id:
            return preset
    raise IntakeError(f"no preset called {preset_id!r}")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _number(value, field: str, allow_none: bool = False) -> float | None:
    if value is None or value == "":
        if allow_none:
            return None
        raise IntakeError(f"{field} is required")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise IntakeError(f"{field} must be a number, got {value!r}") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise IntakeError(f"{field} must be a finite number")
    return parsed


def _blocks(value) -> int:
    blocks = _number(value, "number of blocks", allow_none=True)
    if blocks is None:
        return 1
    if blocks < 1 or blocks != int(blocks):
        raise IntakeError("The number of blocks must be a whole number, 1 or more (1 = no blocking).")
    if blocks > 8:
        raise IntakeError("At most 8 blocks are supported.")
    return int(blocks)


def cv_to_log10_sd(cv_pct: float) -> float:
    """SD on the log10 scale of a lognormal response with this CV.

    sigma_ln = sqrt(ln(1 + CV^2)), exactly, for a lognormal; divide by ln 10 for
    log10. For small CVs this is close to CV / 2.303.
    """
    return math.sqrt(math.log1p((cv_pct / 100.0) ** 2)) / math.log(10.0)


def cv_to_log10_effect(fold: float) -> float:
    """A fold change as a log10 difference; a fall (fold < 1) is the same size as its inverse rise."""
    return abs(math.log10(fold))


def spec_from_dict(payload: dict) -> DesignSpec:
    """Validate a form payload into a :class:`DesignSpec`.

    Errors are written for the person who filled the form in, not for a
    developer reading a stack trace.
    """
    if not isinstance(payload, dict):
        raise IntakeError("expected a form object")

    raw_factors = [f for f in payload.get("factors") or [] if (f or {}).get("name", "").strip()]
    if len(raw_factors) < 2:
        raise IntakeError("Give at least two factors — a design needs something to vary.")
    if len(raw_factors) > MAX_FACTORS:
        raise IntakeError(f"At most {MAX_FACTORS} factors are supported.")

    factors = []
    for i, item in enumerate(raw_factors):
        name = str(item.get("name", "")).strip()
        # A factor given levels but no type is categorical: there is nothing else it could be.
        default_kind = "categorical" if item.get("levels") and item.get("low") in (None, "") else "continuous"
        kind = str(item.get("type") or default_kind).strip().lower()
        if kind not in ("continuous", "categorical"):
            raise IntakeError(f"{name}: factor type must be 'continuous' or 'categorical', got {kind!r}.")
        if kind == "categorical":
            factors.append(_categorical_factor(name, item.get("levels"), str(item.get("units") or "").strip()))
            continue
        low = _number(item.get("low"), f"low value for {name or f'factor {i + 1}'}")
        high = _number(item.get("high"), f"high value for {name or f'factor {i + 1}'}")
        if low == high:
            raise IntakeError(f"{name}: the low and high values are the same, so it is not a factor.")
        if low > high:
            low, high = high, low  # a transposed range is a typo, not an error
        factors.append(Factor(name=name, low=low, high=high, units=str(item.get("units") or "").strip()))

    names = [f.name for f in factors]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise IntakeError(f"Factor names must be unique; repeated: {', '.join(sorted(duplicates))}.")
    constraints = _constraints(payload.get("constraints"), factors)
    disallowed = _disallowed(payload.get("disallowed"), factors)
    text_constraints, text_disallowed = parse_constraint_text(payload.get("constraints_text") or "", factors)
    constraints += text_constraints
    disallowed += text_disallowed

    responses = []
    for item in payload.get("responses") or []:
        name = str((item or {}).get("name", "")).strip()
        if not name:
            continue
        noise_model = str(item.get("noise_model") or "sd").strip().lower()
        if noise_model not in ("sd", "cv"):
            raise IntakeError(f"{name}: noise_model must be 'sd' or 'cv'.")
        cv_pct = fold = None
        if noise_model == "cv":
            # Noise that grows with the level (titres, cell counts) is stated
            # as a CV and the effect as a fold change; on a log scale both
            # become constants, which is what the power maths needs.
            cv_pct = _number(item.get("noise_cv_pct"), f"noise CV % for {name}", allow_none=True)
            fold = _number(item.get("target_fold"), f"target fold change for {name}", allow_none=True)
            if cv_pct is not None and cv_pct <= 0:
                raise IntakeError(f"{name}: the CV must be greater than zero percent.")
            if fold is not None and fold <= 0:
                raise IntakeError(f"{name}: a fold change must be greater than zero (e.g. 1.5 for +50%).")
            if fold is not None and fold == 1:
                raise IntakeError(f"{name}: a fold change of 1 is no change at all.")
            target = None if fold is None else cv_to_log10_effect(fold)
            noise = None if cv_pct is None else cv_to_log10_sd(cv_pct)
        else:
            target = _number(item.get("target_effect"), f"target effect for {name}", allow_none=True)
            noise = _number(item.get("noise_sd"), f"noise SD for {name}", allow_none=True)
        if noise is not None and noise <= 0:
            raise IntakeError(f"{name}: the run-to-run standard deviation must be greater than zero.")
        noise_df = _number(item.get("noise_df"), f"noise degrees of freedom for {name}", allow_none=True)
        if noise_df is not None and noise_df < 1:
            raise IntakeError(f"{name}: the noise degrees of freedom must be at least 1.")
        goal_value = str(item.get("goal") or "screen").strip().lower()
        try:
            goal = ResponseGoal(goal_value)
        except ValueError as exc:
            allowed = ", ".join(g.value for g in ResponseGoal)
            raise IntakeError(f"{name}: unknown goal {goal_value!r}. Choose one of: {allowed}.") from exc
        target_value = _number(item.get("target_value"), f"target value for {name}", allow_none=True)
        responses.append(
            Response(
                name=name,
                units=str(item.get("units") or "").strip(),
                target_effect=target,
                noise_sd=noise,
                noise_df=None if noise_df is None or noise is None else int(noise_df),
                goal=goal,
                target_value=target_value if goal is ResponseGoal.TARGET else None,
                scale="log10" if noise_model == "cv" else "raw",
                noise_cv_pct=cv_pct,
                target_fold=fold,
            )
        )

    order_value = str(payload.get("model_order") or "interaction").strip().lower()
    try:
        model_order = ModelOrder(order_value)
    except ValueError as exc:
        allowed = ", ".join(o.value for o in ModelOrder)
        raise IntakeError(f"Unknown model type {order_value!r}. Choose one of: {allowed}.") from exc

    max_runs = _number(payload.get("max_runs"), "run budget", allow_none=True)
    if max_runs is not None:
        # One run per main-effect coefficient plus the intercept: a k-level
        # categorical factor needs k-1 of them.
        floor = 1 + sum(f.n_levels - 1 if f.is_categorical else 1 for f in factors)
        if max_runs < floor:
            raise IntakeError(
                f"A budget of {int(max_runs)} runs cannot support {len(factors)} factors. "
                f"You need at least {floor}."
            )
        if max_runs > MAX_RUNS_CEILING:
            raise IntakeError("That run budget is implausibly large; check the units.")

    centre = _number(payload.get("n_center_points"), "centre points", allow_none=True)
    losses = _number(payload.get("expected_run_losses"), "expected run losses", allow_none=True)

    return DesignSpec(
        factors=factors,
        responses=responses,
        model_order=model_order,
        max_runs=None if max_runs is None else int(max_runs),
        n_center_points=3 if centre is None else max(int(centre), 0),
        expected_run_losses=1 if losses is None else max(int(losses), 0),
        hard_ranges=bool(payload.get("hard_ranges", False)),
        n_blocks=_blocks(payload.get("n_blocks")),
        constraints=constraints,
        disallowed=disallowed,
    )


def _categorical_factor(name: str, raw_levels, units: str) -> Factor:
    """A categorical factor from a list of level names, or a comma-separated string."""
    if isinstance(raw_levels, str):
        raw_levels = raw_levels.split(",")
    levels = [str(v).strip() for v in (raw_levels or []) if str(v).strip()]
    if len(levels) < 2:
        raise IntakeError(f"{name}: a categorical factor needs at least two levels, e.g. 'media A, media B'.")
    if len(levels) > MAX_LEVELS:
        raise IntakeError(f"{name}: at most {MAX_LEVELS} levels are supported.")
    repeated = sorted({v for v in levels if levels.count(v) > 1})
    if repeated:
        raise IntakeError(f"{name}: level names must be unique; repeated: {', '.join(repeated)}.")
    return Factor(name=name, levels=tuple(levels), units=units)


def _constraints(raw, factors: list[Factor], line_label: str | None = None) -> list[LinearConstraint]:
    """Linear constraints: ``{"coefficients": {"feed": 1, "bolus": 1}, "op": "<=", "bound": 8}``."""
    by_name = {f.name: f for f in factors}
    out = []
    for i, item in enumerate(raw or []):
        label = line_label or f"constraint {i + 1}"
        coefficients = (item or {}).get("coefficients") or {}
        if not isinstance(coefficients, dict) or not coefficients:
            raise IntakeError(f"{label}: give the factors and their weights, e.g. {{'feed': 1, 'bolus': 1}}.")
        parsed = {}
        for name, weight in coefficients.items():
            factor = by_name.get(str(name).strip())
            if factor is None:
                raise IntakeError(f"{label}: there is no factor called {name!r}.")
            if factor.is_categorical:
                raise IntakeError(
                    f"{label}: {name} is categorical; rule out a categorical setting with a disallowed "
                    "combination instead."
                )
            w = _number(weight, f"{label}: weight for {name}")
            if w != 0:
                parsed[factor.name] = w
        if not parsed:
            raise IntakeError(f"{label}: every weight is zero, so it limits nothing.")
        op = str(item.get("op") or "<=").strip()
        if op not in ("<=", ">="):
            raise IntakeError(f"{label}: the operator must be '<=' or '>='.")
        bound = _number(item.get("bound"), f"{label}: bound")
        out.append(LinearConstraint(coefficients=parsed, bound=bound, op=op))
    return out


def _disallowed(raw, factors: list[Factor], line_label: str | None = None) -> list[DisallowedCombination]:
    """Disallowed combinations: ``{"media": ["A"], "cell line": ["L2", "L3"]}`` per entry."""
    by_name = {f.name: f for f in factors}
    out = []
    for i, item in enumerate(raw or []):
        label = line_label or f"disallowed combination {i + 1}"
        if not isinstance(item, dict) or not item:
            raise IntakeError(f"{label}: give each categorical factor and the levels that cannot be run together.")
        parsed = {}
        for name, values in item.items():
            factor = by_name.get(str(name).strip())
            if factor is None:
                raise IntakeError(f"{label}: there is no factor called {name!r}.")
            if not factor.is_categorical:
                raise IntakeError(f"{label}: {name} is continuous; limit it with a linear constraint instead.")
            values = [values] if isinstance(values, str) else list(values or [])
            values = [str(v).strip() for v in values if str(v).strip()]
            unknown = [v for v in values if v not in factor.levels]
            if unknown or not values:
                raise IntakeError(
                    f"{label}: {name} has levels {', '.join(factor.levels)}; got {', '.join(unknown) or 'none'}."
                )
            parsed[factor.name] = tuple(values)
        if len(parsed) == 1:
            ((name, values),) = parsed.items()
            raise IntakeError(
                f"{label}: ruling out {name} = {' or '.join(values)} on its own means it is never run; "
                f"remove it from {name}'s levels instead."
            )
        out.append(DisallowedCombination(levels=parsed))
    return out


_LINEAR_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<op><=|>=|=<|=>|≤|≥)\s*(?P<bound>[-+]?[0-9.]+(?:[eE][-+]?\d+)?)$")
_COEF_RE = re.compile(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\*?\s*")


def _linear_terms(lhs: str, factors: list[Factor], label: str) -> dict[str, float]:
    """``feed + 2*bolus - 0.5 temp`` -> ``{"feed": 1, "bolus": 2, "temp": -0.5}``.

    Scanned left to right, matching the *longest* factor name at each step, so
    names may contain spaces, hyphens or digits ("glucose-6 feed") without
    being split.
    """
    names = sorted((f.name for f in factors if not f.is_categorical), key=len, reverse=True)
    text, pos, out = lhs.strip(), 0, {}
    first = True
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        sign = 1.0
        if pos < len(text) and text[pos] in "+-":
            sign = -1.0 if text[pos] == "-" else 1.0
            pos += 1
        elif not first:
            raise IntakeError(f"{label}: expected + or - before {text[pos:]!r}.")
        coef = 1.0
        number = _COEF_RE.match(text, pos)
        if number and not any(text.lower().startswith(n.lower(), pos) for n in names):
            coef, pos = float(number.group(1)), number.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        name = next((n for n in names if text.lower().startswith(n.lower(), pos)), None)
        if name is None:
            raise IntakeError(f"{label}: no continuous factor name at {text[pos:]!r}.")
        out[name] = out.get(name, 0.0) + sign * coef
        pos += len(name)
        first = False
    if not out:
        raise IntakeError(f"{label}: no factors on the left-hand side.")
    return out


def _match_factor(text: str, factors: list[Factor]) -> Factor | None:
    """The factor whose name is ``text``, ignoring case and surrounding space."""
    wanted = text.strip().lower()
    return next((f for f in factors if f.name.lower() == wanted), None)


def parse_constraint_text(
    text: str, factors: list[Factor]
) -> tuple[list[LinearConstraint], list[DisallowedCombination]]:
    """Constraints typed one per line, as a scientist would write them.

    Two kinds of line::

        feed rate + 2*glucose bolus <= 8          (a linear constraint, natural units)
        never media = A or B with cell line = L2  (a disallowed combination)

    Factor names may contain spaces; ``and`` works as well as ``with``. Blank
    lines and lines starting with ``#`` are ignored. Errors name the line.
    """
    constraints: list[LinearConstraint] = []
    disallowed: list[DisallowedCombination] = []
    for number, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label = f"constraint line {number} ({line!r})"
        if line.lower().startswith("never "):
            parts = re.split(r"\s+(?:with|and)\s+", line[6:].strip(), flags=re.IGNORECASE)
            combo: dict[str, list[str]] = {}
            for part in parts:
                if "=" not in part:
                    raise IntakeError(f"{label}: write each setting as 'factor = level'.")
                name, _, values = part.partition("=")
                factor = _match_factor(name, factors)
                if factor is None:
                    raise IntakeError(f"{label}: there is no factor called {name.strip()!r}.")
                combo[factor.name] = [v.strip() for v in re.split(r"\s+or\s+|,", values) if v.strip()]
            disallowed += _disallowed([combo], factors, label)
            continue
        match = _LINEAR_RE.match(line)
        if not match:
            raise IntakeError(
                f"{label}: expected 'a + 2*b <= 10' or 'never factor = level with factor = level'."
            )
        coefficients = _linear_terms(match["lhs"], factors, label)
        op = {"=<": "<=", "≤": "<=", "=>": ">=", "≥": ">="}.get(match["op"], match["op"])
        constraints += _constraints(
            [{"coefficients": coefficients, "op": op, "bound": match["bound"]}], factors, label
        )
    return constraints, disallowed


def _factor_to_dict(f: Factor) -> dict:
    if f.is_categorical:
        return {"name": f.name, "type": "categorical", "levels": list(f.levels), "units": f.units}
    return {"name": f.name, "low": f.low, "high": f.high, "units": f.units}


def spec_to_dict(spec: DesignSpec) -> dict:
    """Inverse of :func:`spec_from_dict`, so a spec can round-trip to the form."""
    out = {
        "factors": [_factor_to_dict(f) for f in spec.factors],
        "responses": [
            {
                "name": r.name,
                "units": r.units,
                "goal": r.goal.value,
                "target_value": r.target_value,
                "target_effect": r.target_effect,
                "noise_sd": r.noise_sd,
                "noise_df": r.noise_df,
                **(
                    {"noise_model": "cv", "noise_cv_pct": r.noise_cv_pct, "target_fold": r.target_fold}
                    if r.is_log
                    else {}
                ),
            }
            for r in spec.responses
        ],
        "model_order": spec.model_order.value,
        "max_runs": spec.max_runs,
        "n_center_points": spec.n_center_points,
        "expected_run_losses": spec.expected_run_losses,
        "hard_ranges": spec.hard_ranges,
        "n_blocks": spec.n_blocks,
    }
    if spec.constraints:
        out["constraints"] = [
            {"coefficients": dict(c.coefficients), "op": c.op, "bound": c.bound} for c in spec.constraints
        ]
    if spec.disallowed:
        out["disallowed"] = [{name: list(values) for name, values in d.levels.items()} for d in spec.disallowed]
    return out


# --------------------------------------------------------------------------
# Optional: free text -> form fields
# --------------------------------------------------------------------------

EXTRACTION_PROMPT = """You convert a bench scientist's plain-English description of a planned experiment \
into the fields of a design-of-experiments intake form.

Return JSON only, in this shape:
{"factors": [{"name": "pH", "low": 6.8, "high": 7.4, "units": ""},
             {"name": "media", "type": "categorical", "levels": ["media A", "media B", "media C"]}],
 "responses": [{"name": "titer", "units": "g/L", "goal": "screen" | "maximize" | "minimize" | "target", \
"target_value": null, "target_effect": null, "noise_sd": null}],
 "model_order": "main" | "interaction" | "quadratic",
 "max_runs": null,
 "n_center_points": 3,
 "expected_run_losses": 1,
 "notes": ["anything you had to assume or could not find"]}

Rules:
- A factor with named options rather than a numeric range (media, cell line, resin, supplier, \
operator) is "type": "categorical" with its "levels" listed; it has no low or high.
- Extract only what the text actually says. Use null for anything not stated. Do NOT invent ranges, \
target effects, or noise estimates — a guessed range that looks plausible is worse than a blank field, \
because the scientist will not notice it.
- "screening" or "which factors matter" implies model_order "main". "interactions" implies "interaction". \
"optimise", "response surface", "curvature", "set ranges" implies "quadratic".
- Response "goal": "maximise"/"increase"/"best" -> "maximize"; "minimise"/"reduce"/"keep low" -> "minimize"; \
"hit"/"reach"/"hold at"/"target of X" -> "target" with target_value X; otherwise "screen". \
"target_effect" is the smallest change worth detecting, NOT the target value.
- Put every assumption you made in "notes"."""


def extract_form_from_text(
    text: str,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Ask Claude to populate the form from free text. Never invents values.

    Returns a dict shaped like the form, plus a ``notes`` list of assumptions.
    Raises :class:`IntakeError` when unavailable, so the caller can fall back to
    the plain form rather than failing the request.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise IntakeError("Conversational intake needs an ANTHROPIC_API_KEY. Fill the form in directly instead.")
    if not (text or "").strip():
        raise IntakeError("Describe your experiment first.")

    try:
        import anthropic
    except ImportError as exc:
        raise IntakeError("The anthropic package is not installed. Fill the form in directly instead.") from exc

    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=model or os.environ.get("DOE_ADVISOR_MODEL", "claude-sonnet-5"),
            max_tokens=1500,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        body = "".join(block.text for block in message.content if block.type == "text")
        parsed = json.loads(_extract_json(body))
    except Exception as exc:
        raise IntakeError(f"Could not read that description ({type(exc).__name__}). Try the form instead.") from exc

    if not isinstance(parsed, dict):
        raise IntakeError("Could not read that description. Try the form instead.")
    return parsed


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def llm_available() -> bool:
    """Whether the chat box should be shown at all."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True
