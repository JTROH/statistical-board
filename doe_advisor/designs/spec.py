"""Core data model: what the scientist asked for, and what a design is.

Everything downstream (generation, scoring, narration, the memo) is built on
these types. They deliberately hold no statistics — scoring lives in
``properties.py`` so that the same ``Design`` can be re-scored under different
assumptions without being rebuilt.

Coding convention: every design matrix in this package is in **coded units**,
where -1 is a factor's low setting and +1 its high setting. Conversion to real
units happens only at the edges (``Factor.decode``, memo export).

A categorical factor (media, cell line, resin) has no low or high, only named
levels. Its column in a design matrix holds the **level index** (0, 1, ...,
k-1); ``model.model_matrix`` expands it into k-1 effect-coded columns. Nothing
outside ``model.py`` should do arithmetic on a level index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class ModelOrder(StrEnum):
    """How ambitious a model the scientist wants to be able to fit.

    This is the single biggest lever on run count, which is why the explorer
    (Phase 2) exposes it as a slider.
    """

    MAIN = "main"  # main effects only
    INTERACTION = "interaction"  # main effects + two-factor interactions
    QUADRATIC = "quadratic"  # main effects + 2FI + pure quadratics

    @property
    def label(self) -> str:
        return {
            ModelOrder.MAIN: "main effects only",
            ModelOrder.INTERACTION: "main effects and two-factor interactions",
            ModelOrder.QUADRATIC: "main effects, two-factor interactions and curvature",
        }[self]


@dataclass(frozen=True)
class Factor:
    """An input the scientist can set, e.g. pH or feed rate.

    Continuous by default. Give ``levels`` (two or more names, e.g.
    ``("media A", "media B", "media C")``) for a categorical factor; ``low`` and
    ``high`` are then ignored and kept at 0 and k-1 so the level index decodes
    to itself.
    """

    name: str
    low: float = 0.0
    high: float = 1.0
    units: str = ""
    levels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.levels:
            levels = tuple(str(level).strip() for level in self.levels)
            if len(levels) < 2:
                raise ValueError(f"factor {self.name!r}: a categorical factor needs at least 2 levels")
            if any(not level for level in levels):
                raise ValueError(f"factor {self.name!r}: level names cannot be blank")
            if len(set(levels)) != len(levels):
                raise ValueError(f"factor {self.name!r}: level names must be unique")
            object.__setattr__(self, "levels", levels)
            object.__setattr__(self, "low", 0.0)
            object.__setattr__(self, "high", float(len(levels) - 1))
            return
        if self.high == self.low:
            raise ValueError(f"factor {self.name!r}: high and low must differ")

    @property
    def is_categorical(self) -> bool:
        return bool(self.levels)

    @property
    def n_levels(self) -> int:
        """Number of levels for a categorical factor; 0 for a continuous one."""
        return len(self.levels)

    @property
    def center(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2.0

    def decode(self, coded: float | np.ndarray) -> float | np.ndarray:
        """Coded (-1..+1) -> real units. A categorical level index is returned as is."""
        if self.is_categorical:
            return np.asarray(coded, dtype=float)
        return self.center + np.asarray(coded) * self.half_range

    def encode(self, real: float | np.ndarray) -> float | np.ndarray:
        """Real units -> coded (-1..+1). A categorical level index is returned as is."""
        if self.is_categorical:
            return np.asarray(real, dtype=float)
        return (np.asarray(real) - self.center) / self.half_range

    def setting(self, coded: float) -> str:
        """One run's setting as it goes on a run sheet: ``7.1`` or ``media B``."""
        if self.is_categorical:
            return self.levels[int(round(float(coded)))]
        return f"{float(self.decode(coded)):g}"

    def level_index(self, label: str) -> int:
        """Index of a named level; raises ``ValueError`` for an unknown one."""
        try:
            return self.levels.index(str(label).strip())
        except ValueError as exc:
            raise ValueError(f"factor {self.name!r} has no level {label!r}") from exc


@dataclass(frozen=True)
class LinearConstraint:
    """A limit on a weighted sum of continuous factors, in **natural units**.

    ``coefficients`` maps factor name to weight; the constraint is
    ``sum(weight * setting) <= bound`` (or ``>=``). Example: total feed plus
    glucose bolus at most 8 g/L is ``{"feed": 1, "bolus": 1} <= 8``.
    """

    coefficients: dict[str, float]
    bound: float
    op: str = "<="

    def __post_init__(self) -> None:
        if self.op not in ("<=", ">="):
            raise ValueError(f"constraint operator must be '<=' or '>=', got {self.op!r}")
        if not self.coefficients:
            raise ValueError("a constraint needs at least one factor")

    def describe(self) -> str:
        parts = []
        for name, weight in self.coefficients.items():
            sign = "-" if weight < 0 else "+"
            magnitude = abs(weight)
            term = name if magnitude == 1 else f"{magnitude:g}*{name}"
            parts.append(f"{sign} {term}")
        text = " ".join(parts)
        text = text[2:] if text.startswith("+ ") else "-" + text[2:]
        return f"{text} {self.op} {self.bound:g}"


@dataclass(frozen=True)
class DisallowedCombination:
    """Settings that cannot be run together, e.g. resin X with buffer B.

    ``levels`` maps each categorical factor name to the level names that, all
    together, make a run impossible. A run is disallowed when *every* listed
    factor is at one of its listed levels.
    """

    levels: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("a disallowed combination needs at least one factor")
        object.__setattr__(
            self, "levels", {name: tuple(str(v).strip() for v in values) for name, values in self.levels.items()}
        )

    def describe(self) -> str:
        return " with ".join(f"{name} = {' or '.join(values)}" for name, values in self.levels.items())


class ResponseGoal(StrEnum):
    """What the scientist wants to do with a response.

    The goal does not change the arithmetic — power and aliasing are the same
    whichever way you want the response to move — but it changes the advice.
    Finding a best setting or hitting a target needs curvature in the model;
    finding out which factors matter does not.
    """

    SCREEN = "screen"  # which factors move it?
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"  # bring it to a specific value

    @property
    def label(self) -> str:
        return {
            ResponseGoal.SCREEN: "find out which factors move it",
            ResponseGoal.MAXIMIZE: "make it as high as possible",
            ResponseGoal.MINIMIZE: "make it as low as possible",
            ResponseGoal.TARGET: "bring it to a target value",
        }[self]

    @property
    def wants_optimum(self) -> bool:
        """Goals that need the model to describe curvature, not just slopes."""
        return self is not ResponseGoal.SCREEN


@dataclass(frozen=True)
class Response:
    """A measured output, e.g. titer.

    ``target_effect`` and ``noise_sd`` are what make power computable. Both are
    in the response's own units; their ratio is the standardised effect size.

    ``goal`` says what the scientist wants to do with the response, and
    ``target_value`` is the value to hit when the goal is ``TARGET``.
    """

    name: str
    units: str = ""
    target_effect: float | None = None  # smallest change worth detecting
    noise_sd: float | None = None  # run-to-run standard deviation
    # Degrees of freedom behind ``noise_sd`` when it was measured (e.g. the
    # pure-error df of a previous study). None means it is a guess.
    noise_df: int | None = None
    goal: ResponseGoal = ResponseGoal.SCREEN
    target_value: float | None = None  # only meaningful when goal is TARGET
    # "raw", or "log10" when the scientist described the noise as a CV and the
    # target as a fold change. Then ``target_effect`` and ``noise_sd`` hold the
    # log10 equivalents (see ``intake``), the power maths runs unchanged on that
    # scale, and the response must be analysed as log10 afterwards. The
    # originals are kept for display.
    scale: str = "raw"
    noise_cv_pct: float | None = None
    target_fold: float | None = None

    @property
    def standardised_effect(self) -> float | None:
        """Target effect expressed in standard deviations, or None if unknown."""
        if self.target_effect is None or self.noise_sd in (None, 0):
            return None
        return abs(self.target_effect) / abs(self.noise_sd)  # type: ignore[arg-type]

    @property
    def is_log(self) -> bool:
        return self.scale == "log10"

    @property
    def display_units(self) -> str:
        """Units of ``target_effect`` / ``noise_sd`` as stored: log10 units on the log scale."""
        if self.is_log:
            return f"log10 {self.units}".strip()
        return self.units

    @property
    def goal_statement(self) -> str:
        """Plain sentence fragment, e.g. ``make titer as high as possible``."""
        unit = f" {self.units}" if self.units else ""
        if self.goal is ResponseGoal.TARGET and self.target_value is not None:
            return f"bring {self.name} to {self.target_value:g}{unit}"
        return self.goal.label.replace(" it", f" {self.name}", 1)


@dataclass
class DesignSpec:
    """The experiment as stated by the scientist. The input to design generation."""

    factors: list[Factor]
    responses: list[Response] = field(default_factory=list)
    model_order: ModelOrder = ModelOrder.INTERACTION
    max_runs: int | None = None  # hard budget; None = unconstrained
    n_center_points: int = 3
    alpha: float = 0.05
    # How many runs the scientist thinks could plausibly be lost (failed
    # bioreactors, contamination). Drives the robustness axis.
    expected_run_losses: int = 1
    # Runs split into this many blocks (days, bioreactor batches, virus
    # lots). Each block gets its own offset in the model, so a day-to-day
    # shift cannot pose as a factor effect. 1 means no blocking.
    n_blocks: int = 1
    # True when the declared ranges are limits, not just the region of
    # interest — e.g. a harvest window the process cannot go outside. Designs
    # with runs outside them (rotatable axial points) are then disqualified
    # rather than merely flagged, because no scoring axis can see a fixed
    # operating window.
    hard_ranges: bool = False
    # Limits on where runs may go. Classical designs ignore them, so a
    # classical design that breaks one is disqualified; the custom (optimal)
    # generator only ever proposes runs that satisfy them.
    constraints: list[LinearConstraint] = field(default_factory=list)
    disallowed: list[DisallowedCombination] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.factors) < 2:
            raise ValueError("a design needs at least 2 factors")
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate factor names: {names}")
        by_name = {f.name: f for f in self.factors}
        for constraint in self.constraints:
            for name in constraint.coefficients:
                if name not in by_name:
                    raise ValueError(f"constraint names an unknown factor {name!r}")
                if by_name[name].is_categorical:
                    raise ValueError(f"constraint on {name!r}: only continuous factors can be in a linear constraint")
        for combo in self.disallowed:
            if len(combo.levels) == 1:
                ((name, values),) = combo.levels.items()
                raise ValueError(
                    f"disallowed combination on {name!r} alone: a level that can never be run is not a level; "
                    f"remove {', '.join(values)} from the factor instead"
                )
            for name, values in combo.levels.items():
                if name not in by_name:
                    raise ValueError(f"disallowed combination names an unknown factor {name!r}")
                factor = by_name[name]
                if not factor.is_categorical:
                    raise ValueError(
                        f"disallowed combination on {name!r}: only categorical factors can be listed; "
                        "limit a continuous factor with a linear constraint instead"
                    )
                for value in values:
                    factor.level_index(value)

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    @property
    def levels(self) -> tuple[int, ...]:
        """Per factor: the number of levels if categorical, 0 if continuous."""
        return tuple(f.n_levels for f in self.factors)

    @property
    def has_categorical(self) -> bool:
        return any(f.is_categorical for f in self.factors)

    @property
    def is_constrained(self) -> bool:
        return bool(self.constraints or self.disallowed)

    @property
    def primary_response(self) -> Response | None:
        return self.responses[0] if self.responses else None

    def feasible(self, matrix: np.ndarray) -> np.ndarray:
        """Boolean mask: which coded runs satisfy every constraint.

        Continuous columns are coded (-1..+1) and decoded here; categorical
        columns hold level indices. A tiny tolerance keeps a run sitting exactly
        on a constraint boundary feasible despite rounding.
        """
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        ok = np.ones(matrix.shape[0], dtype=bool)
        index = {f.name: j for j, f in enumerate(self.factors)}
        for constraint in self.constraints:
            total = np.zeros(matrix.shape[0])
            for name, weight in constraint.coefficients.items():
                j = index[name]
                total += weight * np.asarray(self.factors[j].decode(matrix[:, j]), dtype=float)
            scale = 1e-9 * max(1.0, abs(constraint.bound))
            ok &= total <= constraint.bound + scale if constraint.op == "<=" else total >= constraint.bound - scale
        for combo in self.disallowed:
            hit = np.ones(matrix.shape[0], dtype=bool)
            for name, values in combo.levels.items():
                j = index[name]
                codes = [self.factors[j].level_index(v) for v in values]
                hit &= np.isin(np.rint(matrix[:, j]).astype(int), codes)
            ok &= ~hit
        return ok


@dataclass
class Design:
    """A concrete set of runs, in coded units.

    ``matrix`` is (n_runs, n_factors). ``family`` names the generator that built
    it; ``detail`` carries generator-specific facts worth showing the user
    (fraction, resolution, alpha, defining relation).
    """

    name: str
    family: str
    matrix: np.ndarray
    factor_names: list[str]
    detail: dict = field(default_factory=dict)
    # Block label (0..b-1) per run, or None for an unblocked design.
    blocks: np.ndarray | None = None
    # Per factor: number of levels if categorical, 0 if continuous. None means
    # every factor is continuous, which is what every classical generator builds.
    levels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError("design matrix must be 2-dimensional")
        if self.levels is not None:
            self.levels = tuple(int(k) for k in self.levels)
            if len(self.levels) != self.matrix.shape[1]:
                raise ValueError(f"design {self.name!r}: one level count per factor is required")
            if not any(self.levels):
                self.levels = None
        if self.blocks is not None:
            self.blocks = np.asarray(self.blocks, dtype=int)
            if self.blocks.shape != (self.matrix.shape[0],):
                raise ValueError(f"design {self.name!r}: one block label per run is required")
        if self.matrix.shape[1] != len(self.factor_names):
            raise ValueError(
                f"design {self.name!r}: matrix has {self.matrix.shape[1]} columns "
                f"but {len(self.factor_names)} factor names"
            )

    @property
    def n_runs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def factor_levels(self) -> tuple[int, ...]:
        """Per factor: number of levels if categorical, 0 if continuous."""
        return self.levels if self.levels is not None else (0,) * self.n_factors

    @property
    def continuous_columns(self) -> np.ndarray:
        """Indices of the continuous factors."""
        return np.array([j for j, k in enumerate(self.factor_levels) if k == 0], dtype=int)

    @property
    def center_mask(self) -> np.ndarray:
        """Runs with every *continuous* factor at its centre.

        A categorical factor has no centre, so with categorical factors a centre
        run is one per categorical setting. With no continuous factor at all,
        there are no centre runs."""
        cols = self.continuous_columns
        if cols.size == 0:
            return np.zeros(self.n_runs, dtype=bool)
        return np.all(np.isclose(self.matrix[:, cols], 0.0), axis=1)

    @property
    def n_center_points(self) -> int:
        return int(np.sum(self.center_mask))

    def decoded(self, factors: list[Factor]) -> np.ndarray:
        """Run matrix in real units, for the run sheet in the memo."""
        if len(factors) != self.n_factors:
            raise ValueError("factor list does not match design width")
        out = np.empty_like(self.matrix)
        for j, factor in enumerate(factors):
            out[:, j] = factor.decode(self.matrix[:, j])
        return out

    @property
    def n_blocks(self) -> int:
        return 1 if self.blocks is None else int(self.blocks.max()) + 1

    def randomised_order(self, seed: int | None = None) -> np.ndarray:
        """A run order to execute in. Randomisation is not cosmetic — it is what
        stops a drifting bioreactor or a warming incubator from masquerading as
        a factor effect.

        A blocked design is run block by block, randomised within each block:
        a block is a day or a batch, so its runs must be done together."""
        rng = np.random.default_rng(seed)
        if self.blocks is None:
            return rng.permutation(self.n_runs)
        return np.concatenate([rng.permutation(np.flatnonzero(self.blocks == b)) for b in range(self.n_blocks)])
