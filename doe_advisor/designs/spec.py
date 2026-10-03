"""Core data model: what the scientist asked for, and what a design is.

Everything downstream (generation, scoring, narration, the memo) is built on
these types. They deliberately hold no statistics — scoring lives in
``properties.py`` so that the same ``Design`` can be re-scored under different
assumptions without being rebuilt.

Coding convention: every design matrix in this package is in **coded units**,
where -1 is a factor's low setting and +1 its high setting. Conversion to real
units happens only at the edges (``Factor.decode``, memo export).
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
    """An input the scientist can set, e.g. pH or feed rate."""

    name: str
    low: float
    high: float
    units: str = ""

    def __post_init__(self) -> None:
        if self.high == self.low:
            raise ValueError(f"factor {self.name!r}: high and low must differ")

    @property
    def center(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2.0

    def decode(self, coded: float | np.ndarray) -> float | np.ndarray:
        """Coded (-1..+1) -> real units."""
        return self.center + np.asarray(coded) * self.half_range

    def encode(self, real: float | np.ndarray) -> float | np.ndarray:
        """Real units -> coded (-1..+1)."""
        return (np.asarray(real) - self.center) / self.half_range


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
    goal: ResponseGoal = ResponseGoal.SCREEN
    target_value: float | None = None  # only meaningful when goal is TARGET

    @property
    def standardised_effect(self) -> float | None:
        """Target effect expressed in standard deviations, or None if unknown."""
        if self.target_effect is None or self.noise_sd in (None, 0):
            return None
        return abs(self.target_effect) / abs(self.noise_sd)  # type: ignore[arg-type]

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
    # True when the declared ranges are limits, not just the region of
    # interest — e.g. a harvest window the process cannot go outside. Designs
    # with runs outside them (rotatable axial points) are then disqualified
    # rather than merely flagged, because no scoring axis can see a fixed
    # operating window.
    hard_ranges: bool = False

    def __post_init__(self) -> None:
        if len(self.factors) < 2:
            raise ValueError("a design needs at least 2 factors")
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate factor names: {names}")

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    @property
    def primary_response(self) -> Response | None:
        return self.responses[0] if self.responses else None


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

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError("design matrix must be 2-dimensional")
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
    def n_center_points(self) -> int:
        return int(np.sum(np.all(np.isclose(self.matrix, 0.0), axis=1)))

    def decoded(self, factors: list[Factor]) -> np.ndarray:
        """Run matrix in real units, for the run sheet in the memo."""
        if len(factors) != self.n_factors:
            raise ValueError("factor list does not match design width")
        out = np.empty_like(self.matrix)
        for j, factor in enumerate(factors):
            out[:, j] = factor.decode(self.matrix[:, j])
        return out

    def randomised_order(self, seed: int | None = None) -> np.ndarray:
        """A run order to execute in. Randomisation is not cosmetic — it is what
        stops a drifting bioreactor or a warming incubator from masquerading as
        a factor effect."""
        rng = np.random.default_rng(seed)
        return rng.permutation(self.n_runs)
