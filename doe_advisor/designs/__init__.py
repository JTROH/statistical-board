"""Design generation and scoring."""

from .alias import AliasReport, alias_report
from .classical import (
    box_behnken,
    central_composite,
    definitive_screening,
    fractional_factorial,
    full_factorial,
    resolution_of,
)
from .model import model_matrix, model_terms, potential_terms, term_label
from .properties import DesignProperties, evaluate
from .spec import Design, DesignSpec, Factor, ModelOrder, Response, ResponseGoal

__all__ = [
    "AliasReport",
    "Design",
    "DesignProperties",
    "DesignSpec",
    "Factor",
    "ModelOrder",
    "Response",
    "ResponseGoal",
    "alias_report",
    "box_behnken",
    "central_composite",
    "definitive_screening",
    "evaluate",
    "fractional_factorial",
    "full_factorial",
    "model_matrix",
    "model_terms",
    "potential_terms",
    "resolution_of",
    "term_label",
]
