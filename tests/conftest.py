"""Shared fixtures for the whole test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = REPO_ROOT / "sample_data"


@pytest.fixture
def sample_data() -> Path:
    return SAMPLE_DATA


@pytest.fixture
def long_csv() -> Path:
    return SAMPLE_DATA / "long.csv"


@pytest.fixture
def wide_csv() -> Path:
    return SAMPLE_DATA / "wide.csv"


@pytest.fixture
def two_json() -> Path:
    return SAMPLE_DATA / "two.json"


@pytest.fixture
def reg_csv() -> Path:
    return SAMPLE_DATA / "reg.csv"


@pytest.fixture
def multifactor_csv() -> Path:
    return SAMPLE_DATA / "multifactor.csv"


@pytest.fixture
def doe_dataset_csv() -> Path:
    """A 4-factor bioreactor DoE: run_order, run_type, paired *_coded/natural
    columns, and a titre response. The canonical example of the shape
    `doe_advisor.export.run_sheet_csv` writes and `stat_board` analyses."""
    return SAMPLE_DATA / "DOE_sample_dataset.csv"
