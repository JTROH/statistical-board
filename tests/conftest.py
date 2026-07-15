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
