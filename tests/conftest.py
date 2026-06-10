"""Pytest configuration and fixtures for qmri tests."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def typical_b_values() -> np.ndarray:
    """Typical clinical b-values for DWI."""
    return np.array([0, 50, 100, 200, 400, 600, 800, 1000], dtype=np.float64)


@pytest.fixture
def simple_b_values() -> np.ndarray:
    """Return simple b-values for basic tests."""
    return np.array([0, 500, 1000, 2000], dtype=np.float64)
