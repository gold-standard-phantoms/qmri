# Testing Guide

qmri uses pytest for testing. Tests are located in the `tests/` directory.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_diffusion/test_adc.py

# Run tests matching a pattern
pytest -k "test_adc"

# Exclude slow tests
pytest -m "not slow"

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Test Structure

Tests mirror the package structure:

```
tests/
├── conftest.py              # Shared fixtures
├── test_diffusion/
│   ├── test_adc.py
│   ├── test_signal.py
│   └── test_calibration.py
├── test_relaxometry/
│   ├── test_t1.py
│   └── test_t2.py
├── test_perfusion/
│   └── test_asl.py
├── test_io/
│   └── test_nifti.py
└── test_cli/
    └── test_adc_commands.py
```

## Writing Tests

### Basic Test Structure

Use descriptive names and the Arrange-Act-Assert pattern:

```python
import numpy as np
import pytest
from qmri.diffusion import adc


def test_fit_single_voxel_returns_expected_adc():
    """Test ADC fitting for a single voxel with known values."""
    # Arrange
    signal = np.array([1000.0, 606.5, 367.9, 135.3])
    b_values = np.array([0, 500, 1000, 2000])
    expected_adc = 1.0e-3  # mm²/s

    # Act
    result = adc.fit(signal, b_values, method="iwlls")

    # Assert
    assert result.adc == pytest.approx(expected_adc, rel=0.01)
    assert result.r_squared > 0.99
```

### Test Categories

#### 1. Perfect Data Recovery

Test that fitting noiseless data recovers ground truth exactly:

```python
def test_fit_recovers_true_adc_from_perfect_data():
    """ADC fitting recovers true values from noiseless data."""
    true_adc = 1.0e-3
    true_s0 = 1000
    b_values = np.array([0, 500, 1000, 2000])
    signal = true_s0 * np.exp(-b_values * true_adc)

    result = adc.fit(signal, b_values, method="lls")

    np.testing.assert_allclose(result.adc, true_adc, rtol=1e-10)
    np.testing.assert_allclose(result.s0, true_s0, rtol=1e-10)
```

#### 2. Noisy Data Robustness

Test that fitting remains accurate with realistic noise:

```python
def test_fit_robust_to_realistic_noise():
    """ADC fitting is accurate with realistic noise levels."""
    rng = np.random.default_rng(42)
    true_adc = 1.0e-3
    b_values = np.array([0, 500, 1000, 2000])
    signal = 1000 * np.exp(-b_values * true_adc)
    signal += rng.normal(0, 20, signal.shape)  # ~SNR 50

    result = adc.fit(signal, b_values, method="iwlls")

    np.testing.assert_allclose(result.adc, true_adc, rtol=0.1)
```

#### 3. Edge Cases

Test boundary conditions and error handling:

```python
def test_fit_raises_on_mismatched_shapes():
    """Fitting raises ValueError for incompatible array shapes."""
    signal = np.array([1000, 500, 250])
    b_values = np.array([0, 500])  # Wrong length

    with pytest.raises(ValueError, match="shape"):
        adc.fit(signal, b_values)


def test_fit_handles_zero_signal():
    """Fitting handles zero signal gracefully."""
    signal = np.array([0.0, 0.0, 0.0, 0.0])
    b_values = np.array([0, 500, 1000, 2000])

    result = adc.fit(signal, b_values)

    assert np.isnan(result.adc) or result.adc == 0
```

#### 4. Multi-Dimensional Data

Test that functions handle volumetric data correctly:

```python
def test_fit_handles_3d_volume():
    """ADC fitting works on 3D volumes."""
    shape = (10, 10, 5, 4)  # (X, Y, Z, B)
    b_values = np.array([0, 500, 1000, 2000])
    true_adc = 1.0e-3

    signal = 1000 * np.exp(-b_values * true_adc)
    signal = np.broadcast_to(signal, shape).copy()

    result = adc.fit(signal, b_values)

    assert result.adc.shape == (10, 10, 5)
    np.testing.assert_allclose(result.adc, true_adc, rtol=1e-6)
```

### Property-Based Testing

Use Hypothesis for edge case discovery:

```python
from hypothesis import given, strategies as st, assume
import hypothesis.extra.numpy as hnp


@given(
    true_adc=st.floats(0.1e-3, 3.0e-3),
    true_s0=st.floats(100, 10000),
)
def test_fit_recovers_true_values(true_adc, true_s0):
    """Property: fitting noiseless data recovers true parameters."""
    b_values = np.array([0, 500, 1000, 2000])
    signal = true_s0 * np.exp(-b_values * true_adc)

    result = adc.fit(signal, b_values, method="lls")

    np.testing.assert_allclose(result.adc, true_adc, rtol=1e-6)
    np.testing.assert_allclose(result.s0, true_s0, rtol=1e-6)
```

### Fixtures

Define shared fixtures in `conftest.py`:

```python
# tests/conftest.py
import pytest
import numpy as np


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def dwi_phantom(rng):
    """Synthetic DWI data with known ground truth."""
    b_values = np.array([0, 500, 1000, 2000])
    true_adc = 1.0e-3
    true_s0 = 1000
    signal = true_s0 * np.exp(-b_values * true_adc)
    signal += rng.normal(0, 20, signal.shape)

    return {
        "signal": signal,
        "b_values": b_values,
        "true_adc": true_adc,
        "true_s0": true_s0,
    }
```

### Markers

Use markers for test categorisation:

```python
@pytest.mark.slow
def test_bootstrap_uncertainty_estimation():
    """Bootstrap takes many iterations (slow)."""
    ...


@pytest.mark.integration
def test_full_pipeline_with_files(tmp_path):
    """End-to-end test with file I/O."""
    ...
```

Configure in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring external resources",
]
```

## Test Coverage

Check coverage with:

```bash
pytest --cov=packages/qmri/src --cov-report=html
```

View the report at `htmlcov/index.html`.

### Coverage Targets

- Core fitting functions: >95%
- Signal generation: >90%
- Utilities: >80%
- CLI: >70%

## Continuous Integration

Tests run automatically on:

- Every push to a branch
- Every pull request
- Nightly against latest dependencies

The CI matrix tests:

- Python 3.10, 3.11, 3.12
- Ubuntu, macOS, Windows
