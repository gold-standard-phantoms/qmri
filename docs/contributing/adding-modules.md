# Adding New Modules

This guide explains how to add new signal models, fitting algorithms, or analysis modules to qmri.

## Overview

qmri follows a consistent pattern for all quantitative MRI modules:

1. **Signal generation function** — Forward model (physics → signal)
2. **Fitting function** — Inverse model (signal → parameters)
3. **Result dataclass** — Structured output with uncertainty
4. **Tests** — Perfect data, noisy data, edge cases
5. **Documentation** — Equations, API reference, user guide

## Step-by-Step: Adding a New Signal Model

We'll use T2* mapping as an example.

### 1. Create the Module

Create a new file in the appropriate domain:

```python
# packages/qmri/src/qmri/relaxometry/t2_star.py
"""T2* relaxation mapping.

This module provides signal generation and fitting functions for T2*
relaxometry using multi-echo gradient echo acquisitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize

from qmri._utils.safe_maths import safe_divide

__all__ = ["T2StarResult", "signal", "fit"]
```

### 2. Define the Result Dataclass

```python
@dataclass(frozen=True)
class T2StarResult:
    """Result of T2* fitting.

    Attributes:
        t2_star: T2* relaxation time in seconds.
        s0: Signal amplitude at TE=0.
        r_squared: Coefficient of determination (0-1).
        residuals: Fit residuals, if requested.
    """

    t2_star: npt.NDArray[np.floating]
    s0: npt.NDArray[np.floating]
    r_squared: npt.NDArray[np.floating]
    residuals: npt.NDArray[np.floating] | None = None
```

### 3. Implement the Signal Function

```python
def signal(
    s0: npt.NDArray[np.floating],
    t2_star: npt.NDArray[np.floating],
    echo_times: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Generate T2*-weighted signal.

    Implements the mono-exponential T2* decay model:

    $$S(TE) = S_0 \exp\left(-\frac{TE}{T_2^*}\right)$$

    Args:
        s0: Signal amplitude at TE=0. Shape ``(...)``.
        t2_star: T2* relaxation time in seconds. Shape ``(...)``.
        echo_times: Echo times in seconds. Shape ``(n_echoes,)``.

    Returns:
        Signal at each echo time. Shape ``(..., n_echoes)``.

    Example:
        >>> s0 = np.array([1000.0])
        >>> t2_star = np.array([0.030])  # 30 ms
        >>> te = np.array([0.005, 0.010, 0.020, 0.040])
        >>> sig = signal(s0, t2_star, te)
        >>> print(sig.round(1))
        [[846.5 716.5 513.4 263.6]]
    """
    # Broadcast parameters with echo times
    s0 = np.asarray(s0)[..., np.newaxis]
    t2_star = np.asarray(t2_star)[..., np.newaxis]
    echo_times = np.asarray(echo_times)

    return s0 * np.exp(-echo_times / t2_star)
```

### 4. Implement the Fitting Function

```python
def fit(
    signal_data: npt.NDArray[np.floating],
    echo_times: npt.NDArray[np.floating],
    *,
    method: Literal["lls", "nlls"] = "lls",
    mask: npt.NDArray[np.bool_] | None = None,
) -> T2StarResult:
    r"""Fit T2* model to multi-echo data.

    Fits the mono-exponential model:

    $$S(TE) = S_0 \exp\left(-\frac{TE}{T_2^*}\right)$$

    Args:
        signal_data: Multi-echo signal. Shape ``(..., n_echoes)``.
        echo_times: Echo times in seconds. Shape ``(n_echoes,)``.
        method: Fitting method:

            - ``"lls"``: Log-linear least squares (fast, biased at low SNR)
            - ``"nlls"``: Non-linear least squares (slower, unbiased)

        mask: Optional boolean mask. Shape ``(...)``.

    Returns:
        Fitted T2*, S0, and R² values.

    Raises:
        ValueError: If signal and echo_times have incompatible shapes.

    Example:
        >>> te = np.array([0.005, 0.010, 0.020, 0.040])
        >>> sig = np.array([847, 717, 513, 264])
        >>> result = fit(sig, te)
        >>> print(f"T2*: {result.t2_star * 1000:.1f} ms")
        T2*: 30.0 ms
    """
    signal_data = np.asarray(signal_data)
    echo_times = np.asarray(echo_times)

    # Validate shapes
    if signal_data.shape[-1] != len(echo_times):
        msg = (
            f"Signal shape {signal_data.shape} incompatible with "
            f"{len(echo_times)} echo times"
        )
        raise ValueError(msg)

    if method == "lls":
        return _fit_lls(signal_data, echo_times, mask)
    elif method == "nlls":
        return _fit_nlls(signal_data, echo_times, mask)
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)


def _fit_lls(
    signal_data: npt.NDArray[np.floating],
    echo_times: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_] | None,
) -> T2StarResult:
    """Linear least squares T2* fitting."""
    # Log-transform for linear fitting
    log_signal = np.log(np.maximum(signal_data, 1e-10))

    # Design matrix: [1, -TE]
    X = np.column_stack([np.ones_like(echo_times), -echo_times])

    # Solve via normal equations
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = log_signal @ X @ XtX_inv.T

    # Extract parameters
    log_s0 = beta[..., 0]
    r2_star = beta[..., 1]  # R2* = 1/T2*

    s0 = np.exp(log_s0)
    t2_star = safe_divide(1.0, r2_star, fill=np.inf)

    # Calculate R²
    predicted = s0[..., np.newaxis] * np.exp(-echo_times * r2_star[..., np.newaxis])
    ss_res = np.sum((signal_data - predicted) ** 2, axis=-1)
    ss_tot = np.sum((signal_data - signal_data.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    r_squared = 1 - safe_divide(ss_res, ss_tot, fill=0.0)

    # Apply mask
    if mask is not None:
        t2_star = np.where(mask, t2_star, np.nan)
        s0 = np.where(mask, s0, np.nan)
        r_squared = np.where(mask, r_squared, np.nan)

    return T2StarResult(t2_star=t2_star, s0=s0, r_squared=r_squared)
```

### 5. Export from Package

Add to `__init__.py`:

```python
# packages/qmri/src/qmri/relaxometry/__init__.py
"""Relaxometry models for T1, T2, and T2* mapping."""

from qmri.relaxometry import t1, t2, t2_star

__all__ = ["t1", "t2", "t2_star"]
```

### 6. Write Tests

Create tests in `tests/test_relaxometry/test_t2_star.py`:

```python
"""Tests for T2* relaxometry."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from qmri.relaxometry import t2_star


class TestSignal:
    """Tests for T2* signal generation."""

    def test_signal_shape(self):
        """Signal has correct output shape."""
        s0 = np.ones((5, 5))
        t2s = np.full((5, 5), 0.030)
        te = np.array([0.005, 0.010, 0.020])

        result = t2_star.signal(s0, t2s, te)

        assert result.shape == (5, 5, 3)

    def test_signal_decays_with_te(self):
        """Signal decreases with increasing echo time."""
        sig = t2_star.signal(s0=1000, t2_star=0.030, echo_times=np.array([0.01, 0.02, 0.03]))

        assert np.all(np.diff(sig) < 0)


class TestFit:
    """Tests for T2* fitting."""

    def test_fit_recovers_true_t2_star(self):
        """Fitting recovers true T2* from perfect data."""
        true_t2_star = 0.030  # 30 ms
        te = np.array([0.005, 0.010, 0.020, 0.040])
        sig = t2_star.signal(1000.0, true_t2_star, te).flatten()

        result = t2_star.fit(sig, te)

        np.testing.assert_allclose(result.t2_star, true_t2_star, rtol=1e-6)

    def test_fit_robust_to_noise(self, rng):
        """Fitting is accurate with realistic noise."""
        true_t2_star = 0.030
        te = np.array([0.005, 0.010, 0.020, 0.040])
        sig = t2_star.signal(1000.0, true_t2_star, te).flatten()
        sig += rng.normal(0, 20, sig.shape)

        result = t2_star.fit(sig, te)

        np.testing.assert_allclose(result.t2_star, true_t2_star, rtol=0.1)

    def test_fit_raises_on_shape_mismatch(self):
        """Fitting raises for incompatible shapes."""
        sig = np.array([100, 80, 60])
        te = np.array([0.01, 0.02])  # Wrong length

        with pytest.raises(ValueError, match="incompatible"):
            t2_star.fit(sig, te)
```

### 7. Add Documentation

#### Equations Page

Add to `docs/equations/relaxometry.md`:

```markdown
## T2* Mapping

T2* relaxation describes signal decay due to both spin-spin relaxation
and magnetic field inhomogeneities:

$$S(TE) = S_0 \exp\left(-\frac{TE}{T_2^*}\right)$$

where:

- $S(TE)$ is the signal at echo time $TE$
- $S_0$ is the signal at $TE = 0$
- $T_2^*$ is the effective transverse relaxation time
```

#### User Guide

Add to `docs/user-guide/relaxometry.md`:

```markdown
## T2* Mapping

T2* mapping uses multi-echo gradient echo acquisitions:

​```python
from qmri.relaxometry import t2_star

# Multi-echo GRE data
echo_times = np.array([0.005, 0.010, 0.020, 0.040])  # seconds
signal = np.array([847, 717, 513, 264])

result = t2_star.fit(signal, echo_times)
print(f"T2*: {result.t2_star * 1000:.1f} ms")
​```
```

#### API Reference

Add to `docs/api/relaxometry.md`:

```markdown
## T2* Mapping

::: qmri.relaxometry.t2_star
    options:
      show_root_heading: true
      members:
        - T2StarResult
        - signal
        - fit
```

## Checklist

Before submitting your new module:

- [ ] Signal function with LaTeX docstring
- [ ] Fitting function with method options
- [ ] Frozen result dataclass
- [ ] Type hints on all public functions
- [ ] Unit tests (perfect data, noisy data, edge cases)
- [ ] Property-based tests if applicable
- [ ] Equations documentation
- [ ] User guide section
- [ ] API reference entry
- [ ] Exports added to `__init__.py`
- [ ] All checks pass (`ruff`, `mypy`, `pytest`)
