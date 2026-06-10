"""Fitting algorithms and utilities.

This module provides:

- Linear least squares (LLS)
- Weighted linear least squares (WLLS)
- Iterative weighted linear least squares (IWLLS)
- Least squares fitting wrapper with sensible defaults
- Covariance estimation and standard errors
- Bootstrap uncertainty estimation (future)

Example:
    ```python
    import numpy as np
    from qmri.fitting import least_squares

    def residual_func(params, x, y):
        return y - params[0] * np.exp(-params[1] * x)

    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([1.0, 0.6, 0.4, 0.2, 0.15])
    result = least_squares.fit(residual_func, x0=[1.0, 0.5], args=(x_data, y_data))
    print(f"Success: {result.success}")
    ```
"""

from qmri.fitting.least_squares import (
    FitResult,
    estimate_covariance,
    fit,
    standard_errors,
)

__all__ = [
    "FitResult",
    "estimate_covariance",
    "fit",
    "standard_errors",
]
