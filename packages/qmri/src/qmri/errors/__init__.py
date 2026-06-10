"""Error metrics and uncertainty propagation.

This module provides:

- R-squared (coefficient of determination)
- RMSE (root mean square error)
- Normalised RMSE
- Residuals
- Uncertainty propagation utilities
- Covariance matrix handling

Example:
    ```python
    import numpy as np
    from qmri.errors import metrics

    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 1.9, 3.2, 3.8])
    print(f"R-squared: {metrics.r_squared(observed, predicted):.4f}")
    # R-squared: 0.9800
    ```
"""

from qmri.errors.metrics import (
    normalised_rmse,
    normalized_rmse,
    r_squared,
    residuals,
    rmse,
)

__all__ = [
    "normalised_rmse",
    "normalized_rmse",
    "r_squared",
    "residuals",
    "rmse",
]
