"""Error metrics for model evaluation.

This module provides standard error metrics for evaluating model fits,
including R-squared, RMSE, and residuals.

All functions work with both 1D arrays and N-D arrays where the comparison
is performed along the last axis.

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

from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def residuals(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    *,
    axis: int | None = None,
) -> NDArray[np.floating]: ...


@overload
def residuals(
    observed: float,
    predicted: float,
    *,
    axis: int | None = None,
) -> float: ...


def residuals(
    observed: NDArray[np.floating] | float,
    predicted: NDArray[np.floating] | float,
    *,
    axis: int | None = None,
) -> NDArray[np.floating] | float:
    """Calculate residuals (observed - predicted).

    Args:
        observed: Observed/measured values.
        predicted: Model-predicted values.
        axis: Axis along which to compute. Not used for simple residuals
            but included for API consistency.

    Returns:
        Residuals with same shape as input.

    Example:
        ```python
        import numpy as np
        from qmri.errors.metrics import residuals

        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 1.9, 3.2])
        print(residuals(observed, predicted))
        # [-0.1  0.1 -0.2]
        ```
    """
    del axis  # unused, but kept for API consistency
    obs = np.asarray(observed)
    pred = np.asarray(predicted)
    result = obs - pred
    if np.isscalar(observed) and np.isscalar(predicted):
        return float(result)
    result_arr: NDArray[np.floating] = result
    return result_arr


def r_squared(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    *,
    axis: int | None = None,
) -> NDArray[np.floating] | float:
    r"""Calculate coefficient of determination (R-squared).

    The R-squared value indicates how well the predicted values explain
    the variance in the observed values. Values range from 0 to 1, where
    1 indicates a perfect fit.

    Args:
        observed: Observed/measured values.
        predicted: Model-predicted values.
        axis: Axis along which to compute R-squared. If None, computed over
            flattened arrays. Default is None.

    Returns:
        R-squared value(s). Returns float if axis is None or input is 1D.

    The coefficient of determination is calculated as:

    $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

    where $SS_{res} = \sum_i (y_i - \hat{y}_i)^2$ is the residual
    sum of squares and $SS_{tot} = \sum_i (y_i - \bar{y})^2$ is
    the total sum of squares.

    When $SS_{tot} = 0$ (constant observed values), returns 0.0.

    Example:
        ```python
        import numpy as np
        from qmri.errors.metrics import r_squared

        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])
        print(f"R-squared: {r_squared(observed, predicted):.4f}")
        # R-squared: 0.9800

        # For multi-dimensional arrays:
        observed_2d = np.array([[1, 2, 3], [4, 5, 6]])
        predicted_2d = np.array([[1.1, 1.9, 3.1], [4.2, 4.8, 6.1]])
        r2 = r_squared(observed_2d, predicted_2d, axis=1)
        print(r2.shape)
        # (2,)
        ```
    """
    obs = np.asarray(observed)
    pred = np.asarray(predicted)

    # Calculate residuals
    resid = obs - pred
    ss_res = np.sum(resid**2, axis=axis, keepdims=True)

    # Calculate total sum of squares
    mean_obs = np.mean(obs, axis=axis, keepdims=True)
    ss_tot = np.sum((obs - mean_obs) ** 2, axis=axis, keepdims=True)

    # Handle division by zero (constant observed values)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)

    # Set R-squared to 0 where SS_tot is 0
    r2 = np.where(ss_tot > 0, r2, 0.0)

    # Remove keepdims dimension
    r2 = np.squeeze(r2)

    # Return float for scalar result
    if r2.ndim == 0:
        return float(r2)
    return r2


def rmse(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    *,
    axis: int | None = None,
) -> NDArray[np.floating] | float:
    r"""Calculate Root Mean Square Error (RMSE).

    RMSE measures the average magnitude of the errors between observed
    and predicted values, with the same units as the input data.

    Args:
        observed: Observed/measured values.
        predicted: Model-predicted values.
        axis: Axis along which to compute RMSE. If None, computed over
            flattened arrays. Default is None.

    Returns:
        RMSE value(s). Returns float if axis is None or input is 1D.

    RMSE is calculated as:

    $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

    Example:
        ```python
        import numpy as np
        from qmri.errors.metrics import rmse

        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])
        print(f"RMSE: {rmse(observed, predicted):.4f}")
        # RMSE: 0.1414
        ```
    """
    obs = np.asarray(observed)
    pred = np.asarray(predicted)

    # Calculate squared errors
    squared_errors = (obs - pred) ** 2

    # Mean squared error
    mse = np.mean(squared_errors, axis=axis)

    # Root mean squared error
    result = np.sqrt(mse)

    # Return float for scalar result
    if np.ndim(result) == 0:
        return float(result)
    result_arr: NDArray[np.floating] = result
    return result_arr


def normalised_rmse(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    *,
    axis: int | None = None,
    method: str = "range",
) -> NDArray[np.floating] | float:
    r"""Calculate Normalised Root Mean Square Error (NRMSE).

    NRMSE is RMSE normalised by a characteristic value of the observed
    data, making it dimensionless and comparable across different scales.

    Args:
        observed: Observed/measured values.
        predicted: Model-predicted values.
        axis: Axis along which to compute NRMSE. If None, computed over
            flattened arrays. Default is None.
        method: Normalisation method (default "range"):

            - "range": normalise by (max - min) of observed values
            - "mean": normalise by mean of observed values
            - "std": normalise by standard deviation of observed values

    Returns:
        NRMSE value(s). Returns float if axis is None or input is 1D.

    NRMSE is calculated as:

    $$NRMSE = \frac{RMSE}{y_{norm}}$$

    where $y_{norm}$ depends on the normalisation method:

    - range: $y_{norm} = y_{max} - y_{min}$
    - mean: $y_{norm} = \bar{y}$
    - std: $y_{norm} = \sigma_y$

    When the normalisation factor is zero, returns infinity.

    Example:
        ```python
        import numpy as np
        from qmri.errors.metrics import normalised_rmse

        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])
        print(f"NRMSE: {normalised_rmse(observed, predicted):.4f}")
        # NRMSE: 0.0471
        ```
    """
    obs = np.asarray(observed)
    pred = np.asarray(predicted)

    # Calculate RMSE
    rmse_val = rmse(obs, pred, axis=axis)

    # Calculate normalisation factor
    if method == "range":
        norm_factor = np.max(obs, axis=axis) - np.min(obs, axis=axis)
    elif method == "mean":
        norm_factor = np.mean(obs, axis=axis)
    elif method == "std":
        norm_factor = np.std(obs, axis=axis)
    else:
        msg = f"Unknown normalisation method: {method}. Use 'range', 'mean', or 'std'."
        raise ValueError(msg)

    # Handle division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = rmse_val / norm_factor

    # Set to inf where norm_factor is 0
    result = np.where(norm_factor > 0, result, np.inf)

    # Return float for scalar result
    if np.ndim(result) == 0:
        return float(result)
    return result


# Alias for American English spelling
normalized_rmse = normalised_rmse
"""Alias for :func:`normalised_rmse` using American English spelling."""
