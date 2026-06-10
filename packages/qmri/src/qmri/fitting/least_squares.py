"""Least squares fitting utilities.

This module provides convenient wrappers around scipy.optimize.least_squares
with sensible defaults for qMRI applications.

Example:
    ```python
    import numpy as np
    from qmri.fitting import least_squares

    def residual_func(params, x, y):
        return y - params[0] * np.exp(-params[1] * x)

    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([1.0, 0.6, 0.4, 0.2, 0.15])
    result = least_squares.fit(residual_func, x0=[1.0, 0.5], args=(x_data, y_data))
    print(f"Amplitude: {result.x[0]:.3f}, Rate: {result.x[1]:.3f}")
    ```
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares as scipy_least_squares


@dataclass(frozen=True)
class FitResult:
    """Result of least squares fitting.

    Attributes:
        x: Solution vector (optimised parameters).
        cost: Value of the cost function at the solution.
        residuals: Residual values at the solution.
        success: Whether the optimisation converged successfully.
        message: Description of the termination reason.
        n_function_evals: Number of function evaluations.
        n_jacobian_evals: Number of Jacobian evaluations.
        jacobian: Jacobian matrix at the solution (if available).
    """

    x: NDArray[np.floating]
    cost: float
    residuals: NDArray[np.floating]
    success: bool
    message: str
    n_function_evals: int
    n_jacobian_evals: int
    jacobian: NDArray[np.floating] | None = None


Method = Literal["trf", "dogbox", "lm"]


def fit(
    residual_func: Callable[..., NDArray[np.floating]],
    x0: NDArray[np.floating] | list[float],
    *,
    args: tuple[Any, ...] = (),
    bounds: (
        tuple[NDArray[np.floating] | list[float], NDArray[np.floating] | list[float]]
        | None
    ) = None,
    method: Method = "lm",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    verbose: int = 0,
) -> FitResult:
    """Fit parameters using least squares optimisation.

    A wrapper around scipy.optimize.least_squares with sensible defaults
    for qMRI applications.

    Args:
        residual_func: Function that computes the residuals: f(x, *args) -> residuals.
            The function should return a 1D array of residuals.
        x0: Initial guess for the parameters.
        args: Additional arguments passed to residual_func.
        bounds: Lower and upper bounds on parameters: (lower, upper).
            Each array must match the size of x0.
            Only used with 'trf' or 'dogbox' methods.
        method: Optimisation method (default "lm"):

            - "lm": Levenberg-Marquardt (fast, no bounds support)
            - "trf": Trust Region Reflective (supports bounds)
            - "dogbox": Dogleg with rectangular trust regions (supports bounds)
        ftol: Tolerance for termination by change of cost function (default 1e-8).
        xtol: Tolerance for termination by change of parameters (default 1e-8).
        gtol: Tolerance for termination by norm of gradient (default 1e-8).
        max_nfev: Maximum number of function evaluations. If None, uses scipy default.
        verbose: Level of algorithm verbosity (default 0, silent).

    Returns:
        Object containing optimisation results.

    Example:
        Simple exponential decay fitting:

        ```python
        import numpy as np
        from qmri.fitting.least_squares import fit

        def exp_residual(params, x, y):
            a, b = params
            return y - a * np.exp(-b * x)

        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        y = np.array([1.0, 0.6, 0.35, 0.22, 0.13])
        result = fit(exp_residual, x0=[1.0, 0.5], args=(x, y))
        print(f"Success: {result.success}, params: {result.x}")

        # With bounds:
        result = fit(
            exp_residual,
            x0=[1.0, 0.5],
            args=(x, y),
            bounds=([0, 0], [10, 5]),
            method="trf"
        )
        ```
    """
    x0_arr = np.asarray(x0, dtype=np.float64)

    # Set up kwargs for scipy
    kwargs: dict[str, Any] = {
        "fun": residual_func,
        "x0": x0_arr,
        "args": args,
        "method": method,
        "ftol": ftol,
        "xtol": xtol,
        "gtol": gtol,
        "verbose": verbose,
    }

    if max_nfev is not None:
        kwargs["max_nfev"] = max_nfev

    # Handle bounds
    if bounds is not None:
        if method == "lm":
            msg = "Bounds are not supported with method='lm'. Use 'trf' or 'dogbox'."
            raise ValueError(msg)
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        kwargs["bounds"] = (lower, upper)

    # Run optimisation
    result: OptimizeResult = scipy_least_squares(**kwargs)

    # Extract Jacobian if available
    jacobian = None
    if hasattr(result, "jac") and result.jac is not None:
        jacobian = np.asarray(result.jac)

    return FitResult(
        x=np.asarray(result.x),
        cost=float(result.cost),
        residuals=np.asarray(result.fun),
        success=bool(result.success),
        message=str(result.message),
        n_function_evals=int(result.nfev),
        n_jacobian_evals=int(result.njev) if result.njev is not None else 0,
        jacobian=jacobian,
    )


def estimate_covariance(
    jacobian: NDArray[np.floating],
    residuals: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Estimate parameter covariance matrix from Jacobian.

    Computes the covariance matrix of the fitted parameters using the
    Jacobian matrix at the solution. This is useful for uncertainty
    estimation.

    Args:
        jacobian: Jacobian matrix at the solution, shape (n_residuals, n_params).
        residuals: Residual values at the solution, shape (n_residuals,).

    Returns:
        Covariance matrix, shape (n_params, n_params).

    The covariance matrix is estimated as:

    $$\text{Cov} = \sigma^2 (J^T J)^{-1}$$

    where $\sigma^2$ is the variance of residuals estimated as
    $\sigma^2 = \frac{\sum r_i^2}{n - p}$ with n residuals
    and p parameters.

    Example:
        ```python
        import numpy as np
        from qmri.fitting.least_squares import fit, estimate_covariance

        # After fitting, estimate uncertainties
        # result = fit(...)
        # cov = estimate_covariance(result.jacobian, result.residuals)
        # std_errors = np.sqrt(np.diag(cov))
        ```
    """
    n_residuals = len(residuals)
    n_params = jacobian.shape[1]

    # Degrees of freedom
    dof = n_residuals - n_params
    if dof <= 0:
        msg = (
            f"Insufficient degrees of freedom: {n_residuals} residuals "
            f"with {n_params} parameters"
        )
        raise ValueError(msg)

    # Estimate variance of residuals
    sigma_squared = np.sum(residuals**2) / dof

    # Compute (J^T J)^-1
    jtj = jacobian.T @ jacobian
    try:
        jtj_inv: NDArray[np.floating] = np.linalg.inv(jtj)
    except np.linalg.LinAlgError as e:
        msg = "Jacobian is singular, cannot estimate covariance"
        raise ValueError(msg) from e

    # Covariance matrix
    covariance: NDArray[np.floating] = sigma_squared * jtj_inv
    return covariance


def standard_errors(
    jacobian: NDArray[np.floating],
    residuals: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute standard errors of fitted parameters.

    A convenience function that extracts the standard errors (square root
    of diagonal elements of covariance matrix) from Jacobian and residuals.

    Args:
        jacobian: Jacobian matrix at the solution, shape (n_residuals, n_params).
        residuals: Residual values at the solution, shape (n_residuals,).

    Returns:
        Standard errors for each parameter, shape (n_params,).

    Example:
        ```python
        import numpy as np
        from qmri.fitting.least_squares import fit, standard_errors

        # After fitting
        # result = fit(...)
        # errors = standard_errors(result.jacobian, result.residuals)
        # print(f"Parameter uncertainties: {errors}")
        ```
    """
    cov = estimate_covariance(jacobian, residuals)
    result: NDArray[np.floating] = np.sqrt(np.diag(cov))
    return result
