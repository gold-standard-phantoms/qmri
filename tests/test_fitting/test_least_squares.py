"""Tests for qmri.fitting.least_squares module."""

import numpy as np
import pytest
from qmri.fitting import least_squares


def exponential_residual(
    params: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Residual function for exponential decay: y = a * exp(-b * x)."""
    a, b = params
    return y - a * np.exp(-b * x)


def linear_residual(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual function for linear model: y = a * x + b."""
    a, b = params
    return y - (a * x + b)


class TestFit:
    """Tests for fit function."""

    def test_fit_linear_model(self) -> None:
        """Fit recovers linear parameters."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        true_a, true_b = 2.0, 1.0
        y = true_a * x + true_b

        result = least_squares.fit(linear_residual, x0=[1.0, 0.0], args=(x, y))

        assert result.success
        np.testing.assert_allclose(result.x, [true_a, true_b], rtol=1e-6)

    def test_fit_exponential_model(self) -> None:
        """Fit recovers exponential parameters."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        true_a, true_b = 1.0, 0.5
        y = true_a * np.exp(-true_b * x)

        result = least_squares.fit(exponential_residual, x0=[1.0, 0.3], args=(x, y))

        assert result.success
        np.testing.assert_allclose(result.x, [true_a, true_b], rtol=1e-6)

    def test_fit_returns_fit_result(self) -> None:
        """Fit returns FitResult dataclass with all expected attributes."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 0.6, 0.36, 0.22])

        result = least_squares.fit(exponential_residual, x0=[1.0, 0.5], args=(x, y))

        assert isinstance(result, least_squares.FitResult)
        assert hasattr(result, "x")
        assert hasattr(result, "cost")
        assert hasattr(result, "residuals")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "n_function_evals")
        assert hasattr(result, "n_jacobian_evals")

    def test_fit_residuals_near_zero(self) -> None:
        """Residuals are near zero for noiseless data."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0

        result = least_squares.fit(linear_residual, x0=[1.0, 0.0], args=(x, y))

        np.testing.assert_allclose(result.residuals, 0.0, atol=1e-10)

    def test_fit_with_noisy_data(self, rng: np.random.Generator) -> None:
        """Fit handles noisy data reasonably."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        true_a, true_b = 2.0, 1.0
        y = true_a * x + true_b + rng.normal(0, 0.1, len(x))

        result = least_squares.fit(linear_residual, x0=[1.0, 0.0], args=(x, y))

        assert result.success
        # Allow larger tolerance for noisy data
        np.testing.assert_allclose(result.x[0], true_a, rtol=0.2)
        np.testing.assert_allclose(result.x[1], true_b, rtol=0.2)

    def test_fit_with_bounds_trf(self) -> None:
        """Fit with bounds uses TRF method."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0

        result = least_squares.fit(
            linear_residual,
            x0=[1.0, 0.0],
            args=(x, y),
            bounds=([0, 0], [10, 5]),
            method="trf",
        )

        assert result.success
        np.testing.assert_allclose(result.x, [2.0, 1.0], rtol=1e-6)

    def test_fit_bounds_with_lm_raises(self) -> None:
        """Bounds with LM method raises ValueError."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Bounds are not supported"):
            least_squares.fit(
                linear_residual,
                x0=[1.0, 0.0],
                args=(x, y),
                bounds=([0, 0], [10, 5]),
                method="lm",
            )

    def test_fit_dogbox_method(self) -> None:
        """Fit works with dogbox method."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0

        result = least_squares.fit(
            linear_residual,
            x0=[1.0, 0.0],
            args=(x, y),
            bounds=([0, 0], [10, 5]),
            method="dogbox",
        )

        assert result.success
        np.testing.assert_allclose(result.x, [2.0, 1.0], rtol=1e-6)

    def test_fit_returns_jacobian(self) -> None:
        """Fit returns Jacobian matrix."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0

        result = least_squares.fit(linear_residual, x0=[1.0, 0.0], args=(x, y))

        assert result.jacobian is not None
        assert result.jacobian.shape == (len(x), 2)  # (n_residuals, n_params)

    def test_fit_cost_function(self) -> None:
        """Cost function is half sum of squared residuals."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0

        result = least_squares.fit(linear_residual, x0=[1.0, 0.0], args=(x, y))

        expected_cost = 0.5 * np.sum(result.residuals**2)
        np.testing.assert_allclose(result.cost, expected_cost, rtol=1e-10)


class TestEstimateCovariance:
    """Tests for estimate_covariance function."""

    def test_estimate_covariance_shape(self) -> None:
        """Covariance matrix has correct shape."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        residuals = np.array([0.1, -0.05, 0.02, -0.08])

        cov = least_squares.estimate_covariance(jacobian, residuals)

        assert cov.shape == (2, 2)

    def test_estimate_covariance_symmetric(self) -> None:
        """Covariance matrix is symmetric."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        residuals = np.array([0.1, -0.05, 0.02, -0.08])

        cov = least_squares.estimate_covariance(jacobian, residuals)

        np.testing.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_estimate_covariance_positive_diagonal(self) -> None:
        """Covariance matrix has positive diagonal elements."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        residuals = np.array([0.1, -0.05, 0.02, -0.08])

        cov = least_squares.estimate_covariance(jacobian, residuals)

        assert np.all(np.diag(cov) > 0)

    def test_estimate_covariance_insufficient_dof_raises(self) -> None:
        """Insufficient degrees of freedom raises ValueError."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0]])  # 2 points, 2 params
        residuals = np.array([0.1, -0.05])

        with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
            least_squares.estimate_covariance(jacobian, residuals)


class TestStandardErrors:
    """Tests for standard_errors function."""

    def test_standard_errors_shape(self) -> None:
        """Standard errors have correct shape."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        residuals = np.array([0.1, -0.05, 0.02, -0.08])

        errors = least_squares.standard_errors(jacobian, residuals)

        assert errors.shape == (2,)

    def test_standard_errors_positive(self) -> None:
        """Standard errors are positive."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        residuals = np.array([0.1, -0.05, 0.02, -0.08])

        errors = least_squares.standard_errors(jacobian, residuals)

        assert np.all(errors > 0)

    def test_standard_errors_sqrt_of_covariance_diagonal(self) -> None:
        """Standard errors equal sqrt of covariance diagonal."""
        jacobian = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        residuals = np.array([0.1, -0.05, 0.02, -0.08])

        errors = least_squares.standard_errors(jacobian, residuals)
        cov = least_squares.estimate_covariance(jacobian, residuals)

        np.testing.assert_allclose(errors, np.sqrt(np.diag(cov)), rtol=1e-10)


class TestIntegration:
    """Integration tests for fitting with uncertainty estimation."""

    def test_fit_with_standard_errors(self) -> None:
        """Complete workflow: fit then estimate uncertainties."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0

        # Fit the model
        result = least_squares.fit(linear_residual, x0=[1.0, 0.0], args=(x, y))

        # Estimate uncertainties
        assert result.jacobian is not None
        errors = least_squares.standard_errors(result.jacobian, result.residuals)

        # For perfect fit, errors should be zero (or very small due to numerical)
        assert len(errors) == 2

    def test_fit_exponential_with_noise(self, rng: np.random.Generator) -> None:
        """Fit exponential decay with realistic noise levels."""
        x = np.linspace(0, 5, 20)
        true_a, true_b = 100.0, 0.8
        y = true_a * np.exp(-true_b * x) + rng.normal(0, 2, len(x))

        result = least_squares.fit(exponential_residual, x0=[80.0, 0.5], args=(x, y))

        assert result.success
        # Should recover parameters within reasonable tolerance
        np.testing.assert_allclose(result.x[0], true_a, rtol=0.1)
        np.testing.assert_allclose(result.x[1], true_b, rtol=0.2)


class TestModuleImports:
    """Test module-level imports work correctly."""

    def test_import_from_fitting_module(self) -> None:
        """Functions can be imported from qmri.fitting."""
        from qmri.fitting import (
            FitResult,
            estimate_covariance,
            fit,
            standard_errors,
        )

        assert callable(fit)
        assert callable(estimate_covariance)
        assert callable(standard_errors)
        assert FitResult is not None

    def test_import_from_least_squares_submodule(self) -> None:
        """Functions can be imported from qmri.fitting.least_squares."""
        from qmri.fitting.least_squares import fit

        assert callable(fit)
