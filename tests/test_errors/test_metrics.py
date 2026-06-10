"""Tests for qmri.errors.metrics module."""

import numpy as np
import pytest
from qmri.errors import metrics


class TestResiduals:
    """Tests for residuals function."""

    def test_residuals_perfect_prediction(self) -> None:
        """Residuals are zero for perfect predictions."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = observed.copy()

        result = metrics.residuals(observed, predicted)

        np.testing.assert_array_equal(result, np.zeros(4))

    def test_residuals_positive(self) -> None:
        """Residuals are positive when observed > predicted."""
        observed = np.array([5.0, 6.0, 7.0])
        predicted = np.array([4.0, 5.0, 6.0])

        result = metrics.residuals(observed, predicted)

        np.testing.assert_array_equal(result, np.array([1.0, 1.0, 1.0]))

    def test_residuals_negative(self) -> None:
        """Residuals are negative when observed < predicted."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])

        result = metrics.residuals(observed, predicted)

        np.testing.assert_array_equal(result, np.array([-1.0, -1.0, -1.0]))

    def test_residuals_2d_array(self) -> None:
        """Residuals work with 2D arrays."""
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.1, 1.9], [3.1, 3.9]])

        result = metrics.residuals(observed, predicted)

        expected = np.array([[-0.1, 0.1], [-0.1, 0.1]])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_residuals_scalar_inputs(self) -> None:
        """Residuals work with scalar inputs."""
        result = metrics.residuals(5.0, 3.0)

        assert result == 2.0
        assert isinstance(result, float)


class TestRSquared:
    """Tests for r_squared function."""

    def test_r_squared_perfect_fit(self) -> None:
        """R-squared is 1.0 for perfect fit."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = observed.copy()

        result = metrics.r_squared(observed, predicted)

        assert result == 1.0

    def test_r_squared_mean_prediction(self) -> None:
        """R-squared is 0.0 when predictions equal the mean."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        mean_val = np.mean(observed)
        predicted = np.full_like(observed, mean_val)

        result = metrics.r_squared(observed, predicted)

        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_r_squared_known_value(self) -> None:
        """R-squared matches known calculation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])

        result = metrics.r_squared(observed, predicted)

        # Manual calculation
        ss_res = np.sum((observed - predicted) ** 2)  # 0.01 + 0.01 + 0.04 + 0.04 = 0.1
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)  # 5.0
        expected = 1.0 - (ss_res / ss_tot)  # 1 - 0.02 = 0.98

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_r_squared_constant_observed(self) -> None:
        """R-squared is 0 when observed values are constant."""
        observed = np.array([5.0, 5.0, 5.0, 5.0])
        predicted = np.array([4.0, 5.0, 6.0, 5.0])

        result = metrics.r_squared(observed, predicted)

        assert result == 0.0

    def test_r_squared_with_axis(self) -> None:
        """R-squared computed along specified axis."""
        observed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        predicted = np.array([[1.1, 1.9, 3.1], [4.2, 4.8, 6.1]])

        result = metrics.r_squared(observed, predicted, axis=1)

        assert result.shape == (2,)
        # Each row should have high R-squared
        assert np.all(result > 0.95)

    def test_r_squared_2d_flattened(self) -> None:
        """R-squared computed over flattened arrays by default."""
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.1, 1.9], [3.1, 3.9]])

        result = metrics.r_squared(observed, predicted)

        # Should return single float
        assert isinstance(result, float)
        assert result > 0.95


class TestRMSE:
    """Tests for rmse function."""

    def test_rmse_perfect_fit(self) -> None:
        """RMSE is 0.0 for perfect fit."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = observed.copy()

        result = metrics.rmse(observed, predicted)

        assert result == 0.0

    def test_rmse_known_value(self) -> None:
        """RMSE matches known calculation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])

        result = metrics.rmse(observed, predicted)

        # Manual: sqrt(mean([0.01, 0.01, 0.04, 0.04])) = sqrt(0.025) ~ 0.158
        expected = np.sqrt(0.1 / 4)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_rmse_constant_error(self) -> None:
        """RMSE equals error magnitude for constant error."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([2.0, 3.0, 4.0, 5.0])  # Constant error of -1.0

        result = metrics.rmse(observed, predicted)

        assert result == 1.0

    def test_rmse_with_axis(self) -> None:
        """RMSE computed along specified axis."""
        observed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        predicted = np.array([[1.1, 2.1, 3.1], [4.0, 5.0, 6.0]])

        result = metrics.rmse(observed, predicted, axis=1)

        assert result.shape == (2,)
        np.testing.assert_allclose(result[0], 0.1, rtol=1e-10)
        np.testing.assert_allclose(result[1], 0.0, rtol=1e-10)

    def test_rmse_2d_flattened(self) -> None:
        """RMSE computed over flattened arrays by default."""
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.5, 2.5], [3.5, 4.5]])

        result = metrics.rmse(observed, predicted)

        # Should return single float
        assert isinstance(result, float)
        assert result == 0.5


class TestNormalisedRMSE:
    """Tests for normalised_rmse function."""

    def test_nrmse_perfect_fit(self) -> None:
        """NRMSE is 0.0 for perfect fit."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = observed.copy()

        result = metrics.normalised_rmse(observed, predicted)

        assert result == 0.0

    def test_nrmse_range_method(self) -> None:
        """NRMSE with range normalisation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])

        result = metrics.normalised_rmse(observed, predicted, method="range")

        rmse_val = metrics.rmse(observed, predicted)
        data_range = 4.0 - 1.0  # max - min
        expected = rmse_val / data_range

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_nrmse_mean_method(self) -> None:
        """NRMSE with mean normalisation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])

        result = metrics.normalised_rmse(observed, predicted, method="mean")

        rmse_val = metrics.rmse(observed, predicted)
        expected = rmse_val / np.mean(observed)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_nrmse_std_method(self) -> None:
        """NRMSE with standard deviation normalisation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])

        result = metrics.normalised_rmse(observed, predicted, method="std")

        rmse_val = metrics.rmse(observed, predicted)
        expected = rmse_val / np.std(observed)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_nrmse_constant_observed_returns_inf(self) -> None:
        """NRMSE returns infinity for constant observed values (range method)."""
        observed = np.array([5.0, 5.0, 5.0, 5.0])
        predicted = np.array([4.0, 5.0, 6.0, 5.0])

        result = metrics.normalised_rmse(observed, predicted, method="range")

        assert np.isinf(result)

    def test_nrmse_invalid_method_raises(self) -> None:
        """Invalid normalisation method raises ValueError."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 1.9, 3.1])

        with pytest.raises(ValueError, match="Unknown normalisation method"):
            metrics.normalised_rmse(observed, predicted, method="invalid")

    def test_normalized_rmse_alias(self) -> None:
        """American English spelling alias works."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])

        result_uk = metrics.normalised_rmse(observed, predicted)
        result_us = metrics.normalized_rmse(observed, predicted)

        assert result_uk == result_us


class TestModuleImports:
    """Test module-level imports work correctly."""

    def test_import_from_errors_module(self) -> None:
        """Functions can be imported from qmri.errors."""
        from qmri.errors import (
            normalised_rmse,
            normalized_rmse,
            r_squared,
            residuals,
            rmse,
        )

        assert callable(r_squared)
        assert callable(rmse)
        assert callable(residuals)
        assert callable(normalised_rmse)
        assert callable(normalized_rmse)

    def test_import_from_metrics_submodule(self) -> None:
        """Functions can be imported from qmri.errors.metrics."""
        from qmri.errors.metrics import r_squared, rmse

        assert callable(r_squared)
        assert callable(rmse)
