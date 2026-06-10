"""Tests for qmri.diffusion.adc module."""

import numpy as np
import pytest
from qmri.diffusion import adc


class TestADCFitSingleVoxel:
    """Tests for single-voxel ADC fitting."""

    def test_fit_perfect_data_lls(self, simple_b_values: np.ndarray) -> None:
        """LLS recovers true ADC from noiseless data."""
        true_adc = 1.0e-3
        true_s0 = 1000.0
        signal = adc.signal_model(true_s0, true_adc, simple_b_values)

        result = adc.fit(signal, simple_b_values, method="lls")

        np.testing.assert_allclose(result.adc, true_adc, rtol=1e-10)
        np.testing.assert_allclose(result.s0, true_s0, rtol=1e-10)
        assert result.r_squared > 0.9999

    def test_fit_perfect_data_wlls(self, simple_b_values: np.ndarray) -> None:
        """WLLS recovers true ADC from noiseless data."""
        true_adc = 1.0e-3
        true_s0 = 1000.0
        signal = adc.signal_model(true_s0, true_adc, simple_b_values)

        result = adc.fit(signal, simple_b_values, method="wlls")

        np.testing.assert_allclose(result.adc, true_adc, rtol=1e-10)
        np.testing.assert_allclose(result.s0, true_s0, rtol=1e-10)
        assert result.r_squared > 0.9999

    def test_fit_perfect_data_iwlls(self, simple_b_values: np.ndarray) -> None:
        """IWLLS recovers true ADC from noiseless data."""
        true_adc = 1.0e-3
        true_s0 = 1000.0
        signal = adc.signal_model(true_s0, true_adc, simple_b_values)

        result = adc.fit(signal, simple_b_values, method="iwlls")

        np.testing.assert_allclose(result.adc, true_adc, rtol=1e-10)
        np.testing.assert_allclose(result.s0, true_s0, rtol=1e-10)
        assert result.r_squared > 0.9999
        assert result.iterations is not None
        assert result.iterations >= 1

    def test_fit_noisy_data(
        self, simple_b_values: np.ndarray, rng: np.random.Generator
    ) -> None:
        """IWLLS handles realistic noise levels."""
        true_adc = 1.0e-3
        true_s0 = 1000.0
        signal = adc.signal_model(true_s0, true_adc, simple_b_values)
        signal = signal + rng.normal(0, 20, signal.shape)  # SNR ~50

        result = adc.fit(signal, simple_b_values, method="iwlls")

        # Allow 10% tolerance for noisy data
        np.testing.assert_allclose(result.adc, true_adc, rtol=0.1)
        assert result.r_squared > 0.9

    def test_fit_different_adc_values(self, simple_b_values: np.ndarray) -> None:
        """Fitting works across range of typical ADC values."""
        adc_values = [0.3e-3, 0.8e-3, 1.5e-3, 3.0e-3]  # CSF to restricted

        for true_adc in adc_values:
            signal = adc.signal_model(1000.0, true_adc, simple_b_values)
            result = adc.fit(signal, simple_b_values)
            np.testing.assert_allclose(result.adc, true_adc, rtol=1e-6)

    def test_fit_invalid_method_raises(self, simple_b_values: np.ndarray) -> None:
        """Invalid method raises ValueError."""
        signal = np.array([1000, 600, 360, 130], dtype=np.float64)

        with pytest.raises(ValueError, match="Unknown method"):
            adc.fit(signal, simple_b_values, method="invalid")  # type: ignore[arg-type]


class TestADCFitVolume:
    """Tests for volume ADC fitting."""

    def test_fit_3d_volume(self, simple_b_values: np.ndarray) -> None:
        """Fitting works on 3D volumes."""
        shape = (4, 4, 4)
        true_adc = 1.0e-3
        true_s0 = 1000.0

        # Create 4D signal volume
        signal_4d = np.zeros((*shape, len(simple_b_values)))
        for i in range(len(simple_b_values)):
            signal_4d[..., i] = true_s0 * np.exp(-simple_b_values[i] * true_adc)

        result = adc.fit(signal_4d, simple_b_values)

        assert result.adc.shape == shape
        assert result.s0.shape == shape
        assert result.r_squared.shape == shape
        np.testing.assert_allclose(result.adc, true_adc, rtol=1e-6)

    def test_fit_with_mask(self, simple_b_values: np.ndarray) -> None:
        """Mask limits processing to specified voxels."""
        shape = (4, 4, 4)
        true_adc = 1.0e-3
        true_s0 = 1000.0

        signal_4d = np.zeros((*shape, len(simple_b_values)))
        for i in range(len(simple_b_values)):
            signal_4d[..., i] = true_s0 * np.exp(-simple_b_values[i] * true_adc)

        # Mask: only centre voxels
        mask = np.zeros(shape, dtype=bool)
        mask[1:3, 1:3, 1:3] = True

        result = adc.fit(signal_4d, simple_b_values, mask=mask)

        # Masked voxels should have ADC values
        assert np.all(result.adc[mask] > 0)
        # Non-masked voxels should be zero
        assert np.all(result.adc[~mask] == 0)


class TestSignalModel:
    """Tests for signal generation."""

    def test_signal_model_b0(self) -> None:
        """Signal at b=0 equals S0."""
        s0 = 1000.0
        adc_val = 1.0e-3
        b_values = np.array([0.0])

        signal = adc.signal_model(s0, adc_val, b_values)

        np.testing.assert_allclose(signal, [s0])

    def test_signal_model_decay(self, simple_b_values: np.ndarray) -> None:
        """Signal decreases with increasing b-value."""
        signal = adc.signal_model(1000.0, 1.0e-3, simple_b_values)

        # Signal should monotonically decrease
        assert np.all(np.diff(signal) < 0)

    def test_signal_model_array_inputs(self, simple_b_values: np.ndarray) -> None:
        """Signal model works with array S0 and ADC."""
        s0 = np.array([1000.0, 800.0])[:, np.newaxis]
        adc_val = np.array([1.0e-3, 0.5e-3])[:, np.newaxis]

        signal = adc.signal_model(s0, adc_val, simple_b_values)

        assert signal.shape == (2, len(simple_b_values))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fit_zero_signals(self, simple_b_values: np.ndarray) -> None:
        """Fitting handles zero signals gracefully."""
        signal = np.zeros_like(simple_b_values)

        result = adc.fit(signal, simple_b_values)

        assert result.adc == 0.0
        assert result.s0 == 0.0
        assert result.r_squared == 0.0

    def test_fit_single_valid_point(self, simple_b_values: np.ndarray) -> None:
        """Fitting with insufficient points returns zeros."""
        signal = np.array([1000, 0, 0, 0], dtype=np.float64)

        result = adc.fit(signal, simple_b_values)

        assert result.adc == 0.0

    def test_fit_negative_signals_ignored(self, simple_b_values: np.ndarray) -> None:
        """Negative signals are excluded from fitting."""
        signal = np.array([1000, -100, 368, 135], dtype=np.float64)

        result = adc.fit(signal, simple_b_values)

        # Should still get reasonable fit from valid points
        assert result.adc > 0
        assert result.r_squared > 0
