"""Tests for DWI phantom generation."""

import numpy as np
import pytest
from qmri.diffusion import adc
from qmri.dro import dwi


class TestGenerate:
    """Tests for dwi.generate()."""

    def test_single_voxel_no_noise(self) -> None:
        """Test single voxel phantom without noise."""
        phantom = dwi.generate(adc=1e-3, s0=1000.0)

        # Check shapes
        assert phantom.signal.shape == (4,)
        assert phantom.b_values.shape == (4,)

        # Check ground truth
        assert phantom.ground_truth["adc"].value == 1e-3
        assert phantom.ground_truth["adc"].units == "mm²/s"
        assert phantom.ground_truth["s0"].value == 1000.0

        # Check noise settings
        assert phantom.snr is None
        assert phantom.seed is None

    def test_single_voxel_with_noise(self) -> None:
        """Test single voxel phantom with noise."""
        phantom = dwi.generate(adc=1e-3, s0=1000.0, snr=50.0, seed=42)

        assert phantom.snr == 50.0
        assert phantom.seed == 42

    def test_single_voxel_fit_accuracy(self) -> None:
        """Test that fitting recovers ground truth ADC."""
        true_adc = 1e-3
        phantom = dwi.generate(adc=true_adc, s0=1000.0, snr=100.0, seed=42)

        result = adc.fit(phantom.signal, phantom.b_values)

        # Should be close to true value with high SNR
        np.testing.assert_allclose(result.adc, true_adc, rtol=0.1)

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        p1 = dwi.generate(adc=1e-3, snr=50.0, seed=42)
        p2 = dwi.generate(adc=1e-3, snr=50.0, seed=42)

        np.testing.assert_array_equal(p1.signal, p2.signal)

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        p1 = dwi.generate(adc=1e-3, snr=50.0, seed=42)
        p2 = dwi.generate(adc=1e-3, snr=50.0, seed=43)

        assert not np.array_equal(p1.signal, p2.signal)

    def test_multi_voxel_phantom(self) -> None:
        """Test multi-voxel phantom generation."""
        adc_map = np.array([[0.5e-3, 1.0e-3], [1.5e-3, 2.0e-3]])
        phantom = dwi.generate(adc=adc_map, snr=50.0, seed=42)

        # Check shape: (2, 2, 4) for 2x2 spatial and 4 b-values
        assert phantom.signal.shape == (2, 2, 4)

    def test_custom_b_values(self) -> None:
        """Test phantom with custom b-values."""
        custom_b = (0, 100, 200, 400, 800)
        phantom = dwi.generate(adc=1e-3, b_values=custom_b)

        assert phantom.b_values.shape == (5,)
        np.testing.assert_array_equal(phantom.b_values, custom_b)

    def test_gaussian_noise(self) -> None:
        """Test phantom with Gaussian noise."""
        phantom = dwi.generate(adc=1e-3, snr=50.0, noise_model="gaussian", seed=42)
        assert phantom.signal.shape == (4,)

    def test_invalid_noise_model(self) -> None:
        """Test that invalid noise model raises error."""
        with pytest.raises(ValueError, match="noise_model must be"):
            dwi.generate(adc=1e-3, noise_model="invalid")  # type: ignore[arg-type]


class TestGenerateCalibrationPhantom:
    """Tests for dwi.generate_calibration_phantom()."""

    def test_default_calibration_phantom(self) -> None:
        """Test default calibration phantom."""
        phantom = dwi.generate_calibration_phantom(seed=42)

        # Default: 6 ADC values, 8 b-values
        assert phantom.signal.shape == (6, 8)
        assert phantom.b_values.shape == (8,)
        assert phantom.snr == 50.0

    def test_custom_adc_values(self) -> None:
        """Test calibration phantom with custom ADC values."""
        adc_values = (0.5e-3, 1.0e-3, 1.5e-3)
        phantom = dwi.generate_calibration_phantom(adc_values=adc_values)

        assert phantom.signal.shape == (3, 8)

    def test_fit_all_voxels(self) -> None:
        """Test that all voxels can be fitted accurately."""
        phantom = dwi.generate_calibration_phantom(snr=100.0, seed=42)

        true_adc_values = phantom.ground_truth["adc"].value
        assert isinstance(true_adc_values, np.ndarray)

        for i, true_adc in enumerate(true_adc_values):
            result = adc.fit(phantom.signal[i], phantom.b_values)
            # With SNR=100, should be within 15%
            np.testing.assert_allclose(result.adc, true_adc, rtol=0.15)
