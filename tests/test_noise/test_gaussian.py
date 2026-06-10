"""Tests for Gaussian noise generation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.noise import gaussian


class TestCalculateSigmaFromSNR:
    """Tests for calculate_sigma_from_snr function."""

    def test_scalar_signal(self) -> None:
        """Test sigma calculation with scalar signal."""
        sigma = gaussian.calculate_sigma_from_snr(1000.0, snr=50.0)
        assert_allclose(sigma, 20.0)

    def test_array_signal(self) -> None:
        """Test sigma calculation with array signal."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])
        sigma = gaussian.calculate_sigma_from_snr(signal, snr=35.0)
        # Mean signal is 700, so sigma = 700/35 = 20
        assert_allclose(sigma, 20.0)

    def test_array_with_zeros(self) -> None:
        """Test that zero values are excluded from mean calculation."""
        signal = np.array([0.0, 1000.0, 800.0, 600.0, 400.0, 0.0])
        sigma = gaussian.calculate_sigma_from_snr(signal, snr=35.0)
        # Mean of non-zero is 700, so sigma = 700/35 = 20
        assert_allclose(sigma, 20.0)

    def test_all_zeros_returns_zero(self) -> None:
        """Test that all-zero signal returns zero sigma."""
        signal = np.zeros(4)
        sigma = gaussian.calculate_sigma_from_snr(signal, snr=50.0)
        assert sigma == 0.0

    def test_negative_snr_raises(self) -> None:
        """Test that negative SNR raises ValueError."""
        with pytest.raises(ValueError, match="SNR must be positive"):
            gaussian.calculate_sigma_from_snr(1000.0, snr=-10.0)

    def test_zero_snr_raises(self) -> None:
        """Test that zero SNR raises ValueError."""
        with pytest.raises(ValueError, match="SNR must be positive"):
            gaussian.calculate_sigma_from_snr(1000.0, snr=0.0)


class TestAddGaussianNoise:
    """Tests for add_noise function."""

    def test_requires_snr_or_sigma(self) -> None:
        """Test that either snr or sigma must be provided."""
        signal = np.array([1000.0])
        with pytest.raises(ValueError, match="Either 'snr' or 'sigma'"):
            gaussian.add_noise(signal)

    def test_snr_and_sigma_mutually_exclusive(self) -> None:
        """Test that snr and sigma cannot both be provided."""
        signal = np.array([1000.0])
        with pytest.raises(ValueError, match="Only one of"):
            gaussian.add_noise(signal, snr=50.0, sigma=20.0)

    def test_negative_sigma_raises(self) -> None:
        """Test that negative sigma raises ValueError."""
        signal = np.array([1000.0])
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            gaussian.add_noise(signal, sigma=-10.0)

    def test_output_shape_matches_input(self) -> None:
        """Test that output has same shape as input."""
        rng = np.random.default_rng(42)
        signal_1d = np.array([1000.0, 800.0, 600.0, 400.0])
        signal_2d = np.ones((5, 5)) * 1000.0
        signal_3d = np.ones((3, 3, 3)) * 1000.0

        noisy_1d = gaussian.add_noise(signal_1d, snr=50.0, rng=rng)
        noisy_2d = gaussian.add_noise(signal_2d, snr=50.0, rng=rng)
        noisy_3d = gaussian.add_noise(signal_3d, snr=50.0, rng=rng)

        assert noisy_1d.shape == signal_1d.shape
        assert noisy_2d.shape == signal_2d.shape
        assert noisy_3d.shape == signal_3d.shape

    def test_scalar_input_returns_scalar(self) -> None:
        """Test that scalar input returns scalar output."""
        rng = np.random.default_rng(42)
        noisy = gaussian.add_noise(1000.0, snr=50.0, rng=rng)
        assert isinstance(noisy, float)

    def test_zero_sigma_returns_unchanged(self) -> None:
        """Test that zero sigma returns original signal."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])
        noisy = gaussian.add_noise(signal, sigma=0.0)
        assert_allclose(noisy, signal)

    def test_reproducibility_with_rng(self) -> None:
        """Test that same RNG seed produces same results."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])

        rng1 = np.random.default_rng(42)
        noisy1 = gaussian.add_noise(signal, snr=50.0, rng=rng1)

        rng2 = np.random.default_rng(42)
        noisy2 = gaussian.add_noise(signal, snr=50.0, rng=rng2)

        assert_allclose(noisy1, noisy2)

    def test_noise_has_correct_statistics(self) -> None:
        """Test that added noise has approximately correct mean and std."""
        rng = np.random.default_rng(42)
        signal = np.ones(100000) * 1000.0
        sigma = 20.0

        noisy = gaussian.add_noise(signal, sigma=sigma, rng=rng)
        noise = noisy - signal

        # Check noise statistics (with tolerance for finite sample)
        assert_allclose(np.mean(noise), 0.0, atol=1.0)
        assert_allclose(np.std(noise), sigma, rtol=0.02)

    def test_snr_based_noise(self, rng: np.random.Generator) -> None:
        """Test that SNR-based noise has correct amplitude."""
        signal = np.ones(100000) * 1000.0
        snr = 50.0
        expected_sigma = 20.0  # 1000 / 50

        noisy = gaussian.add_noise(signal, snr=snr, rng=rng)
        noise = noisy - signal

        assert_allclose(np.std(noise), expected_sigma, rtol=0.02)

    def test_different_rng_produces_different_results(self) -> None:
        """Test that different RNG seeds produce different noise."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])

        rng1 = np.random.default_rng(42)
        noisy1 = gaussian.add_noise(signal, snr=50.0, rng=rng1)

        rng2 = np.random.default_rng(123)
        noisy2 = gaussian.add_noise(signal, snr=50.0, rng=rng2)

        # Should be different (very unlikely to be exactly equal)
        assert not np.allclose(noisy1, noisy2)
