"""Tests for Rician noise generation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.noise import rician


class TestAddRicianNoise:
    """Tests for add_noise function."""

    def test_requires_snr_or_sigma(self) -> None:
        """Test that either snr or sigma must be provided."""
        signal = np.array([1000.0])
        with pytest.raises(ValueError, match="Either 'snr' or 'sigma'"):
            rician.add_noise(signal)

    def test_snr_and_sigma_mutually_exclusive(self) -> None:
        """Test that snr and sigma cannot both be provided."""
        signal = np.array([1000.0])
        with pytest.raises(ValueError, match="Only one of"):
            rician.add_noise(signal, snr=50.0, sigma=20.0)

    def test_negative_sigma_raises(self) -> None:
        """Test that negative sigma raises ValueError."""
        signal = np.array([1000.0])
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            rician.add_noise(signal, sigma=-10.0)

    def test_output_shape_matches_input(self) -> None:
        """Test that output has same shape as input."""
        rng = np.random.default_rng(42)
        signal_1d = np.array([1000.0, 800.0, 600.0, 400.0])
        signal_2d = np.ones((5, 5)) * 1000.0
        signal_3d = np.ones((3, 3, 3)) * 1000.0

        noisy_1d = rician.add_noise(signal_1d, snr=50.0, rng=rng)
        noisy_2d = rician.add_noise(signal_2d, snr=50.0, rng=rng)
        noisy_3d = rician.add_noise(signal_3d, snr=50.0, rng=rng)

        assert noisy_1d.shape == signal_1d.shape
        assert noisy_2d.shape == signal_2d.shape
        assert noisy_3d.shape == signal_3d.shape

    def test_scalar_input_returns_scalar(self) -> None:
        """Test that scalar input returns scalar output."""
        rng = np.random.default_rng(42)
        noisy = rician.add_noise(1000.0, snr=50.0, rng=rng)
        assert isinstance(noisy, float)

    def test_zero_sigma_returns_unchanged(self) -> None:
        """Test that zero sigma returns original signal."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])
        noisy = rician.add_noise(signal, sigma=0.0)
        assert_allclose(noisy, signal)

    def test_reproducibility_with_rng(self) -> None:
        """Test that same RNG seed produces same results."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])

        rng1 = np.random.default_rng(42)
        noisy1 = rician.add_noise(signal, snr=50.0, rng=rng1)

        rng2 = np.random.default_rng(42)
        noisy2 = rician.add_noise(signal, snr=50.0, rng=rng2)

        assert_allclose(noisy1, noisy2)

    def test_rician_output_always_positive(self) -> None:
        """Test that Rician noise always produces positive values."""
        rng = np.random.default_rng(42)
        signal = np.ones(10000) * 100.0  # Low signal

        # Even with very high noise, output should be positive
        noisy = rician.add_noise(signal, snr=1.0, rng=rng)

        assert np.all(noisy >= 0)

    def test_rician_bias_at_low_snr(self) -> None:
        """Test that Rician noise shows positive bias at low SNR.

        At low SNR, Rician noise causes an upward bias in measured signal.
        """
        rng = np.random.default_rng(42)
        signal = np.zeros(100000)  # Zero signal
        sigma = 100.0

        noisy = rician.add_noise(signal, sigma=sigma, rng=rng)

        # For zero signal with Rician noise, expected value is sigma * sqrt(pi/2)
        # This is approximately 1.253 * sigma
        expected_mean = sigma * np.sqrt(np.pi / 2)
        assert_allclose(np.mean(noisy), expected_mean, rtol=0.02)

    def test_rician_approaches_gaussian_at_high_snr(self) -> None:
        """Test that Rician noise approximates Gaussian at high SNR.

        At high SNR, the Rician distribution becomes approximately Gaussian.
        """
        rng = np.random.default_rng(42)
        signal = np.ones(100000) * 1000.0
        sigma = 10.0  # High SNR (100)

        noisy = rician.add_noise(signal, sigma=sigma, rng=rng)

        # At high SNR, mean should be close to original signal
        # (bias becomes negligible)
        assert_allclose(np.mean(noisy), 1000.0, rtol=0.01)

        # Standard deviation should be close to sigma
        assert_allclose(np.std(noisy), sigma, rtol=0.05)

    def test_different_rng_produces_different_results(self) -> None:
        """Test that different RNG seeds produce different noise."""
        signal = np.array([1000.0, 800.0, 600.0, 400.0])

        rng1 = np.random.default_rng(42)
        noisy1 = rician.add_noise(signal, snr=50.0, rng=rng1)

        rng2 = np.random.default_rng(123)
        noisy2 = rician.add_noise(signal, snr=50.0, rng=rng2)

        # Should be different (very unlikely to be exactly equal)
        assert not np.allclose(noisy1, noisy2)


class TestRicianVsGaussian:
    """Tests comparing Rician and Gaussian noise behaviour."""

    def test_rician_mean_exceeds_gaussian_at_low_snr(self) -> None:
        """Test that Rician noise has higher mean than Gaussian at low SNR."""
        from qmri.noise import gaussian

        signal = np.ones(50000) * 100.0  # Low signal
        sigma = 50.0  # SNR = 2

        rng_copy = np.random.default_rng(42)
        noisy_gaussian = gaussian.add_noise(signal.copy(), sigma=sigma, rng=rng_copy)

        rng_copy2 = np.random.default_rng(42)
        noisy_rician = rician.add_noise(signal.copy(), sigma=sigma, rng=rng_copy2)

        # Rician should have higher mean due to positive bias
        assert np.mean(noisy_rician) > np.mean(noisy_gaussian)
