"""Tests for perfusion phantom generation."""

import numpy as np
import pytest
from qmri.dro import perfusion


class TestGeneratePCASL:
    """Tests for perfusion.generate_pcasl()."""

    def test_single_voxel_no_noise(self) -> None:
        """Test single voxel pCASL phantom without noise."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
        )

        # Check ground truth
        assert phantom.ground_truth["perfusion_rate"].value == 60.0
        assert phantom.ground_truth["perfusion_rate"].units == "ml/100g/min"
        assert phantom.ground_truth["m0"].value == 1000.0
        assert phantom.ground_truth["transit_time"].value == 1.0
        assert phantom.ground_truth["t1_tissue"].value == 1.3

        # Check acquisition params
        assert phantom.acquisition_params["label_duration"] == 1.8
        assert phantom.acquisition_params["post_label_delay"] == 1.8
        assert phantom.acquisition_params["label_efficiency"] == 0.85

        # Check noise settings
        assert phantom.snr is None
        assert phantom.seed is None

        # Control should be greater than label (delta_m is positive)
        assert float(phantom.control) > float(phantom.label)

    def test_single_voxel_with_noise(self) -> None:
        """Test single voxel pCASL phantom with noise."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
            snr=50.0,
            seed=42,
        )

        assert phantom.snr == 50.0
        assert phantom.seed == 42

    def test_delta_m_positive(self) -> None:
        """Test that delta_m (control - label) is positive for healthy CBF."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
        )

        delta_m = float(phantom.control) - float(phantom.label)
        assert delta_m > 0

    def test_delta_m_scales_with_cbf(self) -> None:
        """Test that delta_m increases with CBF."""
        p1 = perfusion.generate_pcasl(perfusion_rate=30.0, m0=1000.0)
        p2 = perfusion.generate_pcasl(perfusion_rate=60.0, m0=1000.0)
        p3 = perfusion.generate_pcasl(perfusion_rate=90.0, m0=1000.0)

        dm1 = float(p1.control) - float(p1.label)
        dm2 = float(p2.control) - float(p2.label)
        dm3 = float(p3.control) - float(p3.label)

        # Delta_m should increase with CBF (approximately linearly)
        assert dm2 > dm1
        assert dm3 > dm2

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        p1 = perfusion.generate_pcasl(perfusion_rate=60.0, m0=1000.0, snr=50.0, seed=42)
        p2 = perfusion.generate_pcasl(perfusion_rate=60.0, m0=1000.0, snr=50.0, seed=42)

        np.testing.assert_array_equal(p1.control, p2.control)
        np.testing.assert_array_equal(p1.label, p2.label)

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        p1 = perfusion.generate_pcasl(perfusion_rate=60.0, m0=1000.0, snr=50.0, seed=42)
        p2 = perfusion.generate_pcasl(perfusion_rate=60.0, m0=1000.0, snr=50.0, seed=43)

        assert not np.array_equal(p1.control, p2.control)

    def test_multi_voxel_phantom(self) -> None:
        """Test multi-voxel pCASL phantom."""
        cbf_map = np.array([[40.0, 60.0], [80.0, 100.0]])
        phantom = perfusion.generate_pcasl(
            perfusion_rate=cbf_map,
            m0=1000.0,
            snr=50.0,
            seed=42,
        )

        # Check shape: (2, 2) for 2x2 spatial
        assert phantom.control.shape == (2, 2)
        assert phantom.label.shape == (2, 2)
        assert phantom.m0.shape == (2, 2)

    def test_custom_acquisition_params(self) -> None:
        """Test phantom with custom acquisition parameters."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
            label_duration=2.0,
            post_label_delay=2.0,
            label_efficiency=0.9,
            t1_blood=1.8,
            t1_tissue=1.4,
        )

        assert phantom.acquisition_params["label_duration"] == 2.0
        assert phantom.acquisition_params["post_label_delay"] == 2.0
        assert phantom.acquisition_params["label_efficiency"] == 0.9
        assert phantom.acquisition_params["t1_blood"] == 1.8
        assert phantom.ground_truth["t1_tissue"].value == 1.4

    def test_gaussian_noise(self) -> None:
        """Test phantom with Gaussian noise."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
            snr=50.0,
            noise_model="gaussian",
            seed=42,
        )

        # Should still work
        assert phantom.control is not None
        assert phantom.label is not None

    def test_invalid_noise_model(self) -> None:
        """Test that invalid noise model raises error."""
        with pytest.raises(ValueError, match="noise_model must be"):
            perfusion.generate_pcasl(
                perfusion_rate=60.0,
                m0=1000.0,
                noise_model="invalid",  # type: ignore[arg-type]
            )

    def test_zero_cbf(self) -> None:
        """Test phantom with zero CBF."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=0.0,
            m0=1000.0,
        )

        # Control and label should be equal (no perfusion signal)
        np.testing.assert_allclose(phantom.control, phantom.label)

    def test_high_transit_time(self) -> None:
        """Test phantom with high transit time (bolus not arrived)."""
        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
            transit_time=5.0,  # Very late arrival
            post_label_delay=1.8,  # Short PLD
        )

        # With late arrival, delta_m should be very small or zero
        delta_m = float(phantom.control) - float(phantom.label)
        assert delta_m < 1.0  # Essentially no signal
