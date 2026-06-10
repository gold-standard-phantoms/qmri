"""Tests for relaxometry phantom generation."""

import numpy as np
import pytest
from qmri.dro import relaxometry
from qmri.relaxometry import t1


class TestGenerateT1IR:
    """Tests for relaxometry.generate_t1_ir()."""

    def test_single_voxel_no_noise(self) -> None:
        """Test single voxel IR phantom without noise."""
        ti = [0.1, 0.5, 1.0, 2.0, 3.0]
        phantom = relaxometry.generate_t1_ir(
            t1=1.2,
            inversion_times=ti,
            repetition_time=5.0,
        )

        # Check shapes
        assert phantom.signal.shape == (5,)
        assert phantom.time_points.shape == (5,)

        # Check ground truth
        assert phantom.ground_truth["t1"].value == 1.2
        assert phantom.ground_truth["t1"].units == "s"
        assert phantom.ground_truth["s0"].value == 1000.0
        assert phantom.ground_truth["inversion_efficiency"].value == 1.0

        # Check method info
        assert phantom.method == "ir"
        assert phantom.model == "general"
        assert phantom.repetition_time == 5.0

        # Check noise settings
        assert phantom.snr is None
        assert phantom.seed is None

    def test_single_voxel_with_noise(self) -> None:
        """Test single voxel IR phantom with noise."""
        phantom = relaxometry.generate_t1_ir(
            t1=1.2,
            inversion_times=[0.1, 0.5, 1.0, 2.0],
            snr=100.0,
            seed=42,
        )

        assert phantom.snr == 100.0
        assert phantom.seed == 42

    def test_single_voxel_fit_accuracy(self) -> None:
        """Test that fitting recovers ground truth T1."""
        true_t1 = 1.2
        phantom = relaxometry.generate_t1_ir(
            t1=true_t1,
            inversion_times=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
            repetition_time=5.0,
            snr=200.0,
            seed=42,
        )

        result = t1.fit_ir(
            phantom.signal,
            phantom.time_points,
            repetition_times=phantom.repetition_time,
            model="general",
        )

        # Should be close to true value with high SNR
        np.testing.assert_allclose(float(result.t1), true_t1, rtol=0.15)

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        p1 = relaxometry.generate_t1_ir(
            t1=1.2, inversion_times=[0.1, 0.5, 1.0], snr=50.0, seed=42
        )
        p2 = relaxometry.generate_t1_ir(
            t1=1.2, inversion_times=[0.1, 0.5, 1.0], snr=50.0, seed=42
        )

        np.testing.assert_array_equal(p1.signal, p2.signal)

    def test_classical_model(self) -> None:
        """Test IR phantom with classical model."""
        phantom = relaxometry.generate_t1_ir(
            t1=1.2,
            inversion_times=[0.1, 0.5, 1.0, 2.0],
            model="classical",
        )

        assert phantom.model == "classical"

    def test_multi_voxel_phantom(self) -> None:
        """Test multi-voxel IR phantom."""
        t1_map = np.array([[0.5, 1.0], [1.5, 2.0]])
        phantom = relaxometry.generate_t1_ir(
            t1=t1_map,
            inversion_times=[0.1, 0.5, 1.0, 2.0],
            snr=50.0,
            seed=42,
        )

        # Check shape: (2, 2, 4) for 2x2 spatial and 4 TIs
        assert phantom.signal.shape == (2, 2, 4)

    def test_invalid_model(self) -> None:
        """Test that invalid model raises error."""
        with pytest.raises(ValueError, match="model must be"):
            relaxometry.generate_t1_ir(
                t1=1.2,
                inversion_times=[0.1, 0.5, 1.0],
                model="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_noise_model(self) -> None:
        """Test that invalid noise model raises error."""
        with pytest.raises(ValueError, match="noise_model must be"):
            relaxometry.generate_t1_ir(
                t1=1.2,
                inversion_times=[0.1, 0.5, 1.0],
                noise_model="invalid",  # type: ignore[arg-type]
            )


class TestGenerateT1VTR:
    """Tests for relaxometry.generate_t1_vtr()."""

    def test_single_voxel_no_noise(self) -> None:
        """Test single voxel VTR phantom without noise."""
        tr = [0.5, 1.0, 2.0, 4.0, 8.0]
        phantom = relaxometry.generate_t1_vtr(t1=1.2, repetition_times=tr)

        # Check shapes
        assert phantom.signal.shape == (5,)
        assert phantom.time_points.shape == (5,)

        # Check ground truth
        assert phantom.ground_truth["t1"].value == 1.2
        assert phantom.ground_truth["m"].value == 1000.0

        # Check method info
        assert phantom.method == "vtr"
        assert phantom.model is None
        assert phantom.repetition_time is None

    def test_single_voxel_fit_accuracy(self) -> None:
        """Test that fitting recovers ground truth T1."""
        true_t1 = 1.2
        phantom = relaxometry.generate_t1_vtr(
            t1=true_t1,
            repetition_times=[0.5, 1.0, 2.0, 4.0, 8.0],
            snr=200.0,
            seed=42,
        )

        result = t1.fit_vtr(phantom.signal, phantom.time_points)

        # Should be close to true value with high SNR
        np.testing.assert_allclose(float(result.t1), true_t1, rtol=0.15)

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        p1 = relaxometry.generate_t1_vtr(
            t1=1.2, repetition_times=[0.5, 1.0, 2.0], snr=50.0, seed=42
        )
        p2 = relaxometry.generate_t1_vtr(
            t1=1.2, repetition_times=[0.5, 1.0, 2.0], snr=50.0, seed=42
        )

        np.testing.assert_array_equal(p1.signal, p2.signal)

    def test_multi_voxel_phantom(self) -> None:
        """Test multi-voxel VTR phantom."""
        t1_map = np.array([[0.5, 1.0], [1.5, 2.0]])
        phantom = relaxometry.generate_t1_vtr(
            t1=t1_map,
            repetition_times=[0.5, 1.0, 2.0, 4.0],
            snr=50.0,
            seed=42,
        )

        # Check shape: (2, 2, 4) for 2x2 spatial and 4 TRs
        assert phantom.signal.shape == (2, 2, 4)
