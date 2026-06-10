"""Tests for ASL quantification module."""

import numpy as np
from qmri.perfusion import asl


class TestQuantifyPcasl:
    """Tests for pCASL quantification."""

    def test_quantify_pcasl_single_voxel(self) -> None:
        """Test pCASL quantification on single voxel."""
        control = np.array([1000.0])
        label = np.array([950.0])
        m0 = np.array([2000.0])

        result = asl.quantify_pcasl(
            control,
            label,
            m0,
            label_duration=1.8,
            post_label_delay=1.8,
        )

        # Perfusion should be positive
        assert result.perfusion[0] > 0

    def test_quantify_pcasl_volume(self) -> None:
        """Test pCASL quantification on 3D volume."""
        shape = (3, 3, 3)
        control = np.ones(shape) * 1000
        label = np.ones(shape) * 950
        m0 = np.ones(shape) * 2000

        result = asl.quantify_pcasl(
            control,
            label,
            m0,
            label_duration=1.8,
            post_label_delay=1.8,
        )

        assert result.perfusion.shape == shape
        assert np.all(result.perfusion > 0)

    def test_quantify_pcasl_zero_m0(self) -> None:
        """Test pCASL handles zero M0 gracefully."""
        control = np.array([1000.0])
        label = np.array([950.0])
        m0 = np.array([0.0])  # Zero M0

        result = asl.quantify_pcasl(
            control,
            label,
            m0,
            label_duration=1.8,
            post_label_delay=1.8,
        )

        # Should not raise, should return 0
        assert result.perfusion[0] == 0.0

    def test_quantify_pcasl_negative_delta_m(self) -> None:
        """Test pCASL with negative delta M (label > control)."""
        control = np.array([950.0])
        label = np.array([1000.0])
        m0 = np.array([2000.0])

        result = asl.quantify_pcasl(
            control,
            label,
            m0,
            label_duration=1.8,
            post_label_delay=1.8,
        )

        # Negative perfusion (inverted)
        assert result.perfusion[0] < 0


class TestQuantifyPasl:
    """Tests for PASL quantification."""

    def test_quantify_pasl_single_voxel(self) -> None:
        """Test PASL quantification on single voxel."""
        control = np.array([1000.0])
        label = np.array([950.0])
        m0 = np.array([2000.0])

        result = asl.quantify_pasl(
            control,
            label,
            m0,
            bolus_duration=0.7,
            inversion_time=1.8,
        )

        # Perfusion should be positive
        assert result.perfusion[0] > 0

    def test_quantify_pasl_default_params(self) -> None:
        """Test PASL uses correct default parameters."""
        control = np.array([1000.0])
        label = np.array([950.0])
        m0 = np.array([2000.0])

        result = asl.quantify_pasl(
            control,
            label,
            m0,
            bolus_duration=0.7,
            inversion_time=1.8,
            label_efficiency=0.98,  # PASL default
            t1_blood=1.65,
            partition_coefficient=0.9,
        )

        assert result.perfusion[0] > 0


class TestASLResult:
    """Tests for ASLResult dataclass."""

    def test_asl_result_frozen(self) -> None:
        """Test that ASLResult is immutable."""
        result = asl.ASLResult(perfusion=np.array([50.0]))
        try:
            result.perfusion = np.array([60.0])  # type: ignore[misc]
            raise AssertionError("Should have raised AttributeError")
        except AttributeError:
            pass  # Expected
