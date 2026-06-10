"""Tests for MTR calculation module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.transfer import mtr


class TestCalculateMTR:
    """Tests for calculate_mtr function."""

    def test_calculate_mtr_single_voxel(self) -> None:
        """Test MTR calculation on single voxel."""
        signal_nosat = np.array([1000.0])
        signal_sat = np.array([700.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        # MTR = (1000 - 700) / 1000 * 100 = 30%
        assert_allclose(result.mtr, [30.0])

    def test_calculate_mtr_array(self) -> None:
        """Test MTR calculation on 1D array."""
        signal_nosat = np.array([1000.0, 800.0, 600.0])
        signal_sat = np.array([700.0, 560.0, 420.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        # All should have MTR = 30%
        assert_allclose(result.mtr, [30.0, 30.0, 30.0])

    def test_calculate_mtr_volume(self) -> None:
        """Test MTR calculation on 3D volume."""
        shape = (3, 3, 3)
        signal_nosat = np.ones(shape) * 1000.0
        signal_sat = np.ones(shape) * 600.0

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        assert result.mtr.shape == shape
        # MTR = (1000 - 600) / 1000 * 100 = 40%
        assert_allclose(result.mtr, np.ones(shape) * 40.0)

    def test_calculate_mtr_zero_reference(self) -> None:
        """Test MTR handles zero reference signal gracefully."""
        signal_nosat = np.array([0.0])
        signal_sat = np.array([0.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        # Should not raise, should return 0
        assert result.mtr[0] == 0.0

    def test_calculate_mtr_mixed_zero_reference(self) -> None:
        """Test MTR handles mixed zero and non-zero reference signals."""
        signal_nosat = np.array([1000.0, 0.0, 500.0])
        signal_sat = np.array([700.0, 0.0, 400.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        assert_allclose(result.mtr[0], 30.0)
        assert result.mtr[1] == 0.0  # Zero reference -> zero MTR
        assert_allclose(result.mtr[2], 20.0)

    def test_calculate_mtr_negative_result(self) -> None:
        """Test MTR with saturated signal higher than reference (unusual case)."""
        signal_nosat = np.array([700.0])
        signal_sat = np.array([1000.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        # MTR = (700 - 1000) / 700 * 100 = -42.86% (approximately)
        assert result.mtr[0] < 0

    def test_calculate_mtr_zero_mtr(self) -> None:
        """Test MTR when saturated and reference signals are equal."""
        signal_nosat = np.array([1000.0])
        signal_sat = np.array([1000.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        assert_allclose(result.mtr, [0.0])

    def test_calculate_mtr_full_saturation(self) -> None:
        """Test MTR with complete saturation (100%)."""
        signal_nosat = np.array([1000.0])
        signal_sat = np.array([0.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        assert_allclose(result.mtr, [100.0])

    def test_calculate_mtr_shape_mismatch(self) -> None:
        """Test MTR raises error for mismatched shapes."""
        signal_nosat = np.array([1000.0, 800.0])
        signal_sat = np.array([700.0])

        with pytest.raises(ValueError, match="same shape"):
            mtr.calculate_mtr(signal_nosat, signal_sat)

    def test_calculate_mtr_typical_brain_values(self) -> None:
        """Test MTR with typical brain tissue values (20-50%)."""
        # White matter typically has higher MTR than grey matter
        signal_nosat = np.array([1000.0, 1000.0])
        # WM: ~40%, GM: ~30%
        signal_sat = np.array([600.0, 700.0])

        result = mtr.calculate_mtr(signal_nosat, signal_sat)

        assert 35.0 < result.mtr[0] < 45.0  # White matter range
        assert 25.0 < result.mtr[1] < 35.0  # Grey matter range


class TestMTRResult:
    """Tests for MTRResult dataclass."""

    def test_mtr_result_frozen(self) -> None:
        """Test that MTRResult is immutable."""
        result = mtr.MTRResult(mtr=np.array([30.0]))
        with pytest.raises(AttributeError):
            result.mtr = np.array([40.0])  # type: ignore[misc]

    def test_mtr_result_attributes(self) -> None:
        """Test MTRResult has expected attributes."""
        mtr_values = np.array([30.0, 40.0, 50.0])
        result = mtr.MTRResult(mtr=mtr_values)

        assert hasattr(result, "mtr")
        assert_allclose(result.mtr, mtr_values)
