"""Tests for T2 relaxometry module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.relaxometry import t2


class TestSignalDecay:
    """Tests for T2 decay signal model."""

    def test_signal_decay_basic(self) -> None:
        """Test basic T2 decay signal."""
        te = np.array([0.01, 0.02, 0.04, 0.08])
        signal = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)
        assert signal.shape == (4,)
        # Signal should decrease with TE
        assert np.all(np.diff(signal) < 0)

    def test_signal_decay_with_offset(self) -> None:
        """Test T2 decay signal with offset."""
        te = np.array([0.01, 0.02, 0.04, 0.08])
        signal = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te, offset=50)
        # At very long TE, signal should approach offset
        assert signal[-1] > 50


class TestFitT2:
    """Tests for T2 fitting."""

    def test_fit_single_voxel_perfect_data(self) -> None:
        """Test T2 fitting recovers true values from noiseless data."""
        true_t2 = 0.05  # 50 ms
        true_amp = 1000.0
        te = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10])

        signal = t2.signal_decay(amplitude=true_amp, t2=true_t2, echo_times=te)

        result = t2.fit(signal, te, skip_echoes=0, model="reduced")

        assert_allclose(result.t2, true_t2, rtol=0.01)
        assert_allclose(result.amplitude, true_amp, rtol=0.01)

    def test_fit_full_model(self) -> None:
        """Test T2 fitting with full model (including offset)."""
        true_t2 = 0.05
        true_amp = 1000.0
        true_offset = 50.0
        te = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10])

        signal = t2.signal_decay(
            amplitude=true_amp, t2=true_t2, echo_times=te, offset=true_offset
        )

        result = t2.fit(signal, te, skip_echoes=0, model="full")

        assert_allclose(result.t2, true_t2, rtol=0.05)
        assert result.offset is not None
        assert_allclose(result.offset, true_offset, rtol=0.1)

    def test_fit_skip_echoes(self) -> None:
        """Test that skip_echoes parameter works."""
        te = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08])
        signal = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)

        # Should work with skip_echoes=1 (default)
        result = t2.fit(signal, te, skip_echoes=1)
        assert result.t2 > 0

    def test_fit_volume(self) -> None:
        """Test T2 fitting on a 3D volume."""
        true_t2 = 0.05
        te = np.array([0.01, 0.02, 0.04, 0.08])

        # Create a small 3D volume
        shape = (3, 3, 3)
        signal_1d = t2.signal_decay(amplitude=1000, t2=true_t2, echo_times=te)
        signal = np.broadcast_to(signal_1d, (*shape, len(te))).copy()

        result = t2.fit(signal, te, skip_echoes=0, model="reduced")

        assert result.t2.shape == shape
        assert result.amplitude.shape == shape

    def test_fit_with_mask(self) -> None:
        """Test T2 fitting with a mask."""
        te = np.array([0.01, 0.02, 0.04, 0.08])
        shape = (3, 3, 3)

        signal_1d = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)
        signal = np.broadcast_to(signal_1d, (*shape, len(te))).copy()

        # Create mask with only centre voxel
        mask = np.zeros(shape, dtype=bool)
        mask[1, 1, 1] = True

        result = t2.fit(signal, te, skip_echoes=0, mask=mask, model="reduced")

        # Only masked voxel should have non-zero T2
        assert result.t2[1, 1, 1] > 0
        assert result.t2[0, 0, 0] == 0

    def test_fit_mismatched_dimensions(self) -> None:
        """Test error when signal and echo times don't match."""
        te = np.array([0.01, 0.02, 0.04])
        signal = np.ones(5)  # Wrong size

        with pytest.raises(ValueError, match="time points"):
            t2.fit(signal, te)


class TestT2Result:
    """Tests for T2Result dataclass."""

    def test_t2result_frozen(self) -> None:
        """Test that T2Result is immutable."""
        result = t2.T2Result(
            t2=np.array(0.05),
            amplitude=np.array(1000.0),
            offset=None,
        )
        with pytest.raises(AttributeError):
            result.t2 = np.array(0.1)  # type: ignore[misc]
