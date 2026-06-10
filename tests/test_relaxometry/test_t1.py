"""Tests for T1 relaxometry module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.relaxometry import t1


class TestSignalIR:
    """Tests for inversion recovery signal models."""

    def test_signal_ir_general(self) -> None:
        """Test general IR signal model."""
        ti = np.array([0.1, 0.5, 1.0, 2.0])
        signal = t1.signal_ir(
            s0=1000,
            t1=1.0,
            inversion_times=ti,
            repetition_times=5.0,
            inversion_efficiency=1.0,
        )
        assert signal.shape == (4,)
        # At TI=0, signal should be negative (inverted)
        # At long TI, signal should approach S0

    def test_signal_ir_classical(self) -> None:
        """Test classical IR signal model."""
        ti = np.array([0.1, 0.5, 1.0, 2.0])
        signal = t1.signal_ir_classical(
            s0=1000,
            t1=1.0,
            inversion_times=ti,
            inversion_efficiency=1.0,
        )
        assert signal.shape == (4,)
        # At TI=0, signal ≈ -S0
        assert signal[0] < 0

    def test_signal_vtr(self) -> None:
        """Test VTR signal model."""
        tr = np.array([0.5, 1.0, 2.0, 4.0])
        signal = t1.signal_vtr(m=1000, t1=1.0, repetition_times=tr)
        assert signal.shape == (4,)
        # Signal should increase with TR
        assert np.all(np.diff(signal) > 0)


class TestFitIR:
    """Tests for IR T1 fitting."""

    def test_fit_ir_single_voxel_perfect_data(self) -> None:
        """Test IR fitting recovers true T1 from noiseless data."""
        true_t1 = 1.2
        true_s0 = 1000.0
        ti = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
        tr = 5.0

        signal = t1.signal_ir(
            s0=true_s0,
            t1=true_t1,
            inversion_times=ti,
            repetition_times=tr,
            inversion_efficiency=1.0,
        )
        # Take absolute value (magnitude image)
        signal = np.abs(signal)

        result = t1.fit_ir(signal, ti, repetition_times=tr)

        assert_allclose(result.t1, true_t1, rtol=0.1)

    def test_fit_ir_volume(self) -> None:
        """Test IR fitting on a 3D volume."""
        true_t1 = 1.0
        ti = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        tr = 5.0

        # Create a small 3D volume
        shape = (3, 3, 3)
        signal_1d = np.abs(
            t1.signal_ir(s0=1000, t1=true_t1, inversion_times=ti, repetition_times=tr)
        )
        signal = np.broadcast_to(signal_1d, (*shape, len(ti))).copy()

        result = t1.fit_ir(signal, ti, repetition_times=tr)

        assert result.t1.shape == shape
        assert result.s0.shape == shape
        assert result.inversion_efficiency.shape == shape


class TestFitVTR:
    """Tests for VTR T1 fitting."""

    def test_fit_vtr_single_voxel_perfect_data(self) -> None:
        """Test VTR fitting recovers true T1 from noiseless data."""
        true_t1 = 1.5
        true_m = 1000.0
        tr = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])

        signal = t1.signal_vtr(m=true_m, t1=true_t1, repetition_times=tr)

        result = t1.fit_vtr(signal, tr)

        assert_allclose(result.t1, true_t1, rtol=0.05)
        assert_allclose(result.m, true_m, rtol=0.05)

    def test_fit_vtr_volume(self) -> None:
        """Test VTR fitting on a 3D volume."""
        true_t1 = 1.0
        tr = np.array([0.5, 1.0, 2.0, 4.0])

        # Create a small 3D volume
        shape = (2, 2, 2)
        signal_1d = t1.signal_vtr(m=1000, t1=true_t1, repetition_times=tr)
        signal = np.broadcast_to(signal_1d, (*shape, len(tr))).copy()

        result = t1.fit_vtr(signal, tr)

        assert result.t1.shape == shape
        assert result.m.shape == shape


class TestFitDispatch:
    """Tests for the fit() dispatch function."""

    def test_fit_ir_method(self) -> None:
        """Test fit() with IR method."""
        ti = np.array([0.1, 0.5, 1.0, 2.0])
        signal = np.abs(
            t1.signal_ir(s0=1000, t1=1.0, inversion_times=ti, repetition_times=5.0)
        )

        result = t1.fit(signal, ti, method="ir", repetition_times=5.0)

        assert isinstance(result, t1.T1IRResult)

    def test_fit_vtr_method(self) -> None:
        """Test fit() with VTR method."""
        tr = np.array([0.5, 1.0, 2.0, 4.0])
        signal = t1.signal_vtr(m=1000, t1=1.0, repetition_times=tr)

        result = t1.fit(signal, tr, method="vtr")

        assert isinstance(result, t1.T1VTRResult)

    def test_fit_ir_requires_tr(self) -> None:
        """Test fit() raises error when TR not provided for IR."""
        ti = np.array([0.1, 0.5, 1.0, 2.0])
        signal = np.ones(4)

        with pytest.raises(ValueError, match="repetition_times required"):
            t1.fit(signal, ti, method="ir")
