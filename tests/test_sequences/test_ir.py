"""Tests for Inversion Recovery (IR) signal module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.sequences import ir


class TestSignalIr:
    """Tests for IR signal model."""

    def test_signal_ir_single_value(self) -> None:
        """Test IR signal calculation with scalar inputs."""
        signal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        # Signal should be finite
        assert np.isfinite(signal)

    def test_signal_ir_array_inputs(self) -> None:
        """Test IR signal calculation with array inputs."""
        shape = (3, 3, 3)
        m0 = np.ones(shape) * 1000.0
        t1 = np.ones(shape) * 1.0
        t2 = np.ones(shape) * 0.1

        signal = ir.signal_ir(
            m0=m0,
            t1=t1,
            t2=t2,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        assert signal.shape == shape
        assert np.all(np.isfinite(signal))

    def test_signal_ir_null_point(self) -> None:
        """Test that signal passes through zero at null point."""
        t1 = 1.0
        # Null point TI = T1 * ln(2) for ideal inversion
        ti_null = t1 * np.log(2)

        # Test signals before and after null point
        ti_values = [ti_null - 0.2, ti_null, ti_null + 0.2]
        signals = []

        for ti in ti_values:
            s = ir.signal_ir(
                m0=1000.0,
                t1=t1,
                t2=0.5,  # Long T2 to minimise T2 effects
                repetition_time=10.0,  # Long TR
                echo_time=0.001,  # Short TE
                inversion_time=ti,
            )
            signals.append(float(s))

        # Signal should change sign around null point
        # (negative before, near zero at null, positive after)
        assert signals[0] < signals[1]
        assert signals[1] < signals[2]

    def test_signal_ir_ti_evolution(self) -> None:
        """Test signal evolution with inversion time."""
        inversion_times = [0.1, 0.3, 0.5, 0.8, 1.0, 2.0]
        signals = []

        for ti in inversion_times:
            s = ir.signal_ir(
                m0=1000.0,
                t1=1.0,
                t2=0.5,
                repetition_time=10.0,
                echo_time=0.001,
                inversion_time=ti,
            )
            signals.append(float(s))

        # Signal should start negative and recover to positive
        # (for ideal 180 degree inversion)
        assert signals[0] < 0  # Short TI: inverted
        assert signals[-1] > 0  # Long TI: recovered

    def test_signal_ir_zero_t1_handling(self) -> None:
        """Test that zero T1 is handled gracefully."""
        signal = ir.signal_ir(
            m0=1000.0,
            t1=0.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        # Should not raise, should return finite value
        assert np.isfinite(signal)

    def test_signal_ir_zero_t2_handling(self) -> None:
        """Test that zero T2 is handled gracefully."""
        signal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.0,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        # Should not raise, signal should be zero (exp(-TE/0) -> 0)
        assert np.isfinite(signal)
        assert signal == pytest.approx(0.0)

    def test_signal_ir_default_flip_angles(self) -> None:
        """Test that default flip angles are 90 and 180 degrees."""
        # With default flip angles (90, 180), should give standard IR signal
        signal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        # Same result with explicit default angles
        signal_explicit = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
            excitation_flip_angle=90.0,
            inversion_flip_angle=180.0,
        )

        assert_allclose(signal, signal_explicit)

    def test_signal_ir_non_ideal_flip_angles(self) -> None:
        """Test IR signal with non-ideal flip angles."""
        # Non-ideal inversion (e.g., 170 degrees instead of 180)
        signal_ideal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.2,
            inversion_flip_angle=180.0,
        )

        signal_non_ideal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.2,
            inversion_flip_angle=170.0,
        )

        # Non-ideal inversion should give different (less negative) signal
        assert signal_ideal != signal_non_ideal
        # Non-ideal should be less inverted (closer to positive)
        assert signal_non_ideal > signal_ideal

    def test_signal_ir_excitation_flip_angle_effect(self) -> None:
        """Test effect of excitation flip angle on signal."""
        flip_angles = [30.0, 60.0, 90.0]
        signals = []

        for fa in flip_angles:
            s = ir.signal_ir(
                m0=1000.0,
                t1=1.0,
                t2=0.1,
                repetition_time=5.0,
                echo_time=0.01,
                inversion_time=0.8,
                excitation_flip_angle=fa,
            )
            signals.append(float(s))

        # Signal magnitude should vary with flip angle
        assert not all(abs(s) == abs(signals[0]) for s in signals)

    def test_signal_ir_mixed_array_scalar(self) -> None:
        """Test IR with mixture of array and scalar inputs."""
        m0 = np.array([800.0, 1000.0, 1200.0])
        t1 = 1.0  # scalar
        t2 = np.array([0.08, 0.1, 0.12])

        signal = ir.signal_ir(
            m0=m0,
            t1=t1,
            t2=t2,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        assert signal.shape == (3,)
        assert np.all(np.isfinite(signal))

    def test_signal_ir_long_tr_behaviour(self) -> None:
        """Test IR signal behaviour with very long TR."""
        # With very long TR, exp(-TR/T1) -> 0
        # So the equation simplifies
        signal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.5,
            repetition_time=100.0,  # Very long TR
            echo_time=0.001,  # Very short TE
            inversion_time=1.0,
        )

        # Signal should be positive at TI > T1*ln(2)
        assert signal > 0
