"""Tests for Gradient Echo (GRE) signal module."""

import numpy as np
import pytest
from qmri.sequences import gre


class TestSignalGre:
    """Tests for GRE signal model."""

    def test_signal_gre_single_value(self) -> None:
        """Test GRE signal calculation with scalar inputs."""
        signal = gre.signal_gre(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            t2_star=0.05,
            repetition_time=0.025,
            echo_time=0.005,
            flip_angle=30.0,
        )

        # Signal should be positive
        assert signal > 0

    def test_signal_gre_array_inputs(self) -> None:
        """Test GRE signal calculation with array inputs."""
        shape = (3, 3, 3)
        m0 = np.ones(shape) * 1000.0
        t1 = np.ones(shape) * 1.0
        t2 = np.ones(shape) * 0.1
        t2_star = np.ones(shape) * 0.05

        signal = gre.signal_gre(
            m0=m0,
            t1=t1,
            t2=t2,
            t2_star=t2_star,
            repetition_time=0.025,
            echo_time=0.005,
            flip_angle=30.0,
        )

        assert signal.shape == shape
        assert np.all(signal > 0)

    def test_signal_gre_flip_angle_dependence(self) -> None:
        """Test that signal varies with flip angle."""
        flip_angles = [10.0, 30.0, 50.0, 70.0, 90.0]
        signals = []

        for fa in flip_angles:
            s = gre.signal_gre(
                m0=1000.0,
                t1=1.0,
                t2=0.1,
                t2_star=0.05,
                repetition_time=0.025,
                echo_time=0.005,
                flip_angle=fa,
            )
            signals.append(float(s))

        # Signal should vary with flip angle (not all equal)
        assert not all(s == signals[0] for s in signals)

    def test_signal_gre_ernst_angle(self) -> None:
        """Test signal near Ernst angle gives maximum signal."""
        # Ernst angle for T1=1.0s and TR=0.5s is arccos(exp(-TR/T1))
        t1 = 1.0
        tr = 0.5
        ernst_angle = np.degrees(np.arccos(np.exp(-tr / t1)))

        # Test signals at and around Ernst angle
        angles = [ernst_angle - 20, ernst_angle, ernst_angle + 20]
        signals = []

        for fa in angles:
            s = gre.signal_gre(
                m0=1000.0,
                t1=t1,
                t2=0.1,
                t2_star=0.05,
                repetition_time=tr,
                echo_time=0.005,
                flip_angle=fa,
            )
            signals.append(float(s))

        # Signal at Ernst angle should be near maximum
        assert signals[1] >= signals[0]
        assert signals[1] >= signals[2]

    def test_signal_gre_zero_t1_handling(self) -> None:
        """Test that zero T1 is handled gracefully."""
        signal = gre.signal_gre(
            m0=1000.0,
            t1=0.0,
            t2=0.1,
            t2_star=0.05,
            repetition_time=0.025,
            echo_time=0.005,
            flip_angle=30.0,
        )

        # Should not raise, should return finite value
        assert np.isfinite(signal)

    def test_signal_gre_zero_t2_star_handling(self) -> None:
        """Test that zero T2* is handled gracefully."""
        signal = gre.signal_gre(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            t2_star=0.0,
            repetition_time=0.025,
            echo_time=0.005,
            flip_angle=30.0,
        )

        # Should not raise, signal should be zero (exp(-TE/0) -> 0)
        assert np.isfinite(signal)
        assert signal == pytest.approx(0.0)

    def test_signal_gre_te_decay(self) -> None:
        """Test that signal decreases with increasing TE."""
        echo_times = [0.001, 0.005, 0.01, 0.02]
        signals = []

        for te in echo_times:
            s = gre.signal_gre(
                m0=1000.0,
                t1=1.0,
                t2=0.1,
                t2_star=0.05,
                repetition_time=0.025,
                echo_time=te,
                flip_angle=30.0,
            )
            signals.append(float(s))

        # Signal should decrease with increasing TE (T2* decay)
        assert all(signals[i] > signals[i + 1] for i in range(len(signals) - 1))

    def test_signal_gre_mixed_array_scalar(self) -> None:
        """Test GRE with mixture of array and scalar inputs."""
        m0 = np.array([800.0, 1000.0, 1200.0])
        t1 = 1.0  # scalar
        t2 = np.array([0.08, 0.1, 0.12])
        t2_star = 0.05  # scalar

        signal = gre.signal_gre(
            m0=m0,
            t1=t1,
            t2=t2,
            t2_star=t2_star,
            repetition_time=0.025,
            echo_time=0.005,
            flip_angle=30.0,
        )

        assert signal.shape == (3,)
        assert np.all(signal > 0)
