"""Tests for Spin Echo (SE) signal module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.sequences import se


class TestSignalSe:
    """Tests for SE signal model."""

    def test_signal_se_single_value(self) -> None:
        """Test SE signal calculation with scalar inputs."""
        signal = se.signal_se(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=3.0,
            echo_time=0.08,
        )

        # Signal should be positive
        assert signal > 0

    def test_signal_se_array_inputs(self) -> None:
        """Test SE signal calculation with array inputs."""
        shape = (3, 3, 3)
        m0 = np.ones(shape) * 1000.0
        t1 = np.ones(shape) * 1.0
        t2 = np.ones(shape) * 0.1

        signal = se.signal_se(
            m0=m0,
            t1=t1,
            t2=t2,
            repetition_time=3.0,
            echo_time=0.08,
        )

        assert signal.shape == shape
        assert np.all(signal > 0)

    def test_signal_se_long_tr_approaches_m0(self) -> None:
        """Test that signal approaches M0 with very long TR and short TE."""
        m0 = 1000.0
        signal = se.signal_se(
            m0=m0,
            t1=1.0,
            t2=0.1,
            repetition_time=100.0,  # Very long TR
            echo_time=0.0001,  # Very short TE
        )

        # Signal should be close to M0
        assert_allclose(signal, m0, rtol=0.01)

    def test_signal_se_te_decay(self) -> None:
        """Test that signal decreases with increasing TE."""
        echo_times = [0.01, 0.05, 0.1, 0.2]
        signals = []

        for te in echo_times:
            s = se.signal_se(
                m0=1000.0,
                t1=1.0,
                t2=0.1,
                repetition_time=3.0,
                echo_time=te,
            )
            signals.append(float(s))

        # Signal should decrease with increasing TE (T2 decay)
        assert all(signals[i] > signals[i + 1] for i in range(len(signals) - 1))

    def test_signal_se_tr_saturation(self) -> None:
        """Test that signal increases with TR (T1 recovery)."""
        repetition_times = [0.5, 1.0, 2.0, 5.0]
        signals = []

        for tr in repetition_times:
            s = se.signal_se(
                m0=1000.0,
                t1=1.0,
                t2=0.1,
                repetition_time=tr,
                echo_time=0.01,
            )
            signals.append(float(s))

        # Signal should increase with TR (more T1 recovery)
        assert all(signals[i] < signals[i + 1] for i in range(len(signals) - 1))

    def test_signal_se_zero_t1_handling(self) -> None:
        """Test that zero T1 is handled gracefully."""
        signal = se.signal_se(
            m0=1000.0,
            t1=0.0,
            t2=0.1,
            repetition_time=3.0,
            echo_time=0.08,
        )

        # Should not raise, should return finite value
        assert np.isfinite(signal)

    def test_signal_se_zero_t2_handling(self) -> None:
        """Test that zero T2 is handled gracefully."""
        signal = se.signal_se(
            m0=1000.0,
            t1=1.0,
            t2=0.0,
            repetition_time=3.0,
            echo_time=0.08,
        )

        # Should not raise, signal should be zero (exp(-TE/0) -> 0)
        assert np.isfinite(signal)
        assert signal == pytest.approx(0.0)

    def test_signal_se_t2_weighting(self) -> None:
        """Test T2 contrast: longer T2 gives higher signal at given TE."""
        t2_values = [0.05, 0.1, 0.2]
        signals = []

        for t2 in t2_values:
            s = se.signal_se(
                m0=1000.0,
                t1=1.0,
                t2=t2,
                repetition_time=5.0,
                echo_time=0.1,
            )
            signals.append(float(s))

        # Longer T2 should give higher signal at fixed TE
        assert all(signals[i] < signals[i + 1] for i in range(len(signals) - 1))

    def test_signal_se_mixed_array_scalar(self) -> None:
        """Test SE with mixture of array and scalar inputs."""
        m0 = np.array([800.0, 1000.0, 1200.0])
        t1 = 1.0  # scalar
        t2 = np.array([0.08, 0.1, 0.12])

        signal = se.signal_se(
            m0=m0,
            t1=t1,
            t2=t2,
            repetition_time=3.0,
            echo_time=0.05,
        )

        assert signal.shape == (3,)
        assert np.all(signal > 0)

    def test_signal_se_known_values(self) -> None:
        """Test SE signal against analytically calculated values."""
        m0 = 1000.0
        t1 = 1.0
        t2 = 0.1
        tr = 2.0
        te = 0.05

        # Analytically: S = M0 * (1 - exp(-TR/T1)) * exp(-TE/T2)
        expected = m0 * (1 - np.exp(-tr / t1)) * np.exp(-te / t2)

        signal = se.signal_se(
            m0=m0,
            t1=t1,
            t2=t2,
            repetition_time=tr,
            echo_time=te,
        )

        assert_allclose(signal, expected, rtol=1e-10)
