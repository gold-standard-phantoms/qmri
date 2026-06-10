"""Tests for GKM signal model module."""

import numpy as np
from qmri.perfusion import gkm


class TestSignalGKM:
    """Tests for full GKM signal model."""

    def test_signal_gkm_pcasl_arrived(self) -> None:
        """Test pCASL GKM signal after bolus arrival."""
        result = gkm.signal_gkm(
            perfusion_rate=60.0,  # ml/100g/min
            transit_time=1.0,  # s
            m0_tissue=1000.0,
            label_duration=1.8,
            signal_time=3.6,  # PLD = 1.8s, bolus arrived
            label_efficiency=0.85,
            partition_coefficient=0.9,
            t1_blood=1.65,
            t1_tissue=1.3,
            label_type="pcasl",
        )

        # Should have positive delta_m
        assert result.delta_m > 0

    def test_signal_gkm_pcasl_not_arrived(self) -> None:
        """Test pCASL GKM signal before bolus arrival."""
        result = gkm.signal_gkm(
            perfusion_rate=60.0,
            transit_time=2.0,  # Bolus hasn't arrived yet
            m0_tissue=1000.0,
            label_duration=1.8,
            signal_time=1.5,  # Before ATT
            label_efficiency=0.85,
            partition_coefficient=0.9,
            t1_blood=1.65,
            t1_tissue=1.3,
            label_type="pcasl",
        )

        # Should be zero before arrival
        assert result.delta_m == 0.0

    def test_signal_gkm_pasl_arrived(self) -> None:
        """Test PASL GKM signal after bolus arrival."""
        result = gkm.signal_gkm(
            perfusion_rate=60.0,
            transit_time=0.8,
            m0_tissue=1000.0,
            label_duration=0.7,  # PASL bolus duration
            signal_time=1.8,  # TI
            label_efficiency=0.98,
            partition_coefficient=0.9,
            t1_blood=1.65,
            t1_tissue=1.3,
            label_type="pasl",
        )

        # Should have positive delta_m
        assert result.delta_m > 0

    def test_signal_gkm_volume(self) -> None:
        """Test GKM on a 3D volume."""
        shape = (3, 3, 3)
        f = np.ones(shape) * 60.0
        att = np.ones(shape) * 1.0
        m0 = np.ones(shape) * 1000.0
        lam = np.ones(shape) * 0.9
        t1_t = np.ones(shape) * 1.3

        result = gkm.signal_gkm(
            perfusion_rate=f,
            transit_time=att,
            m0_tissue=m0,
            label_duration=1.8,
            signal_time=3.6,
            label_efficiency=0.85,
            partition_coefficient=lam,
            t1_blood=1.65,
            t1_tissue=t1_t,
            label_type="pcasl",
        )

        assert result.delta_m.shape == shape
        assert np.all(result.delta_m > 0)

    def test_signal_gkm_varying_att(self) -> None:
        """Test GKM with varying transit times."""
        shape = (5,)
        f = np.ones(shape) * 60.0
        att = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # Varying ATT
        m0 = np.ones(shape) * 1000.0

        result = gkm.signal_gkm(
            perfusion_rate=f,
            transit_time=att,
            m0_tissue=m0,
            label_duration=1.8,
            signal_time=3.0,
            label_efficiency=0.85,
            partition_coefficient=0.9,
            t1_blood=1.65,
            t1_tissue=1.3,
            label_type="pcasl",
        )

        # Higher ATT should give lower signal (more T1 decay)
        # But some may not have arrived yet
        assert result.delta_m.shape == shape

    def test_signal_gkm_invalid_label_type(self) -> None:
        """Test GKM raises error for invalid label type."""
        try:
            gkm.signal_gkm(
                perfusion_rate=60.0,
                transit_time=1.0,
                m0_tissue=1000.0,
                label_duration=1.8,
                signal_time=3.6,
                label_efficiency=0.85,
                partition_coefficient=0.9,
                t1_blood=1.65,
                t1_tissue=1.3,
                label_type="invalid",  # type: ignore[arg-type]
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Invalid label_type" in str(e)


class TestSignalGKMSimplified:
    """Tests for simplified GKM signal model."""

    def test_simplified_pcasl(self) -> None:
        """Test simplified pCASL GKM."""
        result = gkm.signal_gkm_simplified(
            perfusion_rate=60.0,
            transit_time=1.0,
            m0_tissue=1000.0,
            label_duration=1.8,
            signal_time=3.6,
            label_efficiency=0.85,
            partition_coefficient=0.9,
            t1_blood=1.65,
            label_type="pcasl",
        )

        assert result.delta_m > 0

    def test_simplified_pasl(self) -> None:
        """Test simplified PASL GKM."""
        result = gkm.signal_gkm_simplified(
            perfusion_rate=60.0,
            transit_time=0.8,
            m0_tissue=1000.0,
            label_duration=0.7,
            signal_time=1.8,
            label_efficiency=0.98,
            partition_coefficient=0.9,
            t1_blood=1.65,
            label_type="pasl",
        )

        assert result.delta_m > 0

    def test_simplified_not_arrived(self) -> None:
        """Test simplified GKM returns zero before arrival."""
        result = gkm.signal_gkm_simplified(
            perfusion_rate=60.0,
            transit_time=2.0,  # Late arrival
            m0_tissue=1000.0,
            label_duration=1.8,
            signal_time=2.0,  # Signal before ATT + tau
            label_efficiency=0.85,
            partition_coefficient=0.9,
            t1_blood=1.65,
            label_type="pcasl",
        )

        # Should be zero before full arrival
        assert result.delta_m == 0.0


class TestGKMResult:
    """Tests for GKMResult dataclass."""

    def test_gkm_result_frozen(self) -> None:
        """Test that GKMResult is immutable."""
        result = gkm.GKMResult(delta_m=np.array([10.0]))
        try:
            result.delta_m = np.array([20.0])  # type: ignore[misc]
            raise AssertionError("Should have raised AttributeError")
        except AttributeError:
            pass  # Expected
