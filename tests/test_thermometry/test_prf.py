"""Tests for PRF thermometry module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.thermometry import prf


class TestSignalPhaseShift:
    """Tests for PRF phase shift calculation."""

    def test_phase_shift_basic(self) -> None:
        """Test basic phase shift calculation."""
        # 10°C temperature change at 3T with 20ms TE
        phase_shift = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        # Phase shift should be negative for positive temperature change
        # (due to negative PRF coefficient)
        assert phase_shift < 0

    def test_phase_shift_zero_temperature(self) -> None:
        """Test that zero temperature change gives zero phase shift."""
        phase_shift = prf.signal_phase_shift(
            temperature_change=0.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert_allclose(phase_shift, 0.0)

    def test_phase_shift_proportional_to_temperature(self) -> None:
        """Test that phase shift scales linearly with temperature."""
        phase_shift_1 = prf.signal_phase_shift(
            temperature_change=5.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        phase_shift_2 = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert_allclose(phase_shift_2, 2 * phase_shift_1)

    def test_phase_shift_proportional_to_echo_time(self) -> None:
        """Test that phase shift scales linearly with echo time."""
        phase_shift_1 = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.010,
            magnetic_field=3.0,
        )
        phase_shift_2 = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert_allclose(phase_shift_2, 2 * phase_shift_1)

    def test_phase_shift_proportional_to_field(self) -> None:
        """Test that phase shift scales linearly with field strength."""
        phase_shift_15t = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=1.5,
        )
        phase_shift_3t = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert_allclose(phase_shift_3t, 2 * phase_shift_15t)

    def test_phase_shift_array_input(self) -> None:
        """Test phase shift with array inputs."""
        temperatures = np.array([5.0, 10.0, 15.0])
        phase_shifts = prf.signal_phase_shift(
            temperature_change=temperatures,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert phase_shifts.shape == (3,)
        # All should be negative (heating causes negative phase shift)
        assert np.all(phase_shifts < 0)

    def test_phase_shift_negative_temperature(self) -> None:
        """Test phase shift for cooling (negative temperature change)."""
        phase_shift = prf.signal_phase_shift(
            temperature_change=-5.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        # Cooling should give positive phase shift
        assert phase_shift > 0

    def test_phase_shift_custom_coefficient(self) -> None:
        """Test phase shift with custom PRF coefficient."""
        # Use a coefficient twice the standard value
        custom_coef = 2 * prf.PRF_THERMAL_COEFFICIENT
        phase_shift_standard = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        phase_shift_custom = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,
            magnetic_field=3.0,
            prf_coefficient=custom_coef,
        )
        assert_allclose(phase_shift_custom, 2 * phase_shift_standard)


class TestCalculateTemperature:
    """Tests for temperature calculation from phase difference."""

    def test_calculate_temperature_basic(self) -> None:
        """Test basic temperature calculation."""
        # First generate a phase shift for known temperature change
        true_delta_t = 10.0
        te = 0.020
        b0 = 3.0
        phase_diff = prf.signal_phase_shift(
            temperature_change=true_delta_t,
            echo_time=te,
            magnetic_field=b0,
        )

        # Now recover the temperature from the phase difference
        result = prf.calculate_temperature(
            phase_difference=phase_diff,
            echo_time=te,
            magnetic_field=b0,
        )

        assert_allclose(result.temperature_change, true_delta_t, rtol=1e-10)

    def test_calculate_temperature_zero_phase(self) -> None:
        """Test that zero phase difference gives zero temperature change."""
        result = prf.calculate_temperature(
            phase_difference=0.0,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert_allclose(result.temperature_change, 0.0)

    def test_calculate_temperature_negative_change(self) -> None:
        """Test temperature calculation for cooling."""
        true_delta_t = -5.0  # Cooling
        te = 0.020
        b0 = 3.0
        phase_diff = prf.signal_phase_shift(
            temperature_change=true_delta_t,
            echo_time=te,
            magnetic_field=b0,
        )

        result = prf.calculate_temperature(
            phase_difference=phase_diff,
            echo_time=te,
            magnetic_field=b0,
        )

        assert_allclose(result.temperature_change, true_delta_t, rtol=1e-10)

    def test_calculate_temperature_array_input(self) -> None:
        """Test temperature calculation with array inputs."""
        true_temps = np.array([5.0, 10.0, 15.0])
        te = 0.020
        b0 = 3.0

        phase_diffs = prf.signal_phase_shift(
            temperature_change=true_temps,
            echo_time=te,
            magnetic_field=b0,
        )

        result = prf.calculate_temperature(
            phase_difference=phase_diffs,
            echo_time=te,
            magnetic_field=b0,
        )

        assert result.temperature_change.shape == (3,)
        assert_allclose(result.temperature_change, true_temps, rtol=1e-10)

    def test_calculate_temperature_volume(self) -> None:
        """Test temperature calculation on a 3D volume."""
        shape = (4, 4, 4)
        te = 0.020
        b0 = 3.0

        # Create a temperature map with varying values
        true_temps = np.linspace(0, 20, np.prod(shape)).reshape(shape)

        phase_diffs = prf.signal_phase_shift(
            temperature_change=true_temps,
            echo_time=te,
            magnetic_field=b0,
        )

        result = prf.calculate_temperature(
            phase_difference=phase_diffs,
            echo_time=te,
            magnetic_field=b0,
        )

        assert result.temperature_change.shape == shape
        assert_allclose(result.temperature_change, true_temps, rtol=1e-10)

    def test_calculate_temperature_different_field_strengths(self) -> None:
        """Test temperature calculation at different field strengths."""
        true_delta_t = 10.0
        te = 0.020

        for b0 in [1.5, 3.0, 7.0]:
            phase_diff = prf.signal_phase_shift(
                temperature_change=true_delta_t,
                echo_time=te,
                magnetic_field=b0,
            )
            result = prf.calculate_temperature(
                phase_difference=phase_diff,
                echo_time=te,
                magnetic_field=b0,
            )
            assert_allclose(result.temperature_change, true_delta_t, rtol=1e-10)

    def test_calculate_temperature_different_echo_times(self) -> None:
        """Test temperature calculation at different echo times."""
        true_delta_t = 10.0
        b0 = 3.0

        for te in [0.010, 0.020, 0.030, 0.040]:
            phase_diff = prf.signal_phase_shift(
                temperature_change=true_delta_t,
                echo_time=te,
                magnetic_field=b0,
            )
            result = prf.calculate_temperature(
                phase_difference=phase_diff,
                echo_time=te,
                magnetic_field=b0,
            )
            assert_allclose(result.temperature_change, true_delta_t, rtol=1e-10)

    def test_result_contains_phase_difference(self) -> None:
        """Test that PRFResult contains the phase difference."""
        phase_diff = -0.5
        result = prf.calculate_temperature(
            phase_difference=phase_diff,
            echo_time=0.020,
            magnetic_field=3.0,
        )
        assert_allclose(result.phase_difference, phase_diff)

    def test_roundtrip_phase_to_temperature_to_phase(self) -> None:
        """Test roundtrip: phase -> temperature -> phase."""
        original_phase = -0.3
        te = 0.020
        b0 = 3.0

        # Phase to temperature
        result = prf.calculate_temperature(
            phase_difference=original_phase,
            echo_time=te,
            magnetic_field=b0,
        )

        # Temperature back to phase
        recovered_phase = prf.signal_phase_shift(
            temperature_change=result.temperature_change,
            echo_time=te,
            magnetic_field=b0,
        )

        assert_allclose(recovered_phase, original_phase, rtol=1e-10)


class TestPRFResult:
    """Tests for PRFResult dataclass."""

    def test_prf_result_frozen(self) -> None:
        """Test that PRFResult is immutable."""
        result = prf.PRFResult(
            temperature_change=np.array(10.0),
            phase_difference=np.array(-0.16),
        )
        with pytest.raises(AttributeError):
            result.temperature_change = np.array(5.0)  # type: ignore[misc]

    def test_prf_result_attributes(self) -> None:
        """Test PRFResult attributes."""
        temp_change = np.array([5.0, 10.0])
        phase_diff = np.array([-0.08, -0.16])
        result = prf.PRFResult(
            temperature_change=temp_change,
            phase_difference=phase_diff,
        )
        assert_allclose(result.temperature_change, temp_change)
        assert_allclose(result.phase_difference, phase_diff)


class TestPRFThermalCoefficient:
    """Tests for PRF thermal coefficient constant."""

    def test_prf_coefficient_value(self) -> None:
        """Test that PRF coefficient has expected value."""
        # Standard value is approximately -0.01 ppm/°C
        assert_allclose(prf.PRF_THERMAL_COEFFICIENT, -0.01e-6, rtol=0.01)

    def test_prf_coefficient_negative(self) -> None:
        """Test that PRF coefficient is negative."""
        # This is physically important: increasing temperature decreases
        # the resonance frequency of water
        assert prf.PRF_THERMAL_COEFFICIENT < 0


class TestPhysicallyRealisticValues:
    """Tests using physically realistic parameter values."""

    def test_typical_hifu_ablation(self) -> None:
        """Test phase shifts typical of HIFU ablation monitoring.

        During HIFU ablation, temperatures can reach 60-80°C from a
        baseline of ~37°C, giving temperature changes of 20-40°C.

        Note: Large temperature changes at 3T with typical echo times
        can produce phase shifts exceeding pi radians, requiring
        phase unwrapping in practice.
        """
        baseline_temp = 37.0
        ablation_temp = 70.0
        delta_t = ablation_temp - baseline_temp

        # Use shorter TE typical of HIFU monitoring to avoid phase wrapping
        te = 0.010  # 10 ms (shorter TE used in HIFU to avoid wrapping)
        b0 = 3.0

        phase_shift = prf.signal_phase_shift(
            temperature_change=delta_t,
            echo_time=te,
            magnetic_field=b0,
        )

        # Phase shift at shorter TE should be substantial
        # For delta_T=33°C, TE=10ms, B0=3T: expect ~2.6 rad
        assert np.abs(phase_shift) > 1.0  # Substantial phase shift
        assert np.abs(phase_shift) < np.pi  # But not wrapped with shorter TE

        # Verify roundtrip
        result = prf.calculate_temperature(
            phase_difference=phase_shift,
            echo_time=te,
            magnetic_field=b0,
        )
        assert_allclose(result.temperature_change, delta_t, rtol=1e-10)

    def test_typical_hyperthermia(self) -> None:
        """Test phase shifts typical of mild hyperthermia.

        Hyperthermia treatments typically aim for 41-43°C from a
        baseline of ~37°C, giving temperature changes of 4-6°C.
        """
        delta_t = 5.0  # Typical hyperthermia range

        te = 0.020  # 20 ms
        b0 = 1.5  # Common clinical field strength

        phase_shift = prf.signal_phase_shift(
            temperature_change=delta_t,
            echo_time=te,
            magnetic_field=b0,
        )

        # Expected phase shift should be relatively small
        assert np.abs(phase_shift) < 0.5  # Less than ~30 degrees

        # Verify roundtrip
        result = prf.calculate_temperature(
            phase_difference=phase_shift,
            echo_time=te,
            magnetic_field=b0,
        )
        assert_allclose(result.temperature_change, delta_t, rtol=1e-10)
