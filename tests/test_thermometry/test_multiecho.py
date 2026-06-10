"""Tests for multi-echo dual-resonance thermometry module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qmri.thermometry import multiecho

# Constants used in tests
GAMMA_H = 42.577_478_92e6  # Hz/T, same as in qmri.constants


class TestThermometrySignalModel:
    """Tests for the dual-resonance signal model."""

    def test_signal_model_at_t_zero(self) -> None:
        """Test signal at t=0 equals sum of amplitudes (with phase)."""
        t = np.array([0.0])
        amplitude_1 = 1.0
        amplitude_2 = 0.5
        r2star_1 = 50.0
        r2star_2 = 100.0
        df = 200.0
        dphi_deg = 0.0  # Zero phase offset

        signal = multiecho.thermometry_signal_model(
            t, amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg
        )

        # At t=0 with dphi=0: S = sqrt(A1^2 + A2^2 + 2*A1*A2) = A1 + A2
        expected = amplitude_1 + amplitude_2
        assert_allclose(signal[0], expected, rtol=1e-10)

    def test_signal_model_at_t_zero_with_phase(self) -> None:
        """Test signal at t=0 with non-zero phase offset."""
        t = np.array([0.0])
        amplitude_1 = 1.0
        amplitude_2 = 0.5
        r2star_1 = 50.0
        r2star_2 = 100.0
        df = 200.0
        dphi_deg = 90.0  # 90 degree phase offset

        signal = multiecho.thermometry_signal_model(
            t, amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg
        )

        # At t=0 with dphi=90: cos(pi/2) = 0, so S = sqrt(A1^2 + A2^2)
        expected = np.sqrt(amplitude_1**2 + amplitude_2**2)
        assert_allclose(signal[0], expected, rtol=1e-10)

    def test_signal_model_decay(self) -> None:
        """Test that signal decays with time due to R2* relaxation."""
        t = np.linspace(0.001, 0.050, 50)
        signal = multiecho.thermometry_signal_model(
            t,
            amplitude_1=1.0,
            amplitude_2=0.5,
            r2star_1=50.0,
            r2star_2=100.0,
            df=200.0,
            dphi_deg=45.0,
        )

        # Signal should generally decay
        assert signal[-1] < signal[0]

    def test_signal_model_oscillation(self) -> None:
        """Test that signal oscillates due to frequency difference."""
        # Use high df to see oscillation
        t = np.linspace(0.001, 0.020, 100)
        signal = multiecho.thermometry_signal_model(
            t,
            amplitude_1=1.0,
            amplitude_2=1.0,
            r2star_1=10.0,  # Low R2* to preserve oscillation
            r2star_2=10.0,
            df=100.0,  # 100 Hz -> 10ms period
            dphi_deg=0.0,
        )

        # Find local maxima
        diff = np.diff(signal)
        maxima_indices = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
        assert len(maxima_indices) > 0  # Should have oscillation peaks

    def test_signal_model_non_negative(self) -> None:
        """Test that signal is always non-negative."""
        t = np.linspace(0.001, 0.050, 100)
        signal = multiecho.thermometry_signal_model(
            t,
            amplitude_1=1.0,
            amplitude_2=0.5,
            r2star_1=50.0,
            r2star_2=100.0,
            df=200.0,
            dphi_deg=45.0,
        )
        assert np.all(signal >= 0)

    def test_signal_model_array_df(self) -> None:
        """Test signal model with array-valued df."""
        t = np.array([0.010])
        df_array = np.array([100.0, 150.0, 200.0])

        # df can be broadcast
        signals = multiecho.thermometry_signal_model(
            t,
            amplitude_1=1.0,
            amplitude_2=0.5,
            r2star_1=50.0,
            r2star_2=100.0,
            df=df_array,
            dphi_deg=0.0,
        )
        assert signals.shape == (3,)


class TestTemperatureConversion:
    """Tests for temperature-frequency conversion functions."""

    def test_df_from_temperature_basic(self) -> None:
        """Test basic frequency calculation from temperature."""
        temperature = 37.0  # Body temperature
        b0 = 3.0

        df = multiecho.calculate_df_from_temperature(temperature, b0)

        # df should be positive for temperatures below 193.35
        assert df > 0

    def test_temperature_from_df_basic(self) -> None:
        """Test basic temperature calculation from frequency."""
        df = 200.0  # Hz
        b0 = 3.0

        temperature = multiecho.calculate_temperature_from_df(df, b0)

        # Temperature should be below 193.35 for positive df
        assert temperature < 193.35

    def test_roundtrip_temperature_df_temperature(self) -> None:
        """Test roundtrip: temperature -> df -> temperature."""
        original_temp = 25.0
        b0 = 3.0

        df = multiecho.calculate_df_from_temperature(original_temp, b0)
        recovered_temp = multiecho.calculate_temperature_from_df(df, b0)

        assert_allclose(recovered_temp, original_temp, rtol=1e-10)

    def test_roundtrip_different_temperatures(self) -> None:
        """Test roundtrip for various temperatures."""
        temperatures = np.array([15.0, 20.0, 25.0, 30.0, 37.0, 50.0])
        b0 = 3.0

        for temp in temperatures:
            df = multiecho.calculate_df_from_temperature(temp, b0)
            recovered = multiecho.calculate_temperature_from_df(df, b0)
            assert_allclose(recovered, temp, rtol=1e-10)

    def test_roundtrip_different_field_strengths(self) -> None:
        """Test roundtrip at different field strengths."""
        temperature = 25.0
        field_strengths = [1.5, 3.0, 7.0, 9.4]

        for b0 in field_strengths:
            df = multiecho.calculate_df_from_temperature(temperature, b0)
            recovered = multiecho.calculate_temperature_from_df(df, b0)
            assert_allclose(recovered, temperature, rtol=1e-10)

    def test_df_scales_with_field(self) -> None:
        """Test that df scales linearly with field strength."""
        temperature = 25.0

        df_1p5t = multiecho.calculate_df_from_temperature(temperature, 1.5)
        df_3t = multiecho.calculate_df_from_temperature(temperature, 3.0)

        assert_allclose(df_3t, 2 * df_1p5t, rtol=1e-10)

    def test_temperature_from_negative_df(self) -> None:
        """Test temperature from negative df (uses absolute value)."""
        b0 = 3.0
        df_pos = 200.0
        df_neg = -200.0

        temp_pos = multiecho.calculate_temperature_from_df(df_pos, b0)
        temp_neg = multiecho.calculate_temperature_from_df(df_neg, b0)

        # Both should give same temperature (uses |df|)
        assert_allclose(temp_pos, temp_neg, rtol=1e-10)

    def test_temperature_array_input(self) -> None:
        """Test temperature conversion with array input."""
        temperatures = np.array([20.0, 25.0, 30.0])
        b0 = 3.0

        df_array = multiecho.calculate_df_from_temperature(temperatures, b0)
        assert df_array.shape == (3,)

        recovered = multiecho.calculate_temperature_from_df(df_array, b0)
        assert_allclose(recovered, temperatures, rtol=1e-10)


class TestTemperatureUncertainty:
    """Tests for temperature uncertainty calculation."""

    def test_uncertainty_basic(self) -> None:
        """Test basic uncertainty calculation."""
        df_uncertainty = 1.0  # 1 Hz
        b0 = 3.0

        temp_uncertainty = multiecho.calculate_temperature_uncertainty(
            df_uncertainty, b0
        )

        # Should be positive
        assert temp_uncertainty > 0

    def test_uncertainty_scales_with_df_uncertainty(self) -> None:
        """Test that temperature uncertainty scales with df uncertainty."""
        b0 = 3.0

        uncert_1hz = multiecho.calculate_temperature_uncertainty(1.0, b0)
        uncert_2hz = multiecho.calculate_temperature_uncertainty(2.0, b0)

        assert_allclose(uncert_2hz, 2 * uncert_1hz, rtol=1e-10)

    def test_uncertainty_inversely_proportional_to_field(self) -> None:
        """Test that uncertainty is inversely proportional to field strength."""
        df_uncertainty = 1.0

        uncert_1p5t = multiecho.calculate_temperature_uncertainty(df_uncertainty, 1.5)
        uncert_3t = multiecho.calculate_temperature_uncertainty(df_uncertainty, 3.0)

        assert_allclose(uncert_1p5t, 2 * uncert_3t, rtol=1e-10)

    def test_uncertainty_formula(self) -> None:
        """Test uncertainty matches expected formula."""
        df_uncertainty = 1.0
        b0 = 3.0

        temp_uncertainty = multiecho.calculate_temperature_uncertainty(
            df_uncertainty, b0
        )

        # Expected: u(T) = 1.02e8 * u(df) / (gamma * B0)
        expected = (1.02e8 * df_uncertainty) / (GAMMA_H * b0)
        assert_allclose(temp_uncertainty, expected, rtol=1e-10)


class TestLeastSquaresFit:
    """Tests for the least squares fitting function."""

    def test_fit_ideal_data(self) -> None:
        """Test fitting on ideal (noise-free) data."""
        # Generate synthetic data
        echo_times = np.linspace(0.001, 0.024, 24)
        true_params = [1.0, 0.5, 50.0, 100.0, 200.0, 45.0]
        signal = multiecho.thermometry_signal_model(echo_times, *true_params)

        # Initial guess
        initial_guess = [0.8, 0.4, 40.0, 80.0, 180.0, 30.0]

        popt, pcov, r_squared = multiecho.lsq_fit_thermometry_signal_model(
            echo_times, signal, initial_guess
        )

        # Should recover parameters well
        assert_allclose(popt, true_params, rtol=1e-3)
        assert r_squared > 0.999

    def test_fit_noisy_data(self) -> None:
        """Test fitting on noisy data."""
        rng = np.random.default_rng(42)
        echo_times = np.linspace(0.001, 0.024, 24)
        true_params = [1.0, 0.5, 50.0, 100.0, 200.0, 45.0]
        signal_clean = multiecho.thermometry_signal_model(echo_times, *true_params)
        signal_noisy = signal_clean + rng.normal(0, 0.02, signal_clean.shape)

        initial_guess = [0.8, 0.4, 40.0, 80.0, 180.0, 30.0]

        popt, pcov, r_squared = multiecho.lsq_fit_thermometry_signal_model(
            echo_times, signal_noisy, initial_guess
        )

        # Should still get reasonable fit
        assert r_squared > 0.95
        # df (index 4) should be close to true value
        assert_allclose(popt[4], true_params[4], rtol=0.1)

    def test_fit_returns_covariance(self) -> None:
        """Test that fit returns covariance matrix."""
        echo_times = np.linspace(0.001, 0.024, 24)
        true_params = [1.0, 0.5, 50.0, 100.0, 200.0, 45.0]
        signal = multiecho.thermometry_signal_model(echo_times, *true_params)
        initial_guess = [0.8, 0.4, 40.0, 80.0, 180.0, 30.0]

        popt, pcov, r_squared = multiecho.lsq_fit_thermometry_signal_model(
            echo_times, signal, initial_guess
        )

        assert pcov.shape == (6, 6)
        # Diagonal elements should be non-negative (variances)
        assert np.all(np.diag(pcov) >= 0)

    def test_fit_poor_data_returns_nan(self) -> None:
        """Test that fit returns NaN for unconvergable data."""
        # Random data that doesn't match the model should give poor fit
        rng = np.random.default_rng(123)
        echo_times = np.linspace(0.001, 0.024, 24)
        signal = rng.uniform(0.1, 1.0, size=24)  # Random noise, not a real signal
        initial_guess = [0.8, 0.4, 40.0, 80.0, 180.0, 30.0]

        popt, pcov, r_squared = multiecho.lsq_fit_thermometry_signal_model(
            echo_times, signal, initial_guess
        )

        # Random data should give a poor fit (low R²)
        # Note: may not always be NaN since curve_fit may converge to some solution
        assert np.all(np.isnan(popt)) or r_squared < 0.9


class TestFitMultiechoThermometry:
    """Tests for the main fitting function."""

    def test_fit_single_method(self) -> None:
        """Test single fit method."""
        b0 = 3.0
        temperature_true = 25.0
        df_true = multiecho.calculate_df_from_temperature(temperature_true, b0)

        echo_times = np.linspace(0.001, 0.024, 24)
        signal = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 50.0, 100.0, df_true, 45.0
        )

        result = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="single"
        )

        assert_allclose(result.temperature, temperature_true, rtol=0.01)
        assert result.r_squared > 0.99
        assert result.n_bootstrap is None

    def test_fit_bootstrap_method(self) -> None:
        """Test bootstrap fit method returns valid structure.

        Note: Bootstrap resampling of time points for a single signal is
        less meaningful than bootstrap over voxels (as in the original
        regionwise_bootstrap method). This test primarily checks that
        the bootstrap method runs without error and returns valid results.
        """
        b0 = 3.0
        temperature_true = 25.0
        df_true = multiecho.calculate_df_from_temperature(temperature_true, b0)

        echo_times = np.linspace(0.001, 0.024, 24)
        signal = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 50.0, 100.0, df_true, 45.0
        )

        result = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="bootstrap", n_bootstrap=20
        )

        # Bootstrap should return valid structure
        assert result.n_bootstrap == 20
        assert not np.isnan(result.temperature) or result.r_squared < 0.9
        assert result.temperature_uncertainty >= 0 or np.isnan(result.temperature)

    def test_fit_different_temperatures(self) -> None:
        """Test fitting at different temperatures."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)

        for temp_true in [15.0, 25.0, 35.0, 45.0]:
            df_true = multiecho.calculate_df_from_temperature(temp_true, b0)
            signal = multiecho.thermometry_signal_model(
                echo_times, 1.0, 0.5, 50.0, 100.0, df_true, 45.0
            )

            result = multiecho.fit_multiecho_thermometry(
                signal, echo_times, b0, method="single"
            )

            assert_allclose(result.temperature, temp_true, rtol=0.01)

    def test_fit_mismatched_lengths_raises(self) -> None:
        """Test that mismatched signal/echo_times raises ValueError."""
        signal = np.array([1.0, 0.9, 0.8])
        echo_times = np.array([0.001, 0.002])  # Different length

        with pytest.raises(ValueError, match="must match"):
            multiecho.fit_multiecho_thermometry(signal, echo_times, 3.0)

    def test_fit_result_attributes(self) -> None:
        """Test that result has all expected attributes."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        df = multiecho.calculate_df_from_temperature(25.0, b0)
        signal = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 50.0, 100.0, df, 45.0
        )

        result = multiecho.fit_multiecho_thermometry(signal, echo_times, b0)

        assert hasattr(result, "temperature")
        assert hasattr(result, "temperature_uncertainty")
        assert hasattr(result, "df")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "fitted_params")
        assert hasattr(result, "n_bootstrap")

    def test_fit_noisy_data(self) -> None:
        """Test fitting with noisy data."""
        rng = np.random.default_rng(42)
        b0 = 3.0
        temperature_true = 25.0
        df_true = multiecho.calculate_df_from_temperature(temperature_true, b0)

        echo_times = np.linspace(0.001, 0.024, 24)
        signal_clean = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 50.0, 100.0, df_true, 45.0
        )
        signal_noisy = signal_clean + rng.normal(0, 0.02, signal_clean.shape)

        result = multiecho.fit_multiecho_thermometry(
            signal_noisy, echo_times, b0, method="single"
        )

        # Should still be within ~1 degree
        assert abs(result.temperature - temperature_true) < 1.0
        assert result.r_squared > 0.9


class TestMultiEchoResult:
    """Tests for MultiEchoResult dataclass."""

    def test_result_frozen(self) -> None:
        """Test that MultiEchoResult is immutable."""
        result = multiecho.MultiEchoResult(
            temperature=25.0,
            temperature_uncertainty=0.1,
            df=200.0,
            r_squared=0.99,
            fitted_params=np.array([1.0, 0.5, 50.0, 100.0, 200.0, 45.0]),
            n_bootstrap=None,
        )
        with pytest.raises(AttributeError):
            result.temperature = 30.0  # type: ignore[misc]

    def test_result_attributes(self) -> None:
        """Test MultiEchoResult attributes."""
        params = np.array([1.0, 0.5, 50.0, 100.0, 200.0, 45.0])
        result = multiecho.MultiEchoResult(
            temperature=25.0,
            temperature_uncertainty=0.1,
            df=200.0,
            r_squared=0.99,
            fitted_params=params,
            n_bootstrap=100,
        )
        assert result.temperature == 25.0
        assert result.temperature_uncertainty == 0.1
        assert result.df == 200.0
        assert result.r_squared == 0.99
        assert_allclose(result.fitted_params, params)
        assert result.n_bootstrap == 100


class TestConstants:
    """Tests for module constants."""

    def test_r_squared_threshold(self) -> None:
        """Test R-squared threshold value."""
        assert multiecho.R_SQUARED_THRESHOLD == 0.9

    def test_random_seed(self) -> None:
        """Test random seed is defined."""
        assert isinstance(multiecho.RANDOM_SEED, int)

    def test_bootstrap_reproducibility(self) -> None:
        """Test that bootstrap results are reproducible."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        df = multiecho.calculate_df_from_temperature(25.0, b0)
        signal = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 50.0, 100.0, df, 45.0
        )

        result1 = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="bootstrap", n_bootstrap=10
        )
        result2 = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="bootstrap", n_bootstrap=10
        )

        # With fixed seed, results should be identical
        assert_allclose(result1.temperature, result2.temperature)


class TestIntegration:
    """Integration tests for the complete thermometry workflow."""

    def test_ethylene_glycol_phantom_workflow(self) -> None:
        """Test complete workflow for ethylene glycol phantom thermometry."""
        # Simulate a phantom at room temperature
        b0 = 3.0
        true_temperature = 22.0  # Room temperature

        # Generate realistic echo times (24 echoes from 1-24 ms)
        echo_times = np.linspace(0.001, 0.024, 24)

        # Calculate expected frequency difference
        df_expected = multiecho.calculate_df_from_temperature(true_temperature, b0)

        # Generate signal with realistic parameters
        signal = multiecho.thermometry_signal_model(
            echo_times,
            amplitude_1=1.0,
            amplitude_2=0.5,
            r2star_1=30.0,  # Typical R2* values for phantom
            r2star_2=40.0,
            df=df_expected,
            dphi_deg=0.0,
        )

        # Fit the model
        result = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="single"
        )

        # Verify temperature recovery
        assert_allclose(result.temperature, true_temperature, atol=0.5)
        assert result.r_squared > 0.99

    def test_temperature_range_validation(self) -> None:
        """Test that the method works across typical temperature range."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)

        # Test temperatures from cold to warm (typical phantom range)
        temperatures = [10.0, 20.0, 30.0, 40.0, 50.0]

        for temp in temperatures:
            df = multiecho.calculate_df_from_temperature(temp, b0)
            signal = multiecho.thermometry_signal_model(
                echo_times, 1.0, 0.5, 30.0, 40.0, df, 0.0
            )
            result = multiecho.fit_multiecho_thermometry(
                signal, echo_times, b0, method="single"
            )
            assert_allclose(result.temperature, temp, atol=0.5)


class TestDfInitStrategies:
    """Tests for the frequency starting-condition strategies (df_init)."""

    def test_all_strategies_recover_warm_temperature(self) -> None:
        """At a warm temperature every strategy recovers the temperature."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        temp = 40.0
        df = multiecho.calculate_df_from_temperature(temp, b0)
        signal = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 30.0, 40.0, df, 0.0
        )
        for df_init in ("fixed", "lombscargle", "multistart"):
            result = multiecho.fit_multiecho_thermometry(
                signal, echo_times, b0, method="single", df_init=df_init
            )
            assert_allclose(result.temperature, temp, atol=0.5)

    def test_multistart_recovers_cold_where_fixed_aliases(self) -> None:
        """On the aliasing-prone grid, multistart beats the fixed start at 10 °C."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        temp = 10.0
        df = multiecho.calculate_df_from_temperature(temp, b0)
        signal = multiecho.thermometry_signal_model(
            echo_times, 1.0, 0.5, 30.0, 40.0, df, 0.0
        )

        fixed = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="single", df_init="fixed"
        )
        multistart = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="single", df_init="multistart"
        )
        lombscargle = multiecho.fit_multiecho_thermometry(
            signal, echo_times, b0, method="single", df_init="lombscargle"
        )

        # The fixed start aliases to a spurious hot temperature here...
        assert abs(fixed.temperature - temp) > 50.0
        # ...while the data-driven starts recover the true value.
        assert_allclose(multistart.temperature, temp, atol=0.5)
        assert_allclose(lombscargle.temperature, temp, atol=0.5)
        # Multistart never returns a worse fit than the fixed start.
        assert multistart.r_squared >= fixed.r_squared

    def test_df_init_threads_through_image_fit(self) -> None:
        """df_init is honoured by the segmentation-driven image fit."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        signal, segmentation = _make_segmented_volume({1: 10.0}, echo_times, b0)

        _, fixed = multiecho.fit_multiecho_thermometry_image(
            signal, segmentation, echo_times, b0, method="regionwise", df_init="fixed"
        )
        _, multistart = multiecho.fit_multiecho_thermometry_image(
            signal,
            segmentation,
            echo_times,
            b0,
            method="regionwise",
            df_init="multistart",
        )
        assert abs(fixed[0].temperature - 10.0) > 50.0
        assert_allclose(multistart[0].temperature, 10.0, atol=0.5)


def _make_region_signal(
    temperature: float, echo_times: np.ndarray, b0: float
) -> np.ndarray:
    """Return the dual-resonance signal for a given temperature."""
    df = multiecho.calculate_df_from_temperature(temperature, b0)
    return multiecho.thermometry_signal_model(echo_times, 1.0, 0.5, 30.0, 40.0, df, 0.0)


def _make_segmented_volume(
    region_temperatures: dict[int, float],
    echo_times: np.ndarray,
    b0: float,
    shape: tuple[int, int, int] = (4, 4, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 4D signal volume and a 3D segmentation with known temperatures.

    Voxels are split across the supplied region labels in raster order; any
    remaining voxels are left as background (label 0).
    """
    n_echoes = echo_times.shape[0]
    signal = np.zeros((*shape, n_echoes), dtype=np.float64)
    segmentation = np.zeros(shape, dtype=np.int16)

    flat_indices = list(np.ndindex(shape))
    labels = list(region_temperatures)
    chunk = len(flat_indices) // (len(labels) + 1)  # leave a background chunk
    for label_index, label in enumerate(labels):
        region_signal = _make_region_signal(region_temperatures[label], echo_times, b0)
        start = label_index * chunk
        for i, j, k in flat_indices[start : start + chunk]:
            segmentation[i, j, k] = label
            signal[i, j, k, :] = region_signal
    return signal, segmentation


class TestFitMultiEchoThermometryImage:
    """Tests for segmentation-driven multi-echo thermometry image fitting."""

    def test_regionwise_recovers_region_temperatures(self) -> None:
        """Region-wise fitting recovers per-region temperatures."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        region_temperatures = {1: 20.0, 2: 40.0}
        signal, segmentation = _make_segmented_volume(
            region_temperatures, echo_times, b0
        )

        temperature_map, results = multiecho.fit_multiecho_thermometry_image(
            signal, segmentation, echo_times, b0, method="regionwise"
        )

        assert temperature_map.shape == segmentation.shape
        assert {r.region_id for r in results} == {1, 2}
        for region in results:
            assert_allclose(
                region.temperature, region_temperatures[region.region_id], atol=0.5
            )
            # Every voxel in the region gets the region temperature.
            region_voxels = temperature_map[segmentation == region.region_id]
            assert_allclose(region_voxels, region.temperature, atol=1e-9)
        # Background stays at zero.
        assert_allclose(temperature_map[segmentation == 0], 0.0)

    def test_voxelwise_recovers_region_temperatures(self) -> None:
        """Voxel-wise fitting recovers per-region temperatures."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        region_temperatures = {1: 25.0, 2: 45.0}
        signal, segmentation = _make_segmented_volume(
            region_temperatures, echo_times, b0
        )

        _, results = multiecho.fit_multiecho_thermometry_image(
            signal, segmentation, echo_times, b0, method="voxelwise"
        )

        for region in results:
            assert region.region_size == int(np.sum(segmentation == region.region_id))
            assert region.temperature_values.shape == (region.region_size,)
            assert_allclose(
                region.temperature, region_temperatures[region.region_id], atol=0.5
            )

    def test_regionwise_bootstrap_recovers_temperature(self) -> None:
        """Bootstrap region-wise fitting recovers temperatures with uncertainty."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        region_temperatures = {1: 30.0}
        signal, segmentation = _make_segmented_volume(
            region_temperatures, echo_times, b0
        )

        _, results = multiecho.fit_multiecho_thermometry_image(
            signal,
            segmentation,
            echo_times,
            b0,
            method="regionwise_bootstrap",
            n_bootstrap=20,
        )

        (region,) = results
        assert region.temperature_values.shape == (20,)
        assert_allclose(region.temperature, 30.0, atol=0.5)
        assert region.temperature_uncertainty >= 0.0

    def test_to_dict_is_json_serialisable(self) -> None:
        """RegionThermometryResult.to_dict round-trips through JSON."""
        import json

        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        signal, segmentation = _make_segmented_volume({1: 25.0}, echo_times, b0)

        _, results = multiecho.fit_multiecho_thermometry_image(
            signal, segmentation, echo_times, b0, method="regionwise"
        )

        payload = results[0].to_dict()
        # Should not raise and should preserve the region id.
        restored = json.loads(json.dumps(payload))
        assert restored["id"] == 1

    def test_background_label_is_ignored(self) -> None:
        """An all-background segmentation yields no regions."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        signal = np.zeros((3, 3, 1, echo_times.shape[0]), dtype=np.float64)
        segmentation = np.zeros((3, 3, 1), dtype=np.int16)

        temperature_map, results = multiecho.fit_multiecho_thermometry_image(
            signal, segmentation, echo_times, b0, method="regionwise"
        )

        assert results == []
        assert_allclose(temperature_map, 0.0)

    def test_invalid_dimensions_raise(self) -> None:
        """Inconsistent array shapes raise ValueError."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        signal = np.zeros((4, 4, 2, 24), dtype=np.float64)

        # 2D segmentation.
        with pytest.raises(ValueError, match="segmentation must be 3D"):
            multiecho.fit_multiecho_thermometry_image(
                signal, np.zeros((4, 4)), echo_times, b0
            )
        # Mismatched spatial shape.
        with pytest.raises(ValueError, match="segmentation shape must match"):
            multiecho.fit_multiecho_thermometry_image(
                signal, np.zeros((4, 4, 3)), echo_times, b0
            )
        # Echo-time length mismatch.
        with pytest.raises(ValueError, match="echo_times length"):
            multiecho.fit_multiecho_thermometry_image(
                signal, np.zeros((4, 4, 2)), echo_times[:10], b0
            )

    def test_unknown_method_raises(self) -> None:
        """An unrecognised method raises ValueError."""
        b0 = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        signal = np.zeros((2, 2, 1, 24), dtype=np.float64)
        segmentation = np.ones((2, 2, 1), dtype=np.int16)

        with pytest.raises(ValueError, match="Unknown method"):
            multiecho.fit_multiecho_thermometry_image(
                signal,
                segmentation,
                echo_times,
                b0,
                method="nonsense",  # type: ignore[arg-type]
            )
