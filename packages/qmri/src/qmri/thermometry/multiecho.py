r"""Multi-echo dual-resonance thermometry for ethylene glycol phantoms.

This module provides functions for MR thermometry using the dual-resonance
model, designed for ethylene glycol phantom calibration and validation.

Theory:
    The dual-resonance model describes the multi-echo magnitude signal from
    a sample containing two chemical species with different resonance frequencies
    (e.g., the CH₂ and OH groups in ethylene glycol).

    The signal model is:

    $$S(t) = \sqrt{A_1^2 e^{-2R_{2,1}^* t} + A_2^2 e^{-2R_{2,2}^* t}
           + 2 A_1 A_2 e^{-(R_{2,1}^* + R_{2,2}^*) t}
           \cos(2\pi \Delta f_{12} t + \Delta\phi_{12})}$$

    where:

    - $A_1, A_2$ are the amplitudes of the two resonance peaks
    - $R_{2,1}^*, R_{2,2}^*$ are the R2* relaxation rates (1/s)
    - $\Delta f_{12}$ is the frequency difference between the peaks (Hz)
    - $\Delta\phi_{12}$ is the initial phase difference (radians)

    For ethylene glycol, the frequency difference is related to temperature by:

    $$T[°C] = 193.35 - 1.02 \times 10^8 \cdot \frac{|\Delta f_{12}|}{\gamma B_0}$$

References:
    .. [1] Sprinkhuizen, S.M., Bakker, C.J.G. and Bartels, L.W. (2010),
           Absolute MR thermometry using time-domain analysis of multi-gradient-echo
           magnitude images. Magn. Reson. Med., 64: 239-248.
           https://doi.org/10.1002/mrm.22429

    .. [2] Raiford, D.S., Fisk, C.L. and Becker, E.D. (1979), Calibration of
           methanol and ethylene glycol nuclear magnetic resonance thermometers.
           Analytical Chemistry 51(12): 2050-2051.
           https://doi.org/10.1021/ac50048a040
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from qmri.constants import GAMMA_H
from qmri.errors.metrics import r_squared as calculate_r_squared
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

__all__ = [
    "R_SQUARED_THRESHOLD",
    "RANDOM_SEED",
    "DfInitMethod",
    "MultiEchoResult",
    "RegionAnalysisMethod",
    "RegionThermometryResult",
    "thermometry_signal_model",
    "calculate_df_from_temperature",
    "calculate_temperature_from_df",
    "calculate_temperature_uncertainty",
    "lsq_fit_thermometry_signal_model",
    "fit_multiecho_thermometry",
    "fit_multiecho_thermometry_image",
]

# Constants
R_SQUARED_THRESHOLD: float = 0.9
"""Default threshold for acceptable fit quality (R² > 0.9)."""

RANDOM_SEED: int = 840275920
"""Fixed seed for reproducibility of bootstrap sampling."""

_DF_UPPER_BOUND: float = 1000.0
"""Upper bound on the fitted frequency difference (Hz)."""

_DEFAULT_DF_GUESS: float = 100.0
"""Fallback starting value for the frequency difference (Hz)."""

DfInitMethod = Literal["multistart", "fixed", "lombscargle"]
"""Strategy for choosing the frequency-difference starting value of the fit.

- ``"multistart"`` (default): fit from both the fixed default and the
  data-driven Lomb-Scargle estimate, and keep the highest-R² result. Most
  robust against frequency aliasing.
- ``"fixed"``: a single fit from the fixed default starting value
  (:data:`_DEFAULT_DF_GUESS`). Cheapest; can alias on cold phantoms.
- ``"lombscargle"``: a single fit seeded from the Lomb-Scargle estimate,
  falling back to the fixed default when no estimate can be made.
"""


def thermometry_signal_model(
    t: NDArray[np.floating],
    amplitude_1: float,
    amplitude_2: float,
    r2star_1: float,
    r2star_2: float,
    df: float | NDArray[np.floating],
    dphi_deg: float,
) -> NDArray[np.floating]:
    r"""Calculate the dual-resonance signal at time t.

    Implements the signal model equation:

    $$S(t) = \sqrt{A_1^2 e^{-2R_{2,1}^* t} + A_2^2 e^{-2R_{2,2}^* t}
           + 2 A_1 A_2 e^{-(R_{2,1}^* + R_{2,2}^*) t}
           \cos(2\pi \Delta f t + \Delta\phi)}$$

    Args:
        t: Echo times in seconds.
        amplitude_1: Amplitude of the first signal component.
        amplitude_2: Amplitude of the second signal component.
        r2star_1: R2* relaxation rate of the first component (1/s).
        r2star_2: R2* relaxation rate of the second component (1/s).
        df: Frequency difference between the two components (Hz).
        dphi_deg: Phase difference between the components (degrees).

    Returns:
        Signal magnitude at each echo time.

    Example:
        ```python
        import numpy as np
        from qmri.thermometry.multiecho import thermometry_signal_model

        echo_times = np.linspace(0.001, 0.024, 24)
        signal = thermometry_signal_model(
            t=echo_times,
            amplitude_1=1.0,
            amplitude_2=0.5,
            r2star_1=50.0,
            r2star_2=100.0,
            df=200.0,
            dphi_deg=45.0,
        )
        ```
    """
    dphi_rad = np.deg2rad(dphi_deg)
    radicand: NDArray[np.float64] = (
        amplitude_1**2 * np.exp(-2 * r2star_1 * t)
        + amplitude_2**2 * np.exp(-2 * r2star_2 * t)
        + 2
        * amplitude_1
        * amplitude_2
        * np.exp(-(r2star_1 + r2star_2) * t)
        * np.cos(2 * np.pi * df * t + dphi_rad)
    )
    # Prevent negative values under the square root
    radicand = np.maximum(radicand, 0)
    return np.sqrt(radicand)


def calculate_df_from_temperature(
    temperature_celsius: float | NDArray[np.floating],
    magnetic_field_tesla: float,
) -> float | NDArray[np.floating]:
    r"""Calculate frequency difference from temperature for ethylene glycol.

    Uses the empirical relationship:

    $$\Delta f = \frac{(193.35 - T) \cdot \gamma \cdot B_0}{1.02 \times 10^8}$$

    This relationship is specific to ethylene glycol and should not be
    used for other substances without recalibration.

    Args:
        temperature_celsius: Temperature in degrees Celsius.
        magnetic_field_tesla: Magnetic field strength in Tesla.

    Returns:
        Frequency difference in Hz.

    Example:
        ```python
        from qmri.thermometry.multiecho import calculate_df_from_temperature

        # At 37°C and 3T
        df = calculate_df_from_temperature(37.0, 3.0)
        print(f"Frequency difference: {df:.1f} Hz")
        ```
    """
    return ((193.35 - temperature_celsius) * GAMMA_H * magnetic_field_tesla) / 1.02e8


def calculate_temperature_from_df(
    df: float | NDArray[np.floating],
    magnetic_field_tesla: float,
) -> float | NDArray[np.floating]:
    r"""Calculate temperature from frequency difference for ethylene glycol.

    Uses the empirical relationship:

    $$T[°C] = 193.35 - \frac{1.02 \times 10^8 \cdot |\Delta f|}{\gamma \cdot B_0}$$

    This relationship is specific to ethylene glycol and should not be
    used for other substances without recalibration.

    Args:
        df: Frequency difference in Hz.
        magnetic_field_tesla: Magnetic field strength in Tesla.

    Returns:
        Temperature in degrees Celsius.

    Example:
        ```python
        from qmri.thermometry.multiecho import calculate_temperature_from_df

        # Convert frequency difference to temperature at 3T
        temperature = calculate_temperature_from_df(200.0, 3.0)
        print(f"Temperature: {temperature:.1f} °C")
        ```
    """
    return 193.35 - (1.02e8 * np.abs(df)) / (GAMMA_H * magnetic_field_tesla)


def calculate_temperature_uncertainty(
    df_uncertainty: float,
    magnetic_field_tesla: float,
) -> float:
    r"""Calculate temperature uncertainty from frequency difference uncertainty.

    Uses uncertainty propagation:

    $$u(T) = \frac{1.02 \times 10^8 \cdot u(\Delta f)}{\gamma \cdot B_0}$$

    Args:
        df_uncertainty: Uncertainty in frequency difference (Hz).
        magnetic_field_tesla: Magnetic field strength in Tesla.

    Returns:
        Uncertainty in temperature (°C).

    Example:
        ```python
        from qmri.thermometry.multiecho import calculate_temperature_uncertainty

        # 1 Hz uncertainty in frequency at 3T
        temp_uncertainty = calculate_temperature_uncertainty(1.0, 3.0)
        print(f"Temperature uncertainty: {temp_uncertainty:.2f} °C")
        ```
    """
    return (1.02e8 * df_uncertainty) / (GAMMA_H * magnetic_field_tesla)


def lsq_fit_thermometry_signal_model(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    initial_guess: list[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Perform least squares fit of the dual-resonance signal model.

    Fits the signal data to the thermometry signal model using
    scipy.optimize.curve_fit with bounded parameters.

    Args:
        echo_times: Array of echo times in seconds.
        signal_values: Array of signal values at each echo time.
        initial_guess: Initial parameter estimates
            [amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg].

    Returns:
        Tuple containing:
            - popt: Optimal parameters [A1, A2, R2*1, R2*2, df, dphi].
            - pcov: Covariance matrix of the parameters.
            - r_squared: Coefficient of determination (R²) of the fit.

    If the fit fails to converge, returns arrays of NaN values.

    Example:
        ```python
        import numpy as np
        from qmri.thermometry.multiecho import (
            thermometry_signal_model,
            lsq_fit_thermometry_signal_model,
        )

        # Generate synthetic data
        echo_times = np.linspace(0.001, 0.024, 24)
        true_params = [1.0, 0.5, 50.0, 100.0, 200.0, 45.0]
        signal = thermometry_signal_model(echo_times, *true_params)

        # Fit the model
        initial_guess = [0.8, 0.4, 40.0, 80.0, 180.0, 30.0]
        popt, pcov, r2 = lsq_fit_thermometry_signal_model(
            echo_times, signal, initial_guess
        )
        print(f"R²: {r2:.4f}")
        ```
    """
    max_amplitude = np.max(signal_values)
    bounds = (
        [0, 0, 1e-3, 1e-3, 0, -360],
        [10 * max_amplitude, 10 * max_amplitude, 1000, 1000, _DF_UPPER_BOUND, 360],
    )
    try:
        popt, pcov, *_ = curve_fit(
            thermometry_signal_model,
            echo_times,
            signal_values,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000,
        )
    except RuntimeError:
        popt = np.array([np.nan] * len(initial_guess), dtype=np.float64)
        pcov = np.full((len(initial_guess), len(initial_guess)), np.nan)

    r_squared_val = calculate_r_squared(
        signal_values, thermometry_signal_model(echo_times, *popt)
    )
    return popt, pcov, float(r_squared_val)


def _estimate_df_guess(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
) -> float | None:
    """Estimate the beat frequency directly from the data.

    The dual-resonance magnitude signal oscillates at the frequency
    difference about a decaying envelope. Removing a log-linear envelope
    estimate and locating the dominant peak of a Lomb-Scargle periodogram
    gives a starting value close to the global minimum, keeping the
    non-linear fit out of the local minima (spaced roughly one over the
    echo-train span apart) that a fixed starting value can fall into.

    Args:
        echo_times: Array of echo times in seconds.
        signal_values: Array of signal values at each echo time.

    Returns:
        Estimated frequency difference in Hz, or None when no usable
        estimate can be made (e.g. non-positive signal or a degenerate
        echo-time grid).
    """
    positive = signal_values > 0
    if np.count_nonzero(positive) < 4:
        return None
    envelope_coeffs = np.polyfit(
        echo_times[positive], np.log(signal_values[positive]), deg=1
    )
    residual = signal_values - np.exp(np.polyval(envelope_coeffs, echo_times))
    residual = residual - float(np.mean(residual))
    if not np.any(residual):
        return None

    sorted_times = np.sort(echo_times)
    spacings = np.diff(sorted_times)
    spacings = spacings[spacings > 0]
    span = float(sorted_times[-1] - sorted_times[0])
    if spacings.size == 0 or span <= 0:
        return None
    # Search from half a cycle over the echo train up to the Nyquist limit.
    frequency_min = 0.5 / span
    frequency_max = min(0.5 / float(np.min(spacings)), _DF_UPPER_BOUND)
    if frequency_max <= frequency_min:
        return None
    frequencies = np.linspace(frequency_min, frequency_max, 256)
    power = lombscargle(echo_times, residual, 2 * np.pi * frequencies)
    return float(frequencies[int(np.argmax(power))])


def _df_start_values(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    df_init: DfInitMethod,
) -> list[float]:
    """Return the frequency-difference starting values for the chosen strategy.

    See :data:`DfInitMethod` for the meaning of each strategy. The fixed default
    is always used as a fallback when a Lomb-Scargle estimate cannot be made, so
    the returned list is never empty.
    """
    guesses: list[float] = []
    if df_init in ("lombscargle", "multistart"):
        estimate = _estimate_df_guess(echo_times, signal_values)
        if estimate is not None:
            guesses.append(estimate)
    if df_init in ("fixed", "multistart"):
        guesses.append(_DEFAULT_DF_GUESS)
    if not guesses:  # lombscargle requested but no estimate was possible
        guesses.append(_DEFAULT_DF_GUESS)
    # Drop near-duplicate starts so multistart does not refit the same basin.
    unique: list[float] = []
    for guess in guesses:
        if not any(abs(guess - kept) <= 1.0 for kept in unique):
            unique.append(guess)
    return unique


def _fit_thermometry(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    df_init: DfInitMethod = "multistart",
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Fit the signal model, seeding the frequency per the chosen strategy.

    Runs :func:`lsq_fit_thermometry_signal_model` once per candidate
    frequency-difference starting value selected by ``df_init`` and returns the
    fit with the highest R². For the single-start strategies (``"fixed"`` and
    ``"lombscargle"``) this is just that one fit.

    Args:
        echo_times: Array of echo times in seconds.
        signal_values: Array of signal values at each echo time.
        df_init: Frequency starting-value strategy (see :data:`DfInitMethod`).

    Returns:
        Tuple of (popt, pcov, r_squared) for the best fit.
    """
    max_signal = float(np.max(signal_values))
    df_guesses = _df_start_values(echo_times, signal_values, df_init)

    best: tuple[NDArray[np.floating], NDArray[np.floating], float] | None = None
    for df_guess in df_guesses:
        initial_guess = [
            max_signal / 2.0,
            max_signal / 2.0,
            10.0,
            10.0,
            df_guess,
            0.0,
        ]
        popt, pcov, r_squared_val = lsq_fit_thermometry_signal_model(
            echo_times, signal_values, initial_guess
        )
        if best is None or (
            not np.isnan(r_squared_val)
            and (np.isnan(best[2]) or r_squared_val > best[2])
        ):
            best = (popt, pcov, r_squared_val)
    assert best is not None  # df_guesses is never empty
    return best


@dataclass(frozen=True)
class MultiEchoResult:
    """Result of multi-echo dual-resonance thermometry fitting.

    Attributes:
        temperature: Estimated temperature in degrees Celsius.
        temperature_uncertainty: Uncertainty in temperature (°C).
            For single fits, from covariance matrix.
            For bootstrap fits, from standard deviation of bootstrap samples.
        df: Fitted frequency difference in Hz.
        r_squared: Coefficient of determination (R²) of the fit.
            For bootstrap, this is the mean R² across samples.
        fitted_params: Fitted parameters [A1, A2, R2*1, R2*2, df, dphi_deg].
            For bootstrap, these are the mean parameters.
        n_bootstrap: Number of bootstrap samples (None for single fit).
    """

    temperature: float
    temperature_uncertainty: float
    df: float
    r_squared: float
    fitted_params: NDArray[np.floating]
    n_bootstrap: int | None = None


def fit_multiecho_thermometry(
    signal: NDArray[np.floating],
    echo_times: NDArray[np.floating],
    magnetic_field_tesla: float,
    method: Literal["single", "bootstrap"] = "single",
    n_bootstrap: int = 100,
    r_squared_threshold: float = R_SQUARED_THRESHOLD,
    df_init: DfInitMethod = "multistart",
) -> MultiEchoResult:
    """Fit multi-echo signal to dual-resonance model for thermometry.

    This function fits the dual-resonance signal model to multi-echo
    magnitude data and converts the fitted frequency difference to
    temperature using the ethylene glycol calibration.

    Args:
        signal: Multi-echo magnitude signal array. Shape should be (n_echoes,).
        echo_times: Array of echo times in seconds.
        magnetic_field_tesla: Magnetic field strength in Tesla.
        method: Fitting method. Options:
            - "single": Single least-squares fit (default).
            - "bootstrap": Bootstrap resampling for uncertainty estimation.
        n_bootstrap: Number of bootstrap samples (default 100).
            Only used when method="bootstrap".
        r_squared_threshold: Minimum R² for accepting a fit (default 0.9).
            For bootstrap, samples below threshold are excluded.
        df_init: Frequency starting-value strategy — ``"multistart"`` (default),
            ``"fixed"`` or ``"lombscargle"``. See :data:`DfInitMethod`.

    Returns:
        MultiEchoResult containing temperature, uncertainty, and fit parameters.

    Raises:
        ValueError: If signal and echo_times have different lengths.

    Example:
        ```python
        import numpy as np
        from qmri.thermometry.multiecho import (
            thermometry_signal_model,
            calculate_df_from_temperature,
            fit_multiecho_thermometry,
        )

        # Generate synthetic data at 25°C
        magnetic_field = 3.0
        echo_times = np.linspace(0.001, 0.024, 24)
        df_true = calculate_df_from_temperature(25.0, magnetic_field)
        signal = thermometry_signal_model(
            echo_times, 1.0, 0.5, 50.0, 100.0, df_true, 45.0
        )

        # Fit the model
        result = fit_multiecho_thermometry(
            signal, echo_times, magnetic_field, method="single"
        )
        temp = result.temperature
        uncert = result.temperature_uncertainty
        print(f"Temperature: {temp:.1f} ± {uncert:.2f} °C")
        print(f"R²: {result.r_squared:.4f}")

        # With bootstrap uncertainty
        result_bs = fit_multiecho_thermometry(
            signal, echo_times, magnetic_field,
            method="bootstrap", n_bootstrap=50
        )
        print(f"Bootstrap uncertainty: {result_bs.temperature_uncertainty:.2f} °C")
        ```

    Note:
        The temperature-frequency calibration is specific to ethylene glycol.
        For other substances, the fitted frequency difference (df) can still
        be used, but the temperature conversion will not be valid.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times)

    if len(signal) != len(echo_times):
        msg = (
            f"Signal length ({len(signal)}) must match "
            f"echo_times length ({len(echo_times)})"
        )
        raise ValueError(msg)

    if method == "single":
        popt, pcov, r_squared_val = _fit_thermometry(echo_times, signal, df_init)
        df = popt[4]
        temperature = float(calculate_temperature_from_df(df, magnetic_field_tesla))

        # Calculate uncertainty from covariance matrix
        param_uncertainties = np.sqrt(np.diag(pcov))
        df_uncert = param_uncertainties[4]
        df_uncertainty = df_uncert if not np.isnan(df_uncert) else np.nan
        temp_uncertainty = calculate_temperature_uncertainty(
            df_uncertainty, magnetic_field_tesla
        )

        return MultiEchoResult(
            temperature=temperature,
            temperature_uncertainty=temp_uncertainty,
            df=float(df),
            r_squared=r_squared_val,
            fitted_params=popt,
            n_bootstrap=None,
        )

    elif method == "bootstrap":
        rng = np.random.default_rng(seed=RANDOM_SEED)
        n_points = len(signal)

        temperatures_list: list[float] = []
        r_squared_list: list[float] = []
        fitted_params_list: list[NDArray[np.floating]] = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n_points, size=n_points, replace=True)
            resampled_echo_times = echo_times[indices]
            resampled_signal = signal[indices]

            # Sort by echo time for fitting
            sort_idx = np.argsort(resampled_echo_times)
            resampled_echo_times = resampled_echo_times[sort_idx]
            resampled_signal = resampled_signal[sort_idx]

            # Fit
            popt, _, r_squared_val = _fit_thermometry(
                resampled_echo_times, resampled_signal, df_init
            )

            df = popt[4]
            temp = float(calculate_temperature_from_df(df, magnetic_field_tesla))

            temperatures_list.append(temp)
            r_squared_list.append(r_squared_val)
            fitted_params_list.append(popt)

        temperatures = np.array(temperatures_list)
        r_squared_values = np.array(r_squared_list)
        fitted_params_arr = np.array(fitted_params_list)

        # Filter by R² threshold
        good_fits = r_squared_values >= r_squared_threshold

        if np.any(good_fits):
            mean_temperature = float(np.mean(temperatures[good_fits]))
            temp_uncertainty = float(np.std(temperatures[good_fits]))
            mean_r_squared = float(np.mean(r_squared_values[good_fits]))
            mean_params = np.mean(fitted_params_arr[good_fits], axis=0)
            mean_df = float(mean_params[4])
        else:
            mean_temperature = np.nan
            temp_uncertainty = np.nan
            mean_r_squared = np.nan
            mean_params = np.array([np.nan] * 6)
            mean_df = np.nan

        return MultiEchoResult(
            temperature=mean_temperature,
            temperature_uncertainty=temp_uncertainty,
            df=mean_df,
            r_squared=mean_r_squared,
            fitted_params=mean_params,
            n_bootstrap=n_bootstrap,
        )


RegionAnalysisMethod = Literal["regionwise", "voxelwise", "regionwise_bootstrap"]
"""Analysis methods for image-based (segmentation-driven) thermometry fitting."""


@dataclass(frozen=True)
class RegionThermometryResult:
    """Per-region results from segmentation-driven multi-echo thermometry.

    A "region" is the set of voxels sharing a single non-zero integer label in
    the segmentation image. The interpretation of the per-fit arrays depends on
    the analysis method:

    - ``regionwise``: a single fit of the region-mean signal (arrays length 1).
    - ``voxelwise``: one fit per voxel (arrays length ``region_size``).
    - ``regionwise_bootstrap``: one fit per bootstrap sample (arrays length
      ``n_bootstrap``).

    Attributes:
        region_id: The integer label of the region in the segmentation.
        region_size: Number of voxels in the region.
        temperature: Representative region temperature in °C. For ``voxelwise``
            this is the inverse-variance weighted mean of voxel temperatures; for
            the region methods it is the temperature of the (mean-signal) fit, or
            the mean across bootstrap samples passing the R² threshold.
        temperature_uncertainty: Standard uncertainty (coverage factor
            ``coverage_factor``) of ``temperature`` in °C.
        coverage_factor: Coverage factor k for ``temperature_uncertainty`` (k=1).
        temperature_values: Temperature estimate from each individual fit (°C).
        temperature_uncertainty_values: Per-fit standard uncertainty in °C,
            derived from the fitted Δf covariance.
        r_squared: Coefficient of determination R² for each individual fit.
        fitted_params: Fitted parameters per fit, shape ``(n_fits, 6)``.
        mean_fitted_params: Mean of ``fitted_params`` over fits passing the R²
            threshold, shape ``(6,)`` (NaN if no fit passes).
        signal_values: Signal values fed to each fit, shape ``(n_fits, n_echoes)``.
    """

    region_id: int
    region_size: int
    temperature: float
    temperature_uncertainty: float
    coverage_factor: float
    temperature_values: NDArray[np.floating]
    temperature_uncertainty_values: NDArray[np.floating]
    r_squared: NDArray[np.floating]
    fitted_params: NDArray[np.floating]
    mean_fitted_params: NDArray[np.floating]
    signal_values: NDArray[np.floating]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary of the region results."""
        return {
            "id": self.region_id,
            "temperature": self.temperature,
            "temperature_uncertainty": [
                self.temperature_uncertainty,
                self.coverage_factor,
            ],
            "region_size": self.region_size,
            "mean_fitted_params": self.mean_fitted_params.tolist(),
            "region_temperature_values": self.temperature_values.tolist(),
            "region_temperature_uncertainty_values": (
                self.temperature_uncertainty_values.tolist()
            ),
            "fitted_params": self.fitted_params.tolist(),
            "signal_values": self.signal_values.tolist(),
            "r_squared": self.r_squared.tolist(),
        }


def fit_multiecho_thermometry_image(
    signal: NDArray[np.floating],
    segmentation: NDArray[np.floating],
    echo_times: NDArray[np.floating],
    magnetic_field_tesla: float,
    method: RegionAnalysisMethod = "regionwise",
    n_bootstrap: int = 100,
    r_squared_threshold: float = R_SQUARED_THRESHOLD,
    df_init: DfInitMethod = "multistart",
) -> tuple[NDArray[np.floating], list[RegionThermometryResult]]:
    r"""Fit multi-echo thermometry over a segmented image volume.

    The segmentation defines discrete regions by integer label; label ``0`` is
    treated as background and ignored. Each non-zero region is fitted with the
    dual-resonance model and the fitted frequency difference is converted to
    temperature with the ethylene-glycol calibration.

    The arrays must be co-located in world space: ``signal`` is the 4D
    magnitude volume ``(nx, ny, nz, n_echoes)`` and ``segmentation`` is the 3D
    label map ``(nx, ny, nz)``. ``echo_times`` must have length ``n_echoes`` and
    be in seconds.

    Args:
        signal: Multi-echo magnitude data, shape ``(nx, ny, nz, n_echoes)``.
        segmentation: Integer label map, shape ``(nx, ny, nz)``.
        echo_times: Echo times in seconds, shape ``(n_echoes,)``.
        magnetic_field_tesla: Magnetic field strength $B_0$ in Tesla.
        method: Analysis method:

            - ``"regionwise"``: fit the mean signal within each region once and
              assign the resulting temperature to every voxel in the region.
              Uncertainty comes from the fitted Δf covariance.
            - ``"voxelwise"``: fit each voxel independently; the region summary
              is an inverse-variance weighted mean of voxel temperatures.
            - ``"regionwise_bootstrap"``: resample region voxels with
              replacement, fit each sample's mean signal, and summarise with the
              mean and standard deviation over samples with $R^2 \geq$
              ``r_squared_threshold``.
        n_bootstrap: Number of bootstrap samples (``regionwise_bootstrap`` only).
        r_squared_threshold: Minimum R² for a fit to contribute to
            ``mean_fitted_params`` and to bootstrap summaries.
        df_init: Frequency starting-value strategy — ``"multistart"`` (default),
            ``"fixed"`` or ``"lombscargle"``. See :data:`DfInitMethod`.

    Returns:
        A tuple ``(temperature_map, results)`` where ``temperature_map`` is a 3D
        array of temperatures in °C co-located with the segmentation, and
        ``results`` is a list of :class:`RegionThermometryResult`, one per
        non-empty region (in ascending label order).

    Raises:
        ValueError: If the array dimensions or echo-time length are inconsistent,
            or if ``method`` is not recognised.
    """
    signal = np.asarray(signal, dtype=np.float64)
    segmentation = np.asarray(segmentation)
    echo_times = np.asarray(echo_times, dtype=np.float64)

    if signal.ndim != 4:
        msg = f"signal must be 4D (nx, ny, nz, n_echoes), got {signal.ndim}D"
        raise ValueError(msg)
    if segmentation.ndim != 3:
        msg = f"segmentation must be 3D (nx, ny, nz), got {segmentation.ndim}D"
        raise ValueError(msg)
    if segmentation.shape != signal.shape[:3]:
        msg = (
            "segmentation shape must match the spatial shape of signal: "
            f"{segmentation.shape} != {signal.shape[:3]}"
        )
        raise ValueError(msg)
    if echo_times.shape[0] != signal.shape[-1]:
        msg = (
            f"echo_times length ({echo_times.shape[0]}) must match the number "
            f"of echoes ({signal.shape[-1]})"
        )
        raise ValueError(msg)
    if method not in ("regionwise", "voxelwise", "regionwise_bootstrap"):
        msg = (
            f"Unknown method: {method!r}. Use 'regionwise', 'voxelwise', or "
            "'regionwise_bootstrap'."
        )
        raise ValueError(msg)

    n_echoes = signal.shape[-1]
    n_params = 6

    temperature_map: NDArray[np.float64] = np.zeros(
        segmentation.shape, dtype=np.float64
    )

    regions = np.unique(segmentation)
    regions = regions[regions != 0]  # exclude background
    results: list[RegionThermometryResult] = []

    for region in regions:
        region_mask = segmentation == region
        region_size = int(np.sum(region_mask))
        if region_size == 0:
            continue

        temperature_values_list: list[float] = []
        uncertainty_values_list: list[float] = []
        fitted_params_list: list[NDArray[np.floating]] = []
        signal_values_list: list[NDArray[np.floating]] = []
        r_squared_list: list[float] = []

        region_temperature: float = float("nan")
        region_uncertainty: float = float("nan")

        if method == "regionwise":
            region_signal = np.array(
                [
                    float(np.mean(signal[..., echo][region_mask]))
                    for echo in range(n_echoes)
                ]
            )
            popt, pcov, r_squared_value = _fit_thermometry(
                echo_times, region_signal, df_init
            )
            df = float(popt[4])
            df_uncertainty = float(np.sqrt(np.diag(pcov))[4])
            region_temperature = float(
                calculate_temperature_from_df(df, magnetic_field_tesla)
            )
            region_uncertainty = calculate_temperature_uncertainty(
                df_uncertainty, magnetic_field_tesla
            )
            temperature_map[region_mask] = region_temperature

            temperature_values_list.append(region_temperature)
            uncertainty_values_list.append(region_uncertainty)
            fitted_params_list.append(popt)
            signal_values_list.append(region_signal)
            r_squared_list.append(r_squared_value)

        elif method == "voxelwise":
            for index in np.argwhere(region_mask):
                i, j, k = (int(index[0]), int(index[1]), int(index[2]))
                voxel_signal = signal[i, j, k, :]
                popt, pcov, r_squared_value = _fit_thermometry(
                    echo_times, voxel_signal, df_init
                )
                df = float(popt[4])
                df_uncertainty = float(np.sqrt(np.diag(pcov))[4])
                voxel_temperature = float(
                    calculate_temperature_from_df(df, magnetic_field_tesla)
                )
                voxel_uncertainty = calculate_temperature_uncertainty(
                    df_uncertainty, magnetic_field_tesla
                )
                temperature_map[i, j, k] = voxel_temperature

                temperature_values_list.append(voxel_temperature)
                uncertainty_values_list.append(voxel_uncertainty)
                fitted_params_list.append(popt)
                signal_values_list.append(voxel_signal)
                r_squared_list.append(r_squared_value)

            uncertainty_values = np.array(uncertainty_values_list)
            weights = np.divide(
                1.0,
                uncertainty_values**2,
                where=uncertainty_values != 0,
                out=np.zeros_like(uncertainty_values),
            )
            region_temperature = float(
                np.average(np.array(temperature_values_list), weights=weights)
            )
            weight_sum = float(np.sum(weights))
            region_uncertainty = (
                float(np.sqrt(1.0 / weight_sum)) if weight_sum > 0 else float("nan")
            )

        else:  # regionwise_bootstrap
            rng = np.random.default_rng(seed=RANDOM_SEED)
            flat_signal = signal.reshape(-1, n_echoes)
            region_flat_indices = np.where(region_mask.flatten())[0]
            for _ in range(n_bootstrap):
                sampled_indices = rng.choice(
                    region_flat_indices, size=region_size, replace=True
                )
                region_signal = np.mean(flat_signal[sampled_indices], axis=0)
                popt, pcov, r_squared_value = _fit_thermometry(
                    echo_times, region_signal, df_init
                )
                df = float(popt[4])
                df_uncertainty = float(np.sqrt(np.diag(pcov))[4])
                temperature_values_list.append(
                    float(calculate_temperature_from_df(df, magnetic_field_tesla))
                )
                uncertainty_values_list.append(
                    calculate_temperature_uncertainty(
                        df_uncertainty, magnetic_field_tesla
                    )
                )
                fitted_params_list.append(popt)
                signal_values_list.append(region_signal)
                r_squared_list.append(r_squared_value)

            temperature_values = np.array(temperature_values_list)
            r_squared_arr = np.array(r_squared_list)
            passing = r_squared_arr >= r_squared_threshold
            if np.any(passing):
                region_temperature = float(np.mean(temperature_values[passing]))
                region_uncertainty = float(np.std(temperature_values[passing]))
            temperature_map[region_mask] = region_temperature

        r_squared = np.array(r_squared_list)
        fitted_params = np.array(fitted_params_list)
        passing_mask = r_squared >= r_squared_threshold
        if fitted_params.size > 0 and np.any(passing_mask):
            mean_fitted_params = np.mean(fitted_params[passing_mask], axis=0)
        else:
            mean_fitted_params = np.full(n_params, np.nan)

        results.append(
            RegionThermometryResult(
                region_id=int(region),
                region_size=region_size,
                temperature=region_temperature,
                temperature_uncertainty=region_uncertainty,
                coverage_factor=1.0,
                temperature_values=np.array(temperature_values_list),
                temperature_uncertainty_values=np.array(uncertainty_values_list),
                r_squared=r_squared,
                fitted_params=fitted_params,
                mean_fitted_params=mean_fitted_params,
                signal_values=np.array(signal_values_list),
            )
        )

    return temperature_map, results
