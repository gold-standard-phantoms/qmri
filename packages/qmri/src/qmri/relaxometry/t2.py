r"""T2 relaxometry fitting and signal models.

This module provides functions for T2 mapping using multi-echo spin echo
(MESE) data.

Signal Models

**Full Model (with offset)**:

$$S(TE) = A \exp\left(-\frac{TE}{T_2}\right) + C$$

where A is the signal amplitude and C is an offset term.

**Reduced Model (no offset)**:

$$S(TE) = A \exp\left(-\frac{TE}{T_2}\right)$$

References:
    .. [1] Milford, D., et al. "Mono-Exponential Fitting in T2-Relaxometry:
           Relevance of Offset and First Echo." PLOS ONE 10(12), 2015.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

__all__ = [
    "T2Result",
    "fit",
    "signal_decay",
]


@dataclass(frozen=True)
class T2Result:
    """Result of T2 fitting.

    Attributes:
        t2: T2 relaxation time in seconds.
        amplitude: Signal amplitude (A = k * S0).
        offset: Offset term (only for full model).
    """

    t2: NDArray[np.floating]
    amplitude: NDArray[np.floating]
    offset: NDArray[np.floating] | None = None


def signal_decay(
    amplitude: NDArray[np.floating] | float,
    t2: NDArray[np.floating] | float,
    echo_times: NDArray[np.floating],
    offset: NDArray[np.floating] | float = 0.0,
) -> NDArray[np.floating]:
    r"""Calculate T2 decay signal.

    Implements the T2 decay equation:

    $$S(TE) = A \exp\left(-\frac{TE}{T_2}\right) + C$$

    Args:
        amplitude: Signal amplitude (A).
        t2: T2 relaxation time in seconds.
        echo_times: Echo times (TE) in seconds.
        offset: Offset term (default 0.0).

    Returns:
        Signal intensity at each echo time.

    Example:
        ```python
        import numpy as np
        from qmri.relaxometry import t2

        te = np.array([0.01, 0.02, 0.04, 0.08])
        signal = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)
        ```
    """
    a = np.asarray(amplitude)
    t2_val = np.asarray(t2)
    te = np.asarray(echo_times)
    c = np.asarray(offset)

    result: NDArray[np.floating] = a * np.exp(-te / t2_val) + c
    return result


def _residual_t2_full(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    te: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Residual function for full T2 model."""
    result: NDArray[np.floating] = signal - (x[1] * np.exp(-te / x[0]) + x[2])
    return result


def _residual_t2_reduced(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    te: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Residual function for reduced T2 model."""
    result: NDArray[np.floating] = signal - x[1] * np.exp(-te / x[0])
    return result


def _fit_t2_voxel(
    signal: NDArray[np.floating],
    echo_times: NDArray[np.floating],
    model: Literal["full", "reduced"],
) -> NDArray[np.floating] | None:
    """Fit T2 model to a single voxel."""
    te = np.asarray(echo_times)

    if model == "full":
        # Initial guess: T2=0.1s, A=first signal, offset=0
        x0 = np.array([0.1, signal[0], 0.0])
        residual_func = _residual_t2_full
    else:  # reduced
        # Initial guess: T2=0.1s, A=first signal
        x0 = np.array([0.1, signal[0]])
        residual_func = _residual_t2_reduced

    try:
        result = least_squares(
            residual_func,
            x0,
            args=(signal, te),
            method="lm",
            max_nfev=1000000,
        )
        if result.success:
            return_val: NDArray[np.floating] = result.x
            return return_val
    except Exception:  # noqa: BLE001
        pass

    return None


def fit(
    signal: NDArray[np.floating],
    echo_times: NDArray[np.floating],
    *,
    model: Literal["full", "reduced"] = "full",
    mask: NDArray[np.bool_] | None = None,
    skip_echoes: int = 1,
    max_t2: float = 10.0,
) -> T2Result:
    """Fit T2 using mono-exponential decay model.

    Args:
        signal: Signal intensities. Last dimension corresponds to echo times.
        echo_times: Echo times (TE) in seconds, in ascending order.
        model: Model to use. "full" includes offset term (default).
        mask: Boolean mask for processing.
        skip_echoes: Number of initial echoes to skip (default 1).
            Skipping the first echo reduces fitting errors.
        max_t2: Maximum valid T2 value in seconds (default 10.0).

    Returns:
        Fitted T2, amplitude, and offset (if full model).

    Skipping the first echo is recommended as it helps minimise
    errors in T2 fitting due to stimulated echoes and imperfect
    refocusing pulses [1]_.

    Example:
        ```python
        import numpy as np
        from qmri.relaxometry import t2

        te = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
        signal = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)
        result = t2.fit(signal, te, skip_echoes=0)
        print(f"T2 = {result.t2 * 1000:.1f} ms")
        ```
    """
    signal = np.asarray(signal)
    te = np.asarray(echo_times)

    if signal.shape[-1] != len(te):
        msg = (
            f"Signal has {signal.shape[-1]} time points, "
            f"but {len(te)} echo times provided"
        )
        raise ValueError(msg)

    # Apply echo skipping
    if skip_echoes > 0:
        signal = signal[..., skip_echoes:]
        te = te[skip_echoes:]

    n_params = 3 if model == "full" else 2

    # Handle 1D case (single voxel)
    if signal.ndim == 1:
        result = _fit_t2_voxel(signal, te, model)
        if result is None:
            result = np.zeros(n_params)
        t2_val = np.clip(result[0], 0, max_t2)
        if not np.isfinite(t2_val) or t2_val <= 0:
            t2_val = 0.0
        return T2Result(
            t2=np.asarray(t2_val),
            amplitude=np.asarray(result[1]),
            offset=np.asarray(result[2]) if model == "full" else None,
        )

    # Multi-dimensional case
    spatial_shape = signal.shape[:-1]
    flat_signal = signal.reshape(-1, signal.shape[-1])
    n_voxels = flat_signal.shape[0]

    flat_mask = np.ones(n_voxels, dtype=bool)
    if mask is not None:
        flat_mask = mask.reshape(-1)

    flat_result = np.zeros((n_voxels, n_params), dtype=np.float64)

    for idx in range(n_voxels):
        if not flat_mask[idx]:
            continue
        voxel_result = _fit_t2_voxel(flat_signal[idx], te, model)
        if voxel_result is not None:
            flat_result[idx] = voxel_result

    # Clean up T2 values
    t2_flat = flat_result[:, 0]
    t2_flat[~np.isfinite(t2_flat)] = 0.0
    t2_flat[t2_flat <= 0] = 0.0
    t2_flat[t2_flat > max_t2] = 0.0

    result_shape = spatial_shape
    return T2Result(
        t2=t2_flat.reshape(result_shape),
        amplitude=flat_result[:, 1].reshape(result_shape),
        offset=(flat_result[:, 2].reshape(result_shape) if model == "full" else None),
    )
