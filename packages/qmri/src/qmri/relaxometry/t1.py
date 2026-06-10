r"""T1 relaxometry fitting and signal models.

This module provides functions for T1 mapping using inversion recovery (IR)
and variable repetition time (VTR) methods.

Signal Models

**Inversion Recovery (General)**:

$$S = S_0 \left(1 - 2 \alpha \exp\left(-\frac{TI}{T_1}\right)
+ \exp\left(-\frac{TR}{T_1}\right)\right)$$

where $\alpha$ is the inversion efficiency (ideally 1.0).

**Inversion Recovery (Classical)**:

$$S = S_0 \left(1 - 2 \alpha \exp\left(-\frac{TI}{T_1}\right)\right)$$

This assumes TR >> T1 so the final term is negligible.

**Variable TR (VTR)**:

$$S = M \left(1 - \exp\left(-\frac{TR}{T_1}\right)\right)$$

where M is the equilibrium magnetisation.

References:
    .. [1] Barral, J.K., et al. "A robust methodology for in vivo T1 mapping."
           MRM 64(4):1057-67, 2010.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

__all__ = [
    "T1Model",
    "T1Result",
    "T1IRResult",
    "T1VTRResult",
    "fit",
    "fit_ir",
    "fit_vtr",
    "signal_ir",
    "signal_ir_classical",
    "signal_vtr",
]


class T1Model(str, Enum):
    """Model to use for T1 mapping."""

    GENERAL = "general"
    """General IR model including TR recovery term."""

    CLASSICAL = "classical"
    """Classical IR model assuming TR >> T1."""

    VTR = "vtr"
    """Variable TR saturation recovery model."""


@dataclass(frozen=True)
class T1Result:
    """Base result of T1 fitting.

    Attributes:
        t1: T1 relaxation time in seconds.
    """

    t1: NDArray[np.floating]


@dataclass(frozen=True)
class T1IRResult(T1Result):
    """Result of inversion recovery T1 fitting.

    Attributes:
        t1: T1 relaxation time in seconds.
        s0: Signal amplitude at equilibrium.
        inversion_efficiency: Inversion efficiency (ideally 1.0).
    """

    s0: NDArray[np.floating]
    inversion_efficiency: NDArray[np.floating]


@dataclass(frozen=True)
class T1VTRResult(T1Result):
    """Result of variable TR T1 fitting.

    Attributes:
        t1: T1 relaxation time in seconds.
        m: Equilibrium magnetisation.
    """

    m: NDArray[np.floating]


def signal_ir(
    s0: NDArray[np.floating] | float,
    t1: NDArray[np.floating] | float,
    inversion_times: NDArray[np.floating],
    repetition_times: NDArray[np.floating] | float,
    inversion_efficiency: NDArray[np.floating] | float = 1.0,
) -> NDArray[np.floating]:
    r"""Calculate inversion recovery signal (general model).

    Implements the general IR signal equation:

    $$S = S_0 \left(1 - 2 \alpha \exp\left(-\frac{TI}{T_1}\right)
    + \exp\left(-\frac{TR}{T_1}\right)\right)$$

    Args:
        s0: Signal amplitude at equilibrium.
        t1: T1 relaxation time in seconds.
        inversion_times: Inversion times (TI) in seconds.
        repetition_times: Repetition time(s) (TR) in seconds. Can be scalar or array
            matching inversion_times.
        inversion_efficiency: Inversion efficiency (default 1.0).

    Returns:
        Signal intensity at each inversion time.

    Example:
        ```python
        import numpy as np
        from qmri.relaxometry import t1

        ti = np.array([0.1, 0.5, 1.0, 2.0])
        signal = t1.signal_ir(s0=1000, t1=1.0, inversion_times=ti,
                              repetition_times=5.0)
        ```
    """
    s0 = np.asarray(s0)
    t1_val = np.asarray(t1)
    ti = np.asarray(inversion_times)
    tr = np.asarray(repetition_times)
    alpha = np.asarray(inversion_efficiency)

    result: NDArray[np.floating] = s0 * (
        1 - 2 * alpha * np.exp(-ti / t1_val) + np.exp(-tr / t1_val)
    )
    return result


def signal_ir_classical(
    s0: NDArray[np.floating] | float,
    t1: NDArray[np.floating] | float,
    inversion_times: NDArray[np.floating],
    inversion_efficiency: NDArray[np.floating] | float = 1.0,
) -> NDArray[np.floating]:
    r"""Calculate inversion recovery signal (classical model).

    Implements the classical IR signal equation (assumes TR >> T1):

    $$S = S_0 \left(1 - 2 \alpha \exp\left(-\frac{TI}{T_1}\right)\right)$$

    Args:
        s0: Signal amplitude at equilibrium.
        t1: T1 relaxation time in seconds.
        inversion_times: Inversion times (TI) in seconds.
        inversion_efficiency: Inversion efficiency (default 1.0).

    Returns:
        Signal intensity at each inversion time.
    """
    s0 = np.asarray(s0)
    t1_val = np.asarray(t1)
    ti = np.asarray(inversion_times)
    alpha = np.asarray(inversion_efficiency)

    result: NDArray[np.floating] = s0 * (1 - 2 * alpha * np.exp(-ti / t1_val))
    return result


def signal_vtr(
    m: NDArray[np.floating] | float,
    t1: NDArray[np.floating] | float,
    repetition_times: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Calculate variable TR signal.

    Implements the VTR signal equation:

    $$S = M \left(1 - \exp\left(-\frac{TR}{T_1}\right)\right)$$

    Args:
        m: Equilibrium magnetisation.
        t1: T1 relaxation time in seconds.
        repetition_times: Repetition times (TR) in seconds.

    Returns:
        Signal intensity at each TR.

    Example:
        ```python
        import numpy as np
        from qmri.relaxometry import t1

        tr = np.array([0.5, 1.0, 2.0, 4.0])
        signal = t1.signal_vtr(m=1000, t1=1.0, repetition_times=tr)
        ```
    """
    m = np.asarray(m)
    t1_val = np.asarray(t1)
    tr = np.asarray(repetition_times)

    result: NDArray[np.floating] = m * (1 - np.exp(-tr / t1_val))
    return result


def _residual_ir_general(
    x: NDArray[np.floating],
    voxel_signal: NDArray[np.floating],
    ti: NDArray[np.floating],
    tr: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Residual function for general IR model."""
    result: NDArray[np.floating] = voxel_signal - x[1] * (
        1 - 2 * x[2] * np.exp(-ti / x[0]) + np.exp(-tr / x[0])
    )
    return result


def _residual_ir_classical(
    x: NDArray[np.floating],
    voxel_signal: NDArray[np.floating],
    ti: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Residual function for classical IR model."""
    result: NDArray[np.floating] = voxel_signal - x[1] * (
        1 - 2 * x[2] * np.exp(-ti / x[0])
    )
    return result


def _fit_ir_voxel(
    signal: NDArray[np.floating],
    inversion_times: NDArray[np.floating],
    repetition_times: NDArray[np.floating] | float,
    model: Literal["general", "classical"],
) -> NDArray[np.floating] | None:
    """Fit IR model to a single voxel with polarity restoration."""
    ti = np.asarray(inversion_times)
    tr = np.asarray(repetition_times)
    n_ti = len(ti)

    best_cost = np.inf
    best_result: NDArray[np.floating] | None = None

    # Try different polarity restoration points
    for restore_idx in range(n_ti + 1):
        # Apply polarity restoration
        voxel_signal = signal.copy()
        voxel_signal[:restore_idx] = -voxel_signal[:restore_idx]

        # Initial guess: T1=1s, S0=last signal, inv_eff=1
        x0 = np.array([1.0, voxel_signal[-1], 1.0])

        try:
            if model == "general":
                result = least_squares(
                    _residual_ir_general,
                    x0,
                    args=(voxel_signal, ti, tr),
                    method="lm",
                    max_nfev=100000,
                )
            else:  # classical
                result = least_squares(
                    _residual_ir_classical,
                    x0,
                    args=(voxel_signal, ti),
                    method="lm",
                    max_nfev=100000,
                )
            if result.success:
                cost = np.sum(np.abs(result.fun))
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.x
        except Exception:  # noqa: BLE001
            continue

    return best_result


def _residual_vtr(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    tr: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Residual function for VTR model."""
    result: NDArray[np.floating] = signal - x[1] * (1 - np.exp(-tr / x[0]))
    return result


def _fit_vtr_voxel(
    signal: NDArray[np.floating],
    repetition_times: NDArray[np.floating],
) -> NDArray[np.floating] | None:
    """Fit VTR model to a single voxel."""
    tr = np.asarray(repetition_times)

    # Initial guess: T1=1s, M=first signal
    x0 = np.array([1.0, signal[0]])

    try:
        result = least_squares(
            _residual_vtr,
            x0,
            args=(signal, tr),
            method="lm",
            max_nfev=100000,
        )
        if result.success:
            return_val: NDArray[np.floating] = result.x
            return return_val
    except Exception:  # noqa: BLE001
        pass

    return None


def fit_ir(
    signal: NDArray[np.floating],
    inversion_times: NDArray[np.floating],
    repetition_times: NDArray[np.floating] | float,
    *,
    model: Literal["general", "classical"] = "general",
    mask: NDArray[np.bool_] | None = None,
    max_t1: float = 20.0,
) -> T1IRResult:
    """Fit T1 using inversion recovery.

    Args:
        signal: Signal intensities. Last dimension corresponds to inversion times.
        inversion_times: Inversion times (TI) in seconds, in ascending order.
        repetition_times: Repetition time(s) (TR) in seconds.
        model: IR model to use (default "general").
        mask: Boolean mask for processing.
        max_t1: Maximum valid T1 value in seconds (default 20.0).

    Returns:
        Fitted T1, S0, and inversion efficiency maps.

    Example:
        ```python
        import numpy as np
        from qmri.relaxometry import t1

        ti = np.array([0.1, 0.5, 1.0, 2.0])
        # Generate synthetic data
        signal = t1.signal_ir(s0=1000, t1=1.2, inversion_times=ti,
                              repetition_times=5.0)
        result = t1.fit_ir(signal, ti, repetition_times=5.0)
        print(f"T1 = {result.t1:.2f} s")
        ```
    """
    signal = np.asarray(signal)
    ti = np.asarray(inversion_times)
    tr = np.asarray(repetition_times)

    if signal.shape[-1] != len(ti):
        msg = (
            f"Signal has {signal.shape[-1]} time points, "
            f"but {len(ti)} inversion times provided"
        )
        raise ValueError(msg)

    # Handle 1D case (single voxel)
    if signal.ndim == 1:
        result = _fit_ir_voxel(signal, ti, tr, model)
        if result is None:
            result = np.zeros(3)
        t1_val = np.clip(result[0], 0, max_t1)
        if not np.isfinite(t1_val):
            t1_val = 0.0
        return T1IRResult(
            t1=np.asarray(t1_val),
            s0=np.asarray(result[1]),
            inversion_efficiency=np.asarray(result[2]),
        )

    # Multi-dimensional case
    spatial_shape = signal.shape[:-1]
    flat_signal = signal.reshape(-1, signal.shape[-1])
    n_voxels = flat_signal.shape[0]

    flat_mask = np.ones(n_voxels, dtype=bool)
    if mask is not None:
        flat_mask = mask.reshape(-1)

    # Results array: [T1, S0, inv_eff]
    flat_result = np.zeros((n_voxels, 3), dtype=np.float64)

    for idx in range(n_voxels):
        if not flat_mask[idx]:
            continue
        voxel_result = _fit_ir_voxel(flat_signal[idx], ti, tr, model)
        if voxel_result is not None:
            flat_result[idx] = voxel_result

    # Clean up T1 values
    t1_flat = flat_result[:, 0]
    t1_flat[~np.isfinite(t1_flat)] = 0.0
    t1_flat[t1_flat < 0] = 0.0
    t1_flat[t1_flat > max_t1] = 0.0

    result_shape = spatial_shape
    return T1IRResult(
        t1=t1_flat.reshape(result_shape),
        s0=flat_result[:, 1].reshape(result_shape),
        inversion_efficiency=flat_result[:, 2].reshape(result_shape),
    )


def fit_vtr(
    signal: NDArray[np.floating],
    repetition_times: NDArray[np.floating],
    *,
    mask: NDArray[np.bool_] | None = None,
    max_t1: float = 20.0,
) -> T1VTRResult:
    """Fit T1 using variable TR method.

    Args:
        signal: Signal intensities. Last dimension corresponds to TRs.
        repetition_times: Repetition times (TR) in seconds, in ascending order.
        mask: Boolean mask for processing.
        max_t1: Maximum valid T1 value in seconds (default 20.0).

    Returns:
        Fitted T1 and M maps.

    Example:
        ```python
        import numpy as np
        from qmri.relaxometry import t1

        tr = np.array([0.5, 1.0, 2.0, 4.0])
        signal = t1.signal_vtr(m=1000, t1=1.2, repetition_times=tr)
        result = t1.fit_vtr(signal, tr)
        print(f"T1 = {result.t1:.2f} s")
        ```
    """
    signal = np.asarray(signal)
    tr = np.asarray(repetition_times)

    if signal.shape[-1] != len(tr):
        msg = (
            f"Signal has {signal.shape[-1]} time points, "
            f"but {len(tr)} repetition times provided"
        )
        raise ValueError(msg)

    # Handle 1D case (single voxel)
    if signal.ndim == 1:
        result = _fit_vtr_voxel(signal, tr)
        if result is None:
            result = np.zeros(2)
        t1_val = np.clip(result[0], 0, max_t1)
        if not np.isfinite(t1_val):
            t1_val = 0.0
        return T1VTRResult(
            t1=np.asarray(t1_val),
            m=np.asarray(result[1]),
        )

    # Multi-dimensional case
    spatial_shape = signal.shape[:-1]
    flat_signal = signal.reshape(-1, signal.shape[-1])
    n_voxels = flat_signal.shape[0]

    flat_mask = np.ones(n_voxels, dtype=bool)
    if mask is not None:
        flat_mask = mask.reshape(-1)

    # Results array: [T1, M]
    flat_result = np.zeros((n_voxels, 2), dtype=np.float64)

    for idx in range(n_voxels):
        if not flat_mask[idx]:
            continue
        voxel_result = _fit_vtr_voxel(flat_signal[idx], tr)
        if voxel_result is not None:
            flat_result[idx] = voxel_result

    # Clean up T1 values
    t1_flat = flat_result[:, 0]
    t1_flat[~np.isfinite(t1_flat)] = 0.0
    t1_flat[t1_flat < 0] = 0.0
    t1_flat[t1_flat > max_t1] = 0.0

    result_shape = spatial_shape
    return T1VTRResult(
        t1=t1_flat.reshape(result_shape),
        m=flat_result[:, 1].reshape(result_shape),
    )


def fit(
    signal: NDArray[np.floating],
    time_points: NDArray[np.floating],
    *,
    method: Literal["ir", "ir_classical", "vtr"] = "ir",
    repetition_times: NDArray[np.floating] | float | None = None,
    mask: NDArray[np.bool_] | None = None,
    max_t1: float = 20.0,
) -> T1IRResult | T1VTRResult:
    """Fit T1 using specified method.

    This is a convenience function that dispatches to the appropriate
    fitting function based on the method parameter.

    Args:
        signal: Signal intensities. Last dimension corresponds to time points.
        time_points: Time points in seconds (TI for IR methods, TR for VTR).
        method: Fitting method (default "ir").
        repetition_times: TR values for IR methods (required for "ir" and
            "ir_classical").
        mask: Boolean mask for processing.
        max_t1: Maximum valid T1 value in seconds (default 20.0).

    Returns:
        Fitted parameters depending on method.
    """
    if method in ("ir", "ir_classical"):
        if repetition_times is None:
            msg = f"repetition_times required for method '{method}'"
            raise ValueError(msg)
        model: Literal["general", "classical"] = (
            "general" if method == "ir" else "classical"
        )
        return fit_ir(
            signal,
            time_points,
            repetition_times,
            model=model,
            mask=mask,
            max_t1=max_t1,
        )
    elif method == "vtr":
        return fit_vtr(signal, time_points, mask=mask, max_t1=max_t1)
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)
