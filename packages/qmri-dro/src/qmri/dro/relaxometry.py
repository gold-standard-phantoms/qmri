"""Relaxometry phantom generation for T1/T2 validation.

This module provides functions for generating synthetic T1 relaxometry
data with known ground truth values using inversion recovery (IR) and
variable TR (VTR) methods.

Example:
    ```python
    from qmri.dro import relaxometry
    from qmri.relaxometry import t1

    # Generate IR data with known T1
    phantom = relaxometry.generate_t1_ir(
        t1=1.2,
        inversion_times=[0.1, 0.5, 1.0, 2.0, 3.0],
        repetition_time=5.0,
        snr=100,
        seed=42,
    )

    # Fit and validate
    result = t1.fit_ir(phantom.signal, phantom.time_points, repetition_times=5.0)
    print(f"True T1: {phantom.ground_truth['t1'].value:.2f} s")
    print(f"Fitted T1: {float(result.t1):.2f} s")
    ```
"""

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from qmri.dro._types import GroundTruth, T1Phantom
from qmri.noise import gaussian, rician
from qmri.relaxometry import t1 as t1_module

__all__ = [
    "generate_t1_ir",
    "generate_t1_vtr",
]


@overload
def generate_t1_ir(
    t1: float,
    inversion_times: Sequence[float],
    *,
    s0: float = ...,
    repetition_time: float = ...,
    inversion_efficiency: float = ...,
    model: Literal["general", "classical"] = ...,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> T1Phantom: ...


@overload
def generate_t1_ir(
    t1: NDArray[np.floating],
    inversion_times: Sequence[float],
    *,
    s0: NDArray[np.floating] | float = ...,
    repetition_time: float = ...,
    inversion_efficiency: NDArray[np.floating] | float = ...,
    model: Literal["general", "classical"] = ...,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> T1Phantom: ...


def generate_t1_ir(
    t1: NDArray[np.floating] | float,
    inversion_times: Sequence[float],
    *,
    s0: NDArray[np.floating] | float = 1000.0,
    repetition_time: float = 5.0,
    inversion_efficiency: NDArray[np.floating] | float = 1.0,
    model: Literal["general", "classical"] = "general",
    snr: float | None = None,
    noise_model: Literal["rician", "gaussian"] = "rician",
    seed: int | None = None,
) -> T1Phantom:
    r"""Generate a T1 phantom using inversion recovery.

    Creates synthetic IR signal using either the general or classical
    model with optional noise for validation and testing.

    Args:
        t1: T1 relaxation time in seconds. Can be scalar or array.
        inversion_times: Inversion times (TI) in seconds.
        s0: Signal amplitude at equilibrium. Default is 1000.0.
        repetition_time: Repetition time (TR) in seconds. Default is 5.0.
        inversion_efficiency: Inversion efficiency (0-1). Default is 1.0.
        model: IR model to use. Default is "general".
            - "general": Full model including TR recovery term
            - "classical": Assumes TR >> T1
        snr: Signal-to-noise ratio. If None, no noise is added.
        noise_model: Type of noise to add. Default is "rician".
        seed: Random seed for reproducibility.

    Returns:
        T1Phantom containing the synthetic signal, time points, and ground truth.

    Raises:
        ValueError: If model is not "general" or "classical".
        ValueError: If noise_model is not "rician" or "gaussian".

    Example:
        Single-voxel phantom:

        ```python
        from qmri.dro import relaxometry

        phantom = relaxometry.generate_t1_ir(
            t1=1.2,
            inversion_times=[0.1, 0.5, 1.0, 2.0, 3.0],
            snr=100,
            seed=42,
        )
        print(phantom.signal)
        ```

        Multi-voxel phantom:

        ```python
        import numpy as np
        from qmri.dro import relaxometry

        t1_map = np.array([[0.5, 1.0],
                          [1.5, 2.0]])
        phantom = relaxometry.generate_t1_ir(
            t1=t1_map,
            inversion_times=[0.1, 0.5, 1.0, 2.0],
            snr=50,
            seed=42,
        )
        print(phantom.signal.shape)
        # (2, 2, 4)
        ```

    Notes:
        **General model**:

        $$S = S_0 \\left(1 - 2 \\alpha \\exp\\left(-\\frac{TI}{T_1}\\right)
        + \\exp\\left(-\\frac{TR}{T_1}\\right)\\right)$$

        **Classical model** (assumes TR >> T1):

        $$S = S_0 \\left(1 - 2 \\alpha \\exp\\left(-\\frac{TI}{T_1}\\right)\\right)$$
    """
    if model not in ("general", "classical"):
        msg = f"model must be 'general' or 'classical', got '{model}'"
        raise ValueError(msg)
    if noise_model not in ("rician", "gaussian"):
        msg = f"noise_model must be 'rician' or 'gaussian', got '{noise_model}'"
        raise ValueError(msg)

    ti_arr = np.asarray(inversion_times, dtype=np.float64)
    t1_arr = np.asarray(t1, dtype=np.float64)
    s0_arr = np.asarray(s0, dtype=np.float64)
    alpha_arr = np.asarray(inversion_efficiency, dtype=np.float64)

    # Generate clean signal
    if t1_arr.ndim == 0:
        # Single voxel case
        if model == "general":
            signal = t1_module.signal_ir(
                s0=s0_arr,
                t1=t1_arr,
                inversion_times=ti_arr,
                repetition_times=repetition_time,
                inversion_efficiency=alpha_arr,
            )
        else:
            signal = t1_module.signal_ir_classical(
                s0=s0_arr,
                t1=t1_arr,
                inversion_times=ti_arr,
                inversion_efficiency=alpha_arr,
            )
    else:
        # Multi-voxel case: output shape is (*t1.shape, n_timepoints)
        s0_bc = np.broadcast_to(s0_arr, t1_arr.shape)
        alpha_bc = np.broadcast_to(alpha_arr, t1_arr.shape)

        signal = np.zeros((*t1_arr.shape, len(ti_arr)), dtype=np.float64)
        for i, ti in enumerate(ti_arr):
            if model == "general":
                signal[..., i] = s0_bc * (
                    1
                    - 2 * alpha_bc * np.exp(-ti / t1_arr)
                    + np.exp(-repetition_time / t1_arr)
                )
            else:
                signal[..., i] = s0_bc * (1 - 2 * alpha_bc * np.exp(-ti / t1_arr))

    # Add noise if requested
    if snr is not None:
        rng = np.random.default_rng(seed)
        if noise_model == "rician":
            signal = rician.add_noise(signal, snr=snr, rng=rng)
        else:
            signal = gaussian.add_noise(signal, snr=snr, rng=rng)

    # Build ground truth dictionary
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]] = {
        "t1": GroundTruth(
            value=float(t1) if t1_arr.ndim == 0 else t1_arr,
            units="s",
            description="T1 relaxation time",
        ),
        "s0": GroundTruth(
            value=float(s0) if np.asarray(s0).ndim == 0 else s0_arr,
            units="a.u.",
            description="Signal amplitude at equilibrium",
        ),
        "inversion_efficiency": GroundTruth(
            value=(
                float(inversion_efficiency)
                if np.asarray(inversion_efficiency).ndim == 0
                else alpha_arr
            ),
            units="",
            description="Inversion efficiency",
        ),
    }

    return T1Phantom(
        signal=signal,
        time_points=ti_arr,
        method="ir",
        model=model,
        repetition_time=repetition_time,
        ground_truth=ground_truth,
        snr=snr,
        seed=seed,
    )


@overload
def generate_t1_vtr(
    t1: float,
    repetition_times: Sequence[float],
    *,
    m: float = ...,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> T1Phantom: ...


@overload
def generate_t1_vtr(
    t1: NDArray[np.floating],
    repetition_times: Sequence[float],
    *,
    m: NDArray[np.floating] | float = ...,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> T1Phantom: ...


def generate_t1_vtr(
    t1: NDArray[np.floating] | float,
    repetition_times: Sequence[float],
    *,
    m: NDArray[np.floating] | float = 1000.0,
    snr: float | None = None,
    noise_model: Literal["rician", "gaussian"] = "rician",
    seed: int | None = None,
) -> T1Phantom:
    r"""Generate a T1 phantom using variable TR method.

    Creates synthetic VTR signal for saturation recovery T1 mapping
    with optional noise for validation and testing.

    Args:
        t1: T1 relaxation time in seconds. Can be scalar or array.
        repetition_times: Repetition times (TR) in seconds.
        m: Equilibrium magnetisation. Default is 1000.0.
        snr: Signal-to-noise ratio. If None, no noise is added.
        noise_model: Type of noise to add. Default is "rician".
        seed: Random seed for reproducibility.

    Returns:
        T1Phantom containing the synthetic signal, time points, and ground truth.

    Raises:
        ValueError: If noise_model is not "rician" or "gaussian".

    Example:
        ```python
        from qmri.dro import relaxometry
        from qmri.relaxometry import t1

        phantom = relaxometry.generate_t1_vtr(
            t1=1.2,
            repetition_times=[0.5, 1.0, 2.0, 4.0, 8.0],
            snr=100,
            seed=42,
        )

        result = t1.fit_vtr(phantom.signal, phantom.time_points)
        print(f"True T1: {phantom.ground_truth['t1'].value:.2f} s")
        print(f"Fitted T1: {float(result.t1):.2f} s")
        ```

    Notes:
        The VTR signal model is:

        $$S = M \\left(1 - \\exp\\left(-\\frac{TR}{T_1}\\right)\\right)$$

        where M is the equilibrium magnetisation.
    """
    if noise_model not in ("rician", "gaussian"):
        msg = f"noise_model must be 'rician' or 'gaussian', got '{noise_model}'"
        raise ValueError(msg)

    tr_arr = np.asarray(repetition_times, dtype=np.float64)
    t1_arr = np.asarray(t1, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)

    # Generate clean signal
    if t1_arr.ndim == 0:
        # Single voxel case
        signal = t1_module.signal_vtr(m=m_arr, t1=t1_arr, repetition_times=tr_arr)
    else:
        # Multi-voxel case: output shape is (*t1.shape, n_timepoints)
        m_bc = np.broadcast_to(m_arr, t1_arr.shape)

        signal = np.zeros((*t1_arr.shape, len(tr_arr)), dtype=np.float64)
        for i, tr in enumerate(tr_arr):
            signal[..., i] = m_bc * (1 - np.exp(-tr / t1_arr))

    # Add noise if requested
    if snr is not None:
        rng = np.random.default_rng(seed)
        if noise_model == "rician":
            signal = rician.add_noise(signal, snr=snr, rng=rng)
        else:
            signal = gaussian.add_noise(signal, snr=snr, rng=rng)

    # Build ground truth dictionary
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]] = {
        "t1": GroundTruth(
            value=float(t1) if t1_arr.ndim == 0 else t1_arr,
            units="s",
            description="T1 relaxation time",
        ),
        "m": GroundTruth(
            value=float(m) if np.asarray(m).ndim == 0 else m_arr,
            units="a.u.",
            description="Equilibrium magnetisation",
        ),
    }

    return T1Phantom(
        signal=signal,
        time_points=tr_arr,
        method="vtr",
        model=None,
        repetition_time=None,
        ground_truth=ground_truth,
        snr=snr,
        seed=seed,
    )
