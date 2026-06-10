"""Perfusion phantom generation for ASL validation.

This module provides functions for generating synthetic arterial spin
labelling (ASL) data with known ground truth perfusion values.

Example:
    ```python
    from qmri.dro import perfusion

    # Generate pCASL data with known perfusion
    phantom = perfusion.generate_pcasl(
        perfusion_rate=60.0,  # ml/100g/min
        m0=1000.0,
        transit_time=1.0,
        snr=50,
        seed=42,
    )

    # Access the difference signal
    delta_m = phantom.control - phantom.label
    print(f"True CBF: {phantom.ground_truth['perfusion_rate'].value} ml/100g/min")
    ```
"""

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from qmri.dro._types import ASLPhantom, GroundTruth
from qmri.noise import gaussian, rician
from qmri.perfusion import gkm

__all__ = [
    "generate_pcasl",
]


@overload
def generate_pcasl(
    perfusion_rate: float,
    m0: float,
    *,
    transit_time: float = ...,
    label_duration: float = ...,
    post_label_delay: float = ...,
    label_efficiency: float = ...,
    partition_coefficient: float = ...,
    t1_blood: float = ...,
    t1_tissue: float = ...,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> ASLPhantom: ...


@overload
def generate_pcasl(
    perfusion_rate: NDArray[np.floating],
    m0: NDArray[np.floating] | float,
    *,
    transit_time: NDArray[np.floating] | float = ...,
    label_duration: float = ...,
    post_label_delay: float = ...,
    label_efficiency: float = ...,
    partition_coefficient: NDArray[np.floating] | float = ...,
    t1_blood: float = ...,
    t1_tissue: NDArray[np.floating] | float = ...,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> ASLPhantom: ...


def generate_pcasl(
    perfusion_rate: NDArray[np.floating] | float,
    m0: NDArray[np.floating] | float,
    *,
    transit_time: NDArray[np.floating] | float = 1.0,
    label_duration: float = 1.8,
    post_label_delay: float = 1.8,
    label_efficiency: float = 0.85,
    partition_coefficient: NDArray[np.floating] | float = 0.9,
    t1_blood: float = 1.65,
    t1_tissue: NDArray[np.floating] | float = 1.3,
    snr: float | None = None,
    noise_model: Literal["rician", "gaussian"] = "rician",
    seed: int | None = None,
) -> ASLPhantom:
    r"""Generate a pCASL phantom with known perfusion.

    Creates synthetic pseudo-continuous ASL (pCASL) control and label
    images using the General Kinetic Model with optional noise.

    Args:
        perfusion_rate: Cerebral blood flow (CBF) in ml/100g/min.
            Can be scalar or array for multi-voxel phantom.
        m0: Equilibrium magnetisation (M0) of tissue.
        transit_time: Arterial transit time (ATT) in seconds. Default is 1.0.
        label_duration: Duration of labelling pulse (tau) in seconds.
            Default is 1.8.
        post_label_delay: Post-label delay (PLD) in seconds. Default is 1.8.
        label_efficiency: Labelling efficiency (0-1). Default is 0.85.
        partition_coefficient: Blood-brain partition coefficient (lambda)
            in ml/g. Default is 0.9.
        t1_blood: T1 of arterial blood in seconds. Default is 1.65.
        t1_tissue: T1 of tissue in seconds. Default is 1.3.
        snr: Signal-to-noise ratio. If None, no noise is added.
        noise_model: Type of noise to add. Default is "rician".
        seed: Random seed for reproducibility.

    Returns:
        ASLPhantom containing control, label, and M0 images with ground truth.

    Raises:
        ValueError: If noise_model is not "rician" or "gaussian".

    Example:
        Single-voxel phantom:

        ```python
        from qmri.dro import perfusion

        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
            snr=50,
            seed=42,
        )
        print(f"Control: {phantom.control}")
        print(f"Label: {phantom.label}")
        print(f"Delta M: {phantom.control - phantom.label}")
        ```

        Multi-voxel phantom:

        ```python
        import numpy as np
        from qmri.dro import perfusion

        cbf_map = np.array([[40, 60],
                           [80, 100]])
        phantom = perfusion.generate_pcasl(
            perfusion_rate=cbf_map,
            m0=1000.0,
            snr=50,
            seed=42,
        )
        print(phantom.control.shape)  # (2, 2)
        ```

    Notes:
        The signal is calculated using the General Kinetic Model (GKM):

        $$\\Delta M(t) = 2 M_{0,b} f T_1' \\alpha e^{-\\delta/T_{1,b}}
        (1 - e^{-\\tau/T_1'}) e^{-(t-\\tau-\\delta)/T_1'}$$

        Default parameters are based on ASL White Paper recommendations
        for adult brain imaging at 3T.

        Typical perfusion values:

        - Grey matter: 50-80 ml/100g/min
        - White matter: 20-30 ml/100g/min
        - Tumour: Variable, often elevated
    """
    if noise_model not in ("rician", "gaussian"):
        msg = f"noise_model must be 'rician' or 'gaussian', got '{noise_model}'"
        raise ValueError(msg)

    cbf_arr = np.asarray(perfusion_rate, dtype=np.float64)
    m0_arr = np.asarray(m0, dtype=np.float64)
    att_arr = np.asarray(transit_time, dtype=np.float64)
    lam_arr = np.asarray(partition_coefficient, dtype=np.float64)
    t1_t_arr = np.asarray(t1_tissue, dtype=np.float64)

    # Calculate signal time (time after labelling start)
    signal_time = label_duration + post_label_delay

    # Calculate delta_m using GKM
    gkm_result = gkm.signal_gkm(
        perfusion_rate=cbf_arr,
        transit_time=att_arr,
        m0_tissue=m0_arr,
        label_duration=label_duration,
        signal_time=signal_time,
        label_efficiency=label_efficiency,
        partition_coefficient=lam_arr,
        t1_blood=t1_blood,
        t1_tissue=t1_t_arr,
        label_type="pcasl",
    )

    delta_m = gkm_result.delta_m

    # Control signal is just M0 (equilibrium magnetisation)
    # In reality it would be M0 * signal decay, but for simplicity
    # we use the M0 value as the control baseline
    control: NDArray[np.floating]
    label: NDArray[np.floating]
    if cbf_arr.ndim == 0:
        control = np.asarray(m0_arr, dtype=np.float64)
        label = np.asarray(m0_arr - delta_m, dtype=np.float64)
        m0_out = np.asarray(m0_arr, dtype=np.float64)
    else:
        control = np.broadcast_to(m0_arr, cbf_arr.shape).copy().astype(np.float64)
        label = (control - delta_m).astype(np.float64)
        m0_out = control.copy()

    # Add noise if requested
    if snr is not None:
        rng = np.random.default_rng(seed)
        if noise_model == "rician":
            control = rician.add_noise(control, snr=snr, rng=rng)
            label = rician.add_noise(label, snr=snr, rng=rng)
        else:
            control = gaussian.add_noise(control, snr=snr, rng=rng)
            label = gaussian.add_noise(label, snr=snr, rng=rng)

    # Build ground truth dictionary
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]] = {
        "perfusion_rate": GroundTruth(
            value=float(perfusion_rate) if cbf_arr.ndim == 0 else cbf_arr,
            units="ml/100g/min",
            description="Cerebral blood flow (CBF)",
        ),
        "transit_time": GroundTruth(
            value=float(transit_time) if att_arr.ndim == 0 else att_arr,
            units="s",
            description="Arterial transit time (ATT)",
        ),
        "m0": GroundTruth(
            value=float(m0) if np.asarray(m0).ndim == 0 else m0_arr,
            units="a.u.",
            description="Equilibrium magnetisation",
        ),
        "t1_tissue": GroundTruth(
            value=float(t1_tissue) if t1_t_arr.ndim == 0 else t1_t_arr,
            units="s",
            description="T1 of tissue",
        ),
    }

    # Store acquisition parameters
    acquisition_params = {
        "label_duration": label_duration,
        "post_label_delay": post_label_delay,
        "label_efficiency": label_efficiency,
        "partition_coefficient": (
            float(partition_coefficient)
            if np.asarray(partition_coefficient).ndim == 0
            else float(np.mean(lam_arr))
        ),
        "t1_blood": t1_blood,
    }

    return ASLPhantom(
        control=control,
        label=label,
        m0=m0_out,
        ground_truth=ground_truth,
        acquisition_params=acquisition_params,
        snr=snr,
        seed=seed,
    )
