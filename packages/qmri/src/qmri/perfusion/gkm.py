r"""General Kinetic Model (GKM) for ASL signal generation.

This module implements the Buxton General Kinetic Model for calculating
the ASL perfusion difference signal.

Theory
------

The GKM considers the magnetisation difference $\Delta M(t)$ as the
sum over delivery history weighted by residue and relaxation:

$$\Delta M(t) = 2 M_{0,b} f \lbrace c(t) \ast [r(t) \cdot m(t)] \rbrace$$

where:

- $c(t)$ = delivery function (plug flow)
- $r(t) = e^{-ft/\lambda}$ = residue function
- $m(t) = e^{-t/T_1}$ = magnetisation relaxation

The delivery function differs between labelling schemes:

- **CASL/pCASL**: $c(t) = \alpha e^{-\Delta t / T_{1,b}}$ during labelling
- **PASL**: $c(t) = \alpha e^{-t / T_{1,b}}$ during labelling

References:
    .. [1] Buxton, R.B., et al. "A general kinetic model for quantitative
           perfusion imaging with arterial spin labeling."
           MRM 40(3):383-396, 1998.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GKMResult",
    "signal_gkm",
    "signal_gkm_simplified",
]


@dataclass(frozen=True)
class GKMResult:
    """Result of GKM signal calculation.

    Attributes:
        delta_m: Magnetisation difference (control - label).
    """

    delta_m: NDArray[np.floating]


def signal_gkm(
    perfusion_rate: NDArray[np.floating] | float,
    transit_time: NDArray[np.floating] | float,
    m0_tissue: NDArray[np.floating] | float,
    *,
    label_duration: float,
    signal_time: float,
    label_efficiency: float,
    partition_coefficient: NDArray[np.floating] | float,
    t1_blood: float,
    t1_tissue: NDArray[np.floating] | float,
    label_type: Literal["pcasl", "casl", "pasl"],
) -> GKMResult:
    """Calculate ASL signal using the full General Kinetic Model.

    Args:
        perfusion_rate: Perfusion rate (CBF) in ml/100g/min.
        transit_time: Arterial transit time (ATT) in seconds.
        m0_tissue: Equilibrium magnetisation of tissue.
        label_duration: Duration of labelling pulse in seconds.
        signal_time: Time after labelling start to calculate signal.
        label_efficiency: Labelling efficiency (0-1).
        partition_coefficient: Blood-brain partition coefficient (lambda) in ml/g.
        t1_blood: T1 of arterial blood in seconds.
        t1_tissue: T1 of tissue in seconds.
        label_type: ASL labelling scheme ("pcasl", "casl", or "pasl").

    Returns:
        Calculated magnetisation difference.

    Example:
        ```python
        import numpy as np
        from qmri.perfusion import gkm

        result = gkm.signal_gkm(
            perfusion_rate=60.0,  # ml/100g/min
            transit_time=1.0,     # s
            m0_tissue=1000.0,
            label_duration=1.8,
            signal_time=3.6,      # PLD + tau
            label_efficiency=0.85,
            partition_coefficient=0.9,
            t1_blood=1.65,
            t1_tissue=1.3,
            label_type="pcasl",
        )
        ```
    """
    # Convert to arrays
    f_raw = np.asarray(perfusion_rate) / 6000  # Convert to SI (s^-1)
    att_raw = np.asarray(transit_time)
    m0_raw = np.asarray(m0_tissue)
    lam_raw = np.asarray(partition_coefficient)
    t1_t_raw = np.asarray(t1_tissue)

    tau = label_duration
    t = signal_time
    alpha = label_efficiency
    t1_b = t1_blood

    # Broadcast all arrays to common shape
    bc = np.broadcast(f_raw, att_raw, m0_raw, lam_raw, t1_t_raw)
    broadcast_shape = bc.shape
    if broadcast_shape:
        f = np.broadcast_to(f_raw, broadcast_shape).copy()
        att = np.broadcast_to(att_raw, broadcast_shape).copy()
        m0 = np.broadcast_to(m0_raw, broadcast_shape).copy()
        lam = np.broadcast_to(lam_raw, broadcast_shape).copy()
        t1_t = np.broadcast_to(t1_t_raw, broadcast_shape).copy()
    else:
        f, att, m0, lam, t1_t = f_raw, att_raw, m0_raw, lam_raw, t1_t_raw

    # Create output array with correct shape
    out_shape = broadcast_shape if broadcast_shape else ()
    zeros_out: NDArray[np.floating] = np.zeros(out_shape, dtype=np.float64)

    # Calculate M0 of blood
    m0_b: NDArray[np.floating] = np.divide(
        m0, lam, out=zeros_out.copy(), where=lam != 0
    )

    # Calculate T1'
    f_over_lam: NDArray[np.floating] = np.divide(
        f, lam, out=zeros_out.copy(), where=lam != 0
    )
    one_over_t1_t: NDArray[np.floating] = np.divide(
        1.0, t1_t, out=zeros_out.copy(), where=t1_t != 0
    )
    denom = one_over_t1_t + f_over_lam
    t1_prime: NDArray[np.floating] = np.divide(
        1.0, denom, out=zeros_out.copy(), where=denom != 0
    )

    # Arrival state masks
    not_arrived = t <= att
    arriving = (att < t) & (t < att + tau)
    arrived = t >= att + tau

    # Initialise output
    delta_m = zeros_out.copy()

    if label_type.lower() == "pasl":
        # PASL GKM equations
        k: NDArray[np.floating] = (1 / t1_b if t1_b != 0 else 0) - np.divide(
            1.0, t1_prime, out=zeros_out.copy(), where=t1_prime != 0
        )

        # q_pasl for arriving state
        num_arr = np.exp(k * t) * (np.exp(-k * att) - np.exp(-k * t))
        den_arr = k * (t - att)
        q_arriving: NDArray[np.floating] = np.divide(
            num_arr, den_arr, out=zeros_out.copy(), where=den_arr != 0
        )

        # q_pasl for arrived state
        num_arrd = np.exp(k * t) * (np.exp(-k * att) - np.exp(-k * (att + tau)))
        den_arrd = k * tau
        q_arrived: NDArray[np.floating] = np.divide(
            num_arrd, den_arrd, out=zeros_out.copy(), where=den_arrd != 0
        )

        exp_t1b = np.exp(-t / t1_b) if t1_b > 0 else 0.0

        dm_arriving = 2 * m0_b * f * (t - att) * alpha * exp_t1b * q_arriving
        dm_arrived = 2 * m0_b * f * alpha * tau * exp_t1b * q_arrived

    elif label_type.lower() in ("casl", "pcasl"):
        # CASL/pCASL GKM equations
        q_ss_arriving: NDArray[np.floating] = 1 - np.exp(
            -np.divide(t - att, t1_prime, out=zeros_out.copy(), where=t1_prime != 0)
        )
        q_ss_arrived: NDArray[np.floating] = 1 - np.exp(
            -np.divide(tau, t1_prime, out=zeros_out.copy(), where=t1_prime != 0)
        )

        exp_att_t1b = np.exp(-att / t1_b) if t1_b != 0 else zeros_out.copy()

        dm_arriving = 2 * m0_b * f * t1_prime * alpha * exp_att_t1b * q_ss_arriving
        dm_arrived = (
            2
            * m0_b
            * f
            * t1_prime
            * alpha
            * exp_att_t1b
            * np.exp(
                -np.divide(
                    t - tau - att,
                    t1_prime,
                    out=zeros_out.copy(),
                    where=t1_prime != 0,
                )
            )
            * q_ss_arrived
        )
    else:
        msg = f"Invalid label_type: {label_type}"
        raise ValueError(msg)

    # Combine arrival states
    if isinstance(delta_m, np.ndarray):
        delta_m[not_arrived] = 0.0
        delta_m[arriving] = dm_arriving[arriving]
        delta_m[arrived] = dm_arrived[arrived]
    else:
        if arrived:
            delta_m = float(dm_arrived)
        elif arriving:
            delta_m = float(dm_arriving)
        else:
            delta_m = 0.0

    result_arr: NDArray[np.floating] = np.asarray(delta_m)
    return GKMResult(delta_m=result_arr)


def signal_gkm_simplified(
    perfusion_rate: NDArray[np.floating] | float,
    transit_time: NDArray[np.floating] | float,
    m0_tissue: NDArray[np.floating] | float,
    *,
    label_duration: float,
    signal_time: float,
    label_efficiency: float,
    partition_coefficient: NDArray[np.floating] | float,
    t1_blood: float,
    label_type: Literal["pcasl", "casl", "pasl"],
) -> GKMResult:
    """Calculate ASL signal using simplified (White Paper) GKM.

    This is derived from the single-subtraction quantification equations
    and assumes signal is acquired after the bolus has fully arrived.

    Args:
        perfusion_rate: Perfusion rate (CBF) in ml/100g/min.
        transit_time: Arterial transit time (ATT) in seconds.
        m0_tissue: Equilibrium magnetisation of tissue.
        label_duration: Duration of labelling pulse in seconds.
        signal_time: Time after labelling start to calculate signal.
        label_efficiency: Labelling efficiency (0-1).
        partition_coefficient: Blood-brain partition coefficient (lambda) in ml/g.
        t1_blood: T1 of arterial blood in seconds.
        label_type: ASL labelling scheme ("pcasl", "casl", or "pasl").

    Returns:
        Calculated magnetisation difference.
    """
    # Convert to arrays
    f = np.asarray(perfusion_rate) / 6000  # Convert to SI (s^-1)
    att = np.asarray(transit_time)
    m0 = np.asarray(m0_tissue)
    lam = np.asarray(partition_coefficient)

    tau = label_duration
    t = signal_time
    alpha = label_efficiency
    t1_b = t1_blood

    # Calculate M0 of blood
    m0_b: NDArray[np.floating] = np.divide(
        m0, lam, out=np.zeros_like(lam, dtype=np.float64), where=lam != 0
    )

    # Check if signal arrived
    arrived = t >= att + tau

    if label_type.lower() in ("casl", "pcasl"):
        # Simplified pCASL equation
        dm = (
            2
            * m0_b
            * f
            * t1_b
            * alpha
            * (1 - np.exp(-tau / t1_b))
            * np.exp(-(t - tau) / t1_b)
        )
    elif label_type.lower() == "pasl":
        # Simplified PASL equation
        dm = 2 * m0_b * f * tau * alpha * np.exp(-t / t1_b)
    else:
        msg = f"Invalid label_type: {label_type}"
        raise ValueError(msg)

    # Zero where not arrived
    if isinstance(arrived, np.ndarray):
        dm = np.where(arrived, dm, 0.0)
    elif not arrived:
        dm = np.zeros_like(dm)

    result_arr: NDArray[np.floating] = np.asarray(dm)
    return GKMResult(delta_m=result_arr)
