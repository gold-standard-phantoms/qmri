r"""Gradient Echo (GRE) signal model.

This module provides the signal equation for Gradient Echo MRI sequences
with arbitrary excitation flip angle.

Signal Model
------------

**Gradient Echo (Spoiled GRE / FLASH)**:

$$S = M_0 \cdot \sin(\alpha)
\cdot \frac{1 - E_1}{1 - \cos(\alpha) \cdot E_1 - E_2 \cdot (E_1 - \cos(\alpha))}
\cdot E_2^*$$

where:

- $E_1 = \exp(-TR/T_1)$
- $E_2 = \exp(-TR/T_2)$
- $E_2^* = \exp(-TE/T_2^*)$
- $\alpha$ = excitation flip angle
- $M_0$ = equilibrium magnetisation
- $TR$ = repetition time (s)
- $TE$ = echo time (s)
- $T_1$ = longitudinal relaxation time (s)
- $T_2$ = transverse relaxation time (s)
- $T_2^*$ = effective transverse relaxation time (s)

The equation is from p246 in the book "MRI from Picture to Proton",
second edition, 2006, McRobbie et al.

References:
    .. [1] McRobbie, D.W., et al. "MRI from Picture to Proton."
           Cambridge University Press, 2nd edition, 2006.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "signal_gre",
]

# Tolerance for division safety
_TOL = 1e-12


def signal_gre(
    m0: NDArray[np.floating] | float,
    t1: NDArray[np.floating] | float,
    t2: NDArray[np.floating] | float,
    t2_star: NDArray[np.floating] | float,
    *,
    repetition_time: float,
    echo_time: float,
    flip_angle: float,
) -> NDArray[np.floating]:
    r"""Calculate Gradient Echo (GRE) signal.

    Implements the spoiled GRE signal equation:

    $$S = M_0 \cdot \sin(\alpha)
    \cdot \frac{1 - E_1}{1 - \cos(\alpha) \cdot E_1
    - E_2 \cdot (E_1 - \cos(\alpha))} \cdot E_2^*$$

    Args:
        m0: Equilibrium magnetisation.
        t1: Longitudinal relaxation time (T1) in seconds.
        t2: Transverse relaxation time (T2) in seconds.
        t2_star: Effective transverse relaxation time (T2*) in seconds,
            including time-invariant magnetic field inhomogeneities.
        repetition_time: Repetition time (TR) in seconds.
        echo_time: Echo time (TE) in seconds.
        flip_angle: Excitation flip angle in degrees.

    Returns:
        Signal intensity.

    Note:
        - T2* must be provided and accounts for field inhomogeneities.
        - For regions where T1, T2, or T2* is zero, the signal is set to zero
          to avoid division errors.

    Example:
        ```python
        import numpy as np
        from qmri.sequences import gre

        signal = gre.signal_gre(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            t2_star=0.05,
            repetition_time=0.025,
            echo_time=0.005,
            flip_angle=30.0,
        )
        ```
    """
    m0 = np.asarray(m0)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    t2_star = np.asarray(t2_star)

    # Broadcast arrays to common shape
    m0, t1, t2, t2_star = np.broadcast_arrays(m0, t1, t2, t2_star)

    # Convert flip angle to radians
    alpha = np.radians(flip_angle)

    # Create output array
    signal: NDArray[np.floating] = np.zeros_like(m0, dtype=np.float64)

    # Create mask for valid voxels (non-zero T2*)
    valid_mask = t2_star > _TOL

    if not np.any(valid_mask):
        return signal

    # Calculate exp(-TR/T1) - use 1.0 where T1=0 (fully relaxed)
    exp_tr_t1 = np.ones_like(t1, dtype=np.float64)
    t1_valid = t1 > _TOL
    exp_tr_t1[t1_valid] = np.exp(-repetition_time / t1[t1_valid])

    # Calculate exp(-TR/T2) - use 1.0 where T2=0
    exp_tr_t2 = np.ones_like(t2, dtype=np.float64)
    t2_valid = t2 > _TOL
    exp_tr_t2[t2_valid] = np.exp(-repetition_time / t2[t2_valid])

    # Calculate exp(-TE/T2*) for valid voxels only
    exp_te_t2_star = np.zeros_like(t2_star, dtype=np.float64)
    exp_te_t2_star[valid_mask] = np.exp(-echo_time / t2_star[valid_mask])

    # Calculate numerator and denominator
    numerator = m0 * (1 - exp_tr_t1)
    denominator = (
        1 - np.cos(alpha) * exp_tr_t1 - exp_tr_t2 * (exp_tr_t1 - np.cos(alpha))
    )

    # Safe division
    denom_valid = np.abs(denominator) > _TOL
    combined_valid = valid_mask & denom_valid

    signal[combined_valid] = (
        np.sin(alpha)
        * (numerator[combined_valid] / denominator[combined_valid])
        * exp_te_t2_star[combined_valid]
    )

    return signal
