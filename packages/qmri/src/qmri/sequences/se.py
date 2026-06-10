r"""Spin Echo (SE) signal model.

This module provides the signal equation for Spin Echo MRI sequences
assuming perfect 90 degree excitation and 180 degree refocusing pulses.

Signal Model
------------

**Spin Echo**:

$$S = M_0 \cdot (1 - \exp(-TR/T_1)) \cdot \exp(-TE/T_2)$$

where:

- $M_0$ = equilibrium magnetisation
- $TR$ = repetition time (s)
- $TE$ = echo time (s)
- $T_1$ = longitudinal relaxation time (s)
- $T_2$ = transverse relaxation time (s)

This equation assumes 90 degree excitation and 180 degree refocusing pulses.
The equation is from p69 in the book "MRI from Picture to Proton",
second edition, 2006, McRobbie et al.

References:
    .. [1] McRobbie, D.W., et al. "MRI from Picture to Proton."
           Cambridge University Press, 2nd edition, 2006.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "signal_se",
]

# Tolerance for division safety
_TOL = 1e-12


def signal_se(
    m0: NDArray[np.floating] | float,
    t1: NDArray[np.floating] | float,
    t2: NDArray[np.floating] | float,
    *,
    repetition_time: float,
    echo_time: float,
) -> NDArray[np.floating]:
    r"""Calculate Spin Echo (SE) signal.

    Implements the standard spin echo signal equation assuming
    90 degree excitation and 180 degree refocusing pulses:

    $$S = M_0 \cdot (1 - \exp(-TR/T_1)) \cdot \exp(-TE/T_2)$$

    Args:
        m0: Equilibrium magnetisation.
        t1: Longitudinal relaxation time (T1) in seconds.
        t2: Transverse relaxation time (T2) in seconds.
        repetition_time: Repetition time (TR) in seconds.
        echo_time: Echo time (TE) in seconds.

    Returns:
        Signal intensity.

    Note:
        - Assumes perfect 90 degree excitation and 180 degree refocusing pulses.
        - For regions where T1 or T2 is zero, the signal is set to zero
          to avoid division errors.

    Example:
        ```python
        import numpy as np
        from qmri.sequences import se

        signal = se.signal_se(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=3.0,
            echo_time=0.08,
        )
        ```
    """
    m0 = np.asarray(m0)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    # Broadcast arrays to common shape
    m0, t1, t2 = np.broadcast_arrays(m0, t1, t2)

    # Create output array
    signal: NDArray[np.floating] = np.zeros_like(m0, dtype=np.float64)

    # Create mask for valid voxels (non-zero T2)
    valid_mask = t2 > _TOL

    if not np.any(valid_mask):
        return signal

    # Calculate exp(-TR/T1) with safe division
    exp_tr_t1 = np.ones_like(t1, dtype=np.float64)
    t1_valid = t1 > _TOL
    exp_tr_t1[t1_valid] = np.exp(-repetition_time / t1[t1_valid])

    # Calculate exp(-TE/T2) for valid voxels
    exp_te_t2 = np.zeros_like(t2, dtype=np.float64)
    exp_te_t2[valid_mask] = np.exp(-echo_time / t2[valid_mask])

    # Calculate signal only where T2 is valid
    signal = m0 * (1 - exp_tr_t1) * exp_te_t2

    return signal
