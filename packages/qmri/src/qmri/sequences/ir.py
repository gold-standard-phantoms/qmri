r"""Inversion Recovery (IR) signal model.

This module provides the signal equation for Inversion Recovery MRI sequences
with arbitrary inversion and excitation flip angles.

Signal Model
------------

**Inversion Recovery**:

$$S = \sin(\alpha_1)
\cdot \frac{M_0 (1 - (1 - \cos\alpha_2) E_{TI} - \cos\alpha_2 \cdot E_1)}
{1 - \cos\alpha_1 \cos\alpha_2 E_1} \cdot E_2$$

where:

- $E_1 = \exp(-TR/T_1)$
- $E_{TI} = \exp(-TI/T_1)$
- $E_2 = \exp(-TE/T_2)$
- $\alpha_1$ = excitation flip angle
- $\alpha_2$ = inversion flip angle
- $M_0$ = equilibrium magnetisation
- $TR$ = repetition time (s)
- $TI$ = inversion time (s)
- $TE$ = echo time (s)
- $T_1$ = longitudinal relaxation time (s)
- $T_2$ = transverse relaxation time (s)

The equation is from equation 7 in Tofts, P. "T1-weighted DCE Imaging Concepts:
Modelling, Acquisition and Analysis", ISMRM 2009.

References:
    .. [1] Tofts, P. "T1-weighted DCE Imaging Concepts: Modelling, Acquisition
           and Analysis." ISMRM 2009.
           http://www.paul-tofts-phd.org.uk/talks/ismrm2009_rt.pdf
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "signal_ir",
]

# Tolerance for division safety
_TOL = 1e-12


def signal_ir(
    m0: NDArray[np.floating] | float,
    t1: NDArray[np.floating] | float,
    t2: NDArray[np.floating] | float,
    *,
    repetition_time: float,
    echo_time: float,
    inversion_time: float,
    excitation_flip_angle: float = 90.0,
    inversion_flip_angle: float = 180.0,
) -> NDArray[np.floating]:
    r"""Calculate Inversion Recovery (IR) signal.

    Implements the IR signal equation with arbitrary flip angles:

    $$S = \sin(\alpha_1)
    \cdot \frac{M_0 \cdot (1 - (1 - \cos(\alpha_2)) \cdot E_{TI}
    - \cos(\alpha_2) \cdot E_1)}
    {1 - \cos(\alpha_1) \cdot \cos(\alpha_2) \cdot E_1} \cdot E_2$$

    Args:
        m0: Equilibrium magnetisation.
        t1: Longitudinal relaxation time (T1) in seconds.
        t2: Transverse relaxation time (T2) in seconds.
        repetition_time: Repetition time (TR) in seconds.
        echo_time: Echo time (TE) in seconds.
        inversion_time: Inversion time (TI) in seconds.
        excitation_flip_angle: Excitation pulse flip angle in degrees (default 90.0).
        inversion_flip_angle: Inversion pulse flip angle in degrees (default 180.0).

    Returns:
        Signal intensity.

    Note:
        - Default flip angles correspond to ideal IR sequence (90 degree
          excitation, 180 degree inversion).
        - For regions where T1 or T2 is zero, the signal is set to zero
          to avoid division errors.

    Example:
        ```python
        import numpy as np
        from qmri.sequences import ir

        signal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
        )

        # With non-ideal flip angles:
        signal = ir.signal_ir(
            m0=1000.0,
            t1=1.0,
            t2=0.1,
            repetition_time=5.0,
            echo_time=0.01,
            inversion_time=0.5,
            excitation_flip_angle=85.0,
            inversion_flip_angle=175.0,
        )
        ```
    """
    m0 = np.asarray(m0)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    # Broadcast arrays to common shape
    m0, t1, t2 = np.broadcast_arrays(m0, t1, t2)

    # Convert flip angles to radians
    alpha1 = np.radians(excitation_flip_angle)
    alpha2 = np.radians(inversion_flip_angle)

    # Create output array
    signal: NDArray[np.floating] = np.zeros_like(m0, dtype=np.float64)

    # Create mask for valid voxels (non-zero T2)
    valid_mask = t2 > _TOL

    if not np.any(valid_mask):
        return signal

    # Calculate exp(-TR/T1) - use 1.0 where T1=0 (fully relaxed)
    exp_tr_t1 = np.ones_like(t1, dtype=np.float64)
    t1_valid = t1 > _TOL
    exp_tr_t1[t1_valid] = np.exp(-repetition_time / t1[t1_valid])

    # Calculate exp(-TI/T1) - use 1.0 where T1=0
    exp_ti_t1 = np.ones_like(t1, dtype=np.float64)
    exp_ti_t1[t1_valid] = np.exp(-inversion_time / t1[t1_valid])

    # Calculate exp(-TE/T2) for valid voxels only
    exp_te_t2 = np.zeros_like(t2, dtype=np.float64)
    exp_te_t2[valid_mask] = np.exp(-echo_time / t2[valid_mask])

    # Calculate numerator and denominator
    numerator = m0 * (1 - (1 - np.cos(alpha2)) * exp_ti_t1 - np.cos(alpha2) * exp_tr_t1)
    denominator = 1 - np.cos(alpha1) * np.cos(alpha2) * exp_tr_t1

    # Safe division - combine T2 validity with denominator validity
    denom_valid = np.abs(denominator) > _TOL
    combined_valid = valid_mask & denom_valid

    signal[combined_valid] = (
        np.sin(alpha1)
        * (numerator[combined_valid] / denominator[combined_valid])
        * exp_te_t2[combined_valid]
    )

    return signal
