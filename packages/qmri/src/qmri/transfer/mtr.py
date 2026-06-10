r"""Magnetisation Transfer Ratio (MTR) calculation.

This module provides functions for calculating the Magnetisation Transfer
Ratio (MTR) from images with and without bound pool saturation.

Signal Model:
    The Magnetisation Transfer Ratio is calculated as:

    $$\text{MTR} = 100 \cdot \frac{S_0 - S_s}{S_0}$$

    where:

    - $S_0$ = signal without bound pool saturation
    - $S_s$ = signal with bound pool saturation

    The result is expressed in percentage units (pu).

References:
    .. [1] Tofts, P.S., et al. "Quantitative Magnetization Transfer Imaging."
           Quantitative MRI of the Brain, Chapter 8, 2003.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MTRResult",
    "calculate_mtr",
]

# Tolerance for M0 division to avoid divide-by-zero
_M0_TOL = 1e-6


@dataclass(frozen=True)
class MTRResult:
    """Result of MTR calculation.

    Attributes:
        mtr: Magnetisation Transfer Ratio in percentage units (pu).
    """

    mtr: NDArray[np.floating]


def calculate_mtr(
    signal_nosat: NDArray[np.floating],
    signal_sat: NDArray[np.floating],
) -> MTRResult:
    r"""Calculate Magnetisation Transfer Ratio (MTR).

    Implements the standard MTR equation:

    $$\text{MTR} = 100 \cdot \frac{S_0 - S_s}{S_0}$$

    Args:
        signal_nosat: Signal without bound pool saturation (M0 or reference signal).
        signal_sat: Signal with bound pool saturation applied.

    Returns:
        Calculated MTR in percentage units (pu).

    - Input arrays must have the same shape.
    - Voxels where signal_nosat is zero (or near-zero) will have MTR = 0.
    - Typical MTR values in brain tissue range from 20-50%.

    Example:
        ```python
        import numpy as np
        from qmri.transfer import mtr

        signal_nosat = np.array([1000.0, 800.0, 600.0])
        signal_sat = np.array([700.0, 560.0, 420.0])
        result = mtr.calculate_mtr(signal_nosat, signal_sat)
        print(result.mtr)
        # [30.0, 30.0, 30.0]
        ```
    """
    signal_nosat = np.asarray(signal_nosat)
    signal_sat = np.asarray(signal_sat)

    if signal_nosat.shape != signal_sat.shape:
        msg = (
            f"Input arrays must have the same shape. "
            f"Got signal_nosat: {signal_nosat.shape}, signal_sat: {signal_sat.shape}"
        )
        raise ValueError(msg)

    # Calculate MTR with safe division
    mtr_value: NDArray[np.floating] = np.divide(
        100.0 * (signal_nosat - signal_sat),
        signal_nosat,
        out=np.zeros_like(signal_nosat, dtype=np.float64),
        where=np.abs(signal_nosat) >= _M0_TOL,
    )

    return MTRResult(mtr=mtr_value)
