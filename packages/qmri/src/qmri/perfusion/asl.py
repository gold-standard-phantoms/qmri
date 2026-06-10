r"""Arterial Spin Labelling (ASL) quantification.

This module provides functions for ASL perfusion quantification using
the White Paper consensus equations.

Signal Models
-------------

**(pseudo)Continuous ASL (pCASL/CASL)**:

$$f = \frac{6000 \cdot \lambda \cdot \Delta M \cdot e^{PLD/T_{1,b}}}
{2 \cdot \alpha \cdot T_{1,b} \cdot M_0 \cdot (1 - e^{-\tau/T_{1,b}})}$$

**Pulsed ASL (PASL)**:

$$f = \frac{6000 \cdot \lambda \cdot \Delta M \cdot e^{TI/T_{1,b}}}
{2 \cdot \alpha \cdot TI_1 \cdot M_0}$$

where:

- $f$ = perfusion rate (ml/100g/min)
- $\Delta M$ = control - label signal
- $\lambda$ = blood-brain partition coefficient (ml/g)
- $\tau$ = label duration (s)
- $PLD$ = post-label delay (s)
- $TI$ = inversion time (s)
- $TI_1$ = bolus duration (s)
- $T_{1,b}$ = T1 of arterial blood (s)
- $\alpha$ = labelling efficiency
- $M_0$ = equilibrium magnetisation

References:
    .. [1] Alsop, D.C., et al. "Recommended implementation of arterial
           spin-labeled perfusion MRI for clinical applications."
           MRM 73(1):102-116, 2015.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ASLResult",
    "quantify_pcasl",
    "quantify_pasl",
]

# Tolerance for M0 division
_M0_TOL = 1e-6


@dataclass(frozen=True)
class ASLResult:
    """Result of ASL quantification.

    Attributes:
        perfusion: Cerebral blood flow (CBF) in ml/100g/min.
    """

    perfusion: NDArray[np.floating]


def quantify_pcasl(
    control: NDArray[np.floating],
    label: NDArray[np.floating],
    m0: NDArray[np.floating],
    *,
    label_duration: float,
    post_label_delay: float,
    label_efficiency: float = 0.85,
    t1_blood: float = 1.65,
    partition_coefficient: float = 0.9,
) -> ASLResult:
    r"""Quantify perfusion from pCASL/CASL data.

    Implements the White Paper equation for (pseudo)continuous ASL:

    $$f = \frac{6000 \cdot \lambda \cdot (C - L) \cdot e^{PLD/T_{1,b}}}
    {2 \cdot \alpha \cdot T_{1,b} \cdot M_0 \cdot (1 - e^{-\tau/T_{1,b}})}$$

    Args:
        control: Control image signal.
        label: Label image signal.
        m0: Equilibrium magnetisation image.
        label_duration: Duration of labelling pulse (tau) in seconds.
        post_label_delay: Post-label delay (PLD) in seconds.
        label_efficiency: Labelling efficiency (default 0.85 for pCASL).
        t1_blood: T1 of arterial blood in seconds (default 1.65s at 3T).
        partition_coefficient: Blood-brain partition coefficient in ml/g (default 0.9).

    Returns:
        Quantified perfusion in ml/100g/min.

    Example:
        ```python
        import numpy as np
        from qmri.perfusion import asl

        control = np.array([1000.0])
        label = np.array([950.0])
        m0 = np.array([2000.0])
        result = asl.quantify_pcasl(
            control, label, m0,
            label_duration=1.8,
            post_label_delay=1.8,
        )
        ```
    """
    control = np.asarray(control)
    label = np.asarray(label)
    m0 = np.asarray(m0)

    delta_m = control - label

    numerator = (
        6000 * partition_coefficient * delta_m * np.exp(post_label_delay / t1_blood)
    )
    denominator = (
        2 * label_efficiency * t1_blood * m0 * (1 - np.exp(-label_duration / t1_blood))
    )

    # Safe division avoiding divide-by-zero
    perfusion: NDArray[np.floating] = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(m0, dtype=np.float64),
        where=np.abs(m0) >= _M0_TOL,
    )

    return ASLResult(perfusion=perfusion)


def quantify_pasl(
    control: NDArray[np.floating],
    label: NDArray[np.floating],
    m0: NDArray[np.floating],
    *,
    bolus_duration: float,
    inversion_time: float,
    label_efficiency: float = 0.98,
    t1_blood: float = 1.65,
    partition_coefficient: float = 0.9,
) -> ASLResult:
    r"""Quantify perfusion from PASL data.

    Implements the White Paper equation for pulsed ASL:

    $$f = \frac{6000 \cdot \lambda \cdot (C - L) \cdot e^{TI/T_{1,b}}}
    {2 \cdot \alpha \cdot TI_1 \cdot M_0}$$

    Args:
        control: Control image signal.
        label: Label image signal.
        m0: Equilibrium magnetisation image.
        bolus_duration: Bolus duration (TI1) in seconds.
        inversion_time: Inversion time (TI) in seconds.
        label_efficiency: Labelling efficiency (default 0.98 for PASL).
        t1_blood: T1 of arterial blood in seconds (default 1.65s at 3T).
        partition_coefficient: Blood-brain partition coefficient in ml/g (default 0.9).

    Returns:
        Quantified perfusion in ml/100g/min.

    Example:
        ```python
        import numpy as np
        from qmri.perfusion import asl

        control = np.array([1000.0])
        label = np.array([950.0])
        m0 = np.array([2000.0])
        result = asl.quantify_pasl(
            control, label, m0,
            bolus_duration=0.7,
            inversion_time=1.8,
        )
        ```
    """
    control = np.asarray(control)
    label = np.asarray(label)
    m0 = np.asarray(m0)

    delta_m = control - label

    numerator = (
        6000 * partition_coefficient * delta_m * np.exp(inversion_time / t1_blood)
    )
    denominator = 2 * label_efficiency * bolus_duration * m0

    # Safe division avoiding divide-by-zero
    perfusion: NDArray[np.floating] = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(m0, dtype=np.float64),
        where=np.abs(m0) >= _M0_TOL,
    )

    return ASLResult(perfusion=perfusion)
