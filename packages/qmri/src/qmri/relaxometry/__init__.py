"""Relaxometry fitting and signal models (T1, T2).

This module provides functions for quantitative T1 and T2 mapping.

Submodules:
    t1: T1 relaxometry (inversion recovery, variable TR)
    t2: T2 relaxometry (mono-exponential decay)

Example:
    ```python
    import numpy as np
    from qmri.relaxometry import t1, t2

    # T1 fitting with inversion recovery
    ti = np.array([0.1, 0.5, 1.0, 2.0])
    signal_t1 = t1.signal_ir(s0=1000, t1=1.2, inversion_times=ti,
                             repetition_times=5.0)
    result_t1 = t1.fit_ir(signal_t1, ti, repetition_times=5.0)

    # T2 fitting
    te = np.array([0.01, 0.02, 0.04, 0.08])
    signal_t2 = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)
    result_t2 = t2.fit(signal_t2, te, skip_echoes=0)
    ```
"""

from qmri.relaxometry import t1, t2

__all__ = ["t1", "t2"]
