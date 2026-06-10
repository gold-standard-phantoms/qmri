"""Perfusion models for ASL quantification.

This module provides functions for:

- pCASL/CASL perfusion quantification (White Paper equations)
- PASL perfusion quantification
- General Kinetic Model (GKM) signal generation

Submodules:
    asl: ASL quantification using White Paper equations
    gkm: General Kinetic Model signal generation

Example:
    ```python
    import numpy as np
    from qmri.perfusion import asl, gkm

    # Quantify perfusion from pCASL data
    control = np.array([1000.0])
    label = np.array([950.0])
    m0 = np.array([2000.0])
    result = asl.quantify_pcasl(
        control, label, m0,
        label_duration=1.8,
        post_label_delay=1.8,
    )

    # Generate GKM signal
    result = gkm.signal_gkm(
        perfusion_rate=60.0,
        transit_time=1.0,
        m0_tissue=1000.0,
        label_duration=1.8,
        signal_time=3.6,
        label_efficiency=0.85,
        partition_coefficient=0.9,
        t1_blood=1.65,
        t1_tissue=1.3,
        label_type="pcasl",
    )
    ```
"""

from qmri.perfusion import asl, gkm

__all__ = ["asl", "gkm"]
