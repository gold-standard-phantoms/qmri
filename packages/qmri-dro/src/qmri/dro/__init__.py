"""Digital Reference Objects (DROs) for quantitative MRI validation.

This package provides tools for generating synthetic MRI data with known
ground truth parameters for validation, testing, and education.

Modules:
    dwi: DWI phantom generation for ADC validation
    relaxometry: T1/T2 phantom generation
    perfusion: ASL phantom generation

Example:
    ```python
    from qmri.dro import dwi
    from qmri.diffusion import adc

    # Generate a phantom with known ADC
    phantom = dwi.generate(adc=1e-3, s0=1000, snr=50, seed=42)

    # Fit and compare to ground truth
    result = adc.fit(phantom.signal, phantom.b_values)
    print(f"True ADC: {phantom.ground_truth['adc'].value:.2e}")
    print(f"Fitted ADC: {result.adc:.2e}")
    ```
"""

__author__ = "Gold Standard Phantoms"

from qmri.dro._types import (
    ASLPhantom,
    DWIPhantom,
    GroundTruth,
    T1Phantom,
)

__all__ = [
    "ASLPhantom",
    "DWIPhantom",
    "GroundTruth",
    "T1Phantom",
]
