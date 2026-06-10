"""Type definitions for Digital Reference Objects.

This module defines the dataclasses used to represent DRO phantoms
and their associated ground truth values.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GroundTruth",
    "DWIPhantom",
    "T1Phantom",
    "ASLPhantom",
]

T = TypeVar("T")


@dataclass(frozen=True)
class GroundTruth(Generic[T]):
    """Container for a ground truth parameter value.

    Attributes:
        value: The ground truth parameter value.
        units: Physical units of the parameter.
        description: Human-readable description of the parameter.

    Example:
        ```python
        from qmri.dro import GroundTruth

        adc_gt = GroundTruth(
            value=1e-3,
            units="mm²/s",
            description="Apparent Diffusion Coefficient",
        )
        ```
    """

    value: T
    units: str
    description: str


@dataclass(frozen=True)
class DWIPhantom:
    """Digital Reference Object for diffusion-weighted imaging.

    Contains synthetic DWI signal with known ground truth ADC values,
    along with acquisition parameters and optional noise characteristics.

    Attributes:
        signal: Signal intensities at each b-value. Shape is either (n_bvalues,)
            for single voxel or (..., n_bvalues) for multi-voxel.
        b_values: Diffusion weighting values in s/mm².
        ground_truth: Dictionary of ground truth parameters.
        snr: Signal-to-noise ratio used for noise generation, or None if noiseless.
        seed: Random seed used for reproducibility, or None if not seeded.

    Example:
        ```python
        from qmri.dro import dwi

        phantom = dwi.generate(adc=1e-3, snr=50, seed=42)
        print(phantom.ground_truth["adc"].value)  # 0.001
        print(phantom.signal.shape)  # (4,) for default b-values
        ```
    """

    signal: NDArray[np.floating]
    b_values: NDArray[np.floating]
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]]
    snr: float | None
    seed: int | None


@dataclass(frozen=True)
class T1Phantom:
    """Digital Reference Object for T1 relaxometry.

    Contains synthetic T1 signal with known ground truth values,
    supporting both inversion recovery (IR) and variable TR (VTR) methods.

    Attributes:
        signal: Signal intensities at each time point. Shape is either
            (n_timepoints,) for single voxel or (..., n_timepoints) for multi-voxel.
        time_points: Time points in seconds (TI for IR, TR for VTR).
        method: Acquisition method used ("ir" or "vtr").
        model: For IR method, the model used ("general" or "classical").
        repetition_time: TR value(s) for IR method, or None for VTR.
        ground_truth: Dictionary of ground truth parameters.
        snr: Signal-to-noise ratio used for noise generation, or None if noiseless.
        seed: Random seed used for reproducibility, or None if not seeded.

    Example:
        ```python
        from qmri.dro import relaxometry

        phantom = relaxometry.generate_t1_ir(
            t1=1.2,
            inversion_times=[0.1, 0.5, 1.0, 2.0],
            repetition_time=5.0,
            snr=100,
            seed=42,
        )
        print(phantom.ground_truth["t1"].value)  # 1.2
        ```
    """

    signal: NDArray[np.floating]
    time_points: NDArray[np.floating]
    method: str
    model: str | None
    repetition_time: float | None
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]]
    snr: float | None
    seed: int | None


@dataclass(frozen=True)
class ASLPhantom:
    """Digital Reference Object for arterial spin labelling.

    Contains synthetic ASL control and label images with known ground truth
    perfusion values, along with acquisition parameters.

    Attributes:
        control: Control image signal intensity.
        label: Label image signal intensity.
        m0: Equilibrium magnetisation (M0) image.
        ground_truth: Dictionary of ground truth parameters.
        acquisition_params: Dictionary of acquisition parameters used.
        snr: Signal-to-noise ratio used for noise generation, or None if noiseless.
        seed: Random seed used for reproducibility, or None if not seeded.

    Example:
        ```python
        from qmri.dro import perfusion

        phantom = perfusion.generate_pcasl(
            perfusion_rate=60.0,
            m0=1000.0,
            snr=50,
            seed=42,
        )
        delta_m = phantom.control - phantom.label
        print(phantom.ground_truth["perfusion_rate"].value)  # 60.0
        ```
    """

    control: NDArray[np.floating]
    label: NDArray[np.floating]
    m0: NDArray[np.floating]
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]]
    acquisition_params: dict[str, float]
    snr: float | None
    seed: int | None
