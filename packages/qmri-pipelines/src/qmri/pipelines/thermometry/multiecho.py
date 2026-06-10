r"""Multi-echo MR thermometry pipeline.

This pipeline estimates temperature from multi-echo *magnitude* images and a
segmentation/label map using the dual-resonance model
(:func:`qmri.thermometry.fit_multiecho_thermometry_image`).

At a high level it:

- Loads one or more 4D multi-echo magnitude NIfTI images (the echo dimension is
  the last axis).
- Loads a 3D segmentation/label map co-located with the images.
- Loads echo times (in seconds) for each image, concatenates them, and sorts all
  echoes by echo time.
- Determines the magnetic field strength $B_0$ (Tesla) from an explicit argument
  or from a JSON sidecar (``ImagingFrequency`` in MHz or ``MagneticFieldStrength``
  in Tesla).
- Runs region-wise, voxel-wise or bootstrap region-wise fitting and (optionally)
  writes a temperature map NIfTI and a JSON report.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from qmri.constants import GAMMA_H
from qmri.io import (
    NiftiImage,
    load_nifti_image,
    load_sidecar,
    save_nifti,
)
from qmri.thermometry.multiecho import (
    DfInitMethod,
    RegionAnalysisMethod,
    RegionThermometryResult,
    fit_multiecho_thermometry_image,
)

__all__ = [
    "MultiEchoThermometryReport",
    "run_multiecho_thermometry",
]


@dataclass
class MultiEchoThermometryReport:
    """Structured report from :func:`run_multiecho_thermometry`.

    Attributes:
        input_files: The multi-echo input image paths, in the order supplied.
        segmentation_file: The segmentation/label-map path.
        output_file: Path of the saved temperature map, or ``None`` if outputs
            were not written.
        magnetic_field_tesla: Magnetic field strength used for the calibration.
        analysis_method: The analysis method that was run.
        n_bootstrap: Number of bootstrap samples (``None`` unless the method was
            ``regionwise_bootstrap``).
        echo_times: All echo times in seconds, concatenated and sorted ascending.
        regions: Per-region results, one per non-empty segmentation label.
        acquisition_date_time: Acquisition date/time string per input image
            (``"Unknown"`` when no sidecar metadata was found).
        processing_date: Local date/time the pipeline finished.
        processing_time_seconds: Wall-clock processing time in seconds.
    """

    input_files: list[Path]
    segmentation_file: Path
    output_file: Path | None
    magnetic_field_tesla: float
    analysis_method: str
    n_bootstrap: int | None
    echo_times: list[float]
    regions: list[RegionThermometryResult]
    acquisition_date_time: list[str]
    processing_date: str
    processing_time_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary of the report."""
        return {
            "input_files": [str(f) for f in self.input_files],
            "segmentation_file": str(self.segmentation_file),
            "output_file": str(self.output_file) if self.output_file else None,
            "acquisition_date_time": self.acquisition_date_time,
            "processing_date": self.processing_date,
            "processing_time_seconds": self.processing_time_seconds,
            "magnetic_field_tesla": self.magnetic_field_tesla,
            "analysis_method": self.analysis_method,
            "n_bootstrap": self.n_bootstrap,
            "echo_times": self.echo_times,
            "report": [region.to_dict() for region in self.regions],
        }


def _strip_nifti_suffix(path: Path) -> Path:
    """Return ``path`` with a trailing ``.nii`` or ``.nii.gz`` suffix removed."""
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])
    if name.endswith(".nii"):
        return path.with_name(name[:-4])
    return path


def _load_echo_times(path: Path) -> NDArray[np.float64]:
    """Load a 1D list of echo times (seconds) from a text file."""
    values = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    if values.ndim != 1:
        msg = f"Echo-times file must contain a 1D list of values: {path}"
        raise ValueError(msg)
    return values


def _magnetic_field_from_sidecar(sidecar: dict[str, object]) -> float | None:
    """Derive $B_0$ (Tesla) from a BIDS/JSON sidecar, or ``None`` if absent.

    ``ImagingFrequency`` (in MHz) takes precedence over
    ``MagneticFieldStrength`` (in Tesla).
    """
    imaging_frequency = sidecar.get("ImagingFrequency")
    if isinstance(imaging_frequency, (int, float)):
        return float(imaging_frequency) / (GAMMA_H / 1e6)
    field_strength = sidecar.get("MagneticFieldStrength")
    if isinstance(field_strength, (int, float)):
        return float(field_strength)
    return None


def run_multiecho_thermometry(
    multiecho_files: Sequence[str | Path],
    segmentation_file: str | Path,
    echo_times_files: Sequence[str | Path],
    *,
    method: RegionAnalysisMethod = "regionwise",
    n_bootstrap: int = 100,
    df_init: DfInitMethod = "multistart",
    magnetic_field_tesla: float | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    save_outputs: bool = True,
) -> tuple[NiftiImage, MultiEchoThermometryReport]:
    r"""Run multi-echo thermometry over a set of images and a segmentation.

    Args:
        multiecho_files: One or more 4D multi-echo magnitude NIfTI files. The
            echo dimension must be the last axis. All images must share the same
            spatial shape and affine.
        segmentation_file: A 3D segmentation/label-map NIfTI co-located with the
            multi-echo data. Label ``0`` is treated as background.
        echo_times_files: Echo-time text files (seconds), one per multi-echo
            image, in the same order. Each file's length must match the number of
            echoes in its image.
        method: Analysis method — ``"regionwise"`` (default), ``"voxelwise"`` or
            ``"regionwise_bootstrap"``. See
            :func:`qmri.thermometry.fit_multiecho_thermometry_image`.
        n_bootstrap: Number of bootstrap samples (``regionwise_bootstrap`` only).
        df_init: Frequency starting-value strategy — ``"multistart"`` (default),
            ``"fixed"`` or ``"lombscargle"``. See
            :data:`qmri.thermometry.DfInitMethod`.
        magnetic_field_tesla: Magnetic field strength in Tesla. If ``None``, it is
            read from the first JSON sidecar containing ``ImagingFrequency`` or
            ``MagneticFieldStrength``.
        output_dir: Directory for output files. Defaults to the directory of the
            first input image. Only used when ``save_outputs`` is ``True``.
        output_prefix: Prefix for output filenames. Defaults to the first input
            image's stem.
        save_outputs: If ``True`` (default), write ``<prefix>_temperature_map.nii.gz``
            and ``<prefix>_report.json``.

    Returns:
        A tuple ``(temperature_image, report)`` where ``temperature_image`` is a
        :class:`qmri.io.NiftiImage` of the 3D temperature map (°C) co-located with
        the segmentation, and ``report`` is a :class:`MultiEchoThermometryReport`.

    Raises:
        ValueError: If no images are supplied, the number of images and
            echo-time files differ, image dimensions/affines are inconsistent,
            an image's echo count does not match its echo times, or the magnetic
            field strength cannot be determined.
    """
    start_time = time.perf_counter()

    multiecho_paths = [Path(f) for f in multiecho_files]
    echo_times_paths = [Path(f) for f in echo_times_files]
    segmentation_path = Path(segmentation_file)

    if not multiecho_paths:
        msg = "At least one multi-echo image must be provided."
        raise ValueError(msg)
    if len(multiecho_paths) != len(echo_times_paths):
        msg = (
            f"Number of multi-echo images ({len(multiecho_paths)}) must match the "
            f"number of echo-time files ({len(echo_times_paths)})."
        )
        raise ValueError(msg)

    echo_times_per_image = [_load_echo_times(p) for p in echo_times_paths]
    images = [load_nifti_image(p) for p in multiecho_paths]
    reference = images[0]

    for image, path in zip(images, multiecho_paths, strict=True):
        if image.data.ndim != 4:
            msg = f"Multi-echo image must be 4D (x, y, z, echo): {path}"
            raise ValueError(msg)
        if image.data.shape[:3] != reference.data.shape[:3]:
            msg = "All multi-echo images must share the same spatial shape."
            raise ValueError(msg)
        if not np.allclose(image.affine, reference.affine):
            msg = "All multi-echo images must share the same affine."
            raise ValueError(msg)
    for image, echo_times, path in zip(
        images, echo_times_per_image, multiecho_paths, strict=True
    ):
        if image.data.shape[-1] != echo_times.shape[0]:
            msg = (
                f"Image {path} has {image.data.shape[-1]} echoes but "
                f"{echo_times.shape[0]} echo times were provided."
            )
            raise ValueError(msg)

    # Determine B0 and acquisition times from JSON sidecars.
    sidecars = [load_sidecar(p) for p in multiecho_paths]
    acquisition_date_time: list[str] = []
    detected_field: float | None = None
    for sidecar in sidecars:
        if detected_field is None:
            detected_field = _magnetic_field_from_sidecar(sidecar)
        if "AcquisitionDateTime" in sidecar:
            acquisition_date_time.append(str(sidecar["AcquisitionDateTime"]))
        elif "AcquisitionTime" in sidecar:
            acquisition_date_time.append(str(sidecar["AcquisitionTime"]))
        else:
            acquisition_date_time.append("Unknown")

    field_tesla = (
        magnetic_field_tesla if magnetic_field_tesla is not None else detected_field
    )
    if field_tesla is None:
        msg = (
            "Magnetic field strength could not be determined. Pass "
            "magnetic_field_tesla explicitly, or provide a JSON sidecar with "
            "'ImagingFrequency' (MHz) or 'MagneticFieldStrength' (Tesla)."
        )
        raise ValueError(msg)

    # Concatenate echoes across images and sort by echo time.
    signal = np.concatenate([image.data for image in images], axis=3)
    all_echo_times = np.concatenate(echo_times_per_image)
    order = np.argsort(all_echo_times)
    signal = signal[:, :, :, order]
    sorted_echo_times = all_echo_times[order]

    segmentation_image = load_nifti_image(segmentation_path)
    if (
        segmentation_image.data.ndim != 3
        or segmentation_image.data.shape != reference.data.shape[:3]
    ):
        msg = (
            "Segmentation must be 3D and match the spatial shape of the "
            "multi-echo images."
        )
        raise ValueError(msg)

    temperature_map, regions = fit_multiecho_thermometry_image(
        signal=signal,
        segmentation=segmentation_image.data,
        echo_times=sorted_echo_times,
        magnetic_field_tesla=field_tesla,
        method=method,
        n_bootstrap=n_bootstrap,
        df_init=df_init,
    )

    temperature_image = NiftiImage(
        data=temperature_map,
        header=reference.header,
        affine=reference.affine,
    )

    out_dir: Path | None = None
    prefix: str | None = None
    output_path: Path | None = None
    if save_outputs:
        out_dir = (
            Path(output_dir) if output_dir is not None else multiecho_paths[0].parent
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or _strip_nifti_suffix(multiecho_paths[0]).stem
        output_path = out_dir / f"{prefix}_temperature_map.nii.gz"
        save_nifti(
            temperature_map,
            output_path,
            header=reference.header,
            affine=reference.affine,
        )

    report = MultiEchoThermometryReport(
        input_files=multiecho_paths,
        segmentation_file=segmentation_path,
        output_file=output_path,
        magnetic_field_tesla=float(field_tesla),
        analysis_method=method,
        n_bootstrap=n_bootstrap if method == "regionwise_bootstrap" else None,
        echo_times=[float(t) for t in sorted_echo_times],
        regions=regions,
        acquisition_date_time=acquisition_date_time,
        processing_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        processing_time_seconds=time.perf_counter() - start_time,
    )

    if save_outputs and out_dir is not None and prefix is not None:
        report_path = out_dir / f"{prefix}_report.json"
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report.to_dict(), report_file, indent=2)

    return temperature_image, report
