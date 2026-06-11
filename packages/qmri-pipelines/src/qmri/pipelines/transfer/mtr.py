r"""Magnetisation Transfer Ratio (MTR) pipeline.

This pipeline calculates an MTR map from images acquired with and without
bound-pool saturation, using :func:`qmri.transfer.calculate_mtr`.

At a high level it:

- Loads the saturated and unsaturated images, either as two separate 3D NIfTI
  files or as a single 4D file whose last axis holds the two volumes (ordered
  ``[unsaturated, saturated]``).
- Checks the two images are co-located (matching spatial shape and affine).
- Computes the MTR (percentage units) voxel-wise.
- Optionally writes an MTR map NIfTI and a JSON report.

The MTR is defined as

$$\text{MTR} = 100 \cdot \frac{S_0 - S_s}{S_0}$$

where $S_0$ is the unsaturated signal and $S_s$ the saturated signal.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from qmri.io import NiftiImage, load_nifti_image, save_nifti
from qmri.pipelines._common import strip_nifti_suffix
from qmri.transfer import calculate_mtr

__all__ = [
    "MTRReport",
    "run_mtr",
]


@dataclass
class MTRReport:
    """Structured report from :func:`run_mtr`.

    Attributes:
        input_files: The input image paths. Two entries (unsaturated, saturated)
            in separate-file mode, or one entry in combined-file mode.
        mode: ``"separate"`` (two files) or ``"combined"`` (single 4D file).
        output_file: Path of the saved MTR map, or ``None`` if outputs were not
            written.
        n_valid_voxels: Number of voxels with a non-zero unsaturated signal
            (i.e. voxels where MTR is defined) included in the statistics below.
        mtr_mean: Mean MTR (pu) over the valid voxels.
        mtr_std: Standard deviation of MTR (pu) over the valid voxels.
        mtr_min: Minimum MTR (pu) over the valid voxels.
        mtr_max: Maximum MTR (pu) over the valid voxels.
        processing_date: Local date/time the pipeline finished.
        processing_time_seconds: Wall-clock processing time in seconds.
    """

    input_files: list[Path]
    mode: str
    output_file: Path | None
    n_valid_voxels: int
    mtr_mean: float
    mtr_std: float
    mtr_min: float
    mtr_max: float
    processing_date: str
    processing_time_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary of the report."""
        return {
            "input_files": [str(f) for f in self.input_files],
            "mode": self.mode,
            "output_file": str(self.output_file) if self.output_file else None,
            "n_valid_voxels": self.n_valid_voxels,
            "mtr_mean": self.mtr_mean,
            "mtr_std": self.mtr_std,
            "mtr_min": self.mtr_min,
            "mtr_max": self.mtr_max,
            "processing_date": self.processing_date,
            "processing_time_seconds": self.processing_time_seconds,
        }


def _load_combined(
    saturated_path: Path,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NiftiImage]:
    """Split a single 4D file into (unsaturated, saturated) volumes.

    The last axis must hold exactly two volumes ordered ``[unsaturated,
    saturated]``.
    """
    image = load_nifti_image(saturated_path)
    if image.data.ndim != 4 or image.data.shape[-1] != 2:
        msg = (
            "A combined MTR file must be 4D with exactly two volumes "
            f"(unsaturated, saturated) in the last axis: {saturated_path} has "
            f"shape {image.data.shape}."
        )
        raise ValueError(msg)
    signal_nosat = np.ascontiguousarray(image.data[..., 0])
    signal_sat = np.ascontiguousarray(image.data[..., 1])
    return signal_nosat, signal_sat, image


def _load_separate(
    saturated_path: Path,
    unsaturated_path: Path,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NiftiImage]:
    """Load and co-locate two separate saturated/unsaturated images."""
    saturated = load_nifti_image(saturated_path)
    unsaturated = load_nifti_image(unsaturated_path)
    if saturated.data.shape != unsaturated.data.shape:
        msg = (
            "Saturated and unsaturated images must have the same shape "
            f"(got {saturated.data.shape} and {unsaturated.data.shape})."
        )
        raise ValueError(msg)
    if not np.allclose(saturated.affine, unsaturated.affine):
        msg = "Saturated and unsaturated images must share the same affine."
        raise ValueError(msg)
    return unsaturated.data, saturated.data, saturated


def run_mtr(
    saturated_file: str | Path,
    unsaturated_file: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    save_outputs: bool = True,
) -> tuple[NiftiImage, MTRReport]:
    r"""Calculate a Magnetisation Transfer Ratio (MTR) map.

    Two input modes are supported:

    - **Separate** (``unsaturated_file`` given): ``saturated_file`` and
      ``unsaturated_file`` are individual NIfTI images of the same shape and
      affine.
    - **Combined** (``unsaturated_file`` is ``None``): ``saturated_file`` is a
      single 4D NIfTI whose last axis holds two volumes ordered
      ``[unsaturated, saturated]``.

    Args:
        saturated_file: The image with bound-pool saturation, or — in combined
            mode — a 4D file containing both the unsaturated and saturated
            volumes (in that order).
        unsaturated_file: The image without bound-pool saturation. Omit to use
            combined mode.
        output_dir: Directory for output files. Defaults to the directory of
            ``saturated_file``. Only used when ``save_outputs`` is ``True``.
        output_prefix: Prefix for output filenames. Defaults to the stem of
            ``saturated_file``.
        save_outputs: If ``True`` (default), write ``<prefix>_mtr_map.nii.gz``
            and ``<prefix>_report.json``.

    Returns:
        A tuple ``(mtr_image, report)`` where ``mtr_image`` is a
        :class:`qmri.io.NiftiImage` of the MTR map in percentage units (pu),
        co-located with the input, and ``report`` is an :class:`MTRReport`.

    Raises:
        ValueError: If a combined file is not 4D with two volumes, or the two
            separate images differ in shape or affine.
    """
    start_time = time.perf_counter()

    saturated_path = Path(saturated_file)

    if unsaturated_file is None:
        mode = "combined"
        input_files = [saturated_path]
        signal_nosat, signal_sat, reference = _load_combined(saturated_path)
    else:
        mode = "separate"
        unsaturated_path = Path(unsaturated_file)
        input_files = [unsaturated_path, saturated_path]
        signal_nosat, signal_sat, reference = _load_separate(
            saturated_path, unsaturated_path
        )

    mtr_map = calculate_mtr(signal_nosat, signal_sat).mtr

    mtr_image = NiftiImage(
        data=mtr_map,
        header=reference.header,
        affine=reference.affine,
    )

    valid = np.abs(signal_nosat) > 0
    valid_values = mtr_map[valid]
    if valid_values.size:
        mtr_mean = float(np.mean(valid_values))
        mtr_std = float(np.std(valid_values))
        mtr_min = float(np.min(valid_values))
        mtr_max = float(np.max(valid_values))
    else:
        mtr_mean = mtr_std = mtr_min = mtr_max = 0.0

    out_dir: Path | None = None
    prefix: str | None = None
    output_path: Path | None = None
    if save_outputs:
        out_dir = Path(output_dir) if output_dir is not None else saturated_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or strip_nifti_suffix(saturated_path).stem
        output_path = out_dir / f"{prefix}_mtr_map.nii.gz"
        save_nifti(
            mtr_map,
            output_path,
            header=reference.header,
            affine=reference.affine,
        )

    report = MTRReport(
        input_files=input_files,
        mode=mode,
        output_file=output_path,
        n_valid_voxels=int(valid_values.size),
        mtr_mean=mtr_mean,
        mtr_std=mtr_std,
        mtr_min=mtr_min,
        mtr_max=mtr_max,
        processing_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        processing_time_seconds=time.perf_counter() - start_time,
    )

    if save_outputs and out_dir is not None and prefix is not None:
        report_path = out_dir / f"{prefix}_report.json"
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report.to_dict(), report_file, indent=2)

    return mtr_image, report
