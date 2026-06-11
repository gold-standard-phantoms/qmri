r"""Arterial Spin Labelling (ASL) quantification pipeline.

This pipeline produces a cerebral blood flow (CBF) map from an ASL NIfTI using
the White Paper consensus equations in :mod:`qmri.perfusion.asl`.

At a high level it:

- Loads a 4D ASL NIfTI and the per-volume ``asl_context`` describing each
  volume as ``"control"``, ``"label"`` or ``"m0scan"`` (read from an explicit
  argument, a sibling ``*_aslcontext.tsv`` file, or the JSON sidecar).
- Averages the control, label and M0 volumes (or takes M0 from a separate file).
- Resolves the labelling parameters with the precedence
  *explicit argument > JSON sidecar > sensible defaults*.
- Quantifies CBF (ml/100g/min) using the pCASL/CASL or PASL equation, depending
  on ``ArterialSpinLabelingType``.
- Optionally writes a CBF map NIfTI and a JSON report.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from qmri.io import NiftiImage, load_nifti_image, load_sidecar, save_nifti
from qmri.perfusion.asl import quantify_pasl, quantify_pcasl
from qmri.pipelines._common import strip_nifti_suffix

__all__ = [
    "ASLQuantificationReport",
    "LabelType",
    "run_asl_quantification",
]

LabelType = Literal["pcasl", "casl", "pasl"]

# Default labelling efficiency by label type (White Paper recommendations).
_DEFAULT_LABEL_EFFICIENCY: dict[str, float] = {
    "pcasl": 0.85,
    "casl": 0.85,
    "pasl": 0.98,
}
# Default partition coefficient (ml/g).
_DEFAULT_PARTITION_COEFFICIENT = 0.9
# Default T1 of arterial blood (s) by magnetic field strength (Tesla).
_DEFAULT_T1_BLOOD_BY_FIELD: dict[float, float] = {1.5: 1.35, 3.0: 1.65}
_FALLBACK_T1_BLOOD = 1.65


@dataclass
class ASLQuantificationReport:
    """Structured report from :func:`run_asl_quantification`.

    Attributes:
        asl_file: The input ASL image path.
        m0_file: A separate M0 image path, or ``None`` if M0 came from the ASL
            file's ``m0scan`` volumes.
        output_file: Path of the saved CBF map, or ``None`` if outputs were not
            written.
        label_type: The resolved ASL labelling type used for quantification.
        asl_context: The per-volume context labels used to split the ASL image.
        quantification_parameters: The resolved labelling parameters actually
            used for quantification.
        n_valid_voxels: Number of voxels with non-zero M0 (i.e. voxels where CBF
            is defined) included in the statistics below.
        perfusion_mean: Mean CBF (ml/100g/min) over the valid voxels.
        perfusion_std: Standard deviation of CBF (ml/100g/min) over the valid
            voxels.
        processing_date: Local date/time the pipeline finished.
        processing_time_seconds: Wall-clock processing time in seconds.
    """

    asl_file: Path
    m0_file: Path | None
    output_file: Path | None
    label_type: str
    asl_context: list[str]
    quantification_parameters: dict[str, float]
    n_valid_voxels: int
    perfusion_mean: float
    perfusion_std: float
    processing_date: str
    processing_time_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary of the report."""
        return {
            "asl_file": str(self.asl_file),
            "m0_file": str(self.m0_file) if self.m0_file else None,
            "output_file": str(self.output_file) if self.output_file else None,
            "label_type": self.label_type,
            "asl_context": self.asl_context,
            "quantification_parameters": self.quantification_parameters,
            "n_valid_voxels": self.n_valid_voxels,
            "perfusion_mean": self.perfusion_mean,
            "perfusion_std": self.perfusion_std,
            "processing_date": self.processing_date,
            "processing_time_seconds": self.processing_time_seconds,
        }


def _aslcontext_path(asl_path: Path) -> Path:
    """Return the BIDS sibling ``*_aslcontext.tsv`` path for an ASL image."""
    stem = strip_nifti_suffix(asl_path)
    name = stem.name
    if name.endswith("_asl"):
        name = name[: -len("_asl")]
    return stem.with_name(f"{name}_aslcontext.tsv")


def _load_aslcontext_tsv(path: Path) -> list[str]:
    """Parse a single-column BIDS ``aslcontext.tsv`` into per-volume labels.

    A ``volume_type`` header row, if present, is ignored.
    """
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if rows and rows[0].lower() == "volume_type":
        rows = rows[1:]
    return [row.lower() for row in rows]


def _resolve_asl_context(
    asl_path: Path,
    asl_context: Sequence[str] | None,
    sidecar: dict[str, Any],
) -> list[str]:
    """Resolve the per-volume context labels (explicit > tsv > sidecar)."""
    if asl_context is not None:
        return [label.lower() for label in asl_context]
    tsv_path = _aslcontext_path(asl_path)
    if tsv_path.exists():
        return _load_aslcontext_tsv(tsv_path)
    sidecar_context = sidecar.get("asl_context") or sidecar.get("ASLContext")
    if isinstance(sidecar_context, Sequence) and not isinstance(sidecar_context, str):
        return [str(label).lower() for label in sidecar_context]
    msg = (
        "Could not determine the ASL context. Pass asl_context explicitly, or "
        f"provide a '{tsv_path.name}' file alongside the ASL image."
    )
    raise ValueError(msg)


def _mean_of_volumes(
    data: NDArray[np.floating], context: Sequence[str], wanted: str
) -> NDArray[np.floating] | None:
    """Mean over the last-axis volumes whose context label equals ``wanted``."""
    indices = [i for i, label in enumerate(context) if label == wanted]
    if not indices:
        return None
    return np.asarray(np.mean(data[..., indices], axis=-1), dtype=np.float64)


def _resolve(
    explicit: float | None, sidecar: dict[str, Any], key: str, default: float | None
) -> float | None:
    """Resolve a numeric parameter (explicit > sidecar > default)."""
    if explicit is not None:
        return explicit
    value = sidecar.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _resolve_t1_blood(
    explicit: float | None,
    sidecar: dict[str, Any],
    magnetic_field_tesla: float | None,
) -> float:
    """Resolve T1 of arterial blood (explicit > sidecar > field default)."""
    if explicit is not None:
        return explicit
    sidecar_value = sidecar.get("T1ArterialBlood")
    if isinstance(sidecar_value, (int, float)):
        return float(sidecar_value)
    field_strength = magnetic_field_tesla
    if field_strength is None:
        sidecar_field = sidecar.get("MagneticFieldStrength")
        if isinstance(sidecar_field, (int, float)):
            field_strength = float(sidecar_field)
    if field_strength is not None and field_strength in _DEFAULT_T1_BLOOD_BY_FIELD:
        return _DEFAULT_T1_BLOOD_BY_FIELD[field_strength]
    return _FALLBACK_T1_BLOOD


def run_asl_quantification(
    asl_file: str | Path,
    *,
    asl_context: Sequence[str] | None = None,
    m0_file: str | Path | None = None,
    label_type: LabelType | None = None,
    post_label_delay: float | None = None,
    label_duration: float | None = None,
    bolus_duration: float | None = None,
    label_efficiency: float | None = None,
    t1_blood: float | None = None,
    partition_coefficient: float | None = None,
    magnetic_field_tesla: float | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    save_outputs: bool = True,
) -> tuple[NiftiImage, ASLQuantificationReport]:
    r"""Quantify cerebral blood flow (CBF) from an ASL image.

    The control, label and M0 volumes are identified from ``asl_context`` and
    averaged. CBF is then computed with the White Paper pCASL/CASL or PASL
    equation, depending on the resolved labelling type.

    Labelling parameters are resolved with the precedence
    *explicit argument > JSON sidecar > default*. Sidecar keys follow BIDS
    (``ArterialSpinLabelingType``, ``PostLabelingDelay``, ``LabelingDuration``,
    ``BolusCutOffDelayTime``, ``LabelingEfficiency``,
    ``BloodBrainPartitionCoefficient``, ``T1ArterialBlood``,
    ``MagneticFieldStrength``).

    Args:
        asl_file: A 4D ASL NIfTI containing control/label (and optionally M0)
            volumes.
        asl_context: Per-volume labels (``"control"``, ``"label"``,
            ``"m0scan"``). If ``None``, read from a sibling ``*_aslcontext.tsv``
            or the JSON sidecar.
        m0_file: Optional separate M0 NIfTI. If omitted, M0 is taken from the
            ``m0scan`` volumes of ``asl_file``.
        label_type: ``"pcasl"``, ``"casl"`` or ``"pasl"``. If ``None``, read
            from the sidecar's ``ArterialSpinLabelingType``.
        post_label_delay: Post-label delay (s) for pCASL/CASL, or the inversion
            time (TI, s) for PASL.
        label_duration: Label duration (s); pCASL/CASL only.
        bolus_duration: Bolus duration (TI1, s); PASL only.
        label_efficiency: Labelling efficiency. Defaults to 0.85 (pCASL/CASL) or
            0.98 (PASL).
        t1_blood: T1 of arterial blood (s). Defaults from field strength
            (1.35 s at 1.5 T, 1.65 s at 3 T) or 1.65 s.
        partition_coefficient: Blood-brain partition coefficient (ml/g).
            Defaults to 0.9.
        magnetic_field_tesla: Field strength (T), used only to pick a default
            ``t1_blood`` when neither it nor the sidecar provides one.
        output_dir: Directory for output files. Defaults to the directory of
            ``asl_file``. Only used when ``save_outputs`` is ``True``.
        output_prefix: Prefix for output filenames. Defaults to the stem of
            ``asl_file``.
        save_outputs: If ``True`` (default), write ``<prefix>_cbf.nii.gz`` and
            ``<prefix>_report.json``.

    Returns:
        A tuple ``(cbf_image, report)`` where ``cbf_image`` is a
        :class:`qmri.io.NiftiImage` of the CBF map (ml/100g/min) co-located with
        the input, and ``report`` is an :class:`ASLQuantificationReport`.

    Raises:
        ValueError: If the ASL context cannot be determined or mismatches the
            number of volumes; if control, label or M0 volumes are missing; if
            the labelling type is unknown; or if a required labelling parameter
            (post-label delay, label duration, or bolus duration) is missing.
    """
    start_time = time.perf_counter()

    asl_path = Path(asl_file)
    asl_image = load_nifti_image(asl_path)
    sidecar = load_sidecar(asl_path)

    if asl_image.data.ndim != 4:
        msg = f"ASL image must be 4D (x, y, z, volume): {asl_path}"
        raise ValueError(msg)

    context = _resolve_asl_context(asl_path, asl_context, sidecar)
    n_volumes = asl_image.data.shape[-1]
    if len(context) != n_volumes:
        msg = (
            f"ASL context length ({len(context)}) does not match the number of "
            f"volumes in the image ({n_volumes})."
        )
        raise ValueError(msg)

    control = _mean_of_volumes(asl_image.data, context, "control")
    label = _mean_of_volumes(asl_image.data, context, "label")
    if control is None or label is None:
        msg = "ASL context must contain at least one 'control' and one 'label' volume."
        raise ValueError(msg)

    m0_path: Path | None = None
    if m0_file is not None:
        m0_path = Path(m0_file)
        m0 = load_nifti_image(m0_path).data
        if m0.shape != control.shape:
            msg = (
                f"M0 image shape {m0.shape} does not match the ASL volume shape "
                f"{control.shape}."
            )
            raise ValueError(msg)
    else:
        m0_volumes = _mean_of_volumes(asl_image.data, context, "m0scan")
        if m0_volumes is None:
            msg = (
                "No 'm0scan' volume found in the ASL context and no m0_file was "
                "provided."
            )
            raise ValueError(msg)
        m0 = m0_volumes

    resolved_label_type = (label_type or sidecar.get("ArterialSpinLabelingType") or "")
    resolved_label_type = str(resolved_label_type).lower()
    if resolved_label_type not in ("pcasl", "casl", "pasl"):
        msg = (
            "ASL labelling type must be 'pcasl', 'casl' or 'pasl'. Pass "
            "label_type explicitly or set 'ArterialSpinLabelingType' in the "
            f"sidecar (got {resolved_label_type!r})."
        )
        raise ValueError(msg)

    efficiency = _resolve(
        label_efficiency,
        sidecar,
        "LabelingEfficiency",
        _DEFAULT_LABEL_EFFICIENCY[resolved_label_type],
    )
    partition = _resolve(
        partition_coefficient,
        sidecar,
        "BloodBrainPartitionCoefficient",
        _DEFAULT_PARTITION_COEFFICIENT,
    )
    t1b = _resolve_t1_blood(t1_blood, sidecar, magnetic_field_tesla)
    pld = _resolve(post_label_delay, sidecar, "PostLabelingDelay", None)

    # mypy/ruff: defaults above are never None, so narrow for the call below.
    assert efficiency is not None
    assert partition is not None

    parameters: dict[str, float] = {
        "label_efficiency": efficiency,
        "t1_blood": t1b,
        "partition_coefficient": partition,
    }

    if resolved_label_type in ("pcasl", "casl"):
        duration = _resolve(label_duration, sidecar, "LabelingDuration", None)
        if pld is None or duration is None:
            msg = (
                "pCASL/CASL quantification requires both a post-label delay and a "
                "label duration (via arguments or the sidecar)."
            )
            raise ValueError(msg)
        parameters["post_label_delay"] = pld
        parameters["label_duration"] = duration
        result = quantify_pcasl(
            control,
            label,
            m0,
            label_duration=duration,
            post_label_delay=pld,
            label_efficiency=efficiency,
            t1_blood=t1b,
            partition_coefficient=partition,
        )
    else:  # pasl
        bolus = _resolve(bolus_duration, sidecar, "BolusCutOffDelayTime", None)
        if pld is None or bolus is None:
            msg = (
                "PASL quantification requires both an inversion time "
                "(post_label_delay) and a bolus duration (via arguments or the "
                "sidecar)."
            )
            raise ValueError(msg)
        parameters["inversion_time"] = pld
        parameters["bolus_duration"] = bolus
        result = quantify_pasl(
            control,
            label,
            m0,
            bolus_duration=bolus,
            inversion_time=pld,
            label_efficiency=efficiency,
            t1_blood=t1b,
            partition_coefficient=partition,
        )

    perfusion = result.perfusion
    cbf_image = NiftiImage(
        data=perfusion,
        header=asl_image.header,
        affine=asl_image.affine,
    )

    valid = np.abs(m0) > 0
    valid_values = perfusion[valid]
    perfusion_mean = float(np.mean(valid_values)) if valid_values.size else 0.0
    perfusion_std = float(np.std(valid_values)) if valid_values.size else 0.0

    out_dir: Path | None = None
    prefix: str | None = None
    output_path: Path | None = None
    if save_outputs:
        out_dir = Path(output_dir) if output_dir is not None else asl_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or strip_nifti_suffix(asl_path).stem
        output_path = out_dir / f"{prefix}_cbf.nii.gz"
        save_nifti(
            perfusion,
            output_path,
            header=asl_image.header,
            affine=asl_image.affine,
        )

    report = ASLQuantificationReport(
        asl_file=asl_path,
        m0_file=m0_path,
        output_file=output_path,
        label_type=resolved_label_type,
        asl_context=list(context),
        quantification_parameters=parameters,
        n_valid_voxels=int(valid_values.size),
        perfusion_mean=perfusion_mean,
        perfusion_std=perfusion_std,
        processing_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        processing_time_seconds=time.perf_counter() - start_time,
    )

    if save_outputs and out_dir is not None and prefix is not None:
        report_path = out_dir / f"{prefix}_report.json"
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report.to_dict(), report_file, indent=2)

    return cbf_image, report
