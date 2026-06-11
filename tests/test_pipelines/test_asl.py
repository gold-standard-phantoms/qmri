"""End-to-end tests for the ASL quantification pipeline.

These port the behaviour previously covered by ``mrimagetools``'
``asl_quantification`` pipeline tests, rewritten against the qmri pipeline API
and dropping the dependency on the old filter framework. The White Paper
quantification maths is unit-tested in ``tests/test_perfusion/test_asl.py``;
here we exercise volume identification, parameter resolution (argument vs
sidecar vs default), output writing, and input validation.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from qmri.io import load_nifti_image, save_nifti
from qmri.perfusion.asl import quantify_pasl, quantify_pcasl
from qmri.pipelines.perfusion import ASLQuantificationReport, run_asl_quantification

_SHAPE = (4, 4, 4)
_CONTROL = 1.0
_LABEL = 1.0 - 0.01
_M0 = 1.0


def _volume(label: str) -> np.ndarray:
    """Return a uniform volume for a given context label."""
    value = {"control": _CONTROL, "label": _LABEL, "m0scan": _M0}[label]
    return np.full(_SHAPE, value, dtype=np.float64)


def _write_asl(
    directory: Path,
    context: Sequence[str],
    *,
    sidecar: dict[str, object] | None = None,
    write_aslcontext: bool = False,
    name: str = "sub-01_asl",
) -> Path:
    """Write a 4D ASL image plus optional sidecar / aslcontext.tsv.

    Returns the path to the ASL NIfTI.
    """
    data = np.stack([_volume(label) for label in context], axis=-1)
    asl_path = directory / f"{name}.nii.gz"
    save_nifti(data, asl_path, affine=np.eye(4))

    if sidecar is not None:
        (directory / f"{name}.json").write_text(json.dumps(sidecar))
    if write_aslcontext:
        tsv = "volume_type\n" + "\n".join(context)
        (directory / f"{name.removesuffix('_asl')}_aslcontext.tsv").write_text(tsv)

    return asl_path


def _expected_pcasl(**kwargs: float) -> np.ndarray:
    return quantify_pcasl(
        _volume("control"), _volume("label"), _volume("m0scan"), **kwargs
    ).perfusion


class TestPcaslQuantification:
    """pCASL quantification driven by explicit arguments and by the sidecar."""

    def test_explicit_arguments(self, tmp_path: Path) -> None:
        """Explicit labelling parameters drive the quantification."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        cbf_image, report = run_asl_quantification(
            asl_path,
            asl_context=["m0scan", "control", "label"],
            label_type="pcasl",
            post_label_delay=2.0,
            label_duration=1.5,
            label_efficiency=0.95,
            t1_blood=1.65,
            partition_coefficient=0.9,
            save_outputs=False,
        )

        assert isinstance(report, ASLQuantificationReport)
        assert report.label_type == "pcasl"
        np.testing.assert_allclose(
            cbf_image.data,
            _expected_pcasl(
                label_duration=1.5,
                post_label_delay=2.0,
                label_efficiency=0.95,
                t1_blood=1.65,
                partition_coefficient=0.9,
            ),
        )
        assert report.quantification_parameters == {
            "label_efficiency": 0.95,
            "t1_blood": 1.65,
            "partition_coefficient": 0.9,
            "post_label_delay": 2.0,
            "label_duration": 1.5,
        }

    def test_parameters_and_context_from_sidecar(self, tmp_path: Path) -> None:
        """Labelling parameters and context are read from BIDS metadata."""
        asl_path = _write_asl(
            tmp_path,
            ["m0scan", "control", "label"],
            sidecar={
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 2.0,
                "LabelingDuration": 1.5,
                "LabelingEfficiency": 0.95,
                "BloodBrainPartitionCoefficient": 0.9,
                "T1ArterialBlood": 1.65,
                "MagneticFieldStrength": 3.0,
            },
            write_aslcontext=True,
        )

        # No explicit parameters at all - everything from the sidecar / tsv.
        cbf_image, report = run_asl_quantification(asl_path, save_outputs=False)

        assert report.label_type == "pcasl"
        assert report.asl_context == ["m0scan", "control", "label"]
        np.testing.assert_allclose(
            cbf_image.data,
            _expected_pcasl(
                label_duration=1.5,
                post_label_delay=2.0,
                label_efficiency=0.95,
                t1_blood=1.65,
                partition_coefficient=0.9,
            ),
        )

    def test_explicit_argument_overrides_sidecar(self, tmp_path: Path) -> None:
        """An explicit argument takes precedence over the sidecar value."""
        asl_path = _write_asl(
            tmp_path,
            ["m0scan", "control", "label"],
            sidecar={
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 2.0,
                "LabelingDuration": 1.5,
            },
            write_aslcontext=True,
        )

        _, report = run_asl_quantification(
            asl_path, post_label_delay=1.8, save_outputs=False
        )

        assert report.quantification_parameters["post_label_delay"] == 1.8

    def test_defaults_applied(self, tmp_path: Path) -> None:
        """Missing efficiency/partition/T1 fall back to documented defaults."""
        asl_path = _write_asl(
            tmp_path,
            ["m0scan", "control", "label"],
            sidecar={
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 1.8,
                "LabelingDuration": 1.8,
                "MagneticFieldStrength": 3.0,
            },
            write_aslcontext=True,
        )

        _, report = run_asl_quantification(asl_path, save_outputs=False)

        params = report.quantification_parameters
        assert params["label_efficiency"] == 0.85  # pCASL default
        assert params["partition_coefficient"] == 0.9  # default
        assert params["t1_blood"] == 1.65  # default for 3 T

    def test_t1_blood_default_at_1p5t(self, tmp_path: Path) -> None:
        """T1 of blood defaults to 1.35 s at 1.5 T."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        _, report = run_asl_quantification(
            asl_path,
            asl_context=["m0scan", "control", "label"],
            label_type="pcasl",
            post_label_delay=1.8,
            label_duration=1.8,
            magnetic_field_tesla=1.5,
            save_outputs=False,
        )

        assert report.quantification_parameters["t1_blood"] == 1.35


class TestPaslQuantification:
    """PASL quantification maps PLD to inversion time and bolus duration to TI1."""

    def test_pasl_explicit(self, tmp_path: Path) -> None:
        """PASL CBF matches the White Paper PASL equation."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        cbf_image, report = run_asl_quantification(
            asl_path,
            asl_context=["m0scan", "control", "label"],
            label_type="pasl",
            post_label_delay=2.0,
            bolus_duration=1.0,
            label_efficiency=0.55,
            t1_blood=1.65,
            partition_coefficient=0.9,
            save_outputs=False,
        )

        assert report.label_type == "pasl"
        expected = quantify_pasl(
            _volume("control"),
            _volume("label"),
            _volume("m0scan"),
            bolus_duration=1.0,
            inversion_time=2.0,
            label_efficiency=0.55,
            t1_blood=1.65,
            partition_coefficient=0.9,
        ).perfusion
        np.testing.assert_allclose(cbf_image.data, expected)
        assert report.quantification_parameters["inversion_time"] == 2.0
        assert report.quantification_parameters["bolus_duration"] == 1.0


class TestVolumeHandling:
    """Volume identification, averaging and separate M0 handling."""

    def test_multiple_control_label_pairs_are_averaged(self, tmp_path: Path) -> None:
        """Repeated control/label volumes are averaged before quantification."""
        context = ["m0scan", "control", "label", "control", "label"]
        # Build an image with distinct control/label values to check averaging.
        m0 = np.full(_SHAPE, 1.0)
        control_a, control_b = np.full(_SHAPE, 1.0), np.full(_SHAPE, 1.2)
        label_a, label_b = np.full(_SHAPE, 0.9), np.full(_SHAPE, 0.95)
        data = np.stack([m0, control_a, label_a, control_b, label_b], axis=-1)
        asl_path = tmp_path / "sub-01_asl.nii.gz"
        save_nifti(data, asl_path, affine=np.eye(4))

        cbf_image, _ = run_asl_quantification(
            asl_path,
            asl_context=context,
            label_type="pcasl",
            post_label_delay=1.8,
            label_duration=1.8,
            save_outputs=False,
        )

        expected = quantify_pcasl(
            (control_a + control_b) / 2,
            (label_a + label_b) / 2,
            m0,
            label_duration=1.8,
            post_label_delay=1.8,
        ).perfusion
        np.testing.assert_allclose(cbf_image.data, expected)

    def test_separate_m0_file(self, tmp_path: Path) -> None:
        """A separate M0 file is used in place of m0scan volumes."""
        asl_path = _write_asl(tmp_path, ["control", "label"])
        m0_path = tmp_path / "m0.nii.gz"
        save_nifti(np.full(_SHAPE, 2.0), m0_path, affine=np.eye(4))

        cbf_image, report = run_asl_quantification(
            asl_path,
            asl_context=["control", "label"],
            m0_file=m0_path,
            label_type="pcasl",
            post_label_delay=1.8,
            label_duration=1.8,
            save_outputs=False,
        )

        assert report.m0_file == m0_path
        expected = quantify_pcasl(
            _volume("control"),
            _volume("label"),
            np.full(_SHAPE, 2.0),
            label_duration=1.8,
            post_label_delay=1.8,
        ).perfusion
        np.testing.assert_allclose(cbf_image.data, expected)


class TestOutputs:
    """Writing the CBF map and JSON report to disk."""

    def test_writes_map_and_report(self, tmp_path: Path) -> None:
        """The pipeline writes a CBF map and a valid JSON report."""
        asl_path = _write_asl(
            tmp_path,
            ["m0scan", "control", "label"],
            sidecar={
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 1.8,
                "LabelingDuration": 1.8,
                "MagneticFieldStrength": 3.0,
            },
            write_aslcontext=True,
        )

        cbf_image, report = run_asl_quantification(asl_path, output_dir=tmp_path)

        map_path = tmp_path / "sub-01_asl_cbf.nii.gz"
        report_path = tmp_path / "sub-01_asl_report.json"
        assert report.output_file == map_path
        assert map_path.exists()
        assert report_path.exists()

        saved = load_nifti_image(map_path)
        np.testing.assert_allclose(saved.data, cbf_image.data)

        payload = json.loads(report_path.read_text())
        assert payload["label_type"] == "pcasl"
        assert payload["asl_context"] == ["m0scan", "control", "label"]


class TestValidation:
    """Input validation and clear error messages."""

    def test_missing_context_raises(self, tmp_path: Path) -> None:
        """With no context anywhere, the pipeline raises."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        with pytest.raises(ValueError, match="ASL context"):
            run_asl_quantification(
                asl_path, label_type="pcasl", post_label_delay=1.8,
                label_duration=1.8, save_outputs=False,
            )

    def test_context_length_mismatch_raises(self, tmp_path: Path) -> None:
        """A context whose length disagrees with the image raises."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        with pytest.raises(ValueError, match="does not match the number of volumes"):
            run_asl_quantification(
                asl_path,
                asl_context=["m0scan", "control"],
                label_type="pcasl",
                post_label_delay=1.8,
                label_duration=1.8,
                save_outputs=False,
            )

    def test_missing_label_volume_raises(self, tmp_path: Path) -> None:
        """A context without a label volume raises."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control"])

        with pytest.raises(ValueError, match="'control' and one 'label'"):
            run_asl_quantification(
                asl_path,
                asl_context=["m0scan", "control"],
                label_type="pcasl",
                post_label_delay=1.8,
                label_duration=1.8,
                save_outputs=False,
            )

    def test_missing_m0_raises(self, tmp_path: Path) -> None:
        """No m0scan volume and no m0_file raises."""
        asl_path = _write_asl(tmp_path, ["control", "label"])

        with pytest.raises(ValueError, match="m0scan"):
            run_asl_quantification(
                asl_path,
                asl_context=["control", "label"],
                label_type="pcasl",
                post_label_delay=1.8,
                label_duration=1.8,
                save_outputs=False,
            )

    def test_unknown_label_type_raises(self, tmp_path: Path) -> None:
        """An unknown / missing labelling type raises."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        with pytest.raises(ValueError, match="labelling type"):
            run_asl_quantification(
                asl_path,
                asl_context=["m0scan", "control", "label"],
                post_label_delay=1.8,
                label_duration=1.8,
                save_outputs=False,
            )

    def test_missing_required_parameter_raises(self, tmp_path: Path) -> None:
        """Missing label duration for pCASL raises."""
        asl_path = _write_asl(tmp_path, ["m0scan", "control", "label"])

        with pytest.raises(ValueError, match="label duration"):
            run_asl_quantification(
                asl_path,
                asl_context=["m0scan", "control", "label"],
                label_type="pcasl",
                post_label_delay=1.8,
                save_outputs=False,
            )

    def test_non_4d_image_raises(self, tmp_path: Path) -> None:
        """A 3D ASL image raises."""
        asl_path = tmp_path / "sub-01_asl.nii.gz"
        save_nifti(np.ones(_SHAPE), asl_path, affine=np.eye(4))

        with pytest.raises(ValueError, match="must be 4D"):
            run_asl_quantification(
                asl_path,
                asl_context=["control"],
                label_type="pcasl",
                post_label_delay=1.8,
                label_duration=1.8,
                save_outputs=False,
            )
