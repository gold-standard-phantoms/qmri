"""Tests for the perfusion (ASL) CLI commands."""

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from qmri.cli.main import cli
from qmri.io import save_nifti

_SHAPE = (4, 4, 4)


def _write_asl(
    directory: Path,
    *,
    sidecar: dict[str, object] | None = None,
    aslcontext: list[str] | None = None,
    name: str = "sub-01_asl",
) -> Path:
    """Write a 4D ASL image [m0scan, control, label] plus optional metadata."""
    m0 = np.full(_SHAPE, 1.0)
    control = np.full(_SHAPE, 1.0)
    label = np.full(_SHAPE, 0.99)
    data = np.stack([m0, control, label], axis=-1)
    asl_path = directory / f"{name}.nii.gz"
    save_nifti(data, asl_path, affine=np.eye(4))

    if sidecar is not None:
        (directory / f"{name}.json").write_text(json.dumps(sidecar))
    if aslcontext is not None:
        tsv = "volume_type\n" + "\n".join(aslcontext)
        (directory / f"{name.removesuffix('_asl')}_aslcontext.tsv").write_text(tsv)

    return asl_path


class TestAslCommand:
    """Tests for the `perfusion asl` command."""

    def test_help(self) -> None:
        """The ASL command help describes the parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["perfusion", "asl", "--help"])

        assert result.exit_code == 0
        assert "cerebral blood flow" in result.output.lower()
        assert "--label-type" in result.output
        assert "--post-label-delay" in result.output

    def test_missing_argument(self) -> None:
        """Missing the ASL argument errors."""
        runner = CliRunner()
        result = runner.invoke(cli, ["perfusion", "asl"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output

    def test_execution_with_explicit_args(self, tmp_path: Path) -> None:
        """Running with explicit options writes a CBF map."""
        asl_path = _write_asl(tmp_path)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "perfusion", "asl", str(asl_path),
                "--asl-context", "m0scan,control,label",
                "--label-type", "pcasl",
                "--post-label-delay", "1.8",
                "--label-duration", "1.8",
                "-o", str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "sub-01_asl_cbf.nii.gz").exists()
        assert "Mean CBF" in result.output

    def test_execution_from_sidecar(self, tmp_path: Path) -> None:
        """Parameters and context are read from the BIDS sidecar / tsv."""
        asl_path = _write_asl(
            tmp_path,
            sidecar={
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 1.8,
                "LabelingDuration": 1.8,
                "MagneticFieldStrength": 3.0,
            },
            aslcontext=["m0scan", "control", "label"],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["perfusion", "asl", str(asl_path), "-o", str(tmp_path)]
        )

        assert result.exit_code == 0, result.output
        assert "pcasl" in result.output.lower()

    def test_invalid_label_type(self, tmp_path: Path) -> None:
        """An invalid label type is rejected by click."""
        asl_path = _write_asl(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["perfusion", "asl", str(asl_path), "--label-type", "bogus"]
        )

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    def test_missing_parameters_reports_error(self, tmp_path: Path) -> None:
        """Missing labelling type/parameters produce a handled error."""
        asl_path = _write_asl(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "perfusion", "asl", str(asl_path),
                "--asl-context", "m0scan,control,label",
            ],
        )

        assert result.exit_code != 0
        assert "Error" in result.output


class TestPerfusionGroup:
    """Tests for the perfusion command group."""

    def test_group_help(self) -> None:
        """The perfusion group help lists the asl command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["perfusion", "--help"])

        assert result.exit_code == 0
        assert "asl" in result.output.lower()
