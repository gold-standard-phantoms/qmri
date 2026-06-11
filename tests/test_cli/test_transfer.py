"""Tests for the transfer (MTR) CLI commands."""

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from qmri.cli.main import cli
from qmri.io import save_nifti

_MTR_TRUTH = np.linspace(0.0, 50.0, num=32, dtype=np.float64).reshape(4, 4, 2)


def _write_pair(directory: Path) -> tuple[Path, Path]:
    """Write separate saturated/unsaturated images; return (sat, nosat)."""
    unsaturated = np.ones_like(_MTR_TRUTH)
    saturated = unsaturated * (1.0 - _MTR_TRUTH / 100.0)
    sat_path = directory / "sat.nii.gz"
    nosat_path = directory / "nosat.nii.gz"
    save_nifti(saturated, sat_path, affine=np.eye(4))
    save_nifti(unsaturated, nosat_path, affine=np.eye(4))
    return sat_path, nosat_path


class TestMtrCommand:
    """Tests for the `transfer mtr` command."""

    def test_help(self) -> None:
        """The MTR command help describes the inputs and modes."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transfer", "mtr", "--help"])

        assert result.exit_code == 0
        assert "Magnetisation Transfer Ratio" in result.output
        assert "--unsaturated" in result.output

    def test_missing_argument(self) -> None:
        """Missing the SATURATED argument errors."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transfer", "mtr"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output

    def test_separate_files_execution(self, tmp_path: Path) -> None:
        """Running on separate images writes an MTR map."""
        sat_path, nosat_path = _write_pair(tmp_path)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transfer", "mtr", str(sat_path),
                "-u", str(nosat_path),
                "-o", str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "sat_mtr_map.nii.gz").exists()
        assert "Mean MTR" in result.output

    def test_combined_file_execution(self, tmp_path: Path) -> None:
        """Running on a combined 4D file writes an MTR map."""
        unsaturated = np.ones_like(_MTR_TRUTH)
        saturated = unsaturated * (1.0 - _MTR_TRUTH / 100.0)
        combined = np.stack([unsaturated, saturated], axis=-1)
        combined_path = tmp_path / "mt.nii.gz"
        save_nifti(combined, combined_path, affine=np.eye(4))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["transfer", "mtr", str(combined_path), "-o", str(tmp_path)]
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "mt_mtr_map.nii.gz").exists()

    def test_shape_mismatch_reports_error(self, tmp_path: Path) -> None:
        """A shape mismatch produces a handled error, not a traceback."""
        sat_path, _ = _write_pair(tmp_path)
        wrong = tmp_path / "wrong.nii.gz"
        save_nifti(np.ones((3, 3, 3)), wrong, affine=np.eye(4))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["transfer", "mtr", str(sat_path), "-u", str(wrong)]
        )

        assert result.exit_code != 0
        assert "Error" in result.output


class TestTransferGroup:
    """Tests for the transfer command group."""

    def test_group_help(self) -> None:
        """The transfer group help lists the mtr command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transfer", "--help"])

        assert result.exit_code == 0
        assert "mtr" in result.output.lower()
