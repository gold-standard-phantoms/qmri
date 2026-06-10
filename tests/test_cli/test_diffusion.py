"""Tests for the diffusion CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner
from qmri.cli.main import cli

_rng = np.random.default_rng(42)


class TestADCCommand:
    """Tests for the ADC fitting command."""

    def test_adc_help(self) -> None:
        """Test that the ADC command help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diffusion", "adc", "--help"])

        assert result.exit_code == 0
        assert "Fit Apparent Diffusion Coefficient" in result.output
        assert "--method" in result.output
        assert "--mask" in result.output
        assert "--b0-threshold" in result.output

    def test_adc_missing_arguments(self) -> None:
        """Test that missing arguments produce an error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diffusion", "adc"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output

    def test_adc_missing_output(self, tmp_path: Path) -> None:
        """Test that missing --output produces an error."""
        runner = CliRunner()

        # Create dummy files
        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0 500 1000 2000")

        result = runner.invoke(cli, ["diffusion", "adc", str(dwi_file), str(bval_file)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    @patch("qmri.io.get_affine")
    @patch("qmri.io.load_nifti")
    @patch("qmri.io.save_nifti")
    @patch("qmri.diffusion.adc.fit")
    def test_adc_basic_execution(
        self,
        mock_fit: MagicMock,
        mock_save: MagicMock,
        mock_load: MagicMock,
        mock_get_affine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test basic ADC fitting execution with mocked I/O."""
        # Create test data
        shape = (4, 4, 4, 4)
        data = _rng.random(shape).astype(np.float64) * 1000
        affine = np.eye(4)
        header = MagicMock()

        mock_load.return_value = (data, header)
        mock_get_affine.return_value = affine

        # Create mock result
        mock_result = MagicMock()
        mock_result.adc = _rng.random((4, 4, 4)).astype(np.float64) * 1e-3
        mock_result.s0 = _rng.random((4, 4, 4)).astype(np.float64) * 1000
        mock_result.r_squared = _rng.random((4, 4, 4)).astype(np.float64)
        mock_fit.return_value = mock_result

        # Create input files
        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0 500 1000 2000")
        output_file = tmp_path / "adc.nii.gz"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diffusion",
                "adc",
                str(dwi_file),
                str(bval_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        mock_load.assert_called_once()
        mock_fit.assert_called_once()
        mock_save.assert_called_once()

    @patch("qmri.io.get_affine")
    @patch("qmri.io.load_nifti")
    @patch("qmri.io.save_nifti")
    @patch("qmri.diffusion.adc.fit")
    def test_adc_with_method_option(
        self,
        mock_fit: MagicMock,
        mock_save: MagicMock,
        mock_load: MagicMock,
        mock_get_affine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test ADC fitting with different methods."""
        # Setup mocks
        shape = (4, 4, 4, 4)
        data = _rng.random(shape).astype(np.float64) * 1000
        mock_load.return_value = (data, MagicMock())
        mock_get_affine.return_value = np.eye(4)

        mock_result = MagicMock()
        mock_result.adc = _rng.random((4, 4, 4)).astype(np.float64) * 1e-3
        mock_result.s0 = _rng.random((4, 4, 4)).astype(np.float64) * 1000
        mock_result.r_squared = _rng.random((4, 4, 4)).astype(np.float64)
        mock_fit.return_value = mock_result

        # Create input files
        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0 500 1000 2000")
        output_file = tmp_path / "adc.nii.gz"

        runner = CliRunner()

        for method in ["lls", "wlls", "iwlls"]:
            mock_fit.reset_mock()
            result = runner.invoke(
                cli,
                [
                    "diffusion",
                    "adc",
                    str(dwi_file),
                    str(bval_file),
                    "--method",
                    method,
                    "-o",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0, f"Failed for method {method}: {result.output}"
            # Verify the method was passed correctly
            call_kwargs = mock_fit.call_args.kwargs
            assert call_kwargs["method"] == method.lower()

    @patch("qmri.io.get_affine")
    @patch("qmri.io.load_nifti")
    @patch("qmri.io.save_nifti")
    @patch("qmri.diffusion.adc.fit")
    def test_adc_save_additional_maps(
        self,
        mock_fit: MagicMock,
        mock_save: MagicMock,
        mock_load: MagicMock,
        mock_get_affine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test ADC fitting with --save-s0 and --save-r2 options."""
        # Setup mocks
        shape = (4, 4, 4, 4)
        data = _rng.random(shape).astype(np.float64) * 1000
        mock_load.return_value = (data, MagicMock())
        mock_get_affine.return_value = np.eye(4)

        mock_result = MagicMock()
        mock_result.adc = _rng.random((4, 4, 4)).astype(np.float64) * 1e-3
        mock_result.s0 = _rng.random((4, 4, 4)).astype(np.float64) * 1000
        mock_result.r_squared = _rng.random((4, 4, 4)).astype(np.float64)
        mock_fit.return_value = mock_result

        # Create input files
        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0 500 1000 2000")
        output_file = tmp_path / "adc.nii.gz"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diffusion",
                "adc",
                str(dwi_file),
                str(bval_file),
                "--save-s0",
                "--save-r2",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Should have saved 3 files: ADC, S0, and R2
        assert mock_save.call_count == 3

    def test_adc_invalid_method(self, tmp_path: Path) -> None:
        """Test that an invalid method produces an error."""
        runner = CliRunner()

        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0 500 1000 2000")

        result = runner.invoke(
            cli,
            [
                "diffusion",
                "adc",
                str(dwi_file),
                str(bval_file),
                "--method",
                "invalid",
                "-o",
                str(tmp_path / "adc.nii.gz"),
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()


class TestDiffusionGroup:
    """Tests for the diffusion command group."""

    def test_diffusion_help(self) -> None:
        """Test that the diffusion group help is displayed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diffusion", "--help"])

        assert result.exit_code == 0
        assert "Diffusion imaging analysis commands" in result.output
        assert "adc" in result.output


class TestBValuesFile:
    """Tests for b-values file parsing."""

    @patch("qmri.io.get_affine")
    @patch("qmri.io.load_nifti")
    @patch("qmri.io.save_nifti")
    @patch("qmri.diffusion.adc.fit")
    def test_comma_separated_bvalues(
        self,
        mock_fit: MagicMock,
        mock_save: MagicMock,
        mock_load: MagicMock,
        mock_get_affine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that comma-separated b-values are parsed correctly."""
        # Setup mocks
        shape = (4, 4, 4, 4)
        data = _rng.random(shape).astype(np.float64) * 1000
        mock_load.return_value = (data, MagicMock())
        mock_get_affine.return_value = np.eye(4)

        mock_result = MagicMock()
        mock_result.adc = _rng.random((4, 4, 4)).astype(np.float64) * 1e-3
        mock_result.s0 = _rng.random((4, 4, 4)).astype(np.float64) * 1000
        mock_result.r_squared = _rng.random((4, 4, 4)).astype(np.float64)
        mock_fit.return_value = mock_result

        # Create input files with comma-separated values
        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0,500,1000,2000")
        output_file = tmp_path / "adc.nii.gz"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diffusion",
                "adc",
                str(dwi_file),
                str(bval_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"

    @patch("qmri.io.get_affine")
    @patch("qmri.io.load_nifti")
    @patch("qmri.io.save_nifti")
    @patch("qmri.diffusion.adc.fit")
    def test_newline_separated_bvalues(
        self,
        mock_fit: MagicMock,
        mock_save: MagicMock,
        mock_load: MagicMock,
        mock_get_affine: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that newline-separated b-values are parsed correctly."""
        # Setup mocks
        shape = (4, 4, 4, 4)
        data = _rng.random(shape).astype(np.float64) * 1000
        mock_load.return_value = (data, MagicMock())
        mock_get_affine.return_value = np.eye(4)

        mock_result = MagicMock()
        mock_result.adc = _rng.random((4, 4, 4)).astype(np.float64) * 1e-3
        mock_result.s0 = _rng.random((4, 4, 4)).astype(np.float64) * 1000
        mock_result.r_squared = _rng.random((4, 4, 4)).astype(np.float64)
        mock_fit.return_value = mock_result

        # Create input files with newline-separated values
        dwi_file = tmp_path / "dwi.nii.gz"
        dwi_file.touch()
        bval_file = tmp_path / "bvalues.txt"
        bval_file.write_text("0\n500\n1000\n2000")
        output_file = tmp_path / "adc.nii.gz"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diffusion",
                "adc",
                str(dwi_file),
                str(bval_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
