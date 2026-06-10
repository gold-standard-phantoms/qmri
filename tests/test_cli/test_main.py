"""Tests for the main CLI entry point."""

from click.testing import CliRunner
from qmri.cli.main import cli


class TestCLIMain:
    """Tests for the main CLI group."""

    def test_cli_help(self) -> None:
        """Test that the CLI help message is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "qmri" in result.output.lower()
        assert "diffusion" in result.output
        assert "relaxometry" in result.output

    def test_cli_version(self) -> None:
        """Test that the version option works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_verbose_and_quiet_warning(self) -> None:
        """Test that using both --verbose and --quiet shows a warning."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--quiet", "info"])

        # Should succeed but with a warning
        assert result.exit_code == 0
        assert "Warning" in result.output or "warning" in result.output.lower()

    def test_info_command(self) -> None:
        """Test the info command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "qmri" in result.output.lower()
        assert "Available command groups" in result.output

    def test_diffusion_group_help(self) -> None:
        """Test that the diffusion command group help is displayed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diffusion", "--help"])

        assert result.exit_code == 0
        assert "adc" in result.output.lower()

    def test_relaxometry_group_help(self) -> None:
        """Test that the relaxometry command group help is displayed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["relaxometry", "--help"])

        assert result.exit_code == 0
        assert "t1" in result.output.lower()
        assert "t2" in result.output.lower()


class TestCLIOptions:
    """Tests for global CLI options."""

    def test_output_dir_option(self, tmp_path: "Path") -> None:  # noqa: F821
        """Test that --output-dir creates the directory if needed."""
        runner = CliRunner()
        new_dir = tmp_path / "new_output_dir"

        result = runner.invoke(cli, ["--output-dir", str(new_dir), "--verbose", "info"])

        assert result.exit_code == 0
        assert new_dir.exists()

    def test_verbose_flag(self) -> None:
        """Test that --verbose flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "info"])

        assert result.exit_code == 0

    def test_quiet_flag(self) -> None:
        """Test that --quiet flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--quiet", "info"])

        assert result.exit_code == 0
