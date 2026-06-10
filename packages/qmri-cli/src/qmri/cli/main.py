"""Main CLI entry point for qmri.

This module provides the main command-line interface for qmri,
using click for command structure and rich for pretty console output.
"""

from pathlib import Path

import click
from qmri.cli._utils import console, error_console
from qmri.cli.commands.diffusion import diffusion
from qmri.cli.commands.relaxometry import relaxometry
from qmri.cli.commands.thermometry import thermometry

__all__ = ["cli"]


class QMRIContext:
    """Context object for storing CLI state."""

    def __init__(self) -> None:
        """Initialise the context."""
        self.verbose: bool = False
        self.quiet: bool = False
        self.output_dir: Path | None = None


pass_context = click.make_pass_decorator(QMRIContext, ensure=True)


@click.group()
@click.version_option(package_name="qmri-cli")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output with additional information.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress non-essential output.",
)
@click.option(
    "--output-dir",
    "-d",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Default output directory for generated files.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: bool,
    quiet: bool,
    output_dir: Path | None,
) -> None:
    r"""Provide qmri quantitative MRI analysis tools.

    A command-line interface for quantitative MRI analysis including
    diffusion imaging, relaxometry, and more.

    \b
    Examples:
        # Show help for diffusion commands
        qmri diffusion --help

        # Fit ADC from DWI data
        qmri diffusion adc dwi.nii.gz bvalues.txt -o adc_map.nii.gz

        # Fit T1 from inversion recovery data
        qmri relaxometry t1 ir_data.nii.gz --ti "100,500,1000,2000" -o t1.nii.gz

        # Fit T2 from multi-echo data
        qmri relaxometry t2 mese.nii.gz --te "10,20,40,80" -o t2.nii.gz

        # Estimate temperature from multi-echo magnitude data
        qmri thermometry multiecho echoes.nii.gz -e echo_times.txt \
            -s labels.nii.gz -o results/

    For more information on a specific command, use:
        qmri <command> --help
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store options in context for subcommands
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["output_dir"] = output_dir

    # Validate conflicting options
    if verbose and quiet:
        error_console.print(
            "[yellow]Warning:[/] Both --verbose and --quiet specified. Using --quiet."
        )
        ctx.obj["verbose"] = False

    # Create output directory if specified and doesn't exist
    if output_dir is not None and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            console.print(f"[dim]Created output directory: {output_dir}[/]")


# Register command groups
cli.add_command(diffusion)
cli.add_command(relaxometry)
cli.add_command(thermometry)


@cli.command(name="info")
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display information about qmri and available commands.

    Shows version information, installed packages, and a summary
    of available commands.
    """
    from importlib.metadata import version

    console.print()
    console.print("[bold blue]qmri - Quantitative MRI Analysis Tools[/]")
    console.print()

    # Version information
    try:
        cli_version = version("qmri-cli")
        console.print(f"  qmri-cli version: {cli_version}")
    except Exception:
        console.print("  qmri-cli version: unknown")

    try:
        qmri_version = version("qmri")
        console.print(f"  qmri version: {qmri_version}")
    except Exception:
        console.print("  qmri version: unknown")

    try:
        io_version = version("qmri-io")
        console.print(f"  qmri-io version: {io_version}")
    except Exception:
        console.print("  qmri-io version: unknown")

    try:
        pipelines_version = version("qmri-pipelines")
        console.print(f"  qmri-pipelines version: {pipelines_version}")
    except Exception:
        console.print("  qmri-pipelines version: unknown")

    console.print()
    console.print("[bold]Available command groups:[/]")
    console.print("  diffusion    - Diffusion imaging analysis (ADC, DTI)")
    console.print("  relaxometry  - Relaxation time mapping (T1, T2)")
    console.print("  thermometry  - MR temperature mapping (multi-echo)")
    console.print()
    console.print("Use 'qmri <command> --help' for more information on a command.")


if __name__ == "__main__":
    cli()
