"""Magnetisation transfer commands for qmri CLI.

This module provides commands for magnetisation transfer ratio (MTR) mapping,
wrapping the pipelines in :mod:`qmri.pipelines.transfer`.
"""

import click
from qmri.cli._utils import console, create_progress, error_handler

__all__ = ["transfer"]


@click.group(name="transfer")
def transfer() -> None:
    r"""Magnetisation transfer analysis commands.

    Commands for magnetisation transfer ratio (MTR) mapping.

    \b
    Examples:
        # Separate saturated and unsaturated images
        qmri transfer mtr sat.nii.gz --unsaturated nosat.nii.gz -o results/

        # A single 4D file ordered [unsaturated, saturated]
        qmri transfer mtr mt.nii.gz -o results/
    """
    pass


@transfer.command(name="mtr")
@click.argument("saturated", type=click.Path(exists=True))
@click.option(
    "-u",
    "--unsaturated",
    type=click.Path(exists=True),
    default=None,
    help=(
        "Image without bound-pool saturation. Omit to treat SATURATED as a 4D "
        "file whose last axis holds [unsaturated, saturated]."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: directory of the SATURATED image).",
)
@click.option(
    "--output-prefix",
    default=None,
    help="Output filename prefix (default: stem of the SATURATED image).",
)
@click.pass_context
@error_handler
def mtr(
    ctx: click.Context,
    saturated: str,
    unsaturated: str | None,
    output_dir: str | None,
    output_prefix: str | None,
) -> None:
    r"""Calculate a Magnetisation Transfer Ratio (MTR) map.

    SATURATED is the image acquired with bound-pool saturation. Provide
    ``--unsaturated`` for the reference image, or omit it to treat SATURATED as
    a single 4D file whose last axis holds two volumes ordered
    ``[unsaturated, saturated]``.

    The MTR map is written in percentage units (pu).

    \b
    Examples:
        # Separate images
        qmri transfer mtr sat.nii.gz -u nosat.nii.gz -o results/

        # Combined 4D file ([unsaturated, saturated])
        qmri transfer mtr mt.nii.gz -o results/
    """
    from qmri.pipelines.transfer import run_mtr

    verbose: bool = ctx.obj.get("verbose", False) if ctx.obj else False

    console.print("[bold]Magnetisation transfer ratio[/]")

    with create_progress("Calculating MTR") as progress:
        task = progress.add_task("Processing...", total=None)
        _, report = run_mtr(
            saturated_file=saturated,
            unsaturated_file=unsaturated,
            output_dir=output_dir,
            output_prefix=output_prefix,
            save_outputs=True,
        )
        progress.update(task, completed=True)

    if verbose:
        console.print(f"[dim]Input mode: {report.mode}[/]")
    if report.output_file is not None:
        console.print(f"[green]Saved MTR map to:[/] {report.output_file}")

    console.print()
    console.print("[bold]Summary statistics (valid voxels):[/]")
    console.print(f"  Voxels:   {report.n_valid_voxels}")
    console.print(f"  Mean MTR: {report.mtr_mean:.2f} pu")
    console.print(f"  Std MTR:  {report.mtr_std:.2f} pu")
    console.print(f"  Range:    {report.mtr_min:.2f} – {report.mtr_max:.2f} pu")
