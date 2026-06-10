"""Thermometry commands for qmri CLI.

This module provides commands for MR temperature mapping, wrapping the
pipelines in :mod:`qmri.pipelines.thermometry`.
"""

from typing import TYPE_CHECKING, cast

import click
from qmri.cli._utils import console, create_progress, error_handler

if TYPE_CHECKING:
    from qmri.thermometry import DfInitMethod, RegionAnalysisMethod

__all__ = ["thermometry"]


@click.group(name="thermometry")
def thermometry() -> None:
    r"""Thermometry analysis commands.

    Commands for MR temperature mapping.

    \b
    Examples:
        qmri thermometry multiecho echoes.nii.gz -e echo_times.txt \
            -s labels.nii.gz -o results/
    """
    pass


@thermometry.command(name="multiecho")
@click.argument(
    "multiecho_niftis", nargs=-1, required=True, type=click.Path(exists=True)
)
@click.option(
    "-s",
    "--segmentation",
    required=True,
    type=click.Path(exists=True),
    help="Segmentation/label-map NIfTI (3D).",
)
@click.option(
    "-e",
    "--echo-times",
    "echo_times_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Echo-time text file (seconds), one per multi-echo image, in order.",
)
@click.option(
    "--method",
    type=click.Choice(
        ["regionwise", "voxelwise", "regionwise_bootstrap"], case_sensitive=False
    ),
    default="regionwise",
    help="Analysis method (default: regionwise).",
)
@click.option(
    "--n-bootstrap",
    type=int,
    default=100,
    help="Number of bootstrap iterations (regionwise_bootstrap only).",
)
@click.option(
    "--df-init",
    type=click.Choice(["multistart", "fixed", "lombscargle"], case_sensitive=False),
    default="multistart",
    help=(
        "Frequency starting-value strategy for the fit (default: multistart). "
        "'fixed' uses a single fixed guess; 'lombscargle' seeds from the data; "
        "'multistart' tries both and keeps the best fit."
    ),
)
@click.option(
    "-b",
    "--field-strength",
    "field_strength",
    type=float,
    default=None,
    help="Magnetic field strength in Tesla. Overrides JSON sidecar detection.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: directory of the first input image).",
)
@click.option(
    "--output-prefix",
    default=None,
    help="Output filename prefix (default: stem of the first input image).",
)
@click.pass_context
@error_handler
def multiecho(
    ctx: click.Context,
    multiecho_niftis: tuple[str, ...],
    segmentation: str,
    echo_times_files: tuple[str, ...],
    method: str,
    n_bootstrap: int,
    df_init: str,
    field_strength: float | None,
    output_dir: str | None,
    output_prefix: str | None,
) -> None:
    r"""Estimate temperature from multi-echo magnitude images and a segmentation.

    MULTIECHO_NIFTIS are one or more 4D multi-echo magnitude NIfTI files (the
    echo dimension must be the last axis). Provide one ``--echo-times`` file per
    image, in the same order. Echoes from all images are concatenated and sorted
    by echo time before fitting.

    The temperature calibration is specific to ethylene glycol. The magnetic
    field strength is read from a JSON sidecar (``ImagingFrequency`` or
    ``MagneticFieldStrength``) unless ``--field-strength`` is given.

    \b
    Examples:
        # Single multi-echo image, region-wise fit
        qmri thermometry multiecho echoes.nii.gz -e echo_times.txt \
            -s labels.nii.gz -o results/

        # Two echo blocks, bootstrap uncertainty, explicit field strength
        qmri thermometry multiecho block1.nii.gz block2.nii.gz \
            -e te1.txt -e te2.txt -s labels.nii.gz \
            --method regionwise_bootstrap --n-bootstrap 200 \
            --field-strength 3.0 -o results/
    """
    from qmri.pipelines.thermometry import run_multiecho_thermometry

    verbose: bool = ctx.obj.get("verbose", False) if ctx.obj else False

    if len(multiecho_niftis) != len(echo_times_files):
        msg = (
            f"Number of images ({len(multiecho_niftis)}) must match the number "
            f"of --echo-times files ({len(echo_times_files)})."
        )
        raise click.UsageError(msg)

    analysis_method = cast("RegionAnalysisMethod", method.lower())
    df_init_method = cast("DfInitMethod", df_init.lower())
    console.print(f"[bold]Multi-echo thermometry ({analysis_method})[/]")
    if verbose and field_strength is not None:
        console.print(f"[dim]Field strength: {field_strength} T[/]")

    with create_progress("Fitting thermometry") as progress:
        task = progress.add_task("Processing...", total=None)
        _, report = run_multiecho_thermometry(
            multiecho_files=list(multiecho_niftis),
            segmentation_file=segmentation,
            echo_times_files=list(echo_times_files),
            method=analysis_method,
            n_bootstrap=n_bootstrap,
            df_init=df_init_method,
            magnetic_field_tesla=field_strength,
            output_dir=output_dir,
            output_prefix=output_prefix,
            save_outputs=True,
        )
        progress.update(task, completed=True)

    console.print(f"[green]Field strength:[/] {report.magnetic_field_tesla:.3f} T")
    if report.output_file is not None:
        console.print(f"[green]Saved temperature map to:[/] {report.output_file}")

    console.print()
    console.print("[bold]Per-region temperature:[/]")
    for region in report.regions:
        console.print(
            f"  region {region.region_id}: "
            f"{region.temperature:.2f} ± {region.temperature_uncertainty:.2f} °C "
            f"({region.region_size} voxels)"
        )
