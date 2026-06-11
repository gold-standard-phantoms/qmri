"""Perfusion commands for qmri CLI.

This module provides commands for arterial spin labelling (ASL) perfusion
quantification, wrapping the pipelines in :mod:`qmri.pipelines.perfusion`.
"""

from typing import TYPE_CHECKING, cast

import click
from qmri.cli._utils import console, create_progress, error_handler

if TYPE_CHECKING:
    from qmri.pipelines.perfusion import LabelType

__all__ = ["perfusion"]


@click.group(name="perfusion")
def perfusion() -> None:
    r"""Perfusion analysis commands.

    Commands for arterial spin labelling (ASL) cerebral blood flow mapping.

    \b
    Examples:
        qmri perfusion asl asl.nii.gz --label-type pcasl \
            --post-label-delay 1.8 --label-duration 1.8 -o results/
    """
    pass


@perfusion.command(name="asl")
@click.argument("asl_nifti", type=click.Path(exists=True))
@click.option(
    "--asl-context",
    default=None,
    help=(
        "Comma-separated per-volume labels (e.g. 'm0scan,control,label'). If "
        "omitted, read from a sibling *_aslcontext.tsv or the JSON sidecar."
    ),
)
@click.option(
    "--m0",
    "m0_file",
    type=click.Path(exists=True),
    default=None,
    help="Separate M0 NIfTI. If omitted, M0 is taken from m0scan volumes.",
)
@click.option(
    "--label-type",
    type=click.Choice(["pcasl", "casl", "pasl"], case_sensitive=False),
    default=None,
    help="ASL labelling type. If omitted, read from the JSON sidecar.",
)
@click.option(
    "--post-label-delay",
    type=float,
    default=None,
    help="Post-label delay (s) for pCASL/CASL, or inversion time (s) for PASL.",
)
@click.option(
    "--label-duration",
    type=float,
    default=None,
    help="Label duration (s); pCASL/CASL only.",
)
@click.option(
    "--bolus-duration",
    type=float,
    default=None,
    help="Bolus duration TI1 (s); PASL only.",
)
@click.option(
    "--label-efficiency",
    type=float,
    default=None,
    help="Labelling efficiency (default: 0.85 pCASL/CASL, 0.98 PASL).",
)
@click.option(
    "--t1-blood",
    type=float,
    default=None,
    help="T1 of arterial blood (s). Default depends on field strength.",
)
@click.option(
    "--partition-coefficient",
    type=float,
    default=None,
    help="Blood-brain partition coefficient (ml/g, default: 0.9).",
)
@click.option(
    "-b",
    "--field-strength",
    "field_strength",
    type=float,
    default=None,
    help="Magnetic field strength (T), used to pick a default T1 of blood.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: directory of the ASL image).",
)
@click.option(
    "--output-prefix",
    default=None,
    help="Output filename prefix (default: stem of the ASL image).",
)
@click.pass_context
@error_handler
def asl(
    ctx: click.Context,
    asl_nifti: str,
    asl_context: str | None,
    m0_file: str | None,
    label_type: str | None,
    post_label_delay: float | None,
    label_duration: float | None,
    bolus_duration: float | None,
    label_efficiency: float | None,
    t1_blood: float | None,
    partition_coefficient: float | None,
    field_strength: float | None,
    output_dir: str | None,
    output_prefix: str | None,
) -> None:
    r"""Quantify cerebral blood flow (CBF) from an ASL image.

    ASL_NIFTI is a 4D ASL NIfTI containing control/label (and optionally M0)
    volumes. The control, label and M0 volumes are identified from the ASL
    context and averaged before quantification.

    Labelling parameters are resolved with the precedence
    *command-line option > JSON sidecar > default*. The CBF map is written in
    ml/100g/min.

    \b
    Examples:
        # pCASL with explicit parameters
        qmri perfusion asl asl.nii.gz --label-type pcasl \
            --post-label-delay 1.8 --label-duration 1.8 \
            --asl-context m0scan,control,label -o results/

        # Parameters and context read from the BIDS sidecar / aslcontext.tsv
        qmri perfusion asl sub-01_asl.nii.gz -o results/
    """
    from qmri.pipelines.perfusion import run_asl_quantification

    verbose: bool = ctx.obj.get("verbose", False) if ctx.obj else False

    context = (
        [label.strip() for label in asl_context.split(",") if label.strip()]
        if asl_context is not None
        else None
    )
    label = (
        cast("LabelType", label_type.lower()) if label_type is not None else None
    )

    console.print("[bold]ASL quantification[/]")

    with create_progress("Quantifying CBF") as progress:
        task = progress.add_task("Processing...", total=None)
        _, report = run_asl_quantification(
            asl_file=asl_nifti,
            asl_context=context,
            m0_file=m0_file,
            label_type=label,
            post_label_delay=post_label_delay,
            label_duration=label_duration,
            bolus_duration=bolus_duration,
            label_efficiency=label_efficiency,
            t1_blood=t1_blood,
            partition_coefficient=partition_coefficient,
            magnetic_field_tesla=field_strength,
            output_dir=output_dir,
            output_prefix=output_prefix,
            save_outputs=True,
        )
        progress.update(task, completed=True)

    console.print(f"[green]Labelling type:[/] {report.label_type}")
    if verbose:
        params = ", ".join(
            f"{key}={value}"
            for key, value in report.quantification_parameters.items()
        )
        console.print(f"[dim]Quantification parameters: {params}[/]")
    if report.output_file is not None:
        console.print(f"[green]Saved CBF map to:[/] {report.output_file}")

    console.print()
    console.print("[bold]Summary statistics (valid voxels):[/]")
    console.print(f"  Voxels:   {report.n_valid_voxels}")
    console.print(f"  Mean CBF: {report.perfusion_mean:.2f} ml/100g/min")
    console.print(f"  Std CBF:  {report.perfusion_std:.2f} ml/100g/min")
