"""Relaxometry commands for qmri CLI.

This module provides commands for fitting relaxation time constants (T1, T2)
from MRI data.
"""

from typing import Literal

import click
import numpy as np
from qmri.cli._utils import (
    console,
    create_progress,
    error_handler,
    parse_values_string,
    validate_input_file,
    validate_output_path,
)

__all__ = ["relaxometry"]


@click.group(name="relaxometry")
def relaxometry() -> None:
    """Relaxometry analysis commands.

    Commands for fitting T1 and T2 relaxation times from MRI data.

    Examples:
    --------
        qmri relaxometry t1 ir_data.nii.gz --ti "100,500,1000,2000" -o t1_map.nii.gz

        qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80" -o t2_map.nii.gz
    """
    pass


@relaxometry.command(name="t1")
@click.argument("input_nifti", type=click.Path(exists=True))
@click.option(
    "--ti",
    "ti_values",
    type=str,
    required=True,
    help="Inversion times in ms (comma/space separated).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for the T1 map (NIfTI format).",
)
@click.option(
    "--method",
    type=click.Choice(["ir", "ir_classical", "vtr"], case_sensitive=False),
    default="ir",
    help="Fitting method: ir, ir_classical, or vtr (default: ir).",
)
@click.option(
    "--tr",
    type=float,
    default=None,
    help="Repetition time in milliseconds (required for IR methods).",
)
@click.option(
    "--mask",
    type=click.Path(exists=True),
    default=None,
    help="Optional mask file (NIfTI). Only voxels where mask > 0 will be processed.",
)
@click.option(
    "--max-t1",
    type=float,
    default=20000.0,
    help="Maximum valid T1 value in milliseconds (default: 20000 ms).",
)
@click.option(
    "--save-s0/--no-save-s0",
    default=False,
    help="Also save the S0 (signal amplitude) map.",
)
@click.pass_context
@error_handler
def t1(
    ctx: click.Context,
    input_nifti: str,
    ti_values: str,
    output: str,
    method: str,
    tr: float | None,
    mask: str | None,
    max_t1: float,
    save_s0: bool,
) -> None:
    r"""Fit T1 relaxation time from inversion recovery or VTR data.

    INPUT_NIFTI is the path to the 4D NIfTI file with multiple TI or TR volumes.

    The --ti option specifies inversion times (for IR methods) or repetition
    times (for VTR method) in milliseconds.

    The output T1 map will be in units of milliseconds.

    \b
    Examples:
        # T1 mapping with inversion recovery (general model)
        qmri relaxometry t1 ir_data.nii.gz --ti "100,500,1000,2000" \
            --tr 5000 -o t1.nii.gz

        # T1 mapping with classical IR model (assumes TR >> T1)
        qmri relaxometry t1 ir_data.nii.gz --ti "100,500,1000,2000" \
            --method ir_classical -o t1.nii.gz

        # T1 mapping with variable TR method
        qmri relaxometry t1 vtr_data.nii.gz --ti "500,1000,2000,4000" \
            --method vtr -o t1.nii.gz
    """
    from qmri.io import get_affine, load_nifti, save_nifti
    from qmri.relaxometry import t1 as t1_module

    verbose: bool = ctx.obj.get("verbose", False) if ctx.obj else False

    # Parse TI values (input is in ms, convert to seconds for fitting)
    ti_list = parse_values_string(ti_values)
    ti_array = np.array(ti_list, dtype=np.float64) / 1000.0  # ms -> s

    # Validate TR requirement for IR methods
    fitting_method: Literal["ir", "ir_classical", "vtr"] = method.lower()  # type: ignore[assignment]
    if fitting_method in ("ir", "ir_classical") and tr is None:
        msg = f"--tr is required for method '{method}'"
        raise click.UsageError(msg)

    # Convert TR to seconds
    tr_seconds: float | None = tr / 1000.0 if tr is not None else None

    # Convert max_t1 to seconds
    max_t1_seconds = max_t1 / 1000.0

    if verbose:
        console.print(f"[dim]Inversion times (s): {ti_array}[/]")
        if tr_seconds:
            console.print(f"[dim]TR: {tr_seconds} s[/]")

    # Validate input files
    input_path = validate_input_file(input_nifti, extensions=(".nii", ".nii.gz"))
    output_path = validate_output_path(output)

    # Load input data
    with create_progress("Loading data") as progress:
        task = progress.add_task("Loading NIfTI...", total=None)
        data, header = load_nifti(input_path)
        affine = get_affine(header)
        progress.update(task, completed=True)

    if data.ndim != 4:
        msg = f"Input must be a 4D NIfTI file, got {data.ndim}D"
        raise ValueError(msg)

    if data.shape[-1] != len(ti_array):
        msg = (
            f"Number of volumes ({data.shape[-1]}) does not match "
            f"number of time points ({len(ti_array)})"
        )
        raise ValueError(msg)

    # Load mask if provided
    mask_data: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None
    if mask:
        mask_path = validate_input_file(mask, extensions=(".nii", ".nii.gz"))
        mask_data_raw, _ = load_nifti(mask_path)
        mask_data = mask_data_raw > 0

    # Fit T1
    console.print(f"[bold]Fitting T1 using {fitting_method.upper()} method...[/]")

    with create_progress("Fitting T1") as progress:
        task = progress.add_task("Processing...", total=None)

        result = t1_module.fit(
            signal=data.astype(np.float64),
            time_points=ti_array,
            method=fitting_method,
            repetition_times=tr_seconds,
            mask=mask_data,
            max_t1=max_t1_seconds,
        )

        progress.update(task, completed=True)

    # Convert T1 back to milliseconds for output
    t1_ms = result.t1 * 1000.0

    # Save outputs
    save_nifti(t1_ms, output_path, header=header, affine=affine)
    console.print(f"[green]Saved T1 map to:[/] {output_path}")

    if save_s0:
        stem = output_path.stem.replace(".nii", "")
        s0_path = output_path.with_name(stem + "_s0.nii.gz")
        s0_data = result.s0 if hasattr(result, "s0") else result.m
        save_nifti(s0_data, s0_path, header=header, affine=affine)
        console.print(f"[green]Saved S0 map to:[/] {s0_path}")

    # Print summary statistics
    valid_mask = result.t1 > 0
    if np.any(valid_mask):
        mean_t1 = np.mean(t1_ms[valid_mask])
        std_t1 = np.std(t1_ms[valid_mask])

        console.print()
        console.print("[bold]Summary statistics:[/]")
        console.print(f"  Mean T1: {mean_t1:.1f} ms")
        console.print(f"  Std T1:  {std_t1:.1f} ms")


@relaxometry.command(name="t2")
@click.argument("input_nifti", type=click.Path(exists=True))
@click.option(
    "--te",
    "te_values",
    type=str,
    required=True,
    help="Echo times in milliseconds (comma or space separated, e.g., '10,20,40,80').",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for the T2 map (NIfTI format).",
)
@click.option(
    "--method",
    type=click.Choice(["full", "reduced"], case_sensitive=False),
    default="full",
    help="Model type: full (with offset) or reduced (no offset). Default: full.",
)
@click.option(
    "--mask",
    type=click.Path(exists=True),
    default=None,
    help="Optional mask file (NIfTI). Only voxels where mask > 0 will be processed.",
)
@click.option(
    "--skip-echoes",
    type=int,
    default=1,
    help="Number of initial echoes to skip (default: 1).",
)
@click.option(
    "--max-t2",
    type=float,
    default=5000.0,
    help="Maximum valid T2 value in milliseconds (default: 5000 ms).",
)
@click.option(
    "--save-amplitude/--no-save-amplitude",
    default=False,
    help="Also save the signal amplitude map.",
)
@click.pass_context
@error_handler
def t2(
    ctx: click.Context,
    input_nifti: str,
    te_values: str,
    output: str,
    method: str,
    mask: str | None,
    skip_echoes: int,
    max_t2: float,
    save_amplitude: bool,
) -> None:
    r"""Fit T2 relaxation time from multi-echo spin echo data.

    INPUT_NIFTI is the path to the 4D NIfTI file with multiple echo volumes.

    The --te option specifies echo times in milliseconds.

    The output T2 map will be in units of milliseconds.

    \b
    Examples:
        # T2 mapping with full model (includes offset term)
        qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80,160" -o t2.nii.gz

        # T2 mapping without skipping first echo
        qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80" \
            --skip-echoes 0 -o t2.nii.gz

        # Use reduced model (no offset term)
        qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80" \
            --method reduced -o t2.nii.gz
    """
    from qmri.io import get_affine, load_nifti, save_nifti
    from qmri.relaxometry import t2 as t2_module

    verbose: bool = ctx.obj.get("verbose", False) if ctx.obj else False

    # Parse TE values (input is in ms, convert to seconds for fitting)
    te_list = parse_values_string(te_values)
    te_array = np.array(te_list, dtype=np.float64) / 1000.0  # ms -> s

    # Convert max_t2 to seconds
    max_t2_seconds = max_t2 / 1000.0

    if verbose:
        console.print(f"[dim]Echo times (s): {te_array}[/]")

    # Validate input files
    input_path = validate_input_file(input_nifti, extensions=(".nii", ".nii.gz"))
    output_path = validate_output_path(output)

    # Load input data
    with create_progress("Loading data") as progress:
        task = progress.add_task("Loading NIfTI...", total=None)
        data, header = load_nifti(input_path)
        affine = get_affine(header)
        progress.update(task, completed=True)

    if data.ndim != 4:
        msg = f"Input must be a 4D NIfTI file, got {data.ndim}D"
        raise ValueError(msg)

    if data.shape[-1] != len(te_array):
        msg = (
            f"Number of volumes ({data.shape[-1]}) does not match "
            f"number of echo times ({len(te_array)})"
        )
        raise ValueError(msg)

    # Load mask if provided
    mask_data: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None
    if mask:
        mask_path = validate_input_file(mask, extensions=(".nii", ".nii.gz"))
        mask_data_raw, _ = load_nifti(mask_path)
        mask_data = mask_data_raw > 0

    # Convert method to correct type
    model_type: Literal["full", "reduced"] = method.lower()  # type: ignore[assignment]

    # Fit T2
    console.print(f"[bold]Fitting T2 using {model_type.upper()} model...[/]")

    with create_progress("Fitting T2") as progress:
        task = progress.add_task("Processing...", total=None)

        result = t2_module.fit(
            signal=data.astype(np.float64),
            echo_times=te_array,
            model=model_type,
            mask=mask_data,
            skip_echoes=skip_echoes,
            max_t2=max_t2_seconds,
        )

        progress.update(task, completed=True)

    # Convert T2 back to milliseconds for output
    t2_ms = result.t2 * 1000.0

    # Save outputs
    save_nifti(t2_ms, output_path, header=header, affine=affine)
    console.print(f"[green]Saved T2 map to:[/] {output_path}")

    if save_amplitude:
        amp_path = output_path.with_name(
            output_path.stem.replace(".nii", "") + "_amplitude.nii.gz"
        )
        save_nifti(result.amplitude, amp_path, header=header, affine=affine)
        console.print(f"[green]Saved amplitude map to:[/] {amp_path}")

    # Print summary statistics
    valid_mask = result.t2 > 0
    if np.any(valid_mask):
        mean_t2 = np.mean(t2_ms[valid_mask])
        std_t2 = np.std(t2_ms[valid_mask])

        console.print()
        console.print("[bold]Summary statistics:[/]")
        console.print(f"  Mean T2: {mean_t2:.1f} ms")
        console.print(f"  Std T2:  {std_t2:.1f} ms")
