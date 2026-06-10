"""Diffusion imaging commands for qmri CLI.

This module provides commands for fitting diffusion models such as ADC
from diffusion-weighted MRI data.
"""

from typing import Literal

import click
import numpy as np
from qmri.cli._utils import (
    console,
    create_progress,
    error_handler,
    parse_values_file,
    validate_input_file,
    validate_output_path,
)

__all__ = ["diffusion"]


@click.group(name="diffusion")
def diffusion() -> None:
    """Diffusion imaging analysis commands.

    Commands for fitting diffusion models from DWI data.

    Examples:
    --------
        qmri diffusion adc dwi.nii.gz bvalues.txt -o adc_map.nii.gz

        qmri diffusion adc dwi.nii.gz bvalues.txt --method iwlls -o adc.nii.gz
    """
    pass


@diffusion.command(name="adc")
@click.argument("input_nifti", type=click.Path(exists=True))
@click.argument("bvalues_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for the ADC map (NIfTI format).",
)
@click.option(
    "--method",
    type=click.Choice(["lls", "wlls", "iwlls"], case_sensitive=False),
    default="iwlls",
    help="Fitting method: lls, wlls, or iwlls (default: iwlls).",
)
@click.option(
    "--mask",
    type=click.Path(exists=True),
    default=None,
    help="Optional mask file (NIfTI). Only voxels where mask > 0 will be processed.",
)
@click.option(
    "--b0-threshold",
    type=float,
    default=50.0,
    help="Maximum b-value to consider as b=0 (default: 50 s/mm^2).",
)
@click.option(
    "--save-s0/--no-save-s0",
    default=False,
    help="Also save the S0 (baseline signal) map.",
)
@click.option(
    "--save-r2/--no-save-r2",
    default=False,
    help="Also save the R-squared quality map.",
)
@click.pass_context
@error_handler
def adc(
    ctx: click.Context,
    input_nifti: str,
    bvalues_file: str,
    output: str,
    method: str,
    mask: str | None,
    b0_threshold: float,
    save_s0: bool,
    save_r2: bool,
) -> None:
    r"""Fit Apparent Diffusion Coefficient (ADC) from DWI data.

    INPUT_NIFTI is the path to the 4D diffusion-weighted NIfTI file.

    BVALUES_FILE is a text file containing b-values (one per line or space-separated).

    The output ADC map will be in units of mm^2/s.

    \b
    Examples:
        # Basic ADC fitting with default IWLLS method
        qmri diffusion adc dwi.nii.gz bvalues.txt -o adc.nii.gz

        # Use linear least squares (faster but less accurate)
        qmri diffusion adc dwi.nii.gz bvalues.txt --method lls -o adc.nii.gz

        # Apply a brain mask and save quality metrics
        qmri diffusion adc dwi.nii.gz bvalues.txt --mask brain.nii.gz \
            --save-s0 --save-r2 -o adc.nii.gz
    """
    from qmri.diffusion import adc as adc_module
    from qmri.io import get_affine, load_nifti, save_nifti

    verbose: bool = ctx.obj.get("verbose", False) if ctx.obj else False

    # Validate input files
    input_path = validate_input_file(input_nifti, extensions=(".nii", ".nii.gz"))
    bvalues_path = validate_input_file(bvalues_file)
    output_path = validate_output_path(output)

    # Load b-values
    b_values = np.array(parse_values_file(bvalues_path), dtype=np.float64)

    if verbose:
        console.print(f"[dim]Loaded {len(b_values)} b-values: {b_values}[/]")

    # Load input data
    with create_progress("Loading data") as progress:
        task = progress.add_task("Loading NIfTI...", total=None)
        dwi_data, header = load_nifti(input_path)
        affine = get_affine(header)
        progress.update(task, completed=True)

    if dwi_data.ndim != 4:
        msg = f"Input must be a 4D NIfTI file, got {dwi_data.ndim}D"
        raise ValueError(msg)

    if dwi_data.shape[-1] != len(b_values):
        msg = (
            f"Number of volumes ({dwi_data.shape[-1]}) does not match "
            f"number of b-values ({len(b_values)})"
        )
        raise ValueError(msg)

    # Load mask if provided
    mask_data: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None
    if mask:
        mask_path = validate_input_file(mask, extensions=(".nii", ".nii.gz"))
        mask_data_raw, _ = load_nifti(mask_path)
        mask_data = mask_data_raw > 0

    # Convert method to correct type
    fitting_method: Literal["lls", "wlls", "iwlls"] = method.lower()  # type: ignore[assignment]

    # Fit ADC
    console.print(f"[bold]Fitting ADC using {fitting_method.upper()} method...[/]")

    with create_progress("Fitting ADC") as progress:
        task = progress.add_task("Processing...", total=None)

        result = adc_module.fit(
            signal=dwi_data.astype(np.float64),
            b_values=b_values,
            method=fitting_method,
            mask=mask_data,
        )

        progress.update(task, completed=True)

    # For 4D input, result is ADCMapResult with array attributes
    adc_map = np.asarray(result.adc)
    s0_map = np.asarray(result.s0)
    r2_map = np.asarray(result.r_squared)

    # Save outputs
    save_nifti(adc_map, output_path, header=header, affine=affine)
    console.print(f"[green]Saved ADC map to:[/] {output_path}")

    if save_s0:
        stem = output_path.stem.replace(".nii", "")
        s0_path = output_path.with_name(stem + "_s0.nii.gz")
        save_nifti(s0_map, s0_path, header=header, affine=affine)
        console.print(f"[green]Saved S0 map to:[/] {s0_path}")

    if save_r2:
        stem = output_path.stem.replace(".nii", "")
        r2_path = output_path.with_name(stem + "_r2.nii.gz")
        save_nifti(r2_map, r2_path, header=header, affine=affine)
        console.print(f"[green]Saved R² map to:[/] {r2_path}")

    # Print summary statistics
    valid_mask = adc_map > 0
    if np.any(valid_mask):
        mean_adc = float(np.mean(adc_map[valid_mask]))
        std_adc = float(np.std(adc_map[valid_mask]))
        mean_r2 = float(np.mean(r2_map[valid_mask]))

        console.print()
        console.print("[bold]Summary statistics:[/]")
        console.print(f"  Mean ADC: {mean_adc:.4e} mm²/s")
        console.print(f"  Std ADC:  {std_adc:.4e} mm²/s")
        console.print(f"  Mean R²:  {mean_r2:.4f}")
