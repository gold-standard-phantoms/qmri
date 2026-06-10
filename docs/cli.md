# CLI Reference

The `qmri-cli` package provides a command-line interface for quantitative MRI analysis. It wraps the core `qmri` library functionality in a convenient CLI using [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/) for formatted output.

## Installation

```bash
pip install qmri-cli
```

This will install the `qmri` command along with the core `qmri`, `qmri-io` and
`qmri-pipelines` packages.

## Global Options

All commands support the following global options:

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Enable verbose output with additional information |
| `-q`, `--quiet` | Suppress non-essential output |
| `-d`, `--output-dir PATH` | Default output directory for generated files |
| `--version` | Show version information |
| `--help` | Show help message |

## Commands

### `qmri info`

Display information about qmri and available commands.

```bash
qmri info
```

Shows version information for installed packages and a summary of available commands.

---

## Diffusion Commands

### `qmri diffusion adc`

Fit Apparent Diffusion Coefficient (ADC) from diffusion-weighted imaging (DWI) data.

```bash
qmri diffusion adc INPUT_NIFTI BVALUES_FILE -o OUTPUT [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_NIFTI` | Path to the 4D diffusion-weighted NIfTI file |
| `BVALUES_FILE` | Text file containing b-values (one per line or space-separated) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output` | PATH | *Required* | Output path for the ADC map (NIfTI format) |
| `--method` | `lls` / `wlls` / `iwlls` | `iwlls` | Fitting method |
| `--mask` | PATH | None | Optional mask file (NIfTI). Only voxels where mask > 0 will be processed |
| `--b0-threshold` | FLOAT | 50.0 | Maximum b-value to consider as b=0 (s/mm^2) |
| `--save-s0` / `--no-save-s0` | FLAG | False | Also save the S0 (baseline signal) map |
| `--save-r2` / `--no-save-r2` | FLAG | False | Also save the R-squared quality map |

#### Fitting Methods

- **lls** (Linear Least Squares): Fastest method. Linearises the mono-exponential model using logarithm. Less accurate for low SNR.
- **wlls** (Weighted Linear Least Squares): Applies signal-dependent weights to improve accuracy over LLS.
- **iwlls** (Iterative Weighted Linear Least Squares): Iteratively refines weights for best accuracy. Default and recommended.

#### Output

- ADC map in units of mm^2/s
- Optionally S0 map (baseline signal intensity)
- Optionally R^2 map (coefficient of determination)

#### Examples

```bash
# Basic ADC fitting with default IWLLS method
qmri diffusion adc dwi.nii.gz bvalues.txt -o adc.nii.gz

# Use linear least squares (faster but less accurate)
qmri diffusion adc dwi.nii.gz bvalues.txt --method lls -o adc.nii.gz

# Apply a brain mask and save quality metrics
qmri diffusion adc dwi.nii.gz bvalues.txt --mask brain.nii.gz \
    --save-s0 --save-r2 -o adc.nii.gz

# Verbose output
qmri -v diffusion adc dwi.nii.gz bvalues.txt -o adc.nii.gz
```

---

## Relaxometry Commands

### `qmri relaxometry t1`

Fit T1 relaxation time from inversion recovery (IR) or variable TR (VTR) data.

```bash
qmri relaxometry t1 INPUT_NIFTI --ti "TIMES" -o OUTPUT [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_NIFTI` | Path to the 4D NIfTI file with multiple TI or TR volumes |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--ti` | STRING | *Required* | Inversion times in ms (comma/space separated) |
| `-o`, `--output` | PATH | *Required* | Output path for the T1 map (NIfTI format) |
| `--method` | `ir` / `ir_classical` / `vtr` | `ir` | Fitting method |
| `--tr` | FLOAT | None | Repetition time in milliseconds (required for IR methods) |
| `--mask` | PATH | None | Optional mask file (NIfTI) |
| `--max-t1` | FLOAT | 20000.0 | Maximum valid T1 value in milliseconds |
| `--save-s0` / `--no-save-s0` | FLAG | False | Also save the S0 (signal amplitude) map |

#### Fitting Methods

- **ir**: Inversion recovery with general model (accounts for finite TR)
- **ir_classical**: Classical IR model assuming TR >> T1
- **vtr**: Variable TR method for spoiled gradient echo sequences

#### Output

- T1 map in units of milliseconds

#### Examples

```bash
# T1 mapping with inversion recovery (general model)
qmri relaxometry t1 ir_data.nii.gz --ti "100,500,1000,2000" \
    --tr 5000 -o t1.nii.gz

# T1 mapping with classical IR model (assumes TR >> T1)
qmri relaxometry t1 ir_data.nii.gz --ti "100,500,1000,2000" \
    --method ir_classical -o t1.nii.gz

# T1 mapping with variable TR method
qmri relaxometry t1 vtr_data.nii.gz --ti "500,1000,2000,4000" \
    --method vtr -o t1.nii.gz
```

---

### `qmri relaxometry t2`

Fit T2 relaxation time from multi-echo spin echo data.

```bash
qmri relaxometry t2 INPUT_NIFTI --te "TIMES" -o OUTPUT [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_NIFTI` | Path to the 4D NIfTI file with multiple echo volumes |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--te` | STRING | *Required* | Echo times in milliseconds (comma/space separated) |
| `-o`, `--output` | PATH | *Required* | Output path for the T2 map (NIfTI format) |
| `--method` | `full` / `reduced` | `full` | Model type |
| `--mask` | PATH | None | Optional mask file (NIfTI) |
| `--skip-echoes` | INT | 1 | Number of initial echoes to skip |
| `--max-t2` | FLOAT | 5000.0 | Maximum valid T2 value in milliseconds |
| `--save-amplitude` / `--no-save-amplitude` | FLAG | False | Also save the signal amplitude map |

#### Model Types

- **full**: Includes an offset term to account for noise floor and stimulated echoes
- **reduced**: Simple mono-exponential decay without offset

#### Output

- T2 map in units of milliseconds

#### Examples

```bash
# T2 mapping with full model (includes offset term)
qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80,160" -o t2.nii.gz

# T2 mapping without skipping first echo
qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80" \
    --skip-echoes 0 -o t2.nii.gz

# Use reduced model (no offset term)
qmri relaxometry t2 mese_data.nii.gz --te "10,20,40,80" \
    --method reduced -o t2.nii.gz
```

---

## Thermometry Commands

### `qmri thermometry multiecho`

Estimate absolute temperature from multi-echo magnitude data using the
dual-resonance (ethylene glycol) model, driven by a segmentation. This command
wraps the [`qmri.pipelines.thermometry`](api/pipelines.md) pipeline.

```bash
qmri thermometry multiecho INPUT_NIFTI... -e ECHO_TIMES -s SEGMENTATION -o OUTPUT_DIR [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_NIFTI...` | One or more 4D multi-echo magnitude NIfTI files (echo dimension last). Echoes from all images are concatenated and sorted by echo time before fitting. |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-s`, `--segmentation` | PATH | *Required* | Segmentation/label-map NIfTI (3D). Label `0` is treated as background. |
| `-e`, `--echo-times` | PATH | *Required* | Echo-time text file in **seconds**, one per multi-echo image, in the same order. May be passed multiple times. |
| `--method` | `regionwise` / `voxelwise` / `regionwise_bootstrap` | `regionwise` | Analysis method. |
| `--n-bootstrap` | INT | 100 | Number of bootstrap iterations (`regionwise_bootstrap` only). |
| `--df-init` | `multistart` / `fixed` / `lombscargle` | `multistart` | Frequency starting-value strategy for the fit (see below). |
| `-b`, `--field-strength` | FLOAT | None | Magnetic field strength in Tesla. Overrides JSON sidecar detection. |
| `-o`, `--output-dir` | PATH | Directory of the first input image | Output directory. |
| `--output-prefix` | STRING | Stem of the first input image | Output filename prefix. |

#### Analysis Methods

- **regionwise**: Fit the mean signal within each labelled region once and assign
  the resulting temperature to every voxel in the region. Fastest; uncertainty
  comes from the fitted frequency-difference covariance.
- **voxelwise**: Fit each voxel independently; the region summary is an
  inverse-variance weighted mean of the voxel temperatures.
- **regionwise_bootstrap**: Resample region voxels with replacement, fit each
  sample's mean signal, and summarise with the mean and standard deviation over
  samples passing the R² threshold.

#### Frequency Starting Values (`--df-init`)

The fit must be seeded with a starting value for the frequency difference Δf.
The dual-resonance magnitude signal has local minima spaced roughly one over the
echo-train span apart, so a poor seed can converge to the wrong frequency alias
(a cold ~10 °C phantom can be mis-fitted as ~185 °C on aliasing-prone echo-time
grids).

- **multistart** (default): fit from both a fixed default and a data-driven
  Lomb-Scargle estimate, keeping the higher-R² result. Most robust.
- **fixed**: a single fit from the fixed default. Cheapest; adequate for
  well-conditioned acquisitions but can alias on cold phantoms.
- **lombscargle**: a single fit seeded from the Lomb-Scargle estimate.

On the bundled phantom data all three agree to within ~0.3 °C; `multistart` is
the default as cheap insurance against the aliasing failure on other protocols.

#### Field Strength Detection

The temperature calibration is specific to ethylene glycol and depends on the
magnetic field strength $B_0$. Unless `--field-strength` is given, $B_0$ is read
from a JSON sidecar next to the first input image, using `ImagingFrequency`
(divided by the proton gyromagnetic ratio) or `MagneticFieldStrength`.

#### Output

- A temperature-map NIfTI (°C) co-located with the segmentation.
- A JSON report with per-region temperatures, uncertainties and fit parameters.

#### Examples

```bash
# Single multi-echo image, region-wise fit, B0 from JSON sidecar
qmri thermometry multiecho echoes.nii.gz -e echo_times.txt \
    -s labels.nii.gz -o results/

# Two echo blocks, bootstrap uncertainty, explicit field strength
qmri thermometry multiecho block1.nii.gz block2.nii.gz \
    -e te1.txt -e te2.txt -s labels.nii.gz \
    --method regionwise_bootstrap --n-bootstrap 200 \
    --field-strength 3.0 -o results/
```

---

## Common Usage Patterns

### Processing with Masks

Using a mask significantly speeds up processing by only fitting voxels of interest:

```bash
# Create a brain mask first (using FSL or similar)
bet dwi_b0.nii.gz brain -m

# Use the mask for fitting
qmri diffusion adc dwi.nii.gz bvalues.txt --mask brain_mask.nii.gz -o adc.nii.gz
```

### Batch Processing

For processing multiple subjects, combine with shell loops:

```bash
for subj in sub-01 sub-02 sub-03; do
    qmri diffusion adc ${subj}/dwi.nii.gz ${subj}/bvalues.txt \
        -o ${subj}/adc.nii.gz
done
```

### Quality Control

Save R-squared maps to assess fit quality:

```bash
qmri diffusion adc dwi.nii.gz bvalues.txt --save-r2 -o adc.nii.gz
# Creates: adc.nii.gz, adc_r2.nii.gz
```

### Verbose Output

Use verbose mode for debugging and detailed information:

```bash
qmri -v diffusion adc dwi.nii.gz bvalues.txt -o adc.nii.gz
```
