# Quick Start

This guide demonstrates the core functionality of qmri through practical examples.

## Basic ADC Fitting

The most common use case is fitting Apparent Diffusion Coefficient (ADC) from
diffusion-weighted MRI data.

### Single Voxel Fitting

```python
import numpy as np
from qmri.diffusion import adc

# Define b-values (diffusion weighting) in s/mm²
b_values = np.array([0, 500, 1000, 2000])

# DWI signal intensities at each b-value
signal = np.array([1000, 606, 368, 135])

# Fit ADC using iterative weighted least squares
result = adc.fit(signal, b_values, method="iwlls")

print(f"ADC: {result.adc:.2e} mm²/s")
print(f"S₀: {result.s0:.0f}")
print(f"R²: {result.r_squared:.4f}")
```

Output:

```
ADC: 1.00e-03 mm²/s
S₀: 1000
R²: 1.0000
```

!!! tip "Fitting Methods"
    qmri supports three fitting methods:

    - `"lls"` - Linear Least Squares (fastest, but can be biased at low SNR)
    - `"wlls"` - Weighted Linear Least Squares
    - `"iwlls"` - Iterative Weighted Linear Least Squares (recommended)

### Volume Fitting

For 4D DWI data (3D volume + b-values), the same `fit` function handles the
entire volume:

```python
import numpy as np
from qmri.diffusion import adc

# Simulated 4D DWI data: (64, 64, 30, 4) = (x, y, z, b-values)
b_values = np.array([0, 500, 1000, 2000])
dwi_data = np.random.rand(64, 64, 30, 4) * 1000

# Fit ADC for the entire volume
result = adc.fit(dwi_data, b_values, method="iwlls")

print(f"ADC map shape: {result.adc.shape}")
print(f"S₀ map shape: {result.s0.shape}")
print(f"R² map shape: {result.r_squared.shape}")
```

### Using a Mask

To fit only within a region of interest:

```python
import numpy as np
from qmri.diffusion import adc

# Create a simple spherical mask
shape = (64, 64, 30)
centre = np.array(shape) // 2
coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
distance = sum((c - ctr) ** 2 for c, ctr in zip(coords, centre))
mask = distance < 15 ** 2

# Fit with mask
b_values = np.array([0, 500, 1000, 2000])
dwi_data = np.random.rand(64, 64, 30, 4) * 1000

result = adc.fit(dwi_data, b_values, method="iwlls", mask=mask)
```

!!! note "Masked Voxels"
    Voxels outside the mask will have zero values in the output maps.

## T1/T2 Relaxometry

### T1 Mapping with Inversion Recovery

```python
import numpy as np
from qmri.relaxometry import t1

# Inversion times in seconds
ti = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])

# Generate synthetic IR signal (T1 = 1.2s)
signal = t1.signal_ir(
    s0=1000,
    t1=1.2,
    inversion_times=ti,
    repetition_times=5.0,
    inversion_efficiency=0.95,
)

# Add some noise
signal = signal + np.random.randn(len(ti)) * 10

# Fit T1
result = t1.fit_ir(signal, ti, repetition_times=5.0)

print(f"T1: {result.t1:.3f} s")
print(f"S₀: {result.s0:.0f}")
print(f"Inversion efficiency: {result.inversion_efficiency:.3f}")
```

!!! tip "IR Models"
    Use `model="general"` (default) for the full IR equation, or
    `model="classical"` when TR >> T1.

### T1 Mapping with Variable TR

```python
import numpy as np
from qmri.relaxometry import t1

# Repetition times in seconds
tr = np.array([0.2, 0.5, 1.0, 2.0, 4.0])

# Generate synthetic VTR signal
signal = t1.signal_vtr(m=1000, t1=1.2, repetition_times=tr)

# Fit T1
result = t1.fit_vtr(signal, tr)

print(f"T1: {result.t1:.3f} s")
print(f"M: {result.m:.0f}")
```

### T2 Mapping

```python
import numpy as np
from qmri.relaxometry import t2

# Echo times in seconds
te = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])

# Generate synthetic T2 decay signal (T2 = 50ms)
signal = t2.signal_decay(amplitude=1000, t2=0.05, echo_times=te)

# Add noise
signal = signal + np.random.randn(len(te)) * 5

# Fit T2 (skip first echo to reduce stimulated echo effects)
result = t2.fit(signal, te, model="full", skip_echoes=1)

print(f"T2: {result.t2 * 1000:.1f} ms")
print(f"Amplitude: {result.amplitude:.0f}")
```

!!! warning "First Echo"
    The first echo in multi-echo sequences often contains stimulated echo
    contributions. Setting `skip_echoes=1` (the default) improves fitting
    accuracy.

## Loading and Saving NIFTI Files

With the `qmri-io` package installed, you can easily work with NIFTI files.

### Loading NIFTI Data

```python
from qmri.io.nifti import load_nifti, load_nifti_image, get_voxel_size

# Load data and header separately
data, header = load_nifti("dwi.nii.gz")
print(f"Data shape: {data.shape}")
print(f"Voxel size: {get_voxel_size(header)} mm")

# Or load as a single object
img = load_nifti_image("dwi.nii.gz")
print(f"Data shape: {img.data.shape}")
print(f"Affine:\n{img.affine}")
```

### Complete Workflow: Load, Process, Save

```python
import numpy as np
from qmri.diffusion import adc
from qmri.io.nifti import load_nifti_image, save_nifti

# Load DWI data
img = load_nifti_image("dwi.nii.gz")

# Define b-values (typically from a .bval file)
b_values = np.array([0, 500, 1000, 2000])

# Fit ADC
result = adc.fit(img.data, b_values, method="iwlls")

# Save the ADC map, preserving spatial information
save_nifti(result.adc, "adc_map.nii.gz", affine=img.affine)

# Save the R² quality map
save_nifti(result.r_squared, "r2_map.nii.gz", affine=img.affine)
```

### Loading b-values from a File

```python
import numpy as np

# FSL-style .bval file (space-separated values on one line)
b_values = np.loadtxt("dwi.bval")

# Or if your file has one value per line
b_values = np.loadtxt("bvalues.txt")
```

## Visualisation

With the `qmri-viz` package installed, you can create publication-quality figures.

### Plotting a Parameter Map

```python
import numpy as np
from qmri.viz.maps import plot_parameter_map

# Create or load an ADC map
adc_map = np.random.rand(64, 64, 32) * 3e-3

# Plot a single slice
fig = plot_parameter_map(
    adc_map,
    slice_idx=16,
    cmap="hot",
    vmin=0,
    vmax=3e-3,
    title="ADC Map",
    colorbar_label="ADC (mm²/s)",
)
fig.savefig("adc_slice.png", dpi=150, bbox_inches="tight")
```

### Multi-slice Display

```python
from qmri.viz.maps import plot_multi_slice

fig = plot_multi_slice(
    adc_map,
    n_slices=12,
    axis=2,  # Axial slices
    cmap="hot",
    vmin=0,
    vmax=3e-3,
    title="ADC Map - Multiple Slices",
    colorbar_label="ADC (mm²/s)",
)
fig.savefig("adc_multislice.png", dpi=150, bbox_inches="tight")
```

### Comparing Methods

```python
import numpy as np
from qmri.diffusion import adc
from qmri.viz.maps import compare_maps

# Generate synthetic DWI data
b_values = np.array([0, 500, 1000, 2000])
dwi_data = np.random.rand(64, 64, 32, 4) * 1000

# Fit with different methods
result_lls = adc.fit(dwi_data, b_values, method="lls")
result_iwlls = adc.fit(dwi_data, b_values, method="iwlls")

# Compare results
fig = compare_maps(
    [result_lls.adc, result_iwlls.adc],
    ["LLS", "IWLLS"],
    slice_idx=16,
    cmap="hot",
    suptitle="ADC Fitting Method Comparison",
    colorbar_label="ADC (mm²/s)",
)
fig.savefig("method_comparison.png", dpi=150, bbox_inches="tight")
```

## Complete Example: DWI Processing Pipeline

Here's a complete example combining all the features:

```python
import numpy as np
from qmri.diffusion import adc
from qmri.io.nifti import load_nifti_image, save_nifti
from qmri.viz.maps import plot_parameter_map, plot_multi_slice

# 1. Load data
dwi = load_nifti_image("dwi.nii.gz")
mask_img = load_nifti_image("brain_mask.nii.gz")
b_values = np.loadtxt("dwi.bval")

# 2. Fit ADC with mask
result = adc.fit(
    dwi.data,
    b_values,
    method="iwlls",
    mask=mask_img.data.astype(bool),
)

# 3. Apply quality threshold
adc_map = result.adc.copy()
adc_map[result.r_squared < 0.8] = 0

# 4. Save results
save_nifti(adc_map, "adc_qc.nii.gz", affine=dwi.affine)
save_nifti(result.r_squared, "r2_map.nii.gz", affine=dwi.affine)

# 5. Generate QC figures
fig = plot_multi_slice(
    adc_map,
    n_slices=9,
    cmap="hot",
    vmin=0,
    vmax=3e-3,
    title="ADC Map (R² > 0.8)",
    colorbar_label="ADC (mm²/s)",
)
fig.savefig("adc_qc.png", dpi=150, bbox_inches="tight")

print("Processing complete!")
print(f"Mean ADC (within mask): {np.mean(adc_map[adc_map > 0]):.2e} mm²/s")
```

## Validating Algorithms with Digital Reference Objects

The `qmri-dro` package provides tools to generate synthetic data with known ground
truth for testing and validation.

### Generating and Validating ADC Fitting

```python
from qmri.dro import dwi
from qmri.diffusion import adc

# Generate synthetic DWI with known ADC
phantom = dwi.generate(
    adc=1.0e-3,           # True ADC in mm²/s
    s0=1000.0,            # Baseline signal
    b_values=(0, 500, 1000, 2000),
    snr=50.0,             # Add realistic noise
    seed=42,              # Reproducible results
)

# Fit ADC
result = adc.fit(phantom.signal, phantom.b_values, method="iwlls")

# Compare to ground truth
true_adc = phantom.ground_truth["adc"].value
error = 100 * (result.adc - true_adc) / true_adc

print(f"True ADC: {true_adc:.2e} mm²/s")
print(f"Fitted ADC: {result.adc:.2e} mm²/s")
print(f"Error: {error:+.1f}%")
```

Output:

```
True ADC: 1.00e-03 mm²/s
Fitted ADC: 9.87e-04 mm²/s
Error: -1.3%
```

### Using Calibration Phantoms

For comprehensive validation, use pre-configured calibration phantoms:

```python
from qmri.dro import dwi
from qmri.diffusion import adc

# Generate phantom with multiple ADC values
phantom = dwi.generate_calibration_phantom(snr=50, seed=42)

print("Calibration Results:")
print("-" * 50)
for i, true_adc in enumerate(phantom.ground_truth["adc"].value):
    result = adc.fit(phantom.signal[i], phantom.b_values)
    error = 100 * (result.adc - true_adc) / true_adc
    print(f"True: {true_adc:.2e}  Fitted: {result.adc:.2e}  Error: {error:+5.1f}%")
```

!!! tip "Interactive Tutorials"
    Try our [Jupyter notebooks](../tutorials/index.md) on Binder for hands-on
    learning with DROs:

    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gold-standard-phantoms/qmri/main?labpath=examples%2Fjupyter)

## Next Steps

- Explore the API Reference for complete function signatures:
    - [qmri.diffusion](../api/diffusion.md) - ADC fitting
    - [qmri.relaxometry](../api/relaxometry.md) - T1/T2 mapping
    - [qmri.dro](../api/dro.md) - Digital Reference Objects
- Browse the [User Guide](../user-guide/diffusion.md) for more detailed examples
- Try the [interactive tutorials](../tutorials/index.md) on Binder
