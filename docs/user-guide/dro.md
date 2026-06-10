# Digital Reference Objects (DROs)

Digital Reference Objects (DROs) are synthetic datasets with known ground truth parameters, essential for validating and benchmarking quantitative MRI algorithms.

## Why Use DROs?

DROs are invaluable for:

- **Validation**: Test fitting algorithms against known ground truth
- **Education**: Demonstrate how MRI parameters affect signal
- **Benchmarking**: Compare fitting methods (e.g., LLS vs WLLS vs IWLLS)
- **Testing**: Reproducible synthetic data for unit tests
- **Sensitivity analysis**: Study noise and parameter effects on results

## Quick Start

### DWI Phantom

Generate synthetic DWI data with known ADC:

```python
from qmri.dro import dwi
from qmri.diffusion import adc

# Generate phantom with known ADC
phantom = dwi.generate(
    adc=1e-3,           # mm²/s
    s0=1000.0,          # baseline signal
    b_values=(0, 500, 1000, 2000),  # s/mm²
    snr=50.0,           # signal-to-noise ratio
    seed=42,            # for reproducibility
)

# Access ground truth
print(f"True ADC: {phantom.ground_truth['adc'].value:.2e} mm²/s")

# Fit and compare
result = adc.fit(phantom.signal, phantom.b_values)
print(f"Fitted ADC: {result.adc:.2e} mm²/s")
```

### T1 Phantom (Inversion Recovery)

```python
from qmri.dro import relaxometry
from qmri.relaxometry import t1

# Generate IR data with known T1
phantom = relaxometry.generate_t1_ir(
    t1=1.2,             # seconds
    inversion_times=[0.1, 0.5, 1.0, 2.0, 3.0],
    s0=1000.0,
    repetition_time=5.0,
    snr=100.0,
    seed=42,
)

# Fit and validate
result = t1.fit_ir(
    phantom.signal,
    phantom.time_points,
    repetition_times=phantom.repetition_time
)
print(f"True T1: {phantom.ground_truth['t1'].value:.2f} s")
print(f"Fitted T1: {float(result.t1):.2f} s")
```

### ASL Phantom

```python
from qmri.dro import perfusion

# Generate pCASL data with known perfusion
phantom = perfusion.generate_pcasl(
    perfusion_rate=60.0,  # ml/100g/min
    m0=1000.0,
    transit_time=1.0,
    label_duration=1.8,
    post_label_delay=1.8,
    snr=50.0,
    seed=42,
)

# Access difference signal
delta_m = phantom.control - phantom.label
print(f"True CBF: {phantom.ground_truth['perfusion_rate'].value} ml/100g/min")
```

## Noise Models

DROs support two noise models:

### Rician Noise (Default)

Rician noise is the appropriate model for magnitude MRI images:

$$S_{noisy} = \sqrt{(S + \epsilon_r)^2 + \epsilon_i^2}$$

where $\epsilon_r, \epsilon_i \sim \mathcal{N}(0, \sigma^2)$.

```python
phantom = dwi.generate(adc=1e-3, snr=50, noise_model="rician")
```

### Gaussian Noise

For complex data or high SNR approximations:

```python
phantom = dwi.generate(adc=1e-3, snr=50, noise_model="gaussian")
```

!!! note "Rician Bias"
    At low SNR, Rician noise causes a positive bias in signal estimates.
    Use Rician noise for realistic simulations of magnitude images.

## Multi-Voxel Phantoms

Generate phantoms with spatially varying parameters:

```python
import numpy as np
from qmri.dro import dwi

# Create ADC map
adc_map = np.array([
    [0.5e-3, 1.0e-3],
    [1.5e-3, 2.0e-3]
])

# Generate multi-voxel phantom
phantom = dwi.generate(
    adc=adc_map,
    snr=50.0,
    seed=42,
)

print(f"Signal shape: {phantom.signal.shape}")
# Signal shape: (2, 2, 4)  # (spatial_x, spatial_y, n_bvalues)
```

## Calibration Phantoms

Pre-configured phantoms for standardised testing:

```python
from qmri.dro import dwi

# Generate calibration phantom with multiple ADC values
phantom = dwi.generate_calibration_phantom(
    adc_values=(0.3e-3, 0.7e-3, 1.0e-3, 1.5e-3, 2.0e-3, 3.0e-3),
    b_values=(0, 50, 100, 200, 400, 600, 800, 1000),
    snr=50.0,
    seed=42,
)

# Fit each "tube" and compare to ground truth
for i, true_adc in enumerate(phantom.ground_truth['adc'].value):
    result = adc.fit(phantom.signal[i], phantom.b_values)
    print(f"True: {true_adc:.2e}  Fitted: {result.adc:.2e}")
```

## Ground Truth Access

All phantoms include documented ground truth:

```python
phantom = dwi.generate(adc=1e-3, s0=1000.0, snr=50)

# Ground truth is a dictionary of GroundTruth objects
for name, gt in phantom.ground_truth.items():
    print(f"{name}: {gt.value} {gt.units}")
    print(f"  Description: {gt.description}")
```

## Reproducibility

Use seeds for reproducible results:

```python
# Same seed = identical output
p1 = dwi.generate(adc=1e-3, snr=50, seed=42)
p2 = dwi.generate(adc=1e-3, snr=50, seed=42)
np.testing.assert_array_equal(p1.signal, p2.signal)

# Different seeds = different noise realisations
p3 = dwi.generate(adc=1e-3, snr=50, seed=43)
assert not np.array_equal(p1.signal, p3.signal)
```

## Interactive Tutorials

Try our interactive Jupyter notebooks on Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gold-standard-phantoms/qmri/main?labpath=examples%2Fjupyter)

Available tutorials:

1. **ADC Fitting Workflow** — Single and multi-voxel ADC fitting
2. **T1 Mapping Synthetic** — IR and VTR T1 mapping
3. **ASL Perfusion Quantification** — pCASL signal generation
4. **Method Comparison Benchmark** — LLS vs WLLS vs IWLLS
5. **Noise Sensitivity Analysis** — Gaussian vs Rician noise effects

## API Reference

See the [qmri.dro API documentation](../api/dro.md) for complete function signatures.
