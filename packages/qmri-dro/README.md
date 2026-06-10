# qmri-dro

Digital Reference Objects (DROs) for quantitative MRI validation.

## Overview

`qmri-dro` provides tools for generating synthetic MRI data with known ground
truth parameters. This is essential for:

- **Validation**: Test fitting algorithms against known ground truth
- **Education**: Demonstrate how MRI parameters affect signal
- **Benchmarking**: Compare fitting methods (e.g., LLS vs WLLS vs IWLLS)
- **Testing**: Reproducible synthetic data for unit tests
- **Sensitivity analysis**: Study noise/parameter effects on results

## Installation

```bash
pip install qmri-dro
```

Or with uv:

```bash
uv add qmri-dro
```

## Quick Start

### DWI Phantom

```python
from qmri.dro import dwi
from qmri.diffusion import adc

# Generate a phantom with known ADC
phantom = dwi.generate(adc=1e-3, s0=1000, snr=50, seed=42)

# Fit and compare to ground truth
result = adc.fit(phantom.signal, phantom.b_values)
print(f"True ADC: {phantom.ground_truth['adc'].value:.2e}")
print(f"Fitted ADC: {result.adc:.2e}")
```

### T1 Phantom

```python
from qmri.dro import relaxometry
from qmri.relaxometry import t1

# Generate IR data with known T1
phantom = relaxometry.generate_t1_ir(
    t1=1.2,
    inversion_times=[0.1, 0.5, 1.0, 2.0, 3.0],
    repetition_time=5.0,
    snr=100,
    seed=42,
)

# Fit and validate
result = t1.fit_ir(phantom.signal, phantom.time_points, repetition_times=5.0)
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
    snr=50,
    seed=42,
)

# Access the difference signal
delta_m = phantom.control - phantom.label
print(f"True CBF: {phantom.ground_truth['perfusion_rate'].value} ml/100g/min")
```

## Features

- **Noise models**: Gaussian and Rician noise with reproducible seeding
- **Ground truth tracking**: All phantoms include documented ground truth values
- **Multi-voxel support**: Generate single voxels or calibration phantoms
- **Calibration phantoms**: Pre-configured phantoms for standardised testing

## Documentation

Full documentation: https://gold-standard-phantoms.github.io/qmri

## Licence

MIT
