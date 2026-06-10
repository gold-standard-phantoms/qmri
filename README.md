# qmri

Pure MRI signal models, fitting algorithms, and error propagation for quantitative MRI.

[![PyPI version](https://badge.fury.io/py/qmri.svg)](https://badge.fury.io/py/qmri)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gold-standard-phantoms/qmri/main?labpath=examples%2Fjupyter)

## Overview

`qmri` provides mathematical models for quantitative MRI (qMRI) analysis. It is designed to be:

- **Pure** — Core package depends only on NumPy and SciPy
- **Simple** — Clean function-based API, no complex abstractions
- **Typed** — Full type annotations for IDE support
- **Tested** — Comprehensive test coverage with property-based testing

## Installation

```bash
# Core package (numpy + scipy only)
pip install qmri

# With file I/O support (adds nibabel)
pip install qmri-io

# With end-to-end processing pipelines
pip install qmri-pipelines

# With CLI (adds click, rich)
pip install qmri-cli

# With visualisation (adds matplotlib)
pip install qmri-viz
```

## Quick Start

### ADC Fitting

```python
import numpy as np
from qmri.diffusion import adc

# DWI signal data
b_values = np.array([0, 500, 1000, 2000])
signal = np.array([1000, 606, 368, 135])

# Fit ADC using iterative weighted least squares
result = adc.fit(signal, b_values, method="iwlls")

print(f"ADC: {result.adc:.2e} mm²/s")
print(f"S₀: {result.s0:.0f}")
print(f"R²: {result.r_squared:.4f}")
```

### With File I/O

```python
from qmri.diffusion import adc
from qmri.io import nifti

# Load DWI data
dwi = nifti.load("dwi.nii.gz")

# Fit ADC
result = adc.fit(dwi.data, dwi.b_values, method="iwlls")

# Save result
nifti.save(result.adc, dwi.affine, "adc_map.nii.gz")
```

### Command Line

```bash
# Fit ADC map
qmri adc fit dwi.nii.gz --bval dwi.bval --output adc.nii.gz

# With mask and quality threshold
qmri adc fit dwi.nii.gz --bval dwi.bval --mask brain.nii.gz --r2-threshold 0.8 -o adc.nii.gz
```

## Packages

| Package | Description | Dependencies |
|---------|-------------|--------------|
| `qmri` | Core signal models and fitting | numpy, scipy |
| `qmri-io` | NIFTI/DICOM/BIDS I/O | + nibabel |
| `qmri-pipelines` | End-to-end file-in / file-out workflows | qmri, qmri-io |
| `qmri-cli` | Command-line interface | + click, rich |
| `qmri-viz` | Visualisation utilities | + matplotlib |
| `qmri-dro` | Digital Reference Objects for validation | qmri |

## Modules

### Core (`qmri`)

- `qmri.diffusion` — ADC fitting, DWI signal generation, b-value calibration
- `qmri.relaxometry` — T1 and T2 mapping
- `qmri.perfusion` — ASL quantification, General Kinetic Model
- `qmri.thermometry` — MR thermometry models
- `qmri.transfer` — Magnetisation transfer (MTR)
- `qmri.sequences` — GRE, SE, IR signal equations
- `qmri.fitting` — Least squares algorithms, bootstrap
- `qmri.errors` — R², RMSE, uncertainty propagation
- `qmri.constants` — Physical constants (γ, tissue properties)
- `qmri.dro` — Digital Reference Objects for validation and testing

## Development

```bash
# Clone repository
git clone https://github.com/gold-standard-phantoms/qmri.git
cd qmri

# Install with uv (recommended)
uv sync

# Run tests
uv run pytest

# Type checking
uv run mypy packages/qmri/src

# Linting
uv run ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{qmri,
  author = {Gold Standard Phantoms},
  title = {qmri: Pure MRI signal models for quantitative MRI},
  url = {https://github.com/gold-standard-phantoms/qmri},
  year = {2026}
}
```
