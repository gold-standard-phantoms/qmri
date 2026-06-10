# qmri

**Pure MRI signal models, fitting algorithms, and error propagation for quantitative MRI.**

qmri provides mathematical models for quantitative MRI (qMRI) analysis. It is designed to be:

- **Pure** — Core package depends only on NumPy and SciPy
- **Simple** — Clean function-based API, no complex abstractions
- **Typed** — Full type annotations for IDE support
- **Tested** — Comprehensive test coverage

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

# With Digital Reference Objects for validation
pip install qmri-dro
```

## Quick Example

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

- **[qmri.diffusion](api/diffusion.md)** — ADC fitting, DWI signal generation, b-value calibration
- **[qmri.relaxometry](api/relaxometry.md)** — T1 and T2 mapping
- **[qmri.perfusion](api/perfusion.md)** — ASL quantification, General Kinetic Model
- **[qmri.thermometry](api/thermometry.md)** — MR thermometry models
- **[qmri.fitting](api/fitting.md)** — Least squares algorithms, bootstrap
- **[qmri.errors](api/errors.md)** — R², RMSE, uncertainty propagation
- **[qmri.constants](api/constants.md)** — Physical constants
- **[qmri.dro](api/dro.md)** — Digital Reference Objects for validation and testing

## License

MIT License — see [LICENSE](https://github.com/gold-standard-phantoms/qmri/blob/main/LICENSE) for details.
