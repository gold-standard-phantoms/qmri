# Architecture Overview

This document explains the design philosophy and architecture of qmri.

## Design Philosophy

### Simple Functions Over Complex Frameworks

qmri deliberately avoids complex abstractions like filter pipelines, image containers, or plugin systems. Instead, it provides simple, composable functions that operate on numpy arrays.

**Why?**

1. **Most users don't need pipelines** — Researchers typically run a single operation, not complex workflows
2. **Notebook compatibility** — Simple functions work naturally in Jupyter; frameworks don't
3. **Easier to understand** — `result = fit(data, params)` is clearer than filter wiring
4. **Easier to test** — Pure functions are trivially unit-testable
5. **Flexible composition** — Users combine functions however they need

```python
# qmri approach: simple and clear
from qmri.diffusion import adc
from qmri.io import nifti

dwi = nifti.load("dwi.nii.gz")
result = adc.fit(dwi.data, dwi.b_values)
nifti.save(result.adc, dwi.affine, "adc.nii.gz")
```

### Separation of Concerns

The monorepo structure enforces clean separation:

| Package | Responsibility | Dependencies |
|---------|---------------|--------------|
| `qmri` | Signal physics, fitting algorithms | numpy, scipy |
| `qmri-io` | File formats, metadata | + nibabel |
| `qmri-cli` | Command-line interface | + click, rich |
| `qmri-viz` | Plotting, visualisation | + matplotlib |
| `qmri-dro` | Digital Reference Objects | qmri |

This means:

- A researcher using `qmri` in a notebook doesn't need nibabel installed
- The core physics is testable without mocking file I/O
- Each package can be versioned and released independently

### Pure Functions, Structured Results

Functions should be pure (no side effects) and return structured results:

```python
@dataclass(frozen=True)
class ADCResult:
    """Immutable result container."""
    adc: np.ndarray
    s0: np.ndarray
    r_squared: np.ndarray
    residuals: np.ndarray | None = None


def fit(signal: np.ndarray, b_values: np.ndarray, **options) -> ADCResult:
    """Pure function: arrays in, result out."""
    ...
```

Benefits:

- Results are immutable and hashable
- IDE autocompletion works
- Easy to serialise
- Clear contract

### No Image Format Assumptions

Core functions operate on raw numpy arrays without assuming image structure:

```python
# Works with any array shape
result = adc.fit(signal_4d, b_values)  # (X, Y, Z, B) -> ADCResult

# The I/O package handles format-specific concerns
from qmri.io import nifti
img = nifti.load("dwi.nii.gz")
result = adc.fit(img.data, img.b_values)
nifti.save(result.adc, img.affine, "adc.nii.gz")
```

## Package Structure

### Core Package (`qmri`)

```
packages/qmri/src/qmri/
├── __init__.py
├── py.typed                 # PEP 561 marker
│
├── diffusion/               # Diffusion-weighted imaging
│   ├── __init__.py
│   ├── adc.py              # ADC fitting
│   ├── signal.py           # DWI signal generation
│   └── calibration.py      # B-value calibration
│
├── relaxometry/             # T1/T2 mapping
│   ├── __init__.py
│   ├── t1.py
│   └── t2.py
│
├── perfusion/               # ASL perfusion
│   ├── __init__.py
│   ├── asl.py
│   └── gkm.py
│
├── thermometry/             # MR thermometry
│   └── ...
│
├── transfer/                # Magnetisation transfer
│   └── mtr.py
│
├── sequences/               # MRI pulse sequences
│   └── signal.py           # GRE, SE, IR equations
│
├── fitting/                 # Fitting algorithms
│   ├── least_squares.py    # LLS, WLLS, IWLLS
│   ├── curve_fit.py        # scipy wrappers
│   └── bootstrap.py        # Bootstrap estimation
│
├── errors/                  # Error analysis
│   ├── metrics.py          # R², RMSE
│   ├── propagation.py      # Uncertainty propagation
│   └── covariance.py
│
├── noise/                   # Noise models
│   └── models.py           # Gaussian, Rician
│
├── constants/               # Physical constants
│   ├── physical.py         # γ, tissue properties
│   └── units.py            # Unit conversions
│
└── _utils/                  # Internal utilities
    ├── safe_maths.py       # Safe division
    └── validation.py       # Input validation
```

### I/O Package (`qmri-io`)

```
packages/qmri-io/src/qmri/io/
├── __init__.py
├── nifti.py                # NIFTI loading/saving
├── dicom.py                # DICOM support
├── bids.py                 # BIDS format support
└── sidecar.py              # .bval, .bvec, .json
```

### CLI Package (`qmri-cli`)

```
packages/qmri-cli/src/qmri/cli/
├── __init__.py
├── main.py                 # Entry point
├── adc.py                  # qmri adc ...
├── t1.py                   # qmri t1 ...
├── t2.py                   # qmri t2 ...
└── thermometry.py          # qmri thermometry ...
```

## API Conventions

### Function Signatures

All fitting functions follow a consistent pattern:

```python
def fit_<quantity>(
    signal: np.ndarray,           # Measured signal (N-D array)
    independent_var: np.ndarray,  # e.g., b_values, echo_times
    *,                            # Keyword-only after this
    method: str = "default",      # Fitting method
    mask: np.ndarray | None,      # Optional processing mask
    **options,                    # Method-specific options
) -> <Quantity>Result:
    ...
```

All signal generation functions follow:

```python
def signal_<sequence>(
    params: <Sequence>Params | dict,  # Physical parameters
    independent_var: np.ndarray,       # e.g., echo_times
    **options,
) -> np.ndarray:
    ...
```

### Type Hints

Full type annotations are required:

```python
from typing import Literal
import numpy as np
from numpy.typing import NDArray

def fit(
    signal: NDArray[np.floating],
    b_values: NDArray[np.floating],
    *,
    method: Literal["lls", "wlls", "iwlls"] = "iwlls",
    mask: NDArray[np.bool_] | None = None,
) -> ADCResult:
    ...
```

### Docstrings

All physics functions include LaTeX equations:

```python
def signal_gradient_echo(...) -> np.ndarray:
    r"""Calculate gradient echo signal.

    Implements the steady-state GRE signal equation:

    $$S = M_0 \sin(\theta) \frac{1 - e^{-TR/T_1}}{1 - \cos(\theta) e^{-TR/T_1}} e^{-TE/T_2^*}$$

    Args:
        ...
    """
```

## Dependency Policy

### Core Package

The core `qmri` package has minimal dependencies:

- `numpy>=1.24` — Array operations
- `scipy>=1.10` — Optimisation, special functions

No image I/O, plotting, or CLI libraries.

### Optional Packages

Additional functionality requires optional packages:

```bash
pip install qmri          # Core only
pip install qmri-io       # + nibabel
pip install qmri-cli      # + click, rich
pip install qmri-viz      # + matplotlib
```

### Adding Dependencies

Before adding a dependency to `qmri` core, consider:

1. Is it essential for the physics/maths?
2. Can it be optional (in a separate package)?
3. Is it well-maintained and stable?
4. Does it support all target Python versions?

## Testing Philosophy

### Test Without I/O

Core functions are tested with synthetic numpy arrays, not files:

```python
def test_fit_adc():
    # Synthetic data, no file I/O
    signal = np.array([1000, 606, 368, 135])
    b_values = np.array([0, 500, 1000, 2000])

    result = adc.fit(signal, b_values)

    assert result.adc == pytest.approx(1e-3, rel=0.01)
```

### Property-Based Testing

Use Hypothesis to find edge cases:

```python
@given(
    true_adc=st.floats(0.1e-3, 3.0e-3),
    snr=st.floats(10, 100),
)
def test_fit_accuracy_vs_snr(true_adc, snr):
    """Higher SNR gives more accurate fits."""
    ...
```

### Integration Tests

File I/O tests go in `test_io/` and use fixtures:

```python
def test_nifti_roundtrip(tmp_path):
    # Integration test with actual files
    ...
```

## Versioning

Each package is versioned independently using semantic versioning:

- **Major**: Breaking API changes
- **Minor**: New features, backwards compatible
- **Patch**: Bug fixes

Packages declare compatible version ranges for dependencies:

```toml
[project]
dependencies = [
    "qmri>=1.0.0,<2.0.0",
]
```
