# Code Style Guide

qmri uses strict linting and type checking to maintain code quality and consistency.

## Language

**Use UK English** throughout the codebase:

| Use | Not |
|-----|-----|
| colour | color |
| centre | center |
| behaviour | behavior |
| initialise | initialize |
| optimise | optimize |
| analyse | analyze |
| modelling | modeling |
| labelling | labeling |
| organisation | organization |
| licence (noun) | license |

This applies to:

- Variable and function names
- Docstrings and comments
- Documentation
- Commit messages

## Linting and Formatting

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Check formatting without changing files
ruff format --check .
```

### Enabled Rules

| Code | Category |
|------|----------|
| `E` | pycodestyle errors |
| `F` | pyflakes |
| `I` | isort (import sorting) |
| `UP` | pyupgrade |
| `B` | flake8-bugbear |
| `SIM` | flake8-simplify |
| `TCH` | flake8-type-checking |
| `D` | pydocstyle (Google convention) |
| `NPY` | NumPy-specific rules |

## Type Hints

We use mypy in strict mode:

```bash
mypy packages/qmri/src
```

### Requirements

All public functions must have complete type annotations:

```python
from typing import Literal
import numpy as np
import numpy.typing as npt

def fit(
    signal: npt.NDArray[np.floating],
    b_values: npt.NDArray[np.floating],
    *,
    method: Literal["lls", "wlls", "iwlls"] = "iwlls",
    mask: npt.NDArray[np.bool_] | None = None,
) -> ADCResult:
    ...
```

### Common Patterns

```python
# Optional parameters
def process(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    ...

# Literal types for string options
def fit(method: Literal["lls", "wlls"] = "lls") -> Result:
    ...

# Numpy array types
from numpy.typing import NDArray
def transform(data: NDArray[np.floating]) -> NDArray[np.floating]:
    ...

# Dataclass results
from dataclasses import dataclass

@dataclass(frozen=True)
class ADCResult:
    adc: np.ndarray
    s0: np.ndarray
    r_squared: np.ndarray
```

## Docstrings

Use Google-style docstrings:

```python
def fit(
    signal: npt.NDArray[np.floating],
    b_values: npt.NDArray[np.floating],
    method: Literal["lls", "wlls", "iwlls"] = "iwlls",
) -> ADCResult:
    """Fit the ADC model to diffusion-weighted signal data.

    Implements the Stejskal-Tanner equation:

    $$S(b) = S_0 \\exp(-b \\cdot \\text{ADC})$$

    Args:
        signal: Signal intensity values. Shape ``(n_bvalues,)`` for single
            voxel or ``(..., n_bvalues)`` for volumetric data.
        b_values: B-values in s/mm². Shape ``(n_bvalues,)``.
        method: Fitting method. Options are:

            - ``"lls"``: Linear least squares (fastest, biased at low SNR)
            - ``"wlls"``: Weighted least squares (single iteration)
            - ``"iwlls"``: Iterative weighted least squares (recommended)

    Returns:
        Result containing fitted ADC, S0, and R² values.

    Raises:
        ValueError: If signal and b_values have incompatible shapes.

    Example:
        >>> signal = np.array([1000, 606, 368, 135])
        >>> b_values = np.array([0, 500, 1000, 2000])
        >>> result = fit(signal, b_values)
        >>> print(f"ADC: {result.adc:.2e} mm²/s")
        ADC: 1.00e-03 mm²/s

    References:
        Stejskal, E.O. and Tanner, J.E. (1965). Spin diffusion measurements.
        *J. Chem. Phys.*, 42(1), 288-292.
    """
```

### Docstring Checklist

- [ ] One-line summary (imperative mood)
- [ ] Extended description if needed
- [ ] Mathematical equations (LaTeX) for physics functions
- [ ] All parameters documented with types
- [ ] Return value documented
- [ ] Exceptions documented
- [ ] Working example
- [ ] References for academic sources

## Import Order

Imports are sorted by Ruff (isort rules):

```python
# Standard library
from dataclasses import dataclass
from typing import Literal

# Third-party
import numpy as np
import numpy.typing as npt
from scipy import optimize

# Local
from qmri.fitting import least_squares
```

## Function Design

### Pure Functions

Prefer pure functions over stateful classes:

```python
# Good: Pure function
def fit_adc(signal: np.ndarray, b_values: np.ndarray) -> ADCResult:
    ...

# Avoid: Stateful class
class ADCFitter:
    def __init__(self, method: str):
        self.method = method

    def fit(self, signal, b_values):
        ...
```

### Keyword-Only Arguments

Use keyword-only arguments (after `*`) for optional parameters:

```python
def fit(
    signal: np.ndarray,
    b_values: np.ndarray,
    *,  # Everything after this is keyword-only
    method: str = "iwlls",
    mask: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> ADCResult:
    ...
```

### Result Dataclasses

Return frozen dataclasses for complex results:

```python
@dataclass(frozen=True)
class ADCResult:
    """Result of ADC fitting.

    Attributes:
        adc: Apparent diffusion coefficient in mm²/s.
        s0: Signal at b=0.
        r_squared: Coefficient of determination (0-1).
        residuals: Fit residuals (optional).
    """

    adc: np.ndarray
    s0: np.ndarray
    r_squared: np.ndarray
    residuals: np.ndarray | None = None
```
