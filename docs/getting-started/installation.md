# Installation

This guide covers the various ways to install qmri and its companion packages.

## Prerequisites

qmri requires **Python 3.10 or later**. You can check your Python version with:

```bash
python --version
```

!!! note "Python Version"
    qmri uses modern Python features including type annotations with `|` union
    syntax and `match` statements. Python 3.10+ is required for these features.

## Installation with pip

The core `qmri` package can be installed directly from PyPI:

```bash
pip install qmri
```

This installs only the core package with minimal dependencies (NumPy and SciPy).

## Installation with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. To install qmri with uv:

```bash
uv add qmri
```

Or in an existing project:

```bash
uv pip install qmri
```

!!! tip "Why uv?"
    uv is significantly faster than pip for dependency resolution and
    installation. It's particularly useful for projects with many dependencies
    or when working with virtual environments.

## Optional Dependencies

qmri follows a modular design. The core package is kept minimal, with additional
functionality available through companion packages:

### qmri-io (File I/O)

For loading and saving NIFTI and DICOM files:

=== "pip"

    ```bash
    pip install qmri-io
    ```

=== "uv"

    ```bash
    uv add qmri-io
    ```

This adds [nibabel](https://nipy.org/nibabel/) as a dependency.

### qmri-viz (Visualisation)

For plotting parameter maps and diagnostic figures:

=== "pip"

    ```bash
    pip install qmri-viz
    ```

=== "uv"

    ```bash
    uv add qmri-viz
    ```

This adds [matplotlib](https://matplotlib.org/) as a dependency.

### qmri-cli (Command Line Interface)

For command-line tools:

=== "pip"

    ```bash
    pip install qmri-cli
    ```

=== "uv"

    ```bash
    uv add qmri-cli
    ```

This adds [click](https://click.palletsprojects.com/) and [rich](https://rich.readthedocs.io/) as dependencies.

### qmri-dro (Digital Reference Objects)

For generating synthetic data with known ground truth for validation and testing:

=== "pip"

    ```bash
    pip install qmri-dro
    ```

=== "uv"

    ```bash
    uv add qmri-dro
    ```

This provides tools for generating DWI, T1, and ASL phantoms with configurable parameters and noise models.

### qmri-pipelines (Processing Pipelines)

For end-to-end, file-in / file-out workflows that load images, run a fit, and
write maps and reports (e.g. multi-echo thermometry):

=== "pip"

    ```bash
    pip install qmri-pipelines
    ```

=== "uv"

    ```bash
    uv add qmri-pipelines
    ```

This depends on `qmri` and `qmri-io`.

### Installing All Packages

To install all qmri packages at once:

=== "pip"

    ```bash
    pip install qmri qmri-io qmri-pipelines qmri-viz qmri-cli qmri-dro
    ```

=== "uv"

    ```bash
    uv add qmri qmri-io qmri-pipelines qmri-viz qmri-cli qmri-dro
    ```

## Installing from Git

To install the latest development version directly from the Git repository, you need
to specify the subdirectory for each package since qmri uses a monorepo structure.

### Installing Individual Packages

=== "pip"

    ```bash
    # Core qmri package
    pip install "qmri @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri"

    # Additional packages
    pip install "qmri-io @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-io"
    pip install "qmri-pipelines @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-pipelines"
    pip install "qmri-viz @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-viz"
    pip install "qmri-cli @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-cli"
    pip install "qmri-dro @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-dro"
    ```

=== "uv"

    ```bash
    # Core qmri package
    uv add "qmri @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri"

    # Additional packages
    uv add "qmri-io @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-io"
    uv add "qmri-pipelines @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-pipelines"
    uv add "qmri-viz @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-viz"
    uv add "qmri-cli @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-cli"
    uv add "qmri-dro @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri-dro"
    ```

### Installing a Specific Branch or Tag

You can specify a branch or tag after the repository URL:

```bash
# Install from the main branch
pip install "qmri @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git@main#subdirectory=packages/qmri"

# Install from a specific tag
pip install "qmri @ git+ssh://git@github.com/gold-standard-phantoms/qmri.git@v1.0.0#subdirectory=packages/qmri"
```

### Using HTTPS Instead of SSH

If you don't have SSH keys configured, use HTTPS:

```bash
pip install "qmri @ git+https://github.com/gold-standard-phantoms/qmri.git#subdirectory=packages/qmri"
```

!!! tip "Installing All Packages from Git"
    To install all packages from Git at once, consider using the
    [Development Installation](#development-installation) approach instead,
    which clones the repository and installs all packages in editable mode.

## Development Installation

For contributing to qmri or working with the latest development version, clone
the repository and install in development mode.

### Clone the Repository

```bash
git clone https://github.com/gold-standard-phantoms/qmri.git
cd qmri
```

### Install with uv (Recommended)

```bash
# Install all dependencies including development tools
uv sync
```

This installs all workspace packages in editable mode along with development
dependencies (pytest, mypy, ruff, mkdocs, etc.).

### Install with pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e packages/qmri
pip install -e packages/qmri-io
pip install -e packages/qmri-pipelines
pip install -e packages/qmri-viz
pip install -e packages/qmri-cli
pip install -e packages/qmri-dro

# Install development dependencies
pip install pytest pytest-cov mypy ruff mkdocs mkdocs-material mkdocstrings[python]
```

### Verify Installation

After installation, verify everything is working:

```bash
# Run the test suite
uv run pytest

# Or with pip installation
pytest
```

### Development Commands

Common development commands:

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Type checking
uv run mypy packages/qmri/src

# Linting
uv run ruff check .

# Format code
uv run ruff format .

# Build documentation locally
uv run mkdocs serve
```

## Verifying Your Installation

After installation, you can verify qmri is working correctly:

```python
import numpy as np
from qmri.diffusion import adc

# Generate test data
b_values = np.array([0, 500, 1000, 2000])
signal = np.array([1000, 606, 368, 135])

# Fit ADC
result = adc.fit(signal, b_values, method="iwlls")

print(f"ADC: {result.adc:.2e} mm²/s")
print(f"S₀: {result.s0:.0f}")
print(f"R²: {result.r_squared:.4f}")
```

Expected output:

```
ADC: 1.00e-03 mm²/s
S₀: 1000
R²: 1.0000
```

## Troubleshooting

### ImportError: No module named 'qmri'

Ensure qmri is installed in your active Python environment:

```bash
pip list | grep qmri
```

If not listed, reinstall with `pip install qmri`.

### ImportError: No module named 'nibabel'

You're trying to use I/O functions without qmri-io installed:

```bash
pip install qmri-io
```

### ImportError: No module named 'matplotlib'

You're trying to use visualisation functions without qmri-viz installed:

```bash
pip install qmri-viz
```

### Version Conflicts

If you encounter dependency conflicts, try creating a fresh virtual environment:

```bash
python -m venv fresh-env
source fresh-env/bin/activate
pip install qmri
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to learn how
to use qmri for quantitative MRI analysis.
