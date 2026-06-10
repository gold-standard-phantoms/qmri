# Changelog

All notable changes to qmri will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

Initial release of qmri, providing pure MRI signal models and fitting algorithms for quantitative MRI analysis.

### Added

#### Core (`qmri`)

- **Diffusion imaging**
    - ADC (Apparent Diffusion Coefficient) fitting with multiple methods:
        - LLS (Linear Least Squares)
        - WLLS (Weighted Linear Least Squares)
        - IWLLS (Iterative Weighted Linear Least Squares)
    - DWI signal generation from ADC values
    - Support for single voxel and volumetric data

- **Relaxometry**
    - T1 mapping with multiple acquisition methods:
        - IR (Inversion Recovery with finite TR correction)
        - IR Classical (Inversion Recovery assuming TR >> T1)
        - VTR (Variable TR for spoiled gradient echo)
    - T2 mapping from multi-echo spin echo data:
        - Full model (with offset term)
        - Reduced model (pure mono-exponential)
    - Signal generation functions for both T1 and T2

- **Perfusion**
    - ASL (Arterial Spin Labelling) quantification
    - GKM (General Kinetic Model) for DCE-MRI

- **Thermometry**
    - PRF (Proton Resonance Frequency) shift method for MR thermometry

- **Fitting utilities**
    - Least squares algorithms
    - Bootstrap uncertainty estimation

- **Error metrics**
    - R-squared (coefficient of determination)
    - RMSE (Root Mean Square Error)
    - Uncertainty propagation

- **Constants**
    - Physical constants for MRI calculations

#### I/O (`qmri-io`)

- NIfTI file reading and writing via nibabel
- Affine and header preservation
- BIDS-compatible file handling utilities

#### CLI (`qmri-cli`)

- `qmri diffusion adc` command for ADC fitting
- `qmri relaxometry t1` command for T1 mapping
- `qmri relaxometry t2` command for T2 mapping
- `qmri info` command for version information
- Global options for verbosity and output directory
- Rich console output with progress indicators

#### Visualisation (`qmri-viz`)

- Parameter map plotting utilities
- Diagnostic visualisation tools
- Custom colourmaps for quantitative MRI

### Dependencies

- Core package (`qmri`): NumPy, SciPy only
- I/O package (`qmri-io`): + nibabel
- CLI package (`qmri-cli`): + Click, Rich
- Visualisation package (`qmri-viz`): + Matplotlib

### Notes

- Python 3.10+ required
- Full type annotations throughout
- NumPy-style docstrings
- UK English spelling used throughout
