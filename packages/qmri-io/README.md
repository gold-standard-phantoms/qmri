# qmri-io

NIFTI, DICOM, and BIDS I/O for qmri.

## Installation

```bash
pip install qmri-io
```

## Quick Start

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

## Features

- **NIFTI support** via nibabel
- **BIDS dataset loading** with automatic sidecar parsing
- **DICOM support** (optional, requires pydicom)
- **Sidecar file handling** (.bval, .bvec, .json)

## Design

This package handles file I/O so the core `qmri` package doesn't need nibabel or pydicom as dependencies. Functions return simple dataclasses with numpy arrays, not complex container objects.

## Documentation

Full documentation: https://gold-standard-phantoms.github.io/qmri

## Licence

MIT
