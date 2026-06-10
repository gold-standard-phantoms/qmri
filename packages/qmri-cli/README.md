# qmri-cli

Command-line interface for qmri quantitative MRI analysis.

## Installation

```bash
pip install qmri-cli
```

## Quick Start

```bash
# Fit ADC map
qmri adc fit dwi.nii.gz --bval dwi.bval --output adc.nii.gz

# With brain mask and quality threshold
qmri adc fit dwi.nii.gz \
    --bval dwi.bval \
    --mask brain.nii.gz \
    --r2-threshold 0.8 \
    --output adc.nii.gz

# T1 mapping from inversion recovery
qmri t1 fit-ir t1w.nii.gz \
    --inversion-times 0.1,0.5,1.0,2.0 \
    --output t1_map.nii.gz

# Get help
qmri --help
qmri adc --help
```

## Commands

| Command | Description |
|---------|-------------|
| `qmri adc fit` | ADC fitting from DWI |
| `qmri adc calibrate` | B-value calibration |
| `qmri t1 fit-ir` | T1 mapping (inversion recovery) |
| `qmri t1 fit-vfa` | T1 mapping (variable flip angle) |
| `qmri t2 fit` | T2 mapping |
| `qmri thermometry fit` | MR thermometry |

## Design

The CLI wraps the core `qmri` library with a user-friendly interface. It uses [Click](https://click.palletsprojects.com/) for argument parsing and [Rich](https://rich.readthedocs.io/) for formatted output.

## Documentation

Full documentation: https://gold-standard-phantoms.github.io/qmri/cli/

## Licence

MIT
