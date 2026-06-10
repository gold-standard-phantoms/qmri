# qmri-pipelines

End-to-end quantitative MRI processing pipelines built on
[`qmri`](https://github.com/gold-standard-phantoms/qmri).

Where `qmri` provides pure signal models and `qmri-io` provides file handling,
`qmri-pipelines` ties them together into ready-to-run, file-in / file-out
workflows that load images, run the fit, and write maps and reports.

## Installation

```bash
pip install qmri-pipelines
```

## Available pipelines

### Multi-echo thermometry

Estimate temperature from multi-echo magnitude images and a segmentation using
the dual-resonance (ethylene glycol) model.

```python
from qmri.pipelines.thermometry import run_multiecho_thermometry

temperature_map, report = run_multiecho_thermometry(
    multiecho_files=["echo_block_1.nii.gz", "echo_block_2.nii.gz"],
    segmentation_file="labels.nii.gz",
    echo_times_files=["te_block_1.txt", "te_block_2.txt"],
    method="regionwise",
    output_dir="results/",
)

for region in report.regions:
    print(f"region {region.region_id}: {region.temperature:.2f} °C")
```

The magnetic field strength is read from a JSON sidecar
(`ImagingFrequency` or `MagneticFieldStrength`) unless `magnetic_field_tesla`
is supplied explicitly.

The same pipeline is exposed on the command line via
[`qmri-cli`](https://github.com/gold-standard-phantoms/qmri):

```bash
qmri thermometry multiecho echo_block_1.nii.gz echo_block_2.nii.gz \
    -e te_block_1.txt -e te_block_2.txt \
    -s labels.nii.gz --method regionwise -o results/
```
