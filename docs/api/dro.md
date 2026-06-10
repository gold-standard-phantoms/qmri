# qmri.dro

Digital Reference Objects for quantitative MRI validation.

## Module Overview

The `qmri.dro` module provides tools for generating synthetic MRI data with known ground truth parameters.

## Types

::: qmri.dro.GroundTruth
    options:
      show_root_heading: true
      heading_level: 3

::: qmri.dro.DWIPhantom
    options:
      show_root_heading: true
      heading_level: 3

::: qmri.dro.T1Phantom
    options:
      show_root_heading: true
      heading_level: 3

::: qmri.dro.ASLPhantom
    options:
      show_root_heading: true
      heading_level: 3

## DWI Phantom Generation

::: qmri.dro.dwi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - generate
        - generate_calibration_phantom

## Relaxometry Phantom Generation

::: qmri.dro.relaxometry
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - generate_t1_ir
        - generate_t1_vtr

## Perfusion Phantom Generation

::: qmri.dro.perfusion
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - generate_pcasl
