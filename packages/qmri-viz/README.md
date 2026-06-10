# qmri-viz

Visualisation utilities for qmri quantitative MRI analysis.

## Installation

```bash
pip install qmri-viz
```

## Quick Start

```python
from qmri.diffusion import adc
from qmri.viz import fitting, maps

# Fit data
result = adc.fit(signal, b_values)

# Plot fit diagnostics
fitting.plot_fit(signal, b_values, result)

# Display parameter map
maps.show_slice(result.adc, title="ADC Map", cmap="viridis")
```

## Features

- **Fit diagnostics** — Visualise model fits with residuals
- **Parameter maps** — Display quantitative maps with appropriate colormaps
- **Calibration plots** — Phantom calibration visualisation
- **Multi-slice views** — Browse through 3D volumes

## Design

This package provides matplotlib-based visualisation utilities. It's kept separate from the core `qmri` package so users who don't need plotting avoid the matplotlib dependency.

## Documentation

Full documentation: https://gold-standard-phantoms.github.io/qmri

## Licence

MIT
