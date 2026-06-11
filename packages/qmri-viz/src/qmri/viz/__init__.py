"""Visualisation utilities for qmri.

This package provides visualisation tools for quantitative MRI analysis,
including parameter map plotting, fitting diagnostics, and custom
colourmaps optimised for MRI parameters.

Submodules:
    maps: Parameter map visualisation (single slice, multi-slice, comparisons).
    diagnostics: Fitting diagnostics (residuals, R² histograms, signal decay curves).
    colormaps: Custom colourmaps for ADC, T1, and T2 maps.

Example:
    Plot a parameter map:

    ```python
    import numpy as np
    from qmri.viz import plot_parameter_map
    adc_map = np.random.rand(64, 64, 32) * 3e-3
    fig = plot_parameter_map(adc_map, slice_idx=16, title="ADC Map")
    ```

    Use a custom colourmap:

    ```python
    from qmri.viz import get_adc_cmap, plot_parameter_map
    fig = plot_parameter_map(
        adc_map,
        cmap=get_adc_cmap(),
        vmin=0,
        vmax=3e-3,
    )
    ```

    Plot fitting diagnostics:

    ```python
    from qmri.viz import plot_r_squared_histogram
    r_squared = np.random.beta(10, 2, size=1000)
    fig = plot_r_squared_histogram(r_squared, threshold=0.9)
    ```
"""

from qmri.viz.colormaps import (
    get_adc_cmap,
    get_t1_cmap,
    get_t2_cmap,
    register_qmri_cmaps,
)
from qmri.viz.diagnostics import (
    plot_fit_residuals,
    plot_r_squared_histogram,
    plot_signal_decay,
)
from qmri.viz.maps import (
    compare_maps,
    plot_multi_slice,
    plot_parameter_map,
)

__all__: list[str] = [
    # maps
    "plot_parameter_map",
    "plot_multi_slice",
    "compare_maps",
    # diagnostics
    "plot_fit_residuals",
    "plot_r_squared_histogram",
    "plot_signal_decay",
    # colormaps
    "get_adc_cmap",
    "get_t1_cmap",
    "get_t2_cmap",
    "register_qmri_cmaps",
]
