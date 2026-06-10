"""Custom colourmaps for MRI parameter visualisation.

This module provides specialised colourmaps designed for visualising
quantitative MRI parameters with appropriate perceptual properties
and value ranges.
"""

from __future__ import annotations

import contextlib

import matplotlib as mpl
import matplotlib.colors as mcolors


def _create_segmented_cmap(
    name: str,
    colours: list[tuple[float, float, float]],
    n_bins: int = 256,
) -> mcolors.LinearSegmentedColormap:
    """Create a linear segmented colourmap from a list of colours.

    Args:
        name: Name for the colourmap.
        colours: List of RGB tuples (0-1 range) defining the colourmap.
        n_bins: Number of discrete bins in the colourmap. Default is 256.

    Returns:
        The created colourmap.
    """
    return mcolors.LinearSegmentedColormap.from_list(name, colours, N=n_bins)


def get_adc_cmap() -> mcolors.Colormap:
    """Get a colourmap suitable for ADC maps.

    Creates a colourmap optimised for visualising Apparent Diffusion
    Coefficient (ADC) values in the typical range of 0 to 3e-3 mm²/s.
    The colourmap uses a blue-to-red gradient where:

    - Blue represents low diffusivity (restricted diffusion)
    - Green/yellow represents typical tissue diffusivity
    - Red represents high diffusivity (free water/CSF)

    Returns:
        A matplotlib colourmap for ADC visualisation.

    Example:
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        from qmri.viz.colormaps import get_adc_cmap
        adc_map = np.random.rand(64, 64) * 3e-3
        plt.imshow(adc_map, cmap=get_adc_cmap(), vmin=0, vmax=3e-3)
        plt.colorbar(label="ADC (mm²/s)")
        ```
    """
    # Blue -> Cyan -> Green -> Yellow -> Red
    colours: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.5),  # Dark blue (restricted)
        (0.0, 0.5, 1.0),  # Cyan
        (0.0, 0.8, 0.0),  # Green (typical tissue)
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 0.0),  # Red (free water)
    ]
    return _create_segmented_cmap("adc", colours)


def get_t1_cmap() -> mcolors.Colormap:
    """Get a colourmap suitable for T1 maps.

    Creates a colourmap optimised for visualising T1 relaxation times
    in the typical range of 0 to 4000 ms. The colourmap uses a
    warm colour gradient where:

    - Dark purple represents short T1 (e.g., fat)
    - Orange/yellow represents typical tissue T1
    - Bright white/yellow represents long T1 (e.g., CSF)

    Returns:
        A matplotlib colourmap for T1 visualisation.

    Example:
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        from qmri.viz.colormaps import get_t1_cmap
        t1_map = np.random.rand(64, 64) * 4000
        plt.imshow(t1_map, cmap=get_t1_cmap(), vmin=0, vmax=4000)
        plt.colorbar(label="T1 (ms)")
        ```
    """
    # Dark purple -> Magenta -> Orange -> Yellow -> White
    colours: list[tuple[float, float, float]] = [
        (0.1, 0.0, 0.2),  # Dark purple (short T1)
        (0.5, 0.0, 0.5),  # Magenta
        (0.8, 0.3, 0.0),  # Orange
        (1.0, 0.8, 0.0),  # Yellow
        (1.0, 1.0, 0.9),  # Light yellow/white (long T1)
    ]
    return _create_segmented_cmap("t1", colours)


def get_t2_cmap() -> mcolors.Colormap:
    """Get a colourmap suitable for T2 maps.

    Creates a colourmap optimised for visualising T2 relaxation times
    in the typical range of 0 to 200 ms. The colourmap uses a
    cool colour gradient where:

    - Dark blue represents short T2 (e.g., cortical bone)
    - Cyan/teal represents typical tissue T2
    - Bright green/yellow represents long T2 (e.g., fluids)

    Returns:
        A matplotlib colourmap for T2 visualisation.

    Example:
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        from qmri.viz.colormaps import get_t2_cmap
        t2_map = np.random.rand(64, 64) * 200
        plt.imshow(t2_map, cmap=get_t2_cmap(), vmin=0, vmax=200)
        plt.colorbar(label="T2 (ms)")
        ```
    """
    # Dark blue -> Teal -> Cyan -> Green -> Yellow
    colours: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.3),  # Dark blue (short T2)
        (0.0, 0.4, 0.5),  # Teal
        (0.0, 0.7, 0.7),  # Cyan
        (0.2, 0.9, 0.2),  # Green
        (0.9, 1.0, 0.0),  # Yellow (long T2)
    ]
    return _create_segmented_cmap("t2", colours)


def register_qmri_cmaps() -> None:
    """Register all qMRI colourmaps with matplotlib.

    After calling this function, the colourmaps can be accessed by name:
    'qmri_adc', 'qmri_t1', 'qmri_t2'.

    Example:
        ```python
        from qmri.viz.colormaps import register_qmri_cmaps
        register_qmri_cmaps()
        plt.imshow(data, cmap='qmri_adc')
        ```
    """
    cmaps = {
        "qmri_adc": get_adc_cmap(),
        "qmri_t1": get_t1_cmap(),
        "qmri_t2": get_t2_cmap(),
    }
    for name, cmap in cmaps.items():
        with contextlib.suppress(ValueError):
            mpl.colormaps.register(cmap, name=name)
