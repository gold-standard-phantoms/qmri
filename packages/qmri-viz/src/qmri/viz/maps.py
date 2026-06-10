"""Parameter map visualisation utilities.

This module provides functions for visualising quantitative MRI parameter
maps, including single slices, multi-slice grids, and side-by-side
comparisons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_parameter_map(
    data: NDArray[np.floating],
    slice_idx: int | None = None,
    axis: int = 2,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar_label: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a 2D or 3D parameter map.

    For 3D data, extracts a single slice along the specified axis.
    For 2D data, displays the image directly.

    Args:
        data: 2D or 3D array containing the parameter map values.
        slice_idx: Index of the slice to display for 3D data. If None, uses
            the middle slice. Ignored for 2D data.
        axis: Axis along which to take the slice (0, 1, or 2). Default is 2
            (axial slices for standard orientation).
        cmap: Matplotlib colourmap name. Default is "viridis".
        vmin: Minimum value for colour scaling. If None, uses data minimum.
        vmax: Maximum value for colour scaling. If None, uses data maximum.
        title: Title for the plot.
        colorbar_label: Label for the colourbar.
        ax: Existing axes to plot on. If None, creates a new figure.

    Returns:
        The matplotlib figure containing the plot.

    Raises:
        ValueError: If data is not 2D or 3D, or if axis is out of range.

    Example:
        ```python
        import numpy as np
        from qmri.viz.maps import plot_parameter_map
        adc_map = np.random.rand(64, 64, 32) * 3e-3
        fig = plot_parameter_map(
            adc_map,
            slice_idx=16,
            cmap="hot",
            title="ADC Map",
            colorbar_label="ADC (mm²/s)",
        )
        ```
    """
    if data.ndim not in (2, 3):
        msg = f"Data must be 2D or 3D, got {data.ndim}D"
        raise ValueError(msg)

    if data.ndim == 3:
        if axis not in (0, 1, 2):
            msg = f"Axis must be 0, 1, or 2, got {axis}"
            raise ValueError(msg)

        idx = slice_idx if slice_idx is not None else data.shape[axis] // 2

        # Extract the slice
        slice_data = np.take(data, idx, axis=axis)
    else:
        slice_data = data

    # Create figure if no axes provided
    created_fig: Figure
    created_ax: Axes
    if ax is None:
        created_fig, created_ax = plt.subplots(figsize=(8, 6))
        fig = created_fig
        ax = created_ax
    else:
        parent = ax.get_figure()
        if parent is None:
            msg = "Axes has no associated figure"
            raise ValueError(msg)
        fig = cast("Figure", parent)

    # Plot the image
    im = ax.imshow(
        slice_data.T if data.ndim == 3 else slice_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
    )

    # Add colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    # Add title
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig


def plot_multi_slice(
    data: NDArray[np.floating],
    n_slices: int = 9,
    axis: int = 2,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar_label: str | None = None,
) -> Figure:
    """Plot multiple slices of a 3D parameter map in a grid.

    Automatically selects evenly spaced slices through the volume.

    Args:
        data: 3D array containing the parameter map values.
        n_slices: Number of slices to display. Will be arranged in a
            square-ish grid. Default is 9.
        axis: Axis along which to take slices (0, 1, or 2). Default is 2.
        cmap: Matplotlib colourmap name. Default is "viridis".
        vmin: Minimum value for colour scaling. If None, uses data minimum.
        vmax: Maximum value for colour scaling. If None, uses data maximum.
        title: Overall title for the figure.
        colorbar_label: Label for the shared colourbar.

    Returns:
        The matplotlib figure containing the multi-slice plot.

    Raises:
        ValueError: If data is not 3D or if axis is out of range.

    Example:
        ```python
        import numpy as np
        from qmri.viz.maps import plot_multi_slice
        t1_map = np.random.rand(64, 64, 32) * 4000
        fig = plot_multi_slice(
            t1_map,
            n_slices=12,
            cmap="hot",
            title="T1 Map - Multiple Slices",
        )
        ```
    """
    if data.ndim != 3:
        msg = f"Data must be 3D, got {data.ndim}D"
        raise ValueError(msg)

    if axis not in (0, 1, 2):
        msg = f"Axis must be 0, 1, or 2, got {axis}"
        raise ValueError(msg)

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_slices)))
    n_rows = int(np.ceil(n_slices / n_cols))

    # Calculate slice indices
    n_available = data.shape[axis]
    # Avoid edge slices which are often empty
    start_idx = int(n_available * 0.1)
    end_idx = int(n_available * 0.9)
    slice_indices = np.linspace(start_idx, end_idx, n_slices, dtype=int)

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
    )

    # Determine colour limits
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    # Plot each slice
    im = None
    for ax_item, idx in zip(axes.flat, slice_indices, strict=False):
        slice_data = np.take(data, idx, axis=axis)
        im = ax_item.imshow(
            slice_data.T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        ax_item.set_title(f"Slice {idx}")
        ax_item.set_xticks([])
        ax_item.set_yticks([])

    # Hide unused axes
    for ax_item in axes.flat[len(slice_indices) :]:
        ax_item.set_visible(False)

    # Add shared colourbar
    if im is None:
        msg = "No slices to display"
        raise ValueError(msg)
    cbar = fig.colorbar(im, ax=list(axes.flat), fraction=0.02, pad=0.02)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    # Add title
    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    return fig


def compare_maps(
    maps: list[NDArray[np.floating]],
    titles: list[str],
    slice_idx: int | None = None,
    axis: int = 2,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    suptitle: str | None = None,
    colorbar_label: str | None = None,
    shared_colorbar: bool = True,
) -> Figure:
    """Create a side-by-side comparison of multiple parameter maps.

    Args:
        maps: List of 2D or 3D arrays to compare. All arrays must have the
            same shape.
        titles: List of titles for each map. Must have same length as maps.
        slice_idx: Index of the slice to display for 3D data. If None, uses
            the middle slice.
        axis: Axis along which to take slices for 3D data. Default is 2.
        cmap: Matplotlib colourmap name. Default is "viridis".
        vmin: Minimum value for colour scaling. If None and shared_colorbar
            is True, uses minimum across all maps.
        vmax: Maximum value for colour scaling. If None and shared_colorbar
            is True, uses maximum across all maps.
        suptitle: Overall title for the figure.
        colorbar_label: Label for the colourbar(s).
        shared_colorbar: If True (default), use the same colour scaling for
            all maps.

    Returns:
        The matplotlib figure containing the comparison.

    Raises:
        ValueError: If maps and titles have different lengths, or if maps have
            inconsistent shapes.

    Example:
        ```python
        import numpy as np
        from qmri.viz.maps import compare_maps
        map1 = np.random.rand(64, 64) * 3e-3
        map2 = np.random.rand(64, 64) * 3e-3
        fig = compare_maps(
            [map1, map2],
            ["Method A", "Method B"],
            suptitle="ADC Comparison",
        )
        ```
    """
    if len(maps) != len(titles):
        msg = f"Number of maps ({len(maps)}) must match titles ({len(titles)})"
        raise ValueError(msg)

    if len(maps) == 0:
        msg = "At least one map is required"
        raise ValueError(msg)

    # Check shapes are consistent
    reference_shape = maps[0].shape
    for i, m in enumerate(maps[1:], start=1):
        if m.shape != reference_shape:
            msg = f"Map {i} shape {m.shape} differs from map 0 shape {reference_shape}"
            raise ValueError(msg)

    n_maps = len(maps)

    # Extract slices if 3D
    if maps[0].ndim == 3:
        idx = slice_idx if slice_idx is not None else maps[0].shape[axis] // 2
        slices = [np.take(m, idx, axis=axis).T for m in maps]
    else:
        slices = list(maps)

    # Calculate colour limits if shared
    if shared_colorbar:
        if vmin is None:
            vmin = float(min(np.nanmin(s) for s in slices))
        if vmax is None:
            vmax = float(max(np.nanmax(s) for s in slices))

    # Create figure
    fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4), squeeze=False)
    axes_list = list(axes.flat)

    ims = []
    for ax, slice_data, title in zip(axes_list, slices, titles, strict=True):
        if shared_colorbar:
            im = ax.imshow(
                slice_data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="equal",
            )
        else:
            im = ax.imshow(
                slice_data,
                cmap=cmap,
                origin="lower",
                aspect="equal",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ims.append(im)

    # Add shared colourbar
    if shared_colorbar:
        cbar = fig.colorbar(ims[-1], ax=axes_list, fraction=0.02, pad=0.02)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    # Add suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    fig.tight_layout()
    return fig
