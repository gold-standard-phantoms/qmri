"""Fitting diagnostics visualisation.

This module provides functions for visualising model fitting diagnostics,
including residual plots, R² distributions, and signal decay curves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_fit_residuals(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    x_values: NDArray[np.floating] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot residuals from model fitting.

    Creates a residual plot showing the difference between observed
    and predicted values. Useful for assessing fit quality and
    identifying systematic deviations.

    Args:
        observed: Observed (measured) values.
        predicted: Predicted (fitted) values. Must have same shape as observed.
        x_values: X-axis values (e.g., b-values, TE). If None, uses indices.
        ax: Existing axes to plot on. If None, creates a new figure.

    Returns:
        The matplotlib figure containing the residual plot.

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        ```python
        import numpy as np
        from qmri.viz.diagnostics import plot_fit_residuals
        observed = np.array([1000, 600, 360, 220])
        predicted = np.array([1000, 607, 368, 223])
        b_values = np.array([0, 500, 1000, 1500])
        fig = plot_fit_residuals(observed, predicted, b_values)
        ```
    """
    if observed.shape != predicted.shape:
        msg = (
            f"Observed shape {observed.shape} must match "
            f"predicted shape {predicted.shape}"
        )
        raise ValueError(msg)

    residuals = observed - predicted

    if x_values is None:
        x_values = np.arange(len(observed.flat), dtype=np.float64)

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Flatten for 1D plot if necessary
    x_flat = x_values.flatten()
    residuals_flat = residuals.flatten()

    # Plot residuals
    ax.scatter(x_flat, residuals_flat, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Zero line")

    # Add reference lines at +/- std
    std_residuals = float(np.std(residuals_flat))
    ax.axhline(
        y=std_residuals,
        color="orange",
        linestyle=":",
        linewidth=1,
        label=f"+1 SD ({std_residuals:.2f})",
    )
    ax.axhline(
        y=-std_residuals,
        color="orange",
        linestyle=":",
        linewidth=1,
        label=f"-1 SD ({-std_residuals:.2f})",
    )

    ax.set_xlabel("X values" if x_values is not None else "Index")
    ax.set_ylabel("Residual (Observed - Predicted)")
    ax.set_title("Residual Plot")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_r_squared_histogram(
    r_squared: NDArray[np.floating],
    threshold: float | None = None,
    bins: int = 50,
    ax: Axes | None = None,
) -> Figure:
    """Plot histogram of R² values from fitting.

    Visualises the distribution of goodness-of-fit values across
    a volume or set of fits.

    Args:
        r_squared: Array of R² values (can be any shape, will be flattened).
        threshold: If provided, draws a vertical line indicating a quality
            threshold and reports the percentage of fits above it.
        bins: Number of histogram bins. Default is 50.
        ax: Existing axes to plot on. If None, creates a new figure.

    Returns:
        The matplotlib figure containing the histogram.

    Example:
        ```python
        import numpy as np
        from qmri.viz.diagnostics import plot_r_squared_histogram
        r_squared = np.random.beta(10, 2, size=1000)  # Skewed towards 1
        fig = plot_r_squared_histogram(r_squared, threshold=0.9)
        ```
    """
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure  # type: ignore[assignment]

    r_squared_flat = r_squared.flatten()

    # Filter out NaN and invalid values
    valid_mask = np.isfinite(r_squared_flat) & (r_squared_flat >= 0)
    valid_r2 = r_squared_flat[valid_mask]

    if len(valid_r2) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid R² values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        valid_r2,
        bins=bins,
        range=(0, 1),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )

    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.2f})",
        )

        # Calculate percentage above threshold
        pct_above = 100 * np.sum(valid_r2 >= threshold) / len(valid_r2)
        ax.annotate(
            f"{pct_above:.1f}% above threshold",
            xy=(threshold, ax.get_ylim()[1] * 0.9),
            xytext=(threshold + 0.05, ax.get_ylim()[1] * 0.95),
            fontsize=10,
            ha="left",
        )
        ax.legend(loc="upper left")

    # Add statistics
    mean_r2 = np.mean(valid_r2)
    median_r2 = np.median(valid_r2)
    stats_text = f"Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}\nN: {len(valid_r2)}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Distribution")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_signal_decay(
    signal: NDArray[np.floating],
    x_values: NDArray[np.floating],
    fit_params: dict[str, float] | None = None,
    model_func: Callable[..., NDArray[np.floating]] | None = None,
    xlabel: str = "X",
    ylabel: str = "Signal",
    title: str = "Signal Decay",
    ax: Axes | None = None,
) -> Figure:
    """Plot signal decay curve with optional fitted model.

    Visualises measured signal values and optionally overlays a
    fitted model curve.

    Args:
        signal: Measured signal values.
        x_values: X-axis values (e.g., b-values for diffusion, TE for T2).
        fit_params: Dictionary of fitted parameters to pass to model_func.
            Required if model_func is provided.
        model_func: Model function that takes x_values and **fit_params and
            returns predicted signal. If None, only plots measured data.
        xlabel: Label for x-axis. Default is "X".
        ylabel: Label for y-axis. Default is "Signal".
        title: Plot title. Default is "Signal Decay".
        ax: Existing axes to plot on. If None, creates a new figure.

    Returns:
        The matplotlib figure containing the decay plot.

    Example:
        ```python
        import numpy as np
        from qmri.viz.diagnostics import plot_signal_decay

        def mono_exp(x: np.ndarray, s0: float, adc: float) -> np.ndarray:
            return s0 * np.exp(-x * adc)

        b_values = np.array([0, 500, 1000, 1500, 2000])
        signal = mono_exp(b_values, s0=1000, adc=1e-3)
        signal = signal + np.random.randn(len(signal)) * 20  # Add noise
        fig = plot_signal_decay(
            signal,
            b_values,
            fit_params={"s0": 1000, "adc": 1e-3},
            model_func=mono_exp,
            xlabel="b-value (s/mm²)",
            ylabel="Signal (a.u.)",
        )
        ```
    """
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Plot measured data
    ax.scatter(
        x_values,
        signal,
        s=80,
        color="blue",
        marker="o",
        edgecolors="black",
        linewidth=1,
        label="Measured",
        zorder=3,
    )

    # Plot fitted curve if model provided
    if model_func is not None and fit_params is not None:
        # Create smooth x values for plotting
        x_smooth = np.linspace(
            float(np.min(x_values)),
            float(np.max(x_values)),
            200,
        )
        try:
            y_fitted = model_func(x_smooth, **fit_params)
            ax.plot(
                x_smooth,
                y_fitted,
                color="red",
                linewidth=2,
                label="Fitted model",
                zorder=2,
            )

            # Add parameter text
            param_text = "\n".join(f"{k}: {v:.4g}" for k, v in fit_params.items())
            ax.text(
                0.98,
                0.98,
                param_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
        except (TypeError, ValueError) as e:
            # If model fails, just show data
            ax.text(
                0.98,
                0.98,
                f"Model error: {e}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                color="red",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Ensure y-axis starts from 0 or slightly below minimum
    y_min = min(0, float(np.min(signal)) * 1.1)
    y_max = float(np.max(signal)) * 1.1
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    return fig
