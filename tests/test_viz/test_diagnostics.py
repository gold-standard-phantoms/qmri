"""Tests for qmri.viz.diagnostics module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

# Use non-interactive backend for testing
mpl.use("Agg")

from qmri.viz.diagnostics import (
    plot_fit_residuals,
    plot_r_squared_histogram,
    plot_signal_decay,
)


@pytest.fixture
def sample_observed() -> NDArray[np.floating]:
    """Create sample observed data."""
    return np.array([1000.0, 600.0, 360.0, 216.0], dtype=np.float64)


@pytest.fixture
def sample_predicted() -> NDArray[np.floating]:
    """Create sample predicted data (slightly different from observed)."""
    return np.array([1000.0, 607.0, 368.0, 223.0], dtype=np.float64)


@pytest.fixture
def sample_x_values() -> NDArray[np.floating]:
    """Create sample x values (b-values)."""
    return np.array([0.0, 500.0, 1000.0, 1500.0], dtype=np.float64)


@pytest.fixture
def sample_r_squared() -> NDArray[np.floating]:
    """Create sample R² values."""
    rng = np.random.default_rng(42)
    return rng.beta(10, 2, size=1000).astype(np.float64)


class TestPlotFitResiduals:
    """Tests for plot_fit_residuals function."""

    def test_returns_figure(
        self,
        sample_observed: NDArray[np.floating],
        sample_predicted: NDArray[np.floating],
    ) -> None:
        """Function returns a matplotlib Figure."""
        fig = plot_fit_residuals(sample_observed, sample_predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_x_values(
        self,
        sample_observed: NDArray[np.floating],
        sample_predicted: NDArray[np.floating],
        sample_x_values: NDArray[np.floating],
    ) -> None:
        """Function works with custom x values."""
        fig = plot_fit_residuals(sample_observed, sample_predicted, sample_x_values)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_existing_axes(
        self,
        sample_observed: NDArray[np.floating],
        sample_predicted: NDArray[np.floating],
    ) -> None:
        """Function can plot on existing axes."""
        fig, ax = plt.subplots()
        result_fig = plot_fit_residuals(sample_observed, sample_predicted, ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_mismatched_shapes_raises(
        self, sample_observed: NDArray[np.floating]
    ) -> None:
        """Mismatched shapes raise ValueError."""
        predicted = np.array([1000.0, 600.0], dtype=np.float64)
        with pytest.raises(ValueError, match="must match"):
            plot_fit_residuals(sample_observed, predicted)

    def test_2d_data(self) -> None:
        """Function handles 2D data (flattened)."""
        rng = np.random.default_rng(42)
        observed = rng.random((10, 10)).astype(np.float64)
        predicted = observed + rng.standard_normal((10, 10)) * 0.01
        fig = plot_fit_residuals(observed, predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRSquaredHistogram:
    """Tests for plot_r_squared_histogram function."""

    def test_returns_figure(self, sample_r_squared: NDArray[np.floating]) -> None:
        """Function returns a matplotlib Figure."""
        fig = plot_r_squared_histogram(sample_r_squared)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_threshold(self, sample_r_squared: NDArray[np.floating]) -> None:
        """Function works with threshold parameter."""
        fig = plot_r_squared_histogram(sample_r_squared, threshold=0.9)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_bins(self, sample_r_squared: NDArray[np.floating]) -> None:
        """Function works with custom bin count."""
        fig = plot_r_squared_histogram(sample_r_squared, bins=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_existing_axes(self, sample_r_squared: NDArray[np.floating]) -> None:
        """Function can plot on existing axes."""
        fig, ax = plt.subplots()
        result_fig = plot_r_squared_histogram(sample_r_squared, ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_3d_input(self) -> None:
        """Function handles 3D input (flattened)."""
        rng = np.random.default_rng(42)
        r_squared = rng.random((10, 10, 10)).astype(np.float64)
        fig = plot_r_squared_histogram(r_squared)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_nan_values(self) -> None:
        """Function handles NaN values gracefully."""
        r_squared = np.array([0.9, 0.95, np.nan, 0.85, 0.92], dtype=np.float64)
        fig = plot_r_squared_histogram(r_squared)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_after_filtering(self) -> None:
        """Function handles case where all values are invalid."""
        r_squared = np.array([np.nan, np.nan, -1.0], dtype=np.float64)
        fig = plot_r_squared_histogram(r_squared)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSignalDecay:
    """Tests for plot_signal_decay function."""

    def test_returns_figure(
        self,
        sample_observed: NDArray[np.floating],
        sample_x_values: NDArray[np.floating],
    ) -> None:
        """Function returns a matplotlib Figure."""
        fig = plot_signal_decay(sample_observed, sample_x_values)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_fit(
        self,
        sample_observed: NDArray[np.floating],
        sample_x_values: NDArray[np.floating],
    ) -> None:
        """Function works with fitted model."""

        def mono_exp(
            x: NDArray[np.floating], s0: float, adc: float
        ) -> NDArray[np.floating]:
            return s0 * np.exp(-x * adc)

        fig = plot_signal_decay(
            sample_observed,
            sample_x_values,
            fit_params={"s0": 1000.0, "adc": 1e-3},
            model_func=mono_exp,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(
        self,
        sample_observed: NDArray[np.floating],
        sample_x_values: NDArray[np.floating],
    ) -> None:
        """Custom axis labels are applied."""
        fig = plot_signal_decay(
            sample_observed,
            sample_x_values,
            xlabel="b-value (s/mm²)",
            ylabel="Signal (a.u.)",
            title="Diffusion Decay",
        )
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "b-value (s/mm²)"
        assert ax.get_ylabel() == "Signal (a.u.)"
        assert ax.get_title() == "Diffusion Decay"
        plt.close(fig)

    def test_existing_axes(
        self,
        sample_observed: NDArray[np.floating],
        sample_x_values: NDArray[np.floating],
    ) -> None:
        """Function can plot on existing axes."""
        fig, ax = plt.subplots()
        result_fig = plot_signal_decay(sample_observed, sample_x_values, ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_model_error_handled(
        self,
        sample_observed: NDArray[np.floating],
        sample_x_values: NDArray[np.floating],
    ) -> None:
        """Function handles model errors gracefully."""

        def bad_model(
            x: NDArray[np.floating], wrong_param: str
        ) -> NDArray[np.floating]:
            raise TypeError("Expected float")

        # Should not raise, just show data without fit
        fig = plot_signal_decay(
            sample_observed,
            sample_x_values,
            fit_params={"wrong_param": "not_a_number"},
            model_func=bad_model,  # type: ignore[arg-type]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_t2_decay_example(self) -> None:
        """Function works for T2 decay (different parameter names)."""
        te_values = np.array([10.0, 20.0, 40.0, 80.0, 160.0], dtype=np.float64)
        signal = 1000 * np.exp(-te_values / 80)

        def t2_model(
            x: NDArray[np.floating], s0: float, t2: float
        ) -> NDArray[np.floating]:
            return s0 * np.exp(-x / t2)

        fig = plot_signal_decay(
            signal,
            te_values,
            fit_params={"s0": 1000.0, "t2": 80.0},
            model_func=t2_model,
            xlabel="TE (ms)",
            ylabel="Signal",
            title="T2 Decay",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
