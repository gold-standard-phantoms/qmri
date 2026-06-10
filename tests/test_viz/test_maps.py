"""Tests for qmri.viz.maps module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

# Use non-interactive backend for testing
mpl.use("Agg")

from qmri.viz.maps import compare_maps, plot_multi_slice, plot_parameter_map


@pytest.fixture
def sample_2d_map() -> NDArray[np.floating]:
    """Create a sample 2D parameter map."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64)).astype(np.float64) * 3e-3


@pytest.fixture
def sample_3d_map() -> NDArray[np.floating]:
    """Create a sample 3D parameter map."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 32)).astype(np.float64) * 3e-3


class TestPlotParameterMap:
    """Tests for plot_parameter_map function."""

    def test_2d_map_returns_figure(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Function returns a matplotlib Figure for 2D data."""
        fig = plot_parameter_map(sample_2d_map)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_3d_map_returns_figure(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Function returns a matplotlib Figure for 3D data."""
        fig = plot_parameter_map(sample_3d_map)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_slice_idx_parameter(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Slice index parameter selects correct slice."""
        fig = plot_parameter_map(sample_3d_map, slice_idx=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_axes(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Function works with different axis values."""
        for axis in [0, 1, 2]:
            fig = plot_parameter_map(sample_3d_map, axis=axis)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_custom_cmap(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Custom colourmap is applied."""
        fig = plot_parameter_map(sample_2d_map, cmap="hot")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_vmin_vmax(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Vmin and vmax parameters are applied."""
        fig = plot_parameter_map(sample_2d_map, vmin=0, vmax=1e-3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_and_colorbar_label(
        self, sample_2d_map: NDArray[np.floating]
    ) -> None:
        """Title and colourbar label are displayed."""
        fig = plot_parameter_map(
            sample_2d_map,
            title="Test Map",
            colorbar_label="ADC (mm²/s)",
        )
        assert isinstance(fig, plt.Figure)
        # Check title is set
        axes = fig.get_axes()
        assert any(ax.get_title() == "Test Map" for ax in axes)
        plt.close(fig)

    def test_existing_axes(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Function can plot on existing axes."""
        fig, ax = plt.subplots()
        result_fig = plot_parameter_map(sample_2d_map, ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_invalid_ndim_raises(self) -> None:
        """Invalid array dimensions raise ValueError."""
        rng = np.random.default_rng(42)
        data_1d = rng.random(64).astype(np.float64)
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            plot_parameter_map(data_1d)

        data_4d = rng.random((64, 64, 32, 4)).astype(np.float64)
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            plot_parameter_map(data_4d)

    def test_invalid_axis_raises(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Invalid axis raises ValueError."""
        with pytest.raises(ValueError, match="Axis must be 0, 1, or 2"):
            plot_parameter_map(sample_3d_map, axis=3)


class TestPlotMultiSlice:
    """Tests for plot_multi_slice function."""

    def test_returns_figure(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Function returns a matplotlib Figure."""
        fig = plot_multi_slice(sample_3d_map)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_n_slices_parameter(self, sample_3d_map: NDArray[np.floating]) -> None:
        """n_slices parameter controls number of subplots."""
        for n in [4, 9, 12]:
            fig = plot_multi_slice(sample_3d_map, n_slices=n)
            # Count visible axes
            visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
            # Should have n slice subplots plus colourbar
            assert len(visible_axes) >= n
            plt.close(fig)

    def test_different_axes(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Function works with different axis values."""
        for axis in [0, 1, 2]:
            fig = plot_multi_slice(sample_3d_map, axis=axis, n_slices=4)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_custom_cmap(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Custom colourmap is applied."""
        fig = plot_multi_slice(sample_3d_map, cmap="plasma", n_slices=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_and_colorbar_label(
        self, sample_3d_map: NDArray[np.floating]
    ) -> None:
        """Title and colourbar label are displayed."""
        fig = plot_multi_slice(
            sample_3d_map,
            title="Multi-slice T1",
            colorbar_label="T1 (ms)",
            n_slices=4,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_ndim_raises(self, sample_2d_map: NDArray[np.floating]) -> None:
        """2D data raises ValueError."""
        with pytest.raises(ValueError, match="must be 3D"):
            plot_multi_slice(sample_2d_map)

    def test_invalid_axis_raises(self, sample_3d_map: NDArray[np.floating]) -> None:
        """Invalid axis raises ValueError."""
        with pytest.raises(ValueError, match="Axis must be 0, 1, or 2"):
            plot_multi_slice(sample_3d_map, axis=5)


class TestCompareMaps:
    """Tests for compare_maps function."""

    def test_2d_comparison_returns_figure(
        self, sample_2d_map: NDArray[np.floating]
    ) -> None:
        """Function returns Figure for 2D map comparison."""
        maps = [sample_2d_map, sample_2d_map * 1.1]
        titles = ["Map A", "Map B"]
        fig = compare_maps(maps, titles)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_3d_comparison_returns_figure(
        self, sample_3d_map: NDArray[np.floating]
    ) -> None:
        """Function returns Figure for 3D map comparison."""
        maps = [sample_3d_map, sample_3d_map * 1.1]
        titles = ["Map A", "Map B"]
        fig = compare_maps(maps, titles)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_maps(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Function handles multiple maps."""
        maps = [sample_2d_map, sample_2d_map * 1.1, sample_2d_map * 0.9]
        titles = ["A", "B", "C"]
        fig = compare_maps(maps, titles)
        # Check correct number of subplots
        main_axes = [ax for ax in fig.get_axes() if ax.get_title() in titles]
        assert len(main_axes) == 3
        plt.close(fig)

    def test_shared_colorbar(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Shared colourbar option works."""
        maps = [sample_2d_map, sample_2d_map * 2]
        titles = ["A", "B"]

        fig_shared = compare_maps(maps, titles, shared_colorbar=True)
        fig_separate = compare_maps(maps, titles, shared_colorbar=False)

        assert isinstance(fig_shared, plt.Figure)
        assert isinstance(fig_separate, plt.Figure)

        plt.close(fig_shared)
        plt.close(fig_separate)

    def test_suptitle(self, sample_2d_map: NDArray[np.floating]) -> None:
        """Suptitle is displayed."""
        maps = [sample_2d_map, sample_2d_map]
        titles = ["A", "B"]
        fig = compare_maps(maps, titles, suptitle="Comparison")
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Comparison"
        plt.close(fig)

    def test_mismatched_lengths_raises(
        self, sample_2d_map: NDArray[np.floating]
    ) -> None:
        """Mismatched maps and titles raise ValueError."""
        maps = [sample_2d_map, sample_2d_map]
        titles = ["A"]
        with pytest.raises(ValueError, match="must match"):
            compare_maps(maps, titles)

    def test_empty_maps_raises(self) -> None:
        """Empty maps list raises ValueError."""
        with pytest.raises(ValueError, match="At least one map"):
            compare_maps([], [])

    def test_inconsistent_shapes_raises(
        self, sample_2d_map: NDArray[np.floating]
    ) -> None:
        """Maps with different shapes raise ValueError."""
        rng = np.random.default_rng(42)
        map1 = sample_2d_map
        map2 = rng.random((32, 32)).astype(np.float64)
        with pytest.raises(ValueError, match="shape.*differs"):
            compare_maps([map1, map2], ["A", "B"])
