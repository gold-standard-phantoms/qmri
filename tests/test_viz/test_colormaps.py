"""Tests for qmri.viz.colormaps module."""

import matplotlib as mpl
import matplotlib.colors as mcolors

# Use non-interactive backend for testing
mpl.use("Agg")

from qmri.viz.colormaps import (
    get_adc_cmap,
    get_t1_cmap,
    get_t2_cmap,
    register_qmri_cmaps,
)


class TestGetADCCmap:
    """Tests for get_adc_cmap function."""

    def test_returns_colormap(self) -> None:
        """Function returns a matplotlib Colormap."""
        cmap = get_adc_cmap()
        assert isinstance(cmap, mcolors.Colormap)

    def test_colormap_has_correct_range(self) -> None:
        """Colourmap maps values in [0, 1] to RGBA."""
        cmap = get_adc_cmap()
        # Test at different positions
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rgba = cmap(val)
            assert len(rgba) == 4
            assert all(0 <= c <= 1 for c in rgba)

    def test_colormap_is_continuous(self) -> None:
        """Colourmap produces smooth transitions."""
        cmap = get_adc_cmap()
        # Get colours at nearby values
        c1 = cmap(0.5)
        c2 = cmap(0.51)
        # Colours should be similar (difference < 0.1 per channel)
        for i in range(3):  # RGB channels
            assert abs(c1[i] - c2[i]) < 0.1


class TestGetT1Cmap:
    """Tests for get_t1_cmap function."""

    def test_returns_colormap(self) -> None:
        """Function returns a matplotlib Colormap."""
        cmap = get_t1_cmap()
        assert isinstance(cmap, mcolors.Colormap)

    def test_colormap_has_correct_range(self) -> None:
        """Colourmap maps values in [0, 1] to RGBA."""
        cmap = get_t1_cmap()
        for val in [0.0, 0.5, 1.0]:
            rgba = cmap(val)
            assert len(rgba) == 4
            assert all(0 <= c <= 1 for c in rgba)

    def test_different_from_adc_cmap(self) -> None:
        """T1 colourmap is different from ADC colourmap."""
        adc_cmap = get_adc_cmap()
        t1_cmap = get_t1_cmap()
        # Compare colours at midpoint
        adc_mid = adc_cmap(0.5)
        t1_mid = t1_cmap(0.5)
        # At least one channel should differ significantly
        assert any(abs(adc_mid[i] - t1_mid[i]) > 0.1 for i in range(3))


class TestGetT2Cmap:
    """Tests for get_t2_cmap function."""

    def test_returns_colormap(self) -> None:
        """Function returns a matplotlib Colormap."""
        cmap = get_t2_cmap()
        assert isinstance(cmap, mcolors.Colormap)

    def test_colormap_has_correct_range(self) -> None:
        """Colourmap maps values in [0, 1] to RGBA."""
        cmap = get_t2_cmap()
        for val in [0.0, 0.5, 1.0]:
            rgba = cmap(val)
            assert len(rgba) == 4
            assert all(0 <= c <= 1 for c in rgba)

    def test_different_from_other_cmaps(self) -> None:
        """T2 colourmap is different from ADC and T1 colourmaps."""
        adc_cmap = get_adc_cmap()
        t1_cmap = get_t1_cmap()
        t2_cmap = get_t2_cmap()

        adc_mid = adc_cmap(0.5)
        t1_mid = t1_cmap(0.5)
        t2_mid = t2_cmap(0.5)

        # T2 should differ from both
        assert any(abs(adc_mid[i] - t2_mid[i]) > 0.1 for i in range(3))
        assert any(abs(t1_mid[i] - t2_mid[i]) > 0.1 for i in range(3))


class TestRegisterQmriCmaps:
    """Tests for register_qmri_cmaps function."""

    def test_registers_cmaps(self) -> None:
        """Function registers colourmaps with matplotlib."""
        register_qmri_cmaps()

        # Check that colourmaps are accessible by name
        for name in ["qmri_adc", "qmri_t1", "qmri_t2"]:
            cmap = mpl.colormaps.get_cmap(name)
            assert cmap is not None
            assert isinstance(cmap, mcolors.Colormap)

    def test_idempotent(self) -> None:
        """Function can be called multiple times without error."""
        # Should not raise even if already registered
        register_qmri_cmaps()
        register_qmri_cmaps()
