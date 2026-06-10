"""Tests for safe_divide utility."""

import numpy as np
from numpy.testing import assert_array_equal
from qmri._utils import safe_divide


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_basic_division(self) -> None:
        """Test basic array division."""
        result = safe_divide(
            np.array([10.0, 20.0, 30.0]),
            np.array([2.0, 4.0, 5.0]),
        )
        assert_array_equal(result, [5.0, 5.0, 6.0])

    def test_division_by_zero_returns_fill_value(self) -> None:
        """Test that division by zero returns fill_value."""
        result = safe_divide(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 0.0, 3.0]),
        )
        assert_array_equal(result, [1.0, 0.0, 1.0])

    def test_custom_fill_value(self) -> None:
        """Test custom fill value for division by zero."""
        result = safe_divide(
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0]),
            fill_value=-1.0,
        )
        assert_array_equal(result, [1.0, -1.0])

    def test_scalar_numerator(self) -> None:
        """Test scalar numerator with array divisor."""
        result = safe_divide(10.0, np.array([2.0, 0.0, 5.0]))
        assert_array_equal(result, [5.0, 0.0, 2.0])

    def test_scalar_divisor(self) -> None:
        """Test array numerator with scalar divisor."""
        result = safe_divide(np.array([10.0, 20.0, 30.0]), 5.0)
        assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_both_scalars(self) -> None:
        """Test both scalar inputs."""
        result = safe_divide(10.0, 2.0)
        assert result == 5.0

    def test_scalar_divisor_zero(self) -> None:
        """Test scalar divisor that is zero."""
        result = safe_divide(10.0, 0.0)
        assert result == 0.0

    def test_multidimensional_arrays(self) -> None:
        """Test with multidimensional arrays."""
        num = np.ones((3, 3, 3)) * 10.0
        div = np.ones((3, 3, 3)) * 2.0
        div[1, 1, 1] = 0.0  # Set one element to zero

        result = safe_divide(num, div)

        assert result.shape == (3, 3, 3)
        assert result[0, 0, 0] == 5.0
        assert result[1, 1, 1] == 0.0  # Fill value

    def test_broadcasting(self) -> None:
        """Test broadcasting between different shaped arrays."""
        num = np.array([[1.0, 2.0], [3.0, 4.0]])
        div = np.array([1.0, 0.0])  # Will broadcast

        result = safe_divide(num, div)

        assert result.shape == (2, 2)
        assert_array_equal(result[:, 0], [1.0, 3.0])  # Divided by 1
        assert_array_equal(result[:, 1], [0.0, 0.0])  # Divided by 0

    def test_all_zeros_divisor(self) -> None:
        """Test when all divisor elements are zero."""
        result = safe_divide(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_negative_values(self) -> None:
        """Test with negative values."""
        result = safe_divide(
            np.array([-10.0, 10.0]),
            np.array([2.0, -2.0]),
        )
        assert_array_equal(result, [-5.0, -5.0])
