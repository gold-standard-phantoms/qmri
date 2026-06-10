"""Safe division utilities.

This module provides functions for safe division that handle
divide-by-zero conditions gracefully.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["safe_divide"]


def safe_divide(
    numerator: NDArray[np.floating] | float,
    divisor: NDArray[np.floating] | float,
    *,
    fill_value: float = 0.0,
) -> NDArray[np.floating]:
    """Safely divide arrays, returning fill_value where divisor is zero.

    Args:
        numerator: The numerator for division.
        divisor: The divisor for division.
        fill_value: Value to use where divisor is zero (default 0.0).

    Returns:
        Result of division, with fill_value where divisor was zero.

    Example:
        ```python
        import numpy as np
        from qmri._utils import safe_divide

        safe_divide(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 3.0]))
        # array([1., 0., 1.])

        safe_divide(10.0, np.array([2.0, 0.0, 5.0]))
        # array([5., 0., 2.])
        ```
    """
    num = np.asarray(numerator, dtype=np.float64)
    div = np.asarray(divisor, dtype=np.float64)

    # Broadcast to common shape
    broadcast_shape = np.broadcast(num, div).shape
    if broadcast_shape:
        num = np.broadcast_to(num, broadcast_shape)
        div = np.broadcast_to(div, broadcast_shape)
        out: NDArray[np.floating] = np.full(
            broadcast_shape, fill_value, dtype=np.float64
        )
    else:
        out = np.array(fill_value, dtype=np.float64)

    # Perform division only where divisor is non-zero
    result: NDArray[np.floating] = np.divide(num, div, out=out, where=div != 0)
    return result
