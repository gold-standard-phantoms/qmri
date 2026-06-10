"""MRI sequence signal equations.

This module provides signal equations for:

- Gradient Echo (GRE)
- Spin Echo (SE)
- Inversion Recovery (IR)
"""

from qmri.sequences import gre, ir, se

__all__ = [
    "gre",
    "ir",
    "se",
]
