"""Magnetisation transfer models.

This module provides functions for:

- MTR (Magnetisation Transfer Ratio) calculation
"""

from qmri.transfer import mtr
from qmri.transfer.mtr import MTRResult, calculate_mtr

__all__ = [
    "mtr",
    "MTRResult",
    "calculate_mtr",
]
