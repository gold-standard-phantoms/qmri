"""Internal utilities (not part of public API).

These utilities are used internally by qmri modules but are not
guaranteed to have a stable API.
"""

from qmri._utils.safe_divide import safe_divide

__all__ = ["safe_divide"]
