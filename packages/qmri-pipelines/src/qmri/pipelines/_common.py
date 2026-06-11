"""Internal helpers shared across qmri pipelines."""

from pathlib import Path

__all__ = ["strip_nifti_suffix"]


def strip_nifti_suffix(path: Path) -> Path:
    """Return ``path`` with a trailing ``.nii`` or ``.nii.gz`` suffix removed.

    Args:
        path: A path that may end in ``.nii`` or ``.nii.gz``.

    Returns:
        The path with the NIfTI suffix removed. Paths without a NIfTI suffix
        are returned unchanged.
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])
    if name.endswith(".nii"):
        return path.with_name(name[:-4])
    return path
