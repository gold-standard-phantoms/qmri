"""NIFTI file I/O utilities.

This module provides functions for loading and saving NIFTI files, as well as
extracting metadata from NIFTI headers. It wraps nibabel functionality with
a simplified, type-safe interface.

Example:
    ```python
    from qmri.io.nifti import load_nifti, save_nifti, NiftiImage
    data, header = load_nifti("brain.nii.gz")
    print(f"Data shape: {data.shape}")
    save_nifti(data * 2, "brain_scaled.nii.gz", header=header)
    ```

Notes:
    This module requires nibabel >= 5.0 for full functionality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Type alias for nibabel header (Any when nibabel stubs not available)
NiftiHeader = Any


@dataclass(frozen=True)
class NiftiImage:
    """Container for NIFTI image data and metadata.

    This dataclass provides a convenient way to bundle NIFTI data with its
    associated header and affine transformation matrix.

    Attributes:
        data: The image data array. Shape depends on the image dimensionality
            (e.g., 3D for anatomical, 4D for functional/diffusion).
        header: NIFTI header containing image metadata (voxel dimensions,
            data type, orientation information, etc.).
        affine: 4x4 affine transformation matrix mapping voxel coordinates to
            world (scanner) coordinates in millimetres.

    Example:
        ```python
        from qmri.io.nifti import load_nifti_image
        img = load_nifti_image("brain.nii.gz")
        print(f"Shape: {img.data.shape}")
        print(f"Voxel size: {img.header.get_zooms()[:3]}")
        ```
    """

    data: NDArray[np.floating]
    header: NiftiHeader
    affine: NDArray[np.floating]


def load_nifti(
    path: str | Path,
) -> tuple[NDArray[np.floating], NiftiHeader]:
    """Load a NIFTI file and return the data and header.

    Args:
        path: Path to the NIFTI file. Supports both .nii and .nii.gz formats.

    Returns:
        A tuple containing:

            - data: The image data as a floating-point array. The original
              data type is converted to float64 for numerical stability.
            - header: The NIFTI header containing image metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        nibabel.filebasedimages.ImageFileError: If the file is not a valid
            NIFTI image.

    Example:
        ```python
        from qmri.io.nifti import load_nifti
        data, header = load_nifti("brain.nii.gz")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        ```

    Notes:
        The data is loaded eagerly (not memory-mapped) and converted to float64.
        For very large files, consider using nibabel directly with memory mapping.
    """
    import nibabel as nib

    path = Path(path)
    if not path.exists():
        msg = f"NIFTI file not found: {path}"
        raise FileNotFoundError(msg)

    img = nib.load(path)  # type: ignore[attr-defined]
    data: NDArray[np.floating] = np.asarray(
        img.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    header: NiftiHeader = img.header

    return data, header


def load_nifti_image(path: str | Path) -> NiftiImage:
    """Load a NIFTI file and return a NiftiImage object.

    This is a convenience function that returns all components of a NIFTI
    file bundled in a single dataclass.

    Args:
        path: Path to the NIFTI file. Supports both .nii and .nii.gz formats.

    Returns:
        Dataclass containing data, header, and affine transformation.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        nibabel.filebasedimages.ImageFileError: If the file is not a valid
            NIFTI image.

    Example:
        ```python
        from qmri.io.nifti import load_nifti_image
        img = load_nifti_image("brain.nii.gz")
        voxel_size = get_voxel_size(img.header)
        print(f"Voxel size: {voxel_size} mm")
        ```
    """
    import nibabel as nib

    path = Path(path)
    if not path.exists():
        msg = f"NIFTI file not found: {path}"
        raise FileNotFoundError(msg)

    img = nib.load(path)  # type: ignore[attr-defined]
    data: NDArray[np.floating] = np.asarray(
        img.get_fdata(),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    header: NiftiHeader = img.header
    affine: NDArray[np.floating] = np.asarray(
        img.affine,  # type: ignore[attr-defined]
        dtype=np.float64,
    )

    return NiftiImage(data=data, header=header, affine=affine)


def save_nifti(
    data: NDArray[np.floating],
    path: str | Path,
    header: NiftiHeader | None = None,
    affine: NDArray[np.floating] | None = None,
) -> None:
    """Save an array to a NIFTI file.

    Args:
        data: The image data to save. Can be any dimensionality supported
            by NIFTI.
        path: Output path for the NIFTI file. The extension determines the
            format: use .nii.gz for compressed output (recommended) or .nii
            for uncompressed.
        header: NIFTI header to use. If None, a default header is created.
            The header's data shape and type will be updated to match the data.
        affine: 4x4 affine transformation matrix. If None and header is None,
            uses an identity matrix. If None but header is provided, the
            affine is derived from the header.

    Raises:
        ValueError: If the affine matrix is not 4x4.

    Example:
        Save with default header:

        ```python
        import numpy as np
        from qmri.io.nifti import save_nifti
        data = np.random.rand(64, 64, 30)
        save_nifti(data, "output.nii.gz")
        ```

        Save with custom affine (2mm isotropic voxels):

        ```python
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        save_nifti(data, "output_2mm.nii.gz", affine=affine)
        ```

        Preserve header from original file:

        ```python
        from qmri.io.nifti import load_nifti
        data, header = load_nifti("input.nii.gz")
        processed = data * 2
        save_nifti(processed, "output.nii.gz", header=header)
        ```

    Notes:
        The data is saved as float64 by default. For integer data types,
        convert the array before saving and provide an appropriate header.
    """
    import nibabel as nib

    path = Path(path)
    data = np.asarray(data, dtype=np.float64)

    # Determine the affine matrix
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4):
            msg = f"Affine must be 4x4, got shape {affine.shape}"
            raise ValueError(msg)
        affine_to_use: NDArray[Any] = affine
    elif header is not None:
        # Extract affine from header
        affine_to_use = header.get_best_affine()
    else:
        # Default to identity
        affine_to_use = np.eye(4, dtype=np.float64)

    # Create the NIFTI image
    if header is not None:
        # Create new image with updated header
        img = nib.Nifti1Image(  # type: ignore[attr-defined,no-untyped-call]
            data, affine_to_use, header=header
        )
    else:
        img = nib.Nifti1Image(  # type: ignore[attr-defined,no-untyped-call]
            data, affine_to_use
        )

    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the image
    nib.save(img, path)  # type: ignore[attr-defined]


def get_voxel_size(header: NiftiHeader) -> tuple[float, float, float]:
    """Extract voxel dimensions from a NIFTI header.

    Args:
        header: NIFTI header from which to extract voxel dimensions.

    Returns:
        Voxel dimensions (x, y, z) in millimetres.

    Example:
        ```python
        from qmri.io.nifti import load_nifti, get_voxel_size
        _, header = load_nifti("brain.nii.gz")
        voxel_size = get_voxel_size(header)
        print(f"Voxel size: {voxel_size[0]:.2f} x {voxel_size[1]:.2f} x "
              f"{voxel_size[2]:.2f} mm")
        ```

    Notes:
        This function only returns the first three dimensions (spatial).
        For 4D images (e.g., fMRI, DWI), the fourth dimension (time/volume)
        is not included. Use header.get_zooms() directly to access all
        dimensions including TR for fMRI data.
    """
    zooms = header.get_zooms()
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def get_affine(header: NiftiHeader) -> NDArray[np.floating]:
    """Extract the affine transformation matrix from a NIFTI header.

    Args:
        header: NIFTI header from which to extract the affine.

    Returns:
        4x4 affine transformation matrix mapping voxel indices to
        world coordinates in millimetres.

    Example:
        ```python
        from qmri.io.nifti import load_nifti, get_affine
        _, header = load_nifti("brain.nii.gz")
        affine = get_affine(header)
        print(f"Affine shape: {affine.shape}")
        ```
    """
    affine: NDArray[np.floating] = np.asarray(
        header.get_best_affine(), dtype=np.float64
    )
    return affine
