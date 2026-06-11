"""I/O utilities for qmri - NIFTI, DICOM, and BIDS support.

This package provides file I/O utilities for quantitative MRI processing,
including NIFTI file handling and BIDS-compliant filename parsing.

Example:
    ```python
    from qmri.io import load_nifti, save_nifti, parse_bids_filename
    data, header = load_nifti("sub-01_T1w.nii.gz")
    entities = parse_bids_filename("sub-01_T1w.nii.gz")
    print(f"Subject: {entities['sub']}")
    ```
"""

from qmri.io.bids import (
    BIDS_ENTITY_ORDER,
    build_bids_filename,
    find_sidecar_json,
    get_bids_path_components,
    is_bids_filename,
    load_sidecar,
    parse_bids_filename,
)
from qmri.io.nifti import (
    NiftiHeader,
    NiftiImage,
    get_affine,
    get_voxel_size,
    load_nifti,
    load_nifti_image,
    save_nifti,
)

__all__: list[str] = [
    # NIFTI I/O
    "NiftiHeader",
    "NiftiImage",
    "load_nifti",
    "load_nifti_image",
    "save_nifti",
    "get_voxel_size",
    "get_affine",
    # BIDS utilities
    "BIDS_ENTITY_ORDER",
    "parse_bids_filename",
    "build_bids_filename",
    "find_sidecar_json",
    "load_sidecar",
    "is_bids_filename",
    "get_bids_path_components",
]
