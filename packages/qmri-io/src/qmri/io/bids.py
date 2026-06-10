"""BIDS (Brain Imaging Data Structure) utilities.

This module provides functions for parsing and building BIDS-compliant
filenames, as well as finding and loading associated JSON sidecar files.

Example:
    ```python
    from qmri.io.bids import parse_bids_filename, build_bids_filename
    entities = parse_bids_filename("sub-01_ses-pre_T1w.nii.gz")
    print(entities)
    # {'sub': '01', 'ses': 'pre', 'suffix': 'T1w', 'extension': '.nii.gz'}
    ```

References:
    .. [1] Gorgolewski, K.J., et al. (2016). "The brain imaging data structure,
           a format for organising and describing outputs of neuroimaging
           experiments." Scientific Data, 3:160044.
    .. [2] BIDS Specification: https://bids-specification.readthedocs.io/
"""

import json
from pathlib import Path
from typing import Any

# Standard BIDS entity keys in their canonical order
BIDS_ENTITY_ORDER: tuple[str, ...] = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "dir",
    "run",
    "mod",
    "echo",
    "flip",
    "inv",
    "mt",
    "part",
    "proc",
    "space",
    "split",
    "recording",
    "chunk",
)


def parse_bids_filename(filename: str) -> dict[str, str]:
    """Parse BIDS entities from a filename.

    Extracts key-value pairs from a BIDS-compliant filename following the
    pattern: key1-value1_key2-value2_..._suffix.extension

    Args:
        filename: The filename to parse. Can be just the filename or a full
            path. The function extracts only the filename component.

    Returns:
        Dictionary containing parsed entities. Always includes 'suffix'
        (the final component before the extension) and 'extension'.
        Entity keys are lowercase (e.g., 'sub', 'ses', 'run').

    Example:
        Parse a typical anatomical filename:

        ```python
        from qmri.io.bids import parse_bids_filename
        entities = parse_bids_filename("sub-01_ses-pre_T1w.nii.gz")
        print(entities['sub'])
        # '01'
        print(entities['suffix'])
        # 'T1w'
        ```

        Parse a diffusion-weighted image:

        ```python
        entities = parse_bids_filename("sub-02_ses-01_run-1_dwi.nii.gz")
        print(entities)
        # {'sub': '02', 'ses': '01', 'run': '1', 'suffix': 'dwi',
        #  'extension': '.nii.gz'}
        ```

        Parse from a full path:

        ```python
        entities = parse_bids_filename("/data/bids/sub-01/anat/sub-01_T1w.nii.gz")
        print(entities['sub'])
        # '01'
        ```

    Notes:
        The parser handles compound extensions like .nii.gz correctly.
        Entity keys are normalised to lowercase for consistency.
    """
    # Extract just the filename if a path is provided
    filename = Path(filename).name

    # Handle compound extensions (.nii.gz, .tsv.gz, etc.)
    if filename.endswith(".nii.gz"):
        extension = ".nii.gz"
        stem = filename[: -len(extension)]
    elif filename.endswith(".tsv.gz"):
        extension = ".tsv.gz"
        stem = filename[: -len(extension)]
    elif filename.endswith(".json.gz"):
        extension = ".json.gz"
        stem = filename[: -len(extension)]
    else:
        # Standard single extension
        path_obj = Path(filename)
        extension = path_obj.suffix
        stem = path_obj.stem

    # Split into components
    parts = stem.split("_")

    entities: dict[str, str] = {}

    # Parse entity-value pairs (format: key-value)
    for part in parts[:-1]:  # All but the last component
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key.lower()] = value

    # The last component is the suffix (e.g., T1w, bold, dwi)
    if parts:
        entities["suffix"] = parts[-1]

    # Add extension
    entities["extension"] = extension

    return entities


def build_bids_filename(
    entities: dict[str, str],
    suffix: str,
    extension: str = ".nii.gz",
) -> str:
    """Build a BIDS-compliant filename from entities.

    Constructs a filename following BIDS naming conventions, with entities
    in their canonical order.

    Args:
        entities: Dictionary of BIDS entities (e.g., {'sub': '01', 'ses': 'pre'}).
            Keys should be standard BIDS entity names. Unknown entities are
            appended at the end in alphabetical order.
        suffix: The BIDS suffix (e.g., 'T1w', 'bold', 'dwi', 'adc').
        extension: File extension including the leading dot. Default is '.nii.gz'.

    Returns:
        BIDS-compliant filename.

    Example:
        Build an anatomical filename:

        ```python
        from qmri.io.bids import build_bids_filename
        filename = build_bids_filename({'sub': '01', 'ses': 'pre'}, 'T1w')
        print(filename)
        # sub-01_ses-pre_T1w.nii.gz
        ```

        Build a functional filename with multiple entities:

        ```python
        filename = build_bids_filename(
            {'sub': '02', 'ses': '01', 'task': 'rest', 'run': '1'},
            'bold'
        )
        print(filename)
        # sub-02_ses-01_task-rest_run-1_bold.nii.gz
        ```

        Build a JSON sidecar filename:

        ```python
        filename = build_bids_filename({'sub': '01'}, 'T1w', extension='.json')
        print(filename)
        # sub-01_T1w.json
        ```

    Notes:
        Entities are ordered according to the BIDS specification. The 'sub'
        entity always comes first, followed by 'ses', 'task', etc.
    """
    # Build list of entity-value pairs in canonical order
    parts: list[str] = []

    # Add entities in canonical order
    for key in BIDS_ENTITY_ORDER:
        if key in entities:
            parts.append(f"{key}-{entities[key]}")

    # Add any remaining entities not in the canonical order (sorted alphabetically)
    remaining_keys = sorted(set(entities.keys()) - set(BIDS_ENTITY_ORDER))
    for key in remaining_keys:
        # Skip special keys that aren't entity pairs
        if key not in ("suffix", "extension"):
            parts.append(f"{key}-{entities[key]}")

    # Add suffix
    parts.append(suffix)

    # Join with underscores and add extension
    filename = "_".join(parts) + extension

    return filename


def find_sidecar_json(nifti_path: str | Path) -> Path | None:
    """Find the associated JSON sidecar file for a NIFTI file.

    BIDS associates metadata with NIFTI files through JSON sidecar files
    that share the same filename stem.

    Args:
        nifti_path: Path to the NIFTI file (.nii or .nii.gz).

    Returns:
        Path to the JSON sidecar if it exists, None otherwise.

    Example:
        ```python
        from qmri.io.bids import find_sidecar_json
        json_path = find_sidecar_json("sub-01_T1w.nii.gz")
        if json_path:
            print(f"Found sidecar: {json_path}")
        else:
            print("No sidecar found")
        ```

    Notes:
        The function looks for a JSON file with the same stem as the NIFTI
        file in the same directory. BIDS inheritance rules (looking in parent
        directories) are not currently implemented.
    """
    nifti_path = Path(nifti_path)

    # Get the stem, handling .nii.gz extension
    if nifti_path.name.endswith(".nii.gz"):
        stem = nifti_path.name[:-7]  # Remove .nii.gz
    else:
        stem = nifti_path.stem

    # Look for JSON sidecar in the same directory
    json_path = nifti_path.parent / f"{stem}.json"

    if json_path.exists():
        return json_path

    return None


def load_sidecar(nifti_path: str | Path) -> dict[str, Any]:
    """Load the JSON sidecar associated with a NIFTI file.

    Args:
        nifti_path: Path to the NIFTI file (.nii or .nii.gz).

    Returns:
        Contents of the JSON sidecar file. Returns an empty dictionary
        if no sidecar file is found.

    Example:
        ```python
        from qmri.io.bids import load_sidecar
        metadata = load_sidecar("sub-01_dwi.nii.gz")
        if "EchoTime" in metadata:
            print(f"TE: {metadata['EchoTime']} s")
        ```

        Load diffusion parameters:

        ```python
        metadata = load_sidecar("sub-01_dwi.nii.gz")
        b_values = metadata.get("DiffusionBValue", [])
        print(f"B-values: {b_values}")
        ```

    Notes:
        Common BIDS fields for MRI include:

        - RepetitionTime (TR in seconds)
        - EchoTime (TE in seconds)
        - FlipAngle (in degrees)
        - InversionTime (TI in seconds)
        - DiffusionBValue (b-value in s/mm²)
    """
    json_path = find_sidecar_json(nifti_path)

    if json_path is None:
        return {}

    with open(json_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    return data


def is_bids_filename(filename: str) -> bool:
    """Check if a filename follows BIDS naming conventions.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename appears to follow BIDS conventions.

    Example:
        ```python
        from qmri.io.bids import is_bids_filename
        is_bids_filename("sub-01_T1w.nii.gz")
        # True
        is_bids_filename("brain_image.nii")
        # False
        ```

    Notes:
        This performs a basic check for BIDS compliance:

        - Must contain at least one entity (key-value pair with hyphen)
        - Must have a recognisable extension
        - The 'sub' entity must be present (required in BIDS)
    """
    # Extract just the filename
    filename = Path(filename).name

    # Check for valid extension
    valid_extensions = (".nii", ".nii.gz", ".json", ".tsv", ".tsv.gz", ".bval", ".bvec")
    has_valid_extension = any(filename.endswith(ext) for ext in valid_extensions)

    if not has_valid_extension:
        return False

    # Parse and check for required 'sub' entity
    try:
        entities = parse_bids_filename(filename)
        return "sub" in entities
    except (ValueError, IndexError):
        return False


def get_bids_path_components(path: str | Path) -> dict[str, str]:
    """Extract BIDS path components from a full path.

    Parses a BIDS-organised path to extract subject, session, and datatype
    information from the directory structure.

    Args:
        path: Full path to a BIDS file.

    Returns:
        Dictionary containing path components:

        - 'subject_dir': Subject directory name (e.g., 'sub-01')
        - 'session_dir': Session directory name if present (e.g., 'ses-pre')
        - 'datatype': Data type directory (e.g., 'anat', 'func', 'dwi')
        Empty strings for components that are not found.

    Example:
        ```python
        from qmri.io.bids import get_bids_path_components
        components = get_bids_path_components(
            "/data/bids/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii.gz"
        )
        print(components['subject_dir'])
        # sub-01
        print(components['datatype'])
        # anat
        ```
    """
    path = Path(path)
    # Get directory parts only (exclude the filename)
    parts = path.parent.parts

    result: dict[str, str] = {
        "subject_dir": "",
        "session_dir": "",
        "datatype": "",
    }

    # Known BIDS datatypes
    datatypes = {"anat", "func", "dwi", "fmap", "perf", "meg", "eeg", "ieeg", "beh"}

    for part in parts:
        if part.startswith("sub-"):
            result["subject_dir"] = part
        elif part.startswith("ses-"):
            result["session_dir"] = part
        elif part in datatypes:
            result["datatype"] = part

    return result
