"""Tests for BIDS utilities."""

import json
from pathlib import Path

from qmri.io import (
    BIDS_ENTITY_ORDER,
    build_bids_filename,
    find_sidecar_json,
    get_bids_path_components,
    is_bids_filename,
    load_sidecar,
    parse_bids_filename,
)


class TestParseBidsFilename:
    """Tests for parse_bids_filename function."""

    def test_parse_simple_filename(self) -> None:
        """Test parsing a simple BIDS filename."""
        result = parse_bids_filename("sub-01_T1w.nii.gz")

        assert result["sub"] == "01"
        assert result["suffix"] == "T1w"
        assert result["extension"] == ".nii.gz"

    def test_parse_with_session(self) -> None:
        """Test parsing filename with session."""
        result = parse_bids_filename("sub-01_ses-pre_T1w.nii.gz")

        assert result["sub"] == "01"
        assert result["ses"] == "pre"
        assert result["suffix"] == "T1w"

    def test_parse_multiple_entities(self) -> None:
        """Test parsing filename with multiple entities."""
        result = parse_bids_filename("sub-02_ses-01_task-rest_run-1_bold.nii.gz")

        assert result["sub"] == "02"
        assert result["ses"] == "01"
        assert result["task"] == "rest"
        assert result["run"] == "1"
        assert result["suffix"] == "bold"

    def test_parse_dwi_filename(self) -> None:
        """Test parsing diffusion-weighted image filename."""
        result = parse_bids_filename("sub-03_ses-baseline_acq-multiband_dwi.nii.gz")

        assert result["sub"] == "03"
        assert result["ses"] == "baseline"
        assert result["acq"] == "multiband"
        assert result["suffix"] == "dwi"

    def test_parse_from_full_path(self) -> None:
        """Test parsing extracts filename from full path."""
        result = parse_bids_filename("/data/bids/sub-01/anat/sub-01_T1w.nii.gz")

        assert result["sub"] == "01"
        assert result["suffix"] == "T1w"

    def test_parse_uncompressed_nifti(self) -> None:
        """Test parsing uncompressed .nii extension."""
        result = parse_bids_filename("sub-01_T1w.nii")

        assert result["extension"] == ".nii"
        assert result["suffix"] == "T1w"

    def test_parse_json_extension(self) -> None:
        """Test parsing JSON sidecar filename."""
        result = parse_bids_filename("sub-01_T1w.json")

        assert result["extension"] == ".json"
        assert result["suffix"] == "T1w"

    def test_parse_tsv_extension(self) -> None:
        """Test parsing TSV filename."""
        result = parse_bids_filename("sub-01_ses-01_events.tsv")

        assert result["extension"] == ".tsv"
        assert result["suffix"] == "events"

    def test_parse_compound_values(self) -> None:
        """Test parsing entities with compound values (containing hyphens)."""
        # Some BIDS values can contain multiple hyphens
        result = parse_bids_filename("sub-01_acq-high-res_T1w.nii.gz")

        assert result["sub"] == "01"
        assert result["acq"] == "high-res"

    def test_parse_numeric_entity_values(self) -> None:
        """Test parsing entities with numeric values."""
        result = parse_bids_filename("sub-001_ses-002_run-03_bold.nii.gz")

        assert result["sub"] == "001"
        assert result["ses"] == "002"
        assert result["run"] == "03"

    def test_parse_echo_entity(self) -> None:
        """Test parsing multi-echo data with echo entity."""
        result = parse_bids_filename("sub-01_echo-1_T2starw.nii.gz")

        assert result["echo"] == "1"
        assert result["suffix"] == "T2starw"


class TestBuildBidsFilename:
    """Tests for build_bids_filename function."""

    def test_build_simple_filename(self) -> None:
        """Test building a simple BIDS filename."""
        result = build_bids_filename({"sub": "01"}, "T1w")

        assert result == "sub-01_T1w.nii.gz"

    def test_build_with_session(self) -> None:
        """Test building filename with session."""
        result = build_bids_filename({"sub": "01", "ses": "pre"}, "T1w")

        assert result == "sub-01_ses-pre_T1w.nii.gz"

    def test_build_with_multiple_entities(self) -> None:
        """Test building filename with multiple entities."""
        entities = {"sub": "02", "ses": "01", "task": "rest", "run": "1"}
        result = build_bids_filename(entities, "bold")

        assert result == "sub-02_ses-01_task-rest_run-1_bold.nii.gz"

    def test_build_with_custom_extension(self) -> None:
        """Test building filename with custom extension."""
        result = build_bids_filename({"sub": "01"}, "T1w", extension=".json")

        assert result == "sub-01_T1w.json"

    def test_build_dwi_filename(self) -> None:
        """Test building diffusion filename."""
        entities = {"sub": "03", "ses": "baseline", "acq": "multiband"}
        result = build_bids_filename(entities, "dwi")

        assert result == "sub-03_ses-baseline_acq-multiband_dwi.nii.gz"

    def test_build_maintains_entity_order(self) -> None:
        """Test that entities are ordered correctly."""
        # Provide entities in non-canonical order
        entities = {"run": "1", "sub": "01", "task": "rest", "ses": "pre"}
        result = build_bids_filename(entities, "bold")

        # Should be in canonical order: sub, ses, task, run
        assert result == "sub-01_ses-pre_task-rest_run-1_bold.nii.gz"

    def test_build_with_unknown_entities(self) -> None:
        """Test building with non-standard entities."""
        entities = {"sub": "01", "custom": "value"}
        result = build_bids_filename(entities, "T1w")

        # Unknown entities should be appended after standard ones
        assert "sub-01" in result
        assert "custom-value" in result
        assert result.endswith("T1w.nii.gz")

    def test_build_roundtrip(self) -> None:
        """Test that parse and build are inverse operations."""
        original = "sub-02_ses-01_task-rest_run-1_bold.nii.gz"
        parsed = parse_bids_filename(original)

        # Remove suffix and extension for rebuild
        entities = {k: v for k, v in parsed.items() if k not in ("suffix", "extension")}
        rebuilt = build_bids_filename(entities, parsed["suffix"], parsed["extension"])

        assert rebuilt == original


class TestFindSidecarJson:
    """Tests for find_sidecar_json function."""

    def test_find_existing_sidecar(self, tmp_path: Path) -> None:
        """Test finding an existing JSON sidecar."""
        # Create NIFTI and JSON files
        nifti_file = tmp_path / "sub-01_T1w.nii.gz"
        json_file = tmp_path / "sub-01_T1w.json"
        nifti_file.touch()
        json_file.write_text("{}")

        result = find_sidecar_json(nifti_file)

        assert result is not None
        assert result == json_file

    def test_find_missing_sidecar(self, tmp_path: Path) -> None:
        """Test that None is returned for missing sidecar."""
        nifti_file = tmp_path / "sub-01_T1w.nii.gz"
        nifti_file.touch()

        result = find_sidecar_json(nifti_file)

        assert result is None

    def test_find_sidecar_uncompressed_nifti(self, tmp_path: Path) -> None:
        """Test finding sidecar for uncompressed .nii file."""
        nifti_file = tmp_path / "sub-01_T1w.nii"
        json_file = tmp_path / "sub-01_T1w.json"
        nifti_file.touch()
        json_file.write_text("{}")

        result = find_sidecar_json(nifti_file)

        assert result is not None
        assert result == json_file

    def test_find_sidecar_with_string_path(self, tmp_path: Path) -> None:
        """Test that string paths work correctly."""
        nifti_file = tmp_path / "sub-01_T1w.nii.gz"
        json_file = tmp_path / "sub-01_T1w.json"
        nifti_file.touch()
        json_file.write_text("{}")

        result = find_sidecar_json(str(nifti_file))

        assert result is not None


class TestLoadSidecar:
    """Tests for load_sidecar function."""

    def test_load_existing_sidecar(self, tmp_path: Path) -> None:
        """Test loading an existing JSON sidecar."""
        nifti_file = tmp_path / "sub-01_dwi.nii.gz"
        json_file = tmp_path / "sub-01_dwi.json"
        nifti_file.touch()

        metadata = {
            "RepetitionTime": 2.0,
            "EchoTime": 0.03,
            "FlipAngle": 90,
        }
        json_file.write_text(json.dumps(metadata))

        result = load_sidecar(nifti_file)

        assert result["RepetitionTime"] == 2.0
        assert result["EchoTime"] == 0.03
        assert result["FlipAngle"] == 90

    def test_load_missing_sidecar_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that missing sidecar returns empty dict."""
        nifti_file = tmp_path / "sub-01_T1w.nii.gz"
        nifti_file.touch()

        result = load_sidecar(nifti_file)

        assert result == {}

    def test_load_sidecar_with_nested_data(self, tmp_path: Path) -> None:
        """Test loading sidecar with nested JSON structure."""
        nifti_file = tmp_path / "sub-01_dwi.nii.gz"
        json_file = tmp_path / "sub-01_dwi.json"
        nifti_file.touch()

        metadata = {
            "DiffusionBValue": [0, 1000, 2000],
            "SliceTiming": [0.0, 0.05, 0.1],
            "Manufacturer": "Siemens",
        }
        json_file.write_text(json.dumps(metadata))

        result = load_sidecar(nifti_file)

        assert result["DiffusionBValue"] == [0, 1000, 2000]
        assert len(result["SliceTiming"]) == 3


class TestIsBidsFilename:
    """Tests for is_bids_filename function."""

    def test_valid_bids_filename(self) -> None:
        """Test that valid BIDS filenames are recognised."""
        valid_names = [
            "sub-01_T1w.nii.gz",
            "sub-01_ses-pre_T1w.nii.gz",
            "sub-001_ses-01_task-rest_bold.nii.gz",
            "sub-01_dwi.nii",
            "sub-01_T1w.json",
        ]

        for name in valid_names:
            assert is_bids_filename(name), f"Expected {name} to be valid"

    def test_invalid_bids_filename(self) -> None:
        """Test that non-BIDS filenames are rejected."""
        invalid_names = [
            "brain_image.nii.gz",  # No sub entity
            "T1w.nii.gz",  # No entities
            "ses-01_T1w.nii.gz",  # No sub entity
            "sub-01_T1w.dcm",  # Invalid extension
            "data.txt",  # Not BIDS extension
        ]

        for name in invalid_names:
            assert not is_bids_filename(name), f"Expected {name} to be invalid"

    def test_bids_filename_from_path(self) -> None:
        """Test BIDS validation works with full paths."""
        assert is_bids_filename("/data/bids/sub-01/anat/sub-01_T1w.nii.gz")


class TestGetBidsPathComponents:
    """Tests for get_bids_path_components function."""

    def test_extract_all_components(self) -> None:
        """Test extracting all path components."""
        path = "/data/bids/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii.gz"
        result = get_bids_path_components(path)

        assert result["subject_dir"] == "sub-01"
        assert result["session_dir"] == "ses-pre"
        assert result["datatype"] == "anat"

    def test_extract_without_session(self) -> None:
        """Test extracting components without session."""
        path = "/data/bids/sub-01/anat/sub-01_T1w.nii.gz"
        result = get_bids_path_components(path)

        assert result["subject_dir"] == "sub-01"
        assert result["session_dir"] == ""
        assert result["datatype"] == "anat"

    def test_extract_dwi_datatype(self) -> None:
        """Test extracting dwi datatype."""
        path = "/data/bids/sub-01/dwi/sub-01_dwi.nii.gz"
        result = get_bids_path_components(path)

        assert result["datatype"] == "dwi"

    def test_extract_func_datatype(self) -> None:
        """Test extracting func datatype."""
        path = "/data/bids/sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz"
        result = get_bids_path_components(path)

        assert result["datatype"] == "func"

    def test_missing_components(self) -> None:
        """Test that missing components return empty strings."""
        path = "/some/random/path/file.nii.gz"
        result = get_bids_path_components(path)

        assert result["subject_dir"] == ""
        assert result["session_dir"] == ""
        assert result["datatype"] == ""


class TestBidsEntityOrder:
    """Tests for BIDS_ENTITY_ORDER constant."""

    def test_entity_order_contains_common_entities(self) -> None:
        """Test that common BIDS entities are in the order list."""
        common_entities = ["sub", "ses", "task", "acq", "run", "echo"]

        for entity in common_entities:
            assert entity in BIDS_ENTITY_ORDER

    def test_sub_is_first(self) -> None:
        """Test that 'sub' is the first entity."""
        assert BIDS_ENTITY_ORDER[0] == "sub"

    def test_ses_is_second(self) -> None:
        """Test that 'ses' is the second entity."""
        assert BIDS_ENTITY_ORDER[1] == "ses"
