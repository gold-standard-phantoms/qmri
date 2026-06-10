"""Tests for NIFTI I/O utilities."""

from pathlib import Path

import numpy as np
import pytest
from qmri.io import (
    NiftiImage,
    get_affine,
    get_voxel_size,
    load_nifti,
    load_nifti_image,
    save_nifti,
)


class TestLoadNifti:
    """Tests for load_nifti function."""

    def test_load_nifti_returns_data_and_header(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that load_nifti returns data array and header."""
        # Create a test NIFTI file
        test_data = rng.random((10, 10, 10)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        # Load and verify
        data, header = load_nifti(test_file)

        assert isinstance(data, np.ndarray)
        assert data.shape == (10, 10, 10)
        assert data.dtype == np.float64
        np.testing.assert_array_almost_equal(data, test_data)

    def test_load_nifti_preserves_shape(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that loading preserves the original data shape."""
        shapes = [(5, 5, 5), (10, 20, 30), (64, 64, 30, 4)]

        for shape in shapes:
            test_data = rng.random(shape).astype(np.float64)
            test_file = tmp_path / f"test_{'x'.join(map(str, shape))}.nii.gz"
            save_nifti(test_data, test_file)

            data, _ = load_nifti(test_file)
            assert data.shape == shape

    def test_load_nifti_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError, match="NIFTI file not found"):
            load_nifti("/nonexistent/path/file.nii.gz")

    def test_load_nifti_accepts_string_path(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that load_nifti accepts string paths."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        # Pass as string
        data, _ = load_nifti(str(test_file))
        assert data.shape == (5, 5, 5)


class TestLoadNiftiImage:
    """Tests for load_nifti_image function."""

    def test_returns_nifti_image_dataclass(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that load_nifti_image returns a NiftiImage dataclass."""
        test_data = rng.random((10, 10, 10)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        img = load_nifti_image(test_file)

        assert isinstance(img, NiftiImage)
        assert hasattr(img, "data")
        assert hasattr(img, "header")
        assert hasattr(img, "affine")

    def test_nifti_image_data_matches(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that NiftiImage contains correct data."""
        test_data = rng.random((8, 8, 8)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        img = load_nifti_image(test_file)

        np.testing.assert_array_almost_equal(img.data, test_data)

    def test_nifti_image_affine_shape(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that affine has correct shape."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        img = load_nifti_image(test_file)

        assert img.affine.shape == (4, 4)


class TestSaveNifti:
    """Tests for save_nifti function."""

    def test_save_nifti_creates_file(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that save_nifti creates a file."""
        test_data = rng.random((10, 10, 10)).astype(np.float64)
        test_file = tmp_path / "output.nii.gz"

        save_nifti(test_data, test_file)

        assert test_file.exists()

    def test_save_nifti_roundtrip(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that save/load roundtrip preserves data."""
        original_data = rng.random((12, 12, 12)).astype(np.float64)
        test_file = tmp_path / "roundtrip.nii.gz"

        save_nifti(original_data, test_file)
        loaded_data, _ = load_nifti(test_file)

        np.testing.assert_array_almost_equal(loaded_data, original_data)

    def test_save_nifti_with_custom_affine(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test saving with a custom affine matrix."""
        test_data = rng.random((10, 10, 10)).astype(np.float64)
        custom_affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm isotropic
        test_file = tmp_path / "custom_affine.nii.gz"

        save_nifti(test_data, test_file, affine=custom_affine)

        img = load_nifti_image(test_file)
        np.testing.assert_array_almost_equal(img.affine, custom_affine)

    def test_save_nifti_with_header(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test saving with an existing header."""
        # Create initial file
        initial_data = rng.random((10, 10, 10)).astype(np.float64)
        initial_file = tmp_path / "initial.nii.gz"
        custom_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        save_nifti(initial_data, initial_file, affine=custom_affine)

        # Load and modify
        _, header = load_nifti(initial_file)
        modified_data = initial_data * 2
        modified_file = tmp_path / "modified.nii.gz"

        # Save with original header
        save_nifti(modified_data, modified_file, header=header)

        # Verify
        loaded_img = load_nifti_image(modified_file)
        np.testing.assert_array_almost_equal(loaded_img.data, modified_data)

    def test_save_nifti_invalid_affine_shape(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that invalid affine shape raises ValueError."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        bad_affine = np.eye(3)  # Wrong shape
        test_file = tmp_path / "bad_affine.nii.gz"

        with pytest.raises(ValueError, match="Affine must be 4x4"):
            save_nifti(test_data, test_file, affine=bad_affine)

    def test_save_nifti_creates_parent_directories(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that save_nifti creates parent directories if needed."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        nested_path = tmp_path / "nested" / "dirs" / "output.nii.gz"

        save_nifti(test_data, nested_path)

        assert nested_path.exists()

    def test_save_nifti_uncompressed(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test saving uncompressed .nii files."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        test_file = tmp_path / "uncompressed.nii"

        save_nifti(test_data, test_file)

        assert test_file.exists()
        loaded_data, _ = load_nifti(test_file)
        np.testing.assert_array_almost_equal(loaded_data, test_data)


class TestGetVoxelSize:
    """Tests for get_voxel_size function."""

    def test_get_voxel_size_default(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test voxel size extraction with default affine."""
        test_data = rng.random((10, 10, 10)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)  # Default is identity affine

        _, header = load_nifti(test_file)
        voxel_size = get_voxel_size(header)

        assert len(voxel_size) == 3
        assert all(isinstance(v, float) for v in voxel_size)

    def test_get_voxel_size_custom(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test voxel size extraction with custom affine."""
        test_data = rng.random((10, 10, 10)).astype(np.float64)
        custom_affine = np.diag([2.0, 3.0, 4.0, 1.0])
        test_file = tmp_path / "custom.nii.gz"
        save_nifti(test_data, test_file, affine=custom_affine)

        _, header = load_nifti(test_file)
        voxel_size = get_voxel_size(header)

        assert voxel_size == pytest.approx((2.0, 3.0, 4.0))


class TestGetAffine:
    """Tests for get_affine function."""

    def test_get_affine_returns_4x4(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that get_affine returns a 4x4 matrix."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        _, header = load_nifti(test_file)
        affine = get_affine(header)

        assert affine.shape == (4, 4)
        assert affine.dtype == np.float64

    def test_get_affine_matches_saved(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that extracted affine matches what was saved."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        custom_affine = np.array(
            [
                [1.5, 0.0, 0.0, -10.0],
                [0.0, 1.5, 0.0, -20.0],
                [0.0, 0.0, 2.0, -30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        test_file = tmp_path / "affine_test.nii.gz"
        save_nifti(test_data, test_file, affine=custom_affine)

        _, header = load_nifti(test_file)
        extracted_affine = get_affine(header)

        np.testing.assert_array_almost_equal(extracted_affine, custom_affine)


class TestNiftiImageDataclass:
    """Tests for NiftiImage dataclass."""

    def test_nifti_image_is_frozen(
        self, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Test that NiftiImage is immutable (frozen)."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        img = load_nifti_image(test_file)

        with pytest.raises(AttributeError):
            img.data = np.zeros((5, 5, 5))  # type: ignore[misc]

    def test_nifti_image_fields(self, tmp_path: Path, rng: np.random.Generator) -> None:
        """Test that NiftiImage has all expected fields."""
        test_data = rng.random((5, 5, 5)).astype(np.float64)
        test_file = tmp_path / "test.nii.gz"
        save_nifti(test_data, test_file)

        img = load_nifti_image(test_file)

        # Check all fields exist and have correct types
        assert isinstance(img.data, np.ndarray)
        assert img.data.dtype == np.float64
        assert isinstance(img.affine, np.ndarray)
        assert img.affine.shape == (4, 4)
