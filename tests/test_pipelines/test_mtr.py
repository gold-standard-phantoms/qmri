"""End-to-end tests for the magnetisation transfer ratio (MTR) pipeline.

These port the behaviour previously covered by ``mrimagetools``'
``MtrQuantificationFilter`` and ``mtr_pipeline`` tests, rewritten against the
qmri pipeline API. The numerical model itself is unit-tested in
``tests/test_transfer/test_mtr.py``; here we exercise the file-in/file-out
workflow, the two input modes, output writing, and input validation.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from qmri.io import load_nifti_image, save_nifti
from qmri.pipelines.transfer import MTRReport, run_mtr

# A spatially-varying ground-truth MTR map (percentage units) used throughout.
# With an unsaturated signal of 1, a saturated signal of (1 - MTR/100) yields
# exactly this map, so the pipeline should recover it without error.
_MTR_TRUTH = np.linspace(0.0, 50.0, num=32, dtype=np.float64).reshape(4, 4, 2)


def _write_pair(directory: Path) -> tuple[Path, Path]:
    """Write separate unsaturated/saturated images; return (sat, nosat) paths."""
    affine = np.eye(4)
    unsaturated = np.ones_like(_MTR_TRUTH)
    saturated = unsaturated * (1.0 - _MTR_TRUTH / 100.0)
    sat_path = directory / "sat.nii.gz"
    nosat_path = directory / "nosat.nii.gz"
    save_nifti(saturated, sat_path, affine=affine)
    save_nifti(unsaturated, nosat_path, affine=affine)
    return sat_path, nosat_path


def _write_combined(directory: Path) -> Path:
    """Write a single 4D file ordered [unsaturated, saturated]; return its path."""
    affine = np.eye(4)
    unsaturated = np.ones_like(_MTR_TRUTH)
    saturated = unsaturated * (1.0 - _MTR_TRUTH / 100.0)
    combined = np.stack([unsaturated, saturated], axis=-1)
    combined_path = directory / "mt.nii.gz"
    save_nifti(combined, combined_path, affine=affine)
    return combined_path


class TestRunMtrSeparate:
    """Behaviour with two separate input images."""

    def test_recovers_mtr_and_writes_outputs(self, tmp_path: Path) -> None:
        """The pipeline recovers the known MTR map and writes map + report."""
        sat_path, nosat_path = _write_pair(tmp_path)

        mtr_image, report = run_mtr(
            sat_path, nosat_path, output_dir=tmp_path
        )

        assert isinstance(report, MTRReport)
        assert report.mode == "separate"
        np.testing.assert_allclose(mtr_image.data, _MTR_TRUTH)

        map_path = tmp_path / "sat_mtr_map.nii.gz"
        report_path = tmp_path / "sat_report.json"
        assert report.output_file == map_path
        assert map_path.exists()
        assert report_path.exists()

        saved = load_nifti_image(map_path)
        np.testing.assert_allclose(saved.data, _MTR_TRUTH)

        payload = json.loads(report_path.read_text())
        assert payload["mode"] == "separate"
        assert payload["mtr_max"] == pytest.approx(50.0)

    def test_report_statistics(self, tmp_path: Path) -> None:
        """Summary statistics in the report match the ground truth."""
        sat_path, nosat_path = _write_pair(tmp_path)

        _, report = run_mtr(sat_path, nosat_path, save_outputs=False)

        assert report.output_file is None
        assert report.n_valid_voxels == _MTR_TRUTH.size
        assert report.mtr_mean == pytest.approx(float(np.mean(_MTR_TRUTH)))
        assert report.mtr_min == pytest.approx(0.0)
        assert report.mtr_max == pytest.approx(50.0)

    def test_shape_mismatch_raises(self, tmp_path: Path) -> None:
        """Differently-shaped images raise a clear error."""
        sat_path, _ = _write_pair(tmp_path)
        wrong = tmp_path / "wrong.nii.gz"
        save_nifti(np.ones((3, 3, 3)), wrong, affine=np.eye(4))

        with pytest.raises(ValueError, match="same shape"):
            run_mtr(sat_path, wrong, save_outputs=False)

    def test_affine_mismatch_raises(self, tmp_path: Path) -> None:
        """Images that are not co-located (different affine) raise."""
        sat_path, _ = _write_pair(tmp_path)
        wrong_affine = tmp_path / "wrong_affine.nii.gz"
        save_nifti(np.ones_like(_MTR_TRUTH), wrong_affine, affine=3.0 * np.eye(4))

        with pytest.raises(ValueError, match="same affine"):
            run_mtr(sat_path, wrong_affine, save_outputs=False)


class TestRunMtrCombined:
    """Behaviour with a single combined 4D input image."""

    def test_recovers_mtr_from_combined(self, tmp_path: Path) -> None:
        """A combined [unsaturated, saturated] file yields the same MTR map."""
        combined_path = _write_combined(tmp_path)

        mtr_image, report = run_mtr(combined_path, save_outputs=False)

        assert report.mode == "combined"
        assert report.input_files == [combined_path]
        assert mtr_image.data.shape == _MTR_TRUTH.shape
        np.testing.assert_allclose(mtr_image.data, _MTR_TRUTH)

    def test_wrong_volume_count_raises(self, tmp_path: Path) -> None:
        """A combined file without exactly two volumes raises."""
        three_vols = tmp_path / "three.nii.gz"
        save_nifti(np.ones((4, 4, 2, 3)), three_vols, affine=np.eye(4))

        with pytest.raises(ValueError, match="exactly two volumes"):
            run_mtr(three_vols, save_outputs=False)

    def test_three_dimensional_combined_raises(self, tmp_path: Path) -> None:
        """A 3D file in combined mode raises (must be 4D)."""
        three_d = tmp_path / "three_d.nii.gz"
        save_nifti(np.ones((4, 4, 2)), three_d, affine=np.eye(4))

        with pytest.raises(ValueError, match="must be 4D"):
            run_mtr(three_d, save_outputs=False)


def test_zero_signal_voxels_are_zero_and_excluded(tmp_path: Path) -> None:
    """Voxels with zero unsaturated signal get MTR 0 and are excluded from stats."""
    affine = np.eye(4)
    unsaturated = np.ones((2, 2, 2), dtype=np.float64)
    unsaturated[0, 0, 0] = 0.0  # undefined MTR here
    saturated = np.full((2, 2, 2), 0.7, dtype=np.float64)
    sat_path = tmp_path / "sat.nii.gz"
    nosat_path = tmp_path / "nosat.nii.gz"
    save_nifti(saturated, sat_path, affine=affine)
    save_nifti(unsaturated, nosat_path, affine=affine)

    mtr_image, report = run_mtr(sat_path, nosat_path, save_outputs=False)

    assert mtr_image.data[0, 0, 0] == 0.0
    # 8 voxels total, one excluded.
    assert report.n_valid_voxels == 7
    # The 7 valid voxels all have MTR = 100 * (1 - 0.7) = 30.
    assert report.mtr_mean == pytest.approx(30.0)
