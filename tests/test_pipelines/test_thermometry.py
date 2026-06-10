"""End-to-end tests for the multi-echo thermometry pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
from qmri.io import load_nifti_image, save_nifti
from qmri.pipelines.thermometry import (
    MultiEchoThermometryReport,
    run_multiecho_thermometry,
)
from qmri.thermometry import multiecho

B0 = 3.0
# ImagingFrequency = gamma (MHz/T) * B0, used to populate JSON sidecars.
IMAGING_FREQUENCY_MHZ = 42.577_478_92 * B0


def _region_signal(temperature: float, echo_times: np.ndarray) -> np.ndarray:
    """Dual-resonance signal for a single temperature."""
    df = multiecho.calculate_df_from_temperature(temperature, B0)
    return multiecho.thermometry_signal_model(echo_times, 1.0, 0.5, 30.0, 40.0, df, 0.0)


def _write_dataset(
    directory: Path,
    region_temperatures: dict[int, float],
    echo_times: np.ndarray,
    *,
    sidecar: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    """Write a multi-echo image, segmentation and echo-times file to ``directory``.

    Returns the (image, segmentation, echo-times) paths.
    """
    shape = (4, 4, 2)
    n_echoes = echo_times.shape[0]
    signal = np.zeros((*shape, n_echoes), dtype=np.float64)
    segmentation = np.zeros(shape, dtype=np.int16)

    flat = list(np.ndindex(shape))
    labels = list(region_temperatures)
    chunk = len(flat) // len(labels)
    for label_index, label in enumerate(labels):
        region = _region_signal(region_temperatures[label], echo_times)
        for i, j, k in flat[label_index * chunk : (label_index + 1) * chunk]:
            segmentation[i, j, k] = label
            signal[i, j, k, :] = region

    affine = np.eye(4)
    image_path = directory / "echoes.nii.gz"
    seg_path = directory / "labels.nii.gz"
    te_path = directory / "echo_times.txt"
    save_nifti(signal, image_path, affine=affine)
    save_nifti(segmentation.astype(np.float64), seg_path, affine=affine)
    np.savetxt(te_path, echo_times)

    if sidecar is not None:
        sidecar_path = directory / "echoes.json"
        sidecar_path.write_text(json.dumps(sidecar))

    return image_path, seg_path, te_path


class TestRunMultiEchoThermometry:
    """End-to-end behaviour of run_multiecho_thermometry."""

    def test_writes_outputs_and_recovers_temperatures(self, tmp_path: Path) -> None:
        """The pipeline writes a map + report and recovers region temperatures."""
        echo_times = np.linspace(0.001, 0.024, 24)
        region_temperatures = {1: 20.0, 2: 40.0}
        image_path, seg_path, te_path = _write_dataset(
            tmp_path,
            region_temperatures,
            echo_times,
            sidecar={"MagneticFieldStrength": B0},
        )

        temperature_image, report = run_multiecho_thermometry(
            multiecho_files=[image_path],
            segmentation_file=seg_path,
            echo_times_files=[te_path],
            method="regionwise",
            output_dir=tmp_path,
        )

        assert isinstance(report, MultiEchoThermometryReport)
        assert report.magnetic_field_tesla == pytest.approx(B0)

        # Recovered region temperatures.
        recovered = {r.region_id: r.temperature for r in report.regions}
        assert set(recovered) == {1, 2}
        for label, temp in region_temperatures.items():
            assert recovered[label] == pytest.approx(temp, abs=0.5)

        # Outputs on disk.
        map_path = tmp_path / "echoes_temperature_map.nii.gz"
        report_path = tmp_path / "echoes_report.json"
        assert map_path.exists()
        assert report_path.exists()
        assert report.output_file == map_path

        saved_map = load_nifti_image(map_path)
        assert saved_map.data.shape == (4, 4, 2)
        np.testing.assert_allclose(saved_map.data, temperature_image.data)

        # The JSON report is valid and carries per-region results.
        payload = json.loads(report_path.read_text())
        assert payload["magnetic_field_tesla"] == pytest.approx(B0)
        assert len(payload["report"]) == 2

    def test_detects_field_from_imaging_frequency(self, tmp_path: Path) -> None:
        """B0 is derived from ImagingFrequency when no explicit value is given."""
        echo_times = np.linspace(0.001, 0.024, 24)
        image_path, seg_path, te_path = _write_dataset(
            tmp_path,
            {1: 25.0},
            echo_times,
            sidecar={"ImagingFrequency": IMAGING_FREQUENCY_MHZ},
        )

        _, report = run_multiecho_thermometry(
            multiecho_files=[image_path],
            segmentation_file=seg_path,
            echo_times_files=[te_path],
            save_outputs=False,
        )

        assert report.magnetic_field_tesla == pytest.approx(B0, rel=1e-4)

    def test_explicit_field_overrides_sidecar(self, tmp_path: Path) -> None:
        """An explicit magnetic_field_tesla takes precedence over the sidecar."""
        echo_times = np.linspace(0.001, 0.024, 24)
        image_path, seg_path, te_path = _write_dataset(
            tmp_path,
            {1: 25.0},
            echo_times,
            sidecar={"MagneticFieldStrength": 1.5},
        )

        _, report = run_multiecho_thermometry(
            multiecho_files=[image_path],
            segmentation_file=seg_path,
            echo_times_files=[te_path],
            magnetic_field_tesla=B0,
            save_outputs=False,
        )

        assert report.magnetic_field_tesla == pytest.approx(B0)

    def test_multiple_images_concatenate_and_sort_echoes(self, tmp_path: Path) -> None:
        """Echoes from multiple images are concatenated and sorted by echo time."""
        echo_block_1 = np.linspace(0.013, 0.024, 12)  # later echoes first
        echo_block_2 = np.linspace(0.001, 0.012, 12)
        region_temperatures = {1: 30.0}

        # Build two images that together form a full, contiguous echo train.
        full_echoes = np.concatenate([echo_block_1, echo_block_2])
        shape = (4, 4, 2)
        seg = np.ones(shape, dtype=np.float64)
        affine = np.eye(4)
        region = _region_signal(region_temperatures[1], full_echoes)
        signal_block_1 = np.broadcast_to(region[:12], (*shape, 12)).copy()
        signal_block_2 = np.broadcast_to(region[12:], (*shape, 12)).copy()

        img1 = tmp_path / "block1.nii.gz"
        img2 = tmp_path / "block2.nii.gz"
        seg_path = tmp_path / "labels.nii.gz"
        te1 = tmp_path / "te1.txt"
        te2 = tmp_path / "te2.txt"
        save_nifti(signal_block_1, img1, affine=affine)
        save_nifti(signal_block_2, img2, affine=affine)
        save_nifti(seg, seg_path, affine=affine)
        np.savetxt(te1, echo_block_1)
        np.savetxt(te2, echo_block_2)

        _, report = run_multiecho_thermometry(
            multiecho_files=[img1, img2],
            segmentation_file=seg_path,
            echo_times_files=[te1, te2],
            magnetic_field_tesla=B0,
            save_outputs=False,
        )

        # Echo times in the report are sorted ascending.
        assert report.echo_times == sorted(report.echo_times)
        assert report.regions[0].temperature == pytest.approx(30.0, abs=0.5)

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        """A missing field strength (no sidecar, no argument) raises."""
        echo_times = np.linspace(0.001, 0.024, 24)
        image_path, seg_path, te_path = _write_dataset(tmp_path, {1: 25.0}, echo_times)

        with pytest.raises(ValueError, match="Magnetic field strength"):
            run_multiecho_thermometry(
                multiecho_files=[image_path],
                segmentation_file=seg_path,
                echo_times_files=[te_path],
                save_outputs=False,
            )

    def test_image_echo_count_mismatch_raises(self, tmp_path: Path) -> None:
        """An echo-times file that disagrees with the image raises."""
        echo_times = np.linspace(0.001, 0.024, 24)
        image_path, seg_path, te_path = _write_dataset(tmp_path, {1: 25.0}, echo_times)
        # Overwrite echo-times file with the wrong length.
        np.savetxt(te_path, echo_times[:10])

        with pytest.raises(ValueError, match="echo times were provided"):
            run_multiecho_thermometry(
                multiecho_files=[image_path],
                segmentation_file=seg_path,
                echo_times_files=[te_path],
                magnetic_field_tesla=B0,
                save_outputs=False,
            )

    def test_mismatched_file_counts_raise(self, tmp_path: Path) -> None:
        """Differing numbers of images and echo-time files raise."""
        echo_times = np.linspace(0.001, 0.024, 24)
        image_path, seg_path, te_path = _write_dataset(tmp_path, {1: 25.0}, echo_times)

        with pytest.raises(ValueError, match="must match the number of echo-time"):
            run_multiecho_thermometry(
                multiecho_files=[image_path],
                segmentation_file=seg_path,
                echo_times_files=[te_path, te_path],
                magnetic_field_tesla=B0,
                save_outputs=False,
            )
