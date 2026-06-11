"""End-to-end qmri processing pipelines.

Pipelines combine the pure signal models in :mod:`qmri` with the file handling
in :mod:`qmri.io` to provide ready-to-run, file-in / file-out workflows.

Example:
    ```python
    from qmri.pipelines.thermometry import run_multiecho_thermometry

    temperature_map, report = run_multiecho_thermometry(
        multiecho_files=["echoes.nii.gz"],
        segmentation_file="labels.nii.gz",
        echo_times_files=["echo_times.txt"],
        magnetic_field_tesla=3.0,
    )
    ```
"""

from qmri.pipelines.perfusion.asl import (
    ASLQuantificationReport,
    run_asl_quantification,
)
from qmri.pipelines.thermometry.multiecho import (
    MultiEchoThermometryReport,
    run_multiecho_thermometry,
)
from qmri.pipelines.transfer.mtr import MTRReport, run_mtr

__version__ = "0.1.0"

__all__ = [
    "ASLQuantificationReport",
    "MTRReport",
    "MultiEchoThermometryReport",
    "run_asl_quantification",
    "run_mtr",
    "run_multiecho_thermometry",
]
