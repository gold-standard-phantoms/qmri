"""Guard tests for the qmri PEP 420 namespace package.

The ``qmri`` namespace is spread across every workspace member's
``src/qmri`` directory. This only works while *no* member ships a
``qmri/__init__.py`` — a single one would pin ``qmri.__path__`` to that
member and silently shadow every other package. These tests fail loudly
if that ever regresses.
"""

import importlib.util
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent

SUBPACKAGE_TO_DISTRIBUTION = {
    "io": "qmri-io",
    "cli": "qmri-cli",
    "dro": "qmri-dro",
    "viz": "qmri-viz",
    "pipelines": "qmri-pipelines",
    "thermometry": "qmri",
    "diffusion": "qmri",
    "relaxometry": "qmri",
}


def test_no_member_ships_a_namespace_root_init() -> None:
    """No workspace member may contain src/qmri/__init__.py."""
    offenders = sorted(WORKSPACE_ROOT.glob("packages/*/src/qmri/__init__.py"))
    assert not offenders, (
        f"qmri must remain a PEP 420 namespace package, but found: "
        f"{[str(p.relative_to(WORKSPACE_ROOT)) for p in offenders]}"
    )


def test_namespace_merges_all_workspace_members() -> None:
    """qmri.__path__ must contain every installed member's src/qmri dir."""
    import qmri

    path_entries = [Path(p).resolve() for p in qmri.__path__]
    assert len(path_entries) > 1, (
        f"qmri.__path__ has a single entry ({path_entries}); namespace "
        "merging is broken — a qmri/__init__.py has probably been added."
    )


def test_each_subpackage_resolves_to_its_own_distribution() -> None:
    """Every qmri.<sub> must import from its own package directory."""
    for subpackage, distribution in SUBPACKAGE_TO_DISTRIBUTION.items():
        spec = importlib.util.find_spec(f"qmri.{subpackage}")
        assert spec is not None and spec.origin is not None, (
            f"qmri.{subpackage} is not importable"
        )
        expected = (
            WORKSPACE_ROOT / "packages" / distribution / "src" / "qmri"
        ).resolve()
        origin = Path(spec.origin).resolve()
        assert expected in origin.parents, (
            f"qmri.{subpackage} resolved to {origin}, expected it inside "
            f"{expected} (the {distribution} package)"
        )
