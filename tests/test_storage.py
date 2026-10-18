from pathlib import Path

import pytest

from solar_filament.storage import artifact_path


def test_artifacts_resolve_inside_project_storage(tmp_path):
    root = tmp_path / "project"
    assert artifact_path(root / "runs" / "baseline", root) == root.resolve() / "runs" / "baseline"


def test_artifact_paths_cannot_escape_project_storage(tmp_path):
    root = tmp_path / "project"
    with pytest.raises(ValueError, match="must stay under"):
        artifact_path(root / ".." / "other-project" / "weights.pt", root)
    with pytest.raises(ValueError, match="must stay under"):
        artifact_path(Path(tmp_path) / "weights.pt", root)
