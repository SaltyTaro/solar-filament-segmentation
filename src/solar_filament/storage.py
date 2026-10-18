"""Project paths under one storage root: ``SOLAR_STORAGE_ROOT``, or ``storage/`` in the checkout."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_STORAGE = Path(__file__).resolve().parents[2] / "storage"
DATASET_NAME = "MAGFiLO_1.0_Kaggle_2026"
ANNOTATION_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def storage_root(override: str | Path | None = None) -> Path:
    root = Path(override or os.environ.get("SOLAR_STORAGE_ROOT") or DEFAULT_STORAGE).expanduser()
    if not root.is_absolute():
        raise ValueError("Storage root must be absolute.")
    return root


def dataset_root(root: Path) -> Path:
    return root / "data" / "raw" / DATASET_NAME


def annotations_path(root: Path) -> Path:
    return dataset_root(root) / "train" / ANNOTATION_NAME


def artifact_path(path: str | Path, root: Path) -> Path:
    """Resolve junctions and refuse output outside the dedicated project storage."""
    destination = Path(path).expanduser().resolve()
    project_storage = root.resolve()
    if not destination.is_relative_to(project_storage):
        raise ValueError(f"Project output must stay under {project_storage}: {destination}")
    return destination


def configure_caches(root: Path) -> None:
    """Keep downloads and plotting/model caches under project storage."""
    for variable, directory in {
        "HF_HOME": "huggingface",
        "TORCH_HOME": "torch",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        path = root / "cache" / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
