"""Training-only medoid targets chosen by symmetric instance-mask agreement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solar_filament.coco import CocoIndex
from solar_filament.metrics import overlap_matrix, reference_counts
from solar_filament.storage import annotations_path, artifact_path


def agreement_medoids(identifiers: list[str], similarities: np.ndarray) -> list[str]:
    """Retain all equally central records instead of inventing a stronger tie winner."""
    if similarities.shape != (len(identifiers), len(identifiers)):
        raise ValueError("Agreement matrix must contain one row and column per annotator")
    if (
        not identifiers
        or not np.isfinite(similarities).all()
        or not np.allclose(similarities, similarities.T)
    ):
        raise ValueError("Expected finite symmetric annotator agreement")
    if len(identifiers) == 1:
        return identifiers.copy()
    means = (similarities.sum(axis=1) - np.diag(similarities)) / (len(identifiers) - 1)
    candidates = np.flatnonzero(np.isclose(means, means.max(), atol=1e-12, rtol=0))
    return sorted(identifiers[index] for index in candidates)


def agreement_medoid(identifiers: list[str], similarities: np.ndarray) -> str:
    return agreement_medoids(identifiers, similarities)[0]


def build_medoid_manifest(root: Path, output: Path) -> dict:
    output = artifact_path(output, root)
    index = CocoIndex(annotations_path(root))
    observations = {}
    for filename, records in sorted(index.images_by_filename.items()):
        identifiers = sorted(record["id"] for record in records)
        rles = [index.record_rles(identifier) for identifier in identifiers]
        similarities = np.eye(len(records))
        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                iou = overlap_matrix(rles[left], rles[right])
                forward, reverse = (
                    reference_counts(iou).scores()["pq"],
                    reference_counts(iou.T).scores()["pq"],
                )
                similarities[left, right] = similarities[right, left] = 0.5 * (forward + reverse)
        observations[filename] = {
            "selected_record": agreement_medoid(identifiers, similarities),
            "records": identifiers,
            "pairwise_pq": similarities.tolist(),
        }
    result = {
        "policy": "medoid",
        "physical_images": len(observations),
        "tie_break": "lexicographically smallest annotator-image ID; deterministic and independent of validation results",
        "scope": "Each image's policy uses only that image's supplied training annotations. Training must select policy records only after excluding the held-out fold; evaluation always retains every annotator.",
        "observations": observations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
