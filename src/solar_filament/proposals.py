"""Measure detector proposal coverage before investing in mask refinement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solar_filament.coco import CocoIndex
from solar_filament.storage import annotations_path, artifact_path


def box_iou(gt: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if not len(gt) or not len(predicted):
        return np.zeros((len(gt), len(predicted)), dtype=np.float64)
    left = np.maximum(gt[:, None, :2], predicted[None, :, :2])
    right = np.minimum(gt[:, None, 2:], predicted[None, :, 2:])
    intersection = np.maximum(right - left, 0).prod(axis=2)
    gt_area = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(axis=1)
    pred_area = np.maximum(predicted[:, 2:] - predicted[:, :2], 0).prod(axis=1)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def audit_proposals(root: Path, proposal_directory: Path, output: Path, fold: int = 0) -> dict:
    proposal_directory = artifact_path(proposal_directory, root)
    output = artifact_path(output, root)
    output.mkdir(parents=True, exist_ok=True)
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data/splits/folds.csv")
    names = sorted(folds.loc[folds.fold == fold, "filename"].tolist())
    thresholds = [0.1, 0.3, 0.5, 0.7]
    rows = []
    counts = {threshold: {"proposals": 0, "covered_gt": 0, "total_gt": 0} for threshold in thresholds}
    for name in names:
        path = proposal_directory / (Path(name).stem + ".json")
        if not path.is_file():
            raise ValueError(f"Missing proposal file: {path}")
        proposals = json.loads(path.read_text(encoding="utf-8"))
        for threshold in thresholds:
            selected = [item for item in proposals if item["score"] >= threshold]
            boxes = np.asarray([item["bbox_xyxy"] for item in selected], dtype=float).reshape(-1, 4)
            counts[threshold]["proposals"] += len(selected)
            for record in index.images_by_filename[name]:
                annotations = index.annotations_by_image[record["id"]]
                gt = np.asarray([item["bbox"] for item in annotations], dtype=float).reshape(-1, 4)
                gt[:, 2:] += gt[:, :2]
                overlap = box_iou(gt, boxes)
                best = overlap.max(axis=1) if len(boxes) else np.zeros(len(gt))
                counts[threshold]["total_gt"] += len(gt)
                counts[threshold]["covered_gt"] += int((best > 0.5).sum())
                for number, annotation in enumerate(annotations):
                    rows.append(
                        {
                            "filename": name,
                            "annotator_image": record["id"],
                            "annotation_id": annotation["id"],
                            "threshold": threshold,
                            "area": annotation["area"],
                            "best_box_iou": float(best[number]),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "proposal-per-object.csv", index=False, lineterminator="\n")
    result = {
        "fold": fold,
        "physical_images": len(names),
        "thresholds": [
            {
                "score_threshold": threshold,
                "proposals_per_image": value["proposals"] / len(names),
                "box_recall_at_iou_gt_05": value["covered_gt"] / value["total_gt"],
                "covered_gt": value["covered_gt"],
                "total_gt": value["total_gt"],
            }
            for threshold, value in counts.items()
        ],
        "note": "Bounding-box coverage measures detection geometry. It does not prove that a crop refiner will improve end-to-end PQ.",
    }
    (output / "proposal-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
