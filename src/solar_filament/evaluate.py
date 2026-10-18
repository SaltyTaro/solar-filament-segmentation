"""Evaluate each physical observation against all independent annotator records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from solar_filament.coco import CocoIndex
from solar_filament.metrics import (
    PQCounts,
    one_to_one_counts,
    overlap_diagnostics,
    overlap_matrix,
    reference_counts,
)
from solar_filament.storage import annotations_path
from solar_filament.submission import load_predictions, validate_predictions


def evaluate(root: Path, prediction_file: Path, output: Path, fold: int | None = None) -> dict:
    index = CocoIndex(annotations_path(root))
    if fold is None:
        filenames = sorted(index.images_by_filename)
    else:
        folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
        filenames = sorted(folds.loc[folds["fold"] == fold, "filename"].tolist())
        if not filenames:
            raise ValueError(f"No images in fold {fold}")
    stems = {Path(name).stem for name in filenames}
    predictions = load_predictions(prediction_file)
    validation = validate_predictions(predictions, stems)
    reference_total = PQCounts()
    one_to_one_total = PQCounts()
    rows = []
    object_rows = []
    diagnostics = []
    for filename in tqdm(filenames, desc="Score observations", unit="image"):
        stem = Path(filename).stem
        pred_rles = predictions.get(stem, [])
        for record in index.images_by_filename[filename]:
            gt_rles = index.record_rles(record["id"])
            iou = overlap_matrix(gt_rles, pred_rles)
            reference = reference_counts(iou)
            conventional = one_to_one_counts(iou)
            reference_total += reference
            one_to_one_total += conventional
            detail = overlap_diagnostics(iou)
            gt_union = mask_utils.merge(gt_rles) if gt_rles else None
            pred_union = mask_utils.merge(pred_rles) if pred_rles else None
            semantic_iou = (
                float(mask_utils.iou([gt_union], [pred_union], [0])[0, 0])
                if gt_union is not None and pred_union is not None
                else float(gt_union is None and pred_union is None)
            )
            rows.append(
                {
                    "filename": filename,
                    "annotator_image": record["id"],
                    "site": stem[-2:],
                    "year": int(stem[:4]),
                    "n_gt": len(gt_rles),
                    "n_pred": len(pred_rles),
                    **reference.scores(),
                    "one_to_one_pq": conventional.scores()["pq"],
                    "semantic_iou": semantic_iou,
                    "semantic_dice": 2 * semantic_iou / (1 + semantic_iou),
                    "gt_overlapping_multiple_predictions": detail["gt_overlapping_multiple_predictions"],
                    "predictions_overlapping_multiple_gt": detail["predictions_overlapping_multiple_gt"],
                }
            )
            diagnostics.append({"annotator_image": record["id"], **detail})
            for number, annotation in enumerate(index.annotations_by_image[record["id"]]):
                x, y, width, height = annotation["bbox"]
                best_iou = float(iou[number].max()) if pred_rles else 0.0
                object_rows.append(
                    {
                        "filename": filename,
                        "annotator_image": record["id"],
                        "annotation_id": annotation["id"],
                        "site": stem[-2:],
                        "area": annotation["area"],
                        "maximum_bbox_dimension": max(width, height),
                        "normalized_disk_radius": (
                            ((x + width / 2 - 1024) ** 2 + (y + height / 2 - 1024) ** 2) ** 0.5
                        )
                        / 1024,
                        "best_iou": best_iou,
                        "matched": best_iou > 0.5,
                        "overlapping_predictions": int((iou[number] > 0).sum()),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per-record.csv", index=False, lineterminator="\n")
    pd.DataFrame(object_rows).to_csv(output / "per-object.csv", index=False, lineterminator="\n")
    (output / "overlap-diagnostics.json").write_text(json.dumps(diagnostics) + "\n", encoding="utf-8")
    summary = {
        "prediction_file": str(prediction_file),
        "fold": fold,
        "physical_images": len(filenames),
        "annotator_records": len(rows),
        "reference_pooled": reference_total.scores(),
        "one_to_one_pooled": one_to_one_total.scores(),
        "submission_validation": validation,
        "semantic_union": {
            "mean_iou": float(pd.DataFrame(rows)["semantic_iou"].mean()),
            "mean_dice": float(pd.DataFrame(rows)["semantic_dice"].mean()),
        },
        "notes": ["IoU must be strictly greater than 0.5", "Scores pool counts across annotator records"],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
