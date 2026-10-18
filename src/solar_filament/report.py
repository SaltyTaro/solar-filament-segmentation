"""Summarize pooled results and uncertainty with all annotators grouped together."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solar_filament.storage import artifact_path


def pooled_scores(frame: pd.DataFrame) -> dict:
    tp, fp, fn = (int(frame[column].sum()) for column in ["tp", "fp", "fn"])
    sum_iou = float(frame["sum_iou"].sum())
    denominator = tp + 0.5 * (fp + fn)
    return {
        "physical_images": int(frame.filename.nunique()),
        "annotator_records": len(frame),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pq": sum_iou / denominator if denominator else 0.0,
        "sq": sum_iou / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def bootstrap_pq(frame: pd.DataFrame, replicates: int = 2000, seed: int = 20261003) -> dict:
    frame = frame.copy()
    frame["capture_date"] = frame["filename"].str[:8]
    counts = frame.groupby("capture_date")[["tp", "fp", "fn", "sum_iou"]].sum().to_numpy()
    if not len(counts):
        raise ValueError("No date groups available for bootstrap")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for start in range(0, replicates, 200):
        number = min(200, replicates - start)
        draws = rng.integers(len(counts), size=(number, len(counts)))
        sums = counts[draws].sum(axis=1)
        denominators = sums[:, 0] + 0.5 * (sums[:, 1] + sums[:, 2])
        values[start : start + number] = np.divide(
            sums[:, 3], denominators, out=np.zeros(number), where=denominators > 0
        )
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "date_groups": len(counts),
        "replicates": replicates,
        "seed": seed,
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def build_report(root: Path, evaluation: Path) -> dict:
    evaluation = artifact_path(evaluation, root)
    records = pd.read_csv(evaluation / "per-record.csv")
    objects = pd.read_csv(evaluation / "per-object.csv")
    objects["size_bin"] = pd.cut(
        objects.area, [0, 1000, 5000, float("inf")], labels=["below_1000", "1000_to_5000", "above_5000"]
    )
    size_rows = []
    for name, group in objects.groupby("size_bin", observed=True):
        size_rows.append(
            {
                "size_bin": str(name),
                "annotation_count": len(group),
                "recall_at_iou_gt_05": float(group.matched.mean()),
                "mean_best_iou": float(group.best_iou.mean()),
            }
        )
    result = {
        "pooled": pooled_scores(records),
        "date_grouped_bootstrap": bootstrap_pq(records),
        "by_site": {str(site): pooled_scores(group) for site, group in records.groupby("site")},
        "by_year": {str(year): pooled_scores(group) for year, group in records.groupby("year")},
        "by_annotation_size": size_rows,
        "semantic_union": {
            "mean_iou": float(records.semantic_iou.mean()),
            "median_iou": float(records.semantic_iou.median()),
            "mean_dice": float(records.semantic_dice.mean()),
        },
        "instance_relationships": {
            "gt_overlapping_multiple_predictions": int(records.gt_overlapping_multiple_predictions.sum()),
            "predictions_overlapping_multiple_gt": int(records.predictions_overlapping_multiple_gt.sum()),
        },
        "notes": [
            "Pooled PQ uses all annotator-image records.",
            "Bootstrap resamples UTC capture dates and retains all annotator records of each observation.",
            "This interval describes variation in the fixed evaluated model and does not remove selection bias.",
        ],
    }
    (evaluation / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
