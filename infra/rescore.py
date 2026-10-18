"""Cross-fitted rescoring of fused instances with exact pooled PQ.

A gradient-boosted regressor predicts the fraction of annotators who would
match each fused instance from fusion and image features only (the annotator
count is a training weight, never an input). Each fold is scored by a model
trained on the other folds. The decision threshold is chosen on the other
folds too (nested), so the reported out-of-fold PQ involves no selection on the
scored images. With --save-model, a rescorer fitted on all OOF rows and the
pooled OOF threshold are saved for the test ensemble.

    python infra/rescore.py OOF_FEATURES.csv [...] --output OUT.json [--save-model MODEL.pkl]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from solar_filament.coco import CocoIndex
from solar_filament.metrics import PQCounts
from solar_filament.rescoring import Rescorer, feature_columns
from solar_filament.storage import annotations_path, storage_root
from solar_filament.submission import write_predictions

THRESHOLDS = np.round(np.arange(0.05, 0.81, 0.025), 3)


def pooled_counts(frame: pd.DataFrame, keep: np.ndarray, gt_counts: dict) -> PQCounts:
    """Reference-scorer counts for the kept instances, from stored IoU > .5 pairs."""
    total = PQCounts()
    kept = frame[keep]
    by_stem = dict(tuple(kept.groupby("stem")))
    for stem, counts in gt_counts.items():
        rows = by_stem.get(stem)
        for record, n_gt in enumerate(counts):
            matched_gt, tp, iou_sum, unmatched = set(), 0, 0.0, 0
            if rows is not None:
                for parsed in rows["parsed"]:
                    hits = parsed.get(record, [])
                    if not hits:
                        unmatched += 1
                    for gt, iou in hits:
                        matched_gt.add(gt)
                        tp += 1
                        iou_sum += iou
            total += PQCounts(tp=tp, fp=unmatched, fn=n_gt - len(matched_gt), sum_iou=iou_sum)
    return total


def best_threshold(frame, probabilities, gt_counts):
    scores = [(pooled_counts(frame, probabilities >= threshold, gt_counts).scores()["pq"], threshold) for threshold in THRESHOLDS]
    return max(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-model", type=Path, help="Fit on all OOF rows and pickle the rescorer and threshold")
    parser.add_argument("--save-predictions", type=Path,
                        help="Write each fold's nested-selected instances as oof-f<fold>.csv in scorer format")
    parser.add_argument("--model", choices=("logistic", "hgb-monotone", "hgb"), default="logistic")
    args = parser.parse_args()
    root = storage_root()
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data/splits/folds.csv")
    fold_of = dict(zip(folds.image_stem, folds.fold))
    frame = pd.concat([pd.read_csv(path) for path in args.features], ignore_index=True)
    frame["fold"] = frame.stem.map(fold_of)
    frame["parsed"] = [
        {record: [(gt, iou) for r, gt, iou in json.loads(pairs) if r == record] for record in {r for r, _, _ in json.loads(pairs)}}
        for pairs in frame.pairs
    ]
    covered = sorted(frame.fold.unique())
    gt_counts = {
        stem: [len(index.annotations_by_image[record["id"]]) for record in index.images_by_filename[stem + ".jpeg"]]
        for stem in folds.loc[folds.fold.isin(covered), "image_stem"]
    }
    columns = feature_columns(frame, args.model)
    out_of_fold = np.zeros(len(frame))
    nested = PQCounts()
    per_fold = {}
    for fold in covered:
        train, held = frame.fold != fold, frame.fold == fold
        model = Rescorer(args.model, columns, fold).fit(frame[train])
        out_of_fold[held.to_numpy()] = model.predict(frame[held])
    for fold in covered:
        held = (frame.fold == fold).to_numpy()
        other_counts = {stem: counts for stem, counts in gt_counts.items() if fold_of[stem] != fold}
        own_counts = {stem: counts for stem, counts in gt_counts.items() if fold_of[stem] == fold}
        _, threshold = best_threshold(frame[~held], out_of_fold[~held], other_counts) if len(covered) > 1 else (0, 0.3)
        counts = pooled_counts(frame[held], out_of_fold[held] >= threshold, own_counts)
        nested += counts
        per_fold[int(fold)] = {"threshold_from_other_folds": float(threshold), **counts.scores()}
        if args.save_predictions:
            kept = frame[held & (out_of_fold >= threshold)].sort_values(["stem", "column"])
            write_predictions(args.save_predictions / f"oof-f{fold}.csv",
                              {stem: [{"size": [2048, 2048], "counts": rle} for rle in group.rle] for stem, group in kept.groupby("stem")})
    pooled_best, pooled_threshold = best_threshold(frame, out_of_fold, gt_counts)
    baseline = max((pooled_counts(frame, frame.score.to_numpy() >= threshold, gt_counts).scores()["pq"], threshold)
                   for threshold in THRESHOLDS)
    summary = {
        "features": [str(path) for path in args.features], "folds": [int(fold) for fold in covered],
        "instances": len(frame), "model": args.model, "feature_columns": columns,
        "fused_score_baseline": {"pq": baseline[0], "threshold": float(baseline[1])},
        "rescored_pooled_best": {"pq": pooled_best, "threshold": float(pooled_threshold)},
        "rescored_nested": nested.scores(), "per_fold": per_fold,
    }
    print(json.dumps({key: summary[key] for key in ("fused_score_baseline", "rescored_pooled_best", "rescored_nested")}, indent=2))
    if args.save_model:
        model = Rescorer(args.model, columns, 99).fit(frame)
        with args.save_model.open("wb") as handle:
            pickle.dump({"rescorer": model, "threshold": float(pooled_threshold), "columns": columns,
                         "model": args.model, "features": [str(path) for path in args.features]}, handle)
        summary["saved_model"] = str(args.save_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
