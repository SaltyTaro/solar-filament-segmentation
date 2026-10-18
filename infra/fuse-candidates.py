"""Fuse TTA candidates across views and models, then score pooled PQ over a settings grid.

Each candidate directory is one model; each of its views is a separate source.
Clustering runs once per cluster IoU, fusion once per pixel vote and ownership
once per minimum area; every score threshold is a prefix of the owned list.
With --features, the best setting's owned instances are written with their
fusion features and per-annotator match labels for the learned rescorer.

    python infra/fuse-candidates.py DIR [DIR ...] --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from solar_filament.coco import CocoIndex
from solar_filament.features import instance_features
from solar_filament.fusion import fuse, own_pixels
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.storage import annotations_path, dataset_root, storage_root
from solar_filament.submission import validate_predictions, write_predictions

_INDEX = None


def index() -> CocoIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = CocoIndex(annotations_path(storage_root()))
    return _INDEX


def load_image_candidates(directories: list[Path], stem: str, views: list[int] | None, mask_key: str,
                          scales: list[float] | None = None):
    """Scores are multiplied by the model's scale (capped at 1) to align family calibration."""
    candidates, sources = [], set()
    for number, directory in enumerate(directories):
        scale = scales[number] if scales else 1.0
        family = directory.parent.name if directory.name == "candidates" else directory.name
        for record in json.loads((directory / f"{stem}.json").read_text(encoding="utf-8")):
            if views is not None and record["view"] not in views:
                continue
            rle = record.get(mask_key) or record["rle"]
            source = f"{number}:{record['view']}"
            sources.add(source)
            candidates.append({"score": min(1.0, record["score"] * scale), "source": source, "family": f"m{number}",
                               "rle": {"size": rle["size"], "counts": rle["counts"].encode("ascii")},
                               "model": family})
    return candidates


def evaluate_image(arguments):
    directories, stem, settings, source_count, views, mask_key, want_features, scales = arguments
    candidates = load_image_candidates(directories, stem, views, mask_key, scales)
    records = index().images_by_filename.get(stem + ".jpeg")
    results, features = {}, []
    by_iou = {}
    for (cluster_iou, pixel_vote, minimum_area, maximum_detections) in settings:
        key = (cluster_iou, pixel_vote)
        if key not in by_iou:
            per_family = source_count // len(directories)
            by_iou[key] = fuse(candidates, source_count, cluster_iou, pixel_vote,
                               {f"m{number}": per_family for number in range(len(directories))})
        fused = by_iou[key][:maximum_detections]
        owned = own_pixels(fused, minimum_area, 50000)
        scores = np.asarray([instance["score"] for instance in owned])
        rles = [instance["rle"] for instance in owned]
        matrices = [overlap_matrix(index().record_rles(record["id"]), rles) for record in records] if records else []
        results[(cluster_iou, pixel_vote, minimum_area, maximum_detections)] = (scores, matrices)
        if want_features and (cluster_iou, pixel_vote, minimum_area, maximum_detections) == want_features:
            subset = "train/train_images" if records else "test/test_images"
            image = cv2.imread(str(dataset_root(storage_root()) / subset / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            for column, instance in enumerate(owned):
                matched = [bool((matrix[:, column] > 0.5).any()) for matrix in matrices]
                best = [float(matrix[:, column].max()) if matrix.shape[0] else 0.0 for matrix in matrices]
                pairs = [[record, int(gt), float(matrix[gt, column])] for record, matrix in enumerate(matrices)
                         for gt in np.flatnonzero(matrix[:, column] > 0.5)]
                features.append({
                    "stem": stem, "column": column, "annotators": len(matrices), "image_instances": len(owned),
                    "image_rank": column, "matched_fraction": float(np.mean(matched)) if matched else 0.0,
                    "best_iou_mean": float(np.mean(best)) if best else 0.0, "pairs": json.dumps(pairs),
                    **{name: value for name, value in instance.items() if name not in ("rle", "families", "area")},
                    **instance_features(image, instance["rle"]),
                    "rle": instance["rle"]["counts"].decode("ascii"),
                })
    return stem, results, features


def fixed_image(arguments):
    directories, stem, setting, source_count, views, mask_key, scales = arguments
    cluster_iou, pixel_vote, minimum_area, maximum_detections, threshold = setting
    fused = fuse(load_image_candidates(directories, stem, views, mask_key, scales), source_count, cluster_iou, pixel_vote)
    owned = own_pixels(fused[:maximum_detections], minimum_area, 50000)
    return stem, [instance["rle"] for instance in owned if instance["score"] >= threshold]


def apply_fixed(args, directories, stems, source_count):
    """Write predictions for unlabeled images (the test set) from one setting chosen on held-out folds."""
    values = [args.cluster_iou, args.pixel_vote, args.areas, args.maximum_detections]
    if any(len(value) != 1 for value in values) or args.apply_threshold is None:
        raise ValueError("--write-predictions needs exactly one value per setting and --apply-threshold")
    setting = (*[value[0] for value in values], args.apply_threshold)
    jobs = [(directories, stem, setting, source_count, args.views, args.mask_key, args.scales) for stem in stems]
    with ProcessPoolExecutor(args.workers) as pool:
        predictions = {stem: rles for stem, rles in pool.map(fixed_image, jobs, chunksize=2) if rles}
    write_predictions(args.write_predictions, predictions)
    validation = validate_predictions(predictions, set(stems))
    record = {"directories": [str(path) for path in directories], "setting": dict(zip(
        ("cluster_iou", "pixel_vote", "minimum_area", "maximum_detections", "score_threshold"), setting)),
        "source_count": source_count, "scales": args.scales, "validation": validation}
    args.write_predictions.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


def pooled_grid(results_by_image, thresholds):
    grid = {}
    for _, results in results_by_image:
        for setting, (scores, matrices) in results.items():
            for threshold in thresholds:
                columns = np.flatnonzero(scores >= threshold)
                total = grid.get((*setting, threshold), PQCounts())
                for matrix in matrices:
                    total += reference_counts(matrix[:, columns])
                grid[(*setting, threshold)] = total
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+", help="TTA output directories (with candidates/)")
    parser.add_argument("--views", type=int, nargs="+")
    parser.add_argument("--mask-key", choices=("rle", "rle_low"), default="rle")
    parser.add_argument("--scales", type=float, nargs="+", help="Per-directory score multipliers")
    parser.add_argument("--cluster-iou", type=float, nargs="+", default=[0.5])
    parser.add_argument("--pixel-vote", type=float, nargs="+", default=[0.5])
    parser.add_argument("--areas", type=int, nargs="+", default=[150])
    parser.add_argument("--maximum-detections", type=int, nargs="+", default=[60])
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="Write rescoring features for the best setting")
    parser.add_argument("--write-predictions", type=Path, help="Apply one fixed setting (no scoring) and write a CSV")
    parser.add_argument("--apply-threshold", type=float, help="Fused-score threshold for --write-predictions")
    args = parser.parse_args()
    directories = [directory / "candidates" if (directory / "candidates").is_dir() else directory for directory in args.directories]
    stems = sorted(path.stem for path in directories[0].glob("*.json"))
    for directory in directories[1:]:
        if sorted(path.stem for path in directory.glob("*.json")) != stems:
            raise ValueError(f"{directory} covers different images")
    views = args.views if args.views else list(range(8))
    source_count = len(directories) * len(views)
    if args.write_predictions:
        apply_fixed(args, directories, stems, source_count)
        return
    settings = list(product(args.cluster_iou, args.pixel_vote, args.areas, args.maximum_detections))
    jobs = [(directories, stem, settings, source_count, args.views, args.mask_key, None, args.scales) for stem in stems]
    with ProcessPoolExecutor(args.workers) as pool:
        results_by_image = [(stem, results) for stem, results, _ in pool.map(evaluate_image, jobs, chunksize=2)]
    grid = pooled_grid(results_by_image, args.thresholds)
    rows = [{"cluster_iou": key[0], "pixel_vote": key[1], "minimum_area": key[2], "maximum_detections": key[3],
             "score_threshold": key[4], **counts.scores()} for key, counts in sorted(grid.items())]
    best = max(rows, key=lambda row: row["pq"])
    summary = {"directories": [str(path) for path in directories], "views": views, "mask_key": args.mask_key,
               "physical_images": len(stems), "source_count": source_count, "best": best, "grid": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(stems)} images, {source_count} sources; best pq={best['pq']:.5f} "
          + " ".join(f"{name}={best[name]}" for name in ("cluster_iou", "pixel_vote", "minimum_area", "maximum_detections", "score_threshold"))
          + f" tp={best['tp']} fp={best['fp']} fn={best['fn']}")
    for row in rows:
        if all(row[name] == best[name] for name in ("cluster_iou", "pixel_vote", "minimum_area", "maximum_detections")):
            print(f"  t={row['score_threshold']:.2f} pq={row['pq']:.5f} sq={row['sq']:.4f} rq={row['rq']:.4f} fp={row['fp']} fn={row['fn']}")
    chosen = (best["cluster_iou"], best["pixel_vote"], best["minimum_area"], best["maximum_detections"])
    if args.features:
        jobs = [(directories, stem, [chosen], source_count, args.views, args.mask_key, chosen, args.scales) for stem in stems]
        with ProcessPoolExecutor(args.workers) as pool:
            outputs = list(pool.map(evaluate_image, jobs, chunksize=2))
        frame = pd.DataFrame([row for _, _, features in outputs for row in features])
        frame.to_csv(args.features, index=False, lineterminator="\n")
        print(f"features: {len(frame)} instances -> {args.features}")


if __name__ == "__main__":
    main()
