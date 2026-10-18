"""Two-level test ensemble: per-fold family fusion + rescoring, then fusion across folds.

Level 1 repeats, for each fold's models, exactly the fusion and features used on
that fold's held-out images, and scores the owned instances with the rescorer
fitted on all out-of-fold rows. Level 2 clusters the fold-level instances by
mask IoU, averages their probabilities over all folds (a fold that missed the
instance contributes zero), keeps pixels drawn by at least half of the members,
assigns pixel ownership and applies the out-of-fold threshold.

    python infra/ensemble-test.py --fold-dirs "Y0,M0" "Y1,M1" ... --rescorer MODEL.pkl --output DIR

With --level1-dir, each fold's level-1 instance table is written there (or read
back when present), so extra per-instance features such as the CNN verifier's
probability can be added between the two levels; --level1-only stops after
writing the tables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from solar_filament.fusion import fuse, own_pixels
from solar_filament.storage import dataset_root, storage_root
from solar_filament.submission import validate_predictions, write_predictions

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fold_instances(arguments):
    """Level 1 for one image and one fold: fused, owned instances with features."""
    directories, stem, setting, views = arguments
    fusion = load("fuse-candidates")
    source_count = len(directories) * len(views)
    _, _, features = fusion.evaluate_image((directories, stem, [setting], source_count, views, "rle", setting, None))
    return stem, features


def combine(arguments):
    stem, rows, folds, cluster_iou, pixel_vote, minimum_area, threshold = arguments
    candidates = [{"score": row["probability"], "source": row["fold"], "family": "fold",
                   "rle": {"size": [2048, 2048], "counts": row["rle"].encode("ascii")}} for row in rows]
    fused = fuse(candidates, folds, cluster_iou, pixel_vote)
    owned = own_pixels(fused, minimum_area, 50000)
    return stem, [instance["rle"] for instance in owned if instance["score"] >= threshold], len(owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dirs", nargs="+", required=True,
                        help="One comma-separated list of TTA test directories per fold, families in OOF order")
    parser.add_argument("--rescorer", type=Path, required=True)
    parser.add_argument("--cluster-iou", type=float, default=0.5)
    parser.add_argument("--pixel-vote", type=float, default=0.5)
    parser.add_argument("--minimum-area", type=int, default=150)
    parser.add_argument("--maximum-detections", type=int, default=150)
    parser.add_argument("--threshold", type=float, help="Override the rescorer's OOF threshold")
    parser.add_argument("--views", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--level1-dir", type=Path, help="Write, or read back, per-fold level-1 instance tables")
    parser.add_argument("--level1-only", action="store_true", help="Stop after writing the level-1 tables")
    parser.add_argument("--verifier-weight", type=float, default=0.0,
                        help="Blend weight of the level-1 tables' verifier column with the rescorer probability")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = storage_root()
    saved = pickle.loads(args.rescorer.read_bytes())
    threshold = args.threshold if args.threshold is not None else saved["threshold"]
    stems = sorted(path.stem for path in (dataset_root(root) / "test/test_images").glob("*.jpeg"))
    setting = (args.cluster_iou, args.pixel_vote, args.minimum_area, args.maximum_detections)
    rows = []
    for fold, listing in enumerate(args.fold_dirs):
        table = args.level1_dir / f"level1-f{fold}.csv" if args.level1_dir else None
        if table is not None and table.exists():
            frame = pd.read_csv(table)
        else:
            directories = [Path(item) / "candidates" for item in listing.split(",")]
            jobs = [(directories, stem, setting, args.views) for stem in stems]
            with ProcessPoolExecutor(args.workers) as pool:
                frame = pd.DataFrame([row for _, features in pool.map(fold_instances, jobs, chunksize=4) for row in features])
            if table is not None:
                table.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(table, index=False)
        if args.level1_only:
            print(f"fold {fold}: {len(frame)} instances written to {table}", flush=True)
            continue
        frame["probability"] = saved["rescorer"].predict(frame)
        if args.verifier_weight:
            frame["probability"] = (1 - args.verifier_weight) * frame.probability + args.verifier_weight * frame.verifier
        frame["fold"] = fold
        rows.append(frame[["stem", "fold", "probability", "rle"]])
        print(f"fold {fold}: {len(frame)} instances, {int((frame.probability >= threshold).sum())} above threshold", flush=True)
    if args.level1_only:
        return
    table = pd.concat(rows, ignore_index=True)
    grouped = {stem: group.to_dict("records") for stem, group in table.groupby("stem")}
    jobs = [(stem, grouped.get(stem, []), len(args.fold_dirs), args.cluster_iou, args.pixel_vote, args.minimum_area, threshold)
            for stem in stems]
    with ProcessPoolExecutor(args.workers) as pool:
        predictions = {stem: rles for stem, rles, _ in pool.map(combine, jobs, chunksize=4) if rles}
    args.output.mkdir(parents=True, exist_ok=True)
    csv = args.output / "predictions.csv"
    write_predictions(csv, predictions)
    validation = validate_predictions(predictions, set(stems))
    record = {
        "status": "complete", "physical_images": len(stems), "subset": "test",
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "fold_dirs": args.fold_dirs, "rescorer": str(args.rescorer), "rescorer_model": saved["model"],
        "threshold": threshold, "level1_setting": dict(zip(("cluster_iou", "pixel_vote", "minimum_area", "maximum_detections"), setting)),
        "views": args.views, "validation": validation, "created_utc": datetime.now(timezone.utc).isoformat(),
        **({"level1_dir": str(args.level1_dir)} if args.level1_dir else {}),
        **({"verifier_weight": args.verifier_weight} if args.verifier_weight else {}),
    }
    (args.output / "ensemble.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
