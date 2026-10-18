"""Pooled five-fold out-of-fold evaluation of a family ensemble, plus rescorer comparison.

For each fold, the listed families' held-out TTA candidates are fused with one
fixed setting; fused-score counts are pooled across folds per threshold, and
features are written for the cross-fitted rescorers. Rescorer variants are
compared by nested out-of-fold PQ and the best is saved for the test ensemble.

    python infra/phase2-oof.py --fold-dirs "Y0,M0" "Y1,M1" ... --output DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from solar_filament.metrics import PQCounts

HERE = Path(__file__).resolve().parent


def run(arguments: list[str]) -> str:
    print("+", " ".join(arguments[:3]), "...", flush=True)
    result = subprocess.run([sys.executable, *arguments], check=True, capture_output=True, text=True, cwd=HERE.parent)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dirs", nargs="+", required=True, help="Comma-separated TTA validation dirs per fold")
    parser.add_argument("--cluster-iou", default="0.5")
    parser.add_argument("--pixel-vote", default="0.5")
    parser.add_argument("--maximum-detections", default="150")
    parser.add_argument("--models", nargs="+", default=["logistic", "hgb-monotone", "hgb"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = [f"{value / 100:.2f}" for value in range(20, 66, 5)]
    pooled, features = {}, []
    for fold, listing in enumerate(args.fold_dirs):
        grid = args.output / f"fusion-f{fold}.json"
        table = args.output / f"features-f{fold}.csv"
        if not table.exists():
            run([str(HERE / "fuse-candidates.py"), *listing.split(","), "--cluster-iou", args.cluster_iou,
                 "--pixel-vote", args.pixel_vote, "--maximum-detections", args.maximum_detections,
                 "--thresholds", *thresholds, "--output", str(grid), "--features", str(table)])
        features.append(str(table))
        record = json.loads(grid.read_text(encoding="utf-8"))
        best = record["best"]
        print(f"fold {fold}: {record['physical_images']} images, fused best pq={best['pq']:.5f} at t={best['score_threshold']}", flush=True)
        for row in record["grid"]:
            key = row["score_threshold"]
            pooled[key] = pooled.get(key, PQCounts()) + PQCounts(row["tp"], row["fp"], row["fn"], row["sum_iou"])
    fused = {threshold: counts.scores() for threshold, counts in sorted(pooled.items())}
    fused_best = max(fused.items(), key=lambda item: item[1]["pq"])
    print(f"pooled fused score: best pq={fused_best[1]['pq']:.5f} at t={fused_best[0]} (threshold selected on all folds)")
    summary = {"fold_dirs": args.fold_dirs, "fused_pooled": fused, "fused_best": {"threshold": fused_best[0], **fused_best[1]},
               "rescorers": {}}
    for model in args.models:
        out = args.output / f"rescore-{model}.json"
        run([str(HERE / "rescore.py"), *features, "--model", model, "--output", str(out),
             "--save-model", str(args.output / f"rescorer-{model}.pkl")])
        record = json.loads(out.read_text(encoding="utf-8"))
        summary["rescorers"][model] = {"nested": record["rescored_nested"], "pooled_best": record["rescored_pooled_best"]}
        print(f"rescorer {model}: nested pq={record['rescored_nested']['pq']:.5f}, pooled best {record['rescored_pooled_best']['pq']:.5f} at t={record['rescored_pooled_best']['threshold']}")
    best_model = max(summary["rescorers"], key=lambda name: summary["rescorers"][name]["nested"]["pq"])
    summary["selected_rescorer"] = best_model
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"selected rescorer: {best_model}")


if __name__ == "__main__":
    main()
