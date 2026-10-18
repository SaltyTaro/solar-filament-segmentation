"""Reproduce the final submission from the released weights and the supplied test images alone.

The weights folder is the published release (``infra/export-weights.py``):
one folder per fold model (``config.yaml`` + inference-only ``last.pt``),
``refiner/fold-*.pt``, ``rescorer-hgb.pkl``, the fold split, the submitted
``submission.csv`` and ``manifest.json``. It must lie
under the storage root (for example ``$SOLAR_STORAGE_ROOT/weights/final``),
next to the supplied competition data in ``data/raw/MAGFiLO_1.0_Kaggle_2026``.
Each stage is resumable and runs the same scripts that built the submission:

1. 8-view dihedral TTA of every fold model on the 180 test images (GPU);
2. level 1: per-fold fusion of the five families and instance features;
3. crop refinement of the level-1 masks with the five refiner folds (GPU);
4. adoption: refined masks re-owned in fused order;
5. level 2: nested-OOF rescorer, fusion across the five fold ensembles,
   pixel ownership and the out-of-fold threshold -> ``predictions.csv``.

The result is validated and compared with the submitted CSV's hash, then matched
mask by mask against the released ``submission.csv`` (or ``--reference``).

    python infra/reproduce-submission.py --weights WEIGHTS_DIR --output OUT [--reference SUBMITTED.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from solar_filament.metrics import overlap_matrix
from solar_filament.storage import artifact_path, storage_root
from solar_filament.submission import load_predictions

HERE = Path(__file__).resolve().parent


def run(arguments: list[str]):
    print("+", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], check=True, cwd=HERE.parent)


def compare(candidate: Path, reference: Path) -> dict:
    """Masks of two submissions matched per image at IoU > .5."""
    ours, theirs = load_predictions(candidate), load_predictions(reference)
    matched, ious = 0, []
    for stem in sorted(set(ours) | set(theirs)):
        a, b = ours.get(stem, []), theirs.get(stem, [])
        if a and b:
            matrix = overlap_matrix(b, a)
            best = matrix.max(axis=1)
            matched += int((best > 0.5).sum())
            ious += best[best > 0.5].tolist()
    return {"reproduced_masks": sum(map(len, ours.values())), "reference_masks": sum(map(len, theirs.values())),
            "matched_at_iou_0.5": matched, "matched_median_iou": float(np.median(ious)) if ious else 0.0,
            "matched_mean_iou": float(np.mean(ious)) if ious else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        help="The submitted predictions.csv to compare against; defaults to the release's submission.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Accepted for worker compatibility; stages always resume")
    args = parser.parse_args()
    root = storage_root()
    weights, output = artifact_path(args.weights, root), artifact_path(args.output, root)
    manifest = json.loads((weights / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if hashlib.sha256((weights / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Released file differs from its manifest: {item['path']}")
    output.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        target = output / "tta" / model["run"]
        if not (target / "metadata.json").exists():
            run([str(HERE / "tta-inference.py"), "--family", model["tta_family"], "--run", str(weights / model["run"]),
                 "--subset", "test", "--views", *map(str, manifest["views"]), "--output", str(target)])
    folds = sorted({model["fold"] for model in manifest["models"]})
    fold_dirs = [",".join(str(output / "tta" / model["run"]) for model in manifest["models"] if model["fold"] == fold)
                 for fold in folds]
    setting = manifest["level1_setting"]
    level1 = ["--cluster-iou", str(setting["cluster_iou"]), "--pixel-vote", str(setting["pixel_vote"]),
              "--minimum-area", str(setting["minimum_area"]), "--maximum-detections", str(setting["maximum_detections"]),
              "--views", *map(str, manifest["views"]), "--workers", str(args.workers)]
    rescorer = str(weights / "rescorer-hgb.pkl")
    tables = [output / "level1" / f"level1-f{fold}.csv" for fold in folds]
    if not all(table.exists() for table in tables):
        run([str(HERE / "ensemble-test.py"), "--fold-dirs", *fold_dirs, "--rescorer", rescorer, *level1,
             "--level1-dir", str(output / "level1"), "--level1-only", "--output", str(output / "level1")])
    refined = output / "refined"
    if not all((refined / table.name).exists() for table in tables):
        run([str(HERE / "refine-masks.py"), "tables", "--run", str(weights / "refiner"), "--tables", *map(str, tables),
             "--output", str(refined), "--workers", str(args.workers)])
    adopted = output / "adopted"
    if not all((adopted / table.name).exists() for table in tables):
        run([str(HERE / "refine-masks.py"), "adopt", "--source", str(refined), "--pattern", "level1-f*.csv",
             "--column", manifest["refiner"]["column"], "--minimum-area", str(manifest["refiner"]["minimum_area"]),
             "--output", str(adopted), "--workers", str(args.workers)])
    ensemble = output / "ensemble"
    run([str(HERE / "ensemble-test.py"), "--fold-dirs", *fold_dirs, "--rescorer", rescorer, *level1,
         "--level1-dir", str(adopted), "--threshold", str(manifest["threshold"]), "--output", str(ensemble)])
    csv = ensemble / "predictions.csv"
    record = {"predictions": str(csv), "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
              "submission_csv_sha256": manifest["submission_csv_sha256"]}
    record["identical_to_submission"] = record["csv_sha256"] == record["submission_csv_sha256"]
    reference = args.reference or (weights / "submission.csv" if (weights / "submission.csv").exists() else None)
    if reference:
        record["comparison"] = compare(csv, reference)
    (output / "reproduction.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
