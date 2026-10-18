"""Export the final submission's inference-only weights into a self-contained release folder.

Reads the submission's ``inference-metadata.json`` lineage (25 fold-model
checkpoints with their SHA-256), verifies each training checkpoint, and writes
for each model a folder holding its ``config.yaml`` and a ``last.pt`` with
only what held-out/test inference reads (EMA weights, plus the YOLO
architecture). The crop refiner's fold models and the pickled rescorer are
copied beside them (with the frozen fold split and the submitted CSV when given),
and ``manifest.json`` lists every file with its hash, the
fold/family order of the ensemble and the submitted CSV hash, so
``infra/reproduce-submission.py`` can rebuild the predictions from this
folder and the supplied test images alone.

    python infra/export-weights.py --lineage METADATA.json --runs RUNS_DIR --refiner REFINER_RUN \
        --rescorer RESCORER.pkl --output WEIGHTS_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

FAMILIES = ("yolo11m-1536", "maskrcnn-1536", "yolo11m-2048", "yolo26m-1536", "dense-convnext-small")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tta_family(config: dict) -> str:
    """The infra/tta-inference.py family that loads this model."""
    if config["model"].startswith("dense"):
        return "dense"
    if config["model"].startswith("yolo"):
        return "yolo"
    return "maskrcnn"


def slim(checkpoint: dict) -> dict:
    """EMA weights (or the plain model when no EMA was kept), the YOLO architecture and the config."""
    keep = {"ema": checkpoint.get("ema", checkpoint["model"]), "config": checkpoint.get("config")}
    if "architecture" in checkpoint:
        keep["architecture"] = checkpoint["architecture"]
    return keep


def copy_rescorer(source: Path, target: Path):
    """The pickled rescorer, with its training-table provenance cut to run folder and file name (no host paths)."""
    saved = pickle.loads(source.read_bytes())
    saved["features"] = ["/".join(Path(path).parts[-2:]) for path in saved.get("features", [])]
    target.write_bytes(pickle.dumps(saved))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lineage", type=Path, required=True, help="The submission's inference-metadata.json")
    parser.add_argument("--runs", type=Path, required=True, help="Directory holding the training run folders")
    parser.add_argument("--refiner", type=Path, required=True, help="Refiner run folder with fold-*.pt")
    parser.add_argument("--rescorer", type=Path, required=True)
    parser.add_argument("--splits", type=Path, help="Folder with the frozen folds.csv and folds-metadata.json")
    parser.add_argument("--submission", type=Path, help="The submitted predictions.csv, kept as submission.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.lineage.read_text(encoding="utf-8"))
    ensemble = metadata["ensemble"]
    lineage = metadata["checkpoint_lineage"]
    folds = len(ensemble["fold_dirs"])
    if len(lineage) != folds * len(FAMILIES):
        raise ValueError("Expected one checkpoint per fold and family")
    args.output.mkdir(parents=True, exist_ok=True)
    models, files = [], []
    for number, item in enumerate(lineage):
        run = re.split(r"[\\/]runs[\\/]", item["tta"])[1].replace("\\", "/").split("/")[0]
        source = args.runs / run
        checkpoint_path = source / "last.pt"
        if digest(checkpoint_path) != item["weights_sha256"]:
            raise ValueError(f"{run}: training checkpoint differs from the submission lineage")
        config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
        target = args.output / run
        target.mkdir(exist_ok=True)
        shutil.copy2(source / "config.yaml", target / "config.yaml")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        torch.save(slim(checkpoint), target / "last.pt")
        del checkpoint
        models.append({"run": run, "fold": number // len(FAMILIES), "family": FAMILIES[number % len(FAMILIES)],
                       "tta_family": tta_family(config), "training_checkpoint_sha256": item["weights_sha256"]})
        files += [target / "config.yaml", target / "last.pt"]
        print(f"{run}: {(target / 'last.pt').stat().st_size / 2**20:.0f} MiB", flush=True)
    refiner = args.output / "refiner"
    refiner.mkdir(exist_ok=True)
    for path in sorted(args.refiner.glob("fold-*.pt")):
        shutil.copy2(path, refiner / path.name)
        files.append(refiner / path.name)
    if len(list(refiner.glob("fold-*.pt"))) != folds:
        raise ValueError("Expected one refiner model per fold")
    copy_rescorer(args.rescorer, args.output / "rescorer-hgb.pkl")
    files.append(args.output / "rescorer-hgb.pkl")
    if args.splits:
        (args.output / "splits").mkdir(exist_ok=True)
        for name in ("folds.csv", "folds-metadata.json"):
            shutil.copy2(args.splits / name, args.output / "splits" / name)
            files.append(args.output / "splits" / name)
    if args.submission:
        if digest(args.submission) != metadata["csv_sha256"]:
            raise ValueError("The submission CSV differs from the lineage's hash")
        shutil.copy2(args.submission, args.output / "submission.csv")
        files.append(args.output / "submission.csv")
    manifest = {
        "submission_csv_sha256": metadata["csv_sha256"], "threshold": ensemble["threshold"],
        "level1_setting": ensemble["level1_setting"], "views": ensemble["views"], "families": list(FAMILIES),
        "models": models, "refiner": {"column": "refined_rle_50", "minimum_area": 150},
        "files": [{"path": path.relative_to(args.output).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}
                  for path in files],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(files)} files, {sum(item['bytes'] for item in manifest['files']) / 2**30:.2f} GiB -> {args.output}")


if __name__ == "__main__":
    main()
