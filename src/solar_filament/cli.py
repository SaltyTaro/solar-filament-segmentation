"""Command line entry points for the reproducible project pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solar_filament.storage import artifact_path, configure_caches, storage_root


def main() -> None:
    parser = argparse.ArgumentParser(prog="solar-filament")
    parser.add_argument(
        "--storage-root", help="Project storage root; defaults to SOLAR_STORAGE_ROOT, else storage/ in the checkout"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Download and verify official Kaggle data")
    ingest.add_argument("--verify-only", action="store_true")
    ingest.add_argument("--workers", type=int, default=4)
    folds = commands.add_parser("folds", help="Create deterministic grouped folds")
    folds.add_argument("--folds", type=int, default=5)
    folds.add_argument("--seed", type=int, default=20261003)
    validate = commands.add_parser(
        "validate-submission", help="Validate a CSV against the official test manifest"
    )
    validate.add_argument("predictions", type=Path)
    evaluate = commands.add_parser("evaluate", help="Score held-out predictions with pooled per-annotator PQ")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--fold", type=int)
    evaluate.add_argument("--output", type=Path, required=True)
    classical = commands.add_parser("classical", help="Generate an inexpensive native-resolution reference")
    classical.add_argument("--fold", type=int, default=0)
    classical.add_argument("--threshold", type=float, default=8.0)
    classical.add_argument("--minimum-area", type=int, default=150)
    classical.add_argument("--limit", type=int)
    classical.add_argument("--output", type=Path, required=True)
    commands.add_parser("prepare", help="Audit annotation masks and cache per-annotator union targets")
    fingerprints = commands.add_parser("fingerprints", help="Find image candidates for duplicate review")
    fingerprints.add_argument("--workers", type=int, default=4)
    train = commands.add_parser("train-unet", help="Train the native-resolution grayscale U-Net baseline")
    train.add_argument("--config", type=Path, default=Path("configs/unet-baseline.yaml"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--smoke-steps", type=int, default=0)
    tune = commands.add_parser("tune", help="Compare postprocessing on cached held-out probabilities")
    tune.add_argument("probabilities", type=Path)
    tune.add_argument("--fold", type=int, default=0)
    tune.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Summarize held-out errors and date-grouped PQ uncertainty")
    report.add_argument("evaluation", type=Path)
    detector = commands.add_parser("train-detector", help="Train whole-disk Mask R-CNN instance baseline")
    detector.add_argument("--config", type=Path, default=Path("configs/detector-baseline.yaml"))
    detector.add_argument("--output", type=Path, required=True)
    detector.add_argument("--smoke-steps", type=int, default=0)
    detector.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict-unet", help="Reproduce validation or supplied-test predictions")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--subset", choices=["test", "validation"], default="test")
    predict.add_argument("--fold", type=int, default=0)
    predict.add_argument("--postprocess", type=Path)
    predict.add_argument("--limit", type=int)
    detector_predict = commands.add_parser(
        "predict-detector", help="Reproduce detector predictions from declared weights and supplied JPEGs"
    )
    detector_predict.add_argument("checkpoint", type=Path)
    detector_predict.add_argument("--output", type=Path, required=True)
    detector_predict.add_argument("--subset", choices=["test", "validation"], default="test")
    detector_predict.add_argument("--fold", type=int, default=0)
    detector_predict.add_argument("--limit", type=int)
    detector_predict.add_argument("--postprocess", type=Path)
    detector_cache = commands.add_parser(
        "cache-detector", help="Cache compact native detector mask probabilities"
    )
    detector_cache.add_argument("checkpoint", type=Path)
    detector_cache.add_argument("--output", type=Path, required=True)
    detector_cache.add_argument("--fold", type=int, default=0)
    detector_tune = commands.add_parser(
        "tune-detector", help="Tune detector postprocessing from a verified cache"
    )
    detector_tune.add_argument("cache", type=Path)
    detector_tune.add_argument("--output", type=Path, required=True)
    proposals = commands.add_parser("audit-proposals", help="Measure detector box coverage before refinement")
    proposals.add_argument("proposals", type=Path)
    proposals.add_argument("--output", type=Path, required=True)
    proposals.add_argument("--fold", type=int, default=0)
    hybrid = commands.add_parser("hybrid", help="Test detector boxes with native semantic crop boundaries")
    hybrid.add_argument("probabilities", type=Path)
    hybrid.add_argument("proposals", type=Path)
    hybrid.add_argument("--fold", type=int, default=0)
    hybrid.add_argument("--output", type=Path, required=True)
    refiner = commands.add_parser(
        "train-refiner", help="Train a box-conditioned filament instance crop model"
    )
    refiner.add_argument("proposals", type=Path)
    refiner.add_argument("--config", type=Path, default=Path("configs/refiner-pilot.yaml"))
    refiner.add_argument("--output", type=Path, required=True)
    refiner.add_argument("--smoke-steps", type=int, default=0)
    refiner.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = storage_root(args.storage_root)
    configure_caches(root)
    if hasattr(args, "output"):
        args.output = artifact_path(args.output, root)
    if args.command == "ingest":
        from solar_filament.ingest import ingest as run_ingest

        result = run_ingest(root, args.verify_only, args.workers)
        print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    elif args.command == "folds":
        from solar_filament.folds import create_folds

        print(json.dumps(create_folds(root, args.folds, args.seed), indent=2))
    elif args.command == "validate-submission":
        from solar_filament.submission import load_predictions, validate_predictions

        manifest = json.loads((root / "data" / "manifests" / "kaggle-files.json").read_text(encoding="utf-8"))
        stems = {Path(row["name"]).stem for row in manifest if "/test_images/" in row["name"]}
        print(json.dumps(validate_predictions(load_predictions(args.predictions), stems), indent=2))
    elif args.command == "evaluate":
        from solar_filament.evaluate import evaluate as run_evaluate

        print(json.dumps(run_evaluate(root, args.predictions, args.output, args.fold), indent=2))
    elif args.command == "classical":
        from solar_filament.classical import predict_classical

        print(
            json.dumps(
                predict_classical(
                    root, args.output, args.fold, args.threshold, args.minimum_area, args.limit
                ),
                indent=2,
            )
        )
    elif args.command == "prepare":
        from solar_filament.prepare import prepare_targets

        print(json.dumps(prepare_targets(root), indent=2))
    elif args.command == "fingerprints":
        from solar_filament.prepare import image_fingerprints

        print(json.dumps(image_fingerprints(root, args.workers), indent=2))
    elif args.command == "train-unet":
        from solar_filament.train_unet import train as run_train

        print(json.dumps(run_train(root, args.config, args.output, args.resume, args.smoke_steps), indent=2))
    elif args.command == "tune":
        from solar_filament.tune import tune_probabilities

        print(json.dumps(tune_probabilities(root, args.probabilities, args.output, args.fold), indent=2))
    elif args.command == "report":
        from solar_filament.report import build_report

        print(json.dumps(build_report(root, args.evaluation), indent=2))
    elif args.command == "train-detector":
        from solar_filament.detection import train_detector

        print(
            json.dumps(
                train_detector(root, args.config, args.output, args.smoke_steps, args.resume), indent=2
            )
        )

    elif args.command == "cache-detector":
        from solar_filament.detector_cache import cache_detector

        print(json.dumps(cache_detector(root, args.checkpoint, args.output, args.fold), indent=2))
    elif args.command == "tune-detector":
        from solar_filament.detector_cache import tune_detector_cache

        print(json.dumps(tune_detector_cache(root, args.cache, args.output), indent=2))
    elif args.command == "predict-detector":
        from solar_filament.predict import predict_detector

        print(
            json.dumps(
                predict_detector(
                    root, args.checkpoint, args.output, args.subset, args.fold, args.limit, args.postprocess
                ),
                indent=2,
            )
        )
    elif args.command == "train-refiner":
        from solar_filament.refiner import train_refiner

        print(
            json.dumps(
                train_refiner(root, args.config, args.proposals, args.output, args.smoke_steps, args.resume),
                indent=2,
            )
        )
    elif args.command == "hybrid":
        from solar_filament.hybrid import evaluate_hybrid

        print(
            json.dumps(
                evaluate_hybrid(root, args.probabilities, args.proposals, args.output, args.fold), indent=2
            )
        )
    elif args.command == "audit-proposals":
        from solar_filament.proposals import audit_proposals

        print(json.dumps(audit_proposals(root, args.proposals, args.output, args.fold), indent=2))
    elif args.command == "predict-unet":
        from solar_filament.predict import predict_unet

        print(
            json.dumps(
                predict_unet(
                    root, args.checkpoint, args.output, args.subset, args.fold, args.postprocess, args.limit
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
