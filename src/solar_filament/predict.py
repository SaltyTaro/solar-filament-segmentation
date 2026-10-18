"""Reproduce U-Net predictions on supplied images from a declared checkpoint."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from solar_filament.checkpoint_archive import archive_manifest, restored_checkpoint_bytes
from solar_filament.checkpoint_delta import checkpoint_source, delta_path, restored_bytes
from solar_filament.models import ResNetUNet
from solar_filament.postprocess import probability_instances
from solar_filament.preprocessing import DetectorPreprocessing
from solar_filament.storage import artifact_path, dataset_root
from solar_filament.submission import validate_predictions, write_predictions
from solar_filament.train_unet import predict_image


def file_digest(path: Path) -> str:
    if not path.is_file() and archive_manifest(path).is_file():
        _, record = restored_checkpoint_bytes(path)
        return record["original_sha256"]
    if not path.is_file() and delta_path(path).is_file():
        _, record = restored_bytes(path)
        return record["original_sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def predict_unet(
    root: Path,
    checkpoint_path: Path,
    output: Path,
    subset: str = "test",
    fold: int = 0,
    postprocess_config: Path | None = None,
    limit: int | None = None,
) -> dict:
    checkpoint_path = artifact_path(checkpoint_path, root)
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for declared U-Net inference")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    checkpoint = torch.load(checkpoint_source(checkpoint_path), map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    if subset == "validation" and int(config["fold"]) != fold:
        raise ValueError("This checkpoint was not trained with the requested held-out fold")
    if postprocess_config is not None:
        overrides = yaml.safe_load(postprocess_config.read_text(encoding="utf-8"))
        if "config" in overrides:
            overrides = overrides["config"]
        unknown = set(overrides) - {
            "mask_threshold",
            "minimum_area",
            "maximum_area",
            "seed_threshold",
            "minimum_confidence",
        }
        if unknown:
            raise ValueError(f"Unsupported postprocessing keys: {sorted(unknown)}")
        config.update(overrides)
    output.mkdir(parents=True, exist_ok=True)
    model = ResNetUNet(pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    paths = supplied_paths(root, subset, fold, limit)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_digest(checkpoint_path),
        "subset": subset,
        "fold": fold if subset == "validation" else None,
        "physical_images": len(paths),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "config": config,
        "status": "running",
    }
    (output / "inference-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    predictions = {}
    for number, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid supplied JPEG: {path}")
        probability = (
            predict_image(model, image, torch.device("cuda"), config).astype(np.float16).astype(np.float32)
        )
        rles = probability_instances(probability, config)
        if rles:
            predictions[path.stem] = rles
        if (number + 1) % 20 == 0:
            print(f"U-Net inference {number + 1}/{len(paths)}", flush=True)
    validation = validate_predictions(predictions, {path.stem for path in paths})
    write_predictions(output / "predictions.csv", predictions)
    metadata.update(
        status="complete",
        elapsed_seconds=time.monotonic() - started,
        submission_validation=validation,
        csv_sha256=file_digest(output / "predictions.csv"),
    )
    (output / "inference-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def supplied_paths(root: Path, subset: str, fold: int, limit: int | None = None) -> list[Path]:
    if subset == "test":
        paths = sorted((dataset_root(root) / "test/test_images").glob("*.jpeg"))
    elif subset == "validation":
        folds = pd.read_csv(root / "data/splits/folds.csv")
        image_root = dataset_root(root) / "train/train_images"
        paths = [image_root / name for name in sorted(folds.loc[folds.fold == fold, "filename"].tolist())]
    else:
        raise ValueError("Choose test or validation images")
    if limit is not None:
        if limit < 1:
            raise ValueError("Inference limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError("No supplied images found for inference")
    return paths


def predict_detector(
    root: Path,
    checkpoint_path: Path,
    output: Path,
    subset: str = "test",
    fold: int = 0,
    limit: int | None = None,
    postprocess_config: Path | None = None,
) -> dict:
    """Use only declared weights, configuration and supplied JPEGs for inference."""
    from solar_filament.detection import detector_instances, make_detector

    checkpoint_path = artifact_path(checkpoint_path, root)
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for declared detector inference")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(checkpoint_source(checkpoint_path), map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    if subset == "validation" and int(config["fold"]) != fold:
        raise ValueError("This checkpoint was not trained with the requested held-out fold")
    if postprocess_config is not None:
        overrides = yaml.safe_load(postprocess_config.read_text(encoding="utf-8"))
        if "config" in overrides:
            overrides = overrides["config"]
        unknown = set(overrides) - {"score_threshold", "mask_threshold", "minimum_area", "maximum_area"}
        if unknown:
            raise ValueError(f"Unsupported detector postprocessing keys: {sorted(unknown)}")
        config.update(overrides)
    paths = supplied_paths(root, subset, fold, limit)
    model = make_detector(config, pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "maskrcnn-resnet50-fpn-v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_digest(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "subset": subset,
        "fold": fold if subset == "validation" else None,
        "physical_images": len(paths),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "config": config,
        "status": "running",
    }
    if postprocess_config is not None:
        metadata["postprocess_source"] = str(postprocess_config)
        metadata["postprocess_sha256"] = file_digest(postprocess_config)
    (output / "inference-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    predictions = {}
    started = time.monotonic()
    preprocess = DetectorPreprocessing(config)
    with torch.inference_mode():
        for number, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise ValueError(f"Invalid supplied JPEG: {path}")
            image = preprocess(image)
            tensor = torch.from_numpy(image.astype(np.float32)[None] / 255).repeat(3, 1, 1).cuda()
            with torch.autocast("cuda", enabled=config["mixed_precision"]):
                result = model([tensor])[0]
            rles = detector_instances(result, config)
            if rles:
                predictions[path.stem] = rles
            if (number + 1) % 20 == 0:
                print(f"Detector inference {number + 1}/{len(paths)}", flush=True)
            del result, tensor
    validation = validate_predictions(predictions, {path.stem for path in paths})
    write_predictions(output / "predictions.csv", predictions)
    metadata.update(
        status="complete",
        elapsed_seconds=time.monotonic() - started,
        submission_validation=validation,
        csv_sha256=file_digest(output / "predictions.csv"),
    )
    (output / "inference-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
