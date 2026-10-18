"""Bounded, resumable YOLO segmentation pilot on the project's frozen folds."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from yolo_pilot import ModelEMA, learning_rate, make_model, pack_batch, pretrained_path, schedule_end_to_end

from solar_filament.detection import DetectionDataset, candidate_records, detector_instances, write_candidates
from solar_filament.evaluate import evaluate
from solar_filament.report import build_report
from solar_filament.storage import artifact_path, configure_caches, dataset_root, storage_root
from solar_filament.submission import validate_predictions, write_predictions
from solar_filament.training_iterator import training_iterator


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, record: dict):
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


def optimizer_for(model, config):
    groups = {"weights": [], "normalization": [], "biases": []}
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            group = "biases" if name == "bias" else "normalization" if isinstance(
                module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm),
            ) else "weights"
            groups[group].append(parameter)
    return torch.optim.AdamW([
        {"params": groups["weights"], "weight_decay": config["weight_decay"]},
        {"params": groups["normalization"], "weight_decay": 0},
        {"params": groups["biases"], "weight_decay": 0},
    ], lr=config["learning_rate"])


def save_checkpoint(path, model, ema, optimizer, scaler, config, epoch, step, best_pq):
    """Save only at optimizer boundaries, preserving EMA and random state."""
    record = {
        "model": model.state_dict(), "ema": ema.model.state_dict(), "ema_updates": ema.updates,
        "architecture": model.yaml, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "config": config, "epoch": epoch, "step": step, "best_pq": best_pq,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    if temporary.stat().st_size >= 2**32:
        raise ValueError("Checkpoint exceeds E: FAT32 file limit")
    temporary.replace(path)


@torch.inference_mode()
def predict(model, root, config, output, subset="validation", checkpoint=None):
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    output = artifact_path(output, root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve existing inference; choose a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    # The predictor may fuse convolutions. Use a separate model to preserve
    # train/EMA checkpoint layouts for continuation and optimizer state.
    # Candidates keep every proposal above a low floor. Lower-scored boxes never
    # suppress higher-scored ones, so the thresholded predictions are unchanged.
    save_candidates = bool(config.get("save_candidates", False))
    conf = min(config["score_threshold"], config.get("candidate_threshold", 0.05)) if save_candidates else config["score_threshold"]
    max_det = max(config["maximum_detections"], config.get("candidate_maximum_detections", 100)) if save_candidates else config["maximum_detections"]
    candidate_directory = output / "candidates"
    if save_candidates:
        candidate_directory.mkdir()
    predictor = SegmentationPredictor(overrides={
        "task": "segment", "mode": "predict", "imgsz": config["image_size"],
        "conf": conf, "iou": config["nms_iou"],
        "max_det": max_det, "retina_masks": True,
        "device": 0, "half": config["mixed_precision"], "verbose": False,
        "save": False, "save_txt": False, "save_conf": False,
        "project": str(root / "runs/yolo-predict-cache"), "name": "runtime", "exist_ok": True,
    })
    predictor.setup_model(model=model, verbose=False)
    if subset == "test":
        names = sorted(path.name for path in (dataset_root(root) / "test/test_images").glob("*.jpeg"))
        directory = dataset_root(root) / "test/test_images"
        if len(names) != 180:
            raise ValueError("Test inference must include all 180 supplied images")
    else:
        folds = pd.read_csv(root / "data/splits/folds.csv")
        names = sorted(folds.loc[folds.fold == config["fold"], "filename"])
        if config.get("validation_limit"):
            names = names[:config["validation_limit"]]
        directory = dataset_root(root) / "train/train_images"
    predictions, started = {}, time.monotonic()
    for number, name in enumerate(names):
        image = cv2.imread(str(directory / name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Missing or unexpected native image: {name}")
        result = predictor(source=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), stream=False)[0]
        candidates = []
        if result.masks is not None:
            if tuple(result.masks.data.shape[-2:]) != image.shape:
                raise ValueError("YOLO masks must be restored to native image coordinates")
            scores, masks = result.boxes.conf, result.masks.data
            if save_candidates:
                candidates = candidate_records(scores.float().cpu().numpy(), masks.float().cpu().numpy(), 0.5)
                keep = torch.nonzero(scores >= config["score_threshold"])[:config["maximum_detections"], 0]
                scores, masks = scores[keep], masks[keep]
            instances = detector_instances({"scores": scores, "masks": masks[:, None]}, config)
            if instances:
                predictions[Path(name).stem] = instances
        if save_candidates:
            write_candidates(candidate_directory / (Path(name).stem + ".json"), candidates)
        del result
        if (number + 1) % 20 == 0:
            print(f"YOLO {subset} inference {number + 1}/{len(names)}", flush=True)
    path = output / "predictions.csv"
    write_predictions(path, predictions)
    metadata = {
        "status": "complete", "model": config["model"], "subset": subset,
        "physical_images": len(names), "fold": config["fold"], "config": config,
        "csv_sha256": digest(path), "inference_seconds": time.monotonic() - started,
        "submission_validation": validate_predictions(predictions, {Path(name).stem for name in names}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if checkpoint:
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint training config differs from the run")
        metadata.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), weights="EMA")
    write(output / "inference-metadata.json", metadata)
    if subset == "validation" and not config.get("validation_limit"):
        metrics = evaluate(root, path, output / "evaluation", config["fold"])
        build_report(root, output / "evaluation")
        metadata["pooled"] = metrics["reference_pooled"]
        write(output / "metrics.json", metadata)
        print(f"YOLO full-fold PQ={metadata['pooled']['pq']:.5f}", flush=True)
    return metadata


def train(root, config, output, resume=False, smoke_steps=0):
    if smoke_steps:
        config = {**config, "epochs": 1, "steps_per_epoch": smoke_steps,
                  "gradient_accumulation": 1, "warmup_steps": 0,
                  "validation_limit": 2, "validation_every": 1, "time_limit_minutes": 15}
    accumulation = config["gradient_accumulation"]
    if config["steps_per_epoch"] % accumulation or config["checkpoint_every_steps"] % accumulation:
        raise ValueError("Epoch and checkpoint intervals must end at optimizer boundaries")
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the YOLO training pilot")
    if output.exists() and (output / "config.yaml").exists() and not resume:
        raise ValueError("Preserve existing training; resume it or choose a fresh run")
    output.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=True)
    if resume and (output / "config.yaml").read_text(encoding="utf-8") != config_text:
        raise ValueError("Resume config differs from the original run")
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False) if resume else None
    model = make_model(config, checkpoint["architecture"] if checkpoint else None, pretrained=not resume).cuda()
    ema = ModelEMA(model)
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])
    start_epoch, skip_steps, best_pq = 0, 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema"])
        ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, skip_steps, best_pq = checkpoint["epoch"], checkpoint["step"], checkpoint["best_pq"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
        checkpoint = None
    dataset = DetectionDataset(root, config, config["steps_per_epoch"] * config["batch_size"])
    workers = int(config.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=workers, pin_memory=workers > 0,
                        prefetch_factor=4 if workers else None,
                        collate_fn=functools.partial(pack_batch, image_size=config["image_size"], mask_ratio=config["mask_ratio"]))
    metadata_path = output / "run-metadata.json"
    source_paths = [Path(__file__), Path(__file__).with_name("yolo_pilot.py")]
    model_digest = hashlib.sha256()
    for source_path in sorted((Path(__file__).resolve().parents[1] / "src/solar_filament").glob("*.py")):
        model_digest.update(source_path.name.encode())
        model_digest.update(source_path.read_bytes())
    fingerprints = {
        "source_sha256": {path.name: digest(path) for path in source_paths},
        "shared_model_source_sha256": model_digest.hexdigest(),
        "split_sha256": json.loads((root / "data/splits/folds-metadata.json").read_text(encoding="utf-8"))["split_sha256"],
        "dataset_sha256": json.loads((root / "data/manifests/verified-dataset.json").read_text(encoding="utf-8"))["dataset_sha256"],
    }
    if resume:
        original = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(original[key] != value for key, value in fingerprints.items()):
            raise ValueError("YOLO resume source, dataset or split differs from the original run")
    if not resume:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        write(metadata_path, {
            "model": config["model"], "fold": config["fold"], "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__, "device": torch.cuda.get_device_name(),
            "train_physical_images": len(dataset.names), "sampling_policy": config["sampling_policy"],
            "annotator_policy": config["annotator_policy"], "mask_targets": "Independent COCO raster masks, area-resampled soft boundaries",
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "pretrained_sha256": digest(pretrained_path(config)),
            **fingerprints,
            "code_revision": revision.stdout.strip() if revision.returncode == 0 else os.environ.get("SOLAR_CODE_REVISION", "uncommitted"),
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config["epochs"]):
        dataset.epoch = epoch
        end_to_end = schedule_end_to_end(model, epoch)
        model.train()
        total_loss, batches = 0., 0
        for step, batch in enumerate(training_iterator(loader, skip_steps if epoch == start_epoch else 0)):
            if epoch == start_epoch and step < skip_steps:
                continue
            absolute_step = epoch * config["steps_per_epoch"] + step
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, absolute_step)
            batch = {key: value.cuda() for key, value in batch.items()}
            if end_to_end:
                # YOLO26's auxiliary semantic head: one class, foreground taken from the instance masks.
                batch["sem_masks"] = batch["masks"].new_zeros(len(batch["img"]), *batch["masks"].shape[-2:])
            with torch.autocast("cuda", enabled=config["mixed_precision"]):
                loss_vector, _ = model(batch)
                loss = loss_vector.sum() / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite YOLO loss at epoch {epoch + 1}, step {step + 1}")
            scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * accumulation
            batches += 1
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    ema.update(model)
                if (step + 1) % config["checkpoint_every_steps"] == 0:
                    save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch, step + 1, best_pq)
                if time.monotonic() - started > config["time_limit_minutes"] * 60:
                    save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch, step + 1, best_pq)
                    result = {"status": "time_limit", "epoch": epoch, "step": step + 1, "best_pq": best_pq}
                    write(output / "result.json", result)
                    return result
            if (step + 1) % 40 == 0 or step + 1 == config["steps_per_epoch"]:
                print(f"YOLO epoch={epoch + 1} step={step + 1}/{config['steps_per_epoch']} loss={total_loss / batches:.4f} elapsed_minutes={(time.monotonic() - started) / 60:.1f}", flush=True)
            del batch, loss, loss_vector
        skip_steps = 0
        save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch + 1, 0, best_pq)
        # A negative fold trains on every image and has no held-out validation.
        if config["fold"] >= 0 and ((epoch + 1) % config["validation_every"] == 0 or epoch + 1 == config["epochs"]):
            # Copy EMA before the predictor's fusion/half conversion.
            inference_model = copy.deepcopy(ema.model)
            metrics = predict(inference_model, root, config, output / f"validation-epoch-{epoch + 1:03d}")
            del inference_model
            torch.cuda.empty_cache()
            pq = metrics.get("pooled", {}).get("pq", -1.)
            if pq > best_pq:
                best_pq = pq
                torch.save({"model": ema.model.state_dict(), "architecture": model.yaml, "config": config,
                            "epoch": epoch + 1, "validation_pq": pq}, output / "best-inference.pt")
            save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch + 1, 0, best_pq)
    result = {
        "status": "complete", "epoch": config["epochs"], "best_pq": best_pq,
        "training_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "ema_updates": ema.updates, "physical_image_presentations": config["epochs"] * len(dataset),
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--predict-checkpoint", type=Path)
    parser.add_argument("--subset", choices=("test", "validation"), default="validation")
    args = parser.parse_args()
    root = storage_root()
    configure_caches(root)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if digest(pretrained_path(config)) != config["pretrained_sha256"]:
        raise ValueError("YOLO pretrained weights differ from the frozen manifest")
    if args.predict_checkpoint:
        checkpoint_path = artifact_path(args.predict_checkpoint, root)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Inference config differs from the selected checkpoint")
        model = make_model(config, checkpoint["architecture"], pretrained=False)
        model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        del checkpoint
        predict(model.cuda(), root, config, args.output, args.subset, checkpoint_path)
    else:
        train(root, config, args.output, args.resume, args.smoke_steps)


if __name__ == "__main__":
    main()
