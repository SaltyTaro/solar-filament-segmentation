"""Native-resolution random-annotator U-Net training and full-image validation."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from solar_filament.checkpoint_delta import checkpoint_source
from solar_filament.coco import CocoIndex
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.models import ResNetUNet
from solar_filament.postprocess import probability_instances
from solar_filament.storage import annotations_path, dataset_root
from solar_filament.submission import validate_predictions, write_predictions


class PatchDataset(Dataset):
    """Sample physical images uniformly and choose an independent annotator view."""

    def __init__(self, root: Path, config: dict, length: int):
        self.root = root
        self.config = config
        self.length = length
        self.epoch = 0
        self.index = CocoIndex(annotations_path(root))
        folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
        self.names = sorted(folds.loc[folds["fold"] != config["fold"], "filename"].tolist())
        self.cache = OrderedDict()

    def __len__(self):
        return self.length

    def cached_image(self, name: str) -> np.ndarray:
        key = ("image", name)
        if key not in self.cache:
            image = cv2.imread(
                str(dataset_root(self.root) / "train" / "train_images" / name), cv2.IMREAD_GRAYSCALE
            )
            if image is None:
                raise FileNotFoundError(name)
            self.cache[key] = image
        self.cache.move_to_end(key)
        if len(self.cache) > 80:
            self.cache.popitem(last=False)
        return self.cache[key]

    def cached_target(self, record_id: str) -> np.ndarray:
        key = ("target", record_id)
        if key not in self.cache:
            path = self.root / "data" / "processed" / "union-targets" / f"{record_id}.png"
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = self.index.union_mask(record_id) * 255
            self.cache[key] = (mask > 0).astype(np.uint8)
        self.cache.move_to_end(key)
        if len(self.cache) > 80:
            self.cache.popitem(last=False)
        return self.cache[key]

    def __getitem__(self, sample: int):
        rng = np.random.default_rng(self.config["seed"] + self.epoch * self.length + sample)
        name = self.names[int(rng.integers(len(self.names)))]
        records = self.index.images_by_filename[name]
        record = records[int(rng.integers(len(records)))]
        annotations = self.index.annotations_by_image[record["id"]]
        size = self.config["patch_size"]
        if annotations and rng.random() < self.config["positive_patch_probability"]:
            annotation = annotations[int(rng.integers(len(annotations)))]
            x, y, width, height = annotation["bbox"]
            center_x = x + width / 2 + rng.uniform(-size * 0.2, size * 0.2)
            center_y = y + height / 2 + rng.uniform(-size * 0.2, size * 0.2)
            x0 = int(np.clip(center_x - size / 2, 0, 2048 - size))
            y0 = int(np.clip(center_y - size / 2, 0, 2048 - size))
        else:
            x0, y0 = map(int, rng.integers(0, 2048 - size + 1, size=2))
        image = self.cached_image(name)[y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255
        target = self.cached_target(record["id"])[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
        turns = int(rng.integers(4))
        image, target = np.rot90(image, turns), np.rot90(target, turns)
        if rng.random() < 0.5:
            image, target = np.fliplr(image), np.fliplr(target)
        image = np.clip(image * rng.uniform(0.85, 1.15) + rng.uniform(-0.04, 0.04), 0, 1)
        image = (image - 0.5) / 0.25
        return torch.from_numpy(image.copy()[None]), torch.from_numpy(target.copy()[None])


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    # A 512x512 reduction exceeds float16's range even for moderate probabilities.
    probabilities = logits.float().sigmoid()
    dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dims)
    dice = (2 * intersection + 1) / (probabilities.sum(dim=dims) + targets.sum(dim=dims) + 1)
    return bce + (1 - dice).mean()


def tile_positions(length: int, size: int, stride: int) -> list[int]:
    return sorted(set(range(0, max(length - size, 0) + 1, stride)) | {max(length - size, 0)})


@torch.inference_mode()
def predict_image(model, image: np.ndarray, device: torch.device, config: dict) -> np.ndarray:
    size, stride = config["inference_patch_size"], config["inference_stride"]
    h, w = image.shape
    total = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros_like(total)
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    window = np.maximum(window, 0.01)
    for y in tile_positions(h, size, stride):
        for x in tile_positions(w, size, stride):
            patch = image[y : y + size, x : x + size].astype(np.float32) / 255
            tensor = torch.from_numpy(((patch - 0.5) / 0.25)[None, None]).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probability = model(tensor).sigmoid()[0, 0].float().cpu().numpy()
            total[y : y + size, x : x + size] += probability * window
            weights[y : y + size, x : x + size] += window
    return total / np.maximum(weights, 1e-8)


def probability_rles(probability: np.ndarray, config: dict) -> list[dict]:
    return probability_instances(probability, config)


@torch.inference_mode()
def validate_model(model, root: Path, config: dict, device: torch.device, output: Path) -> dict:
    model.eval()
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
    names = sorted(folds.loc[folds["fold"] == config["fold"], "filename"].tolist())
    if config.get("validation_limit"):
        names = names[: config["validation_limit"]]
    output.mkdir(parents=True, exist_ok=True)
    probability_directory = output / "probabilities"
    probability_directory.mkdir(exist_ok=True)
    save_probabilities = config.get("save_validation_probabilities", True)
    predictions = {}
    pooled = PQCounts()
    records = []
    started = time.monotonic()
    for number, name in enumerate(names):
        image = cv2.imread(str(dataset_root(root) / "train" / "train_images" / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(name)
        probability = predict_image(model, image, device, config).astype(np.float16).astype(np.float32)
        # One small file per image remains far below the FAT32 limit.
        if save_probabilities:
            np.save(probability_directory / (Path(name).stem + ".npy"), probability.astype(np.float16))
        rles = probability_rles(probability, config)
        if rles:
            predictions[Path(name).stem] = rles
        for record in index.images_by_filename[name]:
            counts = reference_counts(overlap_matrix(index.record_rles(record["id"]), rles))
            pooled += counts
            records.append({"filename": name, "annotator_image": record["id"], **counts.scores()})
        if (number + 1) % 20 == 0:
            print(f"validation {number + 1}/{len(names)} pooled_pq={pooled.scores()['pq']:.4f}", flush=True)
    validation = validate_predictions(predictions, {Path(name).stem for name in names})
    write_predictions(output / "predictions.csv", predictions)
    pd.DataFrame(records).to_csv(output / "per-record.csv", index=False, lineterminator="\n")
    metrics = {
        "physical_images": len(names),
        "annotator_records": len(records),
        "pooled": pooled.scores(),
        "inference_seconds": time.monotonic() - started,
        "probability_cache_saved": save_probabilities,
        "submission_validation": validation,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def save_checkpoint(
    path: Path, model, optimizer, scaler, config: dict, epoch: int, step: int, best_pq: float,
    extra: dict | None = None,
) -> None:
    temporary = path.with_suffix(".tmp")
    checkpoint = {
        **(extra or {}),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "step": step,
        "best_pq": best_pq,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    torch.save(checkpoint, temporary)
    if temporary.stat().st_size >= 2**32:
        raise ValueError("Checkpoint exceeds FAT32 limit")
    temporary.replace(path)


def train(root: Path, config_path: Path, output: Path, resume: bool = False, smoke_steps: int = 0) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if smoke_steps:
        config.update(
            steps_per_epoch=smoke_steps,
            epochs=1,
            validation_every=1,
            validation_limit=2,
            pretrained=False,
            time_limit_minutes=10,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; refusing an accidental long CPU training run")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = True
    output.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=True)
    config_digest = hashlib.sha256(config_text.encode()).hexdigest()
    checkpoint_path = output / "last.pt"
    if (output / "config.yaml").exists() and not resume:
        raise ValueError("Run exists; use --resume or a new run directory")
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    model = ResNetUNet(pretrained=config["pretrained"] and not resume).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])
    start_epoch = 0
    skip_steps = 0
    best_pq = -1.0
    if resume:
        checkpoint = torch.load(checkpoint_source(checkpoint_path), map_location=device, weights_only=False)
        if yaml.safe_dump(checkpoint["config"], sort_keys=True) != config_text:
            raise ValueError("Resume configuration differs from the checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, skip_steps = checkpoint["epoch"], checkpoint["step"]
        best_pq = checkpoint["best_pq"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
    dataset = PatchDataset(root, config, config["steps_per_epoch"] * config["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    metadata = {
        "config_sha256": config_digest,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "train_physical_images": len(dataset.names),
        "fold": config["fold"],
        "mixed_precision": config["mixed_precision"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    split_metadata = json.loads(
        (root / "data" / "splits" / "folds-metadata.json").read_text(encoding="utf-8")
    )
    data_metadata = json.loads(
        (root / "data" / "manifests" / "verified-dataset.json").read_text(encoding="utf-8")
    )
    metadata.update(
        split_sha256=split_metadata["split_sha256"], dataset_sha256=data_metadata["dataset_sha256"]
    )
    source_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True)
    metadata["code_revision"] = revision.stdout.strip() if revision.returncode == 0 else "uncommitted"
    source_digest = hashlib.sha256()
    for source in sorted((source_root / "src" / "solar_filament").glob("*.py")):
        source_digest.update(source.name.encode())
        source_digest.update(source.read_bytes())
    metadata["source_sha256"] = source_digest.hexdigest()
    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, config["epochs"]):
        dataset.epoch = epoch
        model.train()
        losses = []
        for step, (images, targets) in enumerate(loader):
            if epoch == start_epoch and step < skip_steps:
                continue
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["mixed_precision"]):
                loss = segmentation_loss(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite segmentation loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if (step + 1) % 20 == 0 or smoke_steps:
                print(
                    f"epoch={epoch + 1} step={step + 1}/{len(loader)} loss={np.mean(losses):.4f} "
                    f"elapsed_minutes={(time.monotonic() - started) / 60:.1f}",
                    flush=True,
                )
            if (step + 1) % config["checkpoint_every_steps"] == 0:
                save_checkpoint(checkpoint_path, model, optimizer, scaler, config, epoch, step + 1, best_pq)
            if time.monotonic() - started > config["time_limit_minutes"] * 60:
                save_checkpoint(checkpoint_path, model, optimizer, scaler, config, epoch, step + 1, best_pq)
                result = {"status": "time_limit", "epoch": epoch, "step": step + 1, "best_pq": best_pq}
                (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
        skip_steps = 0
        save_checkpoint(checkpoint_path, model, optimizer, scaler, config, epoch + 1, 0, best_pq)
        if (epoch + 1) % config["validation_every"] == 0 or epoch + 1 == config["epochs"]:
            metrics = validate_model(
                model, root, config, device, output / f"validation-epoch-{epoch + 1:03d}"
            )
            pq = metrics["pooled"]["pq"]
            print(f"epoch={epoch + 1} validation_pq={pq:.5f}", flush=True)
            if pq > best_pq:
                best_pq = pq
                save_checkpoint(output / "best.pt", model, optimizer, scaler, config, epoch + 1, 0, best_pq)
            save_checkpoint(checkpoint_path, model, optimizer, scaler, config, epoch + 1, 0, best_pq)
        if time.monotonic() - started > config["time_limit_minutes"] * 60:
            result = {"status": "time_limit", "epoch": epoch + 1, "step": 0, "best_pq": best_pq}
            (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    result = {
        "status": "complete",
        "epochs": config["epochs"],
        "best_pq": best_pq,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
