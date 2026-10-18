"""Train and evaluate a box-conditioned native filament instance crop refiner."""

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
import yaml
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

from solar_filament.coco import CocoIndex, annotation_rle, encode_local_mask
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.models import ResNetUNet
from solar_filament.storage import annotations_path, artifact_path, dataset_root
from solar_filament.submission import validate_predictions, write_predictions
from solar_filament.train_unet import save_checkpoint, segmentation_loss


def crop_geometry(box: list[float], shape: tuple[int, int]) -> tuple[int, int, int]:
    height, width = shape
    left, top, right, bottom = box
    side = min(min(shape), max(256, int(np.ceil(max(right - left, bottom - top) * 1.5))))
    x0 = int(np.clip((left + right - side) / 2, 0, width - side))
    y0 = int(np.clip((top + bottom - side) / 2, 0, height - side))
    return x0, y0, side


def crop_inputs(image: np.ndarray, box: list[float], size: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    x0, y0, side = crop_geometry(box, image.shape)
    crop = (
        cv2.resize(
            image[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        / 255
    )
    prompt = np.zeros((side, side), dtype=np.uint8)
    left, top, right, bottom = box
    xa, ya = int(np.clip(np.floor(left - x0), 0, side)), int(np.clip(np.floor(top - y0), 0, side))
    xb, yb = int(np.clip(np.ceil(right - x0), 0, side)), int(np.clip(np.ceil(bottom - y0), 0, side))
    prompt[ya:yb, xa:xb] = 1
    prompt = cv2.resize(prompt, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return np.stack([(crop - 0.5) / 0.25, prompt]), (x0, y0, side)


class InstanceCropDataset(Dataset):
    def __init__(self, root: Path, config: dict, length: int):
        self.root, self.config, self.length = root, config, length
        self.epoch = 0
        self.index = CocoIndex(annotations_path(root))
        folds = pd.read_csv(root / "data/splits/folds.csv")
        self.names = sorted(folds.loc[folds.fold != config["fold"], "filename"].tolist())
        self.cache = OrderedDict()

    def __len__(self):
        return self.length

    def __getitem__(self, sample):
        rng = np.random.default_rng(self.config["seed"] + self.epoch * self.length + sample)
        name = self.names[int(rng.integers(len(self.names)))]
        if name not in self.cache:
            image = cv2.imread(str(dataset_root(self.root) / "train/train_images" / name), 0)
            if image is None:
                raise FileNotFoundError(name)
            self.cache[name] = image
        self.cache.move_to_end(name)
        if len(self.cache) > 48:
            self.cache.popitem(last=False)
        image = self.cache[name]
        records = self.index.images_by_filename[name]
        record = records[int(rng.integers(len(records)))]
        annotations = self.index.annotations_by_image[record["id"]]
        annotation = annotations[int(rng.integers(len(annotations)))]
        left, top, width, height = annotation["bbox"]
        center_x = left + width / 2 + rng.normal(0, max(width * 0.12, 4))
        center_y = top + height / 2 + rng.normal(0, max(height * 0.12, 4))
        width, height = max(4, width * rng.uniform(0.8, 1.25)), max(4, height * rng.uniform(0.8, 1.25))
        box = [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2]
        negative = rng.random() < self.config.get("negative_probability", 0.2)
        if negative:
            accepted = False
            for _ in range(20):
                center_x, center_y = rng.uniform(256, 1792, size=2)
                box = [
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ]
                accepted = all(
                    max(0, min(box[2], a["bbox"][0] + a["bbox"][2]) - max(box[0], a["bbox"][0]))
                    * max(0, min(box[3], a["bbox"][1] + a["bbox"][3]) - max(box[1], a["bbox"][1]))
                    == 0
                    for a in annotations
                )
                if accepted:
                    break
            negative = accepted
            if not accepted:
                left, top, width, height = annotation["bbox"]
                box = [left, top, left + width, top + height]
        inputs, (x0, y0, side) = crop_inputs(image, box, self.config["patch_size"])
        target = np.zeros((self.config["patch_size"], self.config["patch_size"]), dtype=np.float32)
        if not negative:
            full_mask = mask_utils.decode(annotation_rle(annotation))
            target = cv2.resize(
                full_mask[y0 : y0 + side, x0 : x0 + side], target.shape[::-1], interpolation=cv2.INTER_NEAREST
            ).astype(np.float32)
        turns = int(rng.integers(4))
        inputs, target = np.rot90(inputs, turns, axes=(1, 2)), np.rot90(target, turns)
        if rng.random() < 0.5:
            inputs, target = inputs[:, :, ::-1], target[:, ::-1]
        return torch.from_numpy(inputs.copy()), torch.from_numpy(target.copy()[None])


@torch.inference_mode()
def refine_proposals(model, image: np.ndarray, proposals: list[dict], config: dict) -> list[dict]:
    occupied = np.zeros(image.shape, dtype=bool)
    instances = []
    for proposal in sorted(proposals, key=lambda item: -item["score"]):
        if proposal["score"] < config["score_threshold"]:
            continue
        inputs, (x0, y0, side) = crop_inputs(image, proposal["bbox_xyxy"], config["patch_size"])
        tensor = torch.from_numpy(inputs[None]).cuda()
        with torch.autocast("cuda", enabled=config["mixed_precision"]):
            probability = model(tensor).float().sigmoid()[0, 0].cpu().numpy()
        probability = cv2.resize(probability, (side, side), interpolation=cv2.INTER_LINEAR)
        selected = (probability >= config["mask_threshold"]) & ~occupied[y0 : y0 + side, x0 : x0 + side]
        area = int(selected.sum())
        if not config["minimum_area"] <= area <= config["maximum_area"]:
            continue
        instances.append(encode_local_mask(selected, x0, y0, *image.shape))
        occupied[y0 : y0 + side, x0 : x0 + side] |= selected
    return instances


@torch.inference_mode()
def validate_refiner(model, root: Path, config: dict, proposal_directory: Path, output: Path) -> dict:
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data/splits/folds.csv")
    names = sorted(folds.loc[folds.fold == config["fold"], "filename"].tolist())
    if config.get("validation_limit"):
        names = names[: config["validation_limit"]]
    predictions, rows, pooled = {}, [], PQCounts()
    started = time.monotonic()
    for number, name in enumerate(names):
        image = cv2.imread(str(dataset_root(root) / "train/train_images" / name), 0)
        proposals = json.loads((proposal_directory / (Path(name).stem + ".json")).read_text())
        rles = refine_proposals(model, image, proposals, config)
        if rles:
            predictions[Path(name).stem] = rles
        for record in index.images_by_filename[name]:
            counts = reference_counts(overlap_matrix(index.record_rles(record["id"]), rles))
            pooled += counts
            rows.append({"filename": name, "annotator_image": record["id"], **counts.scores()})
        if (number + 1) % 20 == 0:
            print(f"refiner validation {number + 1}/{len(names)} pq={pooled.scores()['pq']:.4f}", flush=True)
    validation = validate_predictions(predictions, {Path(name).stem for name in names})
    write_predictions(output / "predictions.csv", predictions)
    pd.DataFrame(rows).to_csv(output / "per-record.csv", index=False, lineterminator="\n")
    result = {
        "physical_images": len(names),
        "annotator_records": len(rows),
        "pooled": pooled.scores(),
        "submission_validation": validation,
        "inference_seconds": time.monotonic() - started,
        "proposal_directory": str(proposal_directory),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def train_refiner(
    root: Path,
    config_path: Path,
    proposal_directory: Path,
    output: Path,
    smoke_steps: int = 0,
    resume: bool = False,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if smoke_steps:
        config.update(
            epochs=1, steps_per_epoch=smoke_steps, validation_limit=2, pretrained=False, time_limit_minutes=10
        )
    proposal_directory = artifact_path(proposal_directory, root)
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for instance crop refinement")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.yaml").exists() and not resume:
        raise ValueError("Refiner run exists; use --resume or a new directory")
    config_text = yaml.safe_dump(config, sort_keys=True)
    if resume and (output / "config.yaml").read_text(encoding="utf-8") != config_text:
        raise ValueError("Resume configuration differs from the original refiner run")
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    model = ResNetUNet(pretrained=config["pretrained"] and not resume, input_channels=2).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])
    start_epoch, skip_steps = 0, 0
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if yaml.safe_dump(checkpoint["config"], sort_keys=True) != config_text:
            raise ValueError("Resume configuration differs from the checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, skip_steps = checkpoint["epoch"], checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    dataset = InstanceCropDataset(root, config, config["steps_per_epoch"] * config["batch_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    metadata = {
        "model": "box-conditioned-resnet34-unet",
        "fold": config["fold"],
        "train_physical_images": len(dataset.names),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "proposal_directory": str(proposal_directory),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata["split_sha256"] = json.loads((root / "data/splits/folds-metadata.json").read_text())[
        "split_sha256"
    ]
    metadata["dataset_sha256"] = json.loads((root / "data/manifests/verified-dataset.json").read_text())[
        "dataset_sha256"
    ]
    source_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True)
    metadata["code_revision"] = revision.stdout.strip() if revision.returncode == 0 else "uncommitted"
    source_digest = hashlib.sha256()
    for source in sorted((source_root / "src/solar_filament").glob("*.py")):
        source_digest.update(source.name.encode())
        source_digest.update(source.read_bytes())
    metadata["source_sha256"] = source_digest.hexdigest()
    proposal_digest = hashlib.sha256()
    for proposal_file in sorted(proposal_directory.glob("*.json")):
        proposal_digest.update(proposal_file.name.encode())
        proposal_digest.update(proposal_file.read_bytes())
    metadata["proposals_sha256"] = proposal_digest.hexdigest()
    if resume:
        original_metadata = json.loads((output / "run-metadata.json").read_text())
        for key in ("dataset_sha256", "split_sha256", "proposals_sha256"):
            if metadata[key] != original_metadata[key]:
                raise ValueError(f"Resume {key} differs from the original run")
        (output / "resume-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    else:
        (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, config["epochs"]):
        dataset.epoch = epoch
        model.train()
        losses = []
        for step, (images, targets) in enumerate(loader):
            if epoch == start_epoch and step < skip_steps:
                continue
            images, targets = images.cuda(), targets.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["mixed_precision"]):
                loss = segmentation_loss(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite crop-refiner loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if (step + 1) % 20 == 0 or smoke_steps:
                print(
                    f"refiner epoch={epoch + 1} step={step + 1}/{len(loader)} loss={np.mean(losses):.4f}",
                    flush=True,
                )
            if time.monotonic() - started > config["time_limit_minutes"] * 60:
                save_checkpoint(output / "last.pt", model, optimizer, scaler, config, epoch, step + 1, -1)
                result = {"status": "time_limit", "epoch": epoch, "step": step + 1}
                (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
        skip_steps = 0
        save_checkpoint(output / "last.pt", model, optimizer, scaler, config, epoch + 1, 0, -1)
    metrics = validate_refiner(model, root, config, proposal_directory, output / "validation")
    result = {
        "status": "complete",
        "epochs": config["epochs"],
        "pq": metrics["pooled"]["pq"],
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
