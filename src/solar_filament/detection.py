"""Whole-disk Mask R-CNN baseline with independent annotator supervision."""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from solar_filament.annotator_policy import agreement_medoids
from solar_filament.checkpoint import save_inference_checkpoint
from solar_filament.checkpoint_delta import checkpoint_source
from solar_filament.coco import CocoIndex, encode_local_mask
from solar_filament.learning_rate import learning_rate_at_step, validate_learning_rate_schedule
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.preprocessing import DetectorPreprocessing
from solar_filament.sampling import coverage_index
from solar_filament.storage import annotations_path, artifact_path, dataset_root
from solar_filament.submission import validate_predictions, write_predictions
from solar_filament.train_unet import save_checkpoint
from solar_filament.training_iterator import training_iterator


class DetectionDataset(Dataset):
    def __init__(self, root: Path, config: dict, length: int):
        self.root, self.config, self.length = root, config, length
        self.epoch = 0
        self.index = CocoIndex(annotations_path(root))
        folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
        self.names = sorted(folds.loc[folds.fold != config["fold"], "filename"].tolist())
        self.cache = OrderedDict()
        self.preprocess = DetectorPreprocessing(config)
        if config.get("sampling_policy", "random") not in ("random", "coverage"):
            raise ValueError("Detector sampling policy must be random or coverage")
        self.annotator_policy = config.get("annotator_policy", "random")
        if self.annotator_policy in ("medoid", "medoid_ties"):
            manifest = json.loads((root / "data/derived/annotator-medoid.json").read_text())
            self.medoid_records = {
                name: manifest["observations"][name]["selected_record"] for name in self.names
            }
            if self.annotator_policy == "medoid_ties":
                self.medoid_candidates = {
                    name: agreement_medoids(
                        manifest["observations"][name]["records"],
                        np.asarray(manifest["observations"][name]["pairwise_pq"]),
                    )
                    for name in self.names
                }
        elif self.annotator_policy != "random":
            raise ValueError("Detector annotator policy must be random, medoid or medoid_ties")
        # Optional pseudo-labelled external frames (COCO JSON with RLE masks under the storage root).
        self.pseudo_names, self.pseudo_fraction = [], float(config.get("pseudo_fraction", 0.0))
        if config.get("pseudo_annotations"):
            self.pseudo_index = CocoIndex(root / config["pseudo_annotations"])
            self.pseudo_directory = root / config["pseudo_images"]
            self.pseudo_names = sorted(
                name for name, records in self.pseudo_index.images_by_filename.items()
                if self.pseudo_index.annotations_by_image[records[0]["id"]]
            )
            if not self.pseudo_names or not 0 < self.pseudo_fraction < 1:
                raise ValueError("Pseudo labels need annotated frames and a fraction in (0, 1)")

    def __len__(self):
        return self.length

    def __getitem__(self, sample):
        rng = np.random.default_rng(self.config["seed"] + self.epoch * self.length + sample)
        pseudo = False
        if self.pseudo_names:
            # A separate stream decides pseudo draws, leaving the real-image stream unchanged.
            pseudo_rng = np.random.default_rng(
                np.random.SeedSequence([self.config["seed"], self.epoch * self.length + sample, 7331])
            )
            pseudo = pseudo_rng.random() < self.pseudo_fraction
        if pseudo:
            name = self.pseudo_names[int(pseudo_rng.integers(len(self.pseudo_names)))]
            index, directory = self.pseudo_index, self.pseudo_directory
            rng.integers(len(self.names))
        else:
            image_index = int(rng.integers(len(self.names)))
            if self.config.get("sampling_policy", "random") == "coverage":
                image_index = coverage_index(
                    len(self.names), self.epoch * self.length + sample, self.config["seed"]
                )
            name = self.names[image_index]
            index, directory = self.index, dataset_root(self.root) / "train" / "train_images"
        if name not in self.cache:
            image = cv2.imread(str(directory / name), 0)
            if image is None:
                raise FileNotFoundError(name)
            # Process the native image before any geometric/brightness
            # augmentation. Inference uses the identical image-only order.
            self.cache[name] = self.preprocess(image)
        self.cache.move_to_end(name)
        if len(self.cache) > 48:
            self.cache.popitem(last=False)
        image = self.cache[name]
        records = index.images_by_filename[name]
        record = records[int(rng.integers(len(records)))]
        if pseudo:
            pass
        elif self.annotator_policy == "medoid":
            record = next(record for record in records if record["id"] == self.medoid_records[name])
        elif self.annotator_policy == "medoid_ties":
            candidates = self.medoid_candidates[name]
            if len(candidates) < len(records):
                # Keep the original annotation draw and augmentation RNG stream.
                # An independent stream resolves partial ties uniformly; full
                # ties preserve the original random record, including both
                # records of every two-annotator observation.
                policy_rng = np.random.default_rng(
                    np.random.SeedSequence([self.config["seed"], self.epoch * self.length + sample, 9017])
                )
                chosen = candidates[int(policy_rng.integers(len(candidates)))]
                record = next(record for record in records if record["id"] == chosen)
        rles = index.record_rles(record["id"])
        masks = mask_utils.decode(rles).transpose(2, 0, 1)
        boxes = np.asarray(mask_utils.toBbox(rles), dtype=np.float32)
        boxes[:, 2:] += boxes[:, :2]
        height, width = image.shape
        if rng.random() < 0.5:
            image, masks = image[:, ::-1], masks[:, :, ::-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if rng.random() < 0.5:
            image, masks = image[::-1], masks[:, ::-1]
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        if rng.random() < 0.5:
            image, masks = np.rot90(image), np.rot90(masks, axes=(1, 2))
            boxes = boxes[:, [1, 2, 3, 0]]
            boxes[:, [1, 3]] = width - boxes[:, [1, 3]]
        if self.config.get("scale_jitter") or self.config.get("translate_jitter"):
            # Draw only when configured so earlier runs keep their RNG stream.
            image, masks, boxes = random_affine(image, masks, rng, self.config)
        image = np.clip(
            image.astype(np.float32) / 255 * rng.uniform(0.9, 1.1) + rng.uniform(-0.03, 0.03), 0, 1
        )
        tensor = torch.from_numpy(image.copy()[None]).repeat(3, 1, 1)
        target = {
            "boxes": torch.from_numpy(boxes.copy()),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks.copy()),
        }
        return tensor, target


def random_affine(image: np.ndarray, masks: np.ndarray, rng, config: dict):
    """Scale about the disk centre and shift; warp each mask only within its box.

    Instances keep their identity; boxes are recomputed from the warped masks
    and instances with fewer than ``minimum_augmented_area`` pixels left inside
    the frame are dropped.
    """
    low, high = config.get("scale_jitter") or (1.0, 1.0)
    shift = float(config.get("translate_jitter", 0.0))
    if not 0 < low <= high or not 0 <= shift < 0.5:
        raise ValueError("Scale jitter must be positive and translation below half the image")
    height, width = image.shape
    scale = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    tx, ty = rng.uniform(-shift, shift, size=2) * (width, height)
    cx, cy = (width - 1) / 2, (height - 1) / 2
    matrix = np.array([[scale, 0, cx + tx - scale * cx], [0, scale, cy + ty - scale * cy]], dtype=np.float64)
    warped = cv2.warpAffine(
        np.ascontiguousarray(image), matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0
    )
    minimum = int(config.get("minimum_augmented_area", 20))
    kept_masks, kept_boxes = [], []
    for mask in masks:
        rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        if not len(rows):
            continue
        x0, x1, y0, y1 = columns[0], columns[-1] + 1, rows[0], rows[-1] + 1
        ox0 = int(np.floor(scale * x0 + matrix[0, 2])) - 1
        oy0 = int(np.floor(scale * y0 + matrix[1, 2])) - 1
        ow, oh = int(np.ceil(scale * (x1 - x0))) + 3, int(np.ceil(scale * (y1 - y0))) + 3
        local = np.array(
            [[scale, 0, scale * x0 + matrix[0, 2] - ox0], [0, scale, scale * y0 + matrix[1, 2] - oy0]],
            dtype=np.float64,
        )
        patch = cv2.warpAffine(
            np.ascontiguousarray(mask[y0:y1, x0:x1]), local, (ow, oh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        cx0, cy0 = max(ox0, 0), max(oy0, 0)
        cx1, cy1 = min(ox0 + ow, width), min(oy0 + oh, height)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        visible = patch[cy0 - oy0 : cy1 - oy0, cx0 - ox0 : cx1 - ox0]
        if int(visible.sum()) < minimum:
            continue
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[cy0:cy1, cx0:cx1] = visible
        rows, columns = np.flatnonzero(visible.any(axis=1)), np.flatnonzero(visible.any(axis=0))
        kept_masks.append(canvas)
        kept_boxes.append([cx0 + columns[0], cy0 + rows[0], cx0 + columns[-1] + 1, cy0 + rows[-1] + 1])
    if kept_masks:
        return warped, np.stack(kept_masks), np.asarray(kept_boxes, dtype=np.float32)
    return warped, np.zeros((0, height, width), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)


def detection_collate(batch):
    return tuple(zip(*batch))


def make_detector(config: dict, pretrained: bool = True):
    # TorchVision otherwise unfreezes every backbone layer when weights=None,
    # which changes resumed training and checkpoint-based reconstruction.
    trainable_layers = int(config.get("trainable_backbone_layers", 3 if config.get("pretrained", pretrained) else 5))
    if not 0 <= trainable_layers <= 5:
        raise ValueError("Trainable detector backbone layers must be between zero and five")
    mask_roi_size = int(config.get("mask_roi_size", 14))
    if mask_roi_size not in (14, 28):
        raise ValueError("Detector mask ROI size must be 14 or 28")
    anchor_sizes = tuple(config.get("anchor_sizes", (8, 16, 32, 64, 128)))
    if len(anchor_sizes) != 5 or any(not isinstance(size, int) or size <= 0 for size in anchor_sizes):
        raise ValueError("Detector anchor sizes must contain five positive integers")
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=config["image_size"],
        max_size=config["image_size"],
        box_detections_per_img=config.get("maximum_detections", 60),
        box_score_thresh=0.05,
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
        mask_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=mask_roi_size, sampling_ratio=2
        ),
    )
    prefixes = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        prefixes.append("bn1")
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((size,) for size in anchor_sizes), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2
    )
    return model


class DetectorEMA:
    """Exponential moving average of weights with a warm-up ramp on the decay."""

    def __init__(self, model, decay: float = 0.9998, tau: float = 2000):
        self.model = copy.deepcopy(model).eval()
        self.decay, self.tau, self.updates = decay, tau, 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        current = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(current[name].detach(), alpha=1 - decay)
            else:
                value.copy_(current[name])


def candidate_records(scores, masks, mask_threshold: float) -> list[dict]:
    """Every scored proposal as a full-image RLE, before score filtering or pixel ownership.

    These allow CPU threshold, area and fusion sweeps without rerunning the GPU.
    """
    records = []
    for score, probability in zip(scores, masks):
        mask = probability >= mask_threshold
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        rle = encode_local_mask(mask[y0:y1, x0:x1], x0, y0, *mask.shape)
        records.append({"score": float(score), "area": int(len(xs)), "rle": rle})
    return records


def write_candidates(path: Path, records: list[dict]) -> None:
    serializable = [
        {**record, "rle": {"size": record["rle"]["size"], "counts": _counts_text(record["rle"]["counts"])}}
        for record in records
    ]
    path.write_text(json.dumps(serializable), encoding="utf-8")


def _counts_text(counts) -> str:
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def detector_instances(result: dict, config: dict) -> list[dict]:
    """Apply score ordering and assign each native pixel to at most one instance."""
    scores = result["scores"].detach().float().cpu().numpy()
    masks = result["masks"].detach().float().cpu().numpy()[:, 0]
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    instances = []
    for number in np.argsort(-scores):
        if scores[number] < config["score_threshold"]:
            continue
        mask = (masks[number] >= config["mask_threshold"]) & ~occupied
        ys, xs = np.nonzero(mask)
        if not config["minimum_area"] <= len(xs) <= config["maximum_area"]:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        rle = encode_local_mask(mask[y0:y1, x0:x1], x0, y0, *occupied.shape)
        instances.append(rle)
        occupied |= mask
    return instances


@torch.inference_mode()
def validate_detector(model, root: Path, config: dict, output: Path) -> dict:
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    raw_directory = output / "proposals"
    raw_directory.mkdir(exist_ok=True)
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
    names = sorted(folds.loc[folds.fold == config["fold"], "filename"].tolist())
    if config.get("validation_limit"):
        names = names[: config["validation_limit"]]
    predictions, rows = {}, []
    pooled = PQCounts()
    started = time.monotonic()
    preprocess = DetectorPreprocessing(config)
    candidate_directory = None
    if config.get("save_candidates", False):
        candidate_directory = output / "candidates"
        candidate_directory.mkdir(exist_ok=True)
    for number, name in enumerate(names):
        stem = Path(name).stem
        image = cv2.imread(str(dataset_root(root) / "train" / "train_images" / name), 0)
        image = preprocess(image)
        tensor = torch.from_numpy(image.astype(np.float32)[None] / 255).repeat(3, 1, 1).cuda()
        with torch.autocast("cuda", enabled=config["mixed_precision"]):
            result = model([tensor])[0]
        rles = detector_instances(result, config)
        if rles:
            predictions[stem] = rles
        if candidate_directory is not None:
            write_candidates(
                candidate_directory / (stem + ".json"),
                candidate_records(
                    result["scores"].detach().float().cpu().numpy(),
                    result["masks"].detach().float().cpu().numpy()[:, 0],
                    config["mask_threshold"],
                ),
            )
        raw = []
        scores = result["scores"].detach().float().cpu().tolist()
        boxes = result["boxes"].detach().float().cpu().tolist()
        for score, box in zip(scores, boxes):
            raw.append({"score": score, "bbox_xyxy": box})
        (raw_directory / (stem + ".json")).write_text(json.dumps(raw), encoding="utf-8")
        for record in index.images_by_filename[name]:
            counts = reference_counts(overlap_matrix(index.record_rles(record["id"]), rles))
            pooled += counts
            rows.append({"filename": name, "annotator_image": record["id"], **counts.scores()})
        if (number + 1) % 20 == 0:
            print(f"detector validation {number + 1}/{len(names)} pq={pooled.scores()['pq']:.4f}", flush=True)
        del result, tensor
    validation = validate_predictions(predictions, {Path(name).stem for name in names})
    write_predictions(output / "predictions.csv", predictions)
    pd.DataFrame(rows).to_csv(output / "per-record.csv", index=False, lineterminator="\n")
    metrics = {
        "physical_images": len(names),
        "annotator_records": len(rows),
        "pooled": pooled.scores(),
        "submission_validation": validation,
        "inference_seconds": time.monotonic() - started,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def train_detector(
    root: Path, config_path: Path, output: Path, smoke_steps: int = 0, resume: bool = False
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if smoke_steps:
        config.update(
            epochs=1,
            steps_per_epoch=smoke_steps,
            validation_every=1,
            validation_limit=2,
            image_size=config.get("smoke_image_size", 512),
            maximum_detections=10,
            pretrained=False,
            time_limit_minutes=15,
        )
    validate_learning_rate_schedule(config)
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for detector training")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.yaml").exists() and not resume:
        raise ValueError("Detector run exists; choose a new run directory or resume")
    config_text = yaml.safe_dump(config, sort_keys=True)
    if resume and (output / "config.yaml").read_text(encoding="utf-8") != config_text:
        raise ValueError("Resume configuration differs from the original detector run")
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    model = make_detector(config, config["pretrained"] and not resume).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])
    ema = DetectorEMA(model, config.get("ema_decay", 0.9998), config.get("ema_tau", 2000)) if config.get("ema") else None
    start_epoch, skip_steps, best_pq = 0, 0, -1.0
    if resume:
        checkpoint = torch.load(checkpoint_source(output / "last.pt"), map_location="cpu", weights_only=False)
        if yaml.safe_dump(checkpoint["config"], sort_keys=True) != config_text:
            raise ValueError("Detector resume configuration differs")
        model.load_state_dict(checkpoint["model"])
        if ema is not None:
            ema.model.load_state_dict(checkpoint["ema"])
            ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, skip_steps, best_pq = checkpoint["epoch"], checkpoint["step"], checkpoint["best_pq"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
        del checkpoint
    dataset = DetectionDataset(root, config, config["steps_per_epoch"] * config["batch_size"])
    # Samples are seeded by (seed, epoch, index), so worker processes give the
    # same stream; non-persistent workers pick up each new dataset.epoch.
    workers = int(config.get("num_workers", 0))
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], num_workers=workers, collate_fn=detection_collate,
        pin_memory=workers > 0, prefetch_factor=4 if workers else None,
    )

    def checkpoint_extra():
        return {"ema": ema.model.state_dict(), "ema_updates": ema.updates} if ema is not None else None
    metadata = {
        "model": "maskrcnn-resnet50-fpn-v2",
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "train_physical_images": len(dataset.names),
        "fold": config["fold"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling_policy": config.get("sampling_policy", "random"),
        "annotator_policy": config.get("annotator_policy", "random"),
    }
    if config.get("annotator_policy", "random") in ("medoid", "medoid_ties"):
        metadata["annotator_manifest_sha256"] = hashlib.sha256(
            (root / "data/derived/annotator-medoid.json").read_bytes()
        ).hexdigest()
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
    for source in sorted((source_root / "src" / "solar_filament").glob("*.py")):
        source_digest.update(source.name.encode())
        source_digest.update(source.read_bytes())
    metadata["source_sha256"] = source_digest.hexdigest()
    if resume:
        original_metadata = json.loads((output / "run-metadata.json").read_text())
        fingerprint_keys = ["dataset_sha256", "split_sha256"]
        if config.get("annotator_policy", "random") in ("medoid", "medoid_ties"):
            fingerprint_keys.append("annotator_manifest_sha256")
        for key in fingerprint_keys:
            if metadata[key] != original_metadata[key]:
                raise ValueError(f"Resume {key} differs from the original detector run")
        metadata_path = output / "resume-metadata.json"
    else:
        metadata_path = output / "run-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, config["epochs"]):
        dataset.epoch = epoch
        model.train()
        losses = []
        iterator = training_iterator(loader, continuing_epoch=epoch == start_epoch and skip_steps > 0)
        for step, (images, targets) in enumerate(iterator):
            if epoch == start_epoch and step < skip_steps:
                continue
            if config.get("learning_rate_policy", "constant") == "cosine":
                rate = learning_rate_at_step(config, epoch * config["steps_per_epoch"] + step)
                for group in optimizer.param_groups:
                    group["lr"] = rate
            images = [image.cuda() for image in images]
            targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["mixed_precision"]):
                loss_items = model(images, targets)
                loss = sum(loss_items.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite detector loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None and scaler.get_scale() >= old_scale:
                ema.update(model)
            losses.append(float(loss.detach()))
            if (step + 1) % 20 == 0 or smoke_steps:
                print(
                    f"detector epoch={epoch + 1} step={step + 1}/{len(loader)} loss={np.mean(losses):.4f} elapsed_minutes={(time.monotonic() - started) / 60:.1f}"
                    + (f" ema_updates={ema.updates}" if ema is not None else ""),
                    flush=True,
                )
            if (step + 1) % config["checkpoint_every_steps"] == 0:
                save_checkpoint(
                    output / "last.pt", model, optimizer, scaler, config, epoch, step + 1, best_pq,
                    extra=checkpoint_extra(),
                )
            if time.monotonic() - started > config["time_limit_minutes"] * 60:
                save_checkpoint(
                    output / "last.pt", model, optimizer, scaler, config, epoch, step + 1, best_pq,
                    extra=checkpoint_extra(),
                )
                result = {"status": "time_limit", "epoch": epoch, "step": step + 1, "best_pq": best_pq}
                (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
        skip_steps = 0
        save_checkpoint(
            output / "last.pt", model, optimizer, scaler, config, epoch + 1, 0, best_pq, extra=checkpoint_extra()
        )
        # A negative fold trains on every image and has no held-out validation.
        if config["fold"] >= 0 and ((epoch + 1) % config["validation_every"] == 0 or epoch + 1 == config["epochs"]):
            evaluated = ema.model if ema is not None else model
            metrics = validate_detector(evaluated, root, config, output / f"validation-epoch-{epoch + 1:03d}")
            pq = metrics["pooled"]["pq"]
            print(f"detector epoch={epoch + 1} validation_pq={pq:.5f}", flush=True)
            if pq > best_pq:
                best_pq = pq
                if config.get("save_best_checkpoint", True):
                    if config.get("save_best_weights_only", False):
                        save_inference_checkpoint(output / "best.pt", evaluated, config, epoch + 1, best_pq)
                    else:
                        save_checkpoint(
                            output / "best.pt", model, optimizer, scaler, config, epoch + 1, 0, best_pq,
                            extra=checkpoint_extra(),
                        )
            save_checkpoint(
                output / "last.pt", model, optimizer, scaler, config, epoch + 1, 0, best_pq, extra=checkpoint_extra()
            )
        if time.monotonic() - started > config["time_limit_minutes"] * 60:
            result = {"status": "time_limit", "epoch": epoch + 1, "step": 0, "best_pq": best_pq}
            break
    else:
        result = {"status": "complete", "epochs": config["epochs"], "best_pq": best_pq}
    result.update(
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        ema_updates=ema.updates if ema is not None else 0,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
