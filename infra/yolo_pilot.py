"""YOLO instance training from original COCO masks, without polygon conversion."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

_native_cv2_imread = cv2.imread


def pack_batch(samples: list, image_size: int, mask_ratio: int = 2) -> dict:
    """Keep annotator instances separate; area-resample their binary masks.

    Fractional boundary targets preserve the contribution of thin structures.
    Boxes stay in normalized native-image coordinates. Ultralytics receives
    class zero and a separate mask for each instance, including overlaps.
    """
    if image_size < 32 or image_size % 32 or mask_ratio not in (1, 2, 4):
        raise ValueError("Image size must be a multiple of 32; mask ratio must be 1, 2 or 4")
    if not samples:
        raise ValueError("A training batch must contain at least one image")
    images, masks, boxes, batch_indices = [], [], [], []
    mask_size = image_size // mask_ratio
    for image_number, (image, target) in enumerate(samples):
        if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] != image.shape[2]:
            raise ValueError("Training images must be square three-channel tensors")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Training images must be finite and normalized to [0, 1]")
        height, width = image.shape[-2:]
        native_boxes = target["boxes"].float()
        native_masks = target["masks"]
        if native_boxes.shape != (len(native_masks), 4) or native_masks.shape[1:] != (height, width):
            raise ValueError("Boxes and instance masks must describe the same native image")
        if len(native_boxes) and (
            not torch.isfinite(native_boxes).all()
            or (native_boxes[:, :2] < 0).any()
            or (native_boxes[:, 2:] > native_boxes.new_tensor([width, height])).any()
            or (native_boxes[:, 2:] <= native_boxes[:, :2]).any()
        ):
            raise ValueError("Instance boxes must have positive area within the image")
        images.append(F.interpolate(
            image[None], size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True,
        )[0])
        for mask in native_masks:
            array = mask.cpu().numpy()
            # Equivalent to an isin([0, 1]) check for unsigned masks, at a fraction of the cost.
            if array.dtype != np.uint8 or array.max() > 1 or not array.any():
                raise ValueError("Each source instance must have a nonempty binary COCO mask")
            scaled = cv2.resize(array.astype(np.float32), (mask_size, mask_size), interpolation=cv2.INTER_AREA)
            masks.append(torch.from_numpy(scaled))
        centers = (native_boxes[:, :2] + native_boxes[:, 2:]) / 2
        sizes = native_boxes[:, 2:] - native_boxes[:, :2]
        boxes.append(torch.cat((centers, sizes), dim=1) / native_boxes.new_tensor([width, height, width, height]))
        batch_indices.append(torch.full((len(native_boxes),), image_number, dtype=torch.int64))
    count = sum(len(box) for box in boxes)
    return {
        "img": torch.stack(images),
        "batch_idx": torch.cat(batch_indices),
        "cls": torch.zeros(count, 1, dtype=torch.float32),
        "bboxes": torch.cat(boxes),
        "masks": torch.stack(masks) if masks else torch.empty(0, mask_size, mask_size),
    }


def pretrained_path(config: dict) -> Path:
    """Absolute paths stay as written; relative ones resolve under the storage root."""
    from solar_filament.storage import storage_root

    path = Path(config["pretrained_checkpoint"])
    return path if path.is_absolute() else storage_root() / path


def make_model(config: dict, architecture: dict | None = None, pretrained: bool = True):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    # Ultralytics patches Windows cv2.imread to always return three dimensions.
    # The existing COCO dataset expects native two-dimensional grayscale reads.
    cv2.imread = _native_cv2_imread

    source = YOLO(str(pretrained_path(config)), task="segment") if architecture is None else None
    architecture = copy.deepcopy(source.model.yaml if architecture is None else architecture)
    model = SegmentationModel(architecture, nc=1, ch=3, verbose=False)
    model.names = {0: "filament"}
    if pretrained and source is not None:
        model.load(source.model, verbose=False)
    for name, parameter in model.named_parameters():
        if ".dfl." in name:
            parameter.requires_grad_(False)
    model.args = get_cfg(overrides={
        "task": "segment", "overlap_mask": False, "mask_ratio": config["mask_ratio"],
        "epochs": config["epochs"], "imgsz": config["image_size"],
    })
    return model


def schedule_end_to_end(model, epoch: int) -> bool:
    """Set an end-to-end (YOLO26) head's one-to-many/one-to-one loss weights for this epoch.

    The Ultralytics trainer calls criterion.update() after every epoch. Setting
    the weights from the epoch number keeps resumed runs on the same schedule.
    Returns whether the model is end-to-end; other models are left unchanged.
    """
    if not getattr(model, "end2end", False):
        return False
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    criterion = model.criterion
    criterion.updates = epoch
    criterion.o2m = criterion.decay(epoch)
    criterion.o2o = max(criterion.total - criterion.o2m, 0)
    return True


def learning_rate(config: dict, step: int) -> float:
    total = config["epochs"] * config["steps_per_epoch"]
    warmup = min(config.get("warmup_steps", 100), total - 1)
    base, floor = config["learning_rate"], config["minimum_learning_rate"]
    if not 0 <= step < total or not 0 < floor <= base:
        raise ValueError("Invalid learning-rate schedule or training position")
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (base - floor) * (1 + math.cos(math.pi * progress)) / 2


class ModelEMA:
    def __init__(self, model):
        self.model = copy.deepcopy(model).eval()
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = 0.9999 * (1 - math.exp(-self.updates / 2000))
        current = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(current[name].detach(), alpha=1 - decay)
            else:
                value.copy_(current[name])
