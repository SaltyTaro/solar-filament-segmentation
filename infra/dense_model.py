"""Spine-aware dense filament model: a ConvNeXt U-Net at native resolution.

Every annotation is one polygon, 99.5% of instances are a single connected
region and instances of one annotator almost never touch (6 of 5,879 pairs
within 5 px), so instances are the connected components of a filament map.
The model predicts two soft annotator-consensus maps at full resolution: the
filament (fraction of annotators drawing each pixel) and its supplied spine
(the organizer-permitted centre line, drawn 3 px wide). Masks keep the native
pixel grid instead of the quarter-resolution masks of the box and query
families, whose median filament width is about 3 mask pixels.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import convnext_small

from solar_filament.coco import CocoIndex, encode_local_mask
from solar_filament.sampling import coverage_index
from solar_filament.storage import annotations_path, dataset_root, storage_root

GRAY_MEAN, GRAY_STD = 0.449, 0.226
SPINE_WIDTH = 3


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.GELU(),
        )


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs, skip):
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([inputs, skip], dim=1))


class ConvNeXtUNet(nn.Module):
    """ConvNeXt-Small stages at 1/4-1/32, plus convolutions on the image at 1/2 and full resolution."""

    def __init__(self, weights: Path | None = None):
        super().__init__()
        encoder = convnext_small()
        if weights is not None:
            encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        stem = encoder.features[0][0]
        gray = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, bias=True)
        with torch.no_grad():
            # Summing reproduces the stem on three replicated gray channels.
            gray.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
            gray.bias.copy_(stem.bias)
        encoder.features[0][0] = gray
        self.stages = encoder.features
        self.full_resolution = ConvBlock(1, 16)
        self.half_resolution = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
                                  ConvBlock(32, 32))
        self.up32 = Up(768, 384, 256)
        self.up16 = Up(256, 192, 128)
        self.up8 = Up(128, 96, 64)
        self.up4 = Up(64, 32, 48)
        self.up2 = Up(48, 16, 32)
        self.head = nn.Conv2d(32, 2, 1)

    def forward(self, inputs):
        """``inputs``: normalized (B, 1, H, W) with H and W divisible by 32; returns filament and spine logits."""
        features, value = [], inputs
        for number, stage in enumerate(self.stages):
            value = stage(value)
            if number in (1, 3, 5):
                features.append(value)
        decoded = self.up32(value, features[2])
        decoded = self.up16(decoded, features[1])
        decoded = self.up8(decoded, features[0])
        decoded = self.up4(decoded, self.half_resolution(inputs))
        return self.head(self.up2(decoded, self.full_resolution(inputs)))


def make_model(config: dict, pretrained: bool = True) -> ConvNeXtUNet:
    return ConvNeXtUNet(storage_root() / config["pretrained_checkpoint"] if pretrained else None)


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - GRAY_MEAN) / GRAY_STD


def amp_dtype(config: dict) -> torch.dtype:
    """bfloat16 on Ampere and newer GPUs; float16 (with loss scaling in the loop) on T4 or P100."""
    return getattr(torch, config.get("amp_dtype", "bfloat16"))


def spine_points(annotation: dict) -> np.ndarray:
    spine = annotation.get("spine")
    spine = json.loads(spine) if isinstance(spine, str) else spine
    return np.asarray(spine or [], dtype=np.float64).reshape(-1, 2)


def consensus_targets(index: CocoIndex, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of the image's annotators drawing each pixel as filament and as spine (uint8, 0-255)."""
    records = index.images_by_filename[name]
    height, width = records[0]["height"], records[0]["width"]
    filament = np.zeros((height, width), dtype=np.float32)
    spine = np.zeros((height, width), dtype=np.float32)
    for record in records:
        filament += index.union_mask(record["id"])
        canvas = np.zeros((height, width), dtype=np.uint8)
        for annotation in index.annotations_by_image[record["id"]]:
            points = spine_points(annotation)
            if len(points) >= 2:
                cv2.polylines(canvas, [np.round(points).astype(np.int32)], False, 1, SPINE_WIDTH)
        spine += canvas
    scale = 255 / len(records)
    return np.round(filament * scale).astype(np.uint8), np.round(spine * scale).astype(np.uint8)


class DenseDataset(Dataset):
    """Native-resolution crops of training images with their consensus filament and spine maps.

    Crops are scaled by a log-uniform factor, flipped/rotated by one of eight
    dihedral views and brightness-jittered; targets follow the same geometry.
    """

    def __init__(self, root: Path, config: dict, length: int):
        self.root, self.config, self.length, self.epoch = root, config, length, 0
        self.index = CocoIndex(annotations_path(root))
        folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
        self.names = sorted(folds.loc[folds.fold != config["fold"], "filename"].tolist())
        self.cache = OrderedDict()

    def __len__(self):
        return self.length

    def cached(self, name: str):
        if name not in self.cache:
            image = cv2.imread(str(dataset_root(self.root) / "train" / "train_images" / name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(name)
            self.cache[name] = (image, *consensus_targets(self.index, name))
        self.cache.move_to_end(name)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return self.cache[name]

    def __getitem__(self, sample: int):
        rng = np.random.default_rng(self.config["seed"] + self.epoch * self.length + sample)
        name = self.names[coverage_index(len(self.names), self.epoch * self.length + sample, self.config["seed"])]
        image, filament, spine = self.cached(name)
        crop = self.config["crop_size"]
        low, high = self.config.get("scale_jitter") or (1.0, 1.0)
        extent = min(image.shape[0], int(round(crop / np.exp(rng.uniform(np.log(low), np.log(high))))))
        y0, x0 = (int(rng.integers(0, size - extent + 1)) for size in image.shape)
        stack = np.stack([array[y0 : y0 + extent, x0 : x0 + extent] for array in (image, filament, spine)], axis=-1)
        if extent != crop:
            stack = cv2.resize(stack, (crop, crop), interpolation=cv2.INTER_AREA if extent > crop else cv2.INTER_LINEAR)
        view = int(rng.integers(8))
        stack = np.rot90(stack, view % 4)
        stack = stack[:, ::-1] if view >= 4 else stack
        arrays = np.ascontiguousarray(stack.transpose(2, 0, 1)).astype(np.float32) / 255
        arrays[0] = np.clip(arrays[0] * rng.uniform(0.9, 1.1) + rng.uniform(-0.03, 0.03), 0, 1)
        return torch.from_numpy(arrays[:1]), torch.from_numpy(arrays[1:])


def soft_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * targets).sum((-2, -1))
    return (1 - (2 * intersection + 1) / (probability.sum((-2, -1)) + targets.sum((-2, -1)) + 1)).mean()


def loss(model, images: torch.Tensor, targets: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    """BCE + soft Dice on the filament map; the spine map is weighted by ``spine_weight``."""
    with torch.autocast("cuda", dtype=amp_dtype(config), enabled=config["mixed_precision"]):
        logits = model(normalize(images))
    logits = logits.float()
    terms = {}
    for channel, name in enumerate(("filament", "spine")):
        terms[f"{name}_bce"] = F.binary_cross_entropy_with_logits(logits[:, channel], targets[:, channel])
        terms[f"{name}_dice"] = soft_dice(logits[:, channel], targets[:, channel])
    value = terms["filament_bce"] + terms["filament_dice"] + config["spine_weight"] * (terms["spine_bce"] + terms["spine_dice"])
    return value, {key: float(term.detach()) for key, term in terms.items()}


@torch.inference_mode()
def predict(model, image: np.ndarray, config: dict) -> torch.Tensor:
    """Filament and spine probabilities (2, H, W) for one uint8 grayscale image at native resolution."""
    tensor = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255).cuda()[None, None]
    with torch.autocast("cuda", dtype=amp_dtype(config), enabled=config["mixed_precision"]):
        logits = model(normalize(tensor))
    return logits[0].float().sigmoid()


def component_candidates(probability: np.ndarray, spine: np.ndarray, view: int, threshold: float,
                         minimum_area: int) -> list[dict]:
    """Connected components of the thresholded filament map, scored by their mean probability."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((probability >= threshold).astype(np.uint8), connectivity=8)
    records = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < minimum_area:
            continue
        local = labels[y : y + height, x : x + width] == label
        rle = encode_local_mask(local, x, y, *probability.shape)
        records.append({
            "score": float(probability[y : y + height, x : x + width][local].mean()), "area": area, "view": view,
            "spine": float(spine[y : y + height, x : x + width][local].max()),
            "rle": {"size": rle["size"], "counts": rle["counts"].decode("ascii")},
        })
    return records
