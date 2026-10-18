"""Image-derived descriptors of one predicted instance, shared by audits and rescoring."""

from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def instance_features(image: np.ndarray, rle: dict) -> dict:
    x, y, width, height = (int(value) for value in mask_utils.toBbox(rle))
    mask = mask_utils.decode(rle)[y : y + height, x : x + width].astype(bool)
    pad = 12
    y0, y1 = max(y - pad, 0), min(y + height + pad, image.shape[0])
    x0, x1 = max(x - pad, 0), min(x + width + pad, image.shape[1])
    region = image[y0:y1, x0:x1].astype(np.float32)
    inside = np.zeros(region.shape, dtype=bool)
    inside[y - y0 : y - y0 + height, x - x0 : x - x0 + width] = mask
    ring = cv2.dilate(inside.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~inside
    values = region[inside]
    mean_inside = float(values.mean()) if len(values) else 0.0
    mean_ring = float(region[ring].mean()) if ring.any() else mean_inside
    radius = ((x + width / 2 - 1024) ** 2 + (y + height / 2 - 1024) ** 2) ** 0.5 / 1024
    area = int(mask.sum())
    skeleton_proxy = area / max(1.0, 0.5 * (width + height))
    return {
        "area": area,
        "box_long": max(width, height),
        "elongation": max(width, height) / max(1, min(width, height)),
        "fill": float(area / max(1, width * height)),
        "mean_width": skeleton_proxy,
        "mean_inside": mean_inside,
        "minimum_inside": float(np.percentile(values, 5)) if len(values) else 0.0,
        "contrast": mean_ring - mean_inside,
        "relative_contrast": (mean_ring - mean_inside) / max(mean_ring, 1.0),
        "normalized_radius": radius,
    }
