"""Convert native-resolution probability maps into disjoint filament instances."""

from __future__ import annotations

import cv2
import numpy as np

from solar_filament.coco import encode_local_mask


def probability_instances(probability: np.ndarray, config: dict) -> list[dict]:
    threshold = float(config["mask_threshold"])
    minimum_area = int(config.get("minimum_area", 150))
    maximum_area = int(config.get("maximum_area", 50000))
    seed_threshold = float(config.get("seed_threshold", threshold))
    minimum_confidence = float(config.get("minimum_confidence", 0.0))
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Probability map must be a finite two-dimensional array")
    if not 0 <= threshold <= seed_threshold <= 1 or not 0 <= minimum_confidence <= 1:
        raise ValueError("Thresholds must be in [0, 1], with seed threshold at least mask threshold")
    if not 1 <= minimum_area <= maximum_area:
        raise ValueError("Invalid component area limits")
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    instances = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, width, height = stats[component, :4]
        local_mask = labels[y : y + height, x : x + width] == component
        values = probability[y : y + height, x : x + width][local_mask]
        if float(values.max()) < seed_threshold or float(values.mean()) < minimum_confidence:
            continue
        instances.append(encode_local_mask(local_mask, int(x), int(y), *binary.shape))
    return instances
