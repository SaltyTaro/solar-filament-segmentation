"""An inexpensive image-processing reference with native-resolution masks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from solar_filament.storage import dataset_root
from solar_filament.submission import validate_predictions, write_predictions


def filament_candidates(
    image: np.ndarray,
    darkness_threshold: float = 8.0,
    minimum_area: int = 150,
    encoded: bool = False,
) -> list:
    """Normalize local illumination, then extract elongated dark components."""
    _, rough_disk = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(rough_disk, connectivity=8)
    if count <= 1:
        return []
    disk_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = (labels == disk_label).astype(np.uint8)
    disk = cv2.morphologyEx(disk, cv2.MORPH_CLOSE, np.ones((21, 21), dtype=np.uint8))
    disk = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    background = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 20)
    darkness = background - image.astype(np.float32)
    binary = ((darkness > darkness_threshold) & disk.astype(bool)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    masks = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= 50000:
            continue
        x, y, width, height = stats[component, :4]
        local_mask = labels[y : y + height, x : x + width] == component
        ys, xs = np.nonzero(local_mask)
        covariance = np.cov(np.stack([xs, ys]))
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = np.sqrt((eigenvalues[-1] + 1) / (eigenvalues[0] + 1))
        # Compact dark sunspots are common confounders, not reliable filament seeds.
        if elongation < 1.6:
            continue
        mask = np.zeros_like(binary)
        mask[y : y + height, x : x + width] = local_mask
        masks.append(mask_utils.encode(np.asfortranarray(mask)) if encoded else mask.astype(np.uint8))
    return masks


def predict_classical(
    root: Path,
    output: Path,
    fold: int = 0,
    threshold: float = 8.0,
    minimum_area: int = 150,
    limit: int | None = None,
) -> dict:
    folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
    names = sorted(folds.loc[folds["fold"] == fold, "filename"].tolist())
    if limit is not None:
        names = names[:limit]
    predictions = {}
    started = time.monotonic()
    for name in tqdm(names, desc="Classical reference", unit="image"):
        image = cv2.imread(str(dataset_root(root) / "train" / "train_images" / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(name)
        predictions[Path(name).stem] = filament_candidates(image, threshold, minimum_area, encoded=True)
    predictions = {stem: rles for stem, rles in predictions.items() if rles}
    validation = validate_predictions(predictions, {Path(name).stem for name in names})
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(output / "predictions.csv", predictions)
    config = {
        "model": "local-darkness-elongated-components",
        "fold": fold,
        "threshold": threshold,
        "minimum_area": minimum_area,
        "limit": limit,
        "images": len(names),
        "inference_seconds": time.monotonic() - started,
        "validation": validation,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config
