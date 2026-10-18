"""Strict submission parsing and validation before using a Kaggle slot."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from solar_filament.coco import encode_mask, validate_compressed_counts

IMAGE_ID = re.compile(r"^\d{14}(?:Bh|Ch|Lh|Mh|Th|Uh)$")
COLUMNS = ["filament_id", "segmentation_rle"]


def load_predictions(path: Path) -> dict[str, list[dict]]:
    predictions = defaultdict(list)
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected columns {COLUMNS}, found {reader.fieldnames}")
        for number, row in enumerate(reader, start=2):
            identifier = row["filament_id"]
            if not identifier or "_" not in identifier:
                raise ValueError(f"Invalid filament ID on CSV line {number}")
            stem, suffix = identifier.rsplit("_", 1)
            if not IMAGE_ID.fullmatch(stem) or not suffix:
                raise ValueError(f"Invalid filament ID on CSV line {number}")
            if identifier in seen:
                raise ValueError(f"Duplicate filament ID: {identifier}")
            seen.add(identifier)
            counts = row["segmentation_rle"]
            if not counts or not counts.isascii():
                raise ValueError(f"Invalid RLE counts on CSV line {number}")
            predictions[stem].append({"size": [2048, 2048], "counts": counts.encode("ascii")})
    return dict(predictions)


def validate_predictions(predictions: dict[str, list[dict]], expected_stems: set[str]) -> dict:
    unknown = predictions.keys() - expected_stems
    if unknown:
        raise ValueError(f"Predictions contain unknown image IDs: {sorted(unknown)[:5]}")
    areas = []
    for stem, rles in predictions.items():
        for rle in rles:
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
            try:
                validate_compressed_counts(counts, 2048 * 2048)
            except Exception as error:
                raise ValueError(f"Undecodable RLE for {stem}") from error
            if rle.get("size") != [2048, 2048]:
                raise ValueError(f"Invalid mask dimensions for {stem}")
            area = int(mask_utils.area(rle))
            if area == 0:
                raise ValueError(f"Empty predicted instance for {stem}")
            areas.append(area)
        if len(rles) > 1:
            overlaps = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
            np.fill_diagonal(overlaps, 0)
            if (overlaps > 0).any():
                raise ValueError(f"Overlapping predicted instances for {stem}")
    return {
        "expected_images": len(expected_stems),
        "images_with_predictions": len(predictions),
        "images_without_predictions": sorted(expected_stems - predictions.keys()),
        "filament_rows": len(areas),
        "minimum_area": min(areas) if areas else None,
        "maximum_area": max(areas) if areas else None,
        "overlapping_instances": 0,
        "empty_instances": 0,
    }


def write_predictions(path: Path, predictions: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for stem in sorted(predictions):
            for index, rle in enumerate(predictions[stem], start=1):
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("ascii")
                writer.writerow({"filament_id": f"{stem}_{index}", "segmentation_rle": counts})


def masks_to_rles(masks: list[np.ndarray]) -> list[dict]:
    return [encode_mask(mask) for mask in masks]
