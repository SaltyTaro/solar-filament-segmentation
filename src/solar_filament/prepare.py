"""Prepare compact mask targets and image fingerprints on project storage."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from solar_filament.coco import CocoIndex
from solar_filament.storage import annotations_path, dataset_root


def prepare_targets(root: Path) -> dict:
    index = CocoIndex(annotations_path(root))
    target_root = root / "data" / "processed" / "union-targets"
    target_root.mkdir(parents=True, exist_ok=True)
    max_area_error = 0.0
    zero_masks = []
    overlap_records = []
    for record_id in tqdm(sorted(index.images_by_id), desc="Prepare annotator targets", unit="record"):
        rles = index.record_rles(record_id)
        areas = np.asarray(mask_utils.area(rles))
        expected = np.asarray([item["area"] for item in index.annotations_by_image[record_id]])
        if len(areas):
            max_area_error = max(max_area_error, float(np.abs(areas - expected).max()))
            zero_masks.extend(
                item["id"] for item, area in zip(index.annotations_by_image[record_id], areas) if area == 0
            )
            overlap = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
            np.fill_diagonal(overlap, 0)
            if (overlap > 0).any():
                overlap_records.append(record_id)
        destination = target_root / f"{record_id}.png"
        if not destination.exists():
            union = index.union_mask(record_id)
            Image.fromarray(union * 255).save(destination, compress_level=3)
    summary = {
        "annotator_targets": len(index.images_by_id),
        "annotation_masks_checked": len(index.data["annotations"]),
        "maximum_recorded_area_difference_pixels": max_area_error,
        "zero_area_annotation_ids": zero_masks,
        "records_with_overlapping_annotation_masks": overlap_records,
        "target_directory": str(target_root),
    }
    destination = root / "data" / "manifests" / "mask-audit.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def image_fingerprints(root: Path, workers: int = 4) -> dict:
    paths = sorted((dataset_root(root) / "train" / "train_images").glob("*.jpeg"))

    def fingerprint(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)[:8, :8]
        bits = dct > np.median(dct.ravel()[1:])
        bits[0, 0] = False
        return {"filename": path.name, "phash": np.packbits(bits.ravel()).tobytes().hex()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            tqdm(pool.map(fingerprint, paths), total=len(paths), desc="Image fingerprints", unit="image")
        )
    hashes = np.asarray([int(record["phash"], 16) for record in records], dtype=np.uint64)
    pairs = []
    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            distance = int(hashes[first] ^ hashes[second]).bit_count()
            if distance <= 3:
                pairs.append(
                    {
                        "first": records[first]["filename"],
                        "second": records[second]["filename"],
                        "hamming_distance": distance,
                    }
                )
    result = {
        "description": "pHash candidates require visual confirmation; low distance alone is not evidence of duplicate solar observations",
        "images": records,
        "near_duplicate_candidates": pairs,
    }
    (root / "data" / "manifests" / "image-fingerprints.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {"images_fingerprinted": len(records), "near_duplicate_candidates": len(pairs)}
