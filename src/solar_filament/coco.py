"""COCO indexing and lossless mask conversion for the official annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def annotation_rle(annotation: dict, height: int = 2048, width: int = 2048) -> dict:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle["counts"], list):
            return mask_utils.frPyObjects(rle, height, width)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    if not segmentation:
        return encode_mask(np.zeros((height, width), dtype=np.uint8))
    polygons = mask_utils.frPyObjects(segmentation, height, width)
    return mask_utils.merge(polygons)


def encode_mask(mask: np.ndarray) -> dict:
    if mask.ndim != 2:
        raise ValueError("A filament mask must have two dimensions")
    if not np.isin(mask, [0, 1, False, True]).all():
        raise ValueError("A filament mask must be binary")
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


def encode_local_mask(mask: np.ndarray, x: int, y: int, height: int = 2048, width: int = 2048) -> dict:
    """Encode a crop at its full-image location without allocating a dense image."""
    if mask.ndim != 2 or not np.isin(mask, [0, 1, False, True]).all():
        raise ValueError("A local instance mask must be binary and two-dimensional")
    crop_height, crop_width = mask.shape
    if x < 0 or y < 0 or x + crop_width > width or y + crop_height > height:
        raise ValueError("Local mask extends beyond the output image")
    local = np.flatnonzero(mask.T.ravel())
    if not len(local):
        counts = [height * width]
    else:
        global_pixels = (local // crop_height + x) * height + local % crop_height + y
        breaks = np.flatnonzero(np.diff(global_pixels) > 1)
        starts = global_pixels[np.r_[0, breaks + 1]]
        ends = global_pixels[np.r_[breaks, len(global_pixels) - 1]]
        runs = np.empty(2 * len(starts) + 1, dtype=np.int64)
        runs[0] = starts[0]
        runs[1:-1:2] = ends - starts + 1
        runs[2:-1:2] = starts[1:] - ends[:-1] - 1
        runs[-1] = height * width - ends[-1] - 1
        counts = runs.tolist()
    return mask_utils.frPyObjects({"size": [height, width], "counts": counts}, height, width)


def decode_counts(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    if not isinstance(counts, str) or not counts or not counts.isascii():
        raise ValueError("COCO counts must be a nonempty ASCII string")
    validate_compressed_counts(counts, height * width)
    return mask_utils.decode({"size": [height, width], "counts": counts.encode("ascii")})


def validate_compressed_counts(counts: str, pixels: int) -> None:
    """Check compressed COCO runs before passing untrusted input to native code."""
    runs = []
    position = 0
    total = 0
    while position < len(counts):
        value = 0
        shift = 0
        while True:
            if position == len(counts) or shift > 35:
                raise ValueError("Truncated or oversized compressed COCO run")
            character = ord(counts[position]) - 48
            position += 1
            if not 0 <= character <= 63:
                raise ValueError("Invalid character in compressed COCO counts")
            value |= (character & 31) << shift
            shift += 5
            if not character & 32:
                if character & 16:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError("Negative COCO run length")
        runs.append(value)
        total += value
        if total > pixels:
            raise ValueError("COCO counts exceed the mask dimensions")
    if total != pixels:
        raise ValueError("COCO counts do not cover the expected mask dimensions")


class CocoIndex:
    def __init__(self, annotation_file: Path):
        self.path = annotation_file
        self.data = json.loads(annotation_file.read_text(encoding="utf-8"))
        self.images_by_id = {image["id"]: image for image in self.data["images"]}
        if len(self.images_by_id) != len(self.data["images"]):
            raise ValueError("Duplicate annotator-image IDs")
        self.annotations_by_id = {annotation["id"]: annotation for annotation in self.data["annotations"]}
        if len(self.annotations_by_id) != len(self.data["annotations"]):
            raise ValueError("Duplicate annotation IDs")
        self.annotations_by_image = defaultdict(list)
        self.images_by_filename = defaultdict(list)
        for annotation in self.data["annotations"]:
            if annotation["image_id"] not in self.images_by_id:
                raise ValueError("Annotation points to an unknown image record")
            self.annotations_by_image[annotation["image_id"]].append(annotation)
        for image in self.data["images"]:
            self.images_by_filename[image["file_name"]].append(image)

    def record_rles(self, image_id: str) -> list[dict]:
        image = self.images_by_id[image_id]
        return [
            annotation_rle(annotation, image["height"], image["width"])
            for annotation in self.annotations_by_image[image_id]
        ]

    def union_mask(self, image_id: str) -> np.ndarray:
        rles = self.record_rles(image_id)
        image = self.images_by_id[image_id]
        if not rles:
            return np.zeros((image["height"], image["width"]), dtype=np.uint8)
        return mask_utils.decode(mask_utils.merge(rles))
