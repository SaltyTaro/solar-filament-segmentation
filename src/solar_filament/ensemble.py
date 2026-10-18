"""Pair binary instances from models sharing the same held-out observations."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from solar_filament.coco import encode_local_mask
from solar_filament.metrics import overlap_matrix


def fuse_binary_instances(
    first: list[dict], second: list[dict], *, mode: str, retain_unmatched_first: bool,
    pair_iou: float = 0.25, minimum_area: int = 150, maximum_area: int = 50000,
) -> list[dict]:
    """Match identities one-to-one, combine masks, then assign disjoint pixels.

    Binary masks do not retain learned confidence. Paired instances therefore
    precede unmatched first-model instances, ordered by cross-model IoU.
    This declared rule is not confidence averaging or probability ensembling.
    """
    if mode not in ("first", "union", "intersection"):
        raise ValueError("Binary fusion mode must be first, union or intersection")
    if not 0 < pair_iou < 1 or not 0 < minimum_area <= maximum_area:
        raise ValueError("Invalid binary fusion thresholds")
    all_rles = first + second
    if not all_rles:
        return []
    shape = tuple(all_rles[0]["size"])
    if len(shape) != 2 or any(tuple(rle["size"]) != shape for rle in all_rles):
        raise ValueError("All ensemble masks must use the same native image dimensions")
    iou = overlap_matrix(first, second)
    matched, candidates = set(), []
    if first and second:
        valid = iou > pair_iou
        # Maximum cardinality, followed by maximum total correspondence IoU.
        values = valid * (min(len(first), len(second)) + 1 + iou)
        rows, columns = linear_sum_assignment(values, maximize=True)
        for left, right in zip(rows, columns):
            if not valid[left, right]:
                continue
            matched.add(int(left))
            rle = first[left] if mode == "first" else mask_utils.merge(
                [first[left], second[right]], intersect=mode == "intersection"
            )
            candidates.append((float(iou[left, right]), int(left), rle))
    if retain_unmatched_first:
        candidates.extend((-1.0, number, rle) for number, rle in enumerate(first) if number not in matched)
    occupied = np.zeros(shape, dtype=bool)
    instances = []
    for _, _, rle in sorted(candidates, key=lambda row: (-row[0], row[1])):
        # Native ownership remains explicit even when unioned masks touch.
        selected = mask_utils.decode(rle).astype(bool) & ~occupied
        ys, xs = np.nonzero(selected)
        if not minimum_area <= len(xs) <= maximum_area:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        instances.append(encode_local_mask(selected[y0:y1, x0:x1], x0, y0, *shape))
        occupied |= selected
    return instances
