"""Dihedral test-time views and instance fusion across views, folds and model families.

Candidates from every source are clustered greedily in score order by mask
IoU, at most one member per source. A fused instance keeps the pixels drawn by
at least ``pixel_vote`` of its members and scores the mean member score over
all sources, so a candidate missed by a source contributes zero.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from solar_filament.coco import encode_local_mask


def dihedral(image: np.ndarray, view: int) -> np.ndarray:
    if not 0 <= view < 8:
        raise ValueError("Dihedral view must be in [0, 8)")
    image = np.rot90(image, view % 4)
    return image[:, ::-1] if view >= 4 else image


def inverse_dihedral(mask: np.ndarray, view: int) -> np.ndarray:
    """Undo ``dihedral`` on the last two axes."""
    if view >= 4:
        mask = mask[..., :, ::-1]
    return np.rot90(mask, -(view % 4), axes=(-2, -1))


def cluster_candidates(candidates: list[dict], iou_threshold: float = 0.5) -> list[list[dict]]:
    """Greedy score-ordered clustering; each candidate needs ``score``, ``rle`` and ``source``."""
    ordered = sorted(candidates, key=lambda candidate: -candidate["score"])
    clusters, seeds, sources, boxes = [], [], [], []
    for candidate in ordered:
        box = mask_utils.toBbox(candidate["rle"])
        target = None
        if seeds:
            seed_boxes = np.asarray(boxes)
            touching = np.flatnonzero(
                (seed_boxes[:, 0] < box[0] + box[2]) & (box[0] < seed_boxes[:, 0] + seed_boxes[:, 2])
                & (seed_boxes[:, 1] < box[1] + box[3]) & (box[1] < seed_boxes[:, 1] + seed_boxes[:, 3])
            )
            if len(touching):
                overlaps = mask_utils.iou([seeds[number] for number in touching], [candidate["rle"]], [0])[:, 0]
                for order in np.argsort(-overlaps):
                    if overlaps[order] < iou_threshold:
                        break
                    if candidate["source"] not in sources[touching[order]]:
                        target = int(touching[order])
                        break
        if target is None:
            clusters.append([candidate])
            seeds.append(candidate["rle"])
            sources.append({candidate["source"]})
            boxes.append(box)
        else:
            clusters[target].append(candidate)
            sources[target].add(candidate["source"])
    return clusters


def fuse_cluster(members: list[dict], source_count: int, pixel_vote: float = 0.5) -> dict | None:
    boxes = np.asarray([mask_utils.toBbox(member["rle"]) for member in members])
    x0, y0 = int(boxes[:, 0].min()), int(boxes[:, 1].min())
    x1, y1 = int((boxes[:, 0] + boxes[:, 2]).max()), int((boxes[:, 1] + boxes[:, 3]).max())
    votes = np.zeros((y1 - y0, x1 - x0), dtype=np.uint16)
    for member in members:
        votes += mask_utils.decode(member["rle"])[y0:y1, x0:x1]
    mask = votes >= max(1, int(np.ceil(pixel_vote * len(members) - 1e-9)))
    if not mask.any():
        return None
    height, width = members[0]["rle"]["size"]
    scores = np.asarray([member["score"] for member in members])
    return {
        "rle": encode_local_mask(mask, x0, y0, height, width),
        "score": float(scores.sum() / source_count),
        "max_score": float(scores.max()),
        "votes": len(members) / source_count,
        "members": len(members),
        "pixel_agreement": float((votes[mask] / len(members)).mean()),
        "families": sorted({member.get("family", "") for member in members}),
    }


def fuse(candidates: list[dict], source_count: int, iou_threshold: float = 0.5, pixel_vote: float = 0.5,
         family_sources: dict | None = None) -> list[dict]:
    """``family_sources`` maps each family to its source count, adding per-family vote and mean-score features."""
    fused = []
    for members in cluster_candidates(candidates, iou_threshold):
        instance = fuse_cluster(members, source_count, pixel_vote)
        if instance is None:
            continue
        for family, count in (family_sources or {}).items():
            scores = [member["score"] for member in members if member.get("family", "") == family]
            instance[f"votes_{family}"] = len(scores) / count
            instance[f"score_{family}"] = sum(scores) / count
            instance[f"max_score_{family}"] = max(scores, default=0.0)
        fused.append(instance)
    return sorted(fused, key=lambda instance: -instance["score"])


def own_pixels(instances: list[dict], minimum_area: int, maximum_area: int, key: str = "score") -> list[dict]:
    """Assign each pixel to the highest-ranked instance; drop out-of-range areas."""
    occupied = None
    kept = []
    for instance in sorted(instances, key=lambda item: -item[key]):
        mask = mask_utils.decode(instance["rle"]).astype(bool)
        if occupied is None:
            occupied = np.zeros_like(mask)
        mask &= ~occupied
        ys, xs = np.nonzero(mask)
        if not minimum_area <= len(xs) <= maximum_area:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        kept.append({**instance, "rle": encode_local_mask(mask[y0:y1, x0:x1], x0, y0, *mask.shape),
                     "owned_area": int(len(xs))})
        occupied |= mask
    return kept
