"""Pilot detector boxes plus native semantic crops using held-out predictions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from solar_filament.coco import CocoIndex, encode_local_mask
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.probability_cache import load_probability
from solar_filament.storage import annotations_path, artifact_path
from solar_filament.submission import validate_predictions, write_predictions


def hybrid_instances(probability: np.ndarray, proposals: list[dict], config: dict) -> list[dict]:
    height, width = probability.shape
    occupied = np.zeros(probability.shape, dtype=bool)
    instances = []
    for proposal in sorted(proposals, key=lambda item: -item["score"]):
        if proposal["score"] < config["score_threshold"]:
            continue
        left, top, right, bottom = proposal["bbox_xyxy"]
        padding = max(8, min(64, 0.1 * max(right - left, bottom - top)))
        x0, y0 = max(0, int(left - padding)), max(0, int(top - padding))
        x1, y1 = min(width, int(np.ceil(right + padding))), min(height, int(np.ceil(bottom + padding)))
        if x1 <= x0 or y1 <= y0:
            continue
        local_probability = probability[y0:y1, x0:x1]
        binary = (local_probability >= config["mask_threshold"]).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            continue
        roi_left, roi_top = max(0, int(left) - x0), max(0, int(top) - y0)
        roi_right, roi_bottom = (
            min(x1 - x0, int(np.ceil(right)) - x0),
            min(y1 - y0, int(np.ceil(bottom)) - y0),
        )
        best_component, best_value = None, -1.0
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if not config["minimum_area"] <= area <= config["maximum_area"]:
                continue
            selected = labels == component
            inside = selected[roi_top:roi_bottom, roi_left:roi_right]
            if not inside.any():
                continue
            value = float(local_probability[roi_top:roi_bottom, roi_left:roi_right][inside].sum())
            if value > best_value:
                best_component, best_value = component, value
        if best_component is None:
            continue
        selected = (labels == best_component) & ~occupied[y0:y1, x0:x1]
        area = int(selected.sum())
        if not config["minimum_area"] <= area <= config["maximum_area"]:
            continue
        instances.append(encode_local_mask(selected, x0, y0, height, width))
        occupied[y0:y1, x0:x1] |= selected
    return instances


def evaluate_hybrid(
    root: Path, probability_directory: Path, proposal_directory: Path, output: Path, fold: int = 0
) -> dict:
    probability_directory = artifact_path(probability_directory, root)
    proposal_directory = artifact_path(proposal_directory, root)
    output = artifact_path(output, root)
    output.mkdir(parents=True, exist_ok=True)
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data/splits/folds.csv")
    names = sorted(folds.loc[folds.fold == fold, "filename"].tolist())
    candidates = [
        {"score_threshold": score, "mask_threshold": mask, "minimum_area": 150, "maximum_area": 50000}
        for score in [0.1, 0.3]
        for mask in [0.35, 0.5, 0.65]
    ]
    totals = [PQCounts() for _ in candidates]
    predictions = [{} for _ in candidates]
    rows = []
    started = time.monotonic()
    for name in tqdm(names, desc="Native crop hybrid", unit="image"):
        stem = Path(name).stem
        probability = load_probability(probability_directory, stem)
        proposals = json.loads((proposal_directory / (stem + ".json")).read_text(encoding="utf-8"))
        ground_truth = [
            (record["id"], index.record_rles(record["id"])) for record in index.images_by_filename[name]
        ]
        for number, config in enumerate(candidates):
            rles = hybrid_instances(probability, proposals, config)
            if rles:
                predictions[number][stem] = rles
            for record_id, gt in ground_truth:
                counts = reference_counts(overlap_matrix(gt, rles))
                totals[number] += counts
                rows.append(
                    {"candidate": number, "filename": name, "annotator_image": record_id, **counts.scores()}
                )
    ranking = [
        {"candidate": number, **config, **totals[number].scores()} for number, config in enumerate(candidates)
    ]
    ranking.sort(key=lambda item: -item["pq"])
    pd.DataFrame(ranking).to_csv(output / "candidate-ranking.csv", index=False, lineterminator="\n")
    pd.DataFrame(rows).to_csv(output / "candidate-per-record.csv", index=False, lineterminator="\n")
    selected = ranking[0]
    validation = validate_predictions(predictions[selected["candidate"]], {Path(name).stem for name in names})
    write_predictions(output / "predictions.csv", predictions[selected["candidate"]])
    result = {
        "fold": fold,
        "physical_images": len(names),
        "selected": selected,
        "submission_validation": validation,
        "probability_directory": str(probability_directory),
        "proposal_directory": str(proposal_directory),
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Exploratory end-to-end detector/semantic-crop pilot on held-out fold; no trained prompt refiner or independent confirmation yet.",
    }
    (output / "hybrid-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
