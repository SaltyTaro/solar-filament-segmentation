"""Compare bounded postprocessing candidates using cached held-out probabilities."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from solar_filament.coco import CocoIndex
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.postprocess import probability_instances
from solar_filament.probability_cache import cache_file, load_probability
from solar_filament.storage import annotations_path, artifact_path
from solar_filament.submission import validate_predictions, write_predictions


def default_candidates() -> list[dict]:
    candidates = [
        {"mask_threshold": threshold, "minimum_area": area, "maximum_area": 50000}
        for threshold in [0.25, 0.35, 0.45, 0.5, 0.55, 0.65]
        for area in [50, 150, 300]
    ]
    candidates += [
        {"mask_threshold": low, "seed_threshold": high, "minimum_area": 150, "maximum_area": 50000}
        for low, high in [(0.25, 0.55), (0.35, 0.65), (0.45, 0.75)]
    ]
    return candidates


def tune_probabilities(root: Path, probability_directory: Path, output: Path, fold: int = 0) -> dict:
    probability_directory = artifact_path(probability_directory, root)
    output = artifact_path(output, root)
    output.mkdir(parents=True, exist_ok=True)
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
    names = sorted(folds.loc[folds["fold"] == fold, "filename"].tolist())
    if not names:
        raise ValueError(f"No physical images in fold {fold}")
    missing = []
    for name in names:
        try:
            cache_file(probability_directory, Path(name).stem)
        except FileNotFoundError:
            missing.append(name)
    if missing:
        raise ValueError(f"Probability cache is incomplete: {len(missing)} missing images")
    candidates = default_candidates()
    total = [PQCounts() for _ in candidates]
    rows = []
    work_directory = output / "work"
    work_directory.mkdir(exist_ok=True)
    signature = hashlib.sha256(
        json.dumps({"fold": fold, "candidates": candidates}, sort_keys=True).encode()
    ).hexdigest()
    started = time.monotonic()
    cache_digest = hashlib.sha256()
    for name in tqdm(names, desc="Postprocessing sweep", unit="image"):
        stem = Path(name).stem
        probability = load_probability(probability_directory, stem)
        cache_digest.update(stem.encode())
        cache_digest.update(probability.tobytes())
        file_sha256 = hashlib.sha256(probability.tobytes()).hexdigest()
        progress_file = work_directory / (stem + ".json")
        if progress_file.is_file():
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
            if progress["signature"] == signature and progress["file_sha256"] == file_sha256:
                for item in progress["rows"]:
                    total[item["candidate"]] += PQCounts(item["tp"], item["fp"], item["fn"], item["sum_iou"])
                rows.extend(progress["rows"])
                continue
        image_rows = []
        gt = [(record["id"], index.record_rles(record["id"])) for record in index.images_by_filename[name]]
        for number, config in enumerate(candidates):
            predictions = probability_instances(probability, config)
            for record_id, rles in gt:
                counts = reference_counts(overlap_matrix(rles, predictions))
                total[number] += counts
                image_rows.append(
                    {"candidate": number, "filename": name, "annotator_image": record_id, **counts.scores()}
                )
        rows.extend(image_rows)
        progress = {"signature": signature, "file_sha256": file_sha256, "rows": image_rows}
        temporary = progress_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress), encoding="utf-8")
        temporary.replace(progress_file)
    ranking = [
        {"candidate": number, **config, **total[number].scores()} for number, config in enumerate(candidates)
    ]
    ranking.sort(key=lambda item: (-item["pq"], item["candidate"]))
    pd.DataFrame(ranking).to_csv(output / "candidate-ranking.csv", index=False, lineterminator="\n")
    pd.DataFrame(rows).to_csv(output / "candidate-per-record.csv", index=False, lineterminator="\n")
    selected = ranking[0]
    config = candidates[selected["candidate"]]
    predictions = {}
    for name in names:
        stem = Path(name).stem
        probability = load_probability(probability_directory, stem)
        rles = probability_instances(probability, config)
        if rles:
            predictions[stem] = rles
    validation = validate_predictions(predictions, {Path(name).stem for name in names})
    write_predictions(output / "predictions.csv", predictions)
    result = {
        "fold": fold,
        "physical_images": len(names),
        "probability_directory": str(probability_directory),
        "probability_sha256": cache_digest.hexdigest(),
        "candidate_count": len(candidates),
        "selected": selected,
        "config": config,
        "submission_validation": validation,
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Exploratory selection on this fold; confirm frozen parameters with a separately trained fold before promotion.",
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected candidate={selected['candidate']} exploratory_pq={selected['pq']:.5f}", flush=True)
    return result
