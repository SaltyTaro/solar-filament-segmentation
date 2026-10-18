"""Compact native mask probabilities for CPU detector postprocessing sweeps."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from solar_filament.checkpoint_delta import checkpoint_source
from solar_filament.coco import CocoIndex, encode_local_mask
from solar_filament.metrics import PQCounts, overlap_matrix, reference_counts
from solar_filament.preprocessing import DetectorPreprocessing
from solar_filament.storage import annotations_path, artifact_path
from solar_filament.submission import validate_predictions, write_predictions


def pack_probabilities(scores: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    if masks.ndim != 3 or len(scores) != len(masks):
        raise ValueError("Expected one native mask probability image per confidence score")
    if not np.isfinite(masks).all() or np.any(masks < 0) or np.any(masks > 1):
        raise ValueError("Expected finite mask probabilities in [0, 1]")
    geometries, offsets, values = [], [0], []
    for mask in masks:
        ys, xs = np.nonzero(mask > 0)
        if len(xs):
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            crop = mask[y0:y1, x0:x1].astype(np.float32).ravel()
        else:
            x0, y0, x1, y1, crop = 0, 0, 0, 0, np.empty(0, dtype=np.float32)
        geometries.append([x0, y0, x1, y1])
        values.append(crop)
        offsets.append(offsets[-1] + len(crop))
    return {
        "scores": scores.astype(np.float32),
        "geometries": np.asarray(geometries, dtype=np.int32).reshape(-1, 4),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "probabilities": np.concatenate(values) if values else np.empty(0, dtype=np.float32),
        "shape": np.asarray(masks.shape[1:], dtype=np.int32),
    }


def cached_instances(packed: dict[str, np.ndarray], config: dict) -> list[dict]:
    height, width = map(int, packed["shape"])
    occupied = np.zeros((height, width), dtype=bool)
    instances = []
    for number in np.argsort(-packed["scores"]):
        if packed["scores"][number] < config["score_threshold"]:
            continue
        x0, y0, x1, y1 = map(int, packed["geometries"][number])
        if x0 == x1 or y0 == y1:
            continue
        start, stop = map(int, packed["offsets"][number : number + 2])
        probability = packed["probabilities"][start:stop].reshape(y1 - y0, x1 - x0)
        selected = (probability >= config["mask_threshold"]) & ~occupied[y0:y1, x0:x1]
        area = int(selected.sum())
        if not config["minimum_area"] <= area <= config["maximum_area"]:
            continue
        instances.append(encode_local_mask(selected, x0, y0, height, width))
        occupied[y0:y1, x0:x1] |= selected
    return instances


def cache_detector(root: Path, checkpoint_path: Path, output: Path, fold: int) -> dict:
    import torch

    from solar_filament.detection import make_detector
    from solar_filament.predict import file_digest, supplied_paths

    checkpoint_path = artifact_path(checkpoint_path, root)
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for detector probability caching")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    checkpoint = torch.load(checkpoint_source(checkpoint_path), map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    if config["fold"] != fold:
        raise ValueError("The detector checkpoint must exclude the requested validation fold")
    model = make_detector(config, pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    paths = supplied_paths(root, "validation", fold)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    expected = {
        "checkpoint_sha256": file_digest(checkpoint_path),
        "fold": fold,
        "physical_images": len(paths),
        "config": config,
    }
    if manifest_path.exists():
        original = json.loads(manifest_path.read_text())
        if any(original[key] != expected[key] for key in expected):
            raise ValueError("Existing detector cache belongs to a different checkpoint/configuration")
    manifest = {**expected, "checkpoint": str(checkpoint_path), "status": "running", "files": []}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    preprocess = DetectorPreprocessing(config)
    with torch.inference_mode():
        for number, path in enumerate(paths):
            destination = output / (path.stem + ".npz")
            if not destination.exists():
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (2048, 2048):
                    raise ValueError(f"Invalid supplied training JPEG: {path}")
                image = preprocess(image)
                tensor = torch.from_numpy(image.astype(np.float32)[None] / 255).repeat(3, 1, 1).cuda()
                with torch.autocast("cuda", enabled=config["mixed_precision"]):
                    result = model([tensor])[0]
                packed = pack_probabilities(
                    result["scores"].float().cpu().numpy(), result["masks"].float().cpu().numpy()[:, 0]
                )
                temporary = destination.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **packed)
                temporary.replace(destination)
                del result, tensor, packed
            manifest["files"].append(
                {
                    "name": destination.name,
                    "sha256": file_digest(destination),
                    "bytes": destination.stat().st_size,
                }
            )
            if (number + 1) % 20 == 0:
                print(f"Detector probability cache {number + 1}/{len(paths)}", flush=True)
    manifest.update(
        status="complete",
        elapsed_seconds=time.monotonic() - started,
        total_bytes=sum(item["bytes"] for item in manifest["files"]),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "files"}


def tune_detector_cache(root: Path, directory: Path, output: Path) -> dict:
    directory, output = artifact_path(directory, root), artifact_path(output, root)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("A complete held-out detector cache is required")
    index = CocoIndex(annotations_path(root))
    folds = pd.read_csv(root / "data/splits/folds.csv")
    names = sorted(folds.loc[folds.fold == manifest["fold"], "filename"].tolist())
    if len(names) != manifest["physical_images"]:
        raise ValueError("Detector cache fold coverage differs from frozen splits")
    candidates = [
        {"score_threshold": score, "mask_threshold": mask, "minimum_area": 150, "maximum_area": 50000}
        for score in (0.1, 0.2, 0.3, 0.4, 0.5)
        for mask in (0.3, 0.4, 0.5, 0.6, 0.7)
    ]
    totals, rows = [PQCounts() for _ in candidates], []
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for number, name in enumerate(names):
        path = directory / (Path(name).stem + ".npz")
        expected = next(item["sha256"] for item in manifest["files"] if item["name"] == path.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Detector probability cache checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            packed = {key: archive[key] for key in archive.files}
        truths = [
            (record["id"], index.record_rles(record["id"])) for record in index.images_by_filename[name]
        ]
        for candidate, config in enumerate(candidates):
            rles = cached_instances(packed, config)
            for identifier, gt in truths:
                counts = reference_counts(overlap_matrix(gt, rles))
                totals[candidate] += counts
                rows.append(
                    {
                        "candidate": candidate,
                        "filename": name,
                        "annotator_image": identifier,
                        **counts.scores(),
                    }
                )
        if (number + 1) % 20 == 0:
            print(f"Detector postprocessing sweep {number + 1}/{len(names)}", flush=True)
    ranking = sorted(
        [
            {"candidate": number, **config, **totals[number].scores()}
            for number, config in enumerate(candidates)
        ],
        key=lambda item: (-item["pq"], item["candidate"]),
    )
    pd.DataFrame(ranking).to_csv(output / "candidate-ranking.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "candidate-per-record.csv", index=False)
    selected = ranking[0]
    config, predictions = candidates[selected["candidate"]], {}
    for name in names:
        path = directory / (Path(name).stem + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            rles = cached_instances({key: archive[key] for key in archive.files}, config)
        if rles:
            predictions[Path(name).stem] = rles
    validation = validate_predictions(predictions, {Path(name).stem for name in names})
    write_predictions(output / "predictions.csv", predictions)
    result = {
        "fold": manifest["fold"],
        "physical_images": len(names),
        "selected": selected,
        "config": config,
        "candidate_count": len(candidates),
        "baseline_candidate": next(
            item for item in ranking if item["score_threshold"] == 0.3 and item["mask_threshold"] == 0.5
        ),
        "submission_validation": validation,
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Exploratory postprocessing selected on this fold; requires frozen confirmation on separately trained predictions.",
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
