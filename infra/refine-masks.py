"""Cross-fitted crop refiner for fused instance masks.

Each fused instance is cropped from its native image with context (square side
= the longer box side plus max(32 px, a quarter of it) on each side), resized to
a fixed square, and given two channels: the image and the instance's fused mask.
A ResNet-34 U-Net (solar_filament.models.ResNetUNet) predicts the soft
consensus of the annotators' drawings of the same object: the mean mask of the
best-overlapping object (IoU >= .1) of every annotator. Each fold's instances
are refined by a model trained on the other folds (soft-target BCE + Dice, with
dihedral, intensity and mask-prior jitter), and refined masks are pasted back at
native resolution. Refined masks are not yet pixel-disjoint; ownership is
resolved downstream in instance priority order. With ``--dense-maps``, a third
channel carries the dense family's view-averaged filament probability (from
runs whose held-out or test TTA saved ``maps/``), so the refiner can see where
the filament continues beyond a fragmentary fused mask.

The rescorer must learn selection on the masks it will submit, so refinement
runs before it: ``tables`` adds refined columns to level-1 instance tables with
the averaged fold models, and ``adopt`` (CPU) swaps each instance's mask for its
refined mask, re-owns pixels in fused order, adds refined-shape features and,
on training images, recomputes the per-annotator match labels.

    python infra/refine-masks.py train --features DIR --output RUN
    python infra/refine-masks.py tables --run RUN --tables CSV [CSV ...] --output DIR
    python infra/refine-masks.py adopt --source DIR --output DIR [--column refined_rle_50]
    python infra/refine-masks.py predict --run RUN --predictions CSV --output DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from torch.nn import functional as F

from solar_filament.coco import CocoIndex, encode_local_mask
from solar_filament.metrics import overlap_matrix
from solar_filament.models import ResNetUNet
from solar_filament.storage import annotations_path, dataset_root, storage_root
from solar_filament.submission import load_predictions, validate_predictions, write_predictions

MARGIN, MARGIN_FRACTION, OVERLAP = 32, 0.25, 0.1
THRESHOLDS = (0.4, 0.5, 0.6)


def geometry(rle: dict) -> tuple[int, int, int]:
    x, y, width, height = mask_utils.toBbox(rle)
    longer = max(width, height)
    side = int(min(longer + 2 * max(MARGIN, MARGIN_FRACTION * longer), 2048))
    return int(round(x + width / 2 - side / 2)), int(round(y + height / 2 - side / 2)), side


def window(array: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    out = np.zeros((side, side), array.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, array.shape[1]), min(y0 + side, array.shape[0])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def image_path(stem: str) -> Path:
    root = dataset_root(storage_root())
    train = root / "train/train_images" / f"{stem}.jpeg"
    return train if train.exists() else root / "test/test_images" / f"{stem}.jpeg"


def dense_map(stem: str, directories: list[Path]) -> np.ndarray:
    """View-averaged dense filament map of one image, averaged over every given run that holds it.

    Held-out directories hold each training image once (its own fold's model);
    test directories hold every test image once per fold model.
    """
    maps = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for directory in directories
            if (path := directory / "maps" / f"{stem}-filament.png").exists()]
    if not maps:
        raise FileNotFoundError(f"No dense filament map for {stem}")
    return np.round(np.mean(maps, axis=0)).astype(np.uint8)


def extract_image(arguments):
    """Inputs (image, prior mask[, dense map]), soft consensus target and crop geometry for one image's instances."""
    stem, counts, records, size, dense_dirs = arguments
    image = cv2.imread(str(image_path(stem)), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048):
        raise ValueError(f"Missing or unexpected native image: {stem}")
    dense = dense_map(stem, dense_dirs) if dense_dirs else None
    rles = [{"size": [2048, 2048], "counts": value.encode("ascii")} for value in counts]
    matrices = [overlap_matrix(gts, rles) for gts in records]
    inputs = np.zeros((len(rles), 2 + (dense is not None), size, size), np.uint8)
    targets = np.zeros((len(rles), size, size), np.uint8)
    has_target = np.zeros(len(rles), bool)
    boxes = np.zeros((len(rles), 3), np.int32)
    decoded = {}
    for number, rle in enumerate(rles):
        x0, y0, side = geometry(rle)
        boxes[number] = (x0, y0, side)
        mask = mask_utils.decode(rle)
        inputs[number, 0] = cv2.resize(window(image, x0, y0, side), (size, size), interpolation=cv2.INTER_AREA)
        inputs[number, 1] = cv2.resize(window(mask * 255, x0, y0, side), (size, size), interpolation=cv2.INTER_AREA)
        if dense is not None:
            inputs[number, 2] = cv2.resize(window(dense, x0, y0, side), (size, size), interpolation=cv2.INTER_AREA)
        chosen = [(record, int(matrix[:, number].argmax())) for record, matrix in enumerate(matrices)
                  if matrix.shape[0] and matrix[:, number].max() >= OVERLAP]
        if chosen:
            for key in chosen:
                if key not in decoded:
                    decoded[key] = mask_utils.decode(records[key[0]][key[1]])
            soft = np.mean([window(decoded[key], x0, y0, side).astype(np.float32) for key in chosen], axis=0)
            targets[number] = np.round(cv2.resize(soft, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
            has_target[number] = True
    return inputs, targets, has_target, boxes


def pool_map(function, jobs: list, workers: int):
    """Process-pool map over jobs; one worker runs inline (so a notebook can load this file by path)."""
    if workers <= 1:
        yield from map(function, jobs)
        return
    with ProcessPoolExecutor(workers) as pool:
        yield from pool.map(function, jobs, chunksize=2)


def extract(frame: pd.DataFrame, size: int, workers: int, with_targets: bool, dense_dirs: list[Path] | None = None):
    index = CocoIndex(annotations_path(storage_root())) if with_targets else None
    order = list(frame.groupby("stem", sort=False).indices.items())
    jobs = [(stem, frame.rle.iloc[rows].tolist(),
             [index.record_rles(record["id"]) for record in index.images_by_filename[stem + ".jpeg"]] if index else [],
             size, dense_dirs or [])
            for stem, rows in order]
    inputs = np.zeros((len(frame), 2 + bool(dense_dirs), size, size), np.uint8)
    targets = np.zeros((len(frame), size, size), np.uint8)
    has_target = np.zeros(len(frame), bool)
    boxes = np.zeros((len(frame), 3), np.int32)
    for (_, rows), block in zip(order, pool_map(extract_image, jobs, workers)):
        inputs[rows], targets[rows], has_target[rows], boxes[rows] = block
    return inputs, targets, has_target, boxes


def prepare(batch: torch.Tensor) -> torch.Tensor:
    """Image standardized; prior mask and optional dense map mapped to [-1, 1]."""
    image, probabilities = batch[:, :1].float() / 255, batch[:, 1:].float() / 255
    return torch.cat([(image - 0.45) / 0.2, probabilities * 2 - 1], dim=1)


def dihedral(batch: torch.Tensor, view: int) -> torch.Tensor:
    batch = torch.rot90(batch, view % 4, dims=(-2, -1))
    return batch.flip(-1) if view >= 4 else batch


def undo_dihedral(batch: torch.Tensor, view: int) -> torch.Tensor:
    batch = batch.flip(-1) if view >= 4 else batch
    return torch.rot90(batch, -(view % 4), dims=(-2, -1))


def fit(inputs: np.ndarray, targets: np.ndarray, args, seed: int) -> ResNetUNet:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = ResNetUNet(pretrained=True, input_channels=inputs.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(inputs) / args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(args.device, enabled=args.device == "cuda")
    for epoch in range(args.epochs):
        model.train()
        order, total = generator.permutation(len(inputs)), 0.0
        for start in range(0, len(order), args.batch):
            rows = np.sort(order[start:start + args.batch])
            view = int(generator.integers(8))
            batch = dihedral(torch.from_numpy(inputs[rows]).to(args.device), view)
            target = dihedral(torch.from_numpy(targets[rows]).to(args.device), view).float()[:, None] / 255
            values = prepare(batch)
            gain = torch.empty(len(rows), 1, 1, 1, device=args.device).uniform_(0.85, 1.15)
            shift = torch.empty(len(rows), 1, 1, 1, device=args.device).uniform_(-0.15, 0.15)
            # Mask-prior jitter: grow or shrink the fused mask by one crop pixel for a third of samples each.
            prior = values[:, 1:2]
            choice = torch.from_numpy(generator.integers(3, size=len(rows))).to(args.device).view(-1, 1, 1, 1)
            grown, shrunk = F.max_pool2d(prior, 3, 1, 1), -F.max_pool2d(-prior, 3, 1, 1)
            prior = torch.where(choice == 1, grown, torch.where(choice == 2, shrunk, prior))
            values = torch.cat([values[:, :1] * gain + shift, prior, values[:, 2:]], dim=1)
            with torch.autocast(args.device, enabled=args.device == "cuda"):
                logits = model(values).float()
            probability = torch.sigmoid(logits)
            overlap = (probability * target).sum((1, 2, 3))
            dice = 1 - (2 * overlap + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, target) + dice.mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach()) * len(rows)
        print(f"  epoch {epoch + 1}/{args.epochs} loss={total / len(inputs):.4f}", flush=True)
    return model


def predict_crops(models: list, inputs: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """Mean probability over models and four flip views, as uint8 0-255."""
    out = np.zeros((len(inputs), inputs.shape[-2], inputs.shape[-1]), np.uint8)
    for model in models:
        model.eval()
    with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
            total = 0
            for model in models:
                for view in (0, 2, 4, 6):
                    total = total + undo_dihedral(torch.sigmoid(model(prepare(dihedral(batch, view))).float()), view)
            mean = total / (4 * len(models))
            out[start:start + len(batch)] = (mean[:, 0] * 255).round().byte().cpu().numpy()
    return out


def repeat_pass(models: list, inputs: np.ndarray, refined: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """Refine again with the previous pass's binarized mask as the prior channel."""
    again = inputs.copy()
    again[:, 1] = np.where(refined >= 128, 255, 0).astype(np.uint8)
    return predict_crops(models, again, device, batch_size)


def paste(probability: np.ndarray, box: np.ndarray, threshold: float) -> dict | None:
    """Native-resolution refined mask for one crop, or None when empty."""
    x0, y0, side = (int(value) for value in box)
    native = cv2.resize(probability.astype(np.float32) / 255, (side, side), interpolation=cv2.INTER_LINEAR) >= threshold
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, 2048), min(y0 + side, 2048)
    local = native[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    if not local.any():
        return None
    return encode_local_mask(local, sx0, sy0)


def crop_iou(mask: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(mask, target).sum()
    return float(np.logical_and(mask, target).sum() / union) if union else 0.0


def load_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for number, path in enumerate(paths):
        frame = pd.read_csv(path)
        frame["table"], frame["row"] = number, np.arange(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def train(args):
    torch.backends.cudnn.benchmark = True
    root = storage_root()
    paths = sorted(args.features.glob("features-f*.csv"))
    frame = load_tables(paths)
    folds = pd.read_csv(root / "data/splits/folds.csv")
    frame["fold"] = frame.stem.map(dict(zip(folds.image_stem, folds.fold)))
    if frame.fold.isna().any():
        raise ValueError("Every training instance must belong to a declared fold")
    if args.limit:
        frame = frame.groupby("fold", group_keys=False).head(args.limit).reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    inputs, targets, has_target, boxes = extract(frame, args.size, args.workers, with_targets=True, dense_dirs=args.dense_maps)
    print(f"extracted {len(inputs)} crops ({int(has_target.sum())} with annotator targets) in {time.monotonic() - started:.0f}s", flush=True)
    refined = np.zeros((len(frame), args.size, args.size), np.uint8)
    repeated = np.zeros_like(refined)
    diagnostics = {}
    for fold in sorted(frame.fold.unique()):
        held = (frame.fold == fold).to_numpy()
        train_rows = ~held & has_target
        print(f"fold {fold}: train {int(train_rows.sum())}, refine {int(held.sum())}", flush=True)
        model = fit(inputs[train_rows], targets[train_rows], args, seed=args.seed + int(fold))
        refined[held] = predict_crops([model], inputs[held], args.device, args.batch * 2)
        repeated[held] = repeat_pass([model], inputs[held], refined[held], args.device, args.batch * 2)
        torch.save({"model": model.state_dict(), "fold": int(fold), "size": args.size, "channels": inputs.shape[1]},
                   args.output / f"fold-{int(fold)}.pt")
        rows = np.flatnonzero(held & has_target)
        prior_iou = np.array([crop_iou(inputs[row, 1] >= 128, targets[row] >= 128) for row in rows])
        refined_iou = np.array([crop_iou(refined[row] >= 128, targets[row] >= 128) for row in rows])
        repeated_iou = np.array([crop_iou(repeated[row] >= 128, targets[row] >= 128) for row in rows])
        diagnostics[int(fold)] = {
            "instances_with_target": int(len(rows)), "prior_mean_iou": float(prior_iou.mean()),
            "refined_mean_iou": float(refined_iou.mean()), "second_pass_mean_iou": float(repeated_iou.mean()),
            "crossed_above_half": int(((prior_iou <= 0.5) & (refined_iou > 0.5)).sum()),
            "fell_below_half": int(((prior_iou > 0.5) & (refined_iou <= 0.5)).sum()),
            "second_pass_crossed_above_half": int(((prior_iou <= 0.5) & (repeated_iou > 0.5)).sum()),
            "second_pass_fell_below_half": int(((prior_iou > 0.5) & (repeated_iou <= 0.5)).sum()),
        }
        print(f"  crop IoU with consensus: prior {prior_iou.mean():.4f} -> refined {refined_iou.mean():.4f} "
              f"-> second pass {repeated_iou.mean():.4f}; crossed above .5: {diagnostics[int(fold)]['crossed_above_half']} "
              f"({diagnostics[int(fold)]['second_pass_crossed_above_half']}), fell below: {diagnostics[int(fold)]['fell_below_half']} "
              f"({diagnostics[int(fold)]['second_pass_fell_below_half']})", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    columns = add_refined_columns(frame, refined, repeated, boxes)
    output = args.output / "features"
    output.mkdir(parents=True, exist_ok=True)
    for number, path in enumerate(paths):
        original = pd.read_csv(path)
        rows = frame.loc[frame.table == number]
        for column in columns:
            original[column] = ""
            original.loc[rows.row.to_numpy(), column] = rows[column].to_numpy()
        original.to_csv(output / path.name, index=False)
    record = {
        "status": "complete", "instances": len(frame), "with_target": int(has_target.sum()), "size": args.size,
        "epochs": args.epochs, "batch": args.batch, "learning_rate": args.learning_rate, "seed": args.seed,
        "architecture": f"ResNetUNet resnet34-imagenet, {inputs.shape[1]} channels", "features": [str(path) for path in paths],
        "dense_maps": [str(path) for path in args.dense_maps or []],
        "diagnostics": diagnostics, "seconds": time.monotonic() - started, "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("instances", "with_target", "seconds")}), flush=True)


def add_refined_columns(frame: pd.DataFrame, refined: np.ndarray, repeated: np.ndarray | None, boxes: np.ndarray) -> list[str]:
    """refined_rle_{40,50,60} from the first pass and refined_rle_2pass50 from the second (empty = no pixels)."""
    sources = [(f"refined_rle_{int(threshold * 100)}", refined, threshold) for threshold in THRESHOLDS]
    if repeated is not None:
        sources.append(("refined_rle_2pass50", repeated, 0.5))
    for column, probability, threshold in sources:
        pasted = [paste(probability[row], boxes[row], threshold) for row in range(len(frame))]
        frame[column] = [rle["counts"].decode("ascii") if rle else "" for rle in pasted]
    return [column for column, _, _ in sources]


def load_models(run: Path, device: str, dense: bool = False) -> tuple[list, int]:
    models, size = [], 0
    for path in sorted(run.glob("fold-*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("channels", 2) != 2 + dense:
            raise ValueError("Pass --dense-maps exactly when the refiner was trained with them")
        model = ResNetUNet(pretrained=False, input_channels=saved.get("channels", 2))
        model.load_state_dict(saved["model"])
        models.append(model.to(device))
        size = saved["size"]
    if not models:
        raise ValueError("The refiner run has no fold models")
    return models, size


def refine_tables(args):
    """Add refined_rle_* columns to instance tables with the averaged fold models (test level-1 tables)."""
    models, size = load_models(args.run, args.device, bool(args.dense_maps))
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.tables:
        frame = pd.read_csv(path)
        inputs, _, _, boxes = extract(frame, size, args.workers, with_targets=False, dense_dirs=args.dense_maps)
        refined = predict_crops(models, inputs, args.device, args.batch)
        repeated = repeat_pass(models, inputs, refined, args.device, args.batch) if args.passes == 2 else None
        add_refined_columns(frame, refined, repeated, boxes)
        frame.to_csv(args.output / path.name, index=False)
        print(f"{path.name}: {len(frame)} instances refined", flush=True)
    (args.output / "refine.json").write_text(json.dumps({
        "run": str(args.run), "models": len(models), "passes": args.passes, "tables": [str(path) for path in args.tables],
        "dense_maps": [str(path) for path in args.dense_maps or []],
        "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")


def adopt_image(arguments):
    """Re-own one image's refined masks in fused order; match labels against its annotators (if any)."""
    rows, records, minimum_area = arguments
    occupied = np.zeros((2048, 2048), bool)
    kept, rles, extras = [], [], []
    for key, prior_counts, refined_counts in rows:
        prior = mask_utils.decode({"size": [2048, 2048], "counts": prior_counts.encode("ascii")}).astype(bool)
        refined = (mask_utils.decode({"size": [2048, 2048], "counts": refined_counts.encode("ascii")}).astype(bool)
                   if isinstance(refined_counts, str) and refined_counts else prior)
        mask = refined & ~occupied
        area = int(mask.sum())
        if area < minimum_area:
            continue
        occupied |= mask
        union = np.logical_or(prior, refined).sum()
        kept.append(key)
        rles.append(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
        extras.append({"refined_area": area, "refined_area_ratio": area / max(int(prior.sum()), 1),
                       "refined_prior_iou": float(np.logical_and(prior, refined).sum() / union) if union else 0.0})
    matrices = [overlap_matrix(record, rles) for record in records]
    out = []
    for column, (key, rle, extra) in enumerate(zip(kept, rles, extras)):
        matched = [bool((matrix[:, column] > 0.5).any()) for matrix in matrices]
        best = [float(matrix[:, column].max()) if matrix.shape[0] else 0.0 for matrix in matrices]
        pairs = [[record, int(gt), float(matrix[gt, column])] for record, matrix in enumerate(matrices)
                 for gt in np.flatnonzero(matrix[:, column] > 0.5)]
        out.append((key, {"matched_fraction": float(np.mean(matched)) if matched else 0.0,
                          "best_iou_mean": float(np.mean(best)) if best else 0.0, "pairs": json.dumps(pairs),
                          "rle": rle["counts"].decode("ascii"), **extra}))
    return out


def adopt(args):
    """Instance tables whose masks are the refined masks, in the fuse-candidates format the rescorer reads."""
    index = CocoIndex(annotations_path(storage_root()))
    paths = sorted(args.source.glob(args.pattern))
    if not paths:
        raise ValueError(f"No tables match {args.pattern} in {args.source}")
    tables = [pd.read_csv(path) for path in paths]
    jobs = []
    for number, frame in enumerate(tables):
        for stem, group in frame.groupby("stem"):
            group = group.sort_values("column")
            records = [index.record_rles(record["id"]) for record in index.images_by_filename.get(stem + ".jpeg", [])]
            jobs.append(([((number, row), group.rle[row], group[args.column][row]) for row in group.index],
                         records, args.minimum_area))
    updates = {}
    for out in pool_map(adopt_image, jobs, args.workers):
        updates.update(dict(out))
    args.output.mkdir(parents=True, exist_ok=True)
    for number, (path, frame) in enumerate(zip(paths, tables)):
        keep = [row for row in frame.index if (number, row) in updates]
        frame = frame.loc[keep, [name for name in frame.columns if not name.startswith("refined_rle_")]].copy()
        for name in ("matched_fraction", "best_iou_mean", "pairs", "rle", "refined_area", "refined_area_ratio", "refined_prior_iou"):
            frame[name] = [updates[(number, row)][name] for row in frame.index]
        frame.to_csv(args.output / path.name, index=False)
        print(f"{path.name}: {len(tables[number])} -> {len(frame)} instances, matched {frame.matched_fraction.mean():.4f}", flush=True)


def predict(args):
    """Refine a predictions CSV with the averaged fold models, then re-own pixels in CSV (priority) order."""
    predictions = load_predictions(args.predictions)
    rows = [(stem, rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"])
            for stem in sorted(predictions) for rle in predictions[stem]]
    frame = pd.DataFrame(rows, columns=["stem", "rle"])
    models, size = load_models(args.run, args.device, bool(args.dense_maps))
    inputs, _, _, boxes = extract(frame, size, args.workers, with_targets=False, dense_dirs=args.dense_maps)
    refined = predict_crops(models, inputs, args.device, args.batch)
    output = {}
    for stem, group in frame.groupby("stem", sort=False):
        occupied, owned = np.zeros((2048, 2048), bool), []
        for row in group.index:
            rle = paste(refined[row], boxes[row], args.threshold) or {"size": [2048, 2048], "counts": group.rle[row].encode("ascii")}
            mask = mask_utils.decode(rle).astype(bool) & ~occupied
            if mask.sum() >= args.minimum_area:
                ys, xs = np.nonzero(mask)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                owned.append(encode_local_mask(mask[y0:y1, x0:x1], int(x0), int(y0)))
                occupied |= mask
        if owned:
            output[stem] = owned
    args.output.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output / "predictions.csv", output)
    stems = {path.stem for path in (dataset_root(storage_root()) / "test/test_images").glob("*.jpeg")}
    validation = validate_predictions(output, stems)
    (args.output / "refine.json").write_text(json.dumps({
        "source": str(args.predictions), "run": str(args.run), "threshold": args.threshold, "minimum_area": args.minimum_area,
        "models": len(models), "validation": validation, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(validation), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--features", type=Path, required=True, help="Directory of OOF features-f*.csv tables")
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--size", type=int, default=256)
    fitting.add_argument("--epochs", type=int, default=30)
    fitting.add_argument("--batch", type=int, default=32)
    fitting.add_argument("--learning-rate", type=float, default=3e-4)
    fitting.add_argument("--seed", type=int, default=20261011)
    fitting.add_argument("--limit", type=int, default=0, help="Smoke test: instances per fold")
    fitting.add_argument("--workers", type=int, default=8)
    fitting.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    fitting.add_argument("--resume", action="store_true", help="Accepted for worker compatibility")
    fitting.add_argument("--dense-maps", type=Path, nargs="+", help="TTA directories whose maps/ hold dense filament maps (adds a third channel)")
    fitting.set_defaults(handler=train)
    scoring = commands.add_parser("predict")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--threshold", type=float, default=0.5)
    scoring.add_argument("--minimum-area", type=int, default=150)
    scoring.add_argument("--batch", type=int, default=64)
    scoring.add_argument("--workers", type=int, default=8)
    scoring.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    scoring.add_argument("--dense-maps", type=Path, nargs="+", help="TTA directories whose maps/ hold dense filament maps (adds a third channel)")
    scoring.set_defaults(handler=predict)
    tabling = commands.add_parser("tables")
    tabling.add_argument("--run", type=Path, required=True)
    tabling.add_argument("--tables", type=Path, nargs="+", required=True)
    tabling.add_argument("--output", type=Path, required=True)
    tabling.add_argument("--passes", type=int, choices=(1, 2), default=1)
    tabling.add_argument("--batch", type=int, default=64)
    tabling.add_argument("--workers", type=int, default=8)
    tabling.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tabling.add_argument("--resume", action="store_true", help="Accepted for worker compatibility")
    tabling.add_argument("--dense-maps", type=Path, nargs="+", help="TTA directories whose maps/ hold dense filament maps (adds a third channel)")
    tabling.set_defaults(handler=refine_tables)
    adopting = commands.add_parser("adopt")
    adopting.add_argument("--source", type=Path, required=True, help="Directory of tables with refined_rle_* columns")
    adopting.add_argument("--output", type=Path, required=True)
    adopting.add_argument("--pattern", default="*.csv")
    adopting.add_argument("--column", default="refined_rle_50")
    adopting.add_argument("--minimum-area", type=int, default=150)
    adopting.add_argument("--workers", type=int, default=8)
    adopting.set_defaults(handler=adopt)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
