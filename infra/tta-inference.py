"""Dihedral test-time inference that saves per-view proposal candidates.

Supports the project's Mask R-CNN runs, custom-loop YOLO runs and dense
(connected-component) runs; dense runs also save their view-averaged filament and spine maps as PNGs under ``maps/``. By default the final EMA weights are used, so held-out candidates
involve no checkpoint selection on the scored fold. Each image's JSON holds all
views; ``rle`` uses mask threshold .5 and ``rle_low`` (Mask R-CNN only) a lower
threshold for extent studies.

    python infra/tta-inference.py --family maskrcnn --run RUN --subset validation --output OUT
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from solar_filament.coco import encode_local_mask
from solar_filament.fusion import dihedral
from solar_filament.preprocessing import DetectorPreprocessing
from solar_filament.storage import artifact_path, configure_caches, dataset_root, storage_root

NATIVE_IMREAD = cv2.imread


def load_module(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_gpu_mask(mask: torch.Tensor) -> tuple[dict, int] | None:
    rows, columns = torch.nonzero(mask.any(dim=1)), torch.nonzero(mask.any(dim=0))
    if not len(rows):
        return None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1
    crop = mask[y0:y1, x0:x1].cpu().numpy()
    rle = encode_local_mask(crop, x0, y0, *mask.shape)
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}, int(crop.sum())


def gpu_candidates(scores: torch.Tensor, masks: torch.Tensor, view: int, threshold: float,
                   low_threshold: float | None = None) -> list[dict]:
    """Threshold, undo the view on the GPU and encode each mask from its box crop."""
    records = []
    if view >= 4:
        masks = masks.flip(-1)
    masks = torch.rot90(masks, -(view % 4), dims=(-2, -1))
    for score, probability in zip(scores.tolist(), masks):
        encoded = encode_gpu_mask(probability >= threshold)
        if encoded is None:
            continue
        record = {"score": float(score), "area": encoded[1], "view": view, "rle": encoded[0]}
        if low_threshold is not None:
            record["rle_low"] = encode_gpu_mask(probability >= low_threshold)[0]
        records.append(record)
    return records


class MaskRCNNSource:
    def __init__(self, run: Path, weights: str, args):
        from solar_filament.detection import make_detector

        self.config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        path = run / ("best.pt" if weights == "best" else "last.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Released inference-only checkpoints hold the EMA weights alone.
        state = checkpoint["model"] if weights == "best" or "ema" not in checkpoint else checkpoint["ema"]
        self.model = make_detector(self.config, pretrained=False)
        self.model.load_state_dict(state)
        self.model.roi_heads.score_thresh = args.candidate_threshold
        self.model.roi_heads.detections_per_img = args.maximum_candidates
        self.model.cuda().eval()
        self.preprocess = DetectorPreprocessing(self.config)
        self.path, self.low = path, args.low_mask_threshold

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, view: int) -> list[dict]:
        array = np.ascontiguousarray(dihedral(self.preprocess(image), view))
        tensor = torch.from_numpy(array.astype(np.float32)[None] / 255).repeat(3, 1, 1).cuda()
        with torch.autocast("cuda", enabled=self.config["mixed_precision"]):
            result = self.model([tensor])[0]
        scores, masks = result["scores"].float(), result["masks"][:, 0].float()
        return gpu_candidates(scores, masks, view, self.config["mask_threshold"], self.low or None)


class YoloSource:
    def __init__(self, run: Path, weights: str, args):
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor

        self.config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        pilot = load_module("yolo_pilot.py")
        self.path = run / ("best-inference.pt" if weights == "best" else "last.pt")
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        model = pilot.make_model(self.config, checkpoint["architecture"], pretrained=False)
        model.load_state_dict(checkpoint["model"] if weights == "best" else checkpoint["ema"])
        cv2.imread = NATIVE_IMREAD
        self.predictor = SegmentationPredictor(overrides={
            "task": "segment", "mode": "predict", "imgsz": args.image_size or self.config["image_size"],
            "conf": args.candidate_threshold, "iou": self.config.get("nms_iou", 0.7),
            "max_det": args.maximum_candidates, "retina_masks": True, "device": 0, "half": True,
            "verbose": False, "save": False, "project": str(storage_root() / "runs/yolo-predict-cache"),
            "name": "runtime", "exist_ok": True,
        })
        self.predictor.setup_model(model=model.cuda().eval(), verbose=False)

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, view: int) -> list[dict]:
        array = np.ascontiguousarray(dihedral(image, view))
        result = self.predictor(source=cv2.cvtColor(array, cv2.COLOR_GRAY2BGR), stream=False)[0]
        if result.masks is None:
            return []
        return gpu_candidates(result.boxes.conf.float(), result.masks.data.float(), view, 0.5)


class DenseSource:
    """Per-view connected components; probabilities are also averaged over views for ``finish``."""

    def __init__(self, run: Path, args):
        module = load_module("dense_model.py")
        self.config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        self.path = run / "last.pt"
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        self.model = module.make_model(self.config, pretrained=False)
        self.model.load_state_dict(checkpoint["ema"])
        self.model.cuda().eval()
        self.module, self.total, self.views = module, None, 0

    def __call__(self, image: np.ndarray, view: int) -> list[dict]:
        probability = self.module.predict(self.model, dihedral(image, view), self.config)
        if view >= 4:
            probability = probability.flip(-1)
        probability = torch.rot90(probability, -(view % 4), dims=(-2, -1))
        self.total = probability if self.total is None else self.total + probability
        self.views += 1
        filament, spine = probability.cpu().numpy()
        return self.module.component_candidates(filament, spine, view, self.config["component_threshold"],
                                                self.config["minimum_component_area"])

    def finish(self, output: Path, stem: str):
        directory = output / "maps"
        directory.mkdir(exist_ok=True)
        for channel, name in enumerate(("filament", "spine")):
            average = (self.total[channel] / self.views * 255).round().byte().cpu().numpy()
            if not cv2.imwrite(str(directory / f"{stem}-{name}.png"), average):
                raise OSError(f"Could not write the {name} map for {stem}")
        self.total, self.views = None, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("maskrcnn", "yolo", "dense"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", choices=("final", "best"), default="final")
    parser.add_argument("--subset", choices=("validation", "test", "external"), default="validation")
    parser.add_argument("--images-dir", type=Path, help="External unlabeled frames (with --subset external)")
    parser.add_argument("--limit", type=int, help="Only the first N external frames")
    parser.add_argument("--views", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--maximum-candidates", type=int, default=100)
    parser.add_argument("--low-mask-threshold", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, help="YOLO inference size override for multi-scale TTA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Accepted for worker compatibility")
    args = parser.parse_args()
    root = storage_root()
    configure_caches(root)
    run = artifact_path(args.run, root)
    output = artifact_path(args.output, root)
    (output / "candidates").mkdir(parents=True, exist_ok=True)
    if args.family == "maskrcnn":
        source = MaskRCNNSource(run, args.weights, args)
    elif args.family == "dense":
        source = DenseSource(run, args)
    else:
        source = YoloSource(run, args.weights, args)
    if args.subset == "external":
        directory = artifact_path(args.images_dir, root)
        names = sorted(path.name for path in directory.glob("*.jpeg"))[: args.limit]
    elif args.subset == "test":
        directory = dataset_root(root) / "test/test_images"
        names = sorted(path.name for path in directory.glob("*.jpeg"))
        if len(names) != 180:
            raise ValueError("Test inference must include all 180 supplied images")
    else:
        directory = dataset_root(root) / "train/train_images"
        folds = pd.read_csv(root / "data/splits/folds.csv")
        names = sorted(folds.loc[folds.fold == source.config["fold"], "filename"])
    started = time.monotonic()
    for number, name in enumerate(names):
        target = output / "candidates" / (Path(name).stem + ".json")
        if target.exists():
            continue
        image = NATIVE_IMREAD(str(directory / name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Missing or unexpected native image: {name}")
        records = [record for view in args.views for record in source(image, view)]
        if hasattr(source, "finish"):
            source.finish(output, Path(name).stem)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(records), encoding="utf-8")
        temporary.replace(target)
        if (number + 1) % 20 == 0:
            print(f"TTA {args.subset} {number + 1}/{len(names)} elapsed_minutes={(time.monotonic() - started) / 60:.1f}", flush=True)
    metadata = {
        "family": args.family, "run": str(run), "weights": str(source.path),
        "weights_sha256": hashlib.sha256(source.path.read_bytes()).hexdigest(), "subset": args.subset,
        "fold": source.config.get("fold"), "views": args.views, "physical_images": len(names),
        "candidate_threshold": args.candidate_threshold, "maximum_candidates": args.maximum_candidates,
        "low_mask_threshold": args.low_mask_threshold if args.family == "maskrcnn" else None,
        "image_size": args.image_size or source.config.get("image_size", "native"),
        "seconds": time.monotonic() - started, "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
