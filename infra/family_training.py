"""Bounded, resumable training loop shared by model families with their own data and loss.

A family supplies ``make_model(config, pretrained)``, ``dataset(root, config,
length)``, ``collate`` (or None), ``to_device(batch)``, ``loss(model, batch,
config) -> (value, terms)``, ``backbone_prefix``, ``pretrained_file(config)``
and its ``sources``. The loop keeps the YOLO runs' warm-up/cosine schedule,
EMA, per-step learning rates for exact resumption, epoch checkpoints with
random state, a wall-clock limit and the run fingerprints. ``amp_dtype:
float16`` (T4/P100 sessions) adds loss scaling; bfloat16 needs none.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from yolo_pilot import ModelEMA, learning_rate

from solar_filament.storage import artifact_path
from solar_filament.training_iterator import training_iterator

NO_DECAY = ("embed", "relative_position_bias_table", "norm")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, record: dict):
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


def optimizer_for(model, config, backbone_prefix: str):
    """Backbone at ``backbone_multiplier`` of the rate; no decay on norms, biases and embeddings."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = config["backbone_multiplier"] if name.startswith(backbone_prefix) else 1.0
        decay = 0.0 if parameter.ndim <= 1 or any(part in name for part in NO_DECAY) else config["weight_decay"]
        groups.setdefault((scale, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": parameters, "lr_scale": scale, "weight_decay": decay}
                              for (scale, decay), parameters in sorted(groups.items())], lr=config["learning_rate"])


def save_checkpoint(path, model, ema, optimizer, scaler, config, epoch, step):
    record = {
        "model": model.state_dict(), "ema": ema.model.state_dict(), "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "epoch": epoch, "step": step,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    if temporary.stat().st_size >= 2**32:
        raise ValueError("Checkpoint exceeds E: FAT32 file limit")
    temporary.replace(path)


def train(root, config, output, family: SimpleNamespace, resume=False, smoke_steps=0):
    if smoke_steps:
        config = {**config, "epochs": 1, "steps_per_epoch": smoke_steps, "gradient_accumulation": 1,
                  "warmup_steps": 0, "time_limit_minutes": 15}
    accumulation = config["gradient_accumulation"]
    if config["steps_per_epoch"] % accumulation or config["checkpoint_every_steps"] % accumulation:
        raise ValueError("Epoch and checkpoint intervals must end at optimizer boundaries")
    output = artifact_path(output, root)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for {config['model']} training")
    if output.exists() and (output / "config.yaml").exists() and not resume:
        raise ValueError("Preserve existing training; resume it or choose a fresh run")
    output.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=True)
    if resume and (output / "config.yaml").read_text(encoding="utf-8") != config_text:
        raise ValueError("Resume config differs from the original run")
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model = family.make_model(config, pretrained=not resume).cuda()
    ema = ModelEMA(model)
    optimizer = optimizer_for(model, config, family.backbone_prefix)
    scaler = torch.amp.GradScaler("cuda", enabled=config.get("amp_dtype") == "float16")
    start_epoch, skip_steps = 0, 0
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema"])
        ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, skip_steps = checkpoint["epoch"], checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    dataset = family.dataset(root, config, config["steps_per_epoch"] * config["batch_size"])
    workers = int(config.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=workers, pin_memory=workers > 0,
                        prefetch_factor=4 if workers else None, collate_fn=family.collate)
    metadata_path = output / "run-metadata.json"
    sources = [Path(__file__), Path(__file__).with_name("yolo_pilot.py"), *family.sources]
    fingerprints = {
        "source_sha256": {path.name: digest(path) for path in sources},
        "split_sha256": json.loads((root / "data/splits/folds-metadata.json").read_text(encoding="utf-8"))["split_sha256"],
        "dataset_sha256": json.loads((root / "data/manifests/verified-dataset.json").read_text(encoding="utf-8"))["dataset_sha256"],
    }
    if resume:
        original = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(original[key] != value for key, value in fingerprints.items()):
            raise ValueError("Resume source, dataset or split differs from the original run")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        write(metadata_path, {
            "model": config["model"], "fold": config["fold"], "torch": torch.__version__,
            "device": torch.cuda.get_device_name(), "train_physical_images": len(dataset.names),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "pretrained_sha256": digest(family.pretrained_file(config)), **fingerprints,
            "code_revision": revision.stdout.strip() if revision.returncode == 0 else os.environ.get("SOLAR_CODE_REVISION", "uncommitted"),
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config["epochs"]):
        dataset.epoch = epoch
        model.train()
        totals, batches = {}, 0
        for step, batch in enumerate(training_iterator(loader, bool(skip_steps) and epoch == start_epoch)):
            if epoch == start_epoch and step < skip_steps:
                continue
            rate = learning_rate(config, epoch * config["steps_per_epoch"] + step)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["lr_scale"]
            value, terms = family.loss(model, family.to_device(batch), config)
            if not torch.isfinite(value):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch + 1}, step {step + 1}")
            scaler.scale(value / accumulation).backward()
            for key, term in {"loss": float(value.detach()), **terms}.items():
                totals[key] = totals.get(key, 0.0) + term
            batches += 1
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= old_scale:
                    # Skip the EMA when a float16 step overflowed and was not applied.
                    ema.update(model)
                if (step + 1) % config["checkpoint_every_steps"] == 0:
                    save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch, step + 1)
                if time.monotonic() - started > config["time_limit_minutes"] * 60:
                    save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch, step + 1)
                    result = {"status": "time_limit", "epoch": epoch, "step": step + 1}
                    write(output / "result.json", result)
                    return result
            if (step + 1) % 20 == 0 or step + 1 == config["steps_per_epoch"]:
                print(f"{config['model']} epoch={epoch + 1} step={step + 1}/{config['steps_per_epoch']} "
                      + " ".join(f"{key}={total / batches:.4f}" for key, total in totals.items())
                      + f" peak_gib={torch.cuda.max_memory_allocated() / 2**30:.1f}"
                      + f" elapsed_minutes={(time.monotonic() - started) / 60:.1f}", flush=True)
            del batch, value
        skip_steps = 0
        save_checkpoint(output / "last.pt", model, ema, optimizer, scaler, config, epoch + 1, 0)
    result = {
        "status": "complete", "epoch": config["epochs"], "training_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30, "ema_updates": ema.updates,
        "physical_image_presentations": config["epochs"] * len(dataset),
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result
