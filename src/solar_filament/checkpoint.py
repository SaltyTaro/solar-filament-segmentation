"""Compact inference snapshots alongside the full resumable training checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch


def save_inference_checkpoint(path: Path, model, config: dict, epoch: int, pq: float) -> None:
    temporary = path.with_suffix(".tmp")
    record = {
        "model": model.state_dict(),
        "config": dict(config),
        "epoch": epoch,
        "step": 0,
        "best_pq": pq,
        "checkpoint_scope": "inference; resume training with last.pt",
    }
    torch.save(record, temporary)
    if temporary.stat().st_size >= 2**32:
        raise ValueError("Checkpoint exceeds FAT32 limit")
    temporary.replace(path)
