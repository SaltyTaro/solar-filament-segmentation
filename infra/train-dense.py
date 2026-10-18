"""Train the spine-aware dense ConvNeXt U-Net (``dense_model.py``) on the frozen folds.

Held-out and test candidates and averaged probability maps are written
afterwards by ``infra/tta-inference.py --family dense``.

    python infra/train-dense.py --config configs/phase4/dense-convnexts1024-e80-f0.yaml --output RUN
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import dense_model
import yaml
from family_training import digest, train

from solar_filament.storage import configure_caches, storage_root

FAMILY = SimpleNamespace(
    make_model=dense_model.make_model,
    dataset=dense_model.DenseDataset,
    collate=None,
    to_device=lambda batch: [tensor.cuda(non_blocking=True) for tensor in batch],
    loss=lambda model, batch, config: dense_model.loss(model, *batch, config),
    backbone_prefix="stages.",
    pretrained_file=lambda config: storage_root() / config["pretrained_checkpoint"],
    sources=[Path(__file__), Path(dense_model.__file__)],
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    root = storage_root()
    configure_caches(root)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if digest(FAMILY.pretrained_file(config)) != config["pretrained_sha256"]:
        raise ValueError("Dense encoder weights differ from the frozen manifest")
    train(root, config, args.output, FAMILY, args.resume, args.smoke_steps)


if __name__ == "__main__":
    main()
