"""Check dense-family instances, consensus targets and output geometry."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parents[1] / "infra"))
spec = importlib.util.spec_from_file_location("dense_model", Path(__file__).parents[1] / "infra/dense_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def decode(record):
    return mask_utils.decode({"size": record["rle"]["size"], "counts": record["rle"]["counts"].encode()})


def test_components_are_separate_instances_scored_by_mean_probability():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[4:10, 4:30] = 0.9
    probability[40:60, 50:52] = 0.6
    probability[20, 20] = 0.99
    spine = np.zeros_like(probability)
    spine[6, 4:30] = 0.8
    records = module.component_candidates(probability, spine, 5, 0.5, 10)
    assert len(records) == 2
    first, second = sorted(records, key=lambda record: -record["score"])
    assert first["view"] == 5 and first["area"] == 6 * 26
    assert np.isclose(first["score"], 0.9) and np.isclose(second["score"], 0.6)
    assert np.isclose(first["spine"], 0.8) and second["spine"] == 0
    assert decode(first)[4:10, 4:30].all() and decode(first).sum() == 6 * 26


def test_unet_returns_filament_and_spine_logits_at_input_resolution():
    model = module.ConvNeXtUNet().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 64, 96))
    assert output.shape == (1, 2, 64, 96)


class FakeIndex:
    def __init__(self):
        self.images_by_filename = {"a.jpeg": [{"id": "r1", "height": 32, "width": 32}, {"id": "r2", "height": 32, "width": 32}]}
        self.annotations_by_image = {"r1": [{"spine": "[2, 5, 20, 5]"}], "r2": [{"spine": "[]"}]}

    def union_mask(self, record):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[3:8, 2:21] = 1
        if record == "r2":
            mask[20:25, 20:25] = 1
        return mask


def test_consensus_targets_are_annotator_fractions():
    filament, spine = module.consensus_targets(FakeIndex(), "a.jpeg")
    assert filament[5, 10] == 255 and filament[22, 22] == 128 and filament[0, 0] == 0
    assert spine[5, 10] == 128 and spine[5, 25] == 0
