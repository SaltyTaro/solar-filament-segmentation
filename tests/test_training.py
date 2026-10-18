import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from solar_filament.postprocess import probability_instances  # noqa: E402
from solar_filament.train_unet import probability_rles, segmentation_loss, tile_positions  # noqa: E402


def test_half_precision_dice_reduction_retains_expected_value():
    logits = torch.zeros((1, 1, 512, 512), dtype=torch.float16, requires_grad=True)
    target = torch.ones_like(logits, dtype=torch.float32)
    loss = segmentation_loss(logits, target)
    expected = math.log(2) + 1 - (512 * 512 + 1) / (1.5 * 512 * 512 + 1)
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-3)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad < 0).all()


def test_tiled_inference_covers_every_native_image_coordinate():
    positions = tile_positions(2048, 768, 640)
    assert positions[0] == 0
    assert positions[-1] + 768 == 2048
    assert all(right <= left + 768 for left, right in zip(positions, positions[1:]))


def test_training_validation_and_cpu_tuner_encode_identical_masks():
    import numpy as np

    probability = np.zeros((64, 64), dtype=np.float32)
    probability[5:15, 10:20] = 0.8
    probability[30:35, 40:60] = 0.9
    config = {"mask_threshold": 0.5, "minimum_area": 20, "maximum_area": 5000}
    assert probability_rles(probability, config) == probability_instances(probability, config)
