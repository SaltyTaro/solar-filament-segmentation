"""Check instance identity, thin-mask targets and update scheduling."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("yolo_pilot", Path(__file__).parents[1] / "infra/yolo_pilot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(size=64):
    masks = torch.zeros(2, size, size, dtype=torch.uint8)
    masks[0, 8:24, 16:48] = 1
    masks[1, 16:32, 24:56] = 1
    return torch.full((3, size, size), .5), {
        "boxes": torch.tensor([[16, 8, 48, 24], [24, 16, 56, 32]], dtype=torch.float32), "masks": masks,
    }


def test_instance_order_class_and_overlap_survive_batching():
    batch = module.pack_batch([sample(), sample()], 64, 2)
    assert batch["img"].shape == (2, 3, 64, 64)
    assert batch["masks"].shape == (4, 32, 32)
    assert batch["batch_idx"].tolist() == [0, 0, 1, 1]
    assert batch["cls"].tolist() == [[0], [0], [0], [0]]
    assert torch.equal(batch["masks"][0], batch["masks"][2])
    assert (batch["masks"][0] * batch["masks"][1]).sum() > 0
    assert torch.equal(batch["bboxes"][0], torch.tensor([.5, .25, .5, .25]))


def test_one_pixel_filament_keeps_fractional_area():
    image = torch.ones(3, 64, 64)
    mask = torch.zeros(1, 64, 64, dtype=torch.uint8)
    mask[0, 1, 1:63] = 1
    target = {"boxes": torch.tensor([[1., 1., 63., 2.]]), "masks": mask}
    batch = module.pack_batch([(image, target)], 32, 2)
    assert batch["masks"].sum().item() == pytest.approx(62 / 16)
    assert batch["masks"].max().item() <= .25
    assert batch["masks"].min().item() == 0


def test_images_without_instances_keep_image_indices():
    image, target = sample()
    empty = {"boxes": torch.empty(0, 4), "masks": torch.empty(0, 64, 64, dtype=torch.uint8)}
    batch = module.pack_batch([(image, empty), (image, target)], 64)
    assert batch["batch_idx"].tolist() == [1, 1]
    all_empty = module.pack_batch([(image, empty)], 64)
    assert all_empty["bboxes"].shape == (0, 4)
    assert all_empty["masks"].shape == (0, 32, 32)


@pytest.mark.parametrize("bad", ["shape", "outside_box", "empty_mask", "nonbinary_mask", "nan"])
def test_corrupt_targets_are_rejected(bad):
    image, target = sample()
    if bad == "shape":
        target["boxes"] = target["boxes"][:1]
    elif bad == "outside_box":
        target["boxes"][0, 0] = -1
    elif bad == "empty_mask":
        target["masks"][0] = 0
    elif bad == "nonbinary_mask":
        target["masks"][0, 8, 16] = 2
    else:
        image[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        module.pack_batch([(image, target)], 64)


def test_schedule_reaches_floor_and_resume_uses_absolute_position():
    config = {"epochs": 2, "steps_per_epoch": 8, "warmup_steps": 4,
              "learning_rate": .0003, "minimum_learning_rate": .00003}
    rates = [module.learning_rate(config, step) for step in range(16)]
    assert rates[0] == pytest.approx(.000075)
    assert rates[3] == pytest.approx(.0003)
    assert rates[-1] == pytest.approx(.00003)
    assert rates[4:] == sorted(rates[4:], reverse=True)
    assert [module.learning_rate(config, step) for step in range(9, 16)] == rates[9:]


def test_ema_tracks_weights_and_batchnorm_counters():
    model = torch.nn.BatchNorm2d(2)
    ema = module.ModelEMA(model)
    with torch.no_grad():
        model.weight.fill_(2)
        model.num_batches_tracked.fill_(7)
    ema.update(model)
    assert ema.updates == 1
    assert torch.all(ema.model.weight > 1.99)
    assert ema.model.num_batches_tracked.item() == 7
    assert all(not parameter.requires_grad for parameter in ema.model.parameters())


def test_end_to_end_loss_weights_follow_the_ultralytics_epoch_schedule():
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    def build(name):
        model = SegmentationModel(f"{name}.yaml", nc=1, ch=3, verbose=False)
        model.args = get_cfg(overrides={"task": "segment", "overlap_mask": False, "mask_ratio": 2, "epochs": 60})
        return model

    assert ultralytics.__version__
    assert not module.schedule_end_to_end(build("yolo11n-seg"), 5)
    model, reference = build("yolo26n-seg"), None
    for epoch in (0, 1, 30, 59):
        assert module.schedule_end_to_end(model, epoch)
        reference = model.init_criterion()
        for _ in range(epoch):
            reference.update()
        assert model.criterion.o2m == pytest.approx(reference.o2m)
        assert model.criterion.o2o == pytest.approx(reference.o2o)
    module.schedule_end_to_end(model, 3)
    module.schedule_end_to_end(model, 0)
    assert model.criterion.o2m == pytest.approx(.8)
