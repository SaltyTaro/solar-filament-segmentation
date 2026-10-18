import numpy as np
import pytest
from pycocotools import mask as mask_utils

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from solar_filament.detection import detector_instances, make_detector  # noqa: E402


def test_detector_pixel_ownership_follows_confidence_and_keeps_masks_disjoint():
    masks = torch.zeros((2, 1, 64, 64))
    masks[0, 0, 10:20, 10:30] = 0.9
    masks[1, 0, 10:20, 20:40] = 0.9
    result = {"scores": torch.tensor([0.6, 0.9]), "masks": masks}
    config = {"score_threshold": 0.3, "mask_threshold": 0.5, "minimum_area": 50, "maximum_area": 50000}
    instances = detector_instances(result, config)
    decoded = mask_utils.decode(instances)
    assert len(instances) == 2
    assert decoded[..., 0].sum() == 200
    assert decoded[..., 1].sum() == 100
    assert decoded[15, 25, 0] == 1
    assert decoded[15, 25, 1] == 0
    assert np.all(decoded.sum(axis=2) <= 1)


def test_detector_empty_output_is_valid():
    result = {"scores": torch.empty(0), "masks": torch.empty((0, 1, 64, 64))}
    config = {"score_threshold": 0.3, "mask_threshold": 0.5, "minimum_area": 50, "maximum_area": 50000}
    assert detector_instances(result, config) == []


def test_checkpoint_reconstruction_retains_the_pretrained_backbone_freeze_policy():
    model = make_detector({"image_size": 128, "pretrained": True}, pretrained=False)
    parameters = dict(model.backbone.body.named_parameters())
    assert not parameters["conv1.weight"].requires_grad
    assert not parameters["bn1.weight"].requires_grad
    assert not parameters["layer1.0.conv1.weight"].requires_grad
    assert parameters["layer2.0.conv1.weight"].requires_grad
    assert parameters["layer4.0.conv1.weight"].requires_grad
    assert all(parameter.requires_grad for parameter in model.roi_heads.parameters())


def test_nonpretrained_detector_keeps_its_full_backbone_trainable():
    model = make_detector({"image_size": 128, "pretrained": False}, pretrained=False)
    assert all(parameter.requires_grad for parameter in model.backbone.body.parameters())


def test_invalid_backbone_policy_fails_before_model_construction():
    with pytest.raises(ValueError, match="backbone layers"):
        make_detector({"image_size": 128, "trainable_backbone_layers": 6}, pretrained=False)


def test_larger_roi_masks_keep_checkpoint_shapes_and_train_on_thin_native_targets():
    from torchvision.models.detection.roi_heads import maskrcnn_loss

    model = make_detector(
        {"image_size": 128, "pretrained": False, "mask_roi_size": 28, "anchor_sizes": [16, 32, 64, 128, 256]},
        pretrained=False,
    )
    assert model.rpn.anchor_generator.sizes == ((16,), (32,), (64,), (128,), (256,))
    assert model.rpn.anchor_generator.num_anchors_per_location() == [3] * 5
    assert model.rpn.head.cls_logits.out_channels == 3
    assert model.rpn.head.bbox_pred.out_channels == 12
    assert model.roi_heads.mask_roi_pool.output_size == (28, 28)
    model.roi_heads.mask_head.eval()
    features = {str(level): torch.randn(1, 256, side, side) for level, side in enumerate((32, 16, 8, 4))}
    boxes = [torch.tensor([[10.0, 20.0, 110.0, 100.0]])]
    pooled = model.roi_heads.mask_roi_pool(features, boxes, [(128, 128)])
    logits = model.roi_heads.mask_predictor(model.roi_heads.mask_head(pooled))
    assert logits.shape == (1, 2, 56, 56)
    target = torch.zeros((1, 128, 128), dtype=torch.uint8)
    target[0, 48:53, 20:100] = 1
    loss = maskrcnn_loss(logits, boxes, [target], [torch.tensor([1])], [torch.tensor([0])])
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.roi_heads.mask_predictor.mask_fcn_logits.weight.grad).all()
    # ROI pooling has no learned tensors: original pretrained mask-head state
    # shapes remain compatible when spatial output resolution changes.
    assert model.roi_heads.mask_predictor.mask_fcn_logits.weight.shape == (2, 256, 1, 1)


def test_unsupported_roi_size_fails_before_model_construction():
    with pytest.raises(ValueError, match="mask ROI size"):
        make_detector({"image_size": 128, "mask_roi_size": 0}, pretrained=False)


def test_invalid_anchor_geometry_fails_before_model_construction():
    with pytest.raises(ValueError, match="anchor sizes"):
        make_detector({"image_size": 128, "anchor_sizes": [8, 16, 32, 64]}, pretrained=False)
