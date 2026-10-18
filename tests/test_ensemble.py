import numpy as np
import pytest
from pycocotools import mask as mask_utils

from solar_filament.coco import encode_mask
from solar_filament.ensemble import fuse_binary_instances


def rectangle(left, top, right, bottom):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return encode_mask(mask)


def test_paired_thin_instances_keep_native_union_and_intersection_geometry():
    a, b = rectangle(10, 20, 40, 24), rectangle(12, 21, 42, 25)
    union = fuse_binary_instances([a], [b], mode="union", retain_unmatched_first=False, minimum_area=1)
    intersection = fuse_binary_instances([a], [b], mode="intersection", retain_unmatched_first=False, minimum_area=1)
    assert len(union) == len(intersection) == 1
    assert int(mask_utils.area(union[0])) == 156
    assert int(mask_utils.area(intersection[0])) == 84
    assert np.array_equal(mask_utils.toBbox(intersection[0]), [12, 21, 28, 3])


def test_one_second_instance_cannot_confirm_two_first_instances():
    a, b = rectangle(10, 10, 20, 20), rectangle(20, 10, 30, 20)
    merged = rectangle(10, 10, 30, 20)
    consensus = fuse_binary_instances([a, b], [merged], mode="first", retain_unmatched_first=False, minimum_area=1)
    recall_reference = fuse_binary_instances([a, b], [merged], mode="first", retain_unmatched_first=True, minimum_area=1)
    assert len(consensus) == 1
    assert len(recall_reference) == 2


def test_union_and_retained_instances_are_disjoint_after_native_pixel_ownership():
    a, b = rectangle(10, 10, 20, 20), rectangle(20, 10, 30, 20)
    wider_a = rectangle(10, 10, 22, 20)
    result = fuse_binary_instances([a, b], [wider_a], mode="union", retain_unmatched_first=True, minimum_area=1)
    masks = mask_utils.decode(result)
    assert len(result) == 2
    assert np.all(masks.sum(axis=2) <= 1)
    assert list(mask_utils.area(result)) == [120, 80]


def test_no_correspondence_retains_only_declared_first_model_instances():
    a, b = rectangle(10, 10, 20, 20), rectangle(40, 40, 50, 50)
    assert not fuse_binary_instances([a], [b], mode="union", retain_unmatched_first=False, minimum_area=1)
    result = fuse_binary_instances([a], [b], mode="union", retain_unmatched_first=True, minimum_area=1)
    assert len(result) == 1
    assert np.array_equal(mask_utils.decode(result[0]), mask_utils.decode(a))
    assert not fuse_binary_instances([], [], mode="union", retain_unmatched_first=False)


def test_ensemble_refuses_incompatible_native_dimensions():
    with pytest.raises(ValueError, match="native image dimensions"):
        fuse_binary_instances([rectangle(10, 10, 20, 20)], [encode_mask(np.ones((32, 32), dtype=np.uint8))], mode="union", retain_unmatched_first=False)
