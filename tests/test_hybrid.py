import numpy as np
from pycocotools import mask as mask_utils

from solar_filament.hybrid import hybrid_instances


def test_detector_seed_selects_native_component_and_preserves_disjoint_ownership():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:20, 10:30] = 0.8
    probability[40:50, 40:55] = 0.9
    proposals = [
        {"score": 0.9, "bbox_xyxy": [9, 9, 31, 21]},
        {"score": 0.6, "bbox_xyxy": [10, 10, 30, 20]},
        {"score": 0.8, "bbox_xyxy": [39, 39, 56, 51]},
    ]
    config = {"score_threshold": 0.3, "mask_threshold": 0.5, "minimum_area": 50, "maximum_area": 50000}
    instances = hybrid_instances(probability, proposals, config)
    assert len(instances) == 2
    masks = mask_utils.decode(instances)
    assert np.all(masks.sum(axis=2) <= 1)
    assert np.array_equal(masks.sum(axis=2), probability > 0.5)
