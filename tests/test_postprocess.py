import numpy as np
import pytest
from pycocotools import mask as mask_utils

from solar_filament.postprocess import probability_instances


def test_hysteresis_retains_faint_boundary_but_rejects_unseeded_texture():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[5:15, 10:30] = 0.4
    probability[8:12, 15:25] = 0.8
    probability[40:45, 40:50] = 0.4
    instances = probability_instances(
        probability, {"mask_threshold": 0.3, "seed_threshold": 0.7, "minimum_area": 20}
    )
    assert len(instances) == 1
    result = mask_utils.decode(instances[0])
    assert result.sum() == 200
    assert result[5, 10] == 1
    assert result[40, 40] == 0


def test_component_filtering_keeps_geometry_and_disjoint_instances():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[5:10, 10:20] = 0.8
    probability[30:35, 40:50] = 0.9
    probability[55, 55] = 1
    instances = probability_instances(probability, {"mask_threshold": 0.5, "minimum_area": 10})
    assert len(instances) == 2
    masks = mask_utils.decode(instances)
    assert np.array_equal(masks.sum(axis=2), (probability > 0.5) & (probability != 1))


def test_confidence_filter_uses_all_component_pixels():
    probability = np.full((16, 16), 0.4, dtype=np.float32)
    probability[0, 0] = 0.9
    assert not probability_instances(
        probability, {"mask_threshold": 0.3, "minimum_area": 20, "minimum_confidence": 0.6}
    )


def test_invalid_probability_map_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        probability_instances(np.full((16, 16), np.nan), {"mask_threshold": 0.5})
