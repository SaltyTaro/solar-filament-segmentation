import numpy as np
import pytest
from pycocotools import mask as mask_utils

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from solar_filament.detection import detector_instances  # noqa: E402
from solar_filament.detector_cache import cached_instances, pack_probabilities  # noqa: E402


def test_compact_detector_cache_retains_threshold_and_pixel_ownership_exactly():
    masks = np.zeros((3, 64, 64), dtype=np.float32)
    masks[0, 5:15, 5:25] = 0.5
    masks[1, 5:15, 15:35] = 0.7
    masks[2, 45:55, 50:64] = 0.4
    scores = np.array([0.6, 0.9, 0.4], dtype=np.float32)
    packed = pack_probabilities(scores, masks)
    for threshold in (0.4, 0.5, 0.7):
        config = {
            "score_threshold": 0.3,
            "mask_threshold": threshold,
            "minimum_area": 50,
            "maximum_area": 50000,
        }
        native = detector_instances(
            {"scores": torch.from_numpy(scores), "masks": torch.from_numpy(masks[:, None])}, config
        )
        cached = cached_instances(packed, config)
        assert np.array_equal(mask_utils.decode(native), mask_utils.decode(cached))
    assert packed["probabilities"].nbytes < masks.nbytes / 5


def test_compact_detector_cache_retains_empty_detection_geometry():
    packed = pack_probabilities(np.empty(0, dtype=np.float32), np.empty((0, 64, 64), dtype=np.float32))
    assert (
        cached_instances(
            packed, {"score_threshold": 0.3, "mask_threshold": 0.5, "minimum_area": 50, "maximum_area": 50000}
        )
        == []
    )
