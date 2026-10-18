import numpy as np
import pytest

from solar_filament.proposals import box_iou


def test_box_overlap_preserves_detection_geometry():
    truth = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    predicted = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
    result = box_iou(truth, predicted)
    assert result[0, 0] == 1
    assert result[0, 1] == pytest.approx(1 / 3)
    assert np.array_equal(result[1], np.zeros(2))
    assert box_iou(truth, np.empty((0, 4))).shape == (2, 0)
