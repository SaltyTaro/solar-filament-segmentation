import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from solar_filament.refiner import crop_geometry, crop_inputs  # noqa: E402


@pytest.mark.parametrize("box", [[0, 0, 40, 20], [1990, 1990, 2048, 2048], [800, 900, 1000, 1050]])
def test_instance_crop_retains_box_geometry_at_disk_image_borders(box):
    x, y, side = crop_geometry(box, (2048, 2048))
    assert 0 <= x <= 2048 - side
    assert 0 <= y <= 2048 - side
    assert x <= box[0] and y <= box[1]
    assert x + side >= box[2] and y + side >= box[3]
    inputs, geometry = crop_inputs(np.full((2048, 2048), 128, dtype=np.uint8), box, 512)
    assert inputs.shape == (2, 512, 512)
    assert geometry == (x, y, side)
    assert set(np.unique(inputs[1])) <= {0, 1}
    assert inputs[1].sum() > 0


def test_prompt_channel_represents_the_selected_box_in_native_crop_coordinates():
    image = np.zeros((2048, 2048), dtype=np.uint8)
    inputs, (x, y, side) = crop_inputs(image, [1000, 1000, 1100, 1050], 256)
    assert side == 256
    assert inputs[1, int(1020 - y), int(1050 - x)] == 1
    assert inputs[1, 0, 0] == 0
