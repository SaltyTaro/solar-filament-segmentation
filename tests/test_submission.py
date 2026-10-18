import csv

import numpy as np
import pytest

from solar_filament.coco import decode_counts, encode_local_mask, encode_mask
from solar_filament.submission import load_predictions, validate_predictions, write_predictions

STEM = "20110109104734Ch"


def test_coco_round_trip_retains_asymmetric_geometry():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[15:22, 132:171] = 1
    mask[21:32, 169:171] = 1
    rle = encode_mask(mask)
    assert np.array_equal(decode_counts(rle["counts"].decode("ascii")), mask)


@pytest.mark.parametrize("x,y,height,width", [(0, 0, 11, 15), (10, 7, 11, 15), (7, 0, 11, 15)])
def test_local_rle_keeps_global_geometry_at_image_borders(x, y, height, width):
    from pycocotools import mask as mask_utils

    crop = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [0, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=np.uint8)
    expected = np.zeros((height, width), dtype=np.uint8)
    expected[y : y + 4, x : x + 5] = crop
    encoded = encode_local_mask(crop, x, y, height, width)
    assert np.array_equal(mask_utils.decode(encoded), expected)


def test_local_rle_handles_connected_runs_across_columns_and_empty_masks():
    from pycocotools import mask as mask_utils

    for value in [0, 1]:
        mask = np.full((11, 15), value, dtype=np.uint8)
        encoded = encode_local_mask(mask, 0, 0, 11, 15)
        assert np.array_equal(mask_utils.decode(encoded), mask)


def test_csv_round_trip_and_explicit_zero_detection_images(tmp_path):
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[400:410, 100:110] = 1
    path = tmp_path / "submission.csv"
    write_predictions(path, {STEM: [encode_mask(mask)]})
    predictions = load_predictions(path)
    result = validate_predictions(predictions, {STEM, "20110119082634Lh"})
    assert result["filament_rows"] == 1
    assert result["images_without_predictions"] == ["20110119082634Lh"]


def test_overlapping_or_empty_instances_are_rejected():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    with pytest.raises(ValueError, match="Overlapping"):
        validate_predictions({STEM: [encode_mask(mask), encode_mask(mask)]}, {STEM})
    with pytest.raises(ValueError, match="Empty"):
        validate_predictions({STEM: [encode_mask(np.zeros_like(mask))]}, {STEM})


def test_unknown_image_and_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown image"):
        validate_predictions({STEM: []}, set())
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    path = tmp_path / "duplicate.csv"
    counts = encode_mask(mask)["counts"].decode("ascii")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerow([STEM + "_1", counts])
        writer.writerow([STEM + "_1", counts])
    with pytest.raises(ValueError, match="Duplicate"):
        load_predictions(path)


@pytest.mark.parametrize("counts", ["0", "'", "P", "P" * 20])
def test_malformed_rle_is_rejected_before_native_decoding(counts):
    with pytest.raises(ValueError):
        decode_counts(counts)
