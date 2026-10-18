import numpy as np
import pytest

from solar_filament.preprocessing import DetectorPreprocessing


def test_raw_default_keeps_exact_input_and_rejects_invalid_geometry():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert DetectorPreprocessing({})(image) is image
    assert DetectorPreprocessing({"image_preprocessing": "raw"})(image) is image
    for invalid in (None, image.astype(np.float32), image[..., None]):
        with pytest.raises(ValueError, match="uint8 grayscale"):
            DetectorPreprocessing({})(invalid)


def test_clahe_retains_thin_native_geometry_and_does_not_mutate_supplied_pixels():
    yy, xx = np.mgrid[:256, :256]
    image = (120 + xx // 32 + yy // 64).astype(np.uint8)
    image[100:103, 40:210] -= 12
    original = image.copy()
    preprocess = DetectorPreprocessing({"image_preprocessing": "clahe"})
    result = preprocess(image)
    assert np.array_equal(image, original)
    assert result.shape == image.shape and result.dtype == np.uint8
    assert np.array_equal(result, preprocess(image.copy()))
    # At the fixed filament interior and nearby background, the thin dark
    # feature must retain its location and direction without resizing.
    assert np.median(result[100:103, 60:190]) < np.median(result[95:98, 60:190])
    assert not np.array_equal(result, image)


@pytest.mark.parametrize(
    "options",
    [
        {"image_preprocessing": "unknown"},
        {"image_preprocessing": "clahe", "clahe_clip_limit": float("nan")},
        {"image_preprocessing": "clahe", "clahe_clip_limit": 0},
        {"image_preprocessing": "clahe", "clahe_grid_size": True},
        {"image_preprocessing": "clahe", "clahe_grid_size": 8.5},
        {"image_preprocessing": "clahe", "clahe_grid_size": 33},
    ],
)
def test_invalid_preprocessing_configuration_fails_before_training(options):
    with pytest.raises(ValueError):
        DetectorPreprocessing(options)
