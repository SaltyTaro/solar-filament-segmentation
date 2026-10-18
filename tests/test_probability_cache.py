import hashlib

import numpy as np
import pytest

from solar_filament.probability_cache import compress_probability_file, load_probability


def test_lossless_probability_migration_retains_npy_checksum_and_tuned_pixels(tmp_path):
    root = tmp_path.resolve()
    probability = np.random.default_rng(4).random((128, 128)).astype(np.float16)
    path = root / "image.npy"
    np.save(path, probability)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before = load_probability(root, "image")
    record = compress_probability_file(path, root, remove_verified_original=True)
    after = load_probability(root, "image")
    assert record["source_sha256"] == original_hash
    assert record["reconstructed_npy_sha256_verified"]
    assert not path.exists()
    assert np.array_equal(before, after)
    assert np.array_equal(before >= 0.65, after >= 0.65)


def test_migration_refuses_mismatched_existing_compressed_cache(tmp_path):
    root = tmp_path.resolve()
    path = root / "image.npy"
    np.save(path, np.ones((16, 16), dtype=np.float16))
    np.savez_compressed(path.with_suffix(".npz"), probability=np.zeros((16, 16), dtype=np.float16))
    with pytest.raises(ValueError, match="does not match"):
        compress_probability_file(path, root, remove_verified_original=True)
    assert path.exists()
