"""Read native probability caches with lossless compression and strict geometry."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from solar_filament.storage import artifact_path


def cache_file(directory: Path, stem: str) -> Path:
    candidates = [directory / (stem + suffix) for suffix in (".npz", ".npy")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No probability cache for {stem}")


def load_probability(directory: Path, stem: str) -> np.ndarray:
    path = cache_file(directory, stem)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["probability"]:
                raise ValueError(f"Unexpected probability cache members: {path}")
            probability = archive["probability"]
    else:
        probability = np.load(path, allow_pickle=False)
    if probability.ndim != 2 or probability.dtype not in (np.float16, np.float32):
        raise ValueError(f"Invalid probability cache: {path}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"Cache has invalid probabilities: {path}")
    return probability.astype(np.float32)


def compress_probability_file(path: Path, root: Path, remove_verified_original: bool = False) -> dict:
    path = artifact_path(path, root)
    if path.suffix != ".npy":
        raise ValueError("Only uncompressed NPY probability caches can be migrated")
    original = np.load(path, allow_pickle=False)
    if original.ndim != 2 or original.dtype not in (np.float16, np.float32):
        raise ValueError("Expected a native two-dimensional probability array")
    if not np.isfinite(original).all() or np.any(original < 0) or np.any(original > 1):
        raise ValueError("Expected finite probabilities in [0, 1]")
    destination = path.with_suffix(".npz")
    original_bytes = path.stat().st_size
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, probability=original)
        with np.load(temporary, allow_pickle=False) as archive:
            restored = archive["probability"]
        if (
            original.dtype != restored.dtype
            or original.shape != restored.shape
            or original.tobytes() != restored.tobytes()
        ):
            raise ValueError("Compressed probability cache failed byte-for-byte array verification")
        temporary.replace(destination)
    else:
        with np.load(destination, allow_pickle=False) as archive:
            restored = archive["probability"]
        if (
            original.dtype != restored.dtype
            or original.shape != restored.shape
            or original.tobytes() != restored.tobytes()
        ):
            raise ValueError("Existing compressed cache does not match the original")
    reconstructed = io.BytesIO()
    np.save(reconstructed, restored, allow_pickle=False)
    if hashlib.sha256(reconstructed.getvalue()).hexdigest() != original_digest:
        raise ValueError("Restored NPY bytes do not match the original file checksum")
    record = {
        "source": str(path),
        "destination": str(destination),
        "source_sha256": original_digest,
        "compressed_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_bytes": original_bytes,
        "compressed_bytes": destination.stat().st_size,
        "array_bytes_verified": True,
        "reconstructed_npy_sha256_verified": True,
        "original_removed": remove_verified_original,
    }
    audit_file = destination.with_suffix(".compression.json")
    audit_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if remove_verified_original:
        path.unlink()
    return record
