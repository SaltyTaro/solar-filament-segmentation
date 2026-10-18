"""Lossless byte-plane archives preserve original checkpoint hashes and payloads."""

from __future__ import annotations

import hashlib
import io
import json
import zlib
from pathlib import Path

import numpy as np

from solar_filament.storage import artifact_path


def archive_manifest(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".archive.json")


def restored_checkpoint_bytes(path: Path) -> tuple[bytearray, dict]:
    record = json.loads(archive_manifest(path).read_text(encoding="utf-8"))
    archived = path.parent / record["archive_name"]
    if archived.name != record["archive_name"] or archived.parent.resolve() != path.parent.resolve():
        raise ValueError("Checkpoint archive must stay in the original run directory")
    if hashlib.sha256(archived.read_bytes()).hexdigest() != record["archive_sha256"]:
        raise ValueError("Checkpoint archive checksum mismatch")
    restored = bytearray()
    with archived.open("rb") as handle:
        for chunk in record["chunks"]:
            if chunk["offset"] != handle.tell() or not 0 < chunk["native_bytes"] <= 4 * 1024**2:
                raise ValueError("Invalid checkpoint archive chunk geometry")
            compressed = handle.read(chunk["compressed_bytes"])
            planes = zlib.decompress(compressed)
            padded_bytes = (chunk["native_bytes"] + 3) // 4 * 4
            if len(planes) != padded_bytes:
                raise ValueError("Checkpoint archive chunk has invalid decoded size")
            native = np.frombuffer(planes, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()
            restored.extend(native[:chunk["native_bytes"]])
        if handle.read(1):
            raise ValueError("Checkpoint archive contains unexpected trailing bytes")
    if len(restored) != record["native_bytes"] or hashlib.sha256(restored).hexdigest() != record["original_sha256"]:
        raise ValueError("Restored checkpoint checksum mismatch")
    return restored, record


def archived_checkpoint_source(path: Path) -> io.BytesIO:
    values, _ = restored_checkpoint_bytes(path)
    return io.BytesIO(values)


def archive_checkpoint(path: Path, root: Path, *, remove_verified_original: bool = False) -> dict:
    """Archive one completed file; callers must exclude active checkpoint writers."""
    path = artifact_path(path, root)
    if path.suffix != ".pt":
        raise ValueError("Checkpoint archives accept only original .pt files")
    manifest = archive_manifest(path)
    if manifest.exists():
        _, record = restored_checkpoint_bytes(path)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != record["original_sha256"]:
            raise ValueError("Checkpoint differs from its existing archive")
        return record
    native_size, native_digest = path.stat().st_size, hashlib.sha256()
    archived = artifact_path(path.with_suffix(path.suffix + ".zlib"), root)
    temporary = artifact_path(archived.with_suffix(archived.suffix + ".tmp"), root)
    if archived.exists() or temporary.exists():
        raise ValueError("Unrecorded checkpoint archive exists; preserve it for review")
    chunks = []
    with path.open("rb") as source, temporary.open("wb") as destination:
        while native := source.read(4 * 1024**2):
            native_digest.update(native)
            padded = native + b"\0" * ((-len(native)) % 4)
            planes = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
            compressed = zlib.compress(planes, level=6)
            chunks.append({"offset": destination.tell(), "native_bytes": len(native), "compressed_bytes": len(compressed)})
            destination.write(compressed)
    if temporary.stat().st_size >= native_size:
        raise ValueError("Archive does not save storage; original checkpoint retained")
    record = {
        "format": "four-byte-plane zlib chunks; original file bytes preserved",
        "archive_name": archived.name, "native_bytes": native_size,
        "original_sha256": native_digest.hexdigest(), "archive_bytes": temporary.stat().st_size,
        "archive_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
        "chunks": chunks, "original_removed": False,
    }
    temporary.replace(archived)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    restored, _ = restored_checkpoint_bytes(path)
    if bytes(restored) != path.read_bytes():
        raise ValueError("Checkpoint archive differs from the original bytes")
    del restored
    if remove_verified_original:
        if path.stat().st_size != native_size or hashlib.sha256(path.read_bytes()).hexdigest() != record["original_sha256"]:
            raise ValueError("Checkpoint changed while being archived; preserve original")
        path.unlink()
        record["original_removed"] = True
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
