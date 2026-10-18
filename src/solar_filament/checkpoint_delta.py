"""Preserve completed redundant checkpoints as verified byte-exact deltas."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from solar_filament.checkpoint_archive import (
    archive_manifest,
    archived_checkpoint_source,
    restored_checkpoint_bytes,
)
from solar_filament.storage import artifact_path


def delta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".delta.json")


def restored_bytes(path: Path) -> tuple[bytearray, dict]:
    record = json.loads(delta_path(path).read_text(encoding="utf-8"))
    base = path.parent / record["base_name"]
    if base.parent.resolve() != path.parent.resolve() or base.name != record["base_name"]:
        raise ValueError("Checkpoint delta base must be in the same run directory")
    if base.is_file():
        values = bytearray(base.read_bytes())
    elif archive_manifest(base).is_file():
        values, _ = restored_checkpoint_bytes(base)
    else:
        raise FileNotFoundError(base)
    if len(values) != record["bytes"] or hashlib.sha256(values).hexdigest() != record["base_sha256"]:
        raise ValueError("Checkpoint delta base checksum mismatch")
    for patch in record["patches"]:
        offset = patch["offset"]
        replacement = base64.b64decode(patch["bytes_base64"], validate=True)
        if offset < 0 or offset + len(replacement) > len(values):
            raise ValueError("Checkpoint delta extends outside the original file")
        values[offset : offset + len(replacement)] = replacement
    if hashlib.sha256(values).hexdigest() != record["original_sha256"]:
        raise ValueError("Restored checkpoint checksum mismatch")
    return values, record


def checkpoint_source(path: Path):
    """Provide a file or exact reconstructed checkpoint bytes to torch.load."""
    if path.is_file():
        return path
    if archive_manifest(path).is_file():
        return archived_checkpoint_source(path)
    if delta_path(path).is_file():
        values, _ = restored_bytes(path)
        return io.BytesIO(values)
    raise FileNotFoundError(path)


def archive_duplicate_checkpoint(path: Path, base: Path, root: Path, remove_original: bool = False) -> dict:
    path, base = artifact_path(path, root), artifact_path(base, root)
    if path.parent != base.parent or path == base:
        raise ValueError("A checkpoint delta must use another checkpoint from the same run")
    if path.stat().st_size != base.stat().st_size:
        raise ValueError("Redundant checkpoint sizes differ")
    original_digest, base_digest = hashlib.sha256(), hashlib.sha256()
    patches, offset, changed_bytes = [], 0, 0
    with path.open("rb") as original, base.open("rb") as reference:
        while target := original.read(4 * 1024**2):
            source = reference.read(len(target))
            original_digest.update(target)
            base_digest.update(source)
            changed = np.flatnonzero(
                np.frombuffer(target, dtype=np.uint8) != np.frombuffer(source, dtype=np.uint8)
            )
            changed_bytes += len(changed)
            if changed_bytes > 1024**2:
                raise ValueError("Checkpoints differ substantially; preserve both full files")
            if len(changed):
                groups = np.split(changed, np.flatnonzero(np.diff(changed) > 1) + 1)
                for group in groups:
                    start, end = int(group[0]), int(group[-1]) + 1
                    patches.append(
                        {
                            "offset": offset + start,
                            "bytes_base64": base64.b64encode(target[start:end]).decode(),
                        }
                    )
            offset += len(target)
    record = {
        "base_name": base.name,
        "base_sha256": base_digest.hexdigest(),
        "original_sha256": original_digest.hexdigest(),
        "bytes": offset,
        "changed_bytes": changed_bytes,
        "patches": patches,
        "original_removed": False,
    }
    destination = delta_path(path)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    restored, _ = restored_bytes(path)
    if hashlib.sha256(restored).hexdigest() != original_digest.hexdigest():
        raise ValueError("Checkpoint reconstruction differs from its original file")
    del restored
    if remove_original:
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["original_sha256"]:
            raise ValueError("Checkpoint changed during archival; preserve the original")
        path.unlink()
        record["original_removed"] = True
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in record.items() if key != "patches"}
