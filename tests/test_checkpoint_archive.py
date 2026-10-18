import hashlib
import json

import pytest

from solar_filament.checkpoint_archive import archive_checkpoint, archive_manifest
from solar_filament.checkpoint_delta import checkpoint_source


def test_lossless_archive_restores_chunk_boundaries_padding_and_original_hash(tmp_path):
    path = tmp_path / "last.pt"
    original = bytes(range(251)) * 20000 + b"end"
    path.write_bytes(original)
    record = archive_checkpoint(path, tmp_path, remove_verified_original=True)
    assert len(record["chunks"]) == 2
    assert record["original_removed"] and not path.exists()
    assert record["archive_bytes"] < record["native_bytes"]
    assert checkpoint_source(path).read() == original
    assert record["original_sha256"] == hashlib.sha256(original).hexdigest()


def test_archive_rejects_corruption_and_original_file_takes_precedence(tmp_path):
    path = tmp_path / "last.pt"
    path.write_bytes(b"abcde" * 1000)
    archive_checkpoint(path, tmp_path, remove_verified_original=True)
    compressed = path.with_suffix(".pt.zlib")
    compressed.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive checksum"):
        checkpoint_source(path)
    path.write_bytes(b"new normal checkpoint")
    assert checkpoint_source(path) == path


def test_torch_archive_preserves_model_optimizer_and_rng_state(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(16, 8)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(4, 16)).sum().backward()
    optimizer.step()
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
    path = tmp_path / "last.pt"
    torch.save(payload, path)
    archive_checkpoint(path, tmp_path, remove_verified_original=True)
    restored = torch.load(checkpoint_source(path), map_location="cpu", weights_only=False)
    assert torch.equal(restored["model"]["weight"], payload["model"]["weight"])
    assert torch.equal(restored["optimizer"]["state"][0]["exp_avg"], payload["optimizer"]["state"][0]["exp_avg"])
    assert torch.equal(restored["rng"], payload["rng"])


def test_archive_manifest_cannot_escape_run_directory(tmp_path):
    path = tmp_path / "last.pt"
    path.write_bytes(b"abc" * 1000)
    archive_checkpoint(path, tmp_path, remove_verified_original=True)
    manifest = archive_manifest(path)
    data = json.loads(manifest.read_text())
    data["archive_name"] = "../outside.zlib"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="original run directory"):
        checkpoint_source(path)
