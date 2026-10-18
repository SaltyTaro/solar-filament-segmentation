import hashlib
import io

import pytest

from solar_filament.checkpoint_archive import archive_checkpoint
from solar_filament.checkpoint_delta import archive_duplicate_checkpoint, checkpoint_source


def test_delta_archival_restores_exact_original_and_checks_base_checksum(tmp_path):
    base, target = tmp_path / "best.pt", tmp_path / "last.pt"
    base.write_bytes(bytes(range(256)) * 4096)
    original = bytearray(base.read_bytes())
    original[0:4] = b"LAST"
    original[-4:] = b"END!"
    target.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    record = archive_duplicate_checkpoint(target, base, tmp_path, remove_original=True)
    assert record["original_sha256"] == digest
    assert record["original_removed"]
    assert not target.exists()
    assert isinstance(checkpoint_source(target), io.BytesIO)
    assert hashlib.sha256(checkpoint_source(target).read()).hexdigest() == digest
    base.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="base checksum"):
        checkpoint_source(target)


def test_delta_archival_preserves_substantially_different_checkpoints(tmp_path):
    base, target = tmp_path / "best.pt", tmp_path / "last.pt"
    base.write_bytes(b"a" * (2 * 1024**2))
    target.write_bytes(b"b" * (2 * 1024**2))
    with pytest.raises(ValueError, match="substantially"):
        archive_duplicate_checkpoint(target, base, tmp_path, remove_original=True)
    assert target.is_file()


def test_delta_restores_original_after_its_base_is_losslessly_archived(tmp_path):
    base, target = tmp_path / "best.pt", tmp_path / "last.pt"
    base.write_bytes(bytes(range(251)) * 20000)
    original = bytearray(base.read_bytes())
    original[1020:1028] = b"lastcopy"
    target.write_bytes(original)
    archive_duplicate_checkpoint(target, base, tmp_path, remove_original=True)
    archive_checkpoint(base, tmp_path, remove_verified_original=True)
    assert checkpoint_source(target).read() == original
    assert checkpoint_source(base).read() == bytes(range(251)) * 20000
    base.with_suffix(".pt.zlib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive checksum"):
        checkpoint_source(target)


def test_torch_checkpoint_delta_preserves_model_optimizer_and_rng_payload(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1, 4)).sum()
    loss.backward()
    optimizer.step()
    record = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
    base, target = tmp_path / "best.pt", tmp_path / "last.pt"
    torch.save(record, base)
    torch.save(record, target)
    original_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    archived = archive_duplicate_checkpoint(target, base, tmp_path, remove_original=True)
    assert archived["original_sha256"] == original_digest
    restored = torch.load(checkpoint_source(target), weights_only=False)
    assert torch.equal(restored["model"]["weight"], record["model"]["weight"])
    assert torch.equal(restored["rng"], record["rng"])
    assert torch.equal(
        restored["optimizer"]["state"][0]["exp_avg"], record["optimizer"]["state"][0]["exp_avg"]
    )
