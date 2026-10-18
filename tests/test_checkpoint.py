import pytest

torch = pytest.importorskip("torch")

from solar_filament.checkpoint import save_inference_checkpoint  # noqa: E402


def test_inference_checkpoint_restores_model_and_declared_config_without_optimizer(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3)).eval()
    config = {"image_size": 1024, "fold": 0}
    sample = torch.tensor([[0.2, 0.8]])
    expected = model(sample).detach().clone()
    path = tmp_path / "best.pt"
    save_inference_checkpoint(path, model, config, epoch=14, pq=0.4)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    restored = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3)).eval()
    restored.load_state_dict(saved["model"])
    assert torch.equal(restored(sample), expected)
    assert saved["config"] == config
    assert saved["epoch"] == 14
    assert saved["best_pq"] == 0.4
    assert "optimizer" not in saved
    assert not path.with_suffix(".tmp").exists()
