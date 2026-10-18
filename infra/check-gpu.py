"""Verify CUDA kernels and record the training environment under the storage root."""

from __future__ import annotations

import json
import platform

import torch
import torchvision

from solar_filament.storage import artifact_path, storage_root


def main():
    root = storage_root()
    output = artifact_path(root / "artifacts" / "gpu-environment.json", root)
    if not torch.cuda.is_available():
        raise RuntimeError("Installed Torch cannot access CUDA")
    tensor = torch.randn(128, 128, device="cuda", requires_grad=True)
    loss = tensor.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(tensor.grad).all():
        raise RuntimeError("CUDA backward produced invalid gradients")
    properties = torch.cuda.get_device_properties(0)
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_memory_gib": properties.total_memory / 2**30,
        "cuda_forward_backward": "passed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
