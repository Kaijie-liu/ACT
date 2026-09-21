"""One public input, three author noise levels; not Monte Carlo certification."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from recent_moe_deployment import sha256, supervise


REPO = Path("/data1/Kane/MOE/baselines/recent_moe_20260921/dual_rs")
PYTHON = "/data1/Kane/MOE/envs/dual-rs-author-blackwell-20260921/bin/python"
WEIGHTS = Path("/data1/Kane/MOE/baseline_weights/dual_rs_20260921")
ROOT = Path("/data1/Kane/MOE/baseline_runs/dual_rs_install_20260921/native_model_probe")
COMMIT = "6f83aeb7f47466b1dee6295baad8d59a8c94eceb"


def worker():
    start = time.monotonic()
    if subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("author revision drift")
    manifest = json.loads((WEIGHTS / "manifest.json").read_text())
    for row in manifest["files"]:
        if sha256(WEIGHTS / row["filename"]) != row["sha256"]:
            raise ValueError("public model identity mismatch")
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise ValueError("no CUDA; do not silently replace author native path")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024 ** 3:
        raise ValueError("insufficient free GPU memory; no retry in this probe")
    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.manual_seed(1)
    np.random.seed(1)
    # No pickle allowlisting/fallback. All downloadable payloads must be plain tensor states.
    state_paths = [WEIGHTS / "cifar10_uncond_50M_500K.pt", WEIGHTS / "vit/pytorch_model.bin"]
    for path in state_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(
                isinstance(k, str) and isinstance(v, torch.Tensor) and torch.isfinite(v).all()
                for k, v in state.items()):
            raise ValueError("not a finite tensor state_dict")
        del state
    gc.collect()
    sys.path.insert(0, str(REPO / "code"))
    from DRM_classifier import DiffusionRobustModel
    model = DiffusionRobustModel(str(state_paths[0]), str(WEIGHTS / "vit")).eval()
    dataset = CIFAR10("/data1/Kane/MOE/baseline_data/metamoe_20260921", train=False,
                      download=False, transform=ToTensor())
    x, label = dataset[0]
    input_hash = hashlib.sha256(x.numpy().tobytes()).hexdigest()
    x = x.unsqueeze(0).cuda()
    rows = []
    for sigma in [0.25, 0.5, 1.0]:
        t = 0
        while model.diffusion.sqrt_one_minus_alphas_cumprod[t] / model.diffusion.sqrt_alphas_cumprod[t] < sigma * 2:
            t += 1
        torch.cuda.synchronize()
        before = time.monotonic()
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(x, t)
        torch.cuda.synchronize()
        if logits.shape != (1, 10) or not torch.isfinite(logits).all():
            raise ValueError("invalid native classifier output")
        rows.append({"sigma": sigma, "timestep": t, "logits": logits.float().cpu().tolist(),
                     "prediction": int(logits.argmax(1)), "forward_seconds": time.monotonic() - before})
    print(json.dumps({"status": "NATIVE_PRETRAINED_COMPONENT_PROBE_PASS", "author_commit": COMMIT,
                      "weights_manifest_sha256": sha256(WEIGHTS / "manifest.json"),
                      "dataset": "CIFAR10 test", "index": 0, "label": label,
                      "input_float32_3x32x32_bytes_sha256": input_hash,
                      "seed": 1, "batch": 1, "torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(), "free_bytes_at_start": free,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "worker_seconds": time.monotonic() - start, "rows": rows,
                      "sigma_estimator_present": False, "monte_carlo_certification_executed": False,
                      "scope": "unaltered pretrained denoiser+ViT at three sigma levels, not accuracy/CRA"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if a.worker:
        worker()
        return
    result = supervise([PYTHON, str(Path(__file__).resolve()), "--worker"],
                       str(REPO), ROOT, 180, "AUTHOR_CHECKPOINT_SMOKE", REPO, cpu_only=False)
    print(json.dumps({k: v for k, v in result.items() if k != "source_before"}, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
