"""Replay every ordered B1 PGD50 endpoint through the full routed model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.baseline_icml2025_b1_endpoint import (
    _load_model,
    _normalized_batch,
    _pixel_linf,
    _pixel_range,
)
from act.pipeline.moe.baseline_icml2025_b1_smoke import _inside, _sha256


def run(protocol_path: Path, endpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    protocol_path = _inside(protocol_path)
    endpoint_dir = _inside(endpoint_dir)
    output_path = _inside(output_path)
    if output_path.exists():
        raise RuntimeError(f"refuses to overwrite {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary_path = endpoint_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact_path = endpoint_dir / "ordered_endpoint_replay.npz"
    issues: list[str] = []
    if summary.get("status") != "COMPLETED_NOT_INDEPENDENTLY_AUDITED":
        issues.append("endpoint status changed")
    if summary.get("protocol", {}).get("sha256") != _sha256(protocol_path):
        issues.append("endpoint protocol hash changed")
    if summary.get("artifact", {}).get("sha256") != _sha256(artifact_path):
        issues.append("endpoint artifact hash changed")
    checkpoint = Path(summary["checkpoint"]["path"])
    device = torch.device("cuda")
    model, _payload = _load_model(checkpoint, device)

    import torchvision.datasets as datasets

    dataset = datasets.CIFAR10(
        root=str(Path(protocol["run_root"]) / "data"), train=False, download=False
    )
    raw = np.asarray(dataset.data, dtype=np.uint8)
    with np.load(artifact_path, allow_pickle=False) as arrays:
        labels = arrays["labels"].astype(np.int64)
        stored_clean = arrays["clean_predictions"].astype(np.int64)
        stored_adversarial = arrays["adversarial_predictions"].astype(np.int64)
        stored_clean_routes = arrays["clean_routes"].astype(np.int64)
        stored_adversarial_routes = arrays["adversarial_routes"].astype(np.int64)
        adversarial_endpoints = arrays["adversarial_endpoints"].astype(np.float32)
        stored_linf = arrays["pixel_linf"].astype(np.float64)
        stored_minimum_pixels = arrays["minimum_pixels"].astype(np.float64)
        stored_maximum_pixels = arrays["maximum_pixels"].astype(np.float64)
    if not np.array_equal(labels, np.asarray(dataset.targets, dtype=np.int64)):
        issues.append("ordered labels changed")
    clean_rows: list[np.ndarray] = []
    adversarial_rows: list[np.ndarray] = []
    clean_route_rows: list[np.ndarray] = []
    adversarial_route_rows: list[np.ndarray] = []
    replay_linf: list[np.ndarray] = []
    replay_minimum_pixels: list[np.ndarray] = []
    replay_maximum_pixels: list[np.ndarray] = []
    batch_size = int(protocol["endpoint"]["batch_size"])
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        clean = _normalized_batch(raw[start:stop], device)
        adversarial = torch.from_numpy(adversarial_endpoints[start:stop]).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            clean_output = model(clean)
            adversarial_output = model(adversarial)
            clean_routes = model.router(clean)
            adversarial_routes = model.router(adversarial)
        clean_rows.append(clean_output.argmax(dim=1).cpu().numpy())
        adversarial_rows.append(adversarial_output.argmax(dim=1).cpu().numpy())
        clean_route_rows.append(clean_routes.cpu().numpy())
        adversarial_route_rows.append(adversarial_routes.cpu().numpy())
        replay_linf.append(_pixel_linf(clean, adversarial).cpu().numpy())
        minimum_pixels, maximum_pixels = _pixel_range(adversarial)
        replay_minimum_pixels.append(minimum_pixels.cpu().numpy())
        replay_maximum_pixels.append(maximum_pixels.cpu().numpy())
    replay_clean = np.concatenate(clean_rows)
    replay_adversarial = np.concatenate(adversarial_rows)
    replay_clean_routes = np.concatenate(clean_route_rows)
    replay_adversarial_routes = np.concatenate(adversarial_route_rows)
    linf = np.concatenate(replay_linf)
    minimum_pixels = np.concatenate(replay_minimum_pixels)
    maximum_pixels = np.concatenate(replay_maximum_pixels)
    for name, observed, expected in (
        ("clean predictions", replay_clean, stored_clean),
        ("adversarial predictions", replay_adversarial, stored_adversarial),
        ("clean routes", replay_clean_routes, stored_clean_routes),
        ("adversarial routes", replay_adversarial_routes, stored_adversarial_routes),
    ):
        if not np.array_equal(observed, expected):
            issues.append(f"{name} failed replay")
    if not np.allclose(linf, stored_linf, atol=2e-5, rtol=0):
        issues.append("pixel Linf failed replay")
    if not np.allclose(minimum_pixels, stored_minimum_pixels, atol=2e-3, rtol=0):
        issues.append("minimum pixel values failed replay")
    if not np.allclose(maximum_pixels, stored_maximum_pixels, atol=2e-3, rtol=0):
        issues.append("maximum pixel values failed replay")
    if np.any(linf > float(protocol["endpoint"]["epsilon"]) + 2e-5):
        issues.append("replayed endpoint exceeds box")
    if np.any(minimum_pixels < -2e-3) or np.any(maximum_pixels > 255.0 + 2e-3):
        issues.append("replayed endpoint exceeds pixel domain")
    sa = float(100.0 * np.mean(replay_clean == labels))
    ra = float(100.0 * np.mean(replay_adversarial == labels))
    if not np.isclose(sa, summary.get("standard_accuracy_percent"), atol=0, rtol=0):
        issues.append("standard accuracy changed")
    if not np.isclose(ra, summary.get("pgd50_accuracy_percent"), atol=0, rtol=0):
        issues.append("PGD50 accuracy changed")
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "scope": "INDEPENDENT_FULL_MODEL_REPLAY_B1_ORDERED_PGD50_ENDPOINTS",
        "summary_sha256": _sha256(summary_path),
        "artifact_sha256": _sha256(artifact_path),
        "samples_replayed": int(len(labels)),
        "standard_accuracy_percent": sa,
        "pgd50_accuracy_percent": ra,
        "maximum_pixel_linf": float(linf.max()),
        "minimum_pixel_value": float(minimum_pixels.min()),
        "maximum_pixel_value": float(maximum_pixels.max()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.endpoint_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
