"""Prepare fixed identities/selection BEFORE the real step/resume control."""
import argparse
import json
from pathlib import Path

from recent_moe_deployment import git_identity, sha256
from recent_moe_env_inventory import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    base = Path("/data1/Kane/MOE")
    repo = base / "baselines/recent_moe_20260921/dual_rs"
    source = git_identity(repo)
    if source["head"] != "6f83aeb7f47466b1dee6295baad8d59a8c94eceb" or source["status"]:
        raise ValueError("author identity changed")
    labels = repo / "data/sigma_label/map10e4"
    table = labels / "0.250_0.500_1.000_train_100map.npy"
    data = np.load(table, allow_pickle=False)
    if data.shape != (50000, 5) or not np.isfinite(data).all():
        raise ValueError("unexpected author training label table")
    eligible = np.flatnonzero(data[:, -2] != 0).tolist()
    env_path = base / "baseline_runs/dual_rs_install_20260921/environment_frozen.json"
    env = json.loads(env_path.read_text())
    if inventory(list(env)) != env:
        raise ValueError("environment changed")
    data_root = base / "baseline_data/metamoe_20260921"
    manifest = json.loads((data_root / "manifest.json").read_text())
    inputs = {str(data_root / k): v["sha256"] for k, v in manifest["files"].items()
              if k.startswith("cifar-10-batches-py/")}
    for path in [table, labels / "0.250_0.500_1.000_test.npy",
                 labels / "0.250_0.500_1.000_train_100map_class_weights.npy"]:
        inputs[str(path)] = sha256(path)
    weights = base / "baseline_weights/dual_rs_20260921"
    for row in json.loads((weights / "manifest.json").read_text())["files"]:
        inputs[str(weights / row["filename"])] = row["sha256"]
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError("input bytes changed")
    files = ["scripts/dual_rs_training_state.py", "scripts/dual_rs_training_control.py",
             "scripts/audit_dual_rs_training_control.py", "scripts/recent_moe_deployment.py",
             "scripts/recent_moe_env_inventory.py", "scripts/freeze_dual_rs_training_control.py"]
    cfg = {"schema": 1, "protocol": "dual_rs_native_step_resume_control_r1", "author_repo": str(repo),
           "author_commit": source["head"], "author_files": source["tracked_sha256"],
           "python": next(iter(env)), "environment_inventory": str(env_path),
           "environment_inventory_sha256": sha256(env_path), "input_files": inputs,
           "data_root": str(data_root), "label_table": str(table),
           "class_weights": str(labels / "0.250_0.500_1.000_train_100map_class_weights.npy"),
           "diffusion": str(weights / "cifar10_uncond_50M_500K.pt"),
           "output_root": str(base / "baseline_runs/dual_rs_training_control_20260921_r1"),
           "execution_files": {s: sha256(s) for s in files},
           "seed": 1, "loader_batch": 256, "workers": 0, "num_noise_vec": 2,
           "selected_train_indices": eligible[:256], "eligible_training_count": len(eligible),
           "selection": "first 256 raw-order rows passing author's max-radius!=0 training-label filter; no new certification",
           "optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0.01},
           "scheduler": {"milestones": [30, 60, 1000], "gamma": 0.5, "unit": "completed epoch"},
           "noise_sd": 1.0, "sigma_cand": [0.25, 0.5, 1.0], "loss_cl": "softce",
           "loss_con": True, "lbd": 40.0, "eta": 0.5, "adp_con_wght": "max",
           "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
           "total_control_seconds": 300, "minimum_free_gpu_gib": 32,
           "gpu_memory_fraction": 0.40, "restore_gate": "exact tensor/logical equality, no tolerance",
           "stop_rule": "error/deadline stops remaining stages; no retry or threshold relaxation",
           "scope": "one native optimizer update plus next-update fresh-process replay, not trained estimator"}
    if len(cfg["selected_train_indices"]) != 256 or Path(cfg["output_root"]).exists():
        raise ValueError("invalid selection or existing run")
    with a.output.open("x") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": "FROZEN_NOT_EXECUTED", "sha256": sha256(a.output), "eligible": len(eligible)}))


if __name__ == "__main__":
    main()
