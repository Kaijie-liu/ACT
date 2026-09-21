"""Freeze two fixed author-component controls before any real bound queries."""
import argparse
import json
from pathlib import Path

from recent_moe_deployment import sha256
from metamoe_component_control import validate_freeze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    act = Path(__file__).resolve().parents[1]
    base = Path("/data1/Kane/MOE")
    repos = base / "baselines/recent_moe_20260921"
    data = base / "baseline_data/metamoe_20260921"
    env = base / "baseline_runs/metamoe_install_20260921/new_environment_frozen.json"
    sources = ["scripts/metamoe_component_control.py", "scripts/recent_moe_deployment.py",
               "scripts/recent_moe_env_inventory.py", "scripts/freeze_metamoe_component_control.py",
               "configs/recent_moe/metamoe_cpu_requirements_20260921.txt"]
    config = {
        "schema": 1, "protocol": "metamoe_author_component_cpu_control_r1",
        "author_repo": str(repos / "metamoe"),
        "author_commit": "6aed3606e4b226e18e1c9d249485d99bb453f488",
        "backend_repo": str(repos / "metamoe_abcrown"),
        "backend_commit": "58bb93f4886eea7cd1a3dfeb303695f20f61473b",
        "lirpa_commit": "28da3d0148ce320142f4a50b485ac71bca7acc3b",
        "onnx2pytorch_commit": "325959ed128200459a9634668149a722a6a74797",
        "python": str(base / "envs/metamoe-author-cpu-20260921/bin/python"),
        "data_root": str(data), "data_manifest_sha256": sha256(data / "manifest.json"),
        "environment_inventory": str(env), "environment_inventory_sha256": sha256(env),
        "output_root": str(base / "baseline_runs/metamoe_component_20260921_r1"),
        "seed": 100, "seed_source": "unchanged author backend default; paper states 42",
        "epsilon_numerator": 2, "epsilon_denominator": 255,
        "epsilon_space": "author per-dataset normalized tensor, clipped to [-10,10]",
        "solver_seconds": 300, "outer_seconds": 360,
        "outer_includes": "startup, identities/data/model load, export, probe, spec, backend and terminal",
        "installation_cost_separate": True, "conformance_tolerance": 0.0001,
        "execution_files": {name: sha256(act / name) for name in sources},
        "requests": [
            {"id": "cifar10_rt_index0", "dataset": "CIFAR10", "dataset_index": 0,
             "checkpoint": "paper/artifacts/E_0_CNN_AT/cifar10_ultra_verifiable_cnn_best_RT_eps0.031.pth",
             "checkpoint_sha256": "e256e3a423cc0112a29e5e802e35bacfcf70ef90d3c4bba6c890b7b6eadfa9ae"},
            {"id": "mnist_rt_index0", "dataset": "MNIST", "dataset_index": 0,
             "checkpoint": "paper/artifacts/E_1_CNN_AT/mnist_ultra_verifiable_cnn_best_RT_eps0.031.pth",
             "checkpoint_sha256": "e4d32a028646cbb5efbb959ffe2ca97f194cd540d685c1d91a3d85437097432d"}
        ],
        "scope": "author component deployment on fixed raw-order inputs, not paper table/full MoE reproduction",
        "stop_rule": "ERROR stops remaining requests; timeout retained; no replacement/retry",
        "guarantee": "native backend report for exported decimal problem; no source-complete/native-float claim"
    }
    validate_freeze(config)
    if Path(config["output_root"]).exists():
        raise ValueError("prospective result directory already exists")
    with a.output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "FROZEN_NOT_EXECUTED", "sha256": sha256(a.output)}))


if __name__ == "__main__":
    main()
