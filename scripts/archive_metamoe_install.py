"""Compact installation provenance, without raw data/checkpoints or external code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from recent_moe_deployment import git_identity, sha256
from recent_moe_env_inventory import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base = Path("/data1/Kane/MOE")
    run = base / "baseline_runs/metamoe_install_20260921"
    before = json.loads((run / "existing_environments_before.json").read_text())
    if inventory(list(before)) != before:
        raise ValueError("old environments changed")
    frozen = json.loads((run / "new_environment_frozen.json").read_text())
    if inventory(list(frozen)) != frozen:
        raise ValueError("new environment drift")
    python = next(iter(frozen))
    direct = json.loads(subprocess.check_output([python, "-c",
        "import importlib.metadata as m,json; print(json.dumps({n:json.loads(m.distribution(n).read_text('direct_url.json')) for n in ['auto-LiRPA','onnx2pytorch']}))"], text=True))
    repos = {}
    for name in ["metamoe", "metamoe_abcrown", "metamoe_onnx2pytorch"]:
        identity = git_identity(base / "baselines/recent_moe_20260921" / name)
        if identity["status"]:
            raise ValueError("author sources changed")
        files = identity.pop("tracked_sha256")
        identity.update(tracked_files=len(files), tracked_manifest_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest())
        repos[name] = identity
    attempts = {}
    for path in sorted(run.glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        source = receipt.pop("source_before")
        receipt["source_head"] = source["head"] if source else None
        for stream in ["stdout", "stderr"]:
            target = path.parent / f"{stream}.txt"
            if sha256(target) != receipt[f"{stream}_sha256"]:
                raise ValueError("deployment log changed")
        attempts[path.parent.name] = {"receipt": receipt, "path": str(path), "sha256": sha256(path)}
    value = {"schema": 1, "stage": "environment_and_prospective_component_control",
             "user_authorized_dependencies_public_weights_and_data": True,
             "old_environment_package_and_binary_inventories_unchanged": True,
             "old_inventory_sha256": sha256(run / "existing_environments_before.json"),
             "new_environment": frozen, "source_package_direct_urls": direct,
             "repositories": repos, "attempts": attempts,
             "public_data": json.loads((base / "baseline_data/metamoe_20260921/manifest.json").read_text()),
             "component_queries_executed_at_this_preparation_stage": 0,
             "scope": "installation and bounded probes; not original-paper or whole-MoE reproduction"}
    with a.output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"old_environments_unchanged": True, "attempts": len(attempts)}))


if __name__ == "__main__":
    main()
