"""Archive isolated Dual RS staging, preserving grades and any failures."""
import argparse
import hashlib
import json
from pathlib import Path

from recent_moe_deployment import git_identity, sha256
from recent_moe_env_inventory import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    base = Path("/data1/Kane/MOE")
    runs = base / "baseline_runs"
    current = runs / "dual_rs_install_20260921"
    inventories = [runs / "metamoe_install_20260921/existing_environments_before.json",
                   runs / "metamoe_install_20260921/new_environment_frozen.json",
                   runs / "metamoe_install_20260921_r2/new_environment_frozen.json",
                   current / "environment_frozen.json"]
    for path in inventories:
        expected = json.loads(path.read_text())
        if inventory(list(expected)) != expected:
            raise ValueError("environment drift: " + str(path))
    attempts = {}
    for path in sorted(current.glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        source = receipt.pop("source_before", None)
        receipt["source_head"] = source["head"] if source else None
        for name in ["stdout", "stderr"]:
            if sha256(path.parent / f"{name}.txt") != receipt[f"{name}_sha256"]:
                raise ValueError("setup log changed")
        if not receipt["source_unchanged"]:
            raise ValueError("author source changed")
        attempts[path.parent.name] = {"receipt_sha256": sha256(path), "receipt": receipt}
    weights_root = base / "baseline_weights/dual_rs_20260921"
    manifest = json.loads((weights_root / "manifest.json").read_text())
    for row in manifest["files"]:
        if sha256(weights_root / row["filename"]) != row["sha256"]:
            raise ValueError("weight bytes changed")
    source = git_identity(base / "baselines/recent_moe_20260921/dual_rs")
    if source["status"]:
        raise ValueError("author checkout dirty")
    files = source.pop("tracked_sha256")
    source["tracked_manifest_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    probe = json.loads((current / "native_model_probe/stdout.txt").read_text()) if (
        attempts.get("native_model_probe", {}).get("receipt", {}).get("status") == "COMPLETED") else None
    value = {"schema": 1, "stage": "isolated_author_dependency_and_pretrained_component_deployment",
             "author": source, "environment": json.loads(inventories[-1].read_text()),
             "previous_environment_inventories_unchanged": True,
             "inventory_hashes": {str(p): sha256(p) for p in inventories},
             "attempts": attempts, "public_model_manifest": manifest, "native_model_probe": probe,
             "trained_sigma_estimator_checkpoint_available": False,
             "training_executed": False, "fresh_dual_rs_certification_executed": False,
             "scope": "dependency/native component readiness; not original paper accuracy or certification table"}
    if a.check:
        if value != json.loads(a.output.read_text()):
            raise ValueError("saved reread mismatch")
        print("PASS: archive rehash and old-environment inventories")
    else:
        with a.output.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"attempts": len(attempts), "native_model_probe_recorded": probe is not None}))


if __name__ == "__main__":
    main()
