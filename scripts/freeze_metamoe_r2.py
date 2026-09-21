"""Freeze the minimal upstream AvgPool compatibility revision; no new queries."""
import argparse
import json
from pathlib import Path
import subprocess

from metamoe_component_control import validate_freeze
from recent_moe_deployment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.parent.read_text())
    cfg["parent_config_sha256"] = sha256(a.parent)
    cfg["protocol"] = "metamoe_author_component_cpu_control_r2"
    cfg["output_root"] = "/data1/Kane/MOE/baseline_runs/metamoe_component_20260921_r2"
    cfg["python"] = "/data1/Kane/MOE/envs/metamoe-author-cpu-20260921-r2/bin/python"
    cfg["environment_inventory"] = "/data1/Kane/MOE/baseline_runs/metamoe_install_20260921_r2/new_environment_frozen.json"
    cfg["environment_inventory_sha256"] = sha256(cfg["environment_inventory"])
    cfg["onnx2pytorch_commit"] = "8447c42c3192dad383e5598edc74dddac5706ee2"
    direct = json.loads(subprocess.check_output([cfg["python"], "-c",
        "import importlib.metadata as m,json;print(json.dumps({n:json.loads(m.distribution(n).read_text('direct_url.json')) for n in ['onnx2pytorch','auto-LiRPA']}))"], text=True))
    if direct["onnx2pytorch"]["vcs_info"]["commit_id"] != cfg["onnx2pytorch_commit"]:
        raise ValueError("wrong conversion commit")
    if direct["auto-LiRPA"]["vcs_info"]["commit_id"] != cfg["lirpa_commit"]:
        raise ValueError("wrong LiRPA commit")
    old = json.loads(Path(json.loads(a.parent.read_text())["environment_inventory"]).read_text())
    new = json.loads(Path(cfg["environment_inventory"]).read_text())
    if next(iter(old.values()))["environment"]["packages"] != next(iter(new.values()))["environment"]["packages"]:
        raise ValueError("unintended package version change")
    cfg["source_package_direct_urls"] = direct
    cfg["compatibility_change"] = "Upstream immediate successor adds bool count_include_pad mapping; no other source diff except whitespace. R1 retained."
    proof = Path("/data1/Kane/MOE/baseline_runs/metamoe_install_20260921_r2/pool_compatibility/receipt.json")
    receipt = json.loads(proof.read_text())
    if receipt["status"] != "COMPLETED":
        raise ValueError("compatibility probes did not pass")
    for name in ["stdout", "stderr"]:
        if sha256(proof.parent / f"{name}.txt") != receipt[f"{name}_sha256"]:
            raise ValueError("compatibility log changed")
    controls = json.loads((proof.parent / "stdout.txt").read_text())
    if len(controls["pool_controls"]) != 4 or not all(
            row["exact_probe_match"] for row in controls["pool_controls"]):
        raise ValueError("incomplete pool controls")
    cfg["compatibility_controls"] = controls
    cfg["compatibility_receipt_sha256"] = sha256(proof)
    validate_freeze(cfg)
    if Path(cfg["output_root"]).exists():
        raise ValueError("result directory exists")
    with a.output.open("x") as stream:
        json.dump(cfg, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "FROZEN_NOT_EXECUTED", "sha256": sha256(a.output)}))


if __name__ == "__main__":
    main()
