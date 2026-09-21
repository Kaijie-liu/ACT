"""Rehash R1/R2 records and setup; no verification or export calls."""
import argparse
import json
from pathlib import Path
import subprocess

from metamoe_component_control import validate_freeze
from recent_moe_deployment import sha256
from recent_moe_env_inventory import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    base = Path("/data1/Kane/MOE/baseline_runs")
    old_install = base / "metamoe_install_20260921"
    old = json.loads((old_install / "existing_environments_before.json").read_text())
    r1_env = json.loads((old_install / "new_environment_frozen.json").read_text())
    if inventory(list(old)) != old or inventory(list(r1_env)) != r1_env:
        raise ValueError("previous environment drift")
    cfg_file = Path("configs/recent_moe/metamoe_component_control_r2.json")
    cfg = json.loads(cfg_file.read_text())
    validate_freeze(cfg)
    installs = {}
    for path in sorted((base / "metamoe_install_20260921_r2").glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        for name in ["stdout", "stderr"]:
            if sha256(path.parent / f"{name}.txt") != receipt[f"{name}_sha256"]:
                raise ValueError("setup log changed")
        installs[path.parent.name] = {"receipt_sha256": sha256(path), "receipt": receipt}
    runs = {}
    for revision in ["r1", "r2"]:
        config = Path(f"configs/recent_moe/metamoe_component_control_{revision}.json")
        review = Path(f"docs/metamoe_component_control_review_20260921_{revision}.json")
        root = base / f"metamoe_component_20260921_{revision}"
        files = {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(root.rglob("*")) if p.is_file()}
        saved = json.loads(review.read_text())
        if saved["audit"] != "PASS" or saved["config_sha256"] != sha256(config):
            raise ValueError("unbound review")
        runs[revision] = {"config_sha256": sha256(config), "review_sha256": sha256(review),
                          "states": [{"id": r["id"], "status": r["status"]} for r in saved["requests"]],
                          "raw_directory": str(root), "files": files,
                          "total_file_bytes": sum(v["bytes"] for v in files.values())}
    direct = json.loads(subprocess.check_output([cfg["python"], "-c",
        "import importlib.metadata as m,json;print(json.dumps({n:json.loads(m.distribution(n).read_text('direct_url.json')) for n in ['onnx2pytorch','auto-LiRPA']}))"], text=True))
    if direct != cfg["source_package_direct_urls"]:
        raise ValueError("source package drift")
    diag = {}
    for name in ["pool_compatibility", "mnist_export_diagnostic", "mnist_zero_diagnostic"]:
        if installs[name]["receipt"]["status"] != "COMPLETED":
            raise ValueError("diagnostic incomplete")
        diag[name] = json.loads((base / "metamoe_install_20260921_r2" / name / "stdout.txt").read_text())
    result = {"schema": 1, "stage": "author_component_deployment_with_retained_failures",
              "old_five_environments_and_r1_inventory_unchanged": True,
              "r2_environment": json.loads(Path(cfg["environment_inventory"]).read_text()),
              "source_package_direct_urls": direct, "install_and_diagnostic_receipts": installs,
              "diagnostics": diag, "runs": runs,
              "full_moe_comparison_or_paper_table_reproduction": False,
              "scope": "author ONNX/VNNLIB backend control; no source-complete or native-float certificate"}
    if a.check:
        if result != json.loads(a.output.read_text()):
            raise ValueError("archive differs from reread")
        print("PASS: saved-only archive reread matches")
    else:
        with a.output.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps({"archived_revisions": len(runs), "setup_diagnostic_attempts": len(installs)}))


if __name__ == "__main__":
    main()
