"""Archive the successful R2 and its independent saved-only re-audit."""
import argparse
import json
from pathlib import Path

from recent_moe_deployment import sha256


def collect():
    config = Path("configs/recent_moe/dual_rs_training_control_r2.json")
    cfg = json.loads(config.read_text())
    for path, digest in cfg["execution_files"].items():
        if sha256(path) != digest:
            raise ValueError(f"execution identity changed: {path}")
    root = Path(cfg["output_root"])
    receipt = json.loads((root / "receipt.json").read_text())
    terminal = json.loads((root / "outer_terminal.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    reread_path = Path("/data1/Kane/MOE/baseline_runs/dual_rs_training_control_r2_saved_reaudit.json")
    reread = json.loads(reread_path.read_text())
    if {k: v for k, v in audit.items() if k != "audit_seconds"} != {
            k: v for k, v in reread.items() if k != "audit_seconds"}:
        raise ValueError("saved-only audit differs")
    if audit["audit"] != "PASS" or audit["config_sha256"] != sha256(config):
        raise ValueError("audit binding/status")
    if (terminal["status"] != "CONTROL_PASS" or receipt["status"] != "COMPLETED"
            or not receipt["source_unchanged"]
            or receipt["source_before"]["head"] != cfg["author_commit"]):
        raise ValueError("supervision/source gate failed")
    for phase in ("reference", "resume"):
        info = json.loads((root / f"{phase}.json").read_text())
        if sha256(root / f"{phase}_after_step2.pt") != info["file_sha256"]:
            raise ValueError("checkpoint changed after audit")
    for stream in ("stdout", "stderr"):
        if sha256(root / f"{stream}.txt") != receipt[f"{stream}_sha256"]:
            raise ValueError("log changed")
    return {
        "schema": 1, "result": "CONTROL_PASS", "execution_commit": "721dd80d3cd251f9a815df4e0ba5c107060c3ec3",
        "config": str(config), "config_sha256": sha256(config),
        "scope": "log-domain numerical compatibility variant, two real updates and fresh-process exact restore; not trained accuracy or certification",
        "native_r1_remains": "FAILED_NONFINITE_GRADIENT_BEFORE_SECOND_UPDATE",
        "audit": audit, "independent_saved_only_audit": reread,
        "outer_terminal": terminal,
        "cost_seconds": {k: receipt[k] for k in (
            "execution_including_preflight_seconds", "total_with_postflight_seconds")},
        "all_record_hashes": {str(p): sha256(p) for p in sorted(root.iterdir()) if p.is_file()},
        "saved_only_reaudit_sha256": sha256(reread_path),
        "long_training_started": False, "new_certification_started": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    value = collect()
    if args.check:
        if value != json.loads(args.output.read_text()):
            raise ValueError("archive differs")
    else:
        with args.output.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
    print("R2 saved-record archive PASS")
