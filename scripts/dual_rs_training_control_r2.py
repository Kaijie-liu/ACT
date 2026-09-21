"""Versioned log-domain KL control; same inputs, schedule and total R1 budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import dual_rs_training_control as r1
from recent_moe_deployment import sha256, supervise


def enable_compatibility():
    original = r1.build
    def build(cfg):
        if cfg["consistency_implementation"] != "log_domain_default_v1":
            raise ValueError("wrong compatibility identity")
        from dual_rs_log_domain_consistency import consistency_loss
        result = original(cfg)
        result[0].consistency_loss = consistency_loss  # exactly the training callsite
        return result
    r1.build = build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--phase", choices=["reference", "resume", "manager"])
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = Path(cfg["output_root"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
    if sha256(cfg["parent_control_config"]) != cfg["parent_control_config_sha256"]:
        raise ValueError("parent R1 binding changed")
    if a.phase in ["reference", "resume"]:
        enable_compatibility()
        r1.run_phase(cfg, a.config, a.phase)
        return
    if a.phase == "manager":
        stages = []
        for phase in ["reference", "resume", "audit"]:
            r1.write_json(root / f"{phase}_started.json", {"phase": phase, "monotonic": time.monotonic()})
            cmd = ([sys.executable, str(Path(__file__).resolve()), "--config", str(a.config.resolve()), "--phase", phase]
                   if phase != "audit" else [sys.executable, str(Path(__file__).with_name("audit_dual_rs_training_control.py").resolve()),
                                             "--config", str(a.config.resolve()), "--output", str(root / "audit.json")])
            start = time.monotonic()
            with (root / f"{phase}.stdout").open("x") as out, (root / f"{phase}.stderr").open("x") as err:
                code = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], stdout=out, stderr=err).returncode
            stages.append({"phase": phase, "returncode": code, "wall_seconds": time.monotonic() - start})
            r1.write_json(root / f"{phase}_finished.json", stages[-1])
            if code:
                r1.write_json(root / "terminal.json", {"status": "ERROR", "stages": stages})
                raise SystemExit(code)
        r1.write_json(root / "terminal.json", {"status": "CONTROL_PASS", "stages": stages})
        return
    receipt = supervise([cfg["python"], str(Path(__file__).resolve()), "--config", str(a.config.resolve()), "--phase", "manager"],
                        str(Path(__file__).resolve().parents[1]), root, cfg["total_control_seconds"],
                        "DUAL_RS_LOG_DOMAIN_COMPAT_STEP_RESUME", cfg["author_repo"], cpu_only=False)
    terminal = r1.summarize_terminal(root, receipt)
    print(json.dumps(terminal, indent=2))
    raise SystemExit(0 if terminal["status"] == "CONTROL_PASS" else 1)


if __name__ == "__main__":
    main()
