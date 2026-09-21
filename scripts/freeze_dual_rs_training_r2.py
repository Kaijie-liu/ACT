"""Freeze numerical-compatibility R2 without changing the failed R1 artifacts."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from dual_rs_training_control import validate_config


def main():
    path = Path("configs/recent_moe/dual_rs_training_control_r1.json")
    cfg = json.loads(path.read_text())
    validate_config(cfg)
    cfg.update(protocol="dual_rs_log_domain_step_resume_control_r2",
               consistency_implementation="log_domain_default_v1",
               parent_control_config=str(path), parent_control_config_sha256=sha256(path),
               output_root="/data1/Kane/MOE/baseline_runs/dual_rs_training_control_20260921_r2",
               scope="explicit numerical compatibility variant; native math/hyperparameters, not byte-identical author computation")
    for f in ["scripts/dual_rs_log_domain_consistency.py", "scripts/dual_rs_training_control_r2.py",
              "scripts/freeze_dual_rs_training_r2.py"]:
        cfg["execution_files"][f] = sha256(f)
    output = Path("configs/recent_moe/dual_rs_training_control_r2.json")
    if Path(cfg["output_root"]).exists():
        raise ValueError("R2 directory already exists")
    with output.open("x") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(sha256(output))


if __name__ == "__main__":
    main()
