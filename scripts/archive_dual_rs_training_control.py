"""Saved-only failure audit. No checkpoint model construction, GPU or training."""
import argparse
import json
from pathlib import Path
import time

from recent_moe_deployment import sha256


def collect():
    import torch
    import torch.nn.functional as F
    from dual_rs_training_state import digest, load_snapshot
    torch.set_num_threads(2)
    cp = Path("configs/recent_moe/dual_rs_training_control_r1.json")
    dp = Path("configs/recent_moe/dual_rs_failed_step_diagnostic_r1.json")
    cfg, dcfg = json.loads(cp.read_text()), json.loads(dp.read_text())
    if sha256(cp) != dcfg["parent_config_sha256"] or sha256("scripts/diagnose_dual_rs_training_failure.py") != dcfg["script_sha256"]:
        raise ValueError("frozen diagnostic binding changed")
    for path, h in cfg["execution_files"].items():
        if sha256(path) != h:
            raise ValueError("frozen control source changed")
    root, dr = Path(cfg["output_root"]), Path(dcfg["output_root"])
    receipts = {}
    files = {str(cp): sha256(cp), str(dp): sha256(dp)}
    for name, folder in [("control", root), ("diagnostic", dr)]:
        r = json.loads((folder / "receipt.json").read_text())
        if not r["source_unchanged"] or r["source_before"]["head"] != cfg["author_commit"]:
            raise ValueError("source identity mismatch")
        for stream in ["stdout", "stderr"]:
            if sha256(folder / f"{stream}.txt") != r[f"{stream}_sha256"]:
                raise ValueError("log hash mismatch")
        if r["total_with_postflight_seconds"] < r["execution_including_preflight_seconds"]:
            raise ValueError("invalid cost accounting")
        receipts[name] = {k: v for k, v in r.items() if k != "source_before"}
        files.update({str(p): sha256(p) for p in folder.iterdir() if p.is_file()})
    terminal = json.loads((root / "outer_terminal.json").read_text())
    if (receipts["control"]["status"] != "ERROR" or terminal["status"] != "ERROR" or
            [r["status"] for r in terminal["stages"]] != ["ERROR", "NOT_STARTED", "NOT_STARTED"] or
            any((root / name).exists() for name in ["resume.json", "audit.json", "reference_after_step2.pt"])):
        raise ValueError("failure/partial state not preserved")
    first = json.loads((root / "after_step1.json").read_text())
    if first["file_sha256"] != dcfg["checkpoint_sha256"] or first["binding"]["config_sha256"] != sha256(cp):
        raise ValueError("first step binding mismatch")
    state = load_snapshot(root / "after_step1.pt", first["file_sha256"], first["binding"])
    if digest(state) != first["logical_sha256"] or state["cursor"]["global_step"] != 1:
        raise ValueError("saved first state differs")
    def tensors(v):
        if isinstance(v, torch.Tensor):
            yield v
        elif isinstance(v, dict):
            for x in v.values():
                yield from tensors(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from tensors(x)
    if not all(bool(torch.isfinite(t).all()) for t in tensors(state)):
        raise ValueError("saved first state has nonfinite values")
    if any(float(v["step"]) != 1 for v in state["optimizer"]["state"].values()):
        raise ValueError("first snapshot is not one update")
    info = json.loads((dr / "diagnostic.json").read_text())
    if (receipts["diagnostic"]["status"] != "COMPLETED" or not info["no_optimizer_update"] or
            info["diagnostic_config_sha256"] != sha256(dp) or
            info["parent_checkpoint_sha256"] != first["file_sha256"] or
            sha256(dr / "saved_logits.pt") != info["saved_logits_sha256"]):
        raise ValueError("diagnostic record mismatch")
    data = torch.load(dr / "saved_logits.pt", map_location="cpu", weights_only=True)
    derivatives = {}
    # Different implementation: isolate KL and entropy on saved logits directly.
    # This checks the failure mechanism, not a replacement training loss.
    for precision in [torch.float32, torch.float64]:
        for term in ["kl", "entropy"]:
            x = data["logits"].to(precision).detach().requires_grad_(True)
            parts = torch.chunk(x, 2, 0)
            avg = sum(F.softmax(p, 1) for p in parts) / 2
            loss = (sum(F.kl_div(F.log_softmax(p, 1), avg, reduction="none").sum(1) for p in parts) / 2
                    if term == "kl" else -(avg * avg.clamp(min=1e-20).log()).sum(1))
            loss = loss.mean()
            grad, = torch.autograd.grad(loss, x)
            derivatives[f"{term}_{str(precision)}"] = {
                "value_finite": bool(torch.isfinite(loss)),
                "gradient_nonfinite": int((~torch.isfinite(grad)).sum()),
                "average_probability_zeros": int((avg == 0).sum())}
    if not (derivatives["kl_torch.float32"]["gradient_nonfinite"] > 0 and
            derivatives["kl_torch.float32"]["average_probability_zeros"] > 0 and
            derivatives["entropy_torch.float32"]["gradient_nonfinite"] == 0 and
            derivatives["kl_torch.float64"]["gradient_nonfinite"] == 0):
        raise ValueError("saved-only derivative diagnosis not confirmed")
    return {"schema": 1, "archive_audit": "PASS_FAILURE_PRESERVED", "training_resume_control": "FAILED",
            "long_training_recipe_accepted": False, "fresh_certification_executed": False,
            "source_commit": cfg["author_commit"], "frozen_execution_commit": "8fa8d4e34",
            "diagnostic_execution_commit": "1c064a783",
            "source_and_record_hashes": files, "receipts": receipts, "outer_terminal": terminal,
            "first_step": first, "first_snapshot_all_tensors_finite": True,
            "first_snapshot_optimizer_states": len(state["optimizer"]["state"]),
            "diagnostic": info, "independent_saved_logit_cpu_derivatives": derivatives,
            "scope": "fixed-prefix control under isolated compatibility environment; not failure of all author training runs",
            "next_gate": "separately named numerically stable KL evaluation control; no silent dtype/LR/loss change"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    start = time.monotonic()
    value = collect()
    if a.check:
        if value != json.loads(a.output.read_text()):
            raise ValueError("archive saved-only reread differs")
    else:
        with a.output.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps({"archive_audit": value["archive_audit"], "training_resume_control": "FAILED",
                      "saved_only_audit_seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
