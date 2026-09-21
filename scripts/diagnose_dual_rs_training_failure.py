"""One bounded replay of the failed second step; NEVER apply another update."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from recent_moe_deployment import sha256, supervise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if sha256(__file__) != cfg["script_sha256"] or sha256(cfg["parent_config"]) != cfg["parent_config_sha256"]:
        raise ValueError("diagnostic identity mismatch")
    parent = json.loads(Path(cfg["parent_config"]).read_text())
    root = Path(cfg["output_root"])
    if not a.worker:
        r = supervise([parent["python"], str(Path(__file__).resolve()), "--config", str(a.config.resolve()), "--worker"],
                      str(Path(__file__).resolve().parents[1]), root, cfg["seconds"],
                      "FAILED_STEP_DIAGNOSIS_NO_UPDATE", parent["author_repo"], cpu_only=False)
        print(json.dumps({k: v for k, v in r.items() if k != "source_before"}, indent=2))
        raise SystemExit(0 if r["status"] == "COMPLETED" else 1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = parent["cublas_workspace_config"]
    import torch
    from dual_rs_training_control import build, native_step, validate_config, write_json
    from dual_rs_training_state import digest, load_snapshot, restore
    start = time.monotonic()
    validate_config(parent)
    native, model, opt, sched, denoiser, t, weights = build(parent)
    record = json.loads((Path(parent["output_root"]) / "after_step1.json").read_text())
    if record["file_sha256"] != cfg["checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint binding mismatch")
    stored = load_snapshot(Path(parent["output_root"]) / "after_step1.pt", cfg["checkpoint_sha256"], record["binding"])
    restore(stored, model, opt, sched, record["binding"])
    before = digest(opt.state_dict())
    parameter_before = digest(dict(model.named_parameters()))
    # A pre-hook installed before native_step's checker precludes ANY update,
    # even if the failure unexpectedly disappears. It does not alter backward.
    class DiagnosticStop(Exception):
        pass
    def stop(_opt, _args, _kwargs):
        raise DiagnosticStop()
    handle = opt.register_step_pre_hook(stop)
    try:
        native_step(native, model, opt, denoiser, t, weights, stored["batch"], 1)
        raise ValueError("missing diagnostic stop")
    except DiagnosticStop as exc:
        tb = exc.__traceback__
        while tb and tb.tb_frame.f_code is not native.train.__code__:
            tb = tb.tb_next
        if tb is None:
            raise ValueError("native frame missing")
        v = tb.tb_frame.f_locals
        logits = v["outputs"].detach().clone()
        radii = v["radii"].detach().clone()
        classification_weights = v["weights"].detach().clone()
        consistency_weights = (v["con_loss_weights"] * v["weights"][:v["batch_size"]]).detach().clone()
        chunks = torch.chunk(logits, 2, 0)
        avg = sum(torch.softmax(c, 1) for c in chunks) / 2
        result = {"native_loss_finite": bool(torch.isfinite(v["loss"])),
                  "native_loss": float(v["loss"].detach()) if torch.isfinite(v["loss"]) else None,
                  "logits_finite": bool(torch.isfinite(logits).all()),
                  "logit_min": float(logits.min()), "logit_max": float(logits.max()),
                  "average_softmax_zeros": int((avg == 0).sum()), "average_softmax_elements": avg.numel(),
                  "parameter_grad_nonfinite_elements": sum(int((~torch.isfinite(p.grad)).sum())
                                                           for p in model.parameters() if p.grad is not None),
                  "parameter_grad_elements": sum(p.grad.numel() for p in model.parameters() if p.grad is not None)}
        del v, tb
    finally:
        handle.remove()
    if digest(opt.state_dict()) != before or digest(dict(model.named_parameters())) != parameter_before:
        raise ValueError("diagnosis applied an optimizer update")
    import consistency
    comparisons = {}
    # Same frozen logits only: no model/optimizer re-execution, no candidate fix.
    for name in ["softce_float32", "consistency_float32", "consistency_float64"]:
        x = logits.to(torch.float64 if name.endswith("64") else torch.float32).detach().requires_grad_(True)
        loss = ((native.softXEnt(x, radii, "soft_max") * classification_weights).mean()
                if name.startswith("softce") else
                (consistency.consistency_loss(torch.chunk(x, 2, 0), 40., .5)[0] * consistency_weights).mean())
        grad, = torch.autograd.grad(loss, x)
        probabilities = sum(torch.softmax(c, 1) for c in torch.chunk(x, 2, 0)) / 2
        comparisons[name] = {"loss_finite": bool(torch.isfinite(loss)),
                             "loss": float(loss.detach()) if torch.isfinite(loss) else None,
                             "gradient_nonfinite_elements": int((~torch.isfinite(grad)).sum()),
                             "probability_zero_elements": int((probabilities == 0).sum())}
    artifact = root / "saved_logits.pt"
    with artifact.open("xb") as f:
        torch.save({"logits": logits.cpu(), "radii": radii.cpu(),
                    "classification_weights": classification_weights.cpu(),
                    "consistency_weights": consistency_weights.cpu()}, f)
    result.update({"status": "DIAGNOSTIC_COMPLETED", "diagnostic_config_sha256": sha256(a.config),
                   "parent_checkpoint_sha256": cfg["checkpoint_sha256"], "no_optimizer_update": True,
                   "saved_logits_sha256": sha256(artifact), "loss_space_comparison": comparisons,
                   "worker_seconds": time.monotonic() - start,
                   "scope": "fixed failed step and saved logits only; not a training repair or long-run conclusion"})
    write_json(root / "diagnostic.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
