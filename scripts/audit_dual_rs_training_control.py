"""Saved-state replay audit: no dataset, author model, training or GPU required."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def canonical(v):
    """Independent nested tensor equality, not trusting reported hashes."""
    import torch
    if isinstance(v, torch.Tensor):
        return ("tensor", str(v.dtype), tuple(v.shape), v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if isinstance(v, dict):
        return ("dict", tuple((canonical(k), canonical(v[k])) for k in sorted(v, key=lambda k: (type(k).__name__, str(k)))))
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(canonical(x) for x in v))
    return (type(v).__name__, repr(v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    import torch
    cfg = json.loads(a.config.read_text())
    cfg_hash = hashlib.sha256(a.config.read_bytes()).hexdigest()
    root = Path(cfg["output_root"])
    first_info = json.loads((root / "after_step1.json").read_text())
    first_path = root / "after_step1.pt"
    if hashlib.sha256(first_path.read_bytes()).hexdigest() != first_info["file_sha256"]:
        raise ValueError("first checkpoint hash mismatch")
    first_state = torch.load(first_path, map_location="cpu", weights_only=True)
    values = {}
    for phase in ["reference", "resume"]:
        info = json.loads((root / f"{phase}.json").read_text())
        path = root / f"{phase}_after_step2.pt"
        if info["status"] != "COMPLETED" or info["config_sha256"] != cfg_hash or not info["denoiser_unchanged"]:
            raise ValueError("phase status/binding mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["file_sha256"]:
            raise ValueError("phase checkpoint hash mismatch")
        values[phase] = torch.load(path, map_location="cpu", weights_only=True)
    a_state, b_state = values["reference"], values["resume"]
    required = {"schema", "binding", "model", "optimizer", "scheduler", "rng", "cursor", "batch"}
    if set(a_state) != required or set(b_state) != required:
        raise ValueError("incomplete training state")
    equality = {k: canonical(a_state[k]) == canonical(b_state[k]) for k in required}
    if not all(equality.values()) or a_state["cursor"]["global_step"] != 2:
        raise ValueError("fresh-process continuation differs")
    if a_state["binding"]["config_sha256"] != cfg_hash:
        raise ValueError("wrong scope")
    if set(first_state) != required or first_state["binding"] != a_state["binding"]:
        raise ValueError("first checkpoint binding differs")
    for state, step in [(first_state, 1), (a_state, 2), (b_state, 2)]:
        if state["cursor"] != {"epoch": 0, "batch_index": 0, "next_chunk": step, "global_step": step}:
            raise ValueError("incorrect update cursor")
        moments = state["optimizer"]["state"]
        if not moments or any(float(v["step"]) != step or not {"exp_avg", "exp_avg_sq"} <= v.keys()
                              for v in moments.values()):
            raise ValueError("missing moments or incorrect optimizer step count")
    if canonical(first_state["scheduler"]) != canonical(a_state["scheduler"]):
        raise ValueError("epoch scheduler advanced within minibatch")
    if canonical(first_state["model"]) == canonical(a_state["model"]):
        raise ValueError("no continuation update")
    if canonical(first_state["batch"]) != canonical(a_state["batch"]):
        raise ValueError("pending batch changed")
    first_metrics = first_info["metrics"]
    if (not first_info["augmented_images_changed"] or first_metrics["gradient_l2"] <= 0 or
            first_metrics["changed_parameter_tensors"] == 0 or
            first_metrics["effective_original_inputs"] != 128 or first_metrics["noisy_batch"] != 256):
        raise ValueError("native step/data control failed")
    r = json.loads((root / "reference.json").read_text())["second_step"]
    s = json.loads((root / "resume.json").read_text())["second_step"]
    if {k: v for k, v in r.items() if k != "seconds"} != {k: v for k, v in s.items() if k != "seconds"}:
        raise ValueError("next-step diagnostics differ")
    result = {"audit": "PASS", "config_sha256": cfg_hash, "exact_state_component_equality": equality,
              "next_step_metrics_equal_excluding_time": True,
              "optimizer_step_counts": [1, 2], "epoch_scheduler_unchanged_within_batch": True,
              "global_step": 2, "scope": "saved optimizer-update reproducibility, not learned accuracy/certification",
              "audit_seconds": time.monotonic() - start}
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
