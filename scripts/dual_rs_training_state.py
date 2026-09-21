"""Bound, tensor-only training snapshots. No author training-loop replacement."""
import hashlib
import json
import os
from pathlib import Path
import random


def digest(value):
    import torch
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            h.update(str(len(v)).encode())
            for item in v:
                visit(item)
        elif isinstance(v, (str, int, float, bool, type(None))):
            h.update((type(v).__name__ + ":" + repr(v) + ";").encode())
        else:
            raise TypeError(type(v))
    visit(value)
    return h.hexdigest()


def cpu_tree(v):
    import torch
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone()
    if isinstance(v, dict):
        return {k: cpu_tree(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return type(v)(cpu_tree(x) for x in v)
    return v


def rng_state():
    import numpy as np
    import torch
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


REQUIRED = {"schema", "binding", "model", "optimizer", "scheduler", "rng", "cursor", "batch"}


def snapshot(model, optimizer, scheduler, binding, cursor, batch):
    return cpu_tree({"schema": 1, "binding": binding, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "rng": rng_state(), "cursor": cursor, "batch": batch})


def validate(state, binding):
    if set(state) != REQUIRED or state["schema"] != 1 or state["binding"] != binding:
        raise ValueError("checkpoint schema or source binding mismatch")
    if set(state["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("missing RNG state")
    if not state["optimizer"].get("state") or not state["scheduler"]:
        raise ValueError("missing optimizer or scheduler state")
    cursor = state["cursor"]
    if set(cursor) != {"epoch", "batch_index", "next_chunk", "global_step"} or any(
            type(v) is not int or v < 0 for v in cursor.values()):
        raise ValueError("invalid cursor")
    if cursor["next_chunk"] not in (1, 2) or cursor["global_step"] != cursor["next_chunk"]:
        raise ValueError("invalid control step/chunk mapping")
    if len(state["batch"]) != 4 or len(state["batch"][0]) != binding["loader_batch"]:
        raise ValueError("missing materialized pending batch")
    if digest(state["batch"]) != binding["materialized_batch_sha256"]:
        raise ValueError("pending batch identity mismatch")


def restore(state, model, optimizer, scheduler, binding):
    validate(state, binding)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])  # last: model/optimizer construction may consume RNG


def save_snapshot(path, state):
    import torch
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise ValueError("checkpoint target exists")
    with partial.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.link(partial, path)  # atomic publish, refuses an existing final path
    partial.unlink()
    return {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "logical_sha256": digest(state)}


def load_snapshot(path, expected_file_hash, binding):
    import torch
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_file_hash:
        raise ValueError("checkpoint byte identity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    validate(state, binding)
    return state
