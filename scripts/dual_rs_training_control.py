"""Native one-update control and fresh-process resume under ONE outer deadline."""
import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from recent_moe_deployment import git_identity, sha256, supervise
from recent_moe_env_inventory import inventory
from dual_rs_training_state import (digest, load_snapshot, restore,
                                    save_snapshot, snapshot)


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validate_config(cfg):
    fixed = {"seed": 1, "loader_batch": 256, "workers": 0, "num_noise_vec": 2,
             "noise_sd": 1.0, "sigma_cand": [0.25, 0.5, 1.0], "loss_cl": "softce",
             "loss_con": True, "lbd": 40.0, "eta": 0.5, "adp_con_wght": "max",
             "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"}
    if any(cfg[k] != v for k, v in fixed.items()):
        raise ValueError("control recipe mismatch")
    identity = git_identity(cfg["author_repo"])
    if (identity["head"] != cfg["author_commit"] or identity["status"] or
            identity["tracked_sha256"] != cfg["author_files"]):
        raise ValueError("author source mismatch")
    for path, h in {**cfg["execution_files"], **cfg["input_files"]}.items():
        if sha256(path) != h:
            raise ValueError("bound file changed: " + path)
    env = Path(cfg["environment_inventory"])
    if sha256(env) != cfg["environment_inventory_sha256"] or inventory([cfg["python"]]) != json.loads(env.read_text()):
        raise ValueError("environment mismatch")


def build(cfg):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < cfg["minimum_free_gpu_gib"] * 1024**3:
        raise ValueError("GPU resource gate not met")
    torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(Path(cfg["author_repo"]) / "code"))
    native_file = Path(cfg["author_repo"]) / "code/train_sigma_est.py"
    argv = sys.argv
    sys.argv = [str(native_file), "cifar10", "cifar_resnet110", "--noise_sd", "1.0",
                "--num_noise_vec", "2", "--sigma_cand", ".25", ".5", "1.0", "--loss_cl", "softce",
                "--loss_con", "--class_weights", "--lbd", "40", "--eta", ".5", "--adp_con_wght", "max",
                "--workers", "0", "--id", "1"]
    try:
        spec = importlib.util.spec_from_file_location("native_sigma_train_control", native_file)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
    finally:
        sys.argv = argv
    native.seed_everything(cfg["seed"], strict=True)
    import datasets
    datasets.DATASET_LOC = cfg["data_root"]  # location only, native transforms unchanged
    model = native.get_architecture("cifar_resnet110", "cifar10", 3, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                                  weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["scheduler"]["milestones"],
                                                    gamma=cfg["scheduler"]["gamma"])
    denoiser = native.DiffusionModel(cfg["diffusion"]).eval()
    t = 0
    while denoiser.diffusion.sqrt_one_minus_alphas_cumprod[t] / denoiser.diffusion.sqrt_alphas_cumprod[t] < cfg["noise_sd"] * 2:
        t += 1
    weights = np.load(cfg["class_weights"], allow_pickle=False)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid class weights")
    return native, model, optimizer, scheduler, denoiser, t, weights


class StepComplete(Exception):
    pass


def native_step(native, model, optimizer, denoiser, timestep, weights, batch, chunk):
    """Only bound iteration: unmodified author's loss/backward/optimizer path."""
    import torch
    initial_parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
    original_chunk = native._chunk_minibatch
    native._chunk_minibatch = lambda b, n: itertools.islice(original_chunk(b, n), chunk, None)
    def pre(_opt, _args, _kwargs):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("missing/nonfinite gradients before update")
    def post(_opt, _args, _kwargs):
        raise StepComplete()
    before_handle = optimizer.register_step_pre_hook(pre)
    after_handle = optimizer.register_step_post_hook(post)
    start = time.monotonic()
    try:
        native.train([batch], denoiser, timestep, model, optimizer, 0, 1.0, torch.device("cuda"),
                     writer=None, class_weights=weights)
        raise ValueError("native loop did not execute exactly one bounded step")
    except StepComplete as exc:
        tb = exc.__traceback__
        while tb and tb.tb_frame.f_code is not native.train.__code__:
            tb = tb.tb_next
        if tb is None:
            raise ValueError("missing native training frame")
        v = tb.tb_frame.f_locals
        stats = {"loss": float(v["loss"].detach().cpu()),
                 "classification_loss": float(v["loss_cl"].mean().detach().cpu()),
                 "weighted_consistency_loss": float(v["loss_con"].mean().detach().cpu()),
                 "effective_original_inputs": int(v["batch_size"]),
                 "noisy_batch": len(v["outputs"]),
                 "input_sha256": digest(v["inputs"]), "denoised_sha256": digest(v["imgs"]),
                 "output_sha256": digest(v["outputs"]),
                 "gradients_sha256": digest({n: p.grad for n, p in model.named_parameters()}),
                 "gradient_l2": float(torch.linalg.vector_norm(torch.cat([
                     p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])).cpu()),
                 "changed_parameter_tensors": sum(not torch.equal(initial_parameters[n], p)
                                                  for n, p in model.named_parameters())}
        del v, tb
    finally:
        before_handle.remove()
        after_handle.remove()
        native._chunk_minibatch = original_chunk
    if not all(math.isfinite(stats[k]) for k in ["loss", "classification_loss", "weighted_consistency_loss", "gradient_l2"]):
        raise ValueError("nonfinite native metrics")
    if stats["gradient_l2"] <= 0 or stats["changed_parameter_tensors"] == 0:
        raise ValueError("no effective training update")
    stats["seconds"] = time.monotonic() - start
    return stats


def run_phase(cfg, config_path, phase):
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from torchvision.transforms.functional import to_tensor
    start = time.monotonic()
    validate_config(cfg)
    native, model, optimizer, scheduler, denoiser, timestep, weights = build(cfg)
    frozen_denoiser = digest(denoiser.state_dict())
    root = Path(cfg["output_root"])
    if phase == "reference":
        dataset = native.get_dataset("cifar10", "train_sigma_est", cfg["label_table"])
        indices = cfg["selected_train_indices"]
        labels = np.load(cfg["label_table"], allow_pickle=False)
        if np.flatnonzero(labels[:, -2] != 0)[:256].tolist() != indices:
            raise ValueError("selection mismatch")
        batch = default_collate([dataset[i] for i in indices])
        expected = torch.from_numpy(labels[indices])
        if not (torch.equal(batch[1], expected[:, -1].long()) and torch.equal(batch[2], expected[:, -2])
                and torch.equal(batch[3], expected[:, :-2])):
            raise ValueError("native data/label binding mismatch")
        changed = sum(not torch.equal(batch[0][n], to_tensor(dataset.data[i])) for n, i in enumerate(indices))
        if not changed or batch[0].shape != (256, 3, 32, 32) or batch[0].min() < 0 or batch[0].max() > 1:
            raise ValueError("augmentation/input contract failed")
        binding = {"config_sha256": sha256(config_path), "author_commit": cfg["author_commit"],
                   "loader_batch": cfg["loader_batch"], "materialized_batch_sha256": digest(batch),
                   "train_indices_sha256": digest(indices), "denoiser_sha256": frozen_denoiser,
                   "environment_sha256": cfg["environment_inventory_sha256"]}
        first = native_step(native, model, optimizer, denoiser, timestep, weights, batch, 0)
        cursor = {"epoch": 0, "batch_index": 0, "next_chunk": 1, "global_step": 1}
        state = snapshot(model, optimizer, scheduler, binding, cursor, batch)
        identity = save_snapshot(root / "after_step1.pt", state)
        write_json(root / "after_step1.json", {"binding": binding, **identity, "metrics": first,
                                               "augmented_images_changed": changed, "timestep": timestep})
        second = native_step(native, model, optimizer, denoiser, timestep, weights, batch, 1)
    else:
        record = json.loads((root / "after_step1.json").read_text())
        binding = record["binding"]
        if binding["config_sha256"] != sha256(config_path) or binding["denoiser_sha256"] != frozen_denoiser:
            raise ValueError("restore source binding mismatch")
        state = load_snapshot(root / "after_step1.pt", record["file_sha256"], binding)
        batch = state["batch"]
        restore(state, model, optimizer, scheduler, binding)
        restored = snapshot(model, optimizer, scheduler, binding, state["cursor"], batch)
        if digest(restored) != record["logical_sha256"]:
            raise ValueError("restored state differs before continuation")
        second = native_step(native, model, optimizer, denoiser, timestep, weights, batch, 1)
    if digest(denoiser.state_dict()) != frozen_denoiser or any(p.grad is not None for p in denoiser.parameters()):
        raise ValueError("frozen diffusion model was trained")
    cursor = {"epoch": 0, "batch_index": 0, "next_chunk": 2, "global_step": 2}
    final = snapshot(model, optimizer, scheduler, binding, cursor, batch)
    final_id = save_snapshot(root / f"{phase}_after_step2.pt", final)
    write_json(root / f"{phase}.json", {"status": "COMPLETED", "phase": phase,
               "config_sha256": sha256(config_path), "second_step": second, **final_id,
               "component_hashes": {k: digest(v) for k, v in final.items()},
               "denoiser_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "worker_seconds": time.monotonic() - start})


def summarize_terminal(root, receipt):
    """Outer status wins over any late/partial child output; no synthetic PASS."""
    start = time.monotonic()
    root = Path(root)
    stages = []
    for phase in ["reference", "resume", "audit"]:
        end = root / f"{phase}_finished.json"
        begun = (root / f"{phase}_started.json").exists()
        row = json.loads(end.read_text()) if end.exists() else {"phase": phase}
        row["status"] = ("COMPLETED" if row.get("returncode") == 0 else
                         "ERROR" if "returncode" in row else
                         "INTERRUPTED" if begun else "NOT_STARTED")
        stages.append(row)
    audit = root / "audit.json"
    passed = (receipt["status"] == "COMPLETED" and
              all(s["status"] == "COMPLETED" for s in stages) and
              audit.exists() and json.loads(audit.read_text()).get("audit") == "PASS")
    result = {"status": "CONTROL_PASS" if passed else receipt["status"] if receipt["status"] != "COMPLETED" else "ERROR",
              "stages": stages, "deadline_seconds": receipt["deadline_seconds"],
              "execution_including_preflight_seconds": receipt["execution_including_preflight_seconds"],
              "total_with_postflight_seconds": receipt["total_with_postflight_seconds"],
              "postflight_in_execution_budget": False,
              "partial_files_retained": sorted(p.name for p in root.iterdir() if p.is_file()),
              "terminal_accounting_seconds": time.monotonic() - start}
    write_json(root / "outer_terminal.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--phase", choices=["reference", "resume", "manager"])
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = Path(cfg["output_root"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg["cublas_workspace_config"]
    if a.phase in ("reference", "resume"):
        run_phase(cfg, a.config, a.phase)
        return
    if a.phase == "manager":
        # Child phases deliberately inherit this group's deadline; no new sessions.
        stages = []
        for phase in ["reference", "resume", "audit"]:
            write_json(root / f"{phase}_started.json", {"phase": phase, "monotonic": time.monotonic()})
            cmd = ([sys.executable, str(Path(__file__).resolve()), "--config", str(a.config.resolve()), "--phase", phase]
                   if phase != "audit" else [sys.executable, str(Path(__file__).with_name("audit_dual_rs_training_control.py").resolve()),
                                             "--config", str(a.config.resolve()), "--output", str(root / "audit.json")])
            start = time.monotonic()
            with (root / f"{phase}.stdout").open("x") as out, (root / f"{phase}.stderr").open("x") as err:
                code = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], stdout=out, stderr=err).returncode
            stages.append({"phase": phase, "returncode": code, "wall_seconds": time.monotonic() - start})
            write_json(root / f"{phase}_finished.json", stages[-1])
            if code:
                write_json(root / "terminal.json", {"status": "ERROR", "stages": stages})
                raise SystemExit(code)
        write_json(root / "terminal.json", {"status": "CONTROL_PASS", "stages": stages})
        return
    # Expensive identity/environment/data checks are inside each timed worker.
    receipt = supervise([cfg["python"], str(Path(__file__).resolve()), "--config", str(a.config.resolve()), "--phase", "manager"],
                        str(Path(__file__).resolve().parents[1]), root, cfg["total_control_seconds"],
                        "AUTHOR_TRAIN_STEP_RESUME_CONTROL", cfg["author_repo"], cpu_only=False)
    terminal = summarize_terminal(root, receipt)
    print(json.dumps(terminal, indent=2))
    raise SystemExit(0 if terminal["status"] == "CONTROL_PASS" else 1)


if __name__ == "__main__":
    main()
