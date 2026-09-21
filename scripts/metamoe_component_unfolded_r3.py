"""R3: unfused eval-BN ONNX export, SAME two component requests and gates.

Separate source-preserving graph conversion variant, not byte-identical author
export. Original author specification/configuration/backend remain unchanged.
Finite conformance is not a full-domain or native floating equivalence proof.
"""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from recent_moe_deployment import git_identity, sha256, supervise
from recent_moe_env_inventory import inventory


def inspect_vnnlib(text, label, classes=10, dimensions=3072):
    if text.count("(assert") != 2 * dimensions + 1 or "(assert\n    (or\n" not in text:
        raise ValueError("unexpected assertion structure")
    inputs = re.findall(r"\(declare-const X_(\d+) Real\)", text)
    outputs = re.findall(r"\(declare-const Y_(\d+) Real\)", text)
    if inputs != [str(i) for i in range(dimensions)] or outputs != [str(i) for i in range(classes)]:
        raise ValueError("declaration coverage mismatch")
    if not 0 <= label < classes:
        raise ValueError("invalid label")
    lower, upper = {}, {}
    for op, i, number in re.findall(r"\((>=|<=) X_(\d+) ([^\s()]+)\)", text):
        target = lower if op == ">=" else upper
        if int(i) in target:
            raise ValueError("duplicate input bound")
        target[int(i)] = Fraction(number)
    if set(lower) != set(range(dimensions)) or set(upper) != set(lower):
        raise ValueError("input bounds incomplete")
    if any(lower[i] > upper[i] for i in lower):
        raise ValueError("inverted input bounds")
    rows = re.findall(r"\(and \(>= Y_(\d+) Y_(\d+)\)\)", text)
    if rows != [(str(i), str(label)) for i in range(classes) if i != label]:
        raise ValueError("output property coverage mismatch")
    return lower, upper


def parse_backend_result(stdout, returncode, seconds, limit=300):
    if returncode != 0:
        return {"status": "ERROR", "reason": "backend_nonzero_exit"}
    rows = re.findall(r"^Result: (.+?) in ([\d.]+) seconds$", stdout, re.M)
    totals = re.findall(r"Final verified acc: [\d.]+% \(total (\d+) examples\)", stdout)
    if len(rows) != 1 or totals != ["1"]:
        return {"status": "ERROR", "reason": "missing_or_ambiguous_terminal"}
    raw, reported_seconds = rows[0]
    result = {"backend_status": raw, "backend_reported_seconds": float(reported_seconds)}
    if not math.isfinite(seconds) or seconds > limit or "timed out" in raw:
        return dict(result, status="TIMEOUT", reason="late_or_budget_exhausted")
    if raw in {"safe-incomplete", "safe", "safe-complete"}:
        return dict(result, status="BACKEND_POSITIVE", evidence_grade="AUTHOR_BACKEND_REPORTED_ONNX_RESULT")
    if raw.startswith("unsafe"):
        return dict(result, status="BACKEND_UNSAFE_UNREPLAYED",
                    reason="not_a_full_dynamic_MoE_counterexample")
    if raw in {"unknown", "timeout"} or "unknown" in raw:
        return dict(result, status="UNKNOWN", reason="backend_unresolved")
    return dict(result, status="ERROR", reason="unrecognized_terminal")


def validate_freeze(config):
    root = Path(config["author_repo"])
    for repo, expected in [(root, config["author_commit"]),
                           (Path(config["backend_repo"]), config["backend_commit"]),
                           (Path(config["backend_repo"]) / "auto_LiRPA", config["lirpa_commit"])]:
        identity = git_identity(repo)
        if identity["head"] != expected or identity["status"]:
            raise ValueError(f"source identity/cleanliness mismatch: {repo}")
    for relative, digest in config["execution_files"].items():
        if sha256(Path(__file__).resolve().parents[1] / relative) != digest:
            raise ValueError(f"execution source changed: {relative}")
    for item in config["requests"]:
        if sha256(root / item["checkpoint"]) != item["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
    data_root = Path(config["data_root"])
    if sha256(data_root / "manifest.json") != config["data_manifest_sha256"]:
        raise ValueError("dataset manifest changed")
    for name, info in json.loads((data_root / "manifest.json").read_text())["files"].items():
        if sha256(data_root / name) != info["sha256"]:
            raise ValueError(f"dataset file changed: {name}")
    env_path = Path(config["environment_inventory"])
    if sha256(env_path) != config["environment_inventory_sha256"]:
        raise ValueError("environment inventory changed")
    if inventory([config["python"]]) != json.loads(env_path.read_text()):
        raise ValueError("installed environment changed")


def worker(config, index, work):
    start = time.monotonic()
    work.mkdir(parents=True, exist_ok=False)
    events = []
    def phase(name):
        record = {"phase": name, "elapsed_seconds": time.monotonic() - start}
        events.append(record)
        with (work / "phases.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    item = config["requests"][index]
    terminal = {"request": item, "status": "ERROR"}
    try:
        phase("preflight_started")
        validate_freeze(config)
        repo = Path(config["author_repo"])
        for path in [repo, repo / "src/Vision_Transformer_Pytorch",
                     repo / "src/Formal_Neural_Network_Verification/alpha-beta-crown"]:
            sys.path.insert(0, str(path))
        import numpy as np
        import torch
        import onnxruntime as ort
        import yaml
        from metamoe_unfolded_export import export_unfolded
        from create_vnnlib_specs import load_dataset, create_vnnlib_spec
        from verify_expert_abcrown import get_normalization_values, create_verification_config
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        # Both full-model pickles passed restricted weights_only loading under
        # seven reviewed globals beforehand. Their exact bytes are pinned above;
        # this older author-compatible Torch uses the author's original loader.
        checkpoint = repo / item["checkpoint"]
        model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
        if type(model).__name__ != "ModelWrapper" or type(model.model).__name__ != "UltraVerifiableCNN":
            raise ValueError("unexpected author checkpoint topology")
        phase("model_loaded")
        images, labels, classes = load_dataset(item["dataset"], config["data_root"], 1)
        if item["dataset_index"] != 0 or len(images) != 1 or classes != 10:
            raise ValueError("control selection changed")
        x, label = images[0].unsqueeze(0), labels[0]
        np.save(work / "input.npy", x.numpy(), allow_pickle=False)
        mean, std = get_normalization_values(item["dataset"])
        with torch.no_grad():
            original_logits = model(x)[0]
        phase("data_loaded")
        (work / "onnx").mkdir()
        onnx_path = work / "onnx/model.onnx"
        export_info = export_unfolded(model.model, x, onnx_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        errors = []
        for probe in [x, torch.zeros_like(x), torch.randn_like(x)]:
            with torch.no_grad():
                expected = model(probe)[0].numpy()
            actual = session.run(None, {session.get_inputs()[0].name: probe.numpy()})[0]
            error = float(np.max(np.abs(actual - expected)))
            if not math.isfinite(error) or error >= config["conformance_tolerance"]:
                raise ValueError(f"original-to-ONNX numerical conformance failed: {error}")
            errors.append(error)
        phase("onnx_export_and_conformance_complete")
        spec = work / "request.vnnlib"
        epsilon = config["epsilon_numerator"] / config["epsilon_denominator"]
        create_vnnlib_spec(images[0], label, epsilon, classes, str(spec))
        lo, hi = inspect_vnnlib(spec.read_text(), label)
        lower32 = np.clip(images[0].numpy().reshape(-1) - epsilon, -10., 10.)
        upper32 = np.clip(images[0].numpy().reshape(-1) + epsilon, -10., 10.)
        inward = {"lower": sum(lo[i] > Fraction(float(v)) for i, v in enumerate(lower32)),
                  "upper": sum(hi[i] < Fraction(float(v)) for i, v in enumerate(upper32))}
        csv = work / "instances.csv"
        csv.write_text(str(spec) + "\n")
        config_path = work / "backend.yaml"
        create_verification_config(str(checkpoint), str(onnx_path), item["dataset"],
                                   epsilon, 1, config["solver_seconds"], str(config_path),
                                   mean, std, vnnlib_dir=str(work), csv_file=str(csv))
        backend = yaml.safe_load(config_path.read_text())
        backend["general"].update(device="cpu", seed=config["seed"],
                                   results_file=str(work / "backend_results.pkl"))
        config_path.write_text(yaml.safe_dump(backend, sort_keys=False))
        prepared = {"input_sha256": sha256(work / "input.npy"), "label": label,
                    "clean_prediction": int(original_logits.argmax()),
                    "original_logits": original_logits.tolist(), "mean": mean, "std": std,
                    "onnx_sha256": sha256(onnx_path), "vnnlib_sha256": sha256(spec),
                    "config_sha256": sha256(config_path), "property_rows": classes - 1,
                    "export_info": export_info,
                    "conformance_errors": errors, "vnnlib_inward_decimal_coordinates": inward,
                    "epsilon_space": "author normalized input, not raw pixels",
                    "guarantee_object": "serialized ONNX plus decimal VNNLIB; upstream equivalence unproved"}
        (work / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
        phase("backend_started")
        backend_start = time.monotonic()
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
        # Inherit the supervisor-owned process group so an outer deadline also
        # kills this backend and its children. No detached solver processes.
        with (work / "backend.stdout").open("w") as out, (work / "backend.stderr").open("w") as err:
            result = subprocess.run([sys.executable, str(Path(config["backend_repo"]) /
                                    "complete_verifier/abcrown.py"), "--config", str(config_path)],
                                    cwd=work, env=env, stdout=out, stderr=err)
        elapsed = time.monotonic() - backend_start
        phase("backend_finished")
        terminal.update(parse_backend_result((work / "backend.stdout").read_text(),
                                               result.returncode, elapsed, config["solver_seconds"]))
        terminal.update(prepared=prepared, backend_wall_seconds=elapsed,
                        backend_exit_code=result.returncode)
    except Exception as exc:
        terminal.update(status="ERROR", error=repr(exc))
    terminal.update(worker_wall_seconds=time.monotonic() - start, phases=events)
    (work / "terminal.json").write_text(json.dumps(terminal, indent=2) + "\n")
    print(json.dumps(terminal), flush=True)
    if terminal["status"] == "ERROR":
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--worker", type=int)
    p.add_argument("--work", type=Path)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.worker is not None:
        return worker(config, a.worker, a.work.resolve())
    validate_freeze(config)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    terminals = []
    blocked = False
    for index, request in enumerate(config["requests"]):
        if blocked:
            terminals.append({"request": request, "status": "NOT_STARTED_AFTER_ERROR"})
            continue
        out = root / request["id"]
        receipt = supervise([config["python"], str(Path(__file__).resolve()), "--config",
                             str(a.config.resolve()), "--worker", str(index), "--work", str(out / "work")],
                            str(root), out, config["outer_seconds"], "AUTHOR_COMPONENT_CONTROL")
        terminal = {"request": request, "outer": receipt}
        path = out / "work/terminal.json"
        if path.exists():
            terminal["worker"] = json.loads(path.read_text())
        if receipt["status"] == "TIMEOUT":
            terminal["status"] = "OUTER_TIMEOUT"
        elif receipt["status"] != "COMPLETED" or not path.exists():
            terminal["status"] = "ERROR"
        else:
            terminal["status"] = terminal["worker"]["status"]
        blocked = terminal["status"] == "ERROR"
        terminals.append(terminal)
        (root / "progress.json").write_text(json.dumps(terminals, indent=2) + "\n")
    (root / "batch_terminal.json").write_text(json.dumps({"config_sha256": sha256(a.config),
                                                        "requests": terminals}, indent=2) + "\n")
    print(json.dumps([{ "id": t["request"]["id"], "status": t["status"]} for t in terminals]))
    if blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

