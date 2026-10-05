"""Replay every reported ADV in specified complete TLL arms; never repair points."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort

from experiments.neural_hz_20260831.independent_vnnlib_replay import evaluate_vnnlib
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent


def replay(directory, arm_names):
    manifest = json.loads((directory / "preregistered.json").read_text())
    if [j["iid"] for j in manifest["jobs"]] != list(range(32)):
        raise ValueError("expected complete TLL universe")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    records, arm_counts = [], {}
    for arm in arm_names:
        if arm not in {"auto", "one", "uninstrumented"}:
            raise ValueError("unexpected replay arm")
        arm_dir = directory / arm
        summary = json.loads((arm_dir / "summary.json").read_text())
        if summary["completed"] != 32 or summary.get("provenance_drift") is not False:
            raise ValueError("incomplete or drifted arm")
        count = 0
        for job in manifest["jobs"]:
            result_path = arm_dir / f"iid{job['iid']:02d}.json"
            result = json.loads(result_path.read_text())
            exit_record = json.loads((arm_dir / f"iid{job['iid']:02d}.exit.json").read_text())
            if exit_record.get("result_sha256") != _sha256(result_path):
                raise ValueError("result hash differs from completed supervisor record")
            if result.get("iid") != job["iid"] or result.get("provenance") != manifest["provenance"]:
                raise ValueError("result identity/provenance differs from manifest")
            if result["verdict"] != "ADV":
                continue
            count += 1
            if exit_record.get("counted_verdict") != "ADV":
                raise ValueError("reported ADV was rejected by supervisor")
            assets = job["assets"]
            if any(_sha256(Path(p)) != h for p, h in assets.items()):
                raise ValueError("frozen asset changed")
            model_path = next(Path(p) for p in assets if p.endswith(".onnx"))
            spec_path = next(Path(p) for p in assets if "/benchmarks/" in p and p.endswith(".vnnlib"))
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
            if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
                raise ValueError("unexpected TLL ONNX arity")
            info = session.get_inputs()[0]
            dtype = {"tensor(float)": np.float32, "tensor(double)": np.float64}[info.type]
            shape = tuple(1 if not isinstance(d, int) else d for d in info.shape)
            validations = result["concrete_validations"]
            if len(validations) != 1 or validations[0]["valid"] is not True:
                raise ValueError("ADV lacks exactly one validated witness")
            stored = np.asarray(validations[0]["counterexample"], dtype=np.float64)
            native = stored.astype(dtype).reshape(shape)
            y = session.run(None, {info.name: native})[0]
            original = spec_path.read_text()
            native_verdict = evaluate_vnnlib(original, native, y)
            original_box = evaluate_vnnlib(original, stored, y).input_holds
            records.append({"arm": arm, "iid": job["iid"], "result_sha256": _sha256(result_path),
                            "stored_input": stored.tolist(), "native_input": native.reshape(-1).tolist(),
                            "native_output": y.reshape(-1).tolist(), "onnx_input_type": info.type,
                            "native_input_max_abs_cast_change": float(np.max(np.abs(native.reshape(-1).astype(np.float64) - stored))),
                            "stored_input_in_original_box": original_box,
                            "original_vnnlib_evaluation": asdict(native_verdict),
                            "accepted": original_box and native_verdict.all_assertions_hold})
        if count != summary["counts"].get("ADV", 0):
            raise ValueError("ADV replay count differs from completed summary")
        arm_counts[arm] = count
    return {"schema": "tll_campaign_adv_onnx_replay_v2", "formal_baseline": "1870/2413", "formal_gain": 0,
            "onnxruntime_version": ort.__version__, "numpy_version": np.__version__,
            "optimization": "disabled", "provider": "CPUExecutionProvider", "tolerance": 0,
            "scope": "all reported ADV in complete named arms; no rejected-proposal repair",
            "campaign": str(directory.relative_to(ROOT)), "arms": arm_names, "arm_counts": arm_counts,
            "preregistration_sha256": _sha256(directory / "preregistered.json"),
            "runner_sha256": _sha256(Path(__file__)),
            "vnnlib_evaluator_sha256": _sha256(EXPERIMENT / "independent_vnnlib_replay.py"),
            "records": records, "reported_adv_count": len(records),
            "accepted": sum(r["accepted"] for r in records),
            "all_reported_adv_pass": bool(records) and all(r["accepted"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.campaign.resolve(), args.output.resolve()
    if directory.parent != EXPERIMENT / "results" or output.parent != EXPERIMENT / "evidence":
        raise ValueError("paths must stay in isolated experiment")
    if os.path.lexists(output):
        raise FileExistsError(output)
    payload = replay(directory, args.arms)
    _atomic_exclusive_json(output, payload)
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "arm_counts": payload["arm_counts"],
                      "accepted": payload["accepted"], "all_reported_adv_pass": payload["all_reported_adv_pass"]}))


if __name__ == "__main__":
    main()
