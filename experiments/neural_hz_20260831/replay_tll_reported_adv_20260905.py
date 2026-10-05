"""Independently replay only the reported TLL ADV on ONNX and original VNNLIB."""

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
DIRECTORY = EXPERIMENT / "results/tll_current_source_20260905_v2"
OUTPUT = EXPERIMENT / "evidence/tll_reported_adv_onnx_replay_20260905_v1.json"


def main():
    if os.path.lexists(OUTPUT):
        raise FileExistsError(OUTPUT)
    manifest = json.loads((DIRECTORY / "preregistered.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    records = []
    for job in manifest["jobs"]:
        result_path = DIRECTORY / f"iid{job['iid']:02d}.json"
        result = json.loads(result_path.read_text())
        if result["verdict"] != "ADV":
            continue
        assets = job["assets"]
        if any(_sha256(Path(p)) != h for p, h in assets.items()):
            raise ValueError("frozen replay asset changed")
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
            raise ValueError("reported ADV lacks exactly one validated witness")
        stored = np.asarray(validations[0]["counterexample"], dtype=np.float64)
        native = stored.astype(dtype).reshape(shape)
        y = session.run(None, {info.name: native})[0]
        native_verdict = evaluate_vnnlib(spec_path.read_text(), native, y)
        original_box = evaluate_vnnlib(spec_path.read_text(), stored, y).input_holds
        records.append({"iid": job["iid"], "result_sha256": _sha256(result_path),
                        "stored_input": stored.tolist(), "native_input": native.reshape(-1).tolist(),
                        "native_output": y.reshape(-1).tolist(), "onnx_input_type": info.type,
                        "native_input_max_abs_cast_change": float(np.max(np.abs(native.reshape(-1).astype(np.float64) - stored))),
                        "stored_input_in_original_box": original_box,
                        "original_vnnlib_evaluation": asdict(native_verdict),
                        "accepted": original_box and native_verdict.all_assertions_hold})
    record = {"schema": "tll_reported_adv_onnx_replay_v1", "date": "2026-09-05",
              "formal_baseline": "1870/2413", "gain": 0,
              "onnxruntime_version": ort.__version__, "numpy_version": np.__version__,
              "optimization": "disabled", "provider": "CPUExecutionProvider",
              "scope": "reported ADV only, native ONNX dtype; rejected iid26 is not repaired or counted",
              "preregistration_sha256": _sha256(DIRECTORY / "preregistered.json"),
              "runner_sha256": _sha256(Path(__file__)), "records": records,
              "reported_adv_count": len(records), "accepted": sum(row["accepted"] for row in records),
              "all_reported_adv_pass": bool(records) and all(row["accepted"] for row in records)}
    _atomic_exclusive_json(OUTPUT, record)
    print(json.dumps({"output": str(OUTPUT), "sha256": _sha256(OUTPUT),
                      "reported_adv_count": len(records), "accepted": record["accepted"]}))


if __name__ == "__main__":
    main()
