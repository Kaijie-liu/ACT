"""One source-bound admitted ReLU44 qualification with actual publication."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
DIRECTORY = EXPERIMENT / "results/c5_relu44_admitted_20260905_v1"


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    previous = json.loads((EXPERIMENT / "results/c5_integrated_relu63_20260905_v1/preregistered.json").read_text())
    hashes = previous["source_sha256"]
    if any(_sha256(EXPERIMENT / p) != h for p, h in hashes.items()):
        raise ValueError("prior candidate source drift")
    names = [Path(__file__).name, "c5_relu44_admitted_worker_v1.py", "C5_RELU44_ADMITTED_PREREG_20260905.md", "c5_admitted_phase_audit_v1.py", "test_c5_admitted_phase_audit_v1.py", "../../act/back_end/solver/test_neural_hz_compact_relu.py", "CHECKPOINT_C5_RELU36_PROOF_RELU63_20260905_SHA256SUMS",
             "C5_RELU36_BOUNDARY_AUDIT_20260905.md", "test_c5_live_value_contraction_v1.py"]
    hashes.update({name: _sha256(EXPERIMENT / name) for name in names})
    command = list(previous["command"])
    command[0], command[1], command[2] = sys.executable, str(EXPERIMENT / "c5_relu44_admitted_worker_v1.py"), str(DIRECTORY)
    command[command.index("--output") + 1] = str(DIRECTORY / "result.json")
    command[command.index("--stop-after-layer") + 1] = "44"
    provenance = worker._provenance(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / "preregistered.json", {"formal_gain": 0, "source_sha256": hashes,
        "provenance": provenance, "command": command, "wall_cap_s": 240, "memory_gb": 16,
        "target_layer": 44, "candidate_runtime": "c5_runtime_materializer_v2",
        "qualification_sha256": "b45d017ac76a55fdc535d411f4c1ee9aa93496b0d603d18da1b286dc02d7e2c3"})
    record, start = {"formal_gain": 0}, time.monotonic()
    try:
        with (DIRECTORY / "tests.log").open("x") as stream:
            tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                str(EXPERIMENT / "test_c5_admitted_phase_audit_v1.py"),
                str(ROOT / "act/back_end/solver/test_neural_hz_compact_relu.py"),
                str(EXPERIMENT / "test_c5_reference_lower_bound_v1.py"),
                str(EXPERIMENT / "test_c5_live_value_contraction_v1.py"),
                str(EXPERIMENT / "test_c5_runtime_materializer_v2.py"),
                str(EXPERIMENT / "test_c5_live_transaction_v1.py"), str(EXPERIMENT / "test_c5_live_roots_v2.py"),
                str(EXPERIMENT / "test_c5_ordered_union_contraction_v3.py"),
                str(EXPERIMENT / "test_c5_partial_csr_owner_ledger_v3.py")], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record["tests_exit_code"] = tests.returncode
        if tests.returncode:
            return
        with (DIRECTORY / "worker.log").open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record["worker_exit_code"] = result.returncode
    except subprocess.TimeoutExpired as exc:
        record["timeout_s"] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - start, provenance_drift=worker._provenance(ROOT) != provenance,
                      source_drift=any(_sha256(EXPERIMENT / p) != h for p, h in hashes.items()),
                      artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        target_seal = DIRECTORY / 'layer44.snapshot.json'
        if target_seal.exists():
            snapshot = json.loads(target_seal.read_text())
            record['target_snapshot_lists_expression'] = 44 in snapshot['expr_layers']
        evidence = EXPERIMENT / 'evidence/c5_relu44_admitted_20260905_v1.json'
        if evidence.exists():
            record['evidence_sha256'] = _sha256(evidence)
        _atomic_exclusive_json(DIRECTORY / "exit.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
