"""Freeze new accounting instrumentation and retain every exit automatically."""

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
DIRECTORY = EXPERIMENT / "results/c5_full_native_owner_ledger_20260905_v2"


def main():
    if os.path.lexists(DIRECTORY):
        raise FileExistsError(DIRECTORY)
    previous = EXPERIMENT / "results/c5_full_native_transaction_20260905_v1/preregistered.json"
    hashes = json.loads(previous.read_text())["source_sha256"]
    if any(_sha256(EXPERIMENT / path) != digest for path, digest in hashes.items()):
        raise ValueError("original diagnostic source drift")
    names = [Path(__file__).name, "run_c5_full_native_owner_ledger_v2.py", "c5_partial_csr_owner_ledger_v3.py",
             "test_c5_partial_csr_owner_ledger_v3.py", "C5_PARTIAL_CSR_OWNER_PREREG_20260905.md"]
    hashes.update({name: _sha256(EXPERIMENT / name) for name in names})
    provenance = worker._provenance(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / "preregistered.json", {"formal_gain": 0, "provenance": provenance,
        "source_sha256": hashes, "wall_cap_s": 240, "memory_gb": 16, "accounting_only_repetition": True})
    record, start = {"formal_gain": 0}, time.monotonic()
    try:
        tests = ["test_c5_partial_csr_owner_ledger_v3.py", "test_c5_explicit_schema_ledger_v2.py",
                 "test_c5_native_budgeted_materialization_v1.py", "test_s0_c2_whole_state_ledger_prototype.py",
                 "test_c5_ordered_union_contraction_v3.py"]
        with (DIRECTORY / "tests.log").open("x") as stream:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                *[str(EXPERIMENT / name) for name in tests]], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record["tests_exit_code"] = result.returncode
        if result.returncode:
            return
        with (DIRECTORY / "worker.log").open("x") as stream:
            result = subprocess.run([sys.executable, str(EXPERIMENT / "run_c5_full_native_owner_ledger_v2.py")],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record["worker_exit_code"] = result.returncode
    except subprocess.TimeoutExpired as exc:
        record["timeout_s"] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - start, provenance_drift=worker._provenance(ROOT) != provenance,
                      source_drift=any(_sha256(EXPERIMENT / name) != digest for name, digest in hashes.items()),
                      artifacts={path.name: _sha256(path) for path in DIRECTORY.iterdir() if path.is_file()})
        _atomic_exclusive_json(DIRECTORY / "exit.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
