"""Exclusive pre-run tests and one automatically archived bounded shadow."""

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
DIRECTORY = EXPERIMENT / "results/c5_full_native_transaction_20260905_v1"


def main():
    if os.path.lexists(DIRECTORY):
        raise FileExistsError(DIRECTORY)
    names = [Path(__file__).name, "run_c5_full_native_transaction_v1.py", "C5_FULL_NATIVE_TRANSACTION_PREREG_20260905.md",
             "c5_explicit_schema_ledger_v2.py", "c5_native_budgeted_materialization_v1.py",
             "test_c5_explicit_schema_ledger_v2.py", "test_c5_native_budgeted_materialization_v1.py",
             "run_c5_native_boundary_ledger_v1.py", "c5_ordered_union_contraction_v3.py",
             "c5_ordered_row_oracle_v3.py", "c5_live_value_contraction_v1.py", "c5_live_value_union_contraction_v2.py",
             "s0_c2_whole_state_ledger_prototype.py", "../../act/front_end/specs.py", "../../act/front_end/spec_creator_base.py"]
    hashes = {name: _sha256(EXPERIMENT / name) for name in names}
    provenance = worker._provenance(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / "preregistered.json", {
        "formal_gain": 0, "provenance": provenance, "source_sha256": hashes, "wall_cap_s": 240,
        "memory_gb": 16, "whole_sequence_product_cap": 256_000_000, "per_branch_sequence_product_cap": 200_000_000})
    record, start = {"formal_gain": 0}, time.monotonic()
    try:
        tests = ["test_c5_explicit_schema_ledger_v2.py", "test_c5_native_budgeted_materialization_v1.py",
                 "test_s0_c2_whole_state_ledger_prototype.py", "test_c5_ordered_union_contraction_v3.py"]
        with (DIRECTORY / "tests.log").open("x") as stream:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                *[str(EXPERIMENT / p) for p in tests]], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record["tests_exit_code"] = result.returncode
        if result.returncode:
            return
        with (DIRECTORY / "worker.log").open("x") as stream:
            result = subprocess.run([sys.executable, str(EXPERIMENT / "run_c5_full_native_transaction_v1.py")],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record["worker_exit_code"] = result.returncode
    except subprocess.TimeoutExpired as exc:
        record["timeout_s"] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - start, provenance_drift=worker._provenance(ROOT) != provenance,
                      source_drift=any(_sha256(EXPERIMENT / p) != h for p, h in hashes.items()),
                      artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / "exit.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
