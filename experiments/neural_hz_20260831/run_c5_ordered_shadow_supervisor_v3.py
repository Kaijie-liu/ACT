"""Exclusive source-frozen tests + one time-bounded real shadow."""

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
DIRECTORY = EXPERIMENT / "results/c5_ordered_shadow_20260905_v3"


def main():
    if os.path.lexists(DIRECTORY):
        raise FileExistsError(DIRECTORY)
    source_names = ["run_c5_ordered_shadow_supervisor_v3.py", "run_c5_ordered_full_hz_shadow_v3.py",
                    "c5_ordered_union_contraction_v3.py", "c5_ordered_row_oracle_v3.py",
                    "test_c5_ordered_union_contraction_v3.py", "S0_C5_ORDERED_UNION_V3_PREREG_20260905.md",
                    "c5_live_value_contraction_v1.py", "c5_live_value_union_contraction_v2.py"]
    hashes = {name: _sha256(EXPERIMENT / name) for name in source_names}
    provenance = worker._provenance(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / "preregistered.json", {
        "schema": "c5_ordered_shadow_supervisor_v3", "source_sha256": hashes, "provenance": provenance,
        "real_wall_cap_s": 240, "memory_gb": 16, "formal_gain": 0})
    start = time.monotonic()
    result = {"formal_gain": 0}
    try:
        with (DIRECTORY / "tests.log").open("x") as stream:
            tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                str(EXPERIMENT / "test_c5_ordered_union_contraction_v3.py")], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        result["tests_exit_code"] = tests.returncode
        if tests.returncode:
            return
        with (DIRECTORY / "worker.log").open("x") as stream:
            real = subprocess.run([sys.executable, str(EXPERIMENT / "run_c5_ordered_full_hz_shadow_v3.py")],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        result["real_exit_code"] = real.returncode
    except subprocess.TimeoutExpired as exc:
        result["timeout_s"] = exc.timeout
    finally:
        result.update(wall_s=time.monotonic() - start,
                      provenance_drift=worker._provenance(ROOT) != provenance,
                      source_drift=any(_sha256(EXPERIMENT / p) != h for p, h in hashes.items()),
                      artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / "exit.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
