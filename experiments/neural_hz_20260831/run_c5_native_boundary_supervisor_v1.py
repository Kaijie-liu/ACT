"""Bounded exclusive native-boundary diagnostic with automatic exit sealing."""

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
DIRECTORY = EXPERIMENT / "results/c5_native_boundary_20260905_v1"


def main():
    if os.path.lexists(DIRECTORY):
        raise FileExistsError(DIRECTORY)
    names = [Path(__file__).name, "run_c5_native_boundary_ledger_v1.py",
             "C5_NATIVE_BOUNDARY_LEDGER_PREREG_20260905.md", "c5_ordered_union_contraction_v3.py",
             "c5_ordered_row_oracle_v3.py", "s0_c2_whole_state_ledger_prototype.py",
             "run_c5_ordered_full_hz_shadow_v3.py", "c5_live_value_contraction_v1.py",
             "c5_live_value_union_contraction_v2.py"]
    hashes = {name: _sha256(EXPERIMENT / name) for name in names}
    provenance = worker._provenance(ROOT)
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / "preregistered.json", {
        "formal_gain": 0, "provenance": provenance, "source_sha256": hashes, "wall_cap_s": 240, "memory_gb": 16})
    result, start = {"formal_gain": 0}, time.monotonic()
    try:
        with (DIRECTORY / "worker.log").open("x") as stream:
            child = subprocess.run([sys.executable, str(EXPERIMENT / "run_c5_native_boundary_ledger_v1.py")],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        result["exit_code"] = child.returncode
    except subprocess.TimeoutExpired:
        result["timeout"] = True
    finally:
        result.update(wall_s=time.monotonic() - start, provenance_drift=worker._provenance(ROOT) != provenance,
                      source_drift=any(_sha256(EXPERIMENT / p) != h for p, h in hashes.items()),
                      artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / "exit.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
