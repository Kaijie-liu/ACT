"""One preregistered CPU prefix acquisition with an automatic exclusive exit record."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import ANCHOR, ANCHOR_SHA256
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
DIRECTORY = EXPERIMENT / "results/c5_corrected_prefix_20260905_v1"


def main():
    if os.path.lexists(DIRECTORY):
        raise FileExistsError(DIRECTORY)
    if _sha256(ANCHOR) != ANCHOR_SHA256:
        raise ValueError("anchor drift")
    anchor = json.loads(ANCHOR.read_text())
    for name in ("model", "converted_spec"):
        if _sha256(Path(anchor["target"][name])) != anchor["target"][name + "_sha256"]:
            raise ValueError("target asset drift")
    provenance = worker._provenance(ROOT)
    paths = [Path(__file__), EXPERIMENT / "c5_corrected_prefix_worker_v1.py",
             EXPERIMENT / "S0_C5_LIVE_VALUE_SUPPORT_PREREG_20260905.md"]
    hashes = {str(p.relative_to(ROOT)): _sha256(p) for p in paths}
    command = [sys.executable, str(EXPERIMENT / "c5_corrected_prefix_worker_v1.py"), str(DIRECTORY),
               "tinyimagenet_2024", "143", "--arm", "sparse_phase_implicit_relu_census",
               "--act-root", str(ROOT), "--bench-root", "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
               "--vnnlib-root", str(EXPERIMENT / "vnnlib_v2"), "--representation", "sparse",
               "--solver-timeout", "45", "--memory-gb", "16", "--stop-after-layer", "32",
               "--debug-structure", "--output", str(DIRECTORY / "result.json")]
    # Derive the actual converted-spec root from the pinned target, not a guess.
    converted = Path(anchor["target"]["converted_spec"])
    parts = converted.parts
    family_index = parts.index("tinyimagenet_2024")
    command[command.index("--vnnlib-root") + 1] = str(Path(*parts[:family_index]))
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / "preregistered.json", {
        "schema": "c5_corrected_prefix_probe_v1", "formal_gain": 0, "target": anchor["target"],
        "provenance": provenance, "sources": hashes, "command": command,
        "memory_gb": 16, "wall_cap_s": 240, "candidate_enabled": False})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.monotonic()
    record = {"formal_gain": 0}
    try:
        with (DIRECTORY / "worker.log").open("x") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record["exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        record["timeout"] = True
    finally:
        record.update(wall_s=time.monotonic() - start,
                      provenance_drift=worker._provenance(ROOT) != provenance or any(_sha256(ROOT / p) != h for p, h in hashes.items()),
                      artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / "exit.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
