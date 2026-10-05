"""Freeze, test, run and automatically retain one complete support diagnostic."""

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
DIRECTORY = EXPERIMENT / 'results/c6_support_affine_plan_20260905_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    previous = json.loads((EXPERIMENT / 'results/c5_first_terminal_20260905_v1/preregistered.json').read_text())
    hashes = previous['source_sha256']
    if any(_sha256(EXPERIMENT / name) != sha for name, sha in hashes.items()):
        raise ValueError('previous source freeze drift')
    names = [Path(__file__).name, 'c6_support_affine_plan_v1.py', 'test_c6_support_affine_plan_v1.py',
        'c6_support_affine_plan_worker_v1.py', 'C6_SUPPORT_AFFINE_PREREG_20260905.md',
        'C6_SUPPORT_AFFINE_DEVELOPMENT_20260905.md', 'C5_FIRST_TERMINAL_TIMEOUT_AUDIT_20260905.md',
        'CHECKPOINT_C5_RELU44_RELU71_TERMINAL_20260905_SHA256SUMS',
        'evidence/c5_first_terminal_postmortem_20260905_v1.json']
    hashes.update({name: _sha256(EXPERIMENT / name) for name in names})
    tests = ['test_c6_support_affine_plan_v1.py', 'test_c5_zero_suffix_audit_v1.py',
        'test_c5_admitted_phase_audit_v1.py', '../../act/back_end/solver/test_neural_hz_compact_relu.py',
        'test_c5_reference_lower_bound_v1.py', 'test_c5_live_value_contraction_v1.py',
        'test_c5_runtime_materializer_v2.py', 'test_c5_live_transaction_v1.py', 'test_c5_live_roots_v2.py',
        'test_c5_ordered_union_contraction_v3.py', 'test_c5_partial_csr_owner_ledger_v3.py']
    command = [sys.executable, str(EXPERIMENT / 'c6_support_affine_plan_worker_v1.py'), str(DIRECTORY)]
    provenance = worker._provenance(ROOT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'formal_gain': 0, 'source_sha256': hashes,
        'provenance': provenance, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'max_support_visits': 256_000_000, 'branch_product_cap': 200_000_000,
        'whole_product_cap': 256_000_000, 'construction_cap_bytes': 1024**3, 'tests': tests})
    record, start = {'formal_gain': 0}, time.monotonic()
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            checked = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXPERIMENT / name) for name in tests)], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = checked.returncode
        if checked.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            child = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = child.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - start, provenance_drift=worker._provenance(ROOT) != provenance,
            source_drift=any(_sha256(EXPERIMENT / name) != sha for name, sha in hashes.items()),
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
