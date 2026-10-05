"""Exclusive frozen C19 test/worker supervisor with automatic failure retention."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c19_sort_geometry_20260911_v1'
PRIOR = EXP / 'results/c18_owned_emission_20260911_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    if (_sha256(PRIOR / 'exit.json') != '0ae272f39349b4b78c429c7a0731c7ec48ece0fd29e6399d43f80b61ea493ae0'
            or _sha256(PRIOR / 'result.json') != 'e8a12384a17f42cf3829cd4000421872294590394dd2a7af4f4d9a97835eabac'):
        raise ValueError('closed C18 artifacts drift')
    prior = json.loads((PRIOR / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c19_sort_geometry_v1.py', 'c19_sort_geometry_worker_v1.py',
        'test_c19_sort_geometry_v1.py', 'C19_SORT_GEOMETRY_PREREG_20260911.md',
        'CHECKPOINT_C18_OWNED_20260911_SHA256SUMS', 'C18_OWNED_EMISSION_AUDIT_20260911.md',
        str(PRIOR / 'exit.json'), str(PRIOR / 'result.json'),
        'results/c9_live_relu_20260906_v1/relu78.pickle',
        'results/c10_fused_emission_20260908_v1/fused_hz.pickle']
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = ['test_c19_sort_geometry_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(EXP / 'c19_sort_geometry_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'work_cap': 256_000_000, 'branch_work_cap': 200_000_000, 'entry_cap': 64_000_000,
        'construction_cap_bytes': 1024**3, 'uid_limit': 2**20, 'packing_radix': 2**40,
        'solver_authorized': False, 'unit_splice_authorized': False, 'native_ingestion_authorized': False,
        'generation_authorized': False, 'read_only_geometry_authorized': True, 'whole_live_path_authorized': False,
        'partial_acceptance_allowed': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            tests_run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / n) for n in tests)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = tests_run.returncode
        if tests_run.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = completed.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - started,
            source_drift=any(_sha256(EXP / n) != sha for n, sha in hashes.items()),
            provenance_drift=baseline._provenance(ROOT) != provenance,
            artifacts={p.name: _sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
