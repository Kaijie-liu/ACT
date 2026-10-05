"""Exclusive frozen C20 test/worker supervisor with automatic failure retention."""

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
DIRECTORY = EXP / 'results/c20_product_census_20260911_v1'
PRIOR = EXP / 'results/c19_sort_geometry_20260911_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    if (_sha256(PRIOR / 'exit.json') != '07cbacdcc5226dbe7355926c9e83b26cda3983cc446263bdc63b87d54de2090c'
            or _sha256(PRIOR / 'result.json') != 'ff20f60ee364ceaffe7bc212c8140c80b919b2015358286ac669cde58977e7cc'):
        raise ValueError('completed C19 artifacts drift')
    prior = json.loads((PRIOR / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c20_product_census_v1.py', 'c20_product_census_worker_v1.py',
        'test_c20_product_census_v1.py', 'C20_PRODUCT_CENSUS_PREREG_20260911.md',
        'CHECKPOINT_C19_SORT_20260911_SHA256SUMS', 'C19_SORT_GEOMETRY_AUDIT_20260911.md',
        'C19_EXACT_PRODUCT_PAYMENT_DESIGN_20260911.md',
        str(PRIOR / 'exit.json'), str(PRIOR / 'result.json'),
        'results/c9_live_relu_20260906_v1/relu78.pickle',
        'results/c10_fused_emission_20260908_v1/fused_hz.pickle']
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = ['test_c20_product_census_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(EXP / 'c20_product_census_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'work_cap': 256_000_000, 'branch_work_cap': 200_000_000, 'entry_cap': 64_000_000,
        'construction_cap_bytes': 1024**3, 'uid_limit': 2**20, 'packing_radix': 2**40,
        'solver_authorized': False, 'unit_splice_authorized': False, 'native_ingestion_authorized': False,
        'generation_authorized': False, 'read_only_product_classes_authorized': True, 'whole_live_path_authorized': False,
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
