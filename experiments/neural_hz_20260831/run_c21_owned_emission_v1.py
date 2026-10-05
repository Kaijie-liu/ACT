"""Exclusive frozen C21 test/worker supervisor with automatic failure retention."""

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
DIRECTORY = EXP / 'results/c21_owned_emission_20260911_v1'
PRIOR = EXP / 'results/c20_product_census_20260911_v1'


def main():
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    if (_sha256(PRIOR / 'exit.json') != '9126b8f9e9b7a98f83375b4de634ffb76cf718249cac3c7ad6b84ae5308bc2ab'
            or _sha256(PRIOR / 'result.json') != 'cea6ea5da1fd383ebf2f4d23f8f255ec88b539a8f044f63940fbf8f0901ef917'):
        raise ValueError('completed C20 artifacts drift')
    prior = json.loads((PRIOR / 'preregistered.json').read_text())
    hashes = dict(prior['source_sha256'])
    if any(_sha256(EXP / n) != sha for n, sha in hashes.items()):
        raise ValueError('inherited source/library drift')
    names = [Path(__file__).name, 'c17_packed_ownership_v1.py', 'c17_owned_graph_v1.py',
        'c21_exact_products_v1.py', 'c21_owned_rows_v1.py', 'c21_owned_emission_v1.py', 'c17_ownership_audit_v1.py',
        'test_c21_exact_products_v1.py', 'test_c21_owned_emission_v1.py', 'c21_owned_emission_worker_v1.py',
        'C21_OWNED_EMISSION_PREREG_20260911.md', 'C21_OWNED_EMISSION_DEVELOPMENT_20260911.md',
        'CHECKPOINT_C20_PRODUCT_20260911_SHA256SUMS', 'C20_PRODUCT_CENSUS_AUDIT_20260911.md',
        'C19_EXACT_PRODUCT_PAYMENT_DESIGN_20260911.md', str(PRIOR / 'exit.json'), str(PRIOR / 'result.json'),
        'results/c5_first_terminal_20260905_v1/layer75.pickle',
        'results/c5_first_terminal_20260905_v1/composition_events.jsonl',
        'results/c9_live_relu_20260906_v1/relu78.pickle',
        'results/c10_live_relu_20260908_v1/relu78.pickle',
        'results/c14_early_rejection_census_20260910_v1/single_use_factor_table.npz']
    hashes.update({n: _sha256(EXP / n) for n in names})
    tests = ['test_c21_exact_products_v1.py', 'test_c21_owned_emission_v1.py', *prior['tests']]
    provenance = baseline._provenance(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(EXP / 'c21_owned_emission_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'tests': tests, 'command': command, 'wall_cap_s': 240, 'memory_gb': 16,
        'work_cap': 256_000_000, 'branch_work_cap': 200_000_000, 'entry_cap': 64_000_000,
        'construction_cap_bytes': 1024**3, 'uid_limit': 2**20, 'packing_radix': 2**40,
        'solver_authorized': False, 'unit_splice_authorized': False, 'native_ingestion_authorized': True,
        'generation_authorized': 'original_expression_only', 'whole_live_path_authorized': False,
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
