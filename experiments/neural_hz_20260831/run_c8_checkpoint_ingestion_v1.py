"""Fresh checkpoint audit and native SciPy-HiGHS model loading, never solving."""

import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as baseline_worker

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c8_checkpoint_ingestion_20260905_v1'
PREVIOUS = EXP / 'results/c8_dyadic_balance_20260905_v1'


def worker():
    import numpy as np
    from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import Lifted, expression_binding
    from experiments.neural_hz_20260831.c8_dyadic_balance_audit_v1 import audit
    from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    frozen = json.loads((DIRECTORY / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in frozen['source_sha256'].items()):
        raise ValueError('source/library/checkpoint freeze drift')
    checkpoint = PREVIOUS / 'lifted_hz.pickle'
    if _sha256(checkpoint) != frozen['source_sha256'][str(checkpoint)]:
        raise ValueError('checkpoint seal drift')
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    if saved['schema'] != 'c8_dyadic_balance_checkpoint_v1':
        raise ValueError('unexpected checkpoint schema')
    lifted = Lifted(saved['expression'], expression_binding(saved['expression']), saved['hz'],
        saved['definition_graph'], saved['root'], saved['old_n_cont'], saved['old_n_bin'],
        saved['old_n_eq'], saved['keep'], saved['report'])
    lifted.seal = lifted.fingerprint()
    identity = audit(lifted)
    exponents = np.concatenate([n['exponents'][n['needed']] for n in lifted.nodes])
    print(json.dumps({'event': 'checkpoint_audited', 'rows': identity['all_defining_rows_checked'],
        'exponent_range': [int(exponents.min()), int(exponents.max())]}), flush=True)
    native = inspect(lifted.hz)
    record = {'schema': 'c8_checkpoint_ingestion_v1', 'formal_gain': 0,
        'checkpoint_reloaded_and_audited': True, 'identity': identity,
        'auxiliary_exponent_min': int(exponents.min()), 'auxiliary_exponent_max': int(exponents.max()),
        **native, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'provenance': frozen['provenance'], 'source_sha256': frozen['source_sha256'],
        'checkpoint_sha256': _sha256(checkpoint)}
    _atomic_exclusive_json(DIRECTORY / 'result.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'input_matrix_nnz', 'retained_matrix_nnz',
        'different_coefficients', 'solve_called')}), flush=True)


def main():
    import scipy.optimize._highspy._core as core
    import scipy.optimize._highspy._highs_wrapper as wrapper
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    previous_result = json.loads((PREVIOUS / 'result.json').read_text())
    previous_exit = json.loads((PREVIOUS / 'exit.json').read_text())
    if (not previous_result['offline_registered_gates_passed']
            or previous_exit.get('worker_exit_code') != 0 or previous_exit.get('tests_exit_code') != 0
            or previous_exit['source_drift'] or previous_exit['provenance_drift']
            or _sha256(PREVIOUS / 'lifted_hz.pickle') != previous_result['checkpoint_sha256']):
        raise ValueError('offline candidate not qualified for ingestion')
    prior = json.loads((PREVIOUS / 'preregistered.json').read_text())
    hashes = prior['source_sha256']
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('prior source drift')
    for path in (Path(__file__), EXP / 'C8_DYADIC_BALANCE_PREREG_20260905.md',
                 PREVIOUS / 'result.json', PREVIOUS / 'exit.json', PREVIOUS / 'lifted_hz.pickle', Path(core.__file__), Path(wrapper.__file__)):
        hashes[str(path)] = _sha256(path)
    provenance = baseline_worker._provenance(ROOT)
    command = [sys.executable, str(Path(__file__)), '--worker']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', {'source_sha256': hashes, 'provenance': provenance,
        'command': command, 'wall_cap_s': 240, 'memory_gb': 16, 'solve_authorized': False, 'formal_gain': 0})
    started, record = time.monotonic(), {'formal_gain': 0}
    try:
        with (DIRECTORY / 'tests.log').open('x') as stream:
            tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                *(str(EXP / name) for name in prior['tests'])], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        record['tests_exit_code'] = tests.returncode
        if tests.returncode:
            return
        with (DIRECTORY / 'worker.log').open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        record['worker_exit_code'] = completed.returncode
    except subprocess.TimeoutExpired as exc:
        record['timeout_s'] = exc.timeout
    finally:
        record.update(wall_s=time.monotonic() - started, source_drift=any(_sha256(EXP / name) != sha for name, sha in hashes.items()),
            provenance_drift=baseline_worker._provenance(ROOT) != provenance,
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY / 'exit.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    worker() if len(sys.argv) == 2 and sys.argv[1] == '--worker' else main()

