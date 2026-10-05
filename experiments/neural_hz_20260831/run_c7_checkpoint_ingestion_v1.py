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
DIRECTORY = EXP / 'results/c7_checkpoint_ingestion_20260905_v1'
PREVIOUS = EXP / 'results/c7_factored_hz_20260905_v1'


def worker():
    import numpy as np
    import scipy
    import scipy.sparse as sp
    from scipy.optimize._highspy import _core
    from act.back_end.solver.solver_hz import _lower_hz_milp
    from experiments.neural_hz_20260831.c7_factored_hz_v1 import Lifted, expression_binding
    from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import audit
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    frozen = json.loads((DIRECTORY / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in frozen['source_sha256'].items()):
        raise ValueError('source/library freeze drift')
    checkpoint = PREVIOUS / 'lifted_hz.pickle'
    if _sha256(checkpoint) != '6cbe92faf0d6eb5e4113f3e6e0b324a82a00d2d1a6019a73f2c2fd0f522e82df':
        raise ValueError('checkpoint seal drift')
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    lifted = Lifted(saved['expression'], expression_binding(saved['expression']), saved['hz'],
        saved['definition_graph'], saved['root'], saved['old_n_cont'], saved['old_n_bin'],
        saved['old_n_eq'], saved['keep'], saved['report'])
    lifted.seal = lifted.fingerprint()
    identity = audit(lifted)
    highs = _core._Highs()
    options = _core.HighsOptions()
    options.presolve = 'on'
    options.time_limit = 45.
    options.mip_rel_gap = 0.
    high_options_status = highs.passOptions(options)
    thresholds = {'small_matrix_value': options.small_matrix_value, 'large_matrix_value': options.large_matrix_value}
    fields = {}
    for name in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
        data = getattr(lifted.hz, name).data
        nonzero = np.abs(data[data != 0.])
        fields[name] = {'stored': int(data.size), 'nonzero': int(nonzero.size),
            'min_nonzero_abs': float(nonzero.min()) if nonzero.size else None,
            'max_abs': float(nonzero.max()) if nonzero.size else 0.,
            'at_or_below_native_small_threshold': int(np.count_nonzero(nonzero <= options.small_matrix_value)),
            'at_or_above_native_large_threshold': int(np.count_nonzero(nonzero >= options.large_matrix_value))}
    exponents = np.concatenate([n['exponents'][n['needed']] for n in lifted.nodes])
    print(json.dumps({'event': 'checkpoint_audited', 'rows': identity['all_defining_rows_checked'],
        'thresholds': thresholds, 'fields': fields, 'exponent_range': [int(exponents.min()), int(exponents.max())]}), flush=True)
    model = _lower_hz_milp(lifted.hz, prune_unused=True, coalesce_rows=True,
        project_inactive_cont=False, fix_implied_binary=False)
    matrix = model.A.tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    lp = _core.HighsLp()
    lp.num_col_, lp.num_row_ = matrix.shape[1], matrix.shape[0]
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = lp.num_col_, lp.num_row_
    lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
    lp.col_cost_ = np.zeros(lp.num_col_)
    lp.col_lower_, lp.col_upper_ = model.var_lb, model.var_ub
    lp.row_lower_, lp.row_upper_ = model.row_lb, model.row_ub
    lp.integrality_ = [_core.HighsVarType(int(i)) for i in model.integrality]
    ingestion = highs.passModel(lp)
    retained_lp = highs.getLp()
    retained = sp.csc_matrix((np.asarray(retained_lp.a_matrix_.value_, dtype=np.float64),
        np.asarray(retained_lp.a_matrix_.index_, dtype=np.int32),
        np.asarray(retained_lp.a_matrix_.start_, dtype=np.int32)), shape=matrix.shape)
    retained.sort_indices()
    difference = (matrix - retained).tocsc()
    difference.eliminate_zeros()
    checks = {'matrix_all_coefficients_retained': difference.nnz == 0,
        'same_matrix_nnz': matrix.nnz == retained.nnz,
        'column_lower': np.array_equal(retained_lp.col_lower_, model.var_lb),
        'column_upper': np.array_equal(retained_lp.col_upper_, model.var_ub),
        'row_lower': np.array_equal(retained_lp.row_lower_, model.row_lb),
        'row_upper': np.array_equal(retained_lp.row_upper_, model.row_ub),
        'integrality': np.array_equal([int(v) for v in retained_lp.integrality_], model.integrality)}
    passed = all(checks.values()) and ingestion != _core.HighsStatus.kError
    record = {'schema': 'c7_checkpoint_ingestion_v1', 'formal_gain': 0, 'checkpoint_reloaded_and_audited': True,
        'identity': identity, 'fields': fields, 'auxiliary_exponent_min': int(exponents.min()),
        'auxiliary_exponent_max': int(exponents.max()), 'scipy_version': scipy.__version__,
        'native_highs_version': highs.version(), 'native_thresholds': thresholds,
        'options_status': str(high_options_status), 'ingestion_status': str(ingestion),
        'checks': checks, 'input_matrix_nnz': matrix.nnz, 'retained_matrix_nnz': retained.nnz,
        'different_coefficients': difference.nnz,
        'small_nonzero_input_coefficients': int(np.count_nonzero((np.abs(matrix.data) <= options.small_matrix_value) & (matrix.data != 0.))),
        'lowered_n_cont': model.n_cont, 'lowered_n_bin': model.n_bin,
        'passed': passed, 'status': 'INGESTION_PRESERVED' if passed else 'INGESTION_CHANGED_PROBLEM',
        'solve_called': False, 'presolve_called': False, 'terminal_result': None,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'provenance': frozen['provenance'], 'source_sha256': frozen['source_sha256'],
        'checkpoint_sha256': _sha256(checkpoint)}
    _atomic_exclusive_json(DIRECTORY / 'result.json', record)
    print(json.dumps({k:record[k] for k in ('status', 'input_matrix_nnz', 'retained_matrix_nnz', 'different_coefficients', 'solve_called')}), flush=True)


def main():
    import scipy.optimize._highspy._core as core
    import scipy.optimize._highspy._highs_wrapper as wrapper
    if DIRECTORY.exists():
        raise FileExistsError(DIRECTORY)
    prior = json.loads((PREVIOUS / 'preregistered.json').read_text())
    hashes = prior['source_sha256']
    if any(_sha256(EXP / name) != sha for name, sha in hashes.items()):
        raise ValueError('prior source drift')
    for path in (Path(__file__), EXP / 'C7_CHECKPOINT_INGESTION_PREREG_20260905.md',
                 PREVIOUS / 'result.json', PREVIOUS / 'exit.json', Path(core.__file__), Path(wrapper.__file__)):
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
