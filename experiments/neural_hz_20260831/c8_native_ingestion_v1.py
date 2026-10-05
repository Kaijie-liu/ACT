"""Read-only model fidelity check using ACT's installed native SciPy backend."""

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize._highspy import _core

from act.back_end.solver.solver_hz import _lower_hz_milp
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def inspect(hz):
    before = source_digest(hz)
    highs = _core._Highs()
    options = _core.HighsOptions()
    options.presolve = 'on'
    options.time_limit = 45.
    options.mip_rel_gap = 0.
    options_status = highs.passOptions(options)
    fields = {}
    for name in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
        data = getattr(hz, name).data
        nonzero = np.abs(data[data != 0.])
        fields[name] = {'stored': int(data.size), 'nonzero': int(nonzero.size),
            'min_nonzero_abs': float(nonzero.min()) if nonzero.size else None,
            'max_abs': float(nonzero.max()) if nonzero.size else 0.,
            'at_or_below_native_small_threshold': int(np.count_nonzero(nonzero <= options.small_matrix_value)),
            'at_or_above_native_large_threshold': int(np.count_nonzero(nonzero >= options.large_matrix_value))}
    model = _lower_hz_milp(hz, prune_unused=True, coalesce_rows=True,
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
        'integrality': np.array_equal([int(v) for v in retained_lp.integrality_], model.integrality),
        'input_hz_unchanged': before == source_digest(hz)}
    passed = all(checks.values()) and ingestion != _core.HighsStatus.kError and options_status != _core.HighsStatus.kError
    return {'fields': fields, 'scipy_version': scipy.__version__, 'native_highs_version': highs.version(),
        'native_thresholds': {'small_matrix_value': options.small_matrix_value, 'large_matrix_value': options.large_matrix_value},
        'options_status': str(options_status), 'ingestion_status': str(ingestion), 'checks': checks,
        'input_matrix_nnz': matrix.nnz, 'retained_matrix_nnz': retained.nnz, 'different_coefficients': difference.nnz,
        'small_nonzero_input_coefficients': int(np.count_nonzero((np.abs(matrix.data) <= options.small_matrix_value) & (matrix.data != 0.))),
        'lowered_n_cont': model.n_cont, 'lowered_n_bin': model.n_bin, 'passed': passed,
        'status': 'INGESTION_PRESERVED' if passed else 'INGESTION_CHANGED_PROBLEM',
        'solve_called': False, 'presolve_called': False, 'terminal_result': None}
