"""Separate large-interface native capture and explicit equality residual map.

No old capture protocol or limits are patched. Native arithmetic only proposes
a basis. Full original LP checking remains mandatory after reconstruction.
"""
import math
from fractions import Fraction
from pathlib import Path
import time
from hashlib import sha256
from single_check_portable.execution import save_new
from lp_sandwich.check import identity, inspect_bounds, validate_statement, rational
from sparse_basis.engine import Budget, POLICY, dimensions, parsed_rows, manifest, Limit

VERSION = '1.14.0'
OPTIONS = {'presolve': 'off', 'solver': 'simplex', 'simplex_scale_strategy': 0,
           'threads': 1, 'parallel': 'off', 'random_seed': 0, 'output_flag': True, 'log_to_console': True,
           'small_matrix_value': 1e-12, 'large_matrix_value': 1e15,
           'infinite_cost': 1e20, 'infinite_bound': 1e20}
CAPTURES = []


def preflight(expected, tick):
    """Reject known import filtering/infinity conversions, never drop a coefficient."""
    values=[]
    for row in expected['rows']:
        tick()
        for _, value in row['entries']:
            if not math.isfinite(value) or abs(value)<=OPTIONS['small_matrix_value']:
                raise Limit('fidelity: coefficient at/below fixed native filtering threshold')
            if abs(value)>=OPTIONS['large_matrix_value']:
                raise Limit('fidelity: large matrix coefficient')
            values.append(abs(value))
        if not math.isfinite(row['upper']) or abs(row['upper'])>=OPTIONS['infinite_bound']:
            raise Limit('fidelity: row bound would become infinite')
        if row['lower']!='-inf' and (not math.isfinite(row['lower']) or abs(row['lower'])>=OPTIONS['infinite_bound']):
            raise Limit('fidelity: row lower bound would become infinite')
    for key,cap in (('cost','infinite_cost'),('lower','infinite_bound'),('upper','infinite_bound')):
        for value in expected[key]:
            tick()
            if not math.isfinite(value) or abs(value)>=OPTIONS[cap]:
                raise Limit('fidelity: nonfinite/too-large '+key)
    return {'matrix_entries':len(values),'min_abs_nonzero':min(values) if values else None,
            'max_abs':max(values) if values else None,'small_matrix_value':OPTIONS['small_matrix_value'],
            'no_scaling':True,'no_pruning':True}


def floating(v):
    out = float(v if isinstance(v, Fraction) else rational(v))
    if not math.isfinite(out):
        raise ValueError('native float conversion overflow')
    return out


def snapshot_input(lp, b):
    n, ne, na = dimensions(lp)
    rows = []
    for kind, i, entries, rhs in parsed_rows(lp, b):
        rows.append({'lower': floating(rhs) if kind == 'E' else '-inf',
                     'upper': floating(rhs), 'entries': [[j, floating(v)] for j, v in entries if v]})
    return {'cost': [floating(b.value(v)) for v in lp['c']], 'offset': floating(b.value(lp['offset'])),
            'lower': [floating(b.value(v)) for v in lp['lower']],
            'upper': [floating(b.value(v)) for v in lp['upper']], 'rows': rows, 'sense': 'minimize'}


def model_snapshot(h, highspy, b):
    """Linear-time sparse model readback, rather than one native row scan per row."""
    model = h.getLp()
    n, m = model.num_col_, model.num_row_
    matrix = model.a_matrix_
    rows = [[] for _ in range(m)]
    if matrix.format_ == highspy.MatrixFormat.kColwise:
        outer, colwise = n, True
    elif matrix.format_ == highspy.MatrixFormat.kRowwise:
        outer, colwise = m, False
    else:
        raise ValueError('unsupported native readback storage')
    ptr, ids, data = matrix.start_, matrix.index_, matrix.value_
    if len(ptr) != outer + 1 or ptr[0] != 0 or ptr[-1] != len(ids) or len(ids) != len(data):
        raise ValueError('native sparse readback dimensions')
    for j in range(outer):
        b.tick()
        if not 0 <= ptr[j] <= ptr[j+1] <= len(data):
            raise ValueError('native sparse pointer')
        for k in range(ptr[j], ptr[j+1]):
            if k % 256 == 0:
                b.tick()
            row, col = (int(ids[k]), j) if colwise else (j, int(ids[k]))
            if not 0 <= row < m or not 0 <= col < n:
                raise ValueError('native sparse index')
            value = float(data[k])
            if value:
                rows[row].append([col, value])
    outrows = []
    for i, entries in enumerate(rows):
        b.tick()
        lower = float(model.row_lower_[i])
        outrows.append({'lower': '-inf' if lower == -math.inf else lower,
                        'upper': float(model.row_upper_[i]), 'entries': sorted(entries)})
    return {'cost': list(map(float, model.col_cost_)), 'offset': float(model.offset_),
            'lower': list(map(float, model.col_lower_)), 'upper': list(map(float, model.col_upper_)),
            'rows': outrows, 'sense': 'minimize' if model.sense_ == highspy.ObjSense.kMinimize else 'unsupported'}


def capture(lp, statement, destination, *, deadline):
    begin = time.monotonic()
    b = Budget(deadline)
    b.tick()
    n, ne, na = dimensions(lp)
    validate_statement(statement, lp, identity(statement), b.tick)
    inspect_bounds(lp, None, None, identity(statement), b.tick)
    root = Path(destination).resolve()
    if not root.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('outside workspace')
    root.mkdir(exist_ok=False)
    expected = snapshot_input(lp, b)
    save_new(root / 'input.json', {'lp': lp, 'statement': statement, 'submitted': expected})
    pre=preflight(expected,b.tick)
    save_new(root / 'preflight.json',pre)
    import highspy
    import highspy._core as core
    h = highspy.Highs()
    if h.version() != VERSION:
        raise ValueError('untested native version')
    def ok(s):
        if s != highspy.HighsStatus.kOk:
            raise ValueError('native API status: ' + str(s))
    for name, value in OPTIONS.items():
        ok(h.setOptionValue(name, value))
    configured={}
    for name,value in OPTIONS.items():
        status,actual=h.getOptionValue(name);ok(status)
        if actual!=value:raise ValueError('native option readback drift')
        configured[name]=actual
    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = n, ne + na
    model.col_cost_, model.col_lower_, model.col_upper_ = expected['cost'], expected['lower'], expected['upper']
    model.offset_, model.sense_ = expected['offset'], highspy.ObjSense.kMinimize
    model.row_lower_ = [-highspy.kHighsInf if r['lower'] == '-inf' else r['lower'] for r in expected['rows']]
    model.row_upper_ = [r['upper'] for r in expected['rows']]
    ptr, indices, vals = [0], [], []
    for row in expected['rows']:
        b.tick()
        for j, v in row['entries']:
            indices.append(j)
            vals.append(v)
        ptr.append(len(vals))
    model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = ptr, indices, vals
    import_start=time.monotonic()
    imported=h.passModel(model)
    save_new(root / 'import_status.json', {'status':str(imported),'options':configured,
              'submitted_sha256':identity(expected),'deadline_monotonic':deadline})
    before = model_snapshot(h, highspy, b)
    save_new(root / 'import.json', {'status':str(imported),'readback':before,'options':configured,
             'submitted_sha256':identity(expected),'seconds':time.monotonic()-import_start})
    ok(imported)
    if before!=expected:raise ValueError('native import changed intended coefficients/constraints/bounds')
    grant = min(10., deadline - time.monotonic())
    if grant <= 0:
        raise TimeoutError('no native call budget')
    ok(h.setOptionValue('time_limit', grant))
    actual = {}
    for name in (*OPTIONS, 'time_limit'):
        status, value = h.getOptionValue(name)
        ok(status)
        actual[name] = value
    save_new(root / 'submission.json', {'expected': expected, 'readback': before, 'options': actual})
    if before != expected or any(actual[k] != v for k, v in OPTIONS.items()):
        raise ValueError('native submission drift')
    b.tick()
    started = time.monotonic()
    run_status = h.run()
    native_seconds = time.monotonic() - started
    basis, point, info = h.getBasis(), h.getSolution(), h.getInfo()
    basic_status, basic_ids = h.getBasicVariables()
    # Capture raw results before model-readback or mapping can fail/timeout.
    raw = {'run_status': str(run_status), 'model_status': str(h.getModelStatus()),
           'basis_valid': bool(basis.valid), 'value_valid': bool(point.value_valid),
           'column_status': [s.name for s in basis.col_status], 'row_status': [s.name for s in basis.row_status],
           'column_values': list(map(float, point.col_value)), 'row_values': list(map(float, point.row_value)),
           'basic_variables_status': str(basic_status), 'basic_variables': list(map(int, basic_ids)),
           'native_objective': float(info.objective_function_value), 'native_seconds': native_seconds,
           'lp_sha256': identity(lp), 'statement_sha256': identity(statement)}
    save_new(root / 'raw_native.json', raw)
    record = {'schema': 'FIDELITY_NATIVE_BASIS_CAPTURE_V2', **raw,
              'input_sha256': identity({'lp': lp, 'statement': statement}),
              'native_binary_sha256': sha256(Path(core.__file__).read_bytes()).hexdigest(),
              'version': h.version(), 'options': actual, 'submitted': expected,
              'readback_before': before, 'readback_after': model_snapshot(h, highspy, b),
              'deadline_monotonic': deadline, 'seconds': time.monotonic() - begin,
              'native_calls': 1, 'policy': dict(POLICY), 'trusted': False,
              'network_SAFE': False, 'network_UNSAFE': False}
    save_new(root / 'capture.json', record)
    CAPTURES.append({'variables': n, 'rows': ne+na, 'basis_valid': raw['basis_valid'],
                     'column_status_counts': {s: raw['column_status'].count(s) for s in set(raw['column_status'])},
                     'row_status_counts': {s: raw['row_status'].count(s) for s in set(raw['row_status'])}})
    b.tick()
    return record


def map_capture(lp, statement, r, expected_capture, *, deadline):
    b = Budget(deadline)
    b.tick()
    n, ne, na = dimensions(lp)
    if (identity(r) != expected_capture or r['schema'] != 'FIDELITY_NATIVE_BASIS_CAPTURE_V2' or
            (r['lp_sha256'], r['statement_sha256'], r['input_sha256']) !=
            (identity(lp), identity(statement), identity({'lp': lp, 'statement': statement}))):
        raise ValueError('native capture identity')
    if (r['version'] != VERSION or r['policy'] != POLICY or
            set(r['options']) != set(OPTIONS) | {'time_limit'} or
            any(r['options'][k] != v for k, v in OPTIONS.items()) or not 0 < r['options']['time_limit'] <= 10):
        raise ValueError('native protocol identity')
    expected = snapshot_input(lp, b)
    preflight(expected,b.tick)
    if any(r[k] != expected for k in ('submitted', 'readback_before', 'readback_after')):
        raise ValueError('original model readback mismatch')
    def unsupported(reason):
        return {'status': 'UNSUPPORTED_MAPPING', 'reason': reason, 'hint': None,
                'capture_sha256': expected_capture, 'network_SAFE': False, 'network_UNSAFE': False}
    if not r['basis_valid'] or not r['value_valid']:
        return unsupported('no valid native basis/point')
    if len(r['column_status']) != n or len(r['row_status']) != ne+na or len(r['column_values']) != n:
        raise ValueError('native dimensions')
    if not all(math.isfinite(v) for v in r['column_values']):
        return unsupported('nonfinite native point')
    basic, anchors = [], []
    for i, status in enumerate(r['column_status']):
        col = {'kind': 'x', 'index': i}
        if status == 'kBasic':
            basic.append(col)
        elif status in ('kLower', 'kUpper'):
            anchors.append({'column': col, 'at': 'lower' if status == 'kLower' else 'upper'})
        else:
            return unsupported('unsupported structural status: ' + status)
    for i, status in enumerate(r['row_status']):
        col = {'kind': 'E_residual' if i < ne else 'A_slack', 'index': i if i < ne else i-ne}
        if status == 'kBasic':
            basic.append(col)
        elif status == 'kUpper' or (i < ne and status == 'kLower'):
            anchors.append({'column': col, 'at': 'zero'})
        else:
            return unsupported('unsupported row status: ' + status)
    if len(basic) != ne+na:
        return unsupported('basis count does not cover original rows')
    candidate = {'lp_sha256': identity(lp), 'statement_sha256': identity(statement), 'x': r['column_values']}
    h = manifest(lp, statement, candidate, basic, anchors, deadline=deadline)
    b.tick()
    return {'status': 'MAPPED_HINT_ONLY', 'hint': h, 'candidate': candidate,
            'capture_sha256': expected_capture, 'equality_residuals_must_be_zero': True,
            'network_SAFE': False, 'network_UNSAFE': False}
