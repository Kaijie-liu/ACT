"""Exact sparse elimination with incidence-driven pivoting and bounded fill-in.

No optimization or mathematical acceptance occurs here. All original LP
constraints must still be checked by the frozen standalone rational checker.
"""
from fractions import Fraction as F
import heapq
import math
import time
from lp_sandwich.check import identity, rational, matrix_rows, validate_statement

POLICY = {'variables': 16384, 'equations': 16384, 'input_nnz': 1000000,
          'live_nnz': 2000000, 'fill_insertions': 1000000, 'heap_entries': 200000,
          'max_bits': 4096, 'operations': 20000000, 'attempts': 1,
          'pivot': 'minimum_active_column_degree_then_shortest_incident_row_then_index'}
COORDINATES = 'ORIGINAL_X_PLUS_POSITIVE_E_AND_A_RESIDUALS_V1'


class Limit(Exception):
    pass


class Singular(Exception):
    pass


class Budget:
    def __init__(self, deadline):
        now = time.monotonic()
        if not math.isfinite(deadline) or deadline > now + 300:
            raise ValueError('bounded original deadline required')
        self.deadline, self.operations = deadline, 0

    def tick(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('original deadline')

    def visit(self):
        self.tick()
        self.operations += 1
        if self.operations > POLICY['operations']:
            raise Limit('operation budget')

    def value(self, x):
        self.visit()
        v = x if isinstance(x, F) else rational(x)
        if max(abs(v.numerator).bit_length(), v.denominator.bit_length()) > POLICY['max_bits']:
            raise Limit('rational bit budget')
        return v


def dimensions(lp):
    n, ne, na = len(lp['c']), len(lp['h']), len(lp['b'])
    if n > POLICY['variables'] or ne + na > POLICY['equations'] or sum(len(lp[k]['data']) for k in ('E', 'A')) > POLICY['input_nnz']:
        raise Limit('large-interface input cap')
    if lp['matrix_format'] != 'csr_v1' or len(lp['lower']) != n or len(lp['upper']) != n:
        raise ValueError('original LP dimensions')
    for k, count in (('E', ne), ('A', na)):
        if lp[k]['shape'] != [count, n]:
            raise ValueError('original matrix dimensions')
    return n, ne, na


def parsed_rows(lp, budget):
    n, ne, na = dimensions(lp)
    for kind, rhs_name in (('E', 'h'), ('A', 'b')):
        for i, entries in enumerate(matrix_rows(lp[kind], n, budget.tick)):
            entries = [(j, budget.value(v)) for j, v in entries]
            rhs = budget.value(lp[rhs_name][i])
            yield kind, i, entries, rhs


def row_identity(kind, index, entries, rhs):
    return {'kind': kind, 'index': index,
            'row_sha256': identity({'entries': [[j, str(v)] for j, v in entries], 'rhs': str(rhs)})}


def manifest(lp, statement, candidate, basic_columns, anchors, *, deadline):
    b = Budget(deadline)
    b.tick()
    return {'schema': 'LARGE_ORIGINAL_BASIS_V1', 'coordinates': COORDINATES,
            'lp_sha256': identity(lp), 'statement_sha256': identity(statement),
            'candidate_sha256': identity(candidate),
            'rows': [row_identity(k, i, entries, rhs) for k, i, entries, rhs in parsed_rows(lp, b)],
            'basic_columns': basic_columns, 'anchors': anchors}


def key(col, n, ne, na):
    if set(col) != {'kind', 'index'} or type(col['index']) is not int:
        raise ValueError('coordinate schema')
    kind, i = col['kind'], col['index']
    sizes = {'x': n, 'E_residual': ne, 'A_slack': na}
    if kind not in sizes or not 0 <= i < sizes[kind]:
        raise ValueError('original coordinate')
    return kind, i


def eliminate(system, budget, stats):
    """Column incidence avoids scanning every previous pivot for each row.

Column-singleton pivots require no other row updates. General pivots update
only incident rows. A bounded lazy heap is compacted from active columns.
The algorithm never constructs a dense m×m array or prunes nonzero rationals.
"""
    m = len(system)
    rows = {i: r for i, (r, _) in enumerate(system)}
    right = {i: v for i, (_, v) in enumerate(system)}
    system.clear()  # Consume newly assembled rows; do not retain a duplicate matrix.
    incidence = {j: set() for j in range(m)}
    for i, row in rows.items():
        budget.visit()
        for j in row:
            if not 0 <= j < m:
                raise ValueError('basis column range')
            incidence[j].add(i)
    active_nnz = sum(map(len, rows.values()))
    pivot_nnz = 0
    heap = [(len(ids), j) for j, ids in incidence.items()]
    heapq.heapify(heap)
    pivots = []

    def observe():
        budget.tick()
        stats['peak_active_nnz'] = max(stats['peak_active_nnz'], active_nnz)
        stats['peak_live_nnz'] = max(stats['peak_live_nnz'], active_nnz + pivot_nnz)
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'], len(heap))
        if active_nnz + pivot_nnz > POLICY['live_nnz']:
            raise Limit('live sparse entries')
        if stats['fill_insertions'] > POLICY['fill_insertions']:
            raise Limit('fill-in insertions')
        if len(heap) > POLICY['heap_entries']:
            raise Limit('pivot heap budget')

    observe()
    while rows:
        budget.visit()
        while heap:
            degree, col = heapq.heappop(heap)
            stats['heap_pops'] += 1
            budget.visit()
            if col in incidence and degree == len(incidence[col]):
                break
        else:
            raise Singular('no active pivot column')
        if not degree:
            raise Singular('rank deficient original selected basis')
        # This is a declared heuristic, not global Markowitz minimization.
        stats['pivot_row_candidates'] += degree
        pivot_id = min(incidence[col], key=lambda i: (len(rows[i]), i))
        pivot = rows.pop(pivot_id)
        rhs = right.pop(pivot_id)
        active_nnz -= len(pivot)
        scale = pivot[col]
        normalized = {j: budget.value(v / scale) for j, v in pivot.items()}
        target_rhs = budget.value(rhs / scale)
        pivot_nnz += len(normalized)
        pivots.append((col, normalized, target_rhs))
        stats['pivots'] += 1
        if degree == 1:
            stats['singleton_column_pivots'] += 1
        affected = sorted(incidence[col] - {pivot_id})
        changed = set(pivot)
        for j in pivot:
            incidence[j].remove(pivot_id)
        del incidence[col]
        for i in affected:
            budget.visit()
            stats['row_updates'] += 1
            row = rows[i]
            factor = row.pop(col)
            active_nnz -= 1
            right[i] = budget.value(right[i] - factor * target_rhs)
            for j, v in normalized.items():
                if j == col:
                    continue
                existed = j in row
                value = budget.value(row.get(j, F(0)) - factor * v)
                if value:
                    row[j] = value
                    if not existed:
                        active_nnz += 1
                        stats['fill_insertions'] += 1
                        incidence[j].add(i)
                elif existed:
                    del row[j]
                    active_nnz -= 1
                    incidence[j].remove(i)
                changed.add(j)
                observe()
            if not row:
                raise Singular('dependent/inconsistent selected square basis; no LP verdict')
        for j in changed:
            if j in incidence:
                heapq.heappush(heap, (len(incidence[j]), j))
        # Compact before growth can become quadratic from stale entries.
        if len(heap) > max(1024, 4 * len(incidence)):
            heap = [(len(ids), j) for j, ids in incidence.items()]
            heapq.heapify(heap)
            stats['heap_rebuilds'] += 1
        observe()
    values = [F(0)] * m
    for col, row, rhs in reversed(pivots):
        for j, v in row.items():
            if j != col:
                rhs = budget.value(rhs - v * values[j])
        values[col] = rhs
    return values


def propose(lp, statement, candidate, hint, expected_statement, expected_hint, *, deadline):
    begin = time.monotonic()
    b = Budget(deadline)
    stats = dict.fromkeys(('peak_active_nnz', 'peak_live_nnz', 'peak_heap_entries', 'fill_insertions',
                          'heap_pops', 'heap_rebuilds', 'pivots', 'pivot_row_candidates',
                          'singleton_column_pivots', 'row_updates'), 0)
    status, error, bundle, residuals, system_hash = 'ERROR', None, None, None, None
    try:
        b.tick()
        n, ne, na = dimensions(lp)
        validate_statement(statement, lp, expected_statement, b.tick)
        for name in ('c', 'lower', 'upper'):
            for v in lp[name]:
                b.value(v)
        b.value(lp['offset'])
        if any(b.value(l) > b.value(u) for l, u in zip(lp['lower'], lp['upper'])):
            raise ValueError('inverted bounds')
        if (set(candidate) != {'lp_sha256', 'statement_sha256', 'x'} or
                candidate['lp_sha256'] != identity(lp) or candidate['statement_sha256'] != expected_statement or len(candidate['x']) != n):
            raise ValueError('candidate identity')
        point = [b.value(x) for x in candidate['x']]
        fields = {'schema', 'coordinates', 'lp_sha256', 'statement_sha256', 'candidate_sha256', 'rows', 'basic_columns', 'anchors'}
        if (set(hint) != fields or hint['schema'] != 'LARGE_ORIGINAL_BASIS_V1' or hint['coordinates'] != COORDINATES or
                identity(hint) != expected_hint or hint['lp_sha256'] != identity(lp) or
                hint['statement_sha256'] != expected_statement or hint['candidate_sha256'] != identity(candidate)):
            raise ValueError('basis binding/coordinates')
        basic = [key(c, n, ne, na) for c in hint['basic_columns']]
        if len(basic) != ne + na or len(set(basic)) != len(basic):
            raise ValueError('square basis count/uniqueness')
        lookup = {k: j for j, k in enumerate(basic)}
        anchors = {}
        for item in hint['anchors']:
            if set(item) != {'column', 'at'}:
                raise ValueError('anchor schema')
            k = key(item['column'], n, ne, na)
            if k in anchors or k in lookup:
                raise ValueError('duplicate/overlapping coordinate')
            at = item['at']
            if k[0] != 'x':
                if at != 'zero':
                    raise ValueError('nonbasic row residual must be anchored zero')
                value = F(0)
            else:
                if at not in ('lower', 'upper', 'candidate'):
                    raise ValueError('unbound anchor')
                value = point[k[1]] if at == 'candidate' else b.value(lp[at][k[1]])
            anchors[k] = value
        allcols = {('x', i) for i in range(n)} | {('E_residual', i) for i in range(ne)} | {('A_slack', i) for i in range(na)}
        if set(basic) | set(anchors) != allcols:
            raise ValueError('missing original coordinate')
        # All original rows remain present, including redundant equalities.
        expected_rows, system = [], []
        for kind, i, entries, rhs in parsed_rows(lp, b):
            expected_rows.append(row_identity(kind, i, entries, rhs))
            coeff = {('x', j): v for j, v in entries if v}
            coeff[('E_residual' if kind == 'E' else 'A_slack', i)] = F(1)
            reduced = {}
            for k, v in coeff.items():
                if k in lookup:
                    reduced[lookup[k]] = v
                else:
                    rhs = b.value(rhs - v * anchors[k])
            system.append((reduced, rhs))
        if hint['rows'] != expected_rows:
            raise ValueError('complete original row identity/order')
        system_hash = identity([{'row': [[j, str(v)] for j, v in sorted(row.items())], 'rhs': str(rhs)} for row, rhs in system])
        vals = eliminate(system, b, stats)
        values = {**anchors, **dict(zip(basic, vals))}
        x = [values['x', i] for i in range(n)]
        residuals = {'E_fixed_zero_required': [str(values['E_residual', i]) for i in range(ne)],
                     'A_nonnegative_required': [str(values['A_slack', i]) for i in range(na)]}
        objective = b.value(lp['offset'])
        for coeff, v in zip(lp['c'], x):
            objective = b.value(objective + b.value(coeff) * v)
        bundle = {'schema': 'LP_SANDWICH_V1', 'lp': lp, 'statement': statement, 'dual': None,
                  'primal': {'lp_sha256': identity(lp), 'statement_sha256': expected_statement,
                             'x': [str(v) for v in x], 'claimed_objective': str(objective)}}
        b.tick()
        status = 'CANDIDATE_ONLY'
    except Limit as exc:
        status, error = 'LIMIT', str(exc)
    except Singular as exc:
        status, error = 'UNRESOLVED_SINGULAR_BASIS', str(exc)
    except TimeoutError as exc:
        status, error = 'TIMEOUT', str(exc)
    except Exception as exc:
        error = repr(exc)
    if status != 'CANDIDATE_ONLY':
        bundle, residuals = None, None
    return {'schema': 'SPARSE_BASIS_PROPOSAL_V1', 'status': status, 'error': error,
            'bundle': bundle, 'row_residuals': residuals, 'stats': stats, 'operations': b.operations,
            'policy': dict(POLICY), 'seconds': time.monotonic() - begin, 'deadline_monotonic': deadline,
            'statement_sha256': expected_statement, 'hint_sha256': expected_hint,
            'assembled_system_sha256': system_hash, 'attempts': 1, 'solver_calls': 0,
            'feasibility_certified': False, 'network_SAFE': False, 'network_UNSAFE': False}
