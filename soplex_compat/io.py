"""Small-control exact export and strict readback; no tolerance or solver trust.

Native objective omits the exact constant offset, which cannot change its
minimizers. The original offset is retained and independently rechecked in the
LP bundle. All coefficients use exact fraction tokens, including binary floats.
"""
from fractions import Fraction as F
from lp_sandwich.check import identity, inspect_bounds, matrix_rows, rational, validate_statement


def export_lp(lp):
    inspect_bounds(lp, None, None, "")
    n = len(lp['c'])
    # This stage intentionally cannot launch the real 7k--9k-variable LPs.
    if n > 8 or len(lp['b']) + len(lp['h']) > 16:
        raise ValueError('compatibility-controls-only size limit')
    def expr(items):
        return ' '.join(('+' if v >= 0 else '-') + f' {abs(v)} x{j}'
                        for j, v in items) or '+ 0 x0'
    lines = ['Minimize', 'obj: ' + expr(enumerate(map(rational, lp['c']))), 'Subject To']
    for mat, rhs, prefix, op in [('A', 'b', 'a', '<='), ('E', 'h', 'e', '=')]:
        for i, row in enumerate(matrix_rows(lp[mat], n, lambda: None)):
            lines.append(f'{prefix}{i}: {expr(row)} {op} {rational(lp[rhs][i])}')
    lines += ['Bounds']
    for j, (lo, hi) in enumerate(zip(lp['lower'], lp['upper'])):
        lines.append(f'{rational(lo)} <= x{j} <= {rational(hi)}')
    return '\n'.join(lines + ['End', ''])


def readback(text):
    lines = iter(text.splitlines())
    if next(lines) != 'SOPLEX_RATIONAL_READBACK_V1': raise ValueError('readback header')
    def count(label):
        name, number = next(lines).split()
        if name != label: raise ValueError('readback section')
        size = int(number)
        if not 0 <= size <= 32: raise ValueError('readback count')
        return size
    columns, rows = {}, {}
    for _ in range(count('COLUMNS')):
        name, c, lo, hi = next(lines).split()
        if name in columns: raise ValueError('duplicate column')
        columns[name] = tuple(map(F, (c, lo, hi)))
    for _ in range(count('ROWS')):
        name, lo, hi, size, *entries = next(lines).split()
        if name in rows or len(entries) != 2*int(size): raise ValueError('row/nnz')
        coefficients = {}
        for j in range(0, len(entries), 2):
            col, val = entries[j:j+2]
            if col not in columns or col in coefficients: raise ValueError('row coordinate')
            coefficients[col] = F(val)
        rows[name] = (None if lo == '-inf' else F(lo), F(hi),
                      {k: v for k, v in coefficients.items() if v})
    if next(lines) != 'END' or list(lines): raise ValueError('incomplete/extra readback')
    return columns, rows


def verify_readback(lp, text):
    columns, rows = readback(text)
    expected_cols = {f'x{j}': tuple(rational(lp[k][j]) for k in ('c', 'lower', 'upper'))
                     for j in range(len(lp['c']))}
    expected_rows = {}
    for mat, rhs, prefix in [('A', 'b', 'a'), ('E', 'h', 'e')]:
        for i, row in enumerate(matrix_rows(lp[mat], len(lp['c']), lambda: None)):
            right = rational(lp[rhs][i])
            expected_rows[f'{prefix}{i}'] = (None if mat == 'A' else right, right,
                                            {f'x{j}': v for j, v in row if v})
    if columns != expected_cols or rows != expected_rows:
        raise ValueError('native rational model differs from original LP')


def parse_point(text, n):
    lines = text.splitlines()
    header = lines[0].split()
    if len(header) != 2 or header[0] != 'SOPLEX_CANDIDATE_V1': raise ValueError('candidate header')
    int(header[1])  # Recorded native status NEVER grants acceptance.
    if lines[1:] == ['NO_POINT', 'END']: return None
    if lines[1] != f'POINT {n}' or len(lines) != n + 3 or lines[-1] != 'END':
        raise ValueError('incomplete candidate')
    point = {}
    for line in lines[2:-1]:
        name, val = line.split()
        if name in point: raise ValueError('duplicate point coordinate')
        point[name] = F(val)
    if set(point) != {f'x{j}' for j in range(n)}: raise ValueError('point coordinate mapping')
    return [str(point[f'x{j}']) for j in range(n)]


def candidate_bundle(lp, statement, expected_statement, text):
    validate_statement(statement, lp, expected_statement, lambda: None)
    x = parse_point(text, len(lp['c']))
    primal = None if x is None else {
        'lp_sha256': identity(lp), 'statement_sha256': expected_statement, 'x': x,
        'claimed_objective': str(rational(lp['offset']) + sum(
            (rational(c)*F(v) for c, v in zip(lp['c'], x)), F(0)))}
    return {'schema': 'LP_SANDWICH_V1', 'statement': statement, 'lp': lp,
            'primal': primal, 'dual': None}
