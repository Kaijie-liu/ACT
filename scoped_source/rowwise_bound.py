"""Opt-in exact finite-box LP checker, one CSR row at a time.

Given original JSON LP/certificate only; no source lowering, feasibility or
whole-request claim. All rows, including zero-dual rows, must be consumed.
The caller owns immutable inputs. Rehashing detects ordinary callback/alias
pollution, not arbitrary concurrent mutate-and-restore attacks.
Deadlines are cooperative: an outer supervisor is still required.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import time


def identity(value):
    # Unchanged canonical identity, NOT a streaming-serialization optimization.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def clock(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline-time.monotonic() > 300:
        raise ValueError('one at-most-300s deadline required')
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('rowwise exact check deadline')
    tick(); return tick


def rational(value):
    if type(value) is int:
        return F(value)
    if type(value) is float and math.isfinite(value):
        return F.from_float(value)
    if type(value) is str:
        try:
            return F(value)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError('finite JSON rational required')


def rows(matrix, expected, tick):
    """Exhaust to validate. At most one new row is built per next() call."""
    if type(matrix) is not dict or not {'shape','data','indices','indptr'} <= set(matrix):
        raise ValueError('CSR fields')
    shape, data, ids, ptr = (matrix[k] for k in ('shape','data','indices','indptr'))
    if (any(type(v) is not list for v in (shape, data, ids, ptr)) or len(shape) != 2
            or any(type(v) is not int or v < 0 for v in shape) or shape != list(expected)
            or len(ptr) != shape[0]+1 or ptr[0] != 0 or ptr[-1] != len(data) or len(ids) != len(data)):
        raise ValueError('CSR dimensions/pointers')
    for i, pointer in enumerate(ptr):
        if i % 256 == 0: tick()
        if type(pointer) is not int or not 0 <= pointer <= len(data):
            raise ValueError('CSR pointer type/range')
    for i in range(shape[0]):
        tick(); a, b = ptr[i:i+2]
        if a > b:
            raise ValueError('CSR pointer order')
        row = []; previous = -1
        for k in range(a, b):
            if k % 256 == 0: tick()
            column = ids[k]
            if type(column) is not int or not previous < column < shape[1]:
                raise ValueError('CSR column type/order/range')
            previous = column
            row.append((column, rational(data[k])))  # parse even explicit zero/zero dual
        tick(); yield row
    tick()


def check_bound(lp, certificate, *, deadline):
    """Same exact bound as the legacy source checker on its JSON LP domain."""
    tick = clock(deadline)
    if type(lp) is not dict or type(certificate) is not dict:
        raise ValueError('LP/certificate object')
    if not {'matrix_format','c','offset','lower','upper','A','b','E','h'} <= set(lp):
        raise ValueError('complete LP fields including offset required')
    if not {'lp_sha256','inequality_dual','equality_dual','claimed_lower_bound'} <= set(certificate):
        raise ValueError('complete bound certificate required')
    source_hash = identity(lp); cert_hash = identity(certificate); tick()
    if lp['matrix_format'] != 'csr_v1' or certificate['lp_sha256'] != source_hash:
        raise ValueError('original LP identity/format')
    for k in ('c','lower','upper','b','h'):
        if type(lp[k]) is not list: raise ValueError('JSON vector required')
    n = len(lp['c'])
    if not n or len(lp['lower']) != n or len(lp['upper']) != n:
        raise ValueError('finite box shape')
    c, low, high = [], [], []
    for j in range(n):
        if j % 256 == 0: tick()
        c.append(rational(lp['c'][j])); low.append(rational(lp['lower'][j])); high.append(rational(lp['upper'][j]))
        if low[-1] > high[-1]: raise ValueError('finite box order')
    residual = c[:]; value = rational(lp['offset']); row_count = entries = zero_dual = 0; max_row = 0
    for matrix, rhs, key, signed in (('A','b','inequality_dual',True), ('E','h','equality_dual',False)):
        multipliers = certificate[key]
        if type(multipliers) is not list or len(multipliers) != len(lp[rhs]):
            raise ValueError('dual/RHS coverage')
        for i, coefficients in enumerate(rows(lp[matrix], (len(lp[rhs]), n), tick)):
            multiplier = rational(multipliers[i])
            if signed and multiplier > 0: raise ValueError('inequality dual sign')
            value += multiplier*rational(lp[rhs][i]); row_count += 1
            entries += len(coefficients); max_row = max(max_row, len(coefficients)); zero_dual += multiplier == 0
            for p, (j, coefficient) in enumerate(coefficients):
                if p % 256 == 0: tick()
                residual[j] -= multiplier*coefficient
            del coefficients  # caller releases it; enumerate can briefly retain the previous row
    for j, r in enumerate(residual):
        if j % 256 == 0: tick()
        value += min(r*low[j], r*high[j])
    if rational(certificate['claimed_lower_bound']) > value:
        raise ValueError('claim exceeds exact residual-compensated bound')
    tick()
    if identity(lp) != source_hash or identity(certificate) != cert_hash:
        raise ValueError('LP/certificate changed during check')
    result = {'checked_lower_bound':str(value), 'residual':list(map(str,residual)),
            'lp_sha256':source_hash, 'rows_checked':row_count, 'entries_checked':entries,
            'zero_dual_rows_checked':zero_dual, 'maximum_row_entries':max_row,
            'scope':'GIVEN_ORIGINAL_FINITE_BOX_LP_ONLY', 'hard_budget_supervision':False}
    tick(); return result
