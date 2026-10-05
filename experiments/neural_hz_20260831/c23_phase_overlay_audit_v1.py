"""Independent full-incidence proof and complete unit-pair discovery for C23."""

from fractions import Fraction as F
import math
import numpy as np

from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, UID_LIMIT, unique_other
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload


class BranchPool:
    """Charge all overlay operations to BOTH whole diagnostic and branch caps."""
    def __init__(self, whole, cap=200_000_000):
        if type(cap) is not int or not 0 <= cap <= 200_000_000:
            raise ValueError('invalid/increased branch cap')
        self.whole, self.cap, self.used, self.parts = whole, cap, 0, {}

    def charge(self, name, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError('invalid branch charge')
        if self.used + amount > self.cap:
            raise MemoryError('overlay branch work exhausted before operation')
        self.whole.charge(name, amount)
        self.used += amount
        self.parts[name] = self.parts.get(name, 0) + amount


def check_append(pre, post):
    """Authentication guard, not an inference from exact=True or an LP result."""
    if (post.frame_id != pre.frame_id or post.n_cont < pre.n_cont
            or post.n_bin < pre.n_bin or not post.exact):
        raise ValueError('phase oracle changed shared frame or exactness')
    for name in ('Ac', 'Ab', 'Auc', 'Aub'):
        old, new = getattr(pre, name), getattr(post, name)
        if (new.shape[0] < old.shape[0]
                or not equal_payload(new[:old.shape[0], :old.shape[1]], old)
                or new[:old.shape[0], old.shape[1]:].nnz):
            raise ValueError('phase oracle changed exact predicate prefix')
    if not equal_payload(post.b[:pre.n_eq], pre.b) or not equal_payload(post.ub[:pre.n_ineq], pre.ub):
        raise ValueError('phase oracle changed original RHS')


def incidence_oracle(hz, eq_uids, le_uids, old_nc, logical_nc, *, pool):
    """Full actual canonical incidence, independent of sparse append events.

Every nonzero continuous predicate coefficient contributes once; sign and
binary terms do not change continuous incidence. Full-width temporary words
avoid filtering/copying coefficient blocks for each MAIN prefix. At most 2^20
unique UIDs imply count<=2^20, sum<2^40 and no signed-int64 overflow.
    """
    if not 0 <= old_nc <= logical_nc <= hz.n_cont <= 64_000_000:
        raise ValueError('invalid oracle global frame')
    nnz = int(hz.Ac.nnz + hz.Auc.nnz)
    rows = hz.n_eq + hz.n_ineq
    if nnz > 64_000_000 or rows > UID_LIMIT:
        raise MemoryError('unchanged oracle entry/UID ceiling exceeded')
    # finite/all + zero/any =4/nnz; indexed gather/add/store=3/nnz.
    # Row UID uniqueness/domain/pointers and scalar amount=12/row;
    # full allocation and final MAIN range/copy are bounded by4/global slot.
    pool.charge('independent_complete_incidence', 7 * nnz + 12 * rows + 4 * hz.n_cont)
    words = np.zeros(hz.n_cont, np.int64)
    seen = set()
    for matrix, uids in ((hz.Ac, eq_uids), (hz.Auc, le_uids)):
        if (len(uids) != matrix.shape[0] or not matrix.has_canonical_format
                or not np.isfinite(matrix.data).all() or np.any(matrix.data == 0.)):
            raise ValueError('oracle requires complete canonical finite nonzero incidence')
        for row, raw in enumerate(uids):
            uid = int(raw)
            if not 0 <= uid < UID_LIMIT or uid in seen:
                raise ValueError('oracle row UID is invalid or reused')
            seen.add(uid)
            a, b = map(int, matrix.indptr[row:row + 2])
            cols = matrix.indices[a:b]
            words[cols] += RADIX + uid
    return words[old_nc:logical_nc].copy()


def verify_all_and_discover(candidate, post, overlay, actual, eq, le, *, whole, branch):
    """One complete stream verifies every word AND discovers all unit pairs.

Consumer column must be FIRST for the C15 splice shape. Testing that necessary
condition directly avoids searching a long consumer row merely to reject it.
The caller binds the complete C10 source/quotient/MAIN-box proof beforehand.
No sealed table is accepted as a selection input.
    """
    main = candidate.logical_n_cont - candidate.old_n_cont
    if len(actual) != main or len(overlay.base) != main:
        raise ValueError('incomplete MAIN proof population')
    whole.charge('complete_UID_lookup', 8 * (len(eq) + len(le)))
    lookup = {int(uid): (False, row) for row, uid in enumerate(eq)}
    lookup.update({int(uid): (True, row) for row, uid in enumerate(le)})
    if len(lookup) != len(eq) + len(le):
        raise ValueError('duplicate physical UID in discovery')
    whole.charge('output_liveness', 4 * int(post.Gc.nnz) + 2 * post.n_cont)
    output_live = np.bincount(post.Gc.indices, minlength=post.n_cont) != 0
    whole.charge('complete_MAIN_equality_and_unit_metadata', 32 * main)
    accepted, candidates, touched = [], 0, 0
    for i, packed in enumerate(overlay.iter_words(pool=branch)):
        if packed != int(actual[i]):
            raise ValueError(f'sparse overlay differs from complete actual incidence at MAIN {i}')
        touched += packed != int(overlay.base[i])
        col = candidate.old_n_cont + i
        d = int(candidate.eq_roots[candidate.old_n_eq + i])
        if d < 0 or output_live[col] or packed // RADIX != 2:
            continue
        a, b = map(int, post.Ac.indptr[d:d + 2])
        if a == b or int(post.Ac.indices[b - 1]) != col:
            continue
        pivot = float(post.Ac.data[b - 1])
        if pivot <= 0. or math.frexp(pivot)[0] != .5:
            continue
        candidates += 1
        whole.charge('single_consumer_shape', 32)
        other = unique_other(packed, int(eq[d]))
        if other not in lookup:
            raise ValueError('unique consumer UID does not resolve')
        inequality, row = lookup[other]
        matrix, rhs = (post.Auc, post.ub) if inequality else (post.Ac, post.b)
        start, stop = map(int, matrix.indptr[row:row + 2])
        if (start == stop or int(matrix.indices[start]) != col
                or abs(float(matrix.data[start])) != pivot
                or post.Ab.indptr[d + 1] != post.Ab.indptr[d]):
            continue
        whole.charge('unit_pair_exact_RHS', 64)
        sign = -1 if matrix.data[start] > 0. else 1
        updated = float(rhs[row]) + sign * float(post.b[d])
        if not math.isfinite(updated) or F(updated) != F(float(rhs[row])) + sign * F(float(post.b[d])):
            raise ValueError('unit RHS update is not exact')
        accepted.append(col)
    whole.charge('accepted_unit_column_array', len(accepted))
    columns = np.asarray(accepted, np.int64)
    return columns, {'all_post_MAIN_columns_checked': main,
        'complete_post_incidence_equal': True, 'MAIN_columns_with_new_incidence': touched,
        'structural_single_consumer_definitions': candidates,
        'all_unit_pairs_discovered': len(columns), 'diagnostic_UID_lookup_entries': len(lookup),
        'independent_dense_oracle_retained_after_proof': False,
        'new_phase_executed': False, 'unit_splice_executed': False, 'formal_gain': 0}
