"""Complete structural screen of exact shared affine definitions; no HZ mutation."""
from collections import Counter
from fractions import Fraction
import hashlib
import math
import struct
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool

DTYPE = np.dtype([('row', '<i4'), ('column', '<i4'), ('width', '<i4'),
                  ('recipe', 'S32'), ('support', 'S32')])
LOW, HIGH = 2.**-20, 2.**40


def normalized(columns, values, rhs):
    """Exact dyadic normalization; callers establish canonical column ordering."""
    pivot = float(values[-1])
    if not math.isfinite(pivot) or not LOW <= pivot <= HIGH or math.frexp(pivot)[0] != .5:
        raise ValueError('positive in-window dyadic pivot required')
    if not math.isfinite(rhs):
        raise ValueError('finite defining constant required')
    if not np.isfinite(values).all() or np.any(np.abs(values) < LOW) or np.any(np.abs(values) > HIGH):
        raise ValueError('unchanged coefficient window required')
    power = math.frexp(pivot)[1] - 1
    body = np.ldexp(values[:-1], -power)
    constant = math.ldexp(rhs, -power)
    if (not math.isfinite(constant) or math.ldexp(constant, power) != rhs
            or not np.array_equal(np.ldexp(body, power), values[:-1])):
        raise ValueError('power-of-two normalization not exactly reversible')
    constant = 0. if constant == 0 else constant
    support = hashlib.sha256(memoryview(columns[:-1]).cast('B')).digest()
    recipe = hashlib.sha256(support)
    recipe.update(memoryview(body).cast('B'))
    recipe.update(struct.pack('<d', constant))
    return recipe.digest(), support


def independently_equal(ac, rhs, left, right, pool):
    """Prove equality from original rows, not the normalization or digest."""
    a, b = int(ac.indptr[left]), int(ac.indptr[left+1])
    c, d = int(ac.indptr[right]), int(ac.indptr[right+1])
    pool.charge('c84_independent_original_row_ratios', 64*(b-a-1)+128)
    if b-a != d-c or not np.array_equal(ac.indices[a:b-1], ac.indices[c:d-1]):
        raise ValueError('recipe hash collision or incomplete same-parent equality')
    p, q = Fraction(float(ac.data[b-1])), Fraction(float(ac.data[d-1]))
    if Fraction(float(rhs[left]))/p != Fraction(float(rhs[right]))/q:
        raise ValueError('different exact defining constant')
    for x, y in zip(ac.data[a:b-1], ac.data[c:d-1], strict=True):
        if Fraction(float(x))/p != Fraction(float(y))/q:
            raise ValueError('different exact defining coefficient')


def census(hz, *, first, limit, pool, enabled=False, observe=None):
    if not enabled:
        return None
    ac, ab, gc = hz.Ac, hz.Ab, hz.Gc
    if (not all(sp.isspmatrix_csr(a) and a.has_canonical_format for a in (ac,ab,gc))
            or ac.dtype != np.float64 or ac.indices.dtype != np.int32
            or not 0 <= first < limit <= hz.n_cont or ac.shape[0] != len(hz.b)
            or ac.shape[1] != hz.n_cont or ab.shape[0] != ac.shape[0]):
        raise ValueError('complete canonical current float64/int32 source required')
    cost = 8*int(ac.nnz)+192*int(hz.n_eq)+16*int(hz.n_cont)
    if cost > 200_000_000:
        raise MemoryError('unchanged nested census cap')
    before = pool.used
    branch = BranchPool(pool)
    branch.charge('c84_complete_dyadic_recipe_screen', cost)
    counts = np.zeros(hz.n_cont, np.int32)
    live = np.zeros(hz.n_cont, bool); live[gc.indices] = True
    # Count all direct MAIN definitions, including scalar/non-dyadic/binary rows.
    # An ambiguous defining pivot is rejected, not selected by favorable shape.
    for row in range(hz.n_eq):
        a,b = int(ac.indptr[row]),int(ac.indptr[row+1])
        if b>a and first <= int(ac.indices[b-1]) < limit:
            counts[int(ac.indices[b-1])] += 1
    table = np.zeros(hz.n_eq, DTYPE); at=0; reasons=Counter()
    for row in range(hz.n_eq):
        a,b = int(ac.indptr[row]),int(ac.indptr[row+1])
        if b-a<3: reasons['fewer_than_two_parents']+=1; continue
        col=int(ac.indices[b-1])
        if not first<=col<limit: reasons['protected_or_non_MAIN_pivot']+=1; continue
        if counts[col]!=1: reasons['nonunique_direct_definition']+=1; continue
        if live[col]: reasons['output_live']+=1; continue
        if ab.indptr[row+1]!=ab.indptr[row]: reasons['binary_definition']+=1; continue
        p=float(ac.data[b-1])
        if not LOW<=p<=HIGH or math.frexp(p)[0] != .5:
            reasons['not_positive_dyadic_pivot']+=1; continue
        key,support=normalized(ac.indices[a:b],ac.data[a:b],float(hz.b[row]))
        table[at]=(row,col,b-a,key,support); at+=1
        if observe and at%32768==0:
            observe(dict(event='complete_affine_recipe_progress',rows_through=row+1,candidates=at))
    view=table[:at]
    # Keep the whole table backing explicit. No claimed physical simplification.
    pairs=[]; support_counts=[]
    for field in ('recipe','support'):
        order=np.argsort(view[field],kind='stable')
        i=0
        while i<at:
            j=i+1
            while j<at and view[field][order[j]]==view[field][order[i]]: j+=1
            if field=='recipe' and j-i>1:
                first_row=int(view['row'][order[i]])
                for pos in range(i+1,j):
                    other=int(view['row'][order[pos]])
                    independently_equal(ac,hz.b,first_row,other,branch)
                    pairs.append((first_row,other))
            if field=='support' and j-i>1: support_counts.append(j-i)
            i=j
    if pool.used-before>200_000_000:
        raise MemoryError('complete matched-proof branch exceeds original200M')
    report=dict(schema='c84_complete_current_affine_recipe_census_v1',
        all_EQ_rows=int(hz.n_eq),all_continuous_predicate_terms=int(ac.nnz),
        first_MAIN=first,limit_MAIN=limit,candidate_rows=at,
        candidate_parent_terms=int(view['width'].sum())-at,rejections=dict(reasons),
        exact_recipe_twin_pairs=[list(p) for p in pairs],exact_recipe_twins=len(pairs),
        support_digest_repeated_groups=len(support_counts),
        support_digest_repeated_rows=sum(support_counts),
        support_digest_max_group=max(support_counts,default=0),
        support_digest_group_size_histogram=dict(sorted(Counter(support_counts).items())),
        support_digest_groups_are_not_equality_proofs=True,
        all_recipe_matches_independently_proved=True,table_rows_used=at,
        table_backing_rows=len(table),table_backing_bytes=table.nbytes,
        complete_screen_work=pool.used-before,complete=True,
        consumer_substitution_proved=False,strict_nnz_reduction_proved=False,
        new_HZ_constructed=False,solver_executed=False,formal_gain=0)
    if at+sum(reasons.values())!=hz.n_eq:
        raise ValueError('incomplete row classification')
    return report,table
