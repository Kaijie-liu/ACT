"""Default-off first-write unit splice from PRE + appended predicate blocks.

AppendView is NOT an HZ or an admission certificate. Its caller must bind the
complete canonical, zero-free, exact source, incidence, and redundant-box
proofs. This component does not issue a Closed receipt. The source is immutable
through discovery/write/audit. No aggregate unspliced matrix is constructed.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan


@dataclass(frozen=True)
class AppendView:
    pre: SparseHZono
    eq_c: sp.csr_matrix
    eq_b: sp.csr_matrix
    eq_rhs: np.ndarray
    le_c: sp.csr_matrix
    le_b: sp.csr_matrix
    le_rhs: np.ndarray
    c: np.ndarray
    Gc: sp.csr_matrix
    Gb: sp.csr_matrix

    @property
    def n_cont(self): return self.Gc.shape[1]
    @property
    def n_bin(self): return self.Gb.shape[1]
    @property
    def n_eq(self): return self.pre.n_eq + self.eq_c.shape[0]
    @property
    def n_ineq(self): return self.pre.n_ineq + self.le_c.shape[0]

    def blocks(self, name):
        return (getattr(self.pre, name), getattr(self, {
            'Ac':'eq_c', 'Ab':'eq_b', 'Auc':'le_c', 'Aub':'le_b',
            'b':'eq_rhs', 'ub':'le_rhs'}[name]))

    def locate(self, kind, row):
        old = self.pre.n_ineq if kind else self.pre.n_eq
        matrices = self.blocks('Auc' if kind else 'Ac')
        rhs = self.blocks('ub' if kind else 'b')
        block = int(row >= old)
        return matrices[block], rhs[block], row-old if block else row

    def validate_shape(self, pool):
        # Full numeric source validation belongs to the explicitly bound proof;
        # checking these metadata fields is not a substitute for that proof.
        pool.charge('append_block_geometry', 384)
        if (type(self.pre) is not SparseHZono or not self.pre.exact or self.pre.frame_id is None
                or self.n_cont < self.pre.n_cont or self.n_bin < self.pre.n_bin
                or max(self.n_cont, self.n_bin, self.n_eq, self.n_ineq) >= 2**31):
            raise ValueError('invalid append frame/geometry')
        mats = [self.Gc, self.Gb, self.eq_c, self.eq_b, self.le_c, self.le_b,
                self.pre.Ac, self.pre.Ab, self.pre.Auc, self.pre.Aub]
        if any(not sp.isspmatrix_csr(m) or m.dtype != np.float64
               or m.indices.dtype != np.int32 or m.indptr.dtype != np.int32
               or not m.has_canonical_format for m in mats):
            raise ValueError('bound canonical float64/int32 CSR blocks required')
        if sum(int(m.nnz) for m in mats) > 64_000_000:
            raise MemoryError('unchanged coefficient entry ceiling')
        for name, count in (('c', self.Gc.shape[0]), ('eq_rhs', self.eq_c.shape[0]),
                            ('le_rhs', self.le_c.shape[0])):
            value = getattr(self, name)
            if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != (count,):
                raise ValueError('append RHS/output geometry')
        if (self.Gb.shape[0] != len(self.c) or self.eq_c.shape != (len(self.eq_rhs), self.n_cont)
                or self.eq_b.shape != (len(self.eq_rhs), self.n_bin)
                or self.le_c.shape != (len(self.le_rhs), self.n_cont)
                or self.le_b.shape != (len(self.le_rhs), self.n_bin)):
            raise ValueError('append continuous/binary geometry')


def row(matrix, index):
    a, b = int(matrix.indptr[index]), int(matrix.indptr[index+1])
    return matrix.indices[a:b], matrix.data[a:b]


def _plans(view, plans, pool):
    n = len(plans)
    pool.charge('writer_plan_scalar_exact_checks', 192*n)
    definitions, keys, rhs = set(), set(), {}
    previous = -1
    for p in plans:
        if (type(p) is not Plan or type(p.column) is not int or not previous < p.column < view.pre.n_cont
                or type(p.definition) is not int or not 0 <= p.definition < view.pre.n_eq
                or type(p.consumer) is not int or type(p.inequality) is not bool
                or not 0 <= p.consumer < (view.n_ineq if p.inequality else view.n_eq)
                or type(p.sign) is not int or p.sign not in (-1, 1)):
            raise ValueError('sorted source-bound plan required')
        previous = p.column
        key = (p.inequality, p.consumer)
        if p.definition in definitions or key in keys:
            raise ValueError('repeated definition/consumer')
        definitions.add(p.definition); keys.add(key)
        pc, pv = row(view.pre.Ac, p.definition)
        cm, crhs, at = view.locate(p.inequality, p.consumer)
        cc, cv = row(cm, at)
        if (not len(pc) or not len(cc) or int(pc[-1]) != p.column or int(cc[0]) != p.column
                or view.pre.Ab.indptr[p.definition] != view.pre.Ab.indptr[p.definition+1]
                or p.pivot <= 0 or not math.isfinite(p.pivot) or math.frexp(p.pivot)[0] != .5
                or not 2.**-20 <= p.pivot <= 2.**40 or float(pv[-1]) != p.pivot
                or float(cv[0]) != -p.sign*p.pivot or float(view.pre.b[p.definition]) != p.offset):
            raise ValueError('plan differs from binary-free ordered unit source')
        pool.charge('writer_tail_binding', 4*(len(cc)-1))
        if tuple(map(int, cc[1:])) != p.tail:
            raise ValueError('changed consumer tail')
        old = float(crhs[at]); new = old+p.sign*p.offset
        if not all(map(math.isfinite, (old, p.offset, new))) or F(new) != F(old)+p.sign*F(p.offset):
            raise ValueError('inexact/nonfinite RHS')
        rhs[key] = new
    if any(not kind and c in definitions for kind, c in keys):
        raise ValueError('selected definitions/consumers are dependent')
    return definitions, rhs


def _schedule(view, name, plans, definitions, pool):
    kind = name in ('Auc', 'Aub', 'ub')
    continuous = name in ('Ac', 'Auc')
    changed = {p.consumer:p for p in plans if p.inequality == kind} if continuous or name in ('b','ub') else {}
    deleted = definitions if not kind else set()
    edits = {**changed, **{d:None for d in deleted}}
    n = len(edits)
    pool.charge('writer_sparse_schedule', 64*(n+2)+4*n*max(1,(n-1).bit_length()))
    return sorted(edits.items())


def _matrix(view, name, plans, definitions, pool, ledger):
    blocks = view.blocks(name)
    edits = _schedule(view, name, plans, definitions, pool)
    continuous = name in ('Ac','Auc')
    old_rows = blocks[0].shape[0]
    total_rows = sum(m.shape[0] for m in blocks)
    out_rows = total_rows - (len(definitions) if name in ('Ac','Ab') else 0)
    nnz = sum(int(m.nnz) for m in blocks)
    for r, p in edits:
        m, at = (blocks[0], r) if r < old_rows else (blocks[1], r-old_rows)
        if p is None:
            nnz -= int(m.indptr[at+1]-m.indptr[at])
        elif continuous:
            nnz += int(view.pre.Ac.indptr[p.definition+1]-view.pre.Ac.indptr[p.definition])-2
    if not 0 <= nnz < 2**31:
        raise MemoryError('first-write compact CSR exhausted')
    # Charge before allocation. Native payload transfers are counted explicitly
    # and compared with the SAME-boundary vstack traffic; they are not free.
    pool.charge('writer_row_pointer_allocation_and_write', 4*(out_rows+1))
    pool.charge('writer_matrix_allocation_and_publication', 128)
    pool.charge('native_coefficient_and_index_transfers', 2*nnz)
    ptr = np.empty(out_rows+1, np.int32); ptr[0] = 0
    indices, data = np.empty(nnz, np.int32), np.empty(nnz, np.float64)
    dest = output_row = copied = parent = negated = segments = 0

    def copy_values(m, a, b, sign=1, producer=False):
        nonlocal dest, copied, parent, negated, segments
        n = b-a
        indices[dest:dest+n] = m.indices[a:b]
        if sign == 1:
            data[dest:dest+n] = m.data[a:b]
        else:
            pool.charge('writer_actual_parent_negations', n)
            np.negative(m.data[a:b], out=data[dest:dest+n]); negated += n
        dest += n; copied += n; parent += n if producer else 0; segments += 1

    def run(m, a, b):
        nonlocal output_row
        if a == b: return
        begin, end = int(m.indptr[a]), int(m.indptr[b])
        np.add(m.indptr[a+1:b+1], dest-begin, out=ptr[output_row+1:output_row+b-a+1])
        copy_values(m, begin, end)
        output_row += b-a

    offset = 0
    for m in blocks:
        cursor = 0
        for global_row, p in edits:
            if not offset <= global_row < offset+m.shape[0]: continue
            at = global_row-offset
            run(m, cursor, at)
            if p is not None:
                a, b = int(m.indptr[at]), int(m.indptr[at+1])
                d = p.definition
                pa, pb = int(view.pre.Ac.indptr[d]), int(view.pre.Ac.indptr[d+1])
                copy_values(view.pre.Ac, pa, pb-1, p.sign, producer=True)
                copy_values(m, a+1, b)
                output_row += 1; ptr[output_row] = dest
            cursor = at+1
        run(m, cursor, m.shape[0]); offset += m.shape[0]
    if (output_row, dest, copied) != (out_rows, nnz, nnz):
        raise ValueError('first-write schedule did not fill each output exactly once')
    width = view.n_cont if continuous else view.n_bin
    result = sp.csr_matrix((data, indices, ptr), shape=(out_rows, width), copy=False)
    # This follows from bound zero-free canonical source + disjoint ordered
    # prefix/tail transfer, NOT from an unproved claim of arbitrary input safety.
    result.has_sorted_indices = True; result.has_canonical_format = True
    result._act_hz_zero_free = True
    if (not np.shares_memory(result.data, data) and nnz
            or not np.shares_memory(result.indices, indices) and nnz
            or not np.shares_memory(result.indptr, ptr)):
        raise ValueError('CSR publication made a hidden buffer copy')
    ledger[name] = dict(source_coefficients=sum(int(m.nnz) for m in blocks),
        written_coefficients=nnz, coefficient_and_index_transfers=2*nnz,
        output_row_pointer_entries=out_rows+1, parent_terms_written_once=parent,
        actual_parent_negations=negated, copied_segments=segments,
        final_buffer_bytes=data.nbytes+indices.nbytes+ptr.nbytes,
        no_constructor_buffer_copy=True)
    return result


def _rhs(view, name, plans, definitions, updated, pool, ledger):
    blocks = view.blocks(name); old_rows = len(blocks[0]); kind = name == 'ub'
    edits = _schedule(view, name, plans, definitions, pool)
    size = sum(map(len, blocks)) - (0 if kind else len(definitions))
    pool.charge('native_RHS_transfers', size)
    out = np.empty(size, np.float64)
    dest = offset = 0
    for source in blocks:
        cursor = 0
        for r, p in edits:
            if not offset <= r < offset+len(source): continue
            at = r-offset; n = at-cursor
            out[dest:dest+n] = source[cursor:at]; dest += n
            if p is not None:
                out[dest] = updated[(kind,r)]; dest += 1
            cursor = at+1
        n = len(source)-cursor
        out[dest:dest+n] = source[cursor:]; dest += n; offset += len(source)
    if dest != size: raise ValueError('RHS first-write size mismatch')
    ledger[name] = dict(source_entries=sum(map(len,blocks)), written_entries=size,
                        final_buffer_bytes=out.nbytes)
    return out


def splice_append(view, plans, *, pool, enabled=False):
    if not enabled: return None
    view.validate_shape(pool)
    if not plans: return None
    definitions, updated = _plans(view, plans, pool)
    ledger = {}
    matrices = {name:_matrix(view,name,plans,definitions,pool,ledger)
                for name in ('Ac','Ab','Auc','Aub')}
    b = _rhs(view,'b',plans,definitions,updated,pool,ledger)
    ub = _rhs(view,'ub',plans,definitions,updated,pool,ledger)
    result = SparseHZono(view.c,view.Gc,view.Gb,matrices['Ac'],matrices['Ab'],b,
        matrices['Auc'],matrices['Aub'],ub,frame_id=view.pre.frame_id,exact=True)
    if any(getattr(result,k) is not m for k,m in matrices.items()):
        raise ValueError('HZ publication copied a predicate matrix')
    original = sum(v['source_coefficients'] for v in ledger.values() if 'source_coefficients' in v)
    current = sum(v['written_coefficients'] for v in ledger.values() if 'written_coefficients' in v)
    if current != original-2*len(plans): raise ValueError('strict exact nnz decrease failed')
    native = sum(v for k,v in pool.parts.items() if k.startswith('native_'))
    report = dict(schema='c30_first_write_component_v1', unit_pairs=len(plans), matrices=ledger,
        aggregate_unspliced_matrix_constructed=False, new_native_relu_executed=False,
        source_proof_required=True, live_admission_certificate=False, formal_gain=0,
        writer_work=pool.used, writer_work_parts=dict(pool.parts),
        native_payload_transfer_work=native, additional_routing_work=pool.used-native,
        baseline_native_payload_transfers=2*original+view.n_eq+view.n_ineq,
        candidate_native_payload_transfers=native)
    return result, report
