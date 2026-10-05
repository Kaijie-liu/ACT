"""Default-off exact unit row splicing with in-row reconstruction coefficients."""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import storage, _plain
from experiments.neural_hz_20260831.c12_shared_affine_census_v1 import row
from experiments.neural_hz_20260831.c13_separated_affine_census_v1 import coefficient_l1_numerator, exact_box
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

MATRICES = ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')
CERTIFICATE = ('columns', 'descriptors', 'offsets')


@dataclass
class UnitSplice:
    hz: SparseHZono
    columns: np.ndarray
    descriptors: np.ndarray
    offsets: np.ndarray
    old_n_cont: int
    logical_n_cont: int
    original_digest: str
    summary: dict
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {'hz', *CERTIFICATE, 'old_n_cont', 'logical_n_cont', 'original_digest', 'summary', 'seal'}:
            raise ValueError('unregistered splice payload')
        if set(vars(self.hz)) != {'c', *MATRICES, 'b', 'ub', 'frame_id', 'exact'}:
            raise ValueError('unregistered HZ payload')
        _plain(self.summary)
        if (type(self.old_n_cont) is not int or type(self.logical_n_cont) is not int
                or not 0 <= self.old_n_cont <= self.logical_n_cont <= self.hz.n_cont):
            raise ValueError('invalid original input frame')
        n = self.columns.size
        for key, dtype in (('columns', np.int32), ('descriptors', np.uint64), ('offsets', np.float64)):
            value = getattr(self, key)
            if type(value) is not np.ndarray or value.dtype != np.dtype(dtype) or value.ndim != 1:
                raise ValueError('invalid compact certificate array')
            if key != 'offsets' and value.size != n:
                raise ValueError('compact certificate width mismatch')
        flags = self.descriptors >> np.uint64(32)
        targets = self.descriptors & np.uint64((1 << 31) - 1)
        eq = (self.descriptors & np.uint64(1 << 31)) == 0
        if (np.any(flags > 255) or np.any((flags & np.uint64(63)) > 60)
                or self.offsets.size != int(np.count_nonzero(flags & np.uint64(128)))
                or not np.isfinite(self.offsets).all() or np.any(self.offsets == 0.)
                or np.any(self.columns < self.old_n_cont) or np.any(self.columns >= self.logical_n_cont)
                or np.any(np.diff(self.columns.astype(np.int64)) <= 0)
                or np.unique(self.descriptors & np.uint64((1 << 32) - 1)).size != n):
            raise ValueError('invalid compact reconstruction metadata')
        if np.any(targets[eq] >= self.hz.n_eq) or np.any(targets[~eq] >= self.hz.n_ineq):
            raise ValueError('reconstruction row outside HZ')
        h = hashlib.sha256(source_digest(self.hz).encode())
        h.update(json.dumps([self.old_n_cont, self.logical_n_cont, self.original_digest,
            self.summary], sort_keys=True, allow_nan=False).encode())
        for key in CERTIFICATE:
            h.update(key.encode())
            h.update(getattr(self, key).tobytes())
        return h.hexdigest()

    def validate(self):
        if self.fingerprint() != self.seal:
            raise ValueError('spliced HZ or reconstruction certificate changed')

    def numeric_roots(self):
        self.validate()
        return {'hz': self.hz, **{key: getattr(self, key) for key in CERTIFICATE}}

    def decoded(self):
        self.validate()
        cursor = 0
        for col, raw in zip(self.columns, self.descriptors):
            descriptor = int(raw)
            flag = descriptor >> 32
            index = descriptor & ((1 << 31) - 1)
            target = -index - 1 if descriptor & (1 << 31) else index
            offset = float(self.offsets[cursor]) if flag & 128 else 0.
            cursor += bool(flag & 128)
            yield int(col), int(target), math.ldexp(1., (flag & 63) - 20), offset, (-1 if flag & 64 else 1)

    def reconstruct_fraction(self, continuous):
        """Exact independent extension; no input sampling, solver repair or old HZ."""
        self.validate()
        if len(continuous) != self.hz.n_cont:
            raise ValueError('reconstruction requires unchanged global width')
        result = [Fraction(v) for v in continuous]
        if any(abs(v) > 1 for v in result):
            raise ValueError('point outside retained latent box')
        for col, target, pivot, offset, sign in self.decoded():
            matrix, index = (self.hz.Ac, target) if target >= 0 else (self.hz.Auc, -target - 1)
            cc, cv = row(matrix, index)
            stop = int(np.searchsorted(cc, col))
            prefix = sum((Fraction(float(v)) * result[int(c)] for c, v in zip(cc[:stop], cv[:stop])), Fraction(0))
            result[col] = (Fraction(offset) - sign * prefix) / Fraction(pivot)
            if abs(result[col]) > 1:
                raise ValueError('certified extension exceeded eliminated box')
        return result


def python_payload_bytes(root):
    """Closed payload graph shallow bytes, excluding numeric owners counted below.

    Python allocator rounding, interpreter/class globals and Torch C++ metadata
    are not claimed exact; the unchanged RSS/tracer cap separately bounds them.
    """
    seen = set()
    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if type(value) is np.ndarray:
            size = sys.getsizeof(value) - (value.nbytes if value.flags.owndata else 0)
            return size + (visit(value.base) if value.base is not None else 0)
        if type(value) is bytearray:
            return sys.getsizeof(value) - len(value)
        if type(value) is memoryview:
            return sys.getsizeof(value) + visit(value.obj)
        if isinstance(value, torch.Tensor):
            return sys.getsizeof(value)
        if type(value) in (SparseHZono, UnitSplice) or sp.isspmatrix_csr(value):
            return sys.getsizeof(value) + visit(vars(value))
        if type(value) is dict:
            return sys.getsizeof(value) + sum(visit(k) + visit(v) for k, v in value.items())
        if type(value) in (list, tuple):
            return sys.getsizeof(value) + sum(visit(v) for v in value)
        if value is None or type(value) in (bool, int, float, str, bytes) or isinstance(value, np.generic):
            return sys.getsizeof(value)
        raise ValueError(f'unregistered Python payload owner: {type(value).__name__}')
    return visit(root)


def _emit(source, original_ac, keep, replacements):
    """Write compact final CSR once; splice already-ordered coefficient blocks."""
    sizes = np.diff(source.indptr).astype(np.int64)
    for target, plan in replacements.items():
        d = plan['definition']
        sizes[target] = int(original_ac.indptr[d + 1] - original_ac.indptr[d] - 1 + sizes[target] - 1)
    widths = sizes[keep]
    nnz = int(widths.sum())
    if nnz >= 2**31:
        raise MemoryError('compact CSR index format exhausted')
    indptr = np.empty(len(widths) + 1, np.int32)
    indptr[0] = 0
    np.cumsum(widths, out=indptr[1:])
    indices, data = np.empty(nnz, np.int32), np.empty(nnz, np.float64)
    target_row = 0
    for old_row in range(source.shape[0]):
        if not keep[old_row]:
            continue
        begin, end = int(indptr[target_row]), int(indptr[target_row + 1])
        cc, cv = row(source, old_row)
        plan = replacements.get(old_row)
        if plan is None:
            indices[begin:end], data[begin:end] = cc, cv
        else:
            pc, pv = row(original_ac, plan['definition'])
            middle = begin + len(pc) - 1
            indices[begin:middle] = pc[:-1]
            data[begin:middle] = pv[:-1] if plan['sign'] == 1 else -pv[:-1]
            indices[middle:end], data[middle:end] = cc[1:], cv[1:]
        target_row += 1
    return sp.csr_matrix((data, indices, indptr), shape=(len(widths), source.shape[1]))


def splice(hz, *, enabled=False, old_n_cont=None, logical_n_cont=None, old_n_eq=None,
           eq_roots=None, eq_scales=None, def_rows=None, max_work=256_000_000,
           max_entries=64_000_000, observe=None):
    if not enabled:
        return None
    pool = WorkPool(max_work)
    if type(max_entries) is not int or not 0 <= max_entries <= 64_000_000:
        raise ValueError('invalid/increased entry ceiling')
    if (any(type(v) is not int or v < 0 for v in (old_n_cont, logical_n_cont, old_n_eq))
            or not old_n_cont <= logical_n_cont <= hz.n_cont or not hz.exact or hz.frame_id is None):
        raise ValueError('invalid protected MAIN prefix')
    matrices = [getattr(hz, k) for k in MATRICES]
    if any(not sp.isspmatrix_csr(m) or m.dtype != np.float64 for m in matrices):
        raise ValueError('float64 canonical CSR required')
    nnz = sum(m.nnz for m in matrices)
    if nnz > max_entries or max(hz.n_cont, hz.n_eq, hz.n_ineq) >= 2**31:
        raise MemoryError('input entries/compact index ceiling exceeded')
    main = logical_n_cont - old_n_cont
    pool.charge('complete_structural_validation', int(8 * nnz + 32 * main))
    frozen = source_digest(hz)
    if any(not m.has_canonical_format or np.any(m.data == 0.) for m in matrices):
        raise ValueError('finite canonical zero-free CSR required')
    if any(np.any((np.abs(m.data) < 2.**-20) | (np.abs(m.data) > 2.**40)) for m in matrices[2:]):
        raise ValueError('input predicate outside fixed coefficient window')
    eq_roots, eq_scales, def_rows = map(np.asarray, (eq_roots, eq_scales, def_rows))
    if eq_roots.shape != (old_n_eq + main,) or eq_scales.shape != eq_roots.shape:
        raise ValueError('incomplete logical MAIN mapping')
    if def_rows.ndim != 1 or logical_n_cont + def_rows.size > hz.n_cont:
        raise ValueError('invalid radix frame')
    holder = SimpleNamespace(old_n_cont=old_n_cont, logical_n_cont=logical_n_cont,
        old_n_eq=old_n_eq, eq_roots=eq_roots, eq_scales=eq_scales)
    removed, _, _, _ = aliases(holder)
    roots = eq_roots[eq_roots >= 0]
    if (np.asarray(def_rows).dtype != np.dtype(np.int64)
            or not np.array_equal(np.sort(np.r_[roots, def_rows]), np.arange(roots.size + len(def_rows)))
            or roots.size + len(def_rows) > hz.n_eq):
        raise ValueError('invalid complete surviving MAIN/radix prefix partition')
    value_degree = np.bincount(hz.Gc.indices, minlength=hz.n_cont)
    degree = np.bincount(hz.Ac.indices, minlength=hz.n_cont) + np.bincount(hz.Auc.indices, minlength=hz.n_cont)
    if np.any(value_degree[removed] + degree[removed]):
        raise ValueError('tagged removed coordinate still active')
    cols = np.arange(old_n_cont, logical_n_cont, dtype=np.int64)
    defs = eq_roots[old_n_eq:]
    remaining = defs >= 0
    valid = np.flatnonzero(remaining & (value_degree[cols] == 0) & (degree[cols] == 2))
    direct = []
    for index in valid:
        cc, cv = row(hz.Ac, int(defs[index]))
        if cc.size and cc[-1] == cols[index] and cv[-1] > 0. and math.frexp(float(cv[-1]))[0] == .5:
            direct.append(index)
    chosen = np.asarray(direct, dtype=np.int64)
    eqc, ineqc = hz.Ac.tocsc(), hz.Auc.tocsc()
    consumers, inequalities, replacements, consumer_widths = [], [], [], []
    for index in chosen:
        col, definition = int(cols[index]), int(defs[index])
        start, stop = eqc.indptr[col:col + 2]
        found = [int(r) for r in eqc.indices[start:stop] if r != definition]
        inequality = not found
        if inequality:
            start, stop = ineqc.indptr[col:col + 2]
            found = [int(r) for r in ineqc.indices[start:stop]]
        if len(found) != 1:
            raise ValueError('single remaining predicate occurrence accounting failed')
        consumer = found[0]
        cmat, bmat = (hz.Auc, hz.Aub) if inequality else (hz.Ac, hz.Ab)
        width = int(cmat.indptr[consumer + 1] - cmat.indptr[consumer]
                    + bmat.indptr[consumer + 1] - bmat.indptr[consumer])
        terms = int(hz.Ac.indptr[definition + 1] - hz.Ac.indptr[definition] - 1
                    + hz.Ab.indptr[definition + 1] - hz.Ab.indptr[definition] + (hz.b[definition] != 0.))
        consumers.append(consumer)
        inequalities.append(inequality)
        replacements.append(terms)
        consumer_widths.append(width)
    plans = []
    for index, consumer, inequality, consumer_width in zip(chosen, consumers, inequalities, consumer_widths):
        col, definition = int(cols[index]), int(defs[index])
        cc, cv = row(hz.Ac, definition)
        bc, bv = row(hz.Ab, definition)
        cmat = hz.Auc if inequality else hz.Ac
        tc, tv = row(cmat, consumer)
        pivot = float(cv[-1])
        consumer_value = float(tv[np.searchsorted(tc, col)])
        if abs(consumer_value) != pivot:
            continue
        # An ordered binary-free prefix is what makes reconstruction reuse the
        # surviving row without retaining the deleted dense definition.
        if len(bc) or not len(tc) or int(tc[0]) != col:
            raise ValueError('unit pair does not admit a binary-free ordered in-row certificate')
        sign = -1 if consumer_value > 0. else 1
        plans.append({'column': col, 'definition': definition, 'consumer': consumer,
            'inequality': bool(inequality), 'sign': sign, 'pivot': pivot,
            'exponent': math.frexp(pivot)[1] - 1, 'offset': float(hz.b[definition]),
            'parent_terms': len(cc) - 1, 'new_row_width': len(cc) - 2 + consumer_width})
    if not plans:
        return None
    defining_rows = {p['definition'] for p in plans}
    consumer_keys = {(p['inequality'], p['consumer']) for p in plans}
    if (len(consumer_keys) != len(plans)
            or any(not p['inequality'] and p['consumer'] in defining_rows for p in plans)):
        raise ValueError('unit row pairs have shared consumers or selected-definition dependencies')
    norm_terms = sum(p['parent_terms'] for p in plans)
    affected_terms = sum(p['new_row_width'] for p in plans)
    extra = {'exact_box_norm': 16 * norm_terms, 'unit_pair_scalars': 32 * len(plans),
        'compact_emission_and_seals': 12 * int(nnz), 'affected_row_assembly': 4 * affected_terms,
        'output_row_metadata': 4 * (hz.n_eq + hz.n_ineq)}
    total = pool.used + sum(extra.values())
    preliminary = {'event': 'unit_splice_complete_preflight',
        'structural_single_use_definitions': len(chosen), 'selected_unit_pairs': len(plans),
        'definition_parent_terms': norm_terms, 'affected_output_coefficients': affected_terms,
        'logical_work_upper': total, 'max_work': max_work,
        'certificate_encoding': 'int32 column + uint64(row31,kind1,exponent6,sign1,offset1) + sparse float64 offsets'}
    if observe:
        observe(preliminary)
    if total > max_work:
        raise MemoryError(f'unit splice complete work ceiling exceeded: {preliminary}')
    for name, amount in extra.items():
        pool.charge(name, int(amount))
    for plan in plans:
        _, values = row(hz.Ac, plan['definition'])
        norm = coefficient_l1_numerator(values[:-1])
        if not exact_box(norm, plan['offset'], plan['pivot']):
            raise ValueError('unit defining box is not proved redundant')
        old_rhs = float((hz.ub if plan['inequality'] else hz.b)[plan['consumer']])
        change = plan['sign'] * plan['offset'] if plan['offset'] else 0.
        new_rhs = old_rhs + change
        if not math.isfinite(new_rhs) or Fraction(new_rhs) != Fraction(old_rhs) + plan['sign'] * Fraction(plan['offset']):
            raise ValueError('unit consumer RHS sum is not exactly representable')
        plan['new_rhs'] = new_rhs

    keep = np.ones(hz.n_eq, bool)
    keep[list(defining_rows)] = False
    eq_plans = {p['consumer']: p for p in plans if not p['inequality']}
    ineq_plans = {p['consumer']: p for p in plans if p['inequality']}
    ac = _emit(hz.Ac, hz.Ac, keep, eq_plans)
    auc = _emit(hz.Auc, hz.Ac, np.ones(hz.n_ineq, bool), ineq_plans) if ineq_plans else hz.Auc
    # Erased defining rows have ZERO binary support, so all binary payload is
    # reused exactly; only the equality row pointer needs shortening.
    binary_indptr = np.r_[hz.Ab.indptr[:-1][keep], hz.Ab.indptr[-1]].astype(np.int32, copy=False)
    ab = sp.csr_matrix((hz.Ab.data, hz.Ab.indices, binary_indptr), shape=(int(keep.sum()), hz.n_bin))
    b, ub = hz.b[keep], hz.ub.copy() if ineq_plans else hz.ub
    eq_map = np.cumsum(keep, dtype=np.int64) - 1
    columns, descriptors, offsets = [], [], []
    for plan in plans:
        index = plan['consumer'] if plan['inequality'] else int(eq_map[plan['consumer']])
        flag = plan['exponent'] + 20
        if not 0 <= flag <= 60:
            raise ValueError('pivot exponent outside compact window')
        if plan['sign'] == -1:
            flag |= 64
        if plan['offset']:
            flag |= 128
            offsets.append(plan['offset'])
        descriptor = index | ((1 << 31) if plan['inequality'] else 0) | (flag << 32)
        columns.append(plan['column'])
        descriptors.append(descriptor)
        if plan['inequality']:
            ub[index] = plan['new_rhs']
        else:
            b[index] = plan['new_rhs']
    new = SparseHZono(hz.c, hz.Gc, hz.Gb, ac, ab, b, auc, hz.Aub, ub, frame_id=hz.frame_id, exact=True)
    new_nnz = sum(getattr(new, name).nnz for name in MATRICES)
    if new_nnz != nnz - 2 * len(plans):
        raise ValueError('strict simultaneous total nnz reduction failed')
    active = np.zeros(hz.n_cont, bool)
    active[columns] = True
    if any(np.any(active[m.indices]) for m in (new.Gc, new.Ac, new.Auc)):
        raise ValueError('spliced column is still live')
    result = UnitSplice(new, np.asarray(columns, np.int32), np.asarray(descriptors, np.uint64),
        np.asarray(offsets, np.float64), old_n_cont, logical_n_cont, frozen,
        {'schema': 'c15_unit_row_splice_v1', 'default_off': True, 'formal_gain': 0,
         'unit_pairs': len(plans), 'original_coefficient_nnz': int(nnz),
         'spliced_coefficient_nnz': int(new_nnz), 'logical_work_upper': pool.used,
         'solver_executed': False, 'whole_live_path_proved': False})
    result.seal = result.fingerprint()
    old_storage, new_storage = storage({'hz': hz}), storage(result.numeric_roots())
    old_python, new_python = python_payload_bytes(hz), python_payload_bytes(result)
    diagnostics = {**preliminary, 'work_parts': dict(pool.parts), 'formal_gain': 0,
        'original_numeric_bytes': old_storage.resident_bytes, 'spliced_numeric_bytes': new_storage.resident_bytes,
        'original_numeric_entries': old_storage.resident_entries, 'spliced_numeric_entries': new_storage.resident_entries,
        'original_python_payload_bytes': old_python, 'spliced_python_payload_bytes': new_python,
        'original_controlled_bytes': old_storage.resident_bytes + old_python,
        'spliced_controlled_bytes': new_storage.resident_bytes + new_python,
        'certificate_numeric_bytes': sum(getattr(result, key).nbytes for key in CERTIFICATE),
        'certificate_numeric_entries': sum(getattr(result, key).size for key in CERTIFICATE),
        'no_original_definition_buffers_retained': True, 'global_width_unchanged': True,
        'all_binary_factors_retained': True, 'strict_nnz_decrease': True,
        'candidate_transformation_constructed': True, 'whole_live_path_proved': False,
        'native_ingestion_executed': False, 'solver_executed': False}
    if new_storage.resident_entries > max_entries:
        raise MemoryError('spliced component plus certificate exceeds entry ceiling')
    if (new_storage.resident_entries >= old_storage.resident_entries
            or new_storage.resident_bytes >= old_storage.resident_bytes
            or diagnostics['spliced_controlled_bytes'] >= diagnostics['original_controlled_bytes']):
        raise ValueError(f'no strict complete component saving including certificate/Python payload: {diagnostics}')
    if source_digest(hz) != frozen:
        raise ValueError('unit splice mutated original HZ')
    result.validate()
    if observe:
        observe({'event': 'unit_splice_component_constructed', **diagnostics})
    return result, diagnostics
