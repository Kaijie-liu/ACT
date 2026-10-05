"""Fused exact affine/radix/alias emission; no old completed HZ/postpass."""

from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_cnn import _sparse_hz_storage_entries
from experiments.neural_hz_20260831.c7_factored_hz_v1 import Lifted, expression_binding, scaled_exact, restricted_row, _source_row
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import graph, box_exponent
from experiments.neural_hz_20260831.c9_radix_predicate_v1 import _row
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool, FusedRowEncoder, fold_rows, aliases
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays


@dataclass
class FusedIntegrated:
    expression: object
    origin_binding: tuple
    hz: object
    nodes: list
    root: int
    old_n_cont: int
    old_n_bin: int
    old_n_eq: int
    logical_n_cont: int
    keep: np.ndarray
    report: dict
    eq_roots: np.ndarray
    eq_scales: np.ndarray
    ineq_roots: np.ndarray
    ineq_scales: np.ndarray
    def_rows: np.ndarray
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {'expression', 'origin_binding', 'hz', 'nodes', 'root', 'old_n_cont',
                'old_n_bin', 'old_n_eq', 'logical_n_cont', 'keep', 'report', 'eq_roots', 'eq_scales',
                'ineq_roots', 'ineq_scales', 'def_rows', 'seal'}:
            raise ValueError('unregistered integrated state')
        base = Lifted(self.expression, self.origin_binding, self.hz, self.nodes, self.root,
            self.old_n_cont, self.old_n_bin, self.old_n_eq, self.keep, self.report)
        h = hashlib.sha256(base.fingerprint().encode())
        h.update(str(self.logical_n_cont).encode())
        h.update(digest_arrays(self.eq_roots, self.eq_scales, self.ineq_roots, self.ineq_scales, self.def_rows).encode())
        return h.hexdigest()

    def validate(self):
        aliases(self)
        if expression_binding(self.expression) != self.origin_binding or self.fingerprint() != self.seal:
            raise ValueError('integrated source/graph/HZ changed')

    def numeric_roots(self):
        self.validate()
        result = {key: getattr(self, key) for key in ('expression', 'hz', 'keep', 'eq_roots',
            'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
        for index, node in enumerate(self.nodes):
            for key in ('support', 'needed', 'slots', 'exponents'):
                result[f'node{index}_{key}'] = node[key]
            for key in ('source', 'op'):
                if key in node:
                    result[f'node{index}_{key}'] = node[key]
        return result


def lift(expr, keep_rows, *, enabled=False, frame_widths=None, max_work=256_000_000,
         max_branch_work=200_000_000, max_entries=64_000_000, observe=None):
    if not enabled:
        return None
    for cap, limit in ((max_work, 256_000_000), (max_branch_work, 200_000_000), (max_entries, 64_000_000)):
        if type(cap) is not int or not 0 <= cap <= limit:
            raise ValueError('invalid or increased global cap')
    frozen = expression_binding(expr)
    keep = np.asarray(keep_rows)
    if keep.dtype != np.dtype(bool) or keep.shape != (expr.n_out,):
        raise ValueError('invalid output mask')
    keep = keep.copy()
    base = zero_transfer_reference(expr)
    old_nc, old_nb = base.n_cont, base.n_bin
    if frame_widths is not None:
        if (len(frame_widths) != 2 or any(type(v) is not int for v in frame_widths)
                or frame_widths[0] < old_nc or frame_widths[1] < old_nb):
            raise ValueError('old shared frame cannot shrink')
        old_nc, old_nb = frame_widths
    nodes, root, report = graph(expr, keep, max_work)
    main_aux = report['auxiliaries']
    logical_nc = old_nc + main_aux
    old_entries = sum(getattr(base, key).nnz for key in ('Ac', 'Ab', 'Auc', 'Aub')) + base.n_eq + base.n_ineq
    old_work = 12 * old_entries
    costs = []
    for node, count in zip(nodes, report['node_counts']):
        count['encoding_work_upper'] = count['encoding_work_upper'] // 16 * 12
        cost = count['encoding_work_upper'] + count['support_work']
        costs.append(cost + max((costs[p] for p in node['parents']), default=0))
    report['affine_work_upper'] = report['support_work'] + sum(n['encoding_work_upper'] for n in report['node_counts'])
    report['total_work_upper'] = int(report['affine_work_upper'] + old_work)
    report['largest_branch_work_upper'] = int(max(costs) + old_work)
    predicate_entries = old_entries + report['continuous_edges'] + report['binary_edges'] + 2 * main_aux
    estimate = _sparse_hz_storage_entries(base) + report['continuous_edges'] + report['binary_edges'] + 2 * main_aux + int(keep.sum()) + 131_072
    report.update(formal_gain=0, source_count=len({id(t.source) for t in expr.terms}), terms=len(expr.terms),
        node_count=len(nodes), old_predicate_work_upper=old_work, radix_work_reserve=16_000_000,
        radix_auxiliary_reserve=16_384, radix_entry_reserve=131_072, estimated_hz_entries=estimate,
        default_off=True, input_binding_unchanged=False, live_publication_executed=False,
        encoding_operations_per_entry=12, whole_state_reduction_proved=False)
    if observe is not None:
        observe('complete_preflight', dict(report))
    if report['total_work_upper'] > max_work or report['largest_branch_work_upper'] > max_branch_work or estimate > max_entries:
        raise MemoryError('integrated complete plus reserves preflight rejected')
    if 4 * 16 * (predicate_entries + 131_072) + 256 * (main_aux + 16_384) > 1024**3:
        raise MemoryError('integrated numeric preallocation envelope exceeds 1 GiB')
    cursor = old_nc
    for node in nodes:
        rows = np.flatnonzero(node['needed'])
        node['slots'] = np.full(node['width'], -1, dtype=np.int64)
        node['slots'][rows] = cursor + np.arange(rows.size)
        node['exponents'] = np.zeros(node['width'], dtype=np.int32)
        cursor += rows.size
    if cursor != logical_nc:
        raise ValueError('complete main-slot reserve mismatch')
    pool = WorkPool(report['total_work_upper'], report['largest_branch_work_upper'], max_work, max_branch_work)
    encoder = FusedRowEncoder(logical_nc, old_nb, predicate_entries, pool=pool)
    eq_roots, eq_scales, ineq_roots, ineq_scales = [], [], [], []
    for cmat, bmat, rhs, inequality, roots, scales in (
            (base.Ac, base.Ab, base.b, False, eq_roots, eq_scales),
            (base.Auc, base.Aub, base.ub, True, ineq_roots, ineq_scales)):
        for index in range(cmat.shape[0]):
            cc, cv = _row(cmat, index)
            bc, bv = _row(bmat, index)
            physical, shift = encoder.encode(cc, cv, bc, bv, float(rhs[index]), inequality=inequality)
            roots.append(physical)
            scales.append(shift)
    for node_index, node in enumerate(nodes):
        for coordinate in np.flatnonzero(node['needed']):
            bc, bv = np.empty(0, dtype=np.int64), np.empty(0)
            constant = 0.
            if node['kind'] == 'source':
                source = node['source']
                (cc, cv), (bc, bv) = _source_row(source, coordinate)
                constant = float(source.c[coordinate])
                powers = np.zeros(cv.size, dtype=np.int64)
                summands = np.concatenate((cv, bv, [constant] if constant != 0. else []))
                unit = box_exponent(summands)
            elif node['kind'] == 'op':
                parent = nodes[node['parents'][0]]
                indices, cv = restricted_row(node['op'], coordinate, parent['needed'])
                cc, powers = parent['slots'][indices], parent['exponents'][indices].astype(np.int64)
                unit = box_exponent(cv, powers)
            else:
                terms = [(nodes[p]['slots'][coordinate], nodes[p]['exponents'][coordinate], number)
                    for p, number in sorted(Counter(node['parents']).items()) if nodes[p]['support'][coordinate]]
                if any(t[2] > 2**53 for t in terms):
                    raise ValueError('inexact source multiplicity')
                cc = np.array([t[0] for t in terms], dtype=np.int64)
                cv = np.array([t[2] for t in terms], dtype=np.float64)
                powers = np.array([t[1] for t in terms], dtype=np.int64)
                unit = box_exponent(cv, powers)
            node['exponents'][coordinate] = unit
            physical, shift = encoder.encode(np.append(cc, node['slots'][coordinate]), np.append(-cv, 1.),
                bc, -bv, constant, cp=np.append(powers, unit))
            eq_roots.append(physical)
            eq_scales.append(shift)
        if observe is not None:
            observe('encoded_node', {'node': node_index, 'kind': node['kind'], 'logical_equalities': len(eq_roots),
                'radix_auxiliaries': len(encoder.def_rows), 'radix_extra_work': encoder.extra_work})
    output_slots = nodes[root]['slots'][nodes[root]['needed']]
    eq_roots, eq_scales, alias_report = fold_rows(encoder, eq_roots, eq_scales,
        old_nc=old_nc, old_eq=base.n_eq, output_slots=output_slots, observe=observe)
    report.update(alias_quotient=alias_report, work_allocation='coupled_fixed_whole_and_branch_caps',
        whole_base_work=pool.whole_base, branch_base_work=pool.branch_base,
        total_work_upper=pool.whole_base + pool.used,
        largest_branch_work_upper=pool.branch_base + pool.used)
    ac, ab, b = encoder.matrices(encoder.eq)
    auc, aub, ub = encoder.matrices(encoder.ineq)
    root_node = nodes[root]
    output_rows = np.flatnonzero(root_node['needed'])
    output_values = scaled_exact(np.ones(output_rows.size), root_node['exponents'][output_rows])
    if np.any(output_values <= 1e-9) or np.any(output_values >= 1e15):
        raise ValueError('output coefficient outside unchanged native thresholds')
    nc = logical_nc + len(encoder.def_rows)
    gc = sp.csr_matrix((output_values, (output_rows, root_node['slots'][output_rows])), shape=(expr.n_out, nc))
    hz = SparseHZono(expr.bias.copy(), gc, sp.csr_matrix((expr.n_out, old_nb)), ac, ab, b, auc, aub, ub,
        frame_id=expr.frame_id, exact=True)
    actual = _sparse_hz_storage_entries(hz)
    if actual > max_entries or actual > estimate:
        raise MemoryError('integrated final storage exceeds complete preflight')
    report.update(actual_hz_entries=actual, radix_auxiliaries=len(encoder.def_rows),
        packed_logical_rows=encoder.packed_rows, radix_relays=encoder.relays, actual_radix_work=encoder.extra_work,
        n_cont=hz.n_cont, n_bin=hz.n_bin, n_eq=hz.n_eq, n_ineq=hz.n_ineq,
        input_binding_unchanged=True, exact_power_two_coefficients=True,
        real_affine_program_identity=True, rounded_materialized_coefficient_identity_claimed=False)
    result = FusedIntegrated(expr, frozen, hz, nodes, root, old_nc, old_nb, base.n_eq, logical_nc, keep, report,
        *(np.asarray(items, dtype=np.int64) for items in (eq_roots, eq_scales, ineq_roots, ineq_scales, encoder.def_rows)))
    result.seal = result.fingerprint()
    result.validate()
    return result
