"""Exact-real shared affine DAG lifted into a genuine nonconvex HZ.

Default-off isolated prototype. No production binding, solver or model lookup.
Fresh continuous factors have exact triangular definitions; old binary and
continuous coordinates and predicates are retained. No long matrix products.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.solver.solver_hz import SparseHZono, sparse_pad_cols
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import live_value_rows, source_digest
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import operator_digest, digest_arrays
from experiments.neural_hz_20260831.c6_support_affine_plan_v2 import SupportEngine


def expression_binding(expr):
    if type(expr) is not cnn.SparseHZAffineExpr or not np.isfinite(expr.bias).all():
        raise ValueError('finite framed affine expression required')
    return (expr.frame_id, expr.n_out, digest_arrays(expr.bias), tuple(
        (id(t.source), source_digest(t.source), tuple((id(op), operator_digest(op)) for op in t.operators))
        for t in expr.terms))


def scaled_exact(values, shifts):
    values = np.asarray(values, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.int64)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        out = np.ldexp(values, shifts)
        back = np.ldexp(out, -shifts)
    if not np.isfinite(out).all() or not np.array_equal(back, values):
        raise ValueError('power-of-two scaling is not exactly reversible')
    return out


def box_exponent(values, parent_exponents=0):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or np.any(values == 0.) or not np.isfinite(values).all():
        raise ValueError('nonzero finite defining summands required')
    exponents = np.frexp(np.abs(values))[1].astype(np.int64) + parent_exponents
    exponent = int(exponents.max()) + (int(values.size) - 1).bit_length()
    if not -1074 <= exponent <= 1023:
        raise ValueError('auxiliary power-of-two box is not finite representable')
    return exponent


def restricted_row(op, row, keep):
    """Emit only retained exact input coefficients; never expand a Conv."""
    if type(op) is sp.csr_matrix:
        start, stop = op.indptr[row:row + 2]
        columns, values = op.indices[start:stop], op.data[start:stop]
        enabled = keep[columns] & (values != 0.)
        return columns[enabled], values[enabled]
    if type(op) is not ImplicitConv2DOp:
        raise ValueError('unsupported row operator')
    if op._row_mask is not None and not op._row_mask[row]:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    _, ci, hi, wi = op.input_shape
    _, co, ho, wo = op.output_shape
    b, within = divmod(int(row), co * ho * wo)
    oc, spatial = divmod(within, ho * wo)
    oh, ow = divmod(spatial, wo)
    cig = op._kernel.shape[1]
    group = oc // (co // op._groups)
    needed = keep.reshape(op.input_shape)[b, group * cig:(group + 1) * cig]
    columns, values = [], []
    for kh in range(op._kernel.shape[2]):
        ih = oh * op._stride[0] - op._padding[0] + kh * op._dilation[0]
        if not 0 <= ih < hi:
            continue
        for kw in range(op._kernel.shape[3]):
            iw = ow * op._stride[1] - op._padding[1] + kw * op._dilation[1]
            if not 0 <= iw < wi:
                continue
            channels = np.flatnonzero(needed[:, ih, iw] & (op._kernel[oc, :, kh, kw] != 0.))
            if channels.size:
                columns.append(((b * ci + group * cig + channels) * hi + ih) * wi + iw)
                values.append(op._kernel[oc, channels, kh, kw])
    if not columns:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    cols, vals = np.concatenate(columns), np.concatenate(values)
    order = np.argsort(cols)
    return cols[order], vals[order]


def _source_row(source, row):
    items = []
    for matrix in (source.Gc, source.Gb):
        start, stop = matrix.indptr[row:row + 2]
        columns, values = matrix.indices[start:stop], matrix.data[start:stop]
        enabled = values != 0.
        items.append((columns[enabled], values[enabled]))
    return items


def graph(expr, keep, max_work):
    nodes, intern = [], {}

    def node(kind, key, **fields):
        if key not in intern:
            intern[key] = len(nodes)
            nodes.append({'kind': kind, **fields})
        return intern[key]

    def factor(terms):
        groups = {}
        for source, ops in terms:
            key = ('op', id(ops[-1])) if ops else ('source', id(source))
            groups.setdefault(key, []).append((source, ops))
        children = []
        for key, members in groups.items():
            if key[0] == 'source':
                source = members[0][0]
                child = node('source', key, source=source, width=source.n_out, parents=())
                children.extend([child] * len(members))
            else:
                op = members[0][1][-1]
                parent = factor([(s, ops[:-1]) for s, ops in members])
                child = node('op', ('op_node', parent, id(op)), op=op, width=op.shape[0], parents=(parent,))
                children.append(child)
        if len(children) == 1:
            return children[0]
        if not children or len({nodes[c]['width'] for c in children}) != 1:
            raise ValueError('invalid exact sum node')
        return node('sum', ('sum', tuple(children)), parents=tuple(children), width=nodes[children[0]]['width'])

    root = factor([(t.source, t.operators) for t in expr.terms])
    if nodes[root]['width'] != expr.n_out:
        raise ValueError('factor graph output width mismatch')
    engine = SupportEngine(max_work)
    for n in nodes:
        before = engine.visits
        if n['kind'] == 'source':
            support = live_value_rows(n['source'])
        elif n['kind'] == 'op':
            support = engine.compute(n['op'], nodes[n['parents'][0]]['support']) != 0
        else:
            support = np.logical_or.reduce([nodes[p]['support'] for p in n['parents']])
        n.update(support=support, needed=np.zeros(n['width'], dtype=bool), support_work=engine.visits - before)
    nodes[root]['needed'] = keep & nodes[root]['support']
    for n in reversed(nodes):
        before = engine.visits
        if n['kind'] == 'op':
            p = nodes[n['parents'][0]]
            p['needed'] |= (engine.compute(n['op'], n['needed'], transpose=True) != 0) & p['support']
        elif n['kind'] == 'sum':
            for pid in n['parents']:
                nodes[pid]['needed'] |= n['needed'] & nodes[pid]['support']
        n['support_work'] += engine.visits - before
    counts = []
    path_costs = []
    for n in nodes:
        nc = nb = centers = 0
        rows = np.flatnonzero(n['needed'])
        if n['kind'] == 'source':
            s = n['source']
            for row in rows:
                c, b = _source_row(s, row)
                nc += c[1].size
                nb += b[1].size
            centers = int(np.count_nonzero(s.c[rows]))
        elif n['kind'] == 'op':
            p = nodes[n['parents'][0]]
            nc = int(engine.compute(n['op'], p['support'])[n['needed']].sum())
        else:
            nc = sum(int(np.count_nonzero(n['needed'] & nodes[p]['support'])) for p in set(n['parents']))
        cost = 8 * (nc + nb + centers + rows.size) + n['support_work']
        counts.append({'kind': n['kind'], 'width': n['width'], 'auxiliaries': int(rows.size),
            'continuous_edges': nc, 'binary_edges': nb, 'center_edges': centers,
            'encoding_work_upper': int(cost - n['support_work']), 'support_work': n['support_work']})
        path_costs.append(cost + max((path_costs[p] for p in n['parents']), default=0))
    total = engine.visits + sum(c['encoding_work_upper'] for c in counts)
    return nodes, root, {'node_counts': counts, 'support_work': engine.visits,
        'total_work_upper': total, 'largest_branch_work_upper': max(path_costs),
        'auxiliaries': sum(c['auxiliaries'] for c in counts),
        'continuous_edges': sum(c['continuous_edges'] for c in counts),
        'binary_edges': sum(c['binary_edges'] for c in counts),
        'temporary_support_cache_bytes': engine.cache_bytes}


@dataclass
class Lifted:
    expression: object
    origin_binding: tuple
    hz: object
    nodes: list
    root: int
    old_n_cont: int
    old_n_bin: int
    old_n_eq: int
    keep: np.ndarray
    report: dict
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {'expression', 'origin_binding', 'hz', 'nodes', 'root', 'old_n_cont',
                'old_n_bin', 'old_n_eq', 'keep', 'report', 'seal'}:
            raise ValueError('unknown lifted state field')
        h = hashlib.sha256(source_digest(self.hz).encode())
        h.update(repr((self.root, self.old_n_cont, self.old_n_bin, self.old_n_eq)).encode())
        h.update(digest_arrays(self.keep).encode())
        try:
            h.update(json.dumps(self.report, sort_keys=True, allow_nan=False).encode())
        except (TypeError, ValueError) as exc:
            raise ValueError('unregistered report payload') from exc
        for n in self.nodes:
            common = {'kind', 'width', 'parents', 'support', 'needed', 'support_work', 'slots', 'exponents'}
            expected = common | ({'source'} if n['kind'] == 'source' else {'op'} if n['kind'] == 'op' else set())
            if set(n) != expected:
                raise ValueError('unknown factor graph field')
            h.update(repr((n['kind'], n['width'], n['parents'])).encode())
            h.update(digest_arrays(n['support'], n['needed'], n['slots'], n['exponents']).encode())
            if n['kind'] == 'source':
                h.update(repr((id(n['source']), source_digest(n['source']))).encode())
            elif n['kind'] == 'op':
                h.update(repr((id(n['op']), operator_digest(n['op']))).encode())
        return h.hexdigest()

    def validate(self):
        if expression_binding(self.expression) != self.origin_binding or self.fingerprint() != self.seal:
            raise ValueError('lifted HZ or source-bound definition graph changed')

    def numeric_roots(self):
        self.validate()
        # All numeric data in this explicit schema are exposed, including the
        # reconstruction arrays and retained original operator/source graph.
        result = {'expression': self.expression, 'hz': self.hz, 'keep': self.keep}
        for i, n in enumerate(self.nodes):
            for key in ('support', 'needed', 'slots', 'exponents'):
                result[f'node{i}_{key}'] = n[key]
            if n['kind'] == 'source':
                result[f'node{i}_source'] = n['source']
            elif n['kind'] == 'op':
                result[f'node{i}_operator'] = n['op']
        return result


def lift(expr, keep_rows, *, enabled=False, frame_widths=None, max_work=256_000_000,
         max_branch_work=200_000_000, max_entries=64_000_000, observe=None):
    if not enabled:
        return None
    for cap, ceiling in ((max_work, 256_000_000), (max_branch_work, 200_000_000), (max_entries, 64_000_000)):
        if type(cap) is not int or not 0 <= cap <= ceiling:
            raise ValueError('invalid or increased construction ceiling')
    frozen = expression_binding(expr)
    keep = np.asarray(keep_rows)
    if keep.dtype != np.dtype(bool) or keep.shape != (expr.n_out,):
        raise ValueError('invalid selected output mask')
    keep = keep.copy()
    base = zero_transfer_reference(expr)
    old_nc, old_nb = base.n_cont, base.n_bin
    if frame_widths is not None:
        if (len(frame_widths) != 2 or any(type(v) is not int for v in frame_widths)
                or frame_widths[0] < old_nc or frame_widths[1] < old_nb):
            raise ValueError('cannot shrink or reinterpret the old frame')
        old_nc, old_nb = frame_widths
    nodes, root, report = graph(expr, keep, max_work)
    aux = report['auxiliaries']
    estimate = cnn._sparse_hz_storage_entries(base) + report['continuous_edges'] + report['binary_edges'] + 2 * aux + int(keep.sum())
    report.update(formal_gain=0, source_count=len({id(t.source) for t in expr.terms}), terms=len(expr.terms),
        node_count=len(nodes), input_binding_unchanged=False, estimated_hz_entries=int(estimate),
        default_off=True, live_publication_executed=False, whole_state_reduction_proved=False)
    if observe is not None:
        observe('preflight', dict(report))
    if report['total_work_upper'] > max_work or report['largest_branch_work_upper'] > max_branch_work:
        raise MemoryError('factored HZ complete work preflight rejected')
    if estimate > max_entries:
        raise MemoryError('factored HZ complete entry preflight rejected')
    # Float64 coefficients and int64 construction indices, plus simultaneous
    # CSR conversion/join copies. Measured construction remains authoritative.
    if 3 * (16 * (report['continuous_edges'] + report['binary_edges'] + aux) + 24 * aux) > 1024**3:
        raise MemoryError('factored HZ preallocation payload bound exceeds 1 GiB')
    ci = np.empty(report['continuous_edges'] + aux, dtype=np.int64)
    cd = np.empty(ci.size, dtype=np.float64)
    bi = np.empty(report['binary_edges'], dtype=np.int64)
    bd = np.empty(bi.size, dtype=np.float64)
    cp, bp, rhs = np.zeros(aux + 1, dtype=np.int64), np.zeros(aux + 1, dtype=np.int64), np.zeros(aux)
    cursor_c = cursor_b = row_id = 0
    for node_id, n in enumerate(nodes):
        rows = np.flatnonzero(n['needed'])
        n['slots'] = np.full(n['width'], -1, dtype=np.int64)
        n['exponents'] = np.zeros(n['width'], dtype=np.int32)
        n['slots'][rows] = old_nc + row_id + np.arange(rows.size)
        for row in rows:
            bcols, bvals = np.empty(0, dtype=np.int64), np.empty(0)
            constant = 0.
            if n['kind'] == 'source':
                s = n['source']
                (cols, vals), (bcols, bvals) = _source_row(s, row)
                constant = float(s.c[row])
                summands = np.concatenate((vals, bvals, [constant] if constant != 0. else []))
                exponent = box_exponent(summands)
                coefficients = -scaled_exact(vals, -exponent)
                binary_coefficients = -scaled_exact(bvals, -exponent)
                rhs[row_id] = scaled_exact(np.array([constant]), -exponent)[0]
            elif n['kind'] == 'op':
                p = nodes[n['parents'][0]]
                coordinates, vals = restricted_row(n['op'], row, p['needed'])
                cols = p['slots'][coordinates]
                if np.any(cols < 0):
                    raise ValueError('undefined parent factor')
                parent_exponents = p['exponents'][coordinates]
                exponent = box_exponent(vals, parent_exponents)
                coefficients = -scaled_exact(vals, parent_exponents.astype(np.int64) - exponent)
                binary_coefficients = bvals
            else:
                pairs = [(nodes[p]['slots'][row], nodes[p]['exponents'][row], count)
                    for p, count in sorted(Counter(n['parents']).items()) if nodes[p]['support'][row]]
                cols = np.asarray([p[0] for p in pairs], dtype=np.int64)
                if np.any(cols < 0) or any(p[2] > 2**53 for p in pairs):
                    raise ValueError('invalid exact sum multiplicity or parent')
                vals = np.asarray([p[2] for p in pairs], dtype=np.float64)
                parent_exponents = np.asarray([p[1] for p in pairs], dtype=np.int64)
                exponent = box_exponent(vals, parent_exponents)
                coefficients = -scaled_exact(vals, parent_exponents - exponent)
                binary_coefficients = bvals
            n['exponents'][row] = exponent
            end = cursor_c + cols.size
            ci[cursor_c:end], cd[cursor_c:end] = cols, coefficients
            ci[end], cd[end] = old_nc + row_id, 1.
            cursor_c = end + 1
            end = cursor_b + bcols.size
            bi[cursor_b:end], bd[cursor_b:end] = bcols, binary_coefficients
            cursor_b = end
            row_id += 1
            cp[row_id], bp[row_id] = cursor_c, cursor_b
        if observe is not None:
            observe('encoded_node', {'node': node_id, 'kind': n['kind'], 'auxiliaries': int(rows.size),
                'encoded_rows': row_id, 'continuous_entries': cursor_c, 'binary_entries': cursor_b})
    if (row_id, cursor_c, cursor_b) != (aux, ci.size, bi.size):
        raise ValueError('actual defining rows do not match complete preflight')
    nc = old_nc + aux
    added_c = sp.csr_matrix((cd, ci, cp), shape=(aux, nc))
    added_b = sp.csr_matrix((bd, bi, bp), shape=(aux, old_nb))
    if not added_c.has_canonical_format or not added_b.has_canonical_format:
        raise ValueError('definition matrices are not canonical')
    root_node = nodes[root]
    output_rows = np.flatnonzero(root_node['needed'])
    output_values = scaled_exact(np.ones(output_rows.size), root_node['exponents'][output_rows])
    gc = sp.csr_matrix((output_values, (output_rows, root_node['slots'][output_rows])), shape=(expr.n_out, nc))
    hz = SparseHZono(expr.bias.copy(), gc, sp.csr_matrix((expr.n_out, old_nb)),
        sp.vstack([sparse_pad_cols(base.Ac, nc), added_c], format='csr'),
        sp.vstack([sparse_pad_cols(base.Ab, old_nb), added_b], format='csr'),
        np.concatenate([base.b, rhs]), sparse_pad_cols(base.Auc, nc), sparse_pad_cols(base.Aub, old_nb),
        base.ub, frame_id=expr.frame_id, exact=True)
    actual_entries = cnn._sparse_hz_storage_entries(hz)
    if actual_entries > estimate or actual_entries > max_entries:
        raise ValueError('lifted HZ entries exceed preflight')
    if expression_binding(expr) != frozen:
        raise ValueError('incoming expression changed during lifting')
    report.update(actual_hz_entries=int(actual_entries), n_cont=hz.n_cont, n_bin=hz.n_bin,
        n_eq=hz.n_eq, n_ineq=hz.n_ineq, input_binding_unchanged=True,
        exact_power_two_coefficients=True, real_affine_program_identity=True,
        rounded_materialized_coefficient_identity_claimed=False)
    result = Lifted(expr, frozen, hz, nodes, root, old_nc, old_nb, base.n_eq, keep, report)
    result.seal = result.fingerprint()
    result.validate()
    return result
