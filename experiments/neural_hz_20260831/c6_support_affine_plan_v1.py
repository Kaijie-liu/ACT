"""Source-bound structural support proofs; isolated, default-off research.

No Conv expansion, model lookup, solver query, or production dispatch here.
The scalar realization is a small-test oracle, not a qualified live backend.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp, _left_compose_rows
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.solver.solver_hz import sparse_hz_linear, sparse_hz_add_same_frame, sparse_hz_add_const
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.asarray(value)
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def operator_digest(op):
    if type(op) is ImplicitConv2DOp:
        if not np.isfinite(op._kernel).all():
            raise ValueError('nonfinite Conv kernel')
        mask = np.ones(0, dtype=bool) if op._row_mask is None else op._row_mask
        if mask.dtype != np.dtype(bool) or (op._row_mask is not None and mask.shape != (op.shape[0],)):
            raise ValueError('invalid Conv row mask')
        geometry = (op.input_shape, op.output_shape, op._stride, op._padding, op._dilation, op._groups)
        return hashlib.sha256((repr(geometry) + digest_arrays(op._kernel, mask)).encode()).hexdigest()
    if type(op) is sp.csr_matrix:
        if op.dtype != np.dtype(np.float64) or not op.has_canonical_format or not np.isfinite(op.data).all():
            raise ValueError('finite canonical float64 CSR required')
        return hashlib.sha256((repr(op.shape) + digest_arrays(op.data, op.indices, op.indptr)).encode()).hexdigest()
    raise ValueError('unsupported affine operator')


class SupportEngine:
    """Exact integer support operations with bounded and memoized geometry work."""

    def __init__(self, max_visits=256_000_000):
        if type(max_visits) is not int or max_visits < 0:
            raise ValueError('invalid support visit cap')
        self.max_visits = max_visits
        self.visits = 0
        self.cache = {}
        self.cache_bytes = 0
        self.hits = 0

    def charge(self, count):
        if self.visits + count > self.max_visits:
            raise MemoryError('support integer edge-visit cap exceeded')
        self.visits += count

    def compute(self, op, mask, *, transpose=False):
        sha = operator_digest(op)
        mask = np.asarray(mask)
        expected = op.shape[0 if transpose else 1]
        if mask.dtype.kind not in 'biu' or mask.shape != (expected,) or np.any(mask < 0) or np.any(mask > 256_000_000):
            raise ValueError('invalid support shape or dtype')
        key = (id(op), sha, transpose, digest_arrays(mask))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        nout = op.shape[1 if transpose else 0]
        # The same shape rule handles zero sources and zero intermediate maps.
        if not mask.any():
            counts = np.zeros(nout, dtype=np.int64)
        elif type(op) is sp.csr_matrix:
            self.charge(op.nnz)
            if transpose:
                multiplicities = np.repeat(mask.astype(np.int64), np.diff(op.indptr))
                multiplicities[op.data == 0.] = 0
                counts = np.zeros(nout, dtype=np.int64)
                np.add.at(counts, op.indices, multiplicities)
            else:
                edges = (op.data != 0.) * mask[op.indices].astype(np.int64)
                cumulative = np.empty(edges.size + 1, dtype=np.int64)
                cumulative[0] = 0
                np.cumsum(edges, dtype=np.int64, out=cumulative[1:])
                counts = np.diff(cumulative[op.indptr])
        else:
            counts = self._conv_counts(op, mask, transpose)
        counts.flags.writeable = False
        if self.cache_bytes + counts.nbytes > 1024**3:
            raise MemoryError('support numeric cache exceeds 1 GiB')
        self.cache_bytes += counts.nbytes
        self.cache[key] = counts
        return counts

    def _conv_counts(self, op, mask, transpose):
        batch, ci, hi, wi = op.input_shape
        _, co, ho, wo = op.output_shape
        _, cig, khn, kwn = op._kernel.shape
        cog = co // op._groups
        counts = np.zeros(op.input_shape if transpose else op.output_shape, dtype=np.int64)
        selected = mask.reshape(op.output_shape if transpose else op.input_shape)
        out_mask = None if op._row_mask is None else op._row_mask.reshape(op.output_shape)
        for kh in range(khn):
            oh = np.arange(ho)
            ih = oh * op._stride[0] - op._padding[0] + kh * op._dilation[0]
            valid = (ih >= 0) & (ih < hi)
            oh, ih = oh[valid], ih[valid]
            for kw in range(kwn):
                ow = np.arange(wo)
                iw = ow * op._stride[1] - op._padding[1] + kw * op._dilation[1]
                valid = (iw >= 0) & (iw < wi)
                ow, iw = ow[valid], iw[valid]
                out_h, out_w = np.repeat(oh, ow.size), np.tile(ow, oh.size)
                in_h, in_w = np.repeat(ih, iw.size), np.tile(iw, ih.size)
                if not out_h.size:
                    continue
                for b in range(batch):
                    for g in range(op._groups):
                        ic, oc = slice(g * cig, (g + 1) * cig), slice(g * cog, (g + 1) * cog)
                        weights = (op._kernel[oc, :, kh, kw] != 0.).astype(np.int64)
                        values = (selected[b, oc][:, out_h, out_w] if transpose
                                  else selected[b, ic][:, in_h, in_w])
                        if transpose and out_mask is not None:
                            values = values * out_mask[b, oc][:, out_h, out_w]
                        if not values.any() or not weights.any():
                            continue
                        self.charge(int(cig * cog * out_h.size))
                        if transpose:
                            counts[b, ic][:, in_h, in_w] += weights.T @ values.astype(np.int64)
                        else:
                            counts[b, oc][:, out_h, out_w] += weights @ values.astype(np.int64)
        if not transpose and out_mask is not None:
            counts[~out_mask] = 0
        return counts.reshape(-1)


@dataclass(frozen=True)
class BoundProgram:
    expression: object
    source_bindings: tuple
    operator_bindings: tuple
    expression_binding: tuple
    supports: tuple
    support_sha256: str
    report: dict

    def validate(self):
        expr = self.expression
        current = (expr.n_out, expr.frame_id, digest_arrays(expr.bias),
                   tuple((id(t.source), tuple(id(op) for op in t.operators)) for t in expr.terms))
        if current != self.expression_binding:
            raise ValueError('expression changed after support proof')
        for source, sha in self.source_bindings:
            if source_digest(source) != sha:
                raise ValueError('source changed after support proof')
        for op, sha in self.operator_bindings:
            if operator_digest(op) != sha:
                raise ValueError('operator changed after support proof')
        if digest_arrays(*(s for term in self.supports for s in term)) != self.support_sha256:
            raise ValueError('support certificate changed')


def plan(expr, keep_rows, *, max_visits=256_000_000, observe=None):
    if type(expr) is not cnn.SparseHZAffineExpr or not expr.terms or not np.isfinite(expr.bias).all():
        raise ValueError('finite complete exact affine expression required')
    keep = np.asarray(keep_rows)
    if keep.dtype != np.dtype(bool) or keep.shape != (expr.n_out,):
        raise ValueError('invalid selected output rows')
    selected_count = int(keep.sum())
    engine, sources, operators, supports, records = SupportEngine(max_visits), {}, {}, [], []
    for term in expr.terms:
        source = term.source
        if source.frame_id != expr.frame_id:
            raise ValueError('shared frame mismatch')
        if id(source) not in sources:
            sources[id(source)] = (source, source_digest(source))
        masks = [live_value_rows(source)]
        width, degrees = source.n_out, []
        for op in term.operators:
            if op.shape[1] != width:
                raise ValueError('affine operator shape mismatch')
            operators[id(op)] = (op, operator_digest(op))
            counts = engine.compute(op, masks[-1])
            degrees.append(counts)
            masks.append(counts != 0)
            width = op.shape[0]
        if width != expr.n_out:
            raise ValueError('affine output width mismatch')
        for mask in masks:
            mask.flags.writeable = False
        supports.append(tuple(masks))
        multiplicities = (keep & masks[-1]).astype(np.int64)
        stages = []
        for position in range(len(term.operators) - 1, -1, -1):
            op = term.operators[position]
            required = multiplicities != 0
            edges = int(degrees[position][required].sum())
            products = int(np.dot(degrees[position], multiplicities))
            input_multiplicities = np.minimum(selected_count, engine.compute(op, multiplicities, transpose=True))
            input_multiplicities = input_multiplicities * masks[position]
            input_required = input_multiplicities != 0
            stages.append({'position': position, 'type': type(op).__name__, 'shape': list(op.shape),
                'forward_input_support': int(masks[position].sum()),
                'forward_output_support': int(masks[position + 1].sum()),
                'required_outputs': int(required.sum()), 'required_inputs': int(input_required.sum()),
                'retained_edges': edges, 'product_upper_bound': products,
                'max_output_multiplicity': int(multiplicities.max(initial=0))})
            multiplicities = input_multiplicities
        total = sum(stage['product_upper_bound'] for stage in stages)
        records.append({'term_index': len(records), 'source_index': list(sources).index(id(source)),
            'source_live_rows': int(masks[0].sum()), 'stages': list(reversed(stages)),
            'product_upper_bound': total, 'within_200m_branch_cap': total <= 200_000_000})
        if observe is not None:
            observe(json.loads(json.dumps(records[-1])), engine.visits)
    total = sum(record['product_upper_bound'] for record in records)
    binding = (expr.n_out, expr.frame_id, digest_arrays(expr.bias),
               tuple((id(t.source), tuple(id(op) for op in t.operators)) for t in expr.terms))
    report = {'schema': 'c6_support_affine_plan_v1', 'formal_gain': 0, 'numerical_contraction_executed': False,
        'live_publication_executed': False, 'whole_state_reduction_proved': False,
        'quarter_work_proved': False, 'selected_output_rows': selected_count, 'terms': records,
        'unique_source_count': len(sources), 'unique_operator_count': len(operators),
        'support_integer_visits': engine.visits, 'support_cache_hits': engine.hits,
        'temporary_support_cache_bytes': engine.cache_bytes,
        'retained_support_bytes': sum(mask.nbytes for masks in supports for mask in masks),
        'uncached_total_product_upper_bound': total, 'suffix_reuse_discount_applied': False,
        'all_branch_caps_certified': all(r['within_200m_branch_cap'] for r in records),
        'whole_256m_product_cap_certified': total <= 256_000_000,
        'interpretation': 'union-support bound; excess is failure to certify, not an executed-work lower bound'}
    bound = BoundProgram(expr, tuple(sources.values()), tuple(operators.values()), binding,
        tuple(supports), digest_arrays(*(s for masks in supports for s in masks)), report)
    bound.validate()
    return bound


def scalar_test_materialize(bound, keep_rows, *, max_nnz=64_000_000):
    """Small-test reference preserving the original scalar/SciPy association.

    Unqualified for real-world dispatch; it intentionally does not optimize
    repeated prefixes. The source-bound matrices are never exposed for reuse.
    """
    bound.validate()
    expr = bound.expression
    keep = np.asarray(keep_rows)
    if keep.dtype != np.dtype(bool) or keep.shape != (expr.n_out,):
        raise ValueError('invalid selected output rows')
    grouped = {}
    for term, masks in zip(expr.terms, bound.supports, strict=True):
        q = sp.diags((keep & masks[-1]).astype(np.float64), format='csr')
        for position in range(len(term.operators) - 1, -1, -1):
            op, live = term.operators[position], masks[position]
            if type(op) is ImplicitConv2DOp:
                def row(index):
                    columns, values = op._row(index)
                    select = live[columns]
                    return columns[select], values[select]
                q = _left_compose_rows(q, operator_shape=op.shape, row=row, max_nnz=max_nnz)
            else:
                filtered = op.copy()
                filtered.data[~live[filtered.indices]] = 0.
                filtered.eliminate_zeros()
                q = cnn._lazy_left_compose(q, filtered, max_nnz)
        key = id(term.source)
        if key in grouped:
            q = (grouped[key][1] + q).tocsr()
            q.eliminate_zeros()
        if q.nnz > max_nnz:
            raise MemoryError('coalesced source map cap exceeded')
        grouped[key] = (term.source, q)
    out = None
    for source, matrix in grouped.values():
        part = sparse_hz_linear(source, matrix)
        out = part if out is None else sparse_hz_add_same_frame(out, part)
        if cnn._sparse_hz_storage_entries(out) > max_nnz:
            raise MemoryError('complete predicate join cap exceeded')
    bound.validate()
    return sparse_hz_add_const(out, expr.bias)
