"""Ordered coefficient demand union, isolated from production dispatch.

Matches the implicit row oracle's retained scalar association. Equivalence is
source-bound; this is not a stand-alone operator on arbitrary input vectors.
"""

import math
import time

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import _sparse_hz_storage_entries
from act.back_end.solver.solver_hz import (
    sparse_hz_linear, sparse_hz_add_same_frame, sparse_hz_add_const,
)
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import (
    BoundContraction, _rows, live_value_rows, source_digest,
)
from experiments.neural_hz_20260831.c5_live_value_union_contraction_v2 import _stationary


def contract(source, inner, middle_scale, outer, selected_rows, output_scale,
             *, max_products=200_000_000, max_nnz=64_000_000):
    started = time.monotonic()
    frozen = source_digest(source)
    if type(inner) is not ImplicitConv2DOp or type(outer) is not ImplicitConv2DOp:
        raise ValueError("two implicit Conv operators required")
    if inner._groups != 1 or outer._groups != 1:
        raise ValueError("C5-v3 supports ordinary Conv only")
    if inner.output_shape != outer.input_shape or source.n_out != inner.shape[1]:
        raise ValueError("source/intermediate shape mismatch")
    for cap in (max_products, max_nnz):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ValueError("invalid cap")
    rows = _rows(selected_rows, outer.shape[0])
    live = live_value_rows(source).reshape(inner.input_shape)
    sigma = _stationary(middle_scale, inner.output_shape, "middle scale").astype(np.float64)
    omega = _stationary(output_scale, outer.output_shape, "output scale").astype(np.float64)
    ai, bo = inner._kernel, outer._kernel
    if not np.isfinite(ai).all() or not np.isfinite(bo).all():
        raise ValueError("nonfinite kernel")
    _, ci, hi, wi = inner.input_shape
    _, cm, hm, wm = inner.output_shape
    _, co, ho, wo = outer.output_shape
    middle = np.arange(cm) if inner._row_mask is None else np.flatnonzero(
        _stationary(inner._row_mask, inner.output_shape, "inner mask"))

    def visits():
        for row_index, row in enumerate(rows):
            if outer._row_mask is not None and not outer._row_mask[row]:
                continue
            batch, within = divmod(int(row), co * ho * wo)
            oc, spatial = divmod(within, ho * wo)
            oh, ow = divmod(spatial, wo)
            points = {}
            for th in range(bo.shape[2]):
                mh = oh * outer._stride[0] - outer._padding[0] + th * outer._dilation[0]
                if not 0 <= mh < hm:
                    continue
                for tw in range(bo.shape[3]):
                    mw = ow * outer._stride[1] - outer._padding[1] + tw * outer._dilation[1]
                    if not 0 <= mw < wm:
                        continue
                    for qh in range(ai.shape[2]):
                        ih = mh * inner._stride[0] - inner._padding[0] + qh * inner._dilation[0]
                        if not 0 <= ih < hi:
                            continue
                        for qw in range(ai.shape[3]):
                            iw = mw * inner._stride[1] - inner._padding[1] + qw * inner._dilation[1]
                            if 0 <= iw < wi:
                                points.setdefault((ih, iw), []).append((th, tw, qh, qw))
            for (ih, iw), pattern in points.items():
                channels = np.flatnonzero(live[batch, :, ih, iw])
                yield row_index, batch, ih, iw, (oc, tuple(pattern)), channels

    demands = {}
    baseline = spatial_products = demanded_entries = upper_products = visits_count = 0
    for _, _, _, _, key, channels in visits():
        visits_count += 1
        cost = len(key[1]) * int(middle.size)
        baseline += cost * ci
        spatial_products += cost * channels.size
        if not channels.size or not middle.size:
            continue
        if key not in demands:
            # Bound numeric metadata before allocating another full channel mask.
            if (len(demands) + 1) * ci * 9 + live.nbytes + rows.nbytes > 1024**3:
                raise MemoryError("C5-v3 numeric demand cache cap exceeded")
            demands[key] = np.zeros(ci, dtype=bool)
        mask = demands[key]
        fresh = int(np.count_nonzero(~mask[channels]))
        demanded_entries += fresh
        upper_products += fresh * cost
        if demanded_entries > 2_000_000 or upper_products > max_products:
            raise MemoryError("C5-v3 demanded coefficient/product cap exceeded")
        mask[channels] = True

    cache, weights = {}, {}
    products = scale_products = additions = peak_workspace = 0
    for key in sorted(demands):
        oc, pattern = key
        channels = np.flatnonzero(demands[key])
        sums = np.zeros(channels.size, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            for m in middle:
                for th, tw, qh, qw in pattern:
                    weight_key = (oc, int(m), th, tw)
                    if weight_key not in weights:
                        first = float(omega[oc]) * float(bo[oc, m, th, tw])
                        weight = first * float(sigma[m])
                        if not math.isfinite(first) or not math.isfinite(weight):
                            raise ValueError("nonfinite ordered scale product")
                        weights[weight_key] = weight
                        scale_products += 2
                    weight = weights[weight_key]
                    if weight == 0.:
                        continue
                    contribution = weight * ai[m, channels, qh, qw]
                    sums += contribution
                    sums[sums == 0.] = 0.  # Original dictionary deletes cancelled zeros.
                    products += channels.size
                    additions += channels.size
        values = np.zeros(ci, dtype=np.float64)
        values[channels] = sums
        cache[key] = values
        peak_workspace = max(peak_workspace, channels.nbytes + sums.nbytes * 3)
    if products > upper_products:
        raise ValueError("ordered product ledger mismatch")

    row_entries = [[] for _ in rows]
    emitted = 0
    for row_index, batch, ih, iw, key, channels in visits():
        if key not in cache:
            continue
        values = cache[key][channels]
        for channel, value in zip(channels, values, strict=True):
            if value != 0.:
                column = ((batch * ci + int(channel)) * hi + ih) * wi + iw
                row_entries[row_index].append((column, float(value)))
                emitted += 1
                if emitted > max_nnz:
                    raise MemoryError("C5-v3 emitted nnz cap exceeded")
    indptr, indices, data = [0], [], []
    for entries in row_entries:
        entries.sort()
        for column, value in entries:
            indices.append(column)
            data.append(value)
        indptr.append(len(data))
    matrix = sp.csr_matrix((np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64),
                            np.asarray(indptr, dtype=np.int64)), shape=(rows.size, source.n_out))
    if not matrix.has_canonical_format or not np.isfinite(matrix.data).all():
        raise ValueError("noncanonical/nonfinite emission")
    if source_digest(source) != frozen:
        raise ValueError("source changed during compilation")
    cache_bytes = sum(a.nbytes for a in cache.values()) + sum(a.nbytes for a in demands.values()) + 8 * len(weights)
    matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    if cache_bytes + matrix_bytes + live.nbytes + rows.nbytes + peak_workspace > 1024**3:
        raise MemoryError("C5-v3 numeric payload cap exceeded")
    return BoundContraction(source, frozen, rows, matrix, {
        "actual_channel_products": products, "channel_product_upper_bound": upper_products,
        "unrestricted_spatial_channel_products": int(baseline), "source_restricted_spatial_products": int(spatial_products),
        "scale_products": scale_products, "additions": additions, "ordered_patterns": len(cache),
        "demanded_coefficients": demanded_entries, "spatial_group_visits": visits_count * 2,
        "source_live_rows": int(live.sum()), "source_total_rows": source.n_out,
        "compile_cache_numeric_bytes": cache_bytes, "compile_cache_retained_after_return": False,
        "source_support_bytes": live.nbytes, "selected_row_bytes": rows.nbytes,
        "emitted_nnz": matrix.nnz, "emitted_csr_numeric_bytes": matrix_bytes,
        "peak_dot_numeric_workspace_bytes": peak_workspace,
        "quarter_product_gate": upper_products * 4 <= baseline if baseline else False,
        "whole_state_reduction_proved": False, "source_and_predicates_retained": True,
        "python_workspace_in_numeric_bytes": False, "elapsed_s": time.monotonic() - started,
    })


def materialize(expr, rows, *, max_products=256_000_000, max_nnz=64_000_000, observe=None):
    """Complete ordered-source transaction, no cache publication or fallback."""
    rows = _rows(rows, expr.n_out)
    grouped, records = {}, []
    remaining = max_products
    if not expr.terms or not np.isfinite(expr.bias).all():
        raise ValueError("invalid complete expression")
    for term in expr.terms:
        if term.source.frame_id != expr.frame_id or len(term.operators) != 4:
            raise ValueError("unsupported term/frame")
        inner, middle, outer, output = term.operators
        scales = []
        for diagonal in (middle, output):
            if not sp.issparse(diagonal) or diagonal.shape[0] != diagonal.shape[1]:
                raise ValueError("CSR diagonal required")
            scale = diagonal.diagonal()
            difference = (diagonal - sp.diags(scale, format="csr")).tocsr()
            difference.eliminate_zeros()
            if difference.nnz:
                raise ValueError("nondiagonal affine operator")
            scales.append(scale)
        if outer.shape[0] != expr.n_out:
            raise ValueError("output dimension mismatch")
        bound = contract(term.source, inner, scales[0], outer, rows, scales[1],
                         max_products=min(200_000_000, remaining), max_nnz=max_nnz)
        if observe is not None:
            # Shadow-only retention is charged separately, never a runtime gate.
            observe(len(records), bound)
        remaining -= bound.stats["channel_product_upper_bound"]
        compact = bound.matrix.tocoo()
        operator = sp.csr_matrix((compact.data, (rows[compact.row], compact.col)),
                                 shape=(expr.n_out, term.source.n_out))
        key = id(term.source)
        if key in grouped:
            operator = (grouped[key][1] + operator).tocsr()
            operator.eliminate_zeros()
        if operator.nnz > max_nnz:
            raise MemoryError("coalesced operator cap exceeded")
        grouped[key] = (term.source, operator)
        records.append(bound.stats)
    parts = []
    for source, operator in grouped.values():
        part = sparse_hz_linear(source, operator)
        if _sparse_hz_storage_entries(part) > max_nnz:
            raise MemoryError("materialized term cap exceeded")
        parts.append(part)
    out = parts[0]
    for part in parts[1:]:
        out = sparse_hz_add_same_frame(out, part)
        if _sparse_hz_storage_entries(out) > max_nnz:
            raise MemoryError("joined HZ cap exceeded")
    return sparse_hz_add_const(out, expr.bias), records
