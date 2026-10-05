"""Exact coefficient-demand union; same source-restricted scalar map as C5-v1."""

import math
import time

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import (
    BoundContraction, _rows, live_value_rows, source_digest,
)


def _stationary(array, shape, name):
    array = np.asarray(array)
    if array.size != math.prod(shape) or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}")
    shaped = array.reshape(shape)
    channels = shaped[0, :, 0, 0].copy()
    if not np.array_equal(shaped, np.broadcast_to(channels[None, :, None, None], shape)):
        raise ValueError(f"nonstationary {name}")
    return channels


def contract(source, inner, middle_scale, outer, selected_rows, *, max_products=200_000_000, max_nnz=64_000_000):
    start = time.monotonic()
    frozen = source_digest(source)
    if type(inner) is not ImplicitConv2DOp or type(outer) is not ImplicitConv2DOp:
        raise ValueError("two implicit Conv operators required")
    if inner._groups != 1 or outer._groups != 1:
        raise ValueError("C5-v2 supports ordinary Conv only")
    if inner.output_shape != outer.input_shape or source.n_out != inner.shape[1]:
        raise ValueError("source/intermediate shape mismatch")
    for cap in (max_products, max_nnz):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ValueError("invalid cap")
    rows = _rows(selected_rows, outer.shape[0])
    live = live_value_rows(source).reshape(inner.input_shape)
    sigma = _stationary(middle_scale, inner.output_shape, "middle scale").astype(np.float64)
    ai, bo = inner._kernel, outer._kernel
    if not np.isfinite(ai).all() or not np.isfinite(bo).all():
        raise ValueError("nonfinite kernel")
    _, ci, hi, wi = inner.input_shape
    _, cm, hm, wm = inner.output_shape
    _, co, ho, wo = outer.output_shape
    middle = np.arange(cm) if inner._row_mask is None else np.flatnonzero(
        _stationary(inner._row_mask, inner.output_shape, "inner mask"))
    ki_h, ki_w = ai.shape[2:]
    ko_h, ko_w = bo.shape[2:]

    def visits():
        for row_index, row in enumerate(rows):
            if outer._row_mask is not None and not outer._row_mask[row]:
                continue
            batch, within = divmod(int(row), co * ho * wo)
            oc, spatial = divmod(within, ho * wo)
            oh, ow = divmod(spatial, wo)
            for th in range(ko_h):
                mh = oh * outer._stride[0] - outer._padding[0] + th * outer._dilation[0]
                if not 0 <= mh < hm:
                    continue
                for tw in range(ko_w):
                    mw = ow * outer._stride[1] - outer._padding[1] + tw * outer._dilation[1]
                    if not 0 <= mw < wm:
                        continue
                    for qh in range(ki_h):
                        ih = mh * inner._stride[0] - inner._padding[0] + qh * inner._dilation[0]
                        if not 0 <= ih < hi:
                            continue
                        for qw in range(ki_w):
                            iw = mw * inner._stride[1] - inner._padding[1] + qw * inner._dilation[1]
                            if 0 <= iw < wi:
                                channels = np.flatnonzero(live[batch, :, ih, iw])
                                yield row_index, batch, ih, iw, (oc, th, tw, qh, qw), channels

    demands = {}
    baseline = v1_products = spatial_visits = demanded_entries = 0
    for _, _, _, _, key, channels in visits():
        spatial_visits += 1
        baseline += middle.size * ci
        v1_products += middle.size * channels.size
        if not channels.size or not middle.size:
            continue
        mask = demands.setdefault(key, np.zeros(ci, dtype=bool))
        fresh = int(np.count_nonzero(~mask[channels]))
        demanded_entries += fresh
        if demanded_entries > 2_000_000 or demanded_entries * middle.size > max_products:
            raise MemoryError("C5-v2 demanded coefficient/product cap exceeded")
        mask[channels] = True
    cache, products, sigma_products, additions = {}, 0, 0, 0
    peak_workspace = 0
    for key in sorted(demands):
        channels = np.flatnonzero(demands[key])
        oc, th, tw, qh, qw = key
        values = np.zeros(ci, dtype=np.float64)
        sums = np.zeros(channels.size, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            for m in middle:
                weight = bo[oc, m, th, tw] * sigma[m]
                sums += weight * ai[m, channels, qh, qw]
                products += channels.size
                sigma_products += 1
                additions += channels.size
        values[channels] = sums
        cache[key] = values
        peak_workspace = max(peak_workspace, channels.nbytes + sums.nbytes * 2)
    if products != demanded_entries * middle.size:
        raise ValueError("demand/product ledger mismatch")
    accumulators = [{} for _ in rows]
    accumulated_entries = 0
    for row_index, batch, ih, iw, key, channels in visits():
        if not channels.size or key not in cache:
            continue
        accumulator = accumulators[row_index]
        values = cache[key][channels]
        for channel, value in zip(channels, values, strict=True):
            column = ((batch * ci + int(channel)) * hi + ih) * wi + iw
            existed = column in accumulator
            total = accumulator.get(column, 0.) + float(value)
            additions += 1
            if not math.isfinite(total):
                raise ValueError("nonfinite spatial accumulation")
            if total == 0.:
                if existed:
                    del accumulator[column]
                    accumulated_entries -= 1
            else:
                accumulator[column] = total
                accumulated_entries += int(not existed)
            if accumulated_entries > max_nnz:
                raise MemoryError("C5-v2 emitted nnz cap exceeded")
    indptr, indices, data = [0], [], []
    for accumulator in accumulators:
        for column in sorted(accumulator):
            indices.append(column)
            data.append(accumulator[column])
        indptr.append(len(data))
    matrix = sp.csr_matrix((np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64),
                            np.asarray(indptr, dtype=np.int64)), shape=(rows.size, source.n_out))
    if source_digest(source) != frozen:
        raise ValueError("source changed during compilation")
    cache_bytes = sum(a.nbytes for a in cache.values()) + sum(a.nbytes for a in demands.values())
    if cache_bytes + matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes + live.nbytes + rows.nbytes > 1024**3:
        raise MemoryError("C5-v2 numeric payload cap exceeded")
    stats = {"actual_channel_products": products, "unrestricted_spatial_channel_products": int(baseline),
             "v1_spatial_products": int(v1_products), "skipped_channel_products": int(baseline) - products,
             "sigma_products": sigma_products, "additions": additions, "spatial_visits": spatial_visits * 2,
             "spatial_enumeration_passes": 2, "demanded_coefficients": demanded_entries,
             "coefficient_keys": len(cache), "compile_cache_numeric_bytes": cache_bytes,
             "compile_cache_retained_after_return": False, "source_live_rows": int(live.sum()),
             "source_total_rows": source.n_out, "source_support_bytes": live.nbytes, "selected_row_bytes": rows.nbytes,
             "emitted_nnz": matrix.nnz, "emitted_csr_numeric_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
             "peak_dot_numeric_workspace_bytes": peak_workspace,
             "quarter_product_gate": products * 4 <= baseline if baseline else False,
             "whole_state_reduction_proved": False, "source_and_predicates_retained": True,
             "python_workspace_in_numeric_bytes": False, "elapsed_s": time.monotonic() - start}
    return BoundContraction(source, frozen, rows, matrix, stats)
