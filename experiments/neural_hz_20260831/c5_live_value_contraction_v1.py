"""Source-bound exact HZ value-support contraction; isolated, not integrated.

The compiled matrix is equivalent only on its bound source HZ, not on arbitrary
vectors. No predicate, latent column, or binary phase is eliminated.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import time

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear


def source_digest(source):
    if type(source) is not SparseHZono or source.exact is not True or source.frame_id is None:
        raise ValueError("source must be an exact framed SparseHZono")
    digest = hashlib.sha256(json.dumps([source.frame_id, source.n_out, source.n_cont, source.n_bin]).encode())
    for name in ("c", "Gc", "Gb", "Ac", "Ab", "b", "Auc", "Aub", "ub"):
        value = getattr(source, name)
        arrays = (value.data, value.indices, value.indptr) if sp.issparse(value) else (value,)
        for a in arrays:
            if not np.isfinite(a).all():
                raise ValueError("source contains nonfinite payload")
            digest.update(str(a.shape).encode())
            digest.update(str(a.dtype).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def live_value_rows(source):
    source_digest(source)
    live = source.c != 0.
    for matrix in (source.Gc, source.Gb):
        canonical = matrix.copy()
        canonical.eliminate_zeros()
        live |= np.diff(canonical.indptr) != 0
    return live


def _rows(rows, total):
    raw = np.asarray(rows)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("rows must be an integer vector")
    out = raw.astype(np.int64, copy=True)
    if np.any(out < 0) or np.any(out >= total) or np.unique(out).size != out.size:
        raise ValueError("rows must be unique and in range")
    return out


@dataclass(frozen=True)
class BoundContraction:
    source: SparseHZono
    source_sha256: str
    selected_rows: np.ndarray
    matrix: sp.csr_matrix
    stats: dict

    def apply(self, bias=None):
        if source_digest(self.source) != self.source_sha256:
            raise ValueError("source changed after compilation")
        if bias is not None and not np.isfinite(np.asarray(bias)).all():
            raise ValueError("nonfinite output bias")
        return sparse_hz_linear(self.source, self.matrix, bias)


def contract(source, inner, middle_scale, outer, selected_rows, *, max_products=200_000_000, max_nnz=64_000_000):
    """Compute R B D A P_live, pruning dead source coordinates before each dot.

    The numerical association is ascending middle-channel sums, then spatial
    tap sums. It is an exact-real identity; this is not a general floating-point
    outward-rounding proof. Whole-runtime HZ integration remains gated.
    """
    start = time.monotonic()
    frozen = source_digest(source)
    if type(inner) is not ImplicitConv2DOp or type(outer) is not ImplicitConv2DOp:
        raise ValueError("two implicit Conv operators required")
    if inner._groups != 1 or outer._groups != 1:
        raise ValueError("C5-v1 supports ordinary Conv only")
    if inner.output_shape != outer.input_shape or source.n_out != inner.shape[1]:
        raise ValueError("source/intermediate shape mismatch")
    for cap in (max_products, max_nnz):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ValueError("invalid cap")
    rows = _rows(selected_rows, outer.shape[0])
    live = live_value_rows(source).reshape(inner.input_shape)
    scale = np.asarray(middle_scale, dtype=np.float64)
    if scale.size != math.prod(inner.output_shape) or not np.isfinite(scale).all():
        raise ValueError("invalid complete middle diagonal")
    scale = scale.reshape(inner.output_shape)
    ai, bo = inner._kernel, outer._kernel
    if not np.isfinite(ai).all() or not np.isfinite(bo).all():
        raise ValueError("nonfinite kernel")
    batch, ci, ih_size, iw_size = inner.input_shape
    _, cm, mh_size, mw_size = inner.output_shape
    _, co, oh_size, ow_size = outer.output_shape
    inner_mask = None if inner._row_mask is None else inner._row_mask.reshape(inner.output_shape)
    out_mask = outer._row_mask
    kh_i, kw_i = ai.shape[2:]
    kh_o, kw_o = bo.shape[2:]
    indptr, indices, data = [0], [], []
    actual_products = baseline_products = spatial_visits = sigma_products = additions = 0
    peak_dot_workspace = 0
    for row in rows:
        accumulator = {}
        b, within = divmod(int(row), co * oh_size * ow_size)
        oc, spatial = divmod(within, oh_size * ow_size)
        oh, ow = divmod(spatial, ow_size)
        if out_mask is not None and not out_mask[row]:
            indptr.append(len(data))
            continue
        for th in range(kh_o):
            mh = oh * outer._stride[0] - outer._padding[0] + th * outer._dilation[0]
            if not 0 <= mh < mh_size:
                continue
            for tw in range(kw_o):
                mw = ow * outer._stride[1] - outer._padding[1] + tw * outer._dilation[1]
                if not 0 <= mw < mw_size:
                    continue
                mid_live = np.ones(cm, dtype=bool) if inner_mask is None else inner_mask[b, :, mh, mw]
                for qh in range(kh_i):
                    ih = mh * inner._stride[0] - inner._padding[0] + qh * inner._dilation[0]
                    if not 0 <= ih < ih_size:
                        continue
                    for qw in range(kw_i):
                        iw = mw * inner._stride[1] - inner._padding[1] + qw * inner._dilation[1]
                        if not 0 <= iw < iw_size:
                            continue
                        spatial_visits += 1
                        baseline_products += int(mid_live.sum()) * ci
                        channels = np.flatnonzero(live[b, :, ih, iw])
                        retained = int(mid_live.sum()) * channels.size
                        if actual_products + retained > max_products:
                            raise MemoryError("C5 product cap exceeded")
                        actual_products += retained
                        if not channels.size:
                            continue
                        values = np.zeros(channels.size, dtype=np.float64)
                        with np.errstate(over="raise", invalid="raise"):
                            for m in np.flatnonzero(mid_live):
                                weight = bo[oc, m, th, tw] * scale[b, m, mh, mw]
                                values += weight * ai[m, channels, qh, qw]
                                sigma_products += 1
                                additions += channels.size
                        peak_dot_workspace = max(peak_dot_workspace, channels.nbytes + values.nbytes * 2)
                        for channel, value in zip(channels, values, strict=True):
                            column = ((b * ci + int(channel)) * ih_size + ih) * iw_size + iw
                            total = accumulator.get(column, 0.) + float(value)
                            additions += 1
                            if not math.isfinite(total):
                                raise ValueError("nonfinite spatial accumulation")
                            if total == 0.:
                                accumulator.pop(column, None)
                            else:
                                accumulator[column] = total
                            if len(data) + len(accumulator) > max_nnz:
                                raise MemoryError("C5 result nnz cap exceeded")
        for column in sorted(accumulator):
            indices.append(column)
            data.append(accumulator[column])
        indptr.append(len(data))
    matrix = sp.csr_matrix((np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64),
                            np.asarray(indptr, dtype=np.int64)), shape=(rows.size, source.n_out))
    if source_digest(source) != frozen:
        raise ValueError("source mutated during contraction")
    matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return BoundContraction(source, frozen, rows, matrix, {
        "actual_channel_products": actual_products, "unrestricted_spatial_channel_products": baseline_products,
        "skipped_channel_products": baseline_products - actual_products,
        "sigma_products": sigma_products, "additions": additions, "spatial_visits": spatial_visits,
        "source_live_rows": int(live.sum()), "source_total_rows": source.n_out,
        "source_support_bytes": live.nbytes, "selected_row_bytes": rows.nbytes,
        "emitted_nnz": matrix.nnz, "emitted_csr_numeric_bytes": matrix_bytes,
        "peak_dot_numeric_workspace_bytes": peak_dot_workspace,
        "quarter_product_gate": actual_products * 4 <= baseline_products if baseline_products else False,
        "whole_state_reduction_proved": False, "source_and_predicates_retained": True,
        "python_workspace_in_numeric_bytes": False, "elapsed_s": time.monotonic() - start})
