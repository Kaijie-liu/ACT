#===- act/back_end/hybridz_tf/tf_transformer.py - HybridZ Transformer TF -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ transfer functions for transformer operators.
#
#===---------------------------------------------------------------------===#

import torch

try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None
    sp = None

import act.back_end.interval_tf.tf_transformer as interval
from act.back_end.hybridz_tf.tf_mlp import _matmul_term_rows
from act.back_end.core import Bounds, Fact
from act.back_end.utils import scale_interval
from act.back_end.solver.solver_hz import (
    hz_add_const,
    hz_concat,
    hz_lift_bounds,
    hz_multiply,
    hz_tighten_bounds,
    sparse_abs_row_sum,
    sparse_hz_add_const,
    sparse_hz_concat,
    sparse_hz_lift_bounds,
    sparse_hz_linear,
)


def _broadcast_flat(value, n_out: int):
    value = value.flatten()
    if value.numel() == n_out:
        return value
    if value.numel() and n_out % value.numel() == 0:
        return value.repeat(n_out // value.numel())
    raise ValueError(f"cannot broadcast {value.numel()} values to {n_out} HZ rows")


def _liftable(bounds: Bounds) -> bool:
    return bool(
        torch.isfinite(bounds.lb).all()
        and torch.isfinite(bounds.ub).all()
        and (bounds.lb <= bounds.ub).all()
    )


def _dense_lift(
    L,
    fact,
    tf,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
):
    hz = tf._hz_cache.get(L.id)
    if hz is None or not _liftable(fact.bounds):
        tf._hz_cache.pop(L.id, None)
        return fact
    out = hz_lift_bounds(
        hz,
        fact.bounds,
        output_equalities=output_equalities,
        equality_rhs=equality_rhs,
        output_inequalities=output_inequalities,
        inequality_rhs=inequality_rhs,
    )
    if max(out.Gc.shape[1] + out.Gb.shape[1], out.c.shape[0]) > tf._HZ_MAX_INPUT_DIM:
        tf._hz_cache.pop(L.id, None)
    else:
        tf._hz_cache[L.id] = out
    return fact


def _softmax_rowsize(L, bounds):
    shape = L.params.get("input_shape")
    axis = int(L.params.get("axis", -1))
    if shape is not None:
        shape = tuple(int(d) for d in shape)
        axis = axis if axis >= 0 else len(shape) + axis
        if axis == len(shape) - 1:
            rowsize = shape[axis]
            if rowsize > 0 and bounds.lb.numel() % rowsize == 0:
                return int(rowsize)
        return None
    return int(bounds.lb.shape[-1])


def _dense_simplex(rowsize: int, n_out: int, dtype, device):
    if rowsize <= 0 or n_out % rowsize:
        return None, None
    groups = n_out // rowsize
    E = torch.zeros((groups, n_out), dtype=dtype, device=device)
    cols = torch.arange(n_out, device=device).view(groups, rowsize)
    E.scatter_(1, cols, 1.0)
    return E, torch.ones(groups, dtype=dtype, device=device)


def _sparse_simplex(rowsize: int, n_out: int):
    if sp is None or rowsize <= 0 or n_out % rowsize:
        return None, None
    groups = n_out // rowsize
    rows = np.repeat(np.arange(groups, dtype=np.int64), rowsize)
    cols = np.arange(n_out, dtype=np.int64)
    E = sp.csr_matrix(
        (np.ones(n_out, dtype=np.float64), (rows, cols)),
        shape=(groups, n_out),
    )
    return E, np.ones(groups, dtype=np.float64)


def _softmax_ratio_inequalities(bounds: Bounds, rowsize: int, differences=None):
    if sp is None or rowsize <= 1 or bounds.lb.numel() % rowsize:
        return None, None
    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1, rowsize)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1, rowsize)
    matrix_rows, matrix_cols, matrix_data = [], [], []
    row = 0
    for group in range(lb.shape[0]):
        if rowsize <= 8:
            pairs = ((i, j) for i in range(rowsize) for j in range(i + 1, rowsize))
        else:
            ref = int(np.argmax((lb[group] + ub[group]) * 0.5))
            pairs = ((i, ref) for i in range(rowsize) if i != ref)
        offset = group * rowsize
        for i, j in pairs:
            if differences is None:
                lower_delta = lb[group, i] - ub[group, j]
                upper_delta = ub[group, i] - lb[group, j]
            else:
                diff_lb, diff_ub = differences
                lower_delta = -diff_ub[group, i, j]
                upper_delta = -diff_lb[group, i, j]
            lower = (
                np.nextafter(np.exp(lower_delta), 0.0)
                if -745.0 < lower_delta <= np.log(1e12)
                else 0.0
            )
            upper = np.inf
            if upper_delta <= np.log(1e12):
                upper = np.nextafter(
                    np.exp(upper_delta) if upper_delta > -745.0 else 0.0,
                    np.inf,
                )
            if np.isfinite(upper):
                matrix_rows.extend((row, row))
                matrix_cols.extend((offset + i, offset + j))
                matrix_data.extend((1.0, -upper))
                row += 1
            if lower > 0.0:
                matrix_rows.extend((row, row))
                matrix_cols.extend((offset + i, offset + j))
                matrix_data.extend((-1.0, lower))
                row += 1
    if row == 0:
        return None, None
    A = sp.csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)),
        shape=(row, bounds.lb.numel()),
        dtype=np.float64,
    )
    return A, np.zeros(row, dtype=np.float64)


def _attention_score_differences(L, bounds: Bounds, rowsize: int, tf):
    tf._softmax_score_contexts.pop(L.id, None)
    preds = tf._net.preds.get(L.id, [])
    if len(preds) != 1:
        return None
    n_scores = bounds.lb.numel()
    scale = np.ones(n_scores, dtype=np.float64)
    shift = np.zeros(n_scores, dtype=np.float64)
    current = preds[0]
    while True:
        score_layer = tf._net.by_id.get(current)
        if score_layer is None:
            return None
        kind = score_layer.kind.upper()
        if kind == "MATMUL":
            break
        layer_preds = tf._net.preds.get(current, [])
        if len(layer_preds) != 1:
            return None
        if kind in {"RESHAPE", "FLATTEN", "SQUEEZE", "UNSQUEEZE"}:
            current = layer_preds[0]
            continue
        if kind == "SCALE":
            value = score_layer.params["a"].detach().cpu().double().numpy().reshape(-1)
            if value.size == 1:
                value = np.full(n_scores, value.item(), dtype=np.float64)
            elif n_scores % value.size == 0:
                value = np.tile(value, n_scores // value.size)
            else:
                return None
            scale *= value
            current = layer_preds[0]
            continue
        if kind == "BIAS":
            value = score_layer.params["c"].detach().cpu().double().numpy().reshape(-1)
            if value.size == 1:
                value = np.full(n_scores, value.item(), dtype=np.float64)
            elif n_scores % value.size == 0:
                value = np.tile(value, n_scores // value.size)
            else:
                return None
            shift += scale * value
            current = layer_preds[0]
            continue
        return None
    score_preds = tf._net.preds.get(score_layer.id, [])
    if len(score_preds) != 2:
        return None
    q_hz = tf._sparse_hz_cache.get(score_preds[0])
    k_hz = tf._sparse_hz_cache.get(score_preds[1])
    if q_hz is None or k_hz is None or q_hz.frame_id != k_hz.frame_id:
        return None
    q_bounds = tf._net.get_predecessor_bounds(
        score_layer.id, tf._after, tf._before, 0
    )
    k_bounds = tf._net.get_predecessor_bounds(
        score_layer.id, tf._after, tf._before, 1
    )
    batch = int(q_bounds.lb.shape[0])
    rows = _matmul_term_rows(score_layer, batch)
    if rows is None:
        return None
    q_rows, k_rows = rows
    if q_rows.shape[0] != n_scores or n_scores % rowsize:
        return None
    groups, reduction = n_scores // rowsize, q_rows.shape[1]
    q_lb = q_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    q_ub = q_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    lower = np.zeros((groups, rowsize, rowsize), dtype=np.float64)
    upper = np.zeros_like(lower)
    score_scale = scale.reshape(groups, rowsize)
    score_shift = shift.reshape(groups, rowsize)
    key_cache = {}
    pairs = rowsize * rowsize
    term_rows = np.repeat(np.arange(pairs, dtype=np.int64), reduction)
    term_cols = np.arange(pairs * reduction, dtype=np.int64)
    for group in range(groups):
        block = k_rows[group * rowsize : (group + 1) * rowsize]
        key = (block.tobytes(), score_scale[group].tobytes())
        delta = key_cache.get(key)
        if delta is None:
            left = np.broadcast_to(
                block[None, :, :], (rowsize, rowsize, reduction)
            ).reshape(-1)
            right = np.broadcast_to(
                block[:, None, :], (rowsize, rowsize, reduction)
            ).reshape(-1)
            left_scale = np.broadcast_to(
                score_scale[group][None, :, None],
                (rowsize, rowsize, reduction),
            ).reshape(-1)
            right_scale = np.broadcast_to(
                score_scale[group][:, None, None],
                (rowsize, rowsize, reduction),
            ).reshape(-1)
            dc = (left_scale * k_hz.c[left] - right_scale * k_hz.c[right]).reshape(
                rowsize, rowsize, reduction
            )
            DGc = (
                sp.diags(left_scale) @ k_hz.Gc[left]
                - sp.diags(right_scale) @ k_hz.Gc[right]
            ).tocsr()
            radius = sparse_abs_row_sum(DGc).reshape(
                rowsize, rowsize, reduction
            )
            if k_hz.n_bin:
                DGb = (
                    sp.diags(left_scale) @ k_hz.Gb[left]
                    - sp.diags(right_scale) @ k_hz.Gb[right]
                ).tocsr()
                radius += sparse_abs_row_sum(DGb).reshape(
                    rowsize, rowsize, reduction
                )
            else:
                DGb = sp.csr_matrix((dc.size, 0), dtype=np.float64)
            delta = (dc, DGc, DGb, dc - radius, dc + radius)
            key_cache[key] = delta
        dc, DGc, DGb, dl, du = delta
        qr = q_rows[group * rowsize]
        ql, qu = q_lb[qr], q_ub[qr]
        corners = np.stack((ql * dl, ql * du, qu * dl, qu * du), axis=0)
        interval_lower = np.min(corners, axis=0).sum(axis=-1)
        interval_upper = np.max(corners, axis=0).sum(axis=-1)

        cx = np.broadcast_to((ql + qu) * 0.5, (pairs, reduction)).reshape(-1)
        cy = ((dl + du) * 0.5).reshape(-1)
        q_operator = sp.csr_matrix(
            (cy, (term_rows, np.tile(qr, pairs))),
            shape=(pairs, q_hz.n_out),
        )
        delta_operator = sp.csr_matrix(
            (cx, (term_rows, term_cols)),
            shape=(pairs, pairs * reduction),
        )
        affine_center = (
            np.asarray(q_operator @ q_hz.c).reshape(-1)
            + np.asarray(delta_operator @ dc.reshape(-1)).reshape(-1)
            - np.add.reduceat(cx * cy, term_cols[::reduction])
        ).reshape(rowsize, rowsize)
        affine_Gc = (
            q_operator @ q_hz.Gc + delta_operator @ DGc
        ).tocsr()
        affine_radius = sparse_abs_row_sum(affine_Gc).reshape(
            rowsize, rowsize
        )
        if q_hz.n_bin:
            affine_Gb = (
                q_operator @ q_hz.Gb + delta_operator @ DGb
            ).tocsr()
            affine_radius += sparse_abs_row_sum(affine_Gb).reshape(
                rowsize, rowsize
            )
        affine_radius += (
            (qu - ql)[None, None, :] * (du - dl) * 0.25
        ).sum(axis=-1)
        lower[group] = np.maximum(interval_lower, affine_center - affine_radius)
        upper[group] = np.minimum(interval_upper, affine_center + affine_radius)
        shift_delta = score_shift[group][None, :] - score_shift[group][:, None]
        lower[group] = np.nextafter(lower[group] + shift_delta, -np.inf)
        upper[group] = np.nextafter(upper[group] + shift_delta, np.inf)
    tf._softmax_score_contexts[L.id] = {
        "q_hz": q_hz,
        "k_hz": k_hz,
        "q_bounds": q_bounds,
        "k_bounds": k_bounds,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "scale": scale,
        "shift": shift,
    }
    return lower, upper


def _softmax_box_from_differences(bounds: Bounds, differences) -> Bounds:
    lower, upper = differences
    with np.errstate(over="ignore"):
        lower_den = 1.0 + np.sum(np.exp(upper), axis=-1) - 1.0
        upper_den = 1.0 + np.sum(np.exp(lower), axis=-1) - 1.0
    prob_lb = np.divide(1.0, lower_den, out=np.zeros_like(lower_den), where=lower_den > 0)
    prob_ub = np.divide(1.0, upper_den, out=np.ones_like(upper_den), where=upper_den > 0)
    prob_lb = np.nextafter(prob_lb, 0.0)
    prob_ub = np.nextafter(prob_ub, np.inf)
    shape = bounds.lb.shape
    return Bounds(
        torch.from_numpy(prob_lb.reshape(-1)).to(bounds.lb).reshape(shape),
        torch.from_numpy(prob_ub.reshape(-1)).to(bounds.ub).reshape(shape),
    )


def _sparse_lift(
    L,
    hz,
    fact,
    tf,
    *,
    simplex_rowsize=None,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
):
    if not _liftable(fact.bounds):
        return None, "nonfinite_sparse_transformer_bounds"
    reservation = tf._sparse_cont_slots_for(hz, L.id, fact.bounds.lb.numel())
    if reservation is None:
        return None, "sparse_transformer_size_limit"
    slots, n_cont = reservation
    E, rhs = output_equalities, equality_rhs
    if simplex_rowsize is not None:
        E, rhs = _sparse_simplex(int(simplex_rowsize), fact.bounds.lb.numel())
    elif isinstance(E, torch.Tensor):
        E = sp.csr_matrix(E.detach().cpu().double().numpy())
        rhs = rhs.detach().cpu().double().numpy()
    A, arhs = output_inequalities, inequality_rhs
    if isinstance(A, torch.Tensor):
        A = sp.csr_matrix(A.detach().cpu().double().numpy())
        arhs = arhs.detach().cpu().double().numpy()
    return (
        sparse_hz_lift_bounds(
            hz,
            fact.bounds,
            slots,
            n_cont,
            output_equalities=E,
            equality_rhs=rhs,
            output_inequalities=A,
            inequality_rhs=arhs,
        ),
        None,
    )


def _softmax_box(bounds: Bounds, rowsize) -> Bounds:
    if rowsize is None or not _liftable(bounds):
        return Bounds(torch.zeros_like(bounds.lb), torch.ones_like(bounds.ub))
    lb = bounds.lb.reshape(-1, rowsize)
    ub = bounds.ub.reshape(-1, rowsize)
    eye = torch.eye(rowsize, dtype=torch.bool, device=lb.device).unsqueeze(0)
    other_ub = ub.unsqueeze(1).expand(-1, rowsize, -1).masked_fill(eye, float("-inf"))
    other_lb = lb.unsqueeze(1).expand(-1, rowsize, -1).masked_fill(eye, float("-inf"))
    lower_den = torch.logaddexp(lb, torch.logsumexp(other_ub, dim=-1))
    upper_den = torch.logaddexp(ub, torch.logsumexp(other_lb, dim=-1))
    return Bounds(
        torch.exp(lb - lower_den).reshape_as(bounds.lb),
        torch.exp(ub - upper_den).reshape_as(bounds.ub),
    )


def _layernorm_box(L, bounds: Bounds) -> Bounds:
    gamma = L.params["gamma"].flatten().to(bounds.lb)
    beta = L.params["beta"].flatten().to(bounds.lb)
    width = int(gamma.numel())
    if width == 0 or bounds.lb.shape[-1] % width:
        raise ValueError("LayerNorm parameters do not divide the flattened input")
    lb = bounds.lb.reshape(-1, width)
    ub = bounds.ub.reshape(-1, width)
    mu_lb = lb.mean(dim=-1, keepdim=True)
    mu_ub = ub.mean(dim=-1, keepdim=True)
    centered_lb = lb - mu_ub
    centered_ub = ub - mu_lb
    if L.params.get("variant", L.params.get("layer_norm", "standard")) == "no_var":
        norm_lb, norm_ub = centered_lb, centered_ub
    else:
        max_abs = torch.maximum(centered_lb.abs(), centered_ub.abs())
        variance_ub = (max_abs * max_abs).mean(dim=-1, keepdim=True)
        eps = bounds.lb.new_tensor(float(L.params.get("eps", 1e-5)))
        inv_lb = torch.rsqrt(variance_ub + eps)
        inv_ub = torch.rsqrt(eps).expand_as(inv_lb)
        norm_lb, norm_ub = scale_interval(
            centered_lb, centered_ub, inv_lb, inv_ub
        )
    out_lb = torch.where(gamma >= 0, gamma * norm_lb + beta, gamma * norm_ub + beta)
    out_ub = torch.where(gamma >= 0, gamma * norm_ub + beta, gamma * norm_lb + beta)
    return Bounds(out_lb.reshape_as(bounds.lb), out_ub.reshape_as(bounds.ub))


def _layernorm_equalities(L, fact):
    gamma = L.params.get("gamma")
    beta = L.params.get("beta")
    if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
        return None, None
    width = int(gamma.numel())
    total = int(fact.bounds.lb.numel())
    if width == 0 or total % width:
        return None, None
    gamma = gamma.flatten().to(fact.bounds.lb)
    beta = beta.flatten().to(fact.bounds.lb)
    if bool((gamma.abs() <= torch.finfo(gamma.dtype).eps).any()):
        return None, None
    weights = gamma.reciprocal()
    groups = total // width
    E = fact.bounds.lb.new_zeros((groups, total))
    cols = torch.arange(total, device=E.device).view(groups, width)
    E.scatter_(1, cols, weights.expand(groups, -1))
    return E, (weights * beta).sum().expand(groups)


def _mha_split_operator(L, n_in: int, n_out: int):
    weight = L.params.get("weight")
    if sp is None or not isinstance(weight, torch.Tensor):
        return None
    hidden = int(L.params.get("hidden_size", weight.shape[1]))
    input_shape = tuple(int(d) for d in L.params.get("input_shape", (1, hidden)))
    seq = int(input_shape[-2]) if len(input_shape) >= 2 else 1
    per_sample = seq * hidden
    if per_sample <= 0 or n_in % per_sample:
        return None
    W = weight.detach().cpu().double().numpy()
    bias = L.params.get("bias")
    b = (
        bias.detach().cpu().double().numpy().reshape(-1)
        if isinstance(bias, torch.Tensor)
        else np.zeros(W.shape[0], dtype=np.float64)
    )
    role = str(L.params.get("role", ""))
    if role in {"query", "key"}:
        position = int(L.params.get("position", 0))
        if position < 0 or position >= seq:
            return None
        single = sp.hstack(
            [sp.csr_matrix((W.shape[0], position * hidden)), sp.csr_matrix(W),
             sp.csr_matrix((W.shape[0], (seq - position - 1) * hidden))],
            format="csr",
        )
        single_bias = b
    elif role == "value":
        feature = int(L.params.get("feature", 0))
        if feature < 0 or feature >= W.shape[0]:
            return None
        single = sp.kron(sp.eye(seq, format="csr"), sp.csr_matrix(W[feature:feature + 1]), format="csr")
        single_bias = np.full(seq, b[feature], dtype=np.float64)
    else:
        single = sp.kron(sp.eye(seq, format="csr"), sp.csr_matrix(W), format="csr")
        single_bias = np.tile(b, seq)
    batch = n_in // per_sample
    operator = sp.block_diag([single] * batch, format="csr")
    full_bias = np.tile(single_bias, batch)
    return (operator, full_bias) if operator.shape == (n_out, n_in) else None


def tf_posenc(L, bounds, tf):
    fact = interval.tf_posenc(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        tf._hz_cache[L.id] = hz_add_const(
            hz, _broadcast_flat(L.params["pos_vec"], hz.c.shape[0])
        )
    return fact


def tf_layernorm(L, bounds, tf):
    fact = interval.tf_layernorm(L, bounds)
    fact = Fact(bounds=_layernorm_box(L, bounds), cons=fact.cons)
    fact.cons.add_box(L.id, L.out_vars, fact.bounds)
    E, rhs = _layernorm_equalities(L, fact)
    return _dense_lift(L, fact, tf, E, rhs)


def tf_gelu(L, bounds, tf):
    return _dense_lift(L, interval.tf_gelu(L, bounds), tf)


def tf_att_scores(L, bounds, tf):
    fact = interval.tf_att_scores(
        L,
        tf._after[L.params["q_src"]].bounds,
        tf._after[L.params["k_src"]].bounds,
    )
    return _dense_lift(L, fact, tf)


def tf_softmax(L, bounds, tf):
    fact = interval.tf_softmax(L, bounds)
    rowsize = _softmax_rowsize(L, bounds)
    softmax_bounds = _softmax_box(bounds, rowsize)
    differences = (
        _attention_score_differences(L, bounds, rowsize, tf)
        if rowsize is not None
        else None
    )
    if differences is not None:
        difference_bounds = _softmax_box_from_differences(bounds, differences)
        softmax_bounds = hz_tighten_bounds(softmax_bounds, difference_bounds)
    tf._softmax_differences[L.id] = differences
    fact = Fact(bounds=softmax_bounds, cons=fact.cons)
    fact.cons.add_box(L.id, L.out_vars, fact.bounds)
    hz = tf._hz_cache.get(L.id)
    E = rhs = A = arhs = None
    if hz is not None:
        if rowsize is not None:
            E, rhs = _dense_simplex(
                rowsize,
                fact.bounds.lb.numel(),
                hz.c.dtype,
                hz.c.device,
            )
            Asp, arhs_np = _softmax_ratio_inequalities(bounds, rowsize, differences)
            if Asp is not None:
                A = torch.from_numpy(Asp.toarray()).to(hz.c)
                arhs = torch.from_numpy(arhs_np).to(hz.c)
    return _dense_lift(L, fact, tf, E, rhs, A, arhs)


def tf_att_mix(L, bounds, tf):
    fact = interval.tf_att_mix(
        L,
        tf._after[L.params["w_src"]].bounds,
        tf._after[L.params["v_src"]].bounds,
    )
    return _dense_lift(L, fact, tf)


def tf_mha_split(L, bounds, tf):
    fact = interval.tf_mha_split(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        affine = _mha_split_operator(L, hz.c.shape[0], fact.bounds.lb.numel())
        if affine is not None:
            operator, bias = affine
            R = torch.from_numpy(operator.toarray()).to(hz.c)
            tf._hz_cache[L.id] = hz_add_const(
                hz_multiply(hz, R), torch.from_numpy(bias).to(hz.c)
            )
            return fact
    return _dense_lift(L, fact, tf)


def tf_mha_join(L, bounds, tf):
    fact = interval.tf_mha_join(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    parts = [tf._hz_cache.get(pid) for pid in tf._net.preds.get(L.id, [])]
    if parts and all(part is not None for part in parts):
        tf._hz_cache[L.id] = hz_concat(parts)
    else:
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_mask_add(L, bounds, tf):
    fact = interval.tf_mask_add(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        tf._hz_cache[L.id] = hz_add_const(
            hz, _broadcast_flat(L.params["M"], hz.c.shape[0])
        )
    return fact


def sparse_hz_apply_layer(L, hz, input_bounds, result, tf):
    kind = L.kind.upper()
    if kind == "POSENC":
        return True, sparse_hz_add_const(
            hz, _broadcast_flat(L.params["pos_vec"], hz.n_out)
        ), None
    if kind == "MASK_ADD":
        return True, sparse_hz_add_const(
            hz, _broadcast_flat(L.params["M"], hz.n_out)
        ), None
    if kind == "MHA_JOIN":
        parts = [tf._sparse_hz_cache.get(pid) for pid in tf._net.preds.get(L.id, [])]
        return (
            (True, sparse_hz_concat(parts), None)
            if parts and all(part is not None for part in parts)
            else (True, None, "missing_sparse_mha_join_input")
        )
    if kind == "SOFTMAX":
        rowsize = _softmax_rowsize(L, input_bounds)
        differences = tf._softmax_differences.get(L.id)
        A, arhs = (
            _softmax_ratio_inequalities(input_bounds, rowsize, differences)
            if rowsize is not None
            else (None, None)
        )
        out, reason = _sparse_lift(
            L,
            hz,
            result,
            tf,
            simplex_rowsize=rowsize,
            output_inequalities=A,
            inequality_rhs=arhs,
        )
        return True, out, reason
    if kind == "MHA_SPLIT":
        affine = _mha_split_operator(L, hz.n_out, result.bounds.lb.numel())
        if affine is not None:
            operator, bias = affine
            return True, sparse_hz_linear(hz, operator, bias), None
        out, reason = _sparse_lift(L, hz, result, tf)
        return True, out, reason
    if kind == "LAYERNORM":
        E, rhs = _layernorm_equalities(L, result)
        out, reason = _sparse_lift(
            L, hz, result, tf, output_equalities=E, equality_rhs=rhs
        )
        return True, out, reason
    if kind in {"GELU", "ATT_SCORES", "ATT_MIX"}:
        out, reason = _sparse_lift(L, hz, result, tf)
        return True, out, reason
    return False, None, None
