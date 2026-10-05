#===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
#===---------------------------------------------------------------------===#

from dataclasses import dataclass
import weakref

import torch
import torch.nn.functional as F
try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None
    sp = None
from act.back_end.core import Bounds, Fact
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_tighten_bounds,
    sparse_hz_add_const,
    sparse_hz_add_same_frame,
    sparse_hz_fast_bounds,
    sparse_hz_linear,
    sparse_hz_scale,
)
from act.back_end.hybridz_tf.tf_mlp import (
    _hz_fact,
    _sparse_apply_relu,
    _sparse_param_vector,
    _sparse_relu_bounds,
)
from act.back_end.utils import avgpool2d_denominators, avgpool2d_output_hw
import act.back_end.interval_tf.tf_cnn as interval
import act.back_end.interval_tf.tf_mlp as interval_mlp


# --- HZ transfer functions (CNN) ---

def tf_conv2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        input_shape = L.params.get("input_shape")
        if input_shape is not None:
            tf._hz_cache[L.id] = hz_conv2d(
                hz_in, L.params["weight"], L.params.get("bias"),
                L.params.get("stride", 1), L.params.get("padding", 0),
                L.params.get("dilation", 1), L.params.get("groups", 1), input_shape,
            )
        else:
            hz_in = None
    fact = interval.tf_conv2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_maxpool2d(L, bounds, tf):
    # MaxPool is not affine in the HZ rows; keep interval bounds for soundness.
    tf._hz_cache[L.id] = None
    return interval.tf_maxpool2d(L, bounds)


def _sparse_available() -> bool:
    return np is not None and sp is not None


def _pair(x):
    return (int(x), int(x)) if isinstance(x, int) else (int(x[0]), int(x[1]))


def _spatial_shape(input_shape):
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    return int(C), int(H), int(W)


def _sparse_apply_per_batch_linear(hz: SparseHZono, W, bias=None) -> SparseHZono:
    Wsp = W.tocsr().astype(np.float64) if sp.issparse(W) else sp.csr_matrix(W)
    in_dim = int(Wsp.shape[1])
    if in_dim == 0 or hz.n_out % in_dim != 0:
        raise ValueError(f"sparse spatial shape mismatch: {hz.n_out} vs {Wsp.shape}")
    B = hz.n_out // in_dim
    M = sp.kron(sp.eye(B, format="csr"), Wsp, format="csr") if B != 1 else Wsp
    b = None
    if bias is not None:
        b0 = np.asarray(bias, dtype=np.float64).reshape(-1)
        b = np.tile(b0, B) if B > 1 else b0
    return sparse_hz_linear(hz, M, b)


@dataclass(frozen=True)
class SparseHZAffineTerm:
    source: SparseHZono
    operators: tuple


@dataclass(frozen=True)
class SparseHZAffineExpr:
    """Exact delayed affine DAG over one shared nonconvex HZ frame."""

    terms: tuple[SparseHZAffineTerm, ...]
    bias: np.ndarray
    n_out: int
    frame_id: int

    def __post_init__(self):
        bias = np.asarray(self.bias, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "bias", bias)
        if not np.all(np.isfinite(bias)):
            raise ValueError("lazy affine bias contains a non-finite value")
        if bias.size != int(self.n_out):
            raise ValueError(
                f"lazy affine bias mismatch: {bias.size} vs {self.n_out}"
            )
        if not self.terms:
            raise ValueError("lazy affine expression requires at least one term")
        for term in self.terms:
            if not term.source.exact or term.source.frame_id != self.frame_id:
                raise ValueError("lazy affine terms require one exact shared frame")
            width = term.source.n_out
            for operator in term.operators:
                if operator.shape[1] != width:
                    raise ValueError(
                        f"lazy affine operator mismatch: {operator.shape} vs {width}"
                    )
                width = int(operator.shape[0])
            if width != int(self.n_out):
                raise ValueError(
                    f"lazy affine term output mismatch: {width} vs {self.n_out}"
                )


def _lazy_prepare_operator(operator):
    """Validate one lazy operator without expanding an implicit descriptor."""
    if sp.issparse(operator):
        prepared = operator.tocsr()
        if not np.all(np.isfinite(prepared.data)):
            raise ValueError("lazy affine operator contains a non-finite value")
        return prepared
    required = (
        "shape",
        "logical_expanded_nnz",
        "resident_entries",
        "resident_bytes",
        "matvec",
        "left_compose",
    )
    if any(not hasattr(operator, name) for name in required):
        raise ValueError("unsupported lazy affine operator")
    shape = tuple(operator.shape)
    if len(shape) != 2 or any(int(value) < 0 for value in shape):
        raise ValueError(f"invalid lazy affine operator shape: {shape}")
    for name in (
        "logical_expanded_nnz",
        "resident_entries",
        "resident_bytes",
    ):
        if int(getattr(operator, name)) < 0:
            raise ValueError(f"invalid lazy affine operator metric: {name}")
    return operator


def _lazy_operator_logical_nnz(operator) -> int:
    return int(
        operator.nnz
        if sp.issparse(operator)
        else operator.logical_expanded_nnz
    )


def _lazy_operator_resident_count(operator) -> int:
    return int(
        operator.nnz
        if sp.issparse(operator)
        else operator.resident_entries
    )


def _lazy_operator_resident_nbytes(operator) -> int:
    if sp.issparse(operator):
        return int(
            operator.data.nbytes
            + operator.indices.nbytes
            + operator.indptr.nbytes
        )
    return int(operator.resident_bytes)


def _lazy_operator_matvec(operator, vector) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != int(operator.shape[1]):
        raise ValueError(
            f"lazy affine matvec mismatch: {values.size} vs {operator.shape}"
        )
    result = (
        np.asarray(operator @ values, dtype=np.float64).reshape(-1)
        if sp.issparse(operator)
        else np.asarray(operator.matvec(values), dtype=np.float64).reshape(-1)
    )
    if result.size != int(operator.shape[0]):
        raise ValueError("lazy affine matvec returned an invalid shape")
    if not np.all(np.isfinite(result)):
        raise ValueError("lazy affine matvec returned a non-finite value")
    return result


def _lazy_left_compose(Q, operator, limit) -> sp.csr_matrix:
    """Return ``Q @ operator`` while keeping implicit operators unexpanded."""
    left = Q.tocsr()
    if left.shape[1] != int(operator.shape[0]):
        raise ValueError(
            f"lazy affine compose mismatch: {left.shape} vs {operator.shape}"
        )
    if not np.all(np.isfinite(left.data)):
        raise ValueError("lazy affine left factor contains a non-finite value")
    if sp.issparse(operator):
        result = (left @ operator).tocsr()
    else:
        try:
            result = operator.left_compose(left, max_nnz=int(limit))
        except MemoryError as exc:
            raise MemoryError(
                "lazy composed operator storage limit"
            ) from exc
        if not sp.issparse(result):
            raise ValueError("lazy affine composition did not return CSR")
        result = result.tocsr()
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    if not np.all(np.isfinite(result.data)):
        raise ValueError("lazy affine composition returned a non-finite value")
    if result.nnz > int(limit):
        raise MemoryError("lazy composed operator storage limit")
    return result


def _lazy_unique_operators(expr: SparseHZAffineExpr):
    unique = {}
    for term in expr.terms:
        for operator in term.operators:
            unique[id(operator)] = operator
    return unique.values()


def _lazy_operator_entries(expr: SparseHZAffineExpr) -> int:
    """Legacy metric: bias plus unique logical expanded operator entries."""
    return int(
        expr.bias.size
        + sum(
            _lazy_operator_logical_nnz(operator)
            for operator in _lazy_unique_operators(expr)
        )
    )


def _lazy_operator_resident_entries(expr: SparseHZAffineExpr) -> int:
    return int(
        expr.bias.size
        + sum(
            _lazy_operator_resident_count(operator)
            for operator in _lazy_unique_operators(expr)
        )
    )


def _lazy_operator_resident_bytes(expr: SparseHZAffineExpr) -> int:
    return int(
        expr.bias.nbytes
        + sum(
            _lazy_operator_resident_nbytes(operator)
            for operator in _lazy_unique_operators(expr)
        )
    )


def _lazy_intern_operator(tf, operator):
    """Content-intern descriptors only when a weak arena is explicitly set."""
    if sp.issparse(operator):
        return operator
    arena = getattr(tf, "_neural_hz_linear_op_arena", None)
    if arena is None:
        return operator
    if not isinstance(arena, weakref.WeakValueDictionary):
        raise ValueError("exact linear-op arena must hold weak values")
    key = getattr(operator, "content_key", None)
    if key is None:
        raise ValueError("exact linear operator has no stable content key")
    existing = arena.get(key)
    if existing is not None:
        return existing
    arena[key] = operator
    return operator


def _record_lazy_conv_operator(tf, layer, operator, keep_rows=None) -> None:
    if sp.issparse(operator) or not isinstance(operator, ImplicitConv2DOp):
        return
    tf._neural_hz_implicit_conv_ops = int(
        getattr(tf, "_neural_hz_implicit_conv_ops", 0)
    ) + 1
    profile = getattr(tf, "_neural_hz_implicit_conv_profile", None)
    if profile is None:
        profile = []
        tf._neural_hz_implicit_conv_profile = profile
    masked_rows = 0
    if keep_rows is not None:
        keep = np.asarray(keep_rows, dtype=bool).reshape(-1)
        if keep.size != operator.shape[0]:
            raise ValueError("implicit Conv2D profile mask shape mismatch")
        masked_rows = int((~keep).sum())
    profile.append(
        {
            "layer_id": int(layer.id),
            "shape": [int(value) for value in operator.shape],
            "logical_expanded_nnz": int(operator.logical_expanded_nnz),
            "resident_entries": int(operator.resident_entries),
            "resident_bytes": int(operator.resident_bytes),
            "masked_rows": int(masked_rows),
        }
    )


def _lazy_from_hz_linear(hz, operator, bias, limit):
    if not hz.exact or hz.frame_id is None:
        raise ValueError("lazy affine source must be an exact framed HZ")
    operator = _lazy_prepare_operator(operator)
    b = (
        np.zeros(operator.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != int(operator.shape[0]) or not np.all(np.isfinite(b)):
        raise ValueError("lazy affine initial bias shape/value mismatch")
    if _lazy_operator_resident_count(operator) + b.size > int(limit):
        raise MemoryError("lazy affine operator storage limit")
    expr = SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(hz, (operator,)),),
        bias=b,
        n_out=int(operator.shape[0]),
        frame_id=int(hz.frame_id),
    )
    return expr


def _lazy_identity(hz):
    if not hz.exact or hz.frame_id is None:
        raise ValueError("lazy affine identity requires an exact framed HZ")
    return SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(hz, ()),),
        bias=np.zeros(hz.n_out, dtype=np.float64),
        n_out=hz.n_out,
        frame_id=int(hz.frame_id),
    )


def _lazy_append_linear(expr, operator, bias, limit):
    operator = _lazy_prepare_operator(operator)
    if operator.shape[1] != expr.n_out:
        raise ValueError(
            f"lazy affine append mismatch: {operator.shape} vs {expr.n_out}"
        )
    b = (
        np.zeros(operator.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != operator.shape[0]:
        raise ValueError("lazy affine appended bias shape mismatch")
    if not np.all(np.isfinite(b)):
        raise ValueError("lazy affine appended bias contains a non-finite value")
    if _lazy_operator_resident_count(operator) + b.size > int(limit):
        raise MemoryError("lazy affine operator storage limit")
    propagated_bias = _lazy_operator_matvec(operator, expr.bias) + b
    if not np.all(np.isfinite(propagated_bias)):
        raise ValueError("lazy affine propagated bias is non-finite")
    out = SparseHZAffineExpr(
        terms=tuple(
            SparseHZAffineTerm(term.source, (*term.operators, operator))
            for term in expr.terms
        ),
        bias=propagated_bias,
        n_out=int(operator.shape[0]),
        frame_id=expr.frame_id,
    )
    return out


def _lazy_add(left, right, limit):
    if left.frame_id != right.frame_id or left.n_out != right.n_out:
        raise ValueError("lazy residual add requires one frame and shape")
    out = SparseHZAffineExpr(
        terms=(*left.terms, *right.terms),
        bias=left.bias + right.bias,
        n_out=left.n_out,
        frame_id=left.frame_id,
    )
    if out.bias.size > int(limit):
        raise MemoryError("lazy affine metadata storage limit")
    return out


def _sparse_hz_storage_entries(hz):
    return int(
        hz.c.size
        + hz.b.size
        + hz.ub.size
        + hz.Gc.nnz
        + hz.Gb.nnz
        + hz.Ac.nnz
        + hz.Ab.nnz
        + hz.Auc.nnz
        + hz.Aub.nnz
    )


def _try_phase_separate_exact_relu(
    expr,
    completed,
    preactivation,
    input_bounds,
    tf,
    layer_id,
    *,
    forced_stable_negative=None,
):
    """Keep exact stable-active rows lazy and only materialize the ReLU core.

    This is an exact row partition, not a relaxation.  The nonlinear source
    retains every predicate and latent slot from ``completed``; only its value
    map is restricted to actually unstable rows.  Stable-active rows are the
    matching row mask of the original preactivation expression.
    """
    if not bool(
        getattr(tf, "_neural_hz_sparse_phase_separated_relu", False)
    ):
        return None
    if (
        not completed.exact
        or completed.frame_id is None
        or completed.frame_id != expr.frame_id
        or completed.n_out != expr.n_out
    ):
        return None

    limit = int(tf._SPARSE_MAX_AFFINE_CELLS)
    completed_storage = _sparse_hz_storage_entries(completed)
    pressure_limit = min(limit, 64_000_000)
    if 3 * completed_storage < 2 * pressure_limit:
        return None

    lower, upper = _sparse_relu_bounds(
        preactivation,
        input_bounds,
        forced_stable_negative=forced_stable_negative,
    )
    stable_active = lower >= 0.0
    unstable = (lower < 0.0) & (upper > 0.0)
    if not np.any(stable_active) or not np.any(unstable):
        return None

    unstable_mask = sp.diags(unstable.astype(np.float64), format="csr")
    core = sparse_hz_linear(completed, unstable_mask)
    if (
        core.n_cont != completed.n_cont
        or core.n_bin != completed.n_bin
        or core.n_eq != completed.n_eq
        or core.n_ineq != completed.n_ineq
        or core.frame_id != completed.frame_id
        or not core.exact
    ):
        raise ValueError("phase-separated ReLU changed latent predicates")
    core_storage = _sparse_hz_storage_entries(core)
    if core_storage > limit:
        return None

    removed_value_nnz = int(
        completed.Gc[stable_active].nnz
        + completed.Gb[stable_active].nnz
    )
    added_entries = int(stable_active.sum()) + int(completed.n_out)
    if removed_value_nnz <= added_entries:
        return None

    active_mask = sp.diags(stable_active.astype(np.float64), format="csr")
    stable_expression = _lazy_append_linear(
        expr,
        active_mask,
        None,
        limit,
    )
    separated = _lazy_add(
        stable_expression,
        _lazy_identity(core),
        limit,
    )
    tf._neural_hz_phase_separated_relus = int(
        getattr(tf, "_neural_hz_phase_separated_relus", 0)
    ) + 1
    tf._neural_hz_phase_separated_profile.append(
        {
            "layer_id": int(layer_id),
            "frame_id": int(completed.frame_id),
            "stable_negative": int(
                completed.n_out - stable_active.sum() - unstable.sum()
            ),
            "stable_positive": int(stable_active.sum()),
            "unstable": int(unstable.sum()),
            "n_cont": int(completed.n_cont),
            "n_bin": int(completed.n_bin),
            "n_eq": int(completed.n_eq),
            "n_ineq": int(completed.n_ineq),
            "completed_storage": int(completed_storage),
            "core_storage": int(core_storage),
            "removed_value_nnz": int(removed_value_nnz),
            "added_mask_bias_entries": int(added_entries),
            "local_persistent_storage": int(core_storage + added_entries),
            "reachable_lazy_operator_entries": int(
                _lazy_operator_entries(separated)
            ),
            "reachable_lazy_operator_resident_entries": int(
                _lazy_operator_resident_entries(separated)
            ),
            "reachable_lazy_operator_resident_bytes": int(
                _lazy_operator_resident_bytes(separated)
            ),
        }
    )
    return separated


def _lazy_materialize(
    expr,
    keep_rows,
    limit,
    *,
    allow_transient_sum=False,
):
    keep = np.asarray(keep_rows, dtype=bool).reshape(-1)
    if keep.size != expr.n_out:
        raise ValueError(f"lazy keep-row mismatch: {keep.size} vs {expr.n_out}")
    row_mask = sp.diags(keep.astype(np.float64), format="csr")
    reverse_prefix_uses = {}
    for term in expr.terms:
        prefix = []
        for inner in reversed(term.operators):
            prefix.append(id(inner))
            key = tuple(prefix)
            reverse_prefix_uses[key] = reverse_prefix_uses.get(key, 0) + 1
    reverse_prefix_cache = {}
    cached_nnz = 0
    grouped = {}
    for term in expr.terms:
        if term.operators:
            operator = row_mask
            prefix = []
            for inner in reversed(term.operators):
                prefix.append(id(inner))
                key = tuple(prefix)
                cached = reverse_prefix_cache.get(key)
                if cached is not None:
                    operator = cached
                    continue
                operator = _lazy_left_compose(operator, inner, limit)
                if (
                    reverse_prefix_uses[key] > 1
                    and cached_nnz + operator.nnz <= int(limit)
                ):
                    reverse_prefix_cache[key] = operator
                    cached_nnz += int(operator.nnz)
        else:
            if term.source.n_out != expr.n_out:
                raise ValueError("lazy identity term output mismatch")
            operator = row_mask
        key = id(term.source)
        if key in grouped:
            source, previous = grouped[key]
            operator = (previous + operator).tocsr()
            operator.eliminate_zeros()
            if operator.nnz > int(limit):
                raise MemoryError("lazy coalesced operator storage limit")
            grouped[key] = (source, operator)
        else:
            grouped[key] = (term.source, operator)

    parts = []
    for source, operator in grouped.values():
        part = sparse_hz_linear(source, operator)
        if _sparse_hz_storage_entries(part) > int(limit):
            raise MemoryError("lazy materialized term storage limit")
        parts.append(part)
    out = parts[0]
    for part in parts[1:]:
        out = sparse_hz_add_same_frame(out, part)
        if (
            not allow_transient_sum
            and _sparse_hz_storage_entries(out) > int(limit)
        ):
            raise MemoryError("lazy residual materialization storage limit")
    return sparse_hz_add_const(out, expr.bias)


def _lazy_checkpoint(expr, limit):
    hz = _lazy_materialize(
        expr,
        np.ones(expr.n_out, dtype=bool),
        limit,
    )
    if _sparse_hz_storage_entries(hz) > int(limit):
        raise MemoryError("lazy checkpoint storage limit")
    return _lazy_identity(hz)


@dataclass(frozen=True)
class SparseHZPhaseSelectiveResult:
    """Exact selective-materialization result for one lazy ReLU."""

    core: SparseHZono
    expression: SparseHZAffineExpr
    output_bounds: Bounds


def _phase_selective_masks(input_bounds, n_out):
    """Return the disjoint interval-certified N/P/U row partition."""
    lower = input_bounds.lb.detach().cpu().reshape(-1).double().numpy()
    upper = input_bounds.ub.detach().cpu().reshape(-1).double().numpy()
    if lower.size != int(n_out) or upper.size != int(n_out):
        raise ValueError(
            f"phase-selective bound mismatch: {lower.size}/{upper.size} "
            f"vs {n_out}"
        )
    if np.any(lower > upper):
        raise ValueError("phase-selective ReLU received inconsistent bounds")
    stable_negative = upper <= 0.0
    stable_positive = (~stable_negative) & (lower >= 0.0)
    unstable = ~(stable_negative | stable_positive)
    if np.any(stable_negative & stable_positive) or np.any(
        stable_negative & unstable
    ) or np.any(stable_positive & unstable):
        raise ValueError("phase-selective ReLU masks are not disjoint")
    if not np.all(stable_negative | stable_positive | unstable):
        raise ValueError("phase-selective ReLU masks are not exhaustive")
    return stable_negative, stable_positive, unstable


def _phase_selective_synthetic_bounds(input_bounds, unstable):
    """Keep the authoritative interval only on U and make all other rows 0."""
    unstable = np.asarray(unstable, dtype=bool).reshape(-1)
    if unstable.size != input_bounds.lb.numel():
        raise ValueError("phase-selective synthetic bound shape mismatch")
    mask = torch.as_tensor(
        unstable,
        dtype=torch.bool,
        device=input_bounds.lb.device,
    ).reshape_as(input_bounds.lb)
    lower = torch.zeros_like(input_bounds.lb)
    upper = torch.zeros_like(input_bounds.ub)
    lower[mask] = input_bounds.lb[mask]
    upper[mask] = input_bounds.ub[mask]
    return Bounds(lb=lower, ub=upper)


def _phase_selective_output_bounds(
    input_bounds,
    core,
    probe,
    unstable,
    probe_positive,
):
    """Build a sound output hint without materializing omitted P rows."""
    base = Bounds(
        lb=torch.clamp(input_bounds.lb, min=0.0),
        ub=torch.clamp(input_bounds.ub, min=0.0),
    )
    candidate_lb = base.lb.clone().reshape(-1)
    candidate_ub = base.ub.clone().reshape(-1)
    core_bounds = sparse_hz_fast_bounds(core)
    probe_bounds = sparse_hz_fast_bounds(probe)

    def install(rows, bounds):
        indices = np.flatnonzero(rows).astype(np.int64)
        if not indices.size:
            return
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=candidate_lb.device,
        )
        lower = bounds.lb.reshape(-1).to(candidate_lb)
        upper = bounds.ub.reshape(-1).to(candidate_ub)
        candidate_lb[index] = lower[index]
        candidate_ub[index] = upper[index]

    install(unstable, core_bounds)
    install(probe_positive, probe_bounds)
    candidate = Bounds(
        lb=candidate_lb.reshape_as(base.lb),
        ub=candidate_ub.reshape_as(base.ub),
    )
    return hz_tighten_bounds(base, candidate)


def _try_phase_selective_exact_relu(expr, input_bounds, tf, layer):
    """Exactly materialize U plus a deterministic savings probe over P.

    Stable-positive rows stay as ``D_P @ expr``.  Only the unstable rows enter
    the exact ReLU graph, but the graph retains the original full output width,
    frame, predicates, and global neuron-indexed ReLU slots.
    """
    if not bool(
        getattr(
            tf,
            "_neural_hz_sparse_phase_selective_materialization",
            False,
        )
    ):
        return None
    if expr.frame_id is None:
        return None

    stable_negative, stable_positive, unstable = _phase_selective_masks(
        input_bounds, expr.n_out
    )
    if not np.any(stable_positive) or not np.any(unstable):
        return None

    positive_rows = np.flatnonzero(stable_positive).astype(np.int64)
    probe_rows = positive_rows[:8]
    probe_positive = np.zeros(expr.n_out, dtype=bool)
    probe_positive[probe_rows] = True
    keep_rows = unstable | probe_positive
    limit = int(tf._SPARSE_MAX_AFFINE_CELLS)
    try:
        probe = _lazy_materialize(
            expr,
            keep_rows,
            limit,
            allow_transient_sum=True,
        )
    except MemoryError:
        return None
    if (
        not probe.exact
        or probe.frame_id != expr.frame_id
        or probe.n_out != expr.n_out
    ):
        raise ValueError("phase-selective probe changed the exact HZ frame")

    probe_generator_nnz = int(
        probe.Gc[probe_rows].nnz + probe.Gb[probe_rows].nnz
    )
    added_mask_bias_entries = int(expr.n_out + positive_rows.size)
    if probe_generator_nnz <= added_mask_bias_entries:
        return None

    unstable_mask = sp.diags(
        unstable.astype(np.float64),
        offsets=0,
        shape=(expr.n_out, expr.n_out),
        format="csr",
    )
    core_input = sparse_hz_linear(probe, unstable_mask)
    outside = ~unstable
    if (
        np.any(core_input.c[outside] != 0.0)
        or core_input.Gc[outside].nnz
        or core_input.Gb[outside].nnz
    ):
        raise ValueError("phase-selective core input leaked outside U")
    if (
        not core_input.exact
        or core_input.frame_id != probe.frame_id
        or core_input.n_cont != probe.n_cont
        or core_input.n_bin != probe.n_bin
        or core_input.n_eq != probe.n_eq
        or core_input.n_ineq != probe.n_ineq
    ):
        raise ValueError("phase-selective masking changed latent predicates")

    synthetic_bounds = _phase_selective_synthetic_bounds(
        input_bounds, unstable
    )
    previous_transient = getattr(
        tf, "_neural_hz_transient_relu_input", False
    )
    tf._neural_hz_transient_relu_input = True
    try:
        core, reason = _sparse_apply_relu(
            layer,
            core_input,
            synthetic_bounds,
            tf,
        )
    finally:
        tf._neural_hz_transient_relu_input = previous_transient
    if core is None:
        return None
    if (
        not core.exact
        or core.frame_id != expr.frame_id
        or core.n_out != expr.n_out
        or core.n_cont < core_input.n_cont
        or core.n_bin < core_input.n_bin
        or core.n_eq < core_input.n_eq
        or core.n_ineq < core_input.n_ineq
    ):
        raise ValueError(
            "phase-selective exact ReLU changed the latent frame"
        )
    if (
        np.any(core.c[outside] != 0.0)
        or core.Gc[outside].nnz
        or core.Gb[outside].nnz
    ):
        raise ValueError("phase-selective ReLU core leaked outside U")
    core_storage = _sparse_hz_storage_entries(core)
    if core_storage > limit:
        return None

    try:
        positive_mask = sp.diags(
            stable_positive.astype(np.float64),
            offsets=0,
            shape=(expr.n_out, expr.n_out),
            format="csr",
        )
        positive_expression = _lazy_append_linear(
            expr,
            positive_mask,
            None,
            limit,
        )
        separated = _lazy_add(
            positive_expression,
            _lazy_identity(core),
            limit,
        )
    except MemoryError:
        return None

    output_bounds = _phase_selective_output_bounds(
        input_bounds,
        core,
        probe,
        unstable,
        probe_positive,
    )
    tf._neural_hz_phase_selective_relus = int(
        getattr(tf, "_neural_hz_phase_selective_relus", 0)
    ) + 1
    profile = getattr(tf, "_neural_hz_phase_selective_profile", None)
    if profile is None:
        profile = []
        tf._neural_hz_phase_selective_profile = profile
    profile.append(
        {
            "layer_id": int(layer.id),
            "frame_id": int(core.frame_id),
            "stable_negative": int(stable_negative.sum()),
            "stable_positive": int(stable_positive.sum()),
            "unstable": int(unstable.sum()),
            "probe_positive": int(probe_rows.size),
            "omitted_positive": int(positive_rows.size - probe_rows.size),
            "probe_generator_nnz": int(probe_generator_nnz),
            "added_mask_bias_entries": int(added_mask_bias_entries),
            "strict_local_saving": int(
                probe_generator_nnz - added_mask_bias_entries
            ),
            "core_storage": int(core_storage),
            "n_out": int(core.n_out),
            "n_cont": int(core.n_cont),
            "n_bin": int(core.n_bin),
            "n_eq": int(core.n_eq),
            "n_ineq": int(core.n_ineq),
            "reachable_lazy_operator_entries": int(
                _lazy_operator_entries(separated)
            ),
            "reachable_lazy_operator_resident_entries": int(
                _lazy_operator_resident_entries(separated)
            ),
            "reachable_lazy_operator_resident_bytes": int(
                _lazy_operator_resident_bytes(separated)
            ),
        }
    )
    return SparseHZPhaseSelectiveResult(
        core=core,
        expression=separated,
        output_bounds=output_bounds,
    )


def sparse_conv2d_matrix_from_layer(layer):
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing conv2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    weight = layer.params["weight"].detach().cpu().double().numpy()
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    groups = int(layer.params.get("groups", 1))
    OC, ICg, KH, KW = weight.shape
    OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    out_per_group = OC // groups
    rows, cols, data = [], [], []
    for oc in range(OC):
        group = oc // out_per_group
        c0 = group * ICg
        for oh in range(OH):
            for ow in range(OW):
                r = oc * OH * OW + oh * OW + ow
                for icg in range(ICg):
                    ic = c0 + icg
                    for kh in range(KH):
                        ih = oh * stride[0] - padding[0] + kh * dilation[0]
                        if ih < 0 or ih >= H:
                            continue
                        for kw in range(KW):
                            iw = ow * stride[1] - padding[1] + kw * dilation[1]
                            if iw < 0 or iw >= W:
                                continue
                            rows.append(r)
                            cols.append(ic * H * W + ih * W + iw)
                            data.append(weight[oc, icg, kh, kw])
    mat = sp.csr_matrix((data, (rows, cols)), shape=(OC * OH * OW, C * H * W))
    bias = layer.params.get("bias")
    b = None
    if bias is not None:
        b = np.repeat(bias.detach().cpu().double().numpy().reshape(-1), OH * OW)
    return mat, b


def sparse_conv2d_matrix_from_layer_csr(layer, keep_rows=None):
    """Build the exact Conv2D operator directly in output-row CSR order."""
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing conv2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    weight = layer.params["weight"].detach().cpu().double().numpy()
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    groups = int(layer.params.get("groups", 1))
    OC, ICg, KH, KW = weight.shape
    if groups <= 0 or C != ICg * groups or OC % groups:
        raise ValueError(
            f"invalid grouped Conv2D shape: C={C}, OC={OC}, "
            f"ICg={ICg}, groups={groups}"
        )
    OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    if OH < 0 or OW < 0:
        raise ValueError(f"invalid Conv2D output shape: OH={OH}, OW={OW}")

    output_rows = OH * OW
    total_rows = OC * output_rows
    if keep_rows is None:
        keep = np.ones(total_rows, dtype=bool)
    else:
        keep = np.asarray(keep_rows, dtype=bool).reshape(-1)
        if keep.size != total_rows:
            raise ValueError(
                f"Conv2D keep-row mismatch: {keep.size} vs {total_rows}"
            )
    keep_by_channel = keep.reshape(OC, output_rows)
    kernel_terms = ICg * KH * KW
    oh = np.repeat(np.arange(OH, dtype=np.int64), OW)
    ow = np.tile(np.arange(OW, dtype=np.int64), OH)
    icg = np.repeat(np.arange(ICg, dtype=np.int64), KH * KW)
    kh = np.tile(np.repeat(np.arange(KH, dtype=np.int64), KW), ICg)
    kw = np.tile(np.arange(KW, dtype=np.int64), ICg * KH)
    ih = (
        oh[:, None] * stride[0]
        - padding[0]
        + kh[None, :] * dilation[0]
    )
    iw = (
        ow[:, None] * stride[1]
        - padding[1]
        + kw[None, :] * dilation[1]
    )
    valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    local_columns = (
        icg[None, :] * H * W + ih * W + iw
    )
    row_counts = valid.sum(axis=1, dtype=np.int64)
    all_row_counts = np.tile(row_counts, OC)
    all_row_counts[~keep] = 0
    total_entries = int(all_row_counts.sum())
    indices = np.empty(total_entries, dtype=np.int64)
    data = np.empty(total_entries, dtype=np.float64)
    out_per_group = OC // groups
    cursor = 0
    for oc in range(OC):
        selected = valid & keep_by_channel[oc, :, None]
        selected_flat = selected.reshape(output_rows * kernel_terms)
        count = int(selected_flat.sum())
        stop = cursor + count
        group = oc // out_per_group
        indices[cursor:stop] = (
            local_columns.reshape(-1)[selected_flat]
            + group * ICg * H * W
        )
        channel_weights = np.broadcast_to(
            weight[oc].reshape(1, kernel_terms),
            (output_rows, kernel_terms),
        )
        data[cursor:stop] = channel_weights.reshape(-1)[selected_flat]
        cursor = stop

    indptr = np.empty(total_rows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(all_row_counts, out=indptr[1:])
    mat = sp.csr_matrix(
        (data, indices, indptr),
        shape=(OC * output_rows, C * H * W),
    )
    bias = layer.params.get("bias")
    b = None
    if bias is not None:
        b = np.repeat(
            bias.detach().cpu().double().numpy().reshape(-1), output_rows
        )
    return mat, b


def _lazy_conv2d_operator_and_bias(layer, tf, keep_rows=None):
    """Build the flag-selected exact lazy Conv2D operator.

    The lazy DAG historically stores one per-sample operator even when the
    layer metadata is NCHW.  Keep that shape contract so flag-off and flag-on
    fail identically for unsupported batched lazy states.
    """
    if not bool(
        getattr(tf, "_neural_hz_sparse_implicit_conv_dag", False)
    ):
        return sparse_conv2d_matrix_from_layer_csr(
            layer, keep_rows=keep_rows
        )
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing conv2d input_shape")
    channels, height, width = _spatial_shape(
        tuple(int(value) for value in input_shape)
    )
    kernel = layer.params["weight"].detach().cpu().double().numpy()
    operator = ImplicitConv2DOp(
        kernel,
        (1, channels, height, width),
        stride=layer.params.get("stride", 1),
        padding=layer.params.get("padding", 0),
        dilation=layer.params.get("dilation", 1),
        groups=layer.params.get("groups", 1),
        row_mask=keep_rows,
    )
    operator = _lazy_intern_operator(tf, operator)
    bias = layer.params.get("bias")
    if bias is None:
        return operator, None
    output_rows = operator.output_shape[2] * operator.output_shape[3]
    vector = np.repeat(
        bias.detach().cpu().double().numpy().reshape(-1), output_rows
    )
    if vector.size != operator.shape[0] or not np.all(np.isfinite(vector)):
        raise ValueError("invalid Conv2D bias for implicit lazy operator")
    return operator, vector


def sparse_convtranspose2d_matrix_from_layer(layer):
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing convtranspose2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    weight = layer.params["weight"].detach().cpu().double().numpy()
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    output_padding = _pair(layer.params.get("output_padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    groups = int(layer.params.get("groups", 1))
    IC, OCg, KH, KW = weight.shape
    OH = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (KH - 1) + output_padding[0] + 1
    OW = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (KW - 1) + output_padding[1] + 1
    in_per_group = IC // groups
    OC = OCg * groups
    rows, cols, data = [], [], []
    for ic in range(IC):
        group = ic // in_per_group
        oc0 = group * OCg
        for ih in range(H):
            for iw in range(W):
                cidx = ic * H * W + ih * W + iw
                for ocg in range(OCg):
                    oc = oc0 + ocg
                    for kh in range(KH):
                        oh = ih * stride[0] - padding[0] + kh * dilation[0]
                        if oh < 0 or oh >= OH:
                            continue
                        for kw in range(KW):
                            ow = iw * stride[1] - padding[1] + kw * dilation[1]
                            if ow < 0 or ow >= OW:
                                continue
                            rows.append(oc * OH * OW + oh * OW + ow)
                            cols.append(cidx)
                            data.append(weight[ic, ocg, kh, kw])
    mat = sp.csr_matrix((data, (rows, cols)), shape=(OC * OH * OW, C * H * W))
    bias = layer.params.get("bias")
    b = None
    if bias is not None:
        b = np.repeat(bias.detach().cpu().double().numpy().reshape(-1), OH * OW)
    return mat, b


def sparse_avgpool2d_matrix_from_layer(layer):
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing avgpool2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    kernel = _pair(layer.params["kernel_size"])
    raw_stride = layer.params.get("stride")
    stride = _pair(raw_stride if raw_stride is not None else layer.params["kernel_size"])
    padding = _pair(layer.params.get("padding", 0))
    ceil_mode = bool(layer.params.get("ceil_mode", False))
    count_include_pad = bool(layer.params.get("count_include_pad", True))
    divisor_override = layer.params.get("divisor_override")
    OH, OW = avgpool2d_output_hw(
        (H, W), kernel, stride, padding, ceil_mode
    )
    denominators = avgpool2d_denominators(
        (H, W),
        (OH, OW),
        kernel,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=torch.float64,
    ).cpu().numpy()
    rows, cols, data = [], [], []
    for c in range(C):
        for oh in range(OH):
            for ow in range(OW):
                r = c * OH * OW + oh * OW + ow
                for kh in range(kernel[0]):
                    ih = oh * stride[0] - padding[0] + kh
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(kernel[1]):
                        iw = ow * stride[1] - padding[1] + kw
                        if iw < 0 or iw >= W:
                            continue
                        rows.append(r)
                        cols.append(c * H * W + ih * W + iw)
                        data.append(1.0 / float(denominators[oh, ow]))
    return sp.csr_matrix((data, (rows, cols)), shape=(C * OH * OW, C * H * W)), None


_DEFERRED_AFFINE_KINDS = {"BIAS", "SCALE", "BN"}


def _deferred_relu_island(layer, tf):
    """Return the one live elementwise-affine path to a ReLU, if isolated."""
    candidates = []
    direct_successors = [int(value) for value in tf._net.succs.get(layer.id, [])]
    for root in direct_successors:
        chain = []
        previous = int(layer.id)
        current = root
        visited = set()
        while current not in visited:
            visited.add(current)
            node = tf._net.by_id.get(current)
            if node is None:
                break
            predecessors = [int(value) for value in tf._net.preds.get(current, [])]
            if predecessors != [previous]:
                break
            kind = node.kind.upper()
            if kind == "RELU":
                candidates.append((root, tuple(chain), node))
                break
            successors = [int(value) for value in tf._net.succs.get(current, [])]
            if kind not in _DEFERRED_AFFINE_KINDS or len(successors) != 1:
                break
            chain.append(node)
            previous, current = current, successors[0]
    if len(candidates) != 1:
        return None
    root, chain, relu = candidates[0]
    for other in direct_successors:
        if other == root:
            continue
        node = tf._net.by_id.get(other)
        if (
            node is None
            or node.kind.upper() not in _DEFERRED_AFFINE_KINDS
            or tf._net.succs.get(other, [])
        ):
            return None
    return chain, relu


def _deferred_affine_bounds(chain, bounds):
    current = bounds
    for layer in chain:
        kind = layer.kind.upper()
        if kind == "BIAS":
            current = interval_mlp.tf_bias(layer, current).bounds
        elif kind == "SCALE":
            current = interval_mlp.tf_scale(layer, current).bounds
        elif kind == "BN":
            current = interval_mlp.tf_bn(layer, current).bounds
        else:  # pragma: no cover - guarded by _deferred_relu_island
            raise ValueError(f"unsupported deferred affine kind: {kind}")
    return current


def _deferred_sparse_affine(chain, hz):
    current = hz
    for layer in chain:
        kind = layer.kind.upper()
        if kind == "BIAS":
            current = sparse_hz_add_const(
                current,
                _sparse_param_vector(layer.params["c"], current.n_out),
            )
        elif kind == "SCALE":
            current = sparse_hz_scale(
                current,
                _sparse_param_vector(layer.params["a"], current.n_out),
            )
        elif kind == "BN":
            current = sparse_hz_scale(
                current,
                _sparse_param_vector(layer.params["A"], current.n_out),
            )
            current = sparse_hz_add_const(
                current,
                _sparse_param_vector(layer.params["c"], current.n_out),
            )
        else:  # pragma: no cover - guarded by _deferred_relu_island
            raise ValueError(f"unsupported deferred affine kind: {kind}")
    return current


def _try_deferred_conv_relu(layer, hz, result, tf):
    island = _deferred_relu_island(layer, tf)
    if island is None or hz.frame_id is None:
        return None
    chain, relu = island
    relu_input_bounds = _deferred_affine_bounds(chain, result.bounds)
    stable_negative = (
        relu_input_bounds.ub.detach().cpu().reshape(-1).numpy() <= 0.0
    )
    if not np.any(stable_negative):
        return None
    if int(relu.id) in tf._sparse_precomputed_relu:
        return True, None, "duplicate_deferred_relu_target"

    W, bias = sparse_conv2d_matrix_from_layer_csr(
        layer,
        keep_rows=~stable_negative,
    )
    if hz.n_out != W.shape[1]:
        return True, None, "deferred_relu_batch_not_supported"
    partial = _sparse_apply_per_batch_linear(hz, W, bias)
    partial = _deferred_sparse_affine(chain, partial)
    completed, reason = _sparse_apply_relu(
        relu,
        partial,
        relu_input_bounds,
        tf,
        forced_stable_negative=stable_negative,
    )
    if completed is None:
        return True, None, reason or "deferred_relu_transform_failed"
    if tf._sparse_storage_entries(completed) > tf._SPARSE_MAX_AFFINE_CELLS:
        return True, None, "deferred_relu_storage_limit"

    tf._sparse_precomputed_relu[int(relu.id)] = (
        completed,
        relu_input_bounds.lb.detach().cpu().clone(),
        relu_input_bounds.ub.detach().cpu().clone(),
        None,
    )
    tf._neural_hz_deferred_relu_layers = int(
        getattr(tf, "_neural_hz_deferred_relu_layers", 0)
    ) + 1
    tf._neural_hz_deferred_zero_rows = int(
        getattr(tf, "_neural_hz_deferred_zero_rows", 0)
    ) + int(stable_negative.sum())
    return True, None, f"deferred_to_relu:{int(relu.id)}"


def _try_deferred_expr_conv_relu(layer, expr, result, tf):
    island = _deferred_relu_island(layer, tf)
    if island is None:
        return None
    chain, relu = island
    relu_input_bounds = _deferred_affine_bounds(chain, result.bounds)
    stable_negative = (
        relu_input_bounds.ub.detach().cpu().reshape(-1).numpy() <= 0.0
    )
    if not np.any(stable_negative):
        return None
    if int(relu.id) in tf._sparse_precomputed_relu:
        return True, None, None, "duplicate_deferred_relu_target"
    limit = int(tf._SPARSE_MAX_AFFINE_CELLS)
    try:
        operator, bias = _lazy_conv2d_operator_and_bias(
            layer,
            tf,
            keep_rows=~stable_negative,
        )
        current = _lazy_append_linear(expr, operator, bias, limit)
        for affine in chain:
            kind = affine.kind.upper()
            if kind == "BIAS":
                current = _lazy_add_const(current, affine.params["c"])
            elif kind == "SCALE":
                scale = _sparse_param_vector(affine.params["a"], current.n_out)
                current = _lazy_append_linear(
                    current,
                    sp.diags(scale, format="csr"),
                    None,
                    limit,
                )
            elif kind == "BN":
                scale = _sparse_param_vector(affine.params["A"], current.n_out)
                current = _lazy_append_linear(
                    current,
                    sp.diags(scale, format="csr"),
                    _sparse_param_vector(affine.params["c"], current.n_out),
                    limit,
                )
        selective = _try_phase_selective_exact_relu(
            current,
            relu_input_bounds,
            tf,
            relu,
        )
        phase_output_bounds = None
        if selective is not None:
            completed = selective.core
            separated = selective.expression
            phase_output_bounds = selective.output_bounds
        else:
            partial = _lazy_materialize(
                current,
                ~stable_negative,
                limit,
                allow_transient_sum=True,
            )
            previous_transient = tf._neural_hz_transient_relu_input
            tf._neural_hz_transient_relu_input = True
            try:
                completed, reason = _sparse_apply_relu(
                    relu,
                    partial,
                    relu_input_bounds,
                    tf,
                    forced_stable_negative=stable_negative,
                )
            finally:
                tf._neural_hz_transient_relu_input = previous_transient
            if completed is None:
                return True, None, None, reason or "deferred_lazy_relu_failed"
            separated = _try_phase_separate_exact_relu(
                current,
                completed,
                partial,
                relu_input_bounds,
                tf,
                relu.id,
                forced_stable_negative=stable_negative,
            )
        if (
            separated is None
            and _sparse_hz_storage_entries(completed) > limit
        ):
            return True, None, None, "deferred_lazy_relu_storage_limit"
    except MemoryError as exc:
        return True, None, None, str(exc).replace(" ", "_")
    except ValueError as exc:
        return True, None, None, f"deferred_lazy_invalid:{type(exc).__name__}"

    _record_lazy_conv_operator(
        tf, layer, operator, keep_rows=~stable_negative
    )
    tf._sparse_precomputed_relu[int(relu.id)] = (
        completed,
        relu_input_bounds.lb.detach().cpu().clone(),
        relu_input_bounds.ub.detach().cpu().clone(),
        separated,
        phase_output_bounds,
    )
    tf._neural_hz_deferred_relu_layers = int(
        getattr(tf, "_neural_hz_deferred_relu_layers", 0)
    ) + 1
    tf._neural_hz_deferred_zero_rows = int(
        getattr(tf, "_neural_hz_deferred_zero_rows", 0)
    ) + int(stable_negative.sum())
    tf._neural_hz_lazy_affine_materializations = int(
        getattr(tf, "_neural_hz_lazy_affine_materializations", 0)
    ) + 1
    reason = (
        "deferred_lazy_phase_selective_to_relu"
        if phase_output_bounds is not None
        else (
            "deferred_lazy_phase_to_relu"
            if separated is not None
            else "deferred_lazy_to_relu"
        )
    )
    return True, None, None, f"{reason}:{int(relu.id)}"


def _lazy_add_const(expr, value):
    vector = _sparse_param_vector(value, expr.n_out)
    return SparseHZAffineExpr(
        terms=expr.terms,
        bias=expr.bias + vector,
        n_out=expr.n_out,
        frame_id=expr.frame_id,
    )


def _lazy_operand(tf, layer_id):
    expr = tf._sparse_affine_expr_cache.get(int(layer_id))
    if expr is not None:
        return expr
    hz = tf._sparse_hz_cache.get(int(layer_id))
    return None if hz is None else _lazy_identity(hz)


def sparse_hz_apply_affine_expr_layer(L, expr, input_bounds, result, tf):
    """Propagate one exact lazy affine expression or materialize at ReLU."""
    limit = int(tf._SPARSE_MAX_AFFINE_CELLS)
    k = L.kind.upper()
    try:
        if k == "BIAS":
            out_expr = _lazy_add_const(expr, L.params["c"])
        elif k == "SCALE":
            scale = _sparse_param_vector(L.params["a"], expr.n_out)
            out_expr = _lazy_append_linear(
                expr,
                sp.diags(scale, format="csr"),
                None,
                limit,
            )
        elif k == "BN":
            scale = _sparse_param_vector(L.params["A"], expr.n_out)
            out_expr = _lazy_append_linear(
                expr,
                sp.diags(scale, format="csr"),
                _sparse_param_vector(L.params["c"], expr.n_out),
                limit,
            )
        elif k == "CONV2D":
            if bool(
                getattr(
                    tf,
                    "_neural_hz_sparse_deferred_relu_materialization",
                    False,
                )
            ):
                deferred = _try_deferred_expr_conv_relu(L, expr, result, tf)
                if deferred is not None:
                    return deferred
            operator, bias = _lazy_conv2d_operator_and_bias(L, tf)
            out_expr = _lazy_append_linear(
                expr, operator, bias, limit
            )
            _record_lazy_conv_operator(tf, L, operator)
        elif k == "DENSE":
            operator = sp.csr_matrix(
                L.params["weight"].detach().cpu().double().numpy()
            )
            bias = L.params.get("bias")
            bias = (
                None
                if bias is None
                else bias.detach().cpu().double().numpy().reshape(-1)
            )
            out_expr = _lazy_append_linear(
                expr, operator, bias, limit
            )
        elif k == "ADD":
            predecessors = [int(value) for value in tf._net.preds.get(L.id, [])]
            if len(predecessors) != 2:
                return True, None, None, "lazy_add_predecessor_count"
            left = _lazy_operand(tf, predecessors[0])
            right = _lazy_operand(tf, predecessors[1])
            if left is None or right is None:
                return True, None, None, "missing_lazy_add_input"
            try:
                out_expr = _lazy_add(left, right, limit)
            except MemoryError:
                operands = [left, right]
                out_expr = None
                for index in sorted(
                    range(2),
                    key=lambda value: _lazy_operator_entries(operands[value]),
                    reverse=True,
                ):
                    try:
                        operands[index] = _lazy_checkpoint(
                            operands[index], limit
                        )
                    except MemoryError:
                        continue
                    tf._neural_hz_lazy_checkpoints = int(
                        getattr(tf, "_neural_hz_lazy_checkpoints", 0)
                    ) + 1
                    try:
                        out_expr = _lazy_add(
                            operands[0], operands[1], limit
                        )
                        break
                    except MemoryError:
                        continue
                if out_expr is None:
                    raise MemoryError("lazy affine operator storage limit")
        elif k in {"FLATTEN", "RESHAPE", "SQUEEZE", "UNSQUEEZE"}:
            out_expr = expr
        elif k == "RELU":
            selective = _try_phase_selective_exact_relu(
                expr,
                input_bounds,
                tf,
                L,
            )
            if selective is not None:
                tf._neural_hz_lazy_affine_materializations = int(
                    getattr(tf, "_neural_hz_lazy_affine_materializations", 0)
                ) + 1
                phase_bounds = getattr(
                    tf, "_sparse_phase_output_bounds", None
                )
                if phase_bounds is None:
                    phase_bounds = {}
                    tf._sparse_phase_output_bounds = phase_bounds
                phase_bounds[int(L.id)] = selective.output_bounds
                return (
                    True,
                    selective.core,
                    selective.expression,
                    None,
                )
            stable_negative = (
                input_bounds.ub.detach().cpu().reshape(-1).numpy() <= 0.0
            )
            partial = _lazy_materialize(
                expr,
                ~stable_negative,
                limit,
                allow_transient_sum=True,
            )
            previous_transient = tf._neural_hz_transient_relu_input
            tf._neural_hz_transient_relu_input = True
            try:
                completed, reason = _sparse_apply_relu(
                    L,
                    partial,
                    input_bounds,
                    tf,
                    forced_stable_negative=stable_negative,
                )
            finally:
                tf._neural_hz_transient_relu_input = previous_transient
            if completed is None:
                return True, None, None, reason or "lazy_relu_transform_failed"
            separated = _try_phase_separate_exact_relu(
                expr,
                completed,
                partial,
                input_bounds,
                tf,
                L.id,
                forced_stable_negative=stable_negative,
            )
            if (
                separated is None
                and _sparse_hz_storage_entries(completed) > limit
            ):
                return True, None, None, "lazy_relu_storage_limit"
            tf._neural_hz_lazy_affine_materializations = int(
                getattr(tf, "_neural_hz_lazy_affine_materializations", 0)
            ) + 1
            return True, completed, separated, None
        else:
            return False, None, None, None
    except MemoryError as exc:
        return True, None, None, str(exc).replace(" ", "_")
    except ValueError as exc:
        return True, None, None, f"lazy_affine_invalid:{type(exc).__name__}"

    tf._neural_hz_lazy_affine_layers = int(
        getattr(tf, "_neural_hz_lazy_affine_layers", 0)
    ) + 1
    return True, None, out_expr, None


def sparse_hz_apply_layer(L, hz: SparseHZono, input_bounds: Bounds, result, tf):
    if not _sparse_available():
        return True, None, "scipy_unavailable"
    k = L.kind.upper()
    if k == "CONV2D":
        if bool(
            getattr(
                tf,
                "_neural_hz_sparse_deferred_relu_materialization",
                False,
            )
        ):
            deferred = _try_deferred_conv_relu(L, hz, result, tf)
            if deferred is not None:
                return deferred
        if bool(getattr(tf, "_neural_hz_sparse_lazy_affine_dag", False)):
            try:
                operator, bias = _lazy_conv2d_operator_and_bias(L, tf)
                expr = _lazy_from_hz_linear(
                    hz,
                    operator,
                    bias,
                    tf._SPARSE_MAX_AFFINE_CELLS,
                )
            except (MemoryError, ValueError) as exc:
                return True, None, f"lazy_affine_start:{type(exc).__name__}"
            _record_lazy_conv_operator(tf, L, operator)
            tf._sparse_affine_expr_cache[int(L.id)] = expr
            tf._neural_hz_lazy_affine_layers = int(
                getattr(tf, "_neural_hz_lazy_affine_layers", 0)
            ) + 1
            return True, None, "lazy_affine_expr"
        builder = (
            sparse_conv2d_matrix_from_layer_csr
            if bool(getattr(tf, "_neural_hz_sparse_conv_csr_builder", False))
            else sparse_conv2d_matrix_from_layer
        )
        W, b = builder(L)
        return True, _sparse_apply_per_batch_linear(hz, W, b), None
    if k == "CONVTRANSPOSE2D":
        W, b = sparse_convtranspose2d_matrix_from_layer(L)
        return True, _sparse_apply_per_batch_linear(hz, W, b), None
    if k == "AVGPOOL2D":
        W, b = sparse_avgpool2d_matrix_from_layer(L)
        return True, _sparse_apply_per_batch_linear(hz, W, b), None
    if k == "MAXPOOL2D":
        return True, None, "unsupported_sparse_maxpool2d"
    return False, None, None


# --- HZ conv2d (zonotope domain) ---

def _conv2d_generators(
    G, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
):
    """Apply conv2d to a generator matrix ``(B*C*H*W, ng)`` and return
    a generator matrix ``(B*n_out_per_sample, ng)``. Each generator
    column is convolved independently per batch element by stacking
    ``ng * B`` images into conv2d's leading "batch" axis.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out.shape
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}, expected 3D or 4D")
    weight = weight.to(hz.c)

    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    c_img = hz.c.view(B, C, H, W)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(hz.c) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out_c.shape
    new_c = out_c.reshape(-1, 1)
    n_out_per_sample = Cp * Hp * Wp

    new_Gc = _conv2d_generators(
        hz.Gc, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _spatial_op_generators(G, op_fn, B, C, H, W, n_out_per_sample):
    """Apply a linear BCHW spatial operator to each HZ generator column.

    If y = S(x) + b and x = c + G xi, then each output generator is
    G'[:, j] = vec(S(unvec(G[:, j]))), without materializing the matrix for S.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = op_fn(imgs)
    _, Cp, Hp, Wp = out.shape
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def _hz_spatial_affine(hz: HZono, op_fn, input_shape, bias=None) -> HZono:
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    out_c = op_fn(hz.c.view(B, C, H, W))
    _, Cp, Hp, Wp = out_c.shape
    if bias is not None:
        out_c = out_c + bias.to(hz.c).view(1, -1, 1, 1)
    n_out = Cp * Hp * Wp
    return HZono(
        c=out_c.reshape(-1, 1),
        Gc=_spatial_op_generators(hz.Gc, op_fn, B, C, H, W, n_out),
        Gb=_spatial_op_generators(hz.Gb, op_fn, B, C, H, W, n_out),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def hz_avgpool2d(
    hz,
    kernel_size,
    stride,
    padding,
    input_shape,
    *,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> HZono:
    op = lambda x: F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride if stride is not None else kernel_size,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    return _hz_spatial_affine(hz, op, input_shape)


def tf_avgpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_avgpool2d(
                hz_in,
                L.params.get("kernel_size"),
                L.params.get("stride"),
                L.params.get("padding", 0),
                ishape,
                ceil_mode=bool(L.params.get("ceil_mode", False)),
                count_include_pad=bool(L.params.get("count_include_pad", True)),
                divisor_override=L.params.get("divisor_override"),
            )
        else:
            hz_in = None
    fact = interval.tf_avgpool2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def hz_convtranspose2d(
    hz, weight, bias, stride, padding, output_padding, dilation, groups, input_shape
) -> HZono:
    weight = weight.to(hz.c)
    op = lambda x: F.conv_transpose2d(
        x,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    return _hz_spatial_affine(hz, op, input_shape, bias=bias)


def tf_convtranspose2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_convtranspose2d(
                hz_in,
                L.params["weight"],
                L.params.get("bias"),
                L.params.get("stride", 1),
                L.params.get("padding", 0),
                L.params.get("output_padding", 0),
                L.params.get("dilation", 1),
                L.params.get("groups", 1),
                ishape,
            )
        else:
            hz_in = None
    fact = interval.tf_convtranspose2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact
