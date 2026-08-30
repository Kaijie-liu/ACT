"""Disconnected candidate for exact ADD -> compact exact-ReLU deferral.

The candidate starts *after* the affine ADD expression has been formed in a
shared sparse-HZ generator frame.  If, and only if, its sole consumer is a
ReLU, that expression is passed directly to the existing exact binary ReLU
primitive.  Otherwise an exact local ADD frame is materialized and retained
for every consumer.

This module is deliberately not wired into verifier dispatch.  Its receipts
have neither proof nor production authority.  It never invokes triangle
relaxation, branch-and-bound, backward propagation, dual tightening, or a
solver.  The only nonlinear call fixes ``compressed=True`` and
``valid_cuts=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType
import time
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.forward_exact_relu_census import (
    sparse_hz_payload_breakdown,
)
from act.back_end.hybridz_tf.tf_mlp import (
    _sparse_fresh_ids,
    sparse_empty,
    sparse_hz_apply_relu_exact,
)
from act.back_end.hybridz_tf import forward_exact_relu_numeric_candidate as _numeric


SCHEMA = "forward_exact_add_relu_deferral_candidate_v1"
EXACT_RELU_KIND = "RELU"


@dataclass(frozen=True)
class ExactAddReLUDeferralBuild:
    """One non-authoritative candidate result.

    ``add`` is absent only on the eligible single-ReLU route, because no local
    ADD frame was created.  ``relu`` is present exactly when a ReLU consumer
    exists.  A multi-consumer build keeps ``add`` even when it also builds the
    ReLU branch.
    """

    add: Optional[SparseHZono]
    relu: Optional[SparseHZono]
    eligible: bool
    used_deferral: bool
    reason: str
    receipt: Mapping[str, Any]

    @property
    def output(self) -> SparseHZono:
        if self.relu is not None:
            return self.relu
        if self.add is not None:
            return self.add
        raise RuntimeError("candidate build has no output")


def _strict_exact_hz(hz: Any, *, name: str) -> SparseHZono:
    if type(hz) is not SparseHZono:
        raise TypeError(f"{name} must be an exact SparseHZono")
    vectors = (hz.c, hz.b, hz.ub, hz.col_ids, hz.bcol_ids)
    for value in vectors:
        if value is None:
            continue
        if type(value) is not np.ndarray or value.ndim != 1:
            raise TypeError(f"{name} contains a non-exact vector")
        if value.dtype not in (
            np.dtype(np.float64),
            np.dtype(np.int64),
        ):
            raise TypeError(f"{name} contains an unsupported vector dtype")
        if value.dtype == np.dtype(np.float64) and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        if not value.flags.c_contiguous or any(stride < 0 for stride in value.strides):
            raise ValueError(f"{name} vectors must be contiguous with positive layout")
    matrices = (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)
    for matrix in matrices:
        if matrix is None:
            continue
        if type(matrix) is not sp.csr_matrix or matrix.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} contains a non-exact CSR matrix")
        if not matrix.has_canonical_format or not matrix.has_sorted_indices:
            raise ValueError(f"{name} CSR matrices must be canonical and sorted")
        if matrix.nnz and (
            not np.all(np.isfinite(matrix.data))
            or np.any(matrix.data == 0.0)
        ):
            raise ValueError(f"{name} CSR data must be finite and explicitly nonzero")
    if hz.col_ids is None or hz.bcol_ids is None:
        raise ValueError(f"{name} requires stable continuous and binary ids")
    return hz


def _strict_consumers(value: Any) -> Tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("consumer_kinds must be an exact tuple")
    if any(type(item) is not str or not item for item in value):
        raise TypeError("consumer kinds must be nonempty exact strings")
    return value


def _strict_bounds(value: Any, *, width: int) -> Bounds:
    if type(value) is not Bounds:
        raise TypeError("pre_bounds must be an exact Bounds")
    lb = value.lb.detach().cpu().double().numpy().reshape(-1)
    ub = value.ub.detach().cpu().double().numpy().reshape(-1)
    if lb.size != width or ub.size != width:
        raise ValueError("pre_bounds width mismatch")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)) or np.any(lb > ub):
        raise ValueError("pre_bounds must be finite and ordered")
    return value


def _rigorous_box_arrays(hz: SparseHZono) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center/radius bounds rounded outward from stored coefficients."""

    lower = np.empty(hz.n_out, dtype=np.float64)
    upper = np.empty(hz.n_out, dtype=np.float64)
    radius_upper = np.empty(hz.n_out, dtype=np.float64)
    for row in range(hz.n_out):
        terms = [
            Fraction.from_float(abs(float(value)))
            for value in hz.Gc.data[hz.Gc.indptr[row] : hz.Gc.indptr[row + 1]]
        ]
        if hz.n_bin:
            terms.extend(
                Fraction.from_float(abs(float(value)))
                for value in hz.Gb.data[hz.Gb.indptr[row] : hz.Gb.indptr[row + 1]]
            )
        radius = sum(terms, Fraction(0))
        try:
            radius_float = float(radius)
        except OverflowError as exc:
            raise ValueError(
                "exact coefficient box overflows finite binary64"
            ) from exc
        if Fraction.from_float(radius_float) < radius:
            radius_float = np.nextafter(radius_float, np.inf)
        center = Fraction.from_float(float(hz.c[row]))
        exact_lower = center - radius
        exact_upper = center + radius
        try:
            lo = float(exact_lower)
            hi = float(exact_upper)
        except OverflowError as exc:
            raise ValueError(
                "exact coefficient box overflows finite binary64"
            ) from exc
        if Fraction.from_float(lo) > exact_lower:
            lo = np.nextafter(lo, -np.inf)
        if Fraction.from_float(hi) < exact_upper:
            hi = np.nextafter(hi, np.inf)
        lower[row] = lo
        upper[row] = hi
        radius_upper[row] = radius_float
    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(radius_upper))
    ):
        raise ValueError("exact coefficient box overflows finite binary64")
    return lower, upper, radius_upper


def _fast_finite_bounds(hz: SparseHZono) -> Bounds:
    lower, upper, _radius = _rigorous_box_arrays(hz)
    return Bounds(
        lb=torch.from_numpy(lower.copy()).reshape(1, -1).double(),
        ub=torch.from_numpy(upper.copy()).reshape(1, -1).double(),
    )


def _state(hz: SparseHZono) -> Mapping[str, int]:
    payload = sparse_hz_payload_breakdown(hz)
    return MappingProxyType(
        {
            "C": int(hz.n_cont),
            "B": int(hz.n_bin),
            "E": int(hz.n_eq),
            "U": int(hz.n_ub),
            "constraint_nnz": int(hz.constraint_nnz),
            "value_nnz": int(hz.value_nnz),
            "payload_bytes": int(payload["payload_bytes"]),
        }
    )


def materialize_exact_add_frame(add_hz: SparseHZono) -> SparseHZono:
    """Create an exact local box frame tied to ``add_hz`` by two upper rows.

    This mirrors the equality-as-two-inequalities shape used by the Operator-HZ
    ADD materializer.  The unconstrained box is only a coordinate frame; the
    signed row pair preserves equality to the original affine expression.
    Rows with zero radius are already constants and need no new coordinate.
    """

    hz = _strict_exact_hz(add_hz, name="add_hz")
    _lower, _upper, radius = _rigorous_box_arrays(hz)
    center = hz.c.copy()
    live = np.flatnonzero(radius > 0.0).astype(np.int32)
    k = int(live.size)
    n_cont = int(hz.n_cont + k)
    n_bin = int(hz.n_bin)

    rows = live
    cols = np.arange(k, dtype=np.int32)
    out_Gc = sp.csr_matrix(
        (radius[live], (rows, hz.n_cont + cols)),
        shape=(hz.n_out, n_cont),
        dtype=np.float64,
    )
    out_Gb = sparse_empty(hz.n_out, n_bin)

    # d = local_y - source_x; enforce d <= 0 and -d <= 0.
    local = sp.csr_matrix(
        (radius[live], (live, hz.n_cont + cols)),
        shape=(hz.n_out, n_cont),
        dtype=np.float64,
    )
    source_c = sp.hstack((hz.Gc, sparse_empty(hz.n_out, k)), format="csr")
    forward_c = (local - source_c).tocsr()
    forward_b = (-hz.Gb).tocsr()
    rhs = hz.c - center

    old_Ac = sp.hstack((hz.Ac, sparse_empty(hz.n_eq, k)), format="csr")
    old_Auc = sp.hstack(
        (
            hz.Auc if hz.Auc is not None else sparse_empty(0, hz.n_cont),
            sparse_empty(hz.n_ub, k),
        ),
        format="csr",
    )
    old_Aub = hz.Aub if hz.Aub is not None else sparse_empty(0, n_bin)
    old_ub = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    Auc = sp.vstack((old_Auc, forward_c, -forward_c), format="csr")
    Aub = sp.vstack((old_Aub, forward_b, -forward_b), format="csr")
    ub = np.concatenate((old_ub, rhs, -rhs))
    for matrix in (out_Gc, out_Gb, old_Ac, Auc, Aub):
        matrix.eliminate_zeros()
        matrix.sort_indices()

    col_ids = np.concatenate((hz.col_ids.copy(), _sparse_fresh_ids(k)))
    return SparseHZono(
        c=center,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=old_Ac,
        Ab=hz.Ab.copy(),
        b=hz.b.copy(),
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        col_ids=col_ids,
        bcol_ids=hz.bcol_ids.copy(),
    )


def _eligibility_reason(
    consumer_kinds: Tuple[str, ...],
    *,
    deadline: Optional[float],
) -> Tuple[bool, str]:
    if deadline is not None:
        if type(deadline) is not float or not math.isfinite(deadline):
            raise TypeError("deadline must be a finite exact float")
        if time.monotonic() >= deadline:
            return False, "deadline_expired"
    if len(consumer_kinds) != 1:
        return False, "consumer_count_not_one"
    if consumer_kinds[0] != EXACT_RELU_KIND:
        return False, "sole_consumer_not_relu"
    return True, "single_exact_relu_consumer"


def build_exact_add_relu_deferral_candidate(
    add_hz: SparseHZono,
    *,
    consumer_kinds: Tuple[str, ...],
    pre_bounds: Optional[Bounds] = None,
    deadline: Optional[float] = None,
) -> ExactAddReLUDeferralBuild:
    """Build one exact candidate route from an already-formed ADD expression."""

    hz = _strict_exact_hz(add_hz, name="add_hz")
    consumers = _strict_consumers(consumer_kinds)
    bounds = (
        _fast_finite_bounds(hz)
        if pre_bounds is None
        else _strict_bounds(pre_bounds, width=hz.n_out)
    )
    if pre_bounds is not None:
        # A disconnected candidate has no proof object for caller-supplied
        # tightened bounds.  Accept only the independently reproducible fast
        # box; any tighter claim must fail closed until production integration
        # can attach an authenticated certificate.
        fast_bounds = _fast_finite_bounds(hz)
        if not (
            np.array_equal(
                bounds.lb.detach().cpu().double().numpy(),
                fast_bounds.lb.detach().cpu().double().numpy(),
            )
            and np.array_equal(
                bounds.ub.detach().cpu().double().numpy(),
                fast_bounds.ub.detach().cpu().double().numpy(),
            )
        ):
            raise ValueError("uncertified tightened pre_bounds are not accepted")
    eligible, reason = _eligibility_reason(consumers, deadline=deadline)
    before = _state(hz)
    started = time.perf_counter_ns()

    if eligible:
        add = None
        relu, phase_counts, _info = sparse_hz_apply_relu_exact(
            hz,
            pre_bounds=bounds,
            compressed=True,
            valid_cuts=False,
            return_info=True,
        )
    else:
        add = materialize_exact_add_frame(hz)
        if EXACT_RELU_KIND in consumers:
            relu, phase_counts, _info = sparse_hz_apply_relu_exact(
                add,
                pre_bounds=bounds,
                compressed=True,
                valid_cuts=False,
                return_info=True,
            )
        else:
            relu = None
            phase_counts = (0, 0, 0)
    wall_ns = time.perf_counter_ns() - started
    output = relu if relu is not None else add
    assert output is not None
    after = _state(output)
    receipt = MappingProxyType(
        {
            "schema": SCHEMA,
            "proof_authority": False,
            "production_authority": False,
            "authenticity_verified": False,
            "forward_only": True,
            "exact_binary_relu": relu is not None,
            "compressed": relu is not None,
            "valid_cuts": False,
            "used_deferral": bool(eligible),
            "reason": reason,
            "consumer_kinds": consumers,
            "phase_counts": tuple(int(v) for v in phase_counts),
            "before": before,
            "after": after,
            "wall_ns": int(wall_ns),
        }
    )
    return ExactAddReLUDeferralBuild(
        add=add,
        relu=relu,
        eligible=bool(eligible),
        used_deferral=bool(eligible),
        reason=reason,
        receipt=receipt,
    )


def benchmark_exact_add_handoff(
    add_hz: SparseHZono,
    *,
    warmup: int = 2,
    repeats: int = 9,
) -> Mapping[str, Any]:
    """Measure only ADD handoff: exact materialization versus deferral no-op."""

    hz = _strict_exact_hz(add_hz, name="add_hz")
    if type(warmup) is not int or type(repeats) is not int:
        raise TypeError("warmup and repeats must be exact integers")
    if warmup < 0 or repeats < 3:
        raise ValueError("benchmark requires warmup >= 0 and repeats >= 3")

    materialized_ns = []
    deferred_ns = []
    for iteration in range(warmup + repeats):
        start = time.perf_counter_ns()
        out = materialize_exact_add_frame(hz)
        materialized = time.perf_counter_ns() - start
        if out.n_out != hz.n_out:
            raise AssertionError("materialized ADD width changed")

        start = time.perf_counter_ns()
        forwarded = hz
        # This is deliberately the measured handoff boundary: ADD arithmetic
        # is complete and the sole exact-ReLU consumer retains the expression.
        if forwarded is not hz:
            raise AssertionError("deferral handoff lost identity")
        deferred = max(1, time.perf_counter_ns() - start)
        if iteration >= warmup:
            materialized_ns.append(int(materialized))
            deferred_ns.append(int(deferred))
    mat_median = float(np.median(np.asarray(materialized_ns, dtype=np.float64)))
    def_median = float(np.median(np.asarray(deferred_ns, dtype=np.float64)))
    return MappingProxyType(
        {
            "schema": "forward_exact_add_handoff_benchmark_v1",
            "measured_stage": "post_add_pre_relu_handoff_only",
            "includes_relu": False,
            "includes_add_arithmetic": False,
            "proof_authority": False,
            "production_authority": False,
            "materialized_median_ns": mat_median,
            "deferred_median_ns": def_median,
            "speedup": float(mat_median / max(def_median, 1.0)),
            "repeats": int(repeats),
        }
    )


def benchmark_exact_add_relu_routes(
    add_hz: SparseHZono,
    *,
    warmup: int = 2,
    repeats: int = 9,
) -> Mapping[str, Any]:
    """Paired timing from a completed ADD through compact exact ReLU.

    Bound preparation and ADD arithmetic happen before the clock.  Each timed
    materialized sample includes local-frame construction plus exact ReLU;
    each deferred sample includes the same exact ReLU directly on ``add_hz``.
    Alternating order reduces systematic first-route bias.
    """

    hz = _strict_exact_hz(add_hz, name="add_hz")
    if type(warmup) is not int or type(repeats) is not int:
        raise TypeError("warmup and repeats must be exact integers")
    if warmup < 0 or repeats < 3:
        raise ValueError("benchmark requires warmup >= 0 and repeats >= 3")
    # Benchmark the repaired stored-real graph, not the historical production
    # primitive whose rounded RHS is known to be inexact.  The completed ADD
    # expression is sealed once at the handoff boundary outside the clock.
    _numeric._freeze_sparse_hz(hz)
    materialized_ns = []
    deferred_ns = []
    last_materialized = None
    last_deferred = None

    def run_materialized():
        start = time.perf_counter_ns()
        local = materialize_exact_add_frame(hz)
        _numeric._freeze_sparse_hz(local)
        out = _numeric.build_forward_exact_relu_numeric_candidate(local).hz
        return out, time.perf_counter_ns() - start

    def run_deferred():
        start = time.perf_counter_ns()
        out = _numeric.build_forward_exact_relu_numeric_candidate(hz).hz
        return out, time.perf_counter_ns() - start

    for iteration in range(warmup + repeats):
        if iteration % 2:
            last_deferred, deferred = run_deferred()
            last_materialized, materialized = run_materialized()
        else:
            last_materialized, materialized = run_materialized()
            last_deferred, deferred = run_deferred()
        if iteration >= warmup:
            materialized_ns.append(int(materialized))
            deferred_ns.append(int(deferred))
    assert last_materialized is not None and last_deferred is not None
    mat_median = float(np.median(np.asarray(materialized_ns, dtype=np.float64)))
    def_median = float(np.median(np.asarray(deferred_ns, dtype=np.float64)))
    return MappingProxyType(
        {
            "schema": "forward_exact_add_relu_route_benchmark_v1",
            "measured_stage": "completed_add_through_compact_exact_relu",
            "includes_add_arithmetic": False,
            "includes_bound_preparation": True,
            "includes_relu": True,
            "compressed": True,
            "valid_cuts": False,
            "stored_real_numeric_exact": True,
            "python_fraction_used_by_candidate": False,
            "proof_authority": False,
            "production_authority": False,
            "materialized": _state(last_materialized),
            "deferred": _state(last_deferred),
            "materialized_median_ns": mat_median,
            "deferred_median_ns": def_median,
            "speedup": float(mat_median / max(def_median, 1.0)),
            "repeats": int(repeats),
        }
    )


__all__ = (
    "ExactAddReLUDeferralBuild",
    "SCHEMA",
    "benchmark_exact_add_handoff",
    "benchmark_exact_add_relu_routes",
    "build_exact_add_relu_deferral_candidate",
    "materialize_exact_add_frame",
)
