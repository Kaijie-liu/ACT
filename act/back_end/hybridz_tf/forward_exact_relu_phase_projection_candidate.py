"""Single-stream float64 exact-ReLU phase-projection candidate generator.

This deliberately narrow verifier path uses one generator stream, one
deterministic analytic-corner property/phase selection, triangular elimination
of the changed phases, and one continuous input-factor LP.  The float64 phase
cell and LP are candidate generation only.  A returned input is checked by ACT
using a zero-width interval and an exact stored-binary64 Fraction property
lower bound; candidate arithmetic is never verdict authority.

There is no input sampling, ONNX execution, PGD, triangle relaxation, BaB,
backward propagation, dual tightening, phase retry, or property-row retry.
Any unsupported input, numerical ambiguity, resource limit, or failed replay
raises :class:`ExactReLUPhaseProjectionUnknown` and must map to UNKNOWN.

The selected CSR is emitted directly on CUDA in the finite all-nonzero affine
domain.  Anything outside that domain fails closed without a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf import forward_exact_relu_live_row_stream_candidate as _live
from act.back_end.hybridz_tf import operator_hz as _oh


_SCHEMA = "act.hybridz.forward_exact_relu_phase_projection_candidate.v3"
_SOLVER_TOLERANCE = 1.0e-9
_INTERIOR_MULTIPLIER = 16.0
_MAX_INPUT_FACTORS = 200_000
_MAX_PHASE_ROWS = 200_000
_MAX_LP_NNZ = 200_000_000
_OWNER_SMALL_MATRIX_VALUE = 1.0e-12
_SUPPORTED = _live._SUPPORTED | frozenset({"SCALE", "BIAS"})


class ExactReLUPhaseProjectionUnknown(RuntimeError):
    """Fail-closed outcome for an unsupported or unverified candidate."""


@dataclass(frozen=True)
class ExactReLUPhaseProjectionReceipt:
    schema: str
    status: str
    selected_property_row: int
    input_factors: int
    phase_rows: int
    initial_phase_changes: int
    lp_rows: int
    lp_nnz: int
    candidate_margin: float
    singleton_margin_lower: float
    setup_seconds: float
    first_center_seconds: float
    first_stream_seconds: float
    target_center_seconds: float
    delta_seconds: float
    expansion_seconds: float
    model_seconds: float
    lp_seconds: float
    singleton_seconds: float
    total_seconds: float
    phase_updates: int = 1
    phase_retries: int = 0
    property_rows_selected: int = 1
    property_row_retries: int = 0
    all_unstable_exact: bool = True
    triangle_rows: int = 0
    input_sampling_used: bool = False
    pgd_used: bool = False
    concrete_onnx_execution_used: bool = False
    bab_used: bool = False
    backward_used: bool = False
    dual_tightening_used: bool = False
    singleton_interval_verified: bool = True
    candidate_authority: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False
    generator_streams: int = 1
    generator_representation: str = "gpu_emitted_selected_csr_v1"
    candidate_outward_error_bands_used: bool = False
    intermediate_phase_or_margin_replay_used: bool = False
    base_model_status: str = "OPTIMAL"
    base_candidate_margin: Optional[float] = None
    updated_model_status: Optional[str] = None
    updated_candidate_margin: Optional[float] = None
    repair_selector_rule: str = "base_positive_none"
    repair_selected_rows: int = 0
    repair_selected_row_ids_sha256: str = ""
    repair_updates: int = 0
    repair_missing_rows_appended: int = 0
    repair_definition_rows_appended: int = 0
    owner_instances: int = 1
    owner_solves: int = 1
    resolves_after_base: int = 0
    dual_ray_requests: int = 0
    dual_selector_used: bool = False
    dual_ray_authority: bool = False
    dual_selector_authority: bool = False
    same_owner_warm_update_used: bool = False
    second_solver_used: bool = False
    fallbacks: int = 0
    runtime_menu_used: bool = False
    retries: int = 0
    activation_split_used: bool = False
    input_split_used: bool = False
    enumeration_used: bool = False
    cross_request_cache_used: bool = False
    fixed_cell_generator_streams: int = 1
    phase_delta_streams: int = 1
    updated_full_target_materialized: bool = False
    existing_x_block_reused: bool = False
    base_solver_x_block_reused_without_reassembly: bool = False
    logical_authority_seal_copy: bool = False
    same_stored_binary64_input_for_box_and_terminal: bool = True
    base_logical_nnz: int = 0
    base_loaded_nnz: int = 0
    base_deleted_tiny_nnz: int = 0
    updated_lp_rows: int = 0
    updated_lp_nnz: int = 0
    updated_logical_nnz: int = 0
    device_program_seconds: float = 0.0
    base_lp_seconds: float = 0.0
    repair_delta_seconds: float = 0.0
    repair_assembly_seconds: float = 0.0
    repair_lp_seconds: float = 0.0


@dataclass(frozen=True)
class ExactReLUPhaseProjectionResult:
    decoded_input: np.ndarray
    input_layer_id: int
    output_layer_id: int
    assert_layer_id: int
    receipt: ExactReLUPhaseProjectionReceipt


def _deadline(deadline: Optional[float], stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise ExactReLUPhaseProjectionUnknown(
            f"phase-projection deadline expired at {stage}"
        )


def _inward_factor_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray,
    rows: np.ndarray,
    tolerance: float,
) -> Tuple[Tuple[float, float], ...]:
    """Derive binary64 factor bounds strictly inside the exact input box."""

    result = []
    for raw_row in rows:
        row = int(raw_row)
        c = Fraction.from_float(float(center[row]))
        r = Fraction.from_float(float(radius[row]))
        if r <= 0:
            raise ExactReLUPhaseProjectionUnknown(
                "active input factor has nonpositive radius"
            )
        exact_lower = (Fraction.from_float(float(lower[row])) - c) / r
        exact_upper = (Fraction.from_float(float(upper[row])) - c) / r
        if exact_lower < -1 or exact_upper > 1:
            raise ExactReLUPhaseProjectionUnknown(
                "represented input box does not enclose the raw BOX"
            )
        lo = float(exact_lower)
        hi = float(exact_upper)
        while Fraction.from_float(lo) < exact_lower:
            lo = float(np.nextafter(lo, np.inf))
        while Fraction.from_float(hi) > exact_upper:
            hi = float(np.nextafter(hi, -np.inf))
        guard = _INTERIOR_MULTIPLIER * tolerance * (
            1.0 + max(abs(lo), abs(hi))
        )
        lo = float(np.nextafter(lo + guard, np.inf))
        hi = float(np.nextafter(hi - guard, -np.inf))
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ExactReLUPhaseProjectionUnknown(
                "input factor interval vanished after inward guard"
            )
        result.append((lo, hi))
    return tuple(result)


def _top1_property(assert_layer: Any, output_width: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        C_value = assert_layer.params["C"]
        threshold_value = assert_layer.params["thresholds"]
        C = np.ascontiguousarray(
            C_value.detach().cpu().double().numpy(), dtype=np.float64
        ).reshape(-1, output_width)
        thresholds = np.ascontiguousarray(
            threshold_value.detach().cpu().double().numpy(), dtype=np.float64
        ).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExactReLUPhaseProjectionUnknown(
            "ASSERT lacks an established linear encoding"
        ) from exc
    if (
        C.shape[0] == 0
        or C.shape[0] != thresholds.size
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(thresholds))
        or np.any(thresholds != 0.0)
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "candidate requires finite zero-threshold TOP1 rows"
        )
    for row in C:
        nonzero = row[row != 0.0]
        if nonzero.size != 2 or sorted(nonzero.tolist()) != [-1.0, 1.0]:
            raise ExactReLUPhaseProjectionUnknown(
                "candidate requires exact two-term TOP1 rows"
            )
    return C, thresholds


def _raw_box_intersection(
    order: Tuple[Any, ...], input_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    lowers = []
    uppers = []
    for layer in order:
        if _oh._kind(layer.kind) != "INPUT_SPEC":
            continue
        if _oh._kind(layer.params.get("kind", "")) != "BOX":
            raise ExactReLUPhaseProjectionUnknown(
                "candidate accepts BOX input specifications only"
            )
        try:
            lower_value = layer.params["lb"]
            upper_value = layer.params["ub"]
            lower = np.ascontiguousarray(
                lower_value.detach().cpu().double().numpy(), dtype=np.float64
            ).reshape(-1)
            upper = np.ascontiguousarray(
                upper_value.detach().cpu().double().numpy(), dtype=np.float64
            ).reshape(-1)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ExactReLUPhaseProjectionUnknown(
                "BOX input specification is malformed"
            ) from exc
        if (
            lower.size != input_width
            or upper.size != input_width
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ExactReLUPhaseProjectionUnknown(
                "BOX input specification is malformed"
            )
        lowers.append(lower)
        uppers.append(upper)
    if not lowers:
        raise ExactReLUPhaseProjectionUnknown(
            "candidate requires at least one BOX input specification"
        )
    effective_lower = np.maximum.reduce(lowers)
    effective_upper = np.minimum.reduce(uppers)
    if np.any(effective_lower > effective_upper):
        raise ExactReLUPhaseProjectionUnknown(
            "BOX input specification intersection is empty"
        )
    return effective_lower, effective_upper


def _exact_singleton_margin_lower(
    row: np.ndarray,
    threshold: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Fraction:
    exact = -Fraction.from_float(float(threshold))
    for coefficient, lower_value, upper_value in zip(row, lower, upper):
        coefficient = float(coefficient)
        if coefficient > 0.0:
            exact += Fraction.from_float(coefficient) * Fraction.from_float(
                float(lower_value)
            )
        elif coefficient < 0.0:
            exact += Fraction.from_float(coefficient) * Fraction.from_float(
                float(upper_value)
            )
    return exact


def _topological(net: Any) -> Tuple[Tuple[Any, ...], Dict[int, Any]]:
    layers = list(net.layers)
    by_id: Dict[int, Any] = {}
    position: Dict[int, int] = {}
    for index, layer in enumerate(layers):
        layer_id = int(layer.id)
        if layer_id in by_id or _oh._kind(layer.kind) not in _SUPPORTED:
            raise ExactReLUPhaseProjectionUnknown(
                "unsupported or duplicate layer"
            )
        by_id[layer_id] = layer
        position[layer_id] = index
    indegree = {layer_id: 0 for layer_id in by_id}
    children = {layer_id: [] for layer_id in by_id}
    for layer_id in by_id:
        for parent_value in net.preds.get(layer_id, []):
            parent = int(parent_value)
            if parent not in by_id:
                raise ExactReLUPhaseProjectionUnknown(
                    "graph predecessor is missing"
                )
            indegree[layer_id] += 1
            children[parent].append(layer_id)
    ready = [
        (position[layer_id], layer_id)
        for layer_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _position, layer_id = heapq.heappop(ready)
        ordered.append(by_id[layer_id])
        for child in sorted(children[layer_id], key=position.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (position[child], child))
    if len(ordered) != len(layers):
        raise ExactReLUPhaseProjectionUnknown("graph is cyclic")
    return tuple(ordered), by_id


def _pointwise_snapshot(layer: Any, *, width: int) -> np.ndarray:
    kind = _oh._kind(layer.kind)
    key = "a" if kind == "SCALE" else "c"
    try:
        value = layer.params[key]
        raw = value.detach().cpu().double().numpy()
        result = np.array(raw, dtype=np.float64, order="C", copy=True).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExactReLUPhaseProjectionUnknown(
            f"{kind} parameter is malformed"
        ) from exc
    if (
        result.size != int(width)
        or not np.all(np.isfinite(result))
        or (kind == "SCALE" and np.any(result == 0.0))
    ):
        raise ExactReLUPhaseProjectionUnknown(
            f"{kind} is outside the finite all-nonzero domain"
        )
    result.setflags(write=False)
    return result


def _float_add_shadow(left: Any, right: Any) -> Any:
    center = np.asarray(left.center) + np.asarray(right.center)
    if not np.all(np.isfinite(center)):
        raise ExactReLUPhaseProjectionUnknown(
            "float candidate ADD overflowed"
        )
    zero = np.zeros(center.size, dtype=np.float64)
    return _live._Shadow(center, zero, np.abs(center))


def _float_relu_shadow(source: Any, frame: Any) -> Any:
    center = np.zeros(source.center.size, dtype=np.float64)
    center[frame.active] = source.center[frame.active]
    if frame.exact.size:
        center[frame.exact] = 0.5 * frame.upper[frame.exact]
    zero = np.zeros(center.size, dtype=np.float64)
    return _live._Shadow(center, zero, np.abs(center))


def _fixed_frame(original: Any, selected: np.ndarray) -> Any:
    empty = np.zeros(0, dtype=np.int64)
    active = np.sort(
        np.concatenate((original.active, original.exact[selected]))
    ).astype(np.int64)
    inactive = np.sort(
        np.concatenate((original.inactive, original.exact[~selected]))
    ).astype(np.int64)
    return _live._PhaseFrame(
        original.lower,
        original.upper,
        active,
        inactive,
        empty,
        empty,
        empty,
        np.zeros(0, dtype=np.float64),
        original.exact.copy(),
        empty,
        empty,
        np.zeros(0, dtype=bool),
        empty,
        empty,
        empty,
    )


def _triangular_input_expansion(
    changes: Any,
    positions: Mapping[int, Mapping[int, int]],
    first_pre: Mapping[int, np.ndarray],
    delta_pre: Mapping[int, np.ndarray],
    *,
    input_width: int,
) -> np.ndarray:
    expansion = np.zeros((len(changes), input_width), dtype=np.float64)
    for index, (layer_id, row, base_active, target_active) in enumerate(changes):
        position = positions[layer_id][row]
        base_q = np.asarray(first_pre[layer_id][position], dtype=np.float64)
        if (not base_active) and target_active:
            expansion[index] = base_q
            if index:
                expansion[index] += np.asarray(
                    delta_pre[layer_id][position, :index]
                    @ expansion[:index],
                    dtype=np.float64,
                )
        elif base_active and (not target_active):
            expansion[index] = -base_q
        else:
            raise ExactReLUPhaseProjectionUnknown(
                "invalid phase change"
            )
    return expansion


def _csr_box_upper(
    matrix: sp.csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    matrix = matrix.tocsr(copy=False)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if matrix.shape[1] != lower.size or lower.shape != upper.shape:
        raise ExactReLUPhaseProjectionUnknown(
            "CSR box dimensions do not match"
        )
    contribution = matrix.data * np.where(
        matrix.data >= 0.0,
        upper[matrix.indices],
        lower[matrix.indices],
    )
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    nonempty = np.diff(matrix.indptr) > 0
    if np.any(nonempty):
        result[nonempty] = np.add.reduceat(
            contribution, matrix.indptr[:-1][nonempty]
        )
    if not np.all(np.isfinite(result)):
        raise ExactReLUPhaseProjectionUnknown(
            "candidate row screening overflowed"
        )
    return result


def _all_nonzero_affine_support_forward(
    snapshot: Any, source_mask: np.ndarray
) -> np.ndarray:
    source_mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    if source_mask.size != snapshot.input_size:
        raise ExactReLUPhaseProjectionUnknown(
            "affine support input width mismatch"
        )
    if snapshot.kind == "DENSE":
        return np.full(
            snapshot.output_size, bool(np.any(source_mask)), dtype=bool
        )
    topology = snapshot.topology
    if topology is None:
        raise ExactReLUPhaseProjectionUnknown(
            "candidate Conv snapshot lost topology"
        )
    batch, in_channels, input_height, input_width = topology.input_shape
    _out_batch, out_channels, _output_height, _output_width = (
        topology.output_shape
    )
    kernel_height, kernel_width = snapshot.weight.shape[2:]
    source = torch.as_tensor(
        source_mask.reshape(batch, in_channels, input_height, input_width),
        dtype=torch.float64,
        device="cuda",
    )
    kernel = torch.ones(
        (
            out_channels,
            snapshot.weight.shape[1],
            kernel_height,
            kernel_width,
        ),
        dtype=torch.float64,
        device="cuda",
    )
    result = torch.nn.functional.conv2d(
        source,
        kernel,
        stride=topology.stride,
        padding=topology.padding,
        dilation=topology.dilation,
        groups=topology.groups,
    )
    return (result > 0.0).detach().cpu().numpy().reshape(-1)


def _all_nonzero_affine_support_backward(
    snapshot: Any, output_mask: np.ndarray
) -> np.ndarray:
    output_mask = np.asarray(output_mask, dtype=bool).reshape(-1)
    if output_mask.size != snapshot.output_size:
        raise ExactReLUPhaseProjectionUnknown(
            "affine demand output width mismatch"
        )
    if snapshot.kind == "DENSE":
        return np.full(
            snapshot.input_size, bool(np.any(output_mask)), dtype=bool
        )
    topology = snapshot.topology
    if topology is None:
        raise ExactReLUPhaseProjectionUnknown(
            "candidate Conv snapshot lost topology"
        )
    batch, in_channels, input_height, input_width = topology.input_shape
    _out_batch, out_channels, output_height, output_width = (
        topology.output_shape
    )
    kernel_height, kernel_width = snapshot.weight.shape[2:]
    base_height = (
        (output_height - 1) * topology.stride[0]
        - 2 * topology.padding[0]
        + topology.dilation[0] * (kernel_height - 1)
        + 1
    )
    base_width = (
        (output_width - 1) * topology.stride[1]
        - 2 * topology.padding[1]
        + topology.dilation[1] * (kernel_width - 1)
        + 1
    )
    output_padding = (
        input_height - base_height,
        input_width - base_width,
    )
    if not (
        0 <= output_padding[0] < topology.stride[0]
        and 0 <= output_padding[1] < topology.stride[1]
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "Conv transpose output padding is malformed"
        )
    source = torch.as_tensor(
        output_mask.reshape(batch, out_channels, output_height, output_width),
        dtype=torch.float64,
        device="cuda",
    )
    kernel = torch.ones(
        (
            out_channels,
            snapshot.weight.shape[1],
            kernel_height,
            kernel_width,
        ),
        dtype=torch.float64,
        device="cuda",
    )
    result = torch.nn.functional.conv_transpose2d(
        source,
        kernel,
        stride=topology.stride,
        padding=topology.padding,
        output_padding=output_padding,
        dilation=topology.dilation,
        groups=topology.groups,
    )
    return (result > 0.0).detach().cpu().numpy().reshape(-1)


def _all_nonzero_live_rows(
    net: Any,
    order: Tuple[Any, ...],
    affines: Mapping[int, Any],
    frames: Mapping[int, Any],
    input_variable_rows: np.ndarray,
    output_layer_id: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    possible = {
        int(layer.id): np.zeros(len(layer.out_vars), dtype=bool)
        for layer in order
    }
    input_layer = next(
        layer for layer in order if _oh._kind(layer.kind) == "INPUT"
    )
    possible[int(input_layer.id)][input_variable_rows] = True
    for layer in order:
        layer_id = int(layer.id)
        kind = _oh._kind(layer.kind)
        if kind == "INPUT":
            continue
        predecessors = tuple(
            int(value) for value in net.preds.get(layer_id, [])
        )
        if kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
            possible[layer_id] = possible[predecessors[0]].copy()
        elif kind in {"CONV2D", "DENSE"}:
            possible[layer_id] = _all_nonzero_affine_support_forward(
                affines[layer_id], possible[predecessors[0]]
            )
        elif kind == "ADD":
            possible[layer_id] = (
                possible[predecessors[0]] | possible[predecessors[1]]
            )
        elif kind in {"SCALE", "BIAS"}:
            possible[layer_id] = possible[predecessors[0]].copy()
        elif kind == "RELU":
            frame = frames[layer_id]
            possible[layer_id][frame.active] = possible[
                predecessors[0]
            ][frame.active]
            possible[layer_id][frame.exact] = True

    demand = {
        int(layer.id): np.zeros(len(layer.out_vars), dtype=bool)
        for layer in order
    }
    demand[int(output_layer_id)][:] = True
    for layer in reversed(order):
        layer_id = int(layer.id)
        kind = _oh._kind(layer.kind)
        predecessors = tuple(
            int(value) for value in net.preds.get(layer_id, [])
        )
        rows = demand[layer_id].copy()
        if kind == "RELU":
            frame = frames[layer_id]
            rows[frame.exact] = True
            demand[layer_id] = rows
            needed = np.zeros(rows.size, dtype=bool)
            needed[frame.active] = rows[frame.active]
            needed[frame.exact] = True
            demand[predecessors[0]] |= needed
        elif kind in {"CONV2D", "DENSE"}:
            demand[predecessors[0]] |= _all_nonzero_affine_support_backward(
                affines[layer_id], rows
            )
        elif kind == "ADD":
            for predecessor in predecessors:
                demand[predecessor] |= rows
        elif kind in {
            "INPUT_SPEC",
            "FLATTEN",
            "ASSERT",
            "SCALE",
            "BIAS",
        } and predecessors:
            demand[predecessors[0]] |= rows
    live = {
        layer_id: np.flatnonzero(
            demand[layer_id] & possible[layer_id]
        ).astype(np.int64)
        for layer_id in demand
    }
    return live, possible


def _row_ids_sha256(values: np.ndarray) -> str:
    ids = np.ascontiguousarray(values, dtype=np.int64).reshape(-1)
    digest = hashlib.sha256()
    digest.update(ids.dtype.str.encode("ascii"))
    digest.update(repr(ids.shape).encode("ascii"))
    digest.update(memoryview(ids).cast("B"))
    return digest.hexdigest()


def _select_optimal_negative_rows(
    *,
    row_value: np.ndarray,
    row_dual: np.ndarray,
    row_ids: np.ndarray,
    loaded_upper: np.ndarray,
    candidate_margin: float,
) -> Tuple[np.ndarray, int, int]:
    """Apply the one frozen optimal-negative selector."""

    values = np.asarray(row_value)
    duals = np.asarray(row_dual)
    ids = np.asarray(row_ids)
    upper = np.asarray(loaded_upper)
    if (
        values.dtype != np.dtype(np.float64)
        or duals.dtype != np.dtype(np.float64)
        or ids.dtype != np.dtype(np.int64)
        or upper.dtype != np.dtype(np.float64)
        or values.ndim != 1
        or values.shape != duals.shape
        or values.shape != ids.shape
        or values.shape != upper.shape
        or not values.size
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(duals))
        or not np.all(np.isfinite(upper))
        or np.any(ids[1:] <= ids[:-1])
        or not math.isfinite(float(candidate_margin))
        or not float(candidate_margin) < 0.0
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "optimal-negative selector frame is malformed"
        )
    residual = upper - values
    tight = residual <= _SOLVER_TOLERANCE * (1.0 + np.abs(upper))
    strict_negative = duals < 0.0
    eligible = tight & strict_negative
    selected = np.ascontiguousarray(ids[eligible], dtype=np.int64)
    if not selected.size:
        raise ExactReLUPhaseProjectionUnknown(
            "optimal-negative selector produced no phase rows"
        )
    return (
        selected,
        int(np.count_nonzero(tight)),
        int(np.count_nonzero(strict_negative)),
    )


def _select_infeasible_ray_rows(
    *,
    row_ray: np.ndarray,
    row_ids: np.ndarray,
    support_row_ids: Tuple[int, ...],
) -> np.ndarray:
    """Apply the one exact-nonzero validated upper-row ray selector."""

    ray = np.asarray(row_ray)
    ids = np.asarray(row_ids)
    if (
        ray.dtype != np.dtype(np.float64)
        or ids.dtype != np.dtype(np.int64)
        or ray.ndim != 1
        or ray.shape != ids.shape
        or not ray.size
        or not np.all(np.isfinite(ray))
        or np.any(ray > 0.0)
        or np.any(ids[1:] <= ids[:-1])
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "infeasible-ray selector frame is malformed"
        )
    selected = np.ascontiguousarray(ids[ray != 0.0], dtype=np.int64)
    if not selected.size or tuple(int(value) for value in selected) != tuple(
        int(value) for value in support_row_ids
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "validated infeasible-ray support mapping drifted"
        )
    return selected


def _terminal_candidate(
    *,
    device_module: Any,
    terminal_program: Any,
    factors: np.ndarray,
    input_rows: np.ndarray,
    input_center: np.ndarray,
    input_radius: np.ndarray,
    input_shape: Tuple[int, ...],
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    property_row: np.ndarray,
    threshold: float,
    output_width: int,
    deadline: Optional[float],
) -> Tuple[np.ndarray, float, float]:
    """Run the sole authority terminal on one shared stored-binary64 object."""

    frozen_factors = np.asarray(factors)
    if (
        frozen_factors.dtype != np.dtype(np.float64)
        or frozen_factors.ndim != 1
        or frozen_factors.shape != input_rows.shape
        or not np.all(np.isfinite(frozen_factors))
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "terminal factor prefix is malformed"
        )
    decoded = np.asarray(raw_lower, dtype=np.float64).copy()
    for column, raw_row in enumerate(input_rows):
        row = int(raw_row)
        exact_value = Fraction.from_float(float(input_center[row]))
        exact_value += Fraction.from_float(
            float(input_radius[row])
        ) * Fraction.from_float(float(frozen_factors[column]))
        decoded[row] = float(exact_value)

    _deadline(deadline, "stored-binary64 terminal input")
    sealed = device_module.seal_terminal_input(decoded.reshape(input_shape))
    stored = sealed.values
    if not (
        stored.shape == (raw_lower.size,)
        and np.all(np.isfinite(stored))
        and np.all(stored >= raw_lower)
        and np.all(stored <= raw_upper)
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "stored-binary64 candidate decoded outside raw BOX"
        )

    singleton_started = time.monotonic()
    point_lower, point_upper = device_module.terminal_interval_forward(
        sealed, terminal_program
    )
    if (
        point_lower.shape != (output_width,)
        or point_upper.shape != (output_width,)
        or not np.all(np.isfinite(point_lower))
        or not np.all(np.isfinite(point_upper))
        or np.any(point_lower > point_upper)
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "terminal device output bounds are malformed"
        )
    exact_margin = _exact_singleton_margin_lower(
        property_row, threshold, point_lower, point_upper
    )
    singleton_seconds = time.monotonic() - singleton_started
    _deadline(deadline, "stored-binary64 terminal return")
    if exact_margin <= 0:
        raise ExactReLUPhaseProjectionUnknown(
            "zero-width outward terminal did not prove the candidate"
        )
    return sealed.values.reshape(input_shape), float(exact_margin), singleton_seconds


def _build_forward_exact_relu_phase_projection_candidate_impl(
    net: Any,
    entry_layer_id: int,
    before: Mapping[int, Any],
    after: Mapping[int, Any],
    *,
    deadline: Optional[float] = None,
    lp_time_limit: float = 30.0,
    device_module: Any,
    owner_module: Any,
    repair_module: Any,
) -> ExactReLUPhaseProjectionResult:
    """Generate one float64 candidate and verify it with the exact terminal rule.

    Candidate phases and the LP have no authority.  The only returned result
    has passed raw-BOX membership, ACT's zero-width forward interval, and the
    stored-binary64 Fraction property check.  Unsupported cases fail closed;
    there is no fallback path.
    """

    started = time.monotonic()
    if type(entry_layer_id) is not int:
        raise ExactReLUPhaseProjectionUnknown(
            "entry layer id is malformed"
        )
    if deadline is not None and (
        type(deadline) not in {int, float}
        or not math.isfinite(float(deadline))
    ):
        raise ExactReLUPhaseProjectionUnknown("deadline is malformed")
    if type(lp_time_limit) not in {int, float} or not (
        math.isfinite(float(lp_time_limit))
        and 0.0 < float(lp_time_limit) <= 30.0
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "LP time limit is malformed"
        )
    deadline = None if deadline is None else float(deadline)
    if not torch.cuda.is_available():
        raise ExactReLUPhaseProjectionUnknown(
            "CUDA is required; no fallback exists"
        )

    order, by_id = _topological(net)
    inputs = [
        layer for layer in order if _oh._kind(layer.kind) == "INPUT"
    ]
    asserts = [
        layer for layer in order if _oh._kind(layer.kind) == "ASSERT"
    ]
    if len(inputs) != 1 or len(asserts) != 1:
        raise ExactReLUPhaseProjectionUnknown(
            "candidate requires exactly one INPUT and one ASSERT"
        )
    input_layer = inputs[0]
    assert_layer = asserts[0]
    output_layer_id = _live._preds(net, assert_layer, 1)[0]
    if entry_layer_id != int(input_layer.id):
        raise ExactReLUPhaseProjectionUnknown(
            "entry layer id does not identify the unique INPUT"
        )
    input_width = len(input_layer.out_vars)
    output_width = len(by_id[output_layer_id].out_vars)
    C, thresholds = _top1_property(assert_layer, output_width)
    represented_lower, represented_upper = _live._facts_box(
        after,
        int(input_layer.id),
        input_width,
        name="phase_projection.input",
    )
    raw_lower, raw_upper = _raw_box_intersection(order, input_width)
    if np.any(represented_lower > raw_lower) or np.any(
        represented_upper < raw_upper
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "represented input box does not enclose the raw BOX"
        )
    try:
        input_tensor = after[int(input_layer.id)].bounds.lb
        input_shape = tuple(int(value) for value in input_tensor.shape)
        if (
            input_tensor.dtype != torch.float64
            or input_tensor.device.type != "cuda"
            or int(input_tensor.numel()) != input_width
        ):
            raise ExactReLUPhaseProjectionUnknown(
                "input fact must be CUDA float64 with exact graph width"
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ExactReLUPhaseProjectionUnknown):
            raise
        raise ExactReLUPhaseProjectionUnknown(
            "input fact is malformed"
        ) from exc

    input_center, input_radius = _oh._enclosing_center_radius(
        represented_lower,
        represented_upper,
        name="phase_projection.input",
    )
    input_rows = np.flatnonzero(input_radius > 0.0).astype(np.int64)
    if not input_rows.size or input_rows.size > _MAX_INPUT_FACTORS:
        raise ExactReLUPhaseProjectionUnknown(
            "candidate requires a non-point input within the factor cap"
        )
    factor_bounds = _inward_factor_bounds(
        raw_lower,
        raw_upper,
        input_center,
        input_radius,
        input_rows,
        _SOLVER_TOLERANCE,
    )
    factor_lower = np.asarray(
        [bound[0] for bound in factor_bounds], dtype=np.float64
    )
    factor_upper = np.asarray(
        [bound[1] for bound in factor_bounds], dtype=np.float64
    )
    original_frames, _n_cont, n_bin = _live._make_phase_frames(
        order,
        before,
        first_continuous_column=int(input_rows.size),
    )
    if n_bin == 0 or n_bin > _MAX_PHASE_ROWS:
        raise ExactReLUPhaseProjectionUnknown(
            "candidate requires unstable ReLUs within the phase cap"
        )

    setup_started = time.monotonic()
    _deadline(deadline, "GPU selected-CSR setup")
    affines: Dict[int, Any] = {}
    pointwise: Dict[int, np.ndarray] = {}
    for layer in order:
        kind = _oh._kind(layer.kind)
        if kind in {"SCALE", "BIAS"}:
            predecessor = _live._preds(net, layer, 1)[0]
            pointwise[int(layer.id)] = _pointwise_snapshot(
                layer, width=len(by_id[predecessor].out_vars)
            )
            continue
        if kind not in {"CONV2D", "DENSE"}:
            continue
        predecessor = _live._preds(net, layer, 1)[0]
        snapshot = _live._affine_snapshot(
            layer, input_size=len(by_id[predecessor].out_vars)
        )
        if np.any(snapshot.weight == 0.0):
            raise ExactReLUPhaseProjectionUnknown(
                "candidate requires all stored affine weights nonzero"
            )
        affines[int(layer.id)] = snapshot
    live_rows, possible_rows = _all_nonzero_live_rows(
        net,
        order,
        affines,
        original_frames,
        input_rows,
        output_layer_id,
    )
    device_matrices: Dict[int, Any] = {}
    for layer_id, snapshot in affines.items():
        _deadline(deadline, f"GPU selected CSR {layer_id}")
        predecessor = _live._preds(net, by_id[layer_id], 1)[0]
        try:
            device_matrices[layer_id] = (
                _live._gpu_selected_affine_matrix(
                    snapshot,
                    live_rows[layer_id],
                    possible_rows[predecessor],
                    name=f"phase_projection.stream[{layer_id}]",
                )
            )
        except _live.ExactReLULiveRowStreamError as exc:
            raise ExactReLUPhaseProjectionUnknown(
                "GPU selected CSR construction failed"
            ) from exc
    programs = device_module.build_request_local_programs(
        net,
        order,
        affines,
        pointwise,
        device_matrices,
        live_rows,
        input_rows=np.ascontiguousarray(input_rows, dtype=np.int64),
        input_radius=np.ascontiguousarray(input_radius, dtype=np.float64),
        assert_layer_id=int(assert_layer.id),
        output_layer_id=int(output_layer_id),
        deadline=deadline,
    )
    candidate_program = programs.candidate
    terminal_program = programs.terminal
    device_program_seconds = float(candidate_program.build_seconds)
    del programs
    setup_seconds = time.monotonic() - setup_started

    def centers(assignments: Optional[Mapping[int, np.ndarray]]):
        shadows: Dict[int, Any] = {}
        pre_centers: Dict[int, np.ndarray] = {}
        selected_map: Dict[int, np.ndarray] = {}
        frames: Dict[int, Any] = {}
        for layer in order:
            layer_id = int(layer.id)
            kind = _oh._kind(layer.kind)
            predecessors = tuple(
                int(value) for value in net.preds.get(layer_id, [])
            )
            if kind == "INPUT":
                zero = np.zeros(input_center.size, dtype=np.float64)
                shadows[layer_id] = _live._Shadow(
                    input_center.copy(), zero, np.abs(input_center)
                )
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                shadows[layer_id] = shadows[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                shadows[layer_id] = device_module.candidate_affine_center(
                    shadows[predecessors[0]],
                    candidate_program.affines[layer_id],
                    layer_id=layer_id,
                )
            elif kind == "SCALE":
                center = (
                    shadows[predecessors[0]].center
                    * pointwise[layer_id]
                )
                if not np.all(np.isfinite(center)):
                    raise ExactReLUPhaseProjectionUnknown(
                        "float candidate SCALE overflowed"
                    )
                zero = np.zeros(center.size, dtype=np.float64)
                shadows[layer_id] = _live._Shadow(
                    center, zero, np.abs(center)
                )
            elif kind == "BIAS":
                center = (
                    shadows[predecessors[0]].center
                    + pointwise[layer_id]
                )
                if not np.all(np.isfinite(center)):
                    raise ExactReLUPhaseProjectionUnknown(
                        "float candidate BIAS overflowed"
                    )
                zero = np.zeros(center.size, dtype=np.float64)
                shadows[layer_id] = _live._Shadow(
                    center, zero, np.abs(center)
                )
            elif kind == "ADD":
                shadows[layer_id] = _float_add_shadow(
                    shadows[predecessors[0]], shadows[predecessors[1]]
                )
            elif kind == "RELU":
                source = shadows[predecessors[0]]
                original = original_frames[layer_id]
                pre_centers[layer_id] = source.center[
                    original.exact
                ].copy()
                default = source.center[original.exact] >= 0.0
                selected = (
                    default
                    if assignments is None
                    else np.asarray(
                        assignments.get(layer_id, default), dtype=bool
                    )
                )
                if selected.shape != (original.exact.size,):
                    raise ExactReLUPhaseProjectionUnknown(
                        "projected phase assignment has the wrong width"
                    )
                selected_map[layer_id] = selected.copy()
                frame = _fixed_frame(original, selected)
                frames[layer_id] = frame
                shadows[layer_id] = _float_relu_shadow(source, frame)
            else:
                raise ExactReLUPhaseProjectionUnknown(
                    f"unsupported graph kind {kind}"
                )
        return (
            selected_map,
            pre_centers,
            shadows[output_layer_id].center,
            frames,
        )

    _deadline(deadline, "first float center")
    first_center_started = time.monotonic()
    (
        first_assign,
        first_pre_center,
        first_output_center,
        first_frames,
    ) = centers(None)
    first_center_seconds = time.monotonic() - first_center_started

    _deadline(deadline, "first generator stream")
    first_schedule = device_module.seal_fixed_phase_schedule(
        candidate_program, first_frames
    )
    first_pre, first_output, first_stream_seconds = (
        device_module.stream_fixed_cell_generators(
            candidate_program, first_schedule
        )
    )
    del first_schedule
    first_objective = np.asarray(C @ first_output, dtype=np.float64)
    first_objective_center = np.asarray(
        C @ first_output_center - thresholds, dtype=np.float64
    )
    first_upper = first_objective_center + np.sum(
        np.abs(first_objective), axis=1, dtype=np.float64
    )
    if not np.all(np.isfinite(first_upper)):
        raise ExactReLUPhaseProjectionUnknown(
            "candidate rival selector overflowed"
        )
    rival = int(np.argmax(first_upper))
    first_coeff = first_objective[rival]
    first_factors = np.where(
        first_coeff >= 0.0, factor_upper, factor_lower
    )

    projected: Dict[int, np.ndarray] = {}
    changes = []
    positions: Dict[int, Dict[int, int]] = {}
    for layer in order:
        layer_id = int(layer.id)
        original = original_frames.get(layer_id)
        if original is None or not original.exact.size:
            continue
        value = first_pre_center[layer_id] + np.asarray(
            first_pre[layer_id] @ first_factors, dtype=np.float64
        )
        if np.any(value == 0.0) or not np.all(np.isfinite(value)):
            raise ExactReLUPhaseProjectionUnknown(
                "float projection hit a zero or nonfinite phase"
            )
        selected = value >= 0.0
        projected[layer_id] = selected
        rows = np.asarray(original.stream_rows, dtype=np.int64)
        positions[layer_id] = {
            int(row): position for position, row in enumerate(rows)
        }
        for position in np.flatnonzero(
            selected != first_assign[layer_id]
        ):
            changes.append(
                (
                    layer_id,
                    int(rows[position]),
                    bool(first_assign[layer_id][position]),
                    bool(selected[position]),
                )
            )

    _deadline(deadline, "projected float center")
    target_center_started = time.monotonic()
    (
        target_assign,
        target_pre_center,
        target_output_center,
        target_frames,
    ) = centers(projected)
    target_center_seconds = time.monotonic() - target_center_started
    width_total = len(changes)
    delta_started = time.monotonic()
    initial_delta_schedule = device_module.seal_delta_schedule(
        candidate_program, first_frames, target_frames, changes
    )
    delta_pre, delta_output, _initial_delta_kernel_seconds = (
        device_module.stream_phase_deltas(
            candidate_program, initial_delta_schedule
        )
    )
    del initial_delta_schedule
    delta_seconds = time.monotonic() - delta_started

    _deadline(deadline, "triangular phase elimination")
    expansion_started = time.monotonic()
    U = _triangular_input_expansion(
        changes,
        positions,
        first_pre,
        delta_pre,
        input_width=int(input_rows.size),
    )
    target_pre = {
        layer_id: np.asarray(first_pre[layer_id], dtype=np.float64)
        + np.asarray(delta_pre[layer_id] @ U, dtype=np.float64)
        for layer_id in first_pre
    }
    target_output = np.asarray(first_output, dtype=np.float64) + np.asarray(
        delta_output @ U, dtype=np.float64
    )
    expansion_seconds = time.monotonic() - expansion_started

    _deadline(deadline, "candidate LP assembly")
    model_started = time.monotonic()
    blocks = []
    rhs = []
    phase_center_parts = []
    base_active_parts = []
    physical_rows = []
    total_phases = 0
    for layer in order:
        layer_id = int(layer.id)
        original = original_frames.get(layer_id)
        if original is None or not original.exact.size:
            continue
        matrix = sp.csr_matrix(target_pre[layer_id])
        selected = target_assign[layer_id]
        blocks.append(
            matrix.multiply(
                np.where(selected, -1.0, 1.0)[:, None]
            ).tocsr()
        )
        center = target_pre_center[layer_id]
        rhs.append(np.where(selected, center, -center))
        phase_center_parts.append(np.asarray(center, dtype=np.float64))
        base_active_parts.append(np.asarray(selected, dtype=np.bool_))
        stream_rows = np.asarray(original.stream_rows, dtype=np.int64)
        if stream_rows.shape != selected.shape:
            raise ExactReLUPhaseProjectionUnknown(
                "base phase-row physical mapping drifted"
            )
        physical_rows.extend(
            (layer_id, int(position), int(row))
            for position, row in enumerate(stream_rows)
        )
        total_phases += int(original.exact.size)
    A = _live._canonical(
        sp.vstack(blocks, format="csr"), name="phase_projection.base_rows"
    )
    b = np.ascontiguousarray(np.concatenate(rhs), dtype=np.float64)
    phase_centers = np.ascontiguousarray(
        np.concatenate(phase_center_parts), dtype=np.float64
    )
    base_active = np.ascontiguousarray(
        np.concatenate(base_active_parts), dtype=np.bool_
    )
    full_row_ids = np.arange(total_phases, dtype=np.int64)
    if (
        A.shape != (total_phases, input_rows.size)
        or phase_centers.shape != (total_phases,)
        or base_active.shape != (total_phases,)
        or len(physical_rows) != total_phases
        or len(set(physical_rows)) != total_phases
        or len({(layer_id, row) for layer_id, _position, row in physical_rows})
        != total_phases
        or A.nnz > _MAX_LP_NNZ
        or not np.all(np.isfinite(A.data))
        or not np.all(np.isfinite(b))
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "candidate LP shape, nnz, or values are invalid"
    )
    row_max = _csr_box_upper(A, factor_lower, factor_upper)
    keep = np.ascontiguousarray(row_max > b, dtype=np.bool_)
    if not np.any(keep):
        raise ExactReLUPhaseProjectionUnknown(
            "base phase screen retained no owner rows"
        )
    screened_A = _live._canonical(
        A[keep].tocsr(), name="phase_projection.screened_base_rows"
    )
    screened_b = np.ascontiguousarray(b[keep], dtype=np.float64)
    objective_coeff = np.asarray(
        C[[rival]] @ target_output, dtype=np.float64
    ).reshape(-1)
    objective_center = float(
        C[rival] @ target_output_center - thresholds[rival]
    )
    if not (
        np.all(np.isfinite(objective_coeff))
        and np.isfinite(objective_center)
    ):
        raise ExactReLUPhaseProjectionUnknown(
            "candidate objective overflowed"
        )
    _deadline(deadline, "candidate input LP")
    row_lower = np.full(screened_A.shape[0], -np.inf, dtype=np.float64)
    base_rows = owner_module.FrozenRows.from_csr(
        screened_A,
        row_lower=row_lower,
        row_upper=screened_b,
        row_ids=np.ascontiguousarray(full_row_ids[keep], dtype=np.int64),
        column_lower=np.ascontiguousarray(factor_lower, dtype=np.float64),
        column_upper=np.ascontiguousarray(factor_upper, dtype=np.float64),
    )
    base_lp_rows_count = int(base_rows.rows)
    base_logical_nnz_count = int(base_rows.logical_nnz)
    base_loaded_nnz_count = int(base_rows.data.size)
    base_deleted_tiny_nnz_count = int(base_rows.deleted_tiny_nnz)
    model_seconds = time.monotonic() - model_started

    owner_started = time.monotonic()
    owner_deadline = owner_started + float(lp_time_limit)
    if deadline is not None:
        owner_deadline = min(owner_deadline, deadline)
    if not math.isfinite(owner_deadline) or owner_deadline <= owner_started:
        raise ExactReLUPhaseProjectionUnknown(
            "phase-projection deadline expired before owner construction"
        )

    base_model_status = "UNRESOLVED"
    updated_model_status: Optional[str] = None
    base_candidate_margin: Optional[float] = None
    updated_candidate_margin: Optional[float] = None
    candidate_margin: Optional[float] = None
    repair_selector_rule = "base_positive_none"
    selected_ordinals = np.empty(0, dtype=np.int64)
    repair_updates = 0
    owner_solves = 0
    dual_ray_requests = 0
    tight_rows = 0
    strict_negative_duals = 0
    repair_delta_seconds = 0.0
    repair_assembly_seconds = 0.0
    repair_lp_seconds = 0.0
    repair_missing_rows = 0
    repair_definition_rows = 0
    updated_lp_rows = int(base_rows.rows)
    updated_lp_nnz = int(base_rows.data.size)
    updated_logical_nnz = int(base_rows.logical_nnz)
    final_factors: Optional[np.ndarray] = None
    base_result: Any = None
    updated_result: Any = None
    repair_plan: Any = None

    base_lp_started = time.monotonic()
    with owner_module.SafeHighsOwner(
        deadline_monotonic=float(owner_deadline)
    ) as highs_owner:
        base_result = highs_owner.solve_base(
            cost=np.ascontiguousarray(-objective_coeff, dtype=np.float64),
            column_lower=np.ascontiguousarray(factor_lower, dtype=np.float64),
            column_upper=np.ascontiguousarray(factor_upper, dtype=np.float64),
            rows=base_rows,
        )
        owner_solves = 1
        base_lp_seconds = time.monotonic() - base_lp_started

        if isinstance(base_result, owner_module.OptimalSelector):
            base_model_status = "OPTIMAL"
            if not np.array_equal(base_result.row_ids, base_rows.row_ids):
                raise ExactReLUPhaseProjectionUnknown(
                    "base optimal row-id mapping drifted"
                )
            base_factors = np.asarray(base_result.factors, dtype=np.float64)
            base_candidate_margin = float(
                objective_center + objective_coeff @ base_factors
            )
            if not math.isfinite(base_candidate_margin):
                raise ExactReLUPhaseProjectionUnknown(
                    "base candidate margin is nonfinite"
                )
            if base_candidate_margin > 0.0:
                final_factors = np.array(
                    base_factors[: input_rows.size],
                    dtype=np.float64,
                    order="C",
                    copy=True,
                )
                candidate_margin = base_candidate_margin
            elif base_candidate_margin < 0.0:
                repair_selector_rule = (
                    "optimal_negative_all_tight_strict_negative_upper_row_dual"
                )
                (
                    selected_ordinals,
                    tight_rows,
                    strict_negative_duals,
                ) = _select_optimal_negative_rows(
                    row_value=base_result.row_value,
                    row_dual=base_result.row_dual,
                    row_ids=base_result.row_ids,
                    loaded_upper=base_rows.upper,
                    candidate_margin=base_candidate_margin,
                )
            else:
                raise ExactReLUPhaseProjectionUnknown(
                    "zero base candidate margin is not repair-eligible"
                )
        elif isinstance(base_result, owner_module.InfeasibleRaySelector):
            base_model_status = "INFEASIBLE"
            dual_ray_requests = 1
            if not np.array_equal(base_result.row_ids, base_rows.row_ids):
                raise ExactReLUPhaseProjectionUnknown(
                    "base infeasible row-id mapping drifted"
                )
            repair_selector_rule = (
                "infeasible_all_exact_nonzero_validated_dual_ray_phase_rows"
            )
            selected_ordinals = _select_infeasible_ray_rows(
                row_ray=base_result.row_ray,
                row_ids=base_result.row_ids,
                support_row_ids=base_result.support_row_ids,
            )
        else:
            raise ExactReLUPhaseProjectionUnknown(
                "base owner status is unresolved"
            )

        if final_factors is None:
            if (
                selected_ordinals.ndim != 1
                or not selected_ordinals.size
                or np.any(selected_ordinals < 0)
                or np.any(selected_ordinals >= total_phases)
                or np.any(selected_ordinals[1:] <= selected_ordinals[:-1])
                or not np.all(keep[selected_ordinals])
            ):
                raise ExactReLUPhaseProjectionUnknown(
                    "repair selector row IDs are outside the frozen base mapping"
                )

            repair_assign = {
                layer_id: np.asarray(value, dtype=np.bool_).copy()
                for layer_id, value in target_assign.items()
            }
            repair_changes = []
            for raw_ordinal in selected_ordinals:
                layer_id, position, row = physical_rows[int(raw_ordinal)]
                base_value = bool(repair_assign[layer_id][position])
                repair_assign[layer_id][position] = not base_value
                repair_changes.append(
                    (layer_id, row, base_value, not base_value)
                )
            repair_frames = {
                layer_id: _fixed_frame(
                    original_frames[layer_id], repair_assign[layer_id]
                )
                for layer_id in original_frames
            }

            _deadline(deadline, "single repair delta")
            repair_delta_started = time.monotonic()
            repair_delta_schedule = device_module.seal_delta_schedule(
                candidate_program,
                target_frames,
                repair_frames,
                repair_changes,
            )
            repair_delta_pre, repair_delta_output, _repair_kernel_seconds = (
                device_module.stream_phase_deltas(
                    candidate_program, repair_delta_schedule
                )
            )
            del repair_delta_schedule
            repair_delta_seconds = time.monotonic() - repair_delta_started

            repair_delta = np.ascontiguousarray(
                np.concatenate(
                    [
                        repair_delta_pre[int(layer.id)]
                        for layer in order
                        if int(layer.id) in original_frames
                    ],
                    axis=0,
                ),
                dtype=np.float64,
            )
            objective_delta = np.ascontiguousarray(
                np.asarray(
                    C[[rival]] @ repair_delta_output, dtype=np.float64
                ).reshape(-1),
                dtype=np.float64,
            )
            repair_assembly_started = time.monotonic()
            repair_plan = repair_module.build_incremental_repair(
                full_oriented_rows=A,
                phase_centers=phase_centers,
                base_active=base_active,
                keep=keep,
                full_row_ids=full_row_ids,
                base_rows=base_rows,
                x_lower=np.ascontiguousarray(factor_lower, dtype=np.float64),
                x_upper=np.ascontiguousarray(factor_upper, dtype=np.float64),
                selected_ordinals=np.ascontiguousarray(
                    selected_ordinals, dtype=np.int64
                ),
                delta=repair_delta,
                objective_delta=objective_delta,
                deadline_monotonic=float(owner_deadline),
            )
            repair_plan.assert_intact()
            repair_assembly_seconds = (
                time.monotonic() - repair_assembly_started
            )
            repair_missing_rows = int(repair_plan.missing_rows_appended)
            repair_definition_rows = int(
                repair_plan.definition_rows_appended
            )
            repair_lp_started = time.monotonic()
            updated_result = highs_owner.apply_incremental_update(
                new_columns=repair_plan.new_columns,
                existing_row_lower=repair_plan.existing_row_lower,
                existing_row_upper=repair_plan.existing_row_upper,
                appended_rows=repair_plan.appended_rows,
            )
            repair_lp_seconds = time.monotonic() - repair_lp_started
            repair_updates = 1
            owner_solves = 2
            updated_lp_rows = int(
                base_rows.rows + repair_plan.appended_rows.rows
            )
            updated_lp_nnz = int(
                base_rows.data.size
                + np.count_nonzero(
                    np.abs(repair_plan.new_columns.data)
                    > _OWNER_SMALL_MATRIX_VALUE
                )
                + repair_plan.appended_rows.data.size
            )
            updated_logical_nnz = int(
                base_rows.logical_nnz
                + repair_plan.new_columns.data.size
                + repair_plan.appended_rows.logical_nnz
            )
            if not isinstance(updated_result, owner_module.OptimalCandidate):
                updated_model_status = "UNRESOLVED"
                raise ExactReLUPhaseProjectionUnknown(
                    "single repair owner status is unresolved"
                )
            updated_model_status = "OPTIMAL"
            full_factors = np.asarray(
                updated_result.factors, dtype=np.float64
            )
            expected_width = input_rows.size + selected_ordinals.size
            if full_factors.shape != (expected_width,) or not np.all(
                np.isfinite(full_factors)
            ):
                raise ExactReLUPhaseProjectionUnknown(
                    "single repair returned malformed factors"
                )
            updated_candidate_margin = float(
                objective_center
                + objective_coeff @ full_factors[: input_rows.size]
                + objective_delta @ full_factors[input_rows.size :]
            )
            if not (
                math.isfinite(updated_candidate_margin)
                and updated_candidate_margin > 0.0
            ):
                raise ExactReLUPhaseProjectionUnknown(
                    "single repair did not produce a positive candidate margin"
                )
            final_factors = np.array(
                full_factors[: input_rows.size],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            candidate_margin = updated_candidate_margin

    owner_state_after_close = highs_owner.state
    if owner_state_after_close != "CLOSED":
        raise ExactReLUPhaseProjectionUnknown(
            "HiGHS owner was not closed before terminal replay"
        )

    # The terminal owns disjoint device weights.  Release all candidate-side
    # native and large Python references before constructing its authority input.
    del highs_owner, base_result, updated_result, repair_plan, base_rows
    del first_pre, first_output, delta_pre, delta_output, target_pre, target_output
    del device_matrices, affines, changes
    if repair_updates:
        del repair_changes, repair_delta_pre, repair_delta_output
        del repair_delta, objective_delta, repair_frames, repair_assign
    A = None
    screened_A = None
    first_pre_center = None
    target_pre_center = None
    candidate_program = None
    centers = None

    if final_factors is None or candidate_margin is None:
        raise ExactReLUPhaseProjectionUnknown(
            "owner closed without a positive candidate"
        )
    decoded_input, exact_margin, singleton_seconds = _terminal_candidate(
        device_module=device_module,
        terminal_program=terminal_program,
        factors=final_factors,
        input_rows=input_rows,
        input_center=input_center,
        input_radius=input_radius,
        input_shape=input_shape,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
        property_row=C[rival],
        threshold=float(thresholds[rival]),
        output_width=output_width,
        deadline=deadline,
    )

    lp_seconds = base_lp_seconds + repair_lp_seconds
    receipt = ExactReLUPhaseProjectionReceipt(
        schema=_SCHEMA,
        status="singleton_verified",
        selected_property_row=rival,
        input_factors=int(input_rows.size),
        phase_rows=total_phases,
        initial_phase_changes=width_total,
        lp_rows=base_lp_rows_count,
        lp_nnz=base_logical_nnz_count,
        candidate_margin=float(candidate_margin),
        singleton_margin_lower=float(exact_margin),
        setup_seconds=float(setup_seconds),
        first_center_seconds=float(first_center_seconds),
        first_stream_seconds=float(first_stream_seconds),
        target_center_seconds=float(target_center_seconds),
        delta_seconds=float(delta_seconds),
        expansion_seconds=float(expansion_seconds),
        model_seconds=float(model_seconds),
        lp_seconds=float(lp_seconds),
        singleton_seconds=float(singleton_seconds),
        total_seconds=float(time.monotonic() - started),
        generator_representation=(
            "request_local_device_program_incremental_lowrank_v1"
        ),
        base_model_status=base_model_status,
        base_candidate_margin=base_candidate_margin,
        updated_model_status=updated_model_status,
        updated_candidate_margin=updated_candidate_margin,
        repair_selector_rule=repair_selector_rule,
        repair_selected_rows=int(selected_ordinals.size),
        repair_selected_row_ids_sha256=_row_ids_sha256(selected_ordinals),
        repair_updates=repair_updates,
        repair_missing_rows_appended=repair_missing_rows,
        repair_definition_rows_appended=repair_definition_rows,
        owner_instances=1,
        owner_solves=owner_solves,
        resolves_after_base=repair_updates,
        dual_ray_requests=dual_ray_requests,
        dual_selector_used=bool(repair_updates),
        dual_ray_authority=False,
        dual_selector_authority=False,
        same_owner_warm_update_used=bool(repair_updates),
        second_solver_used=False,
        fallbacks=0,
        runtime_menu_used=False,
        retries=0,
        activation_split_used=False,
        input_split_used=False,
        enumeration_used=False,
        cross_request_cache_used=False,
        fixed_cell_generator_streams=1,
        phase_delta_streams=1 + repair_updates,
        updated_full_target_materialized=False,
        existing_x_block_reused=bool(repair_updates),
        base_solver_x_block_reused_without_reassembly=bool(repair_updates),
        logical_authority_seal_copy=bool(repair_updates),
        same_stored_binary64_input_for_box_and_terminal=True,
        base_logical_nnz=base_logical_nnz_count,
        base_loaded_nnz=base_loaded_nnz_count,
        base_deleted_tiny_nnz=base_deleted_tiny_nnz_count,
        updated_lp_rows=updated_lp_rows,
        updated_lp_nnz=updated_lp_nnz,
        updated_logical_nnz=updated_logical_nnz,
        device_program_seconds=device_program_seconds,
        base_lp_seconds=base_lp_seconds,
        repair_delta_seconds=repair_delta_seconds,
        repair_assembly_seconds=repair_assembly_seconds,
        repair_lp_seconds=repair_lp_seconds,
    )
    return ExactReLUPhaseProjectionResult(
        decoded_input=decoded_input,
        input_layer_id=int(input_layer.id),
        output_layer_id=int(output_layer_id),
        assert_layer_id=int(assert_layer.id),
        receipt=receipt,
    )


def build_forward_exact_relu_phase_projection_candidate(
    net: Any,
    entry_layer_id: int,
    before: Mapping[int, Any],
    after: Mapping[int, Any],
    *,
    deadline: Optional[float] = None,
    lp_time_limit: float = 30.0,
) -> ExactReLUPhaseProjectionResult:
    """Run the single request-local device/owner/terminal transaction."""

    try:
        from act.back_end.hybridz_tf import (
            phase_projection_device_program as device_module,
        )
        from act.back_end.hybridz_tf import (
            phase_projection_highs_owner as owner_module,
        )
        from act.back_end.hybridz_tf import (
            phase_projection_incremental_repair as repair_module,
        )
    except Exception as exc:
        raise ExactReLUPhaseProjectionUnknown(
            "phase-projection request-local modules are unavailable"
        ) from exc

    try:
        return _build_forward_exact_relu_phase_projection_candidate_impl(
            net,
            entry_layer_id,
            before,
            after,
            deadline=deadline,
            lp_time_limit=lp_time_limit,
            device_module=device_module,
            owner_module=owner_module,
            repair_module=repair_module,
        )
    except ExactReLUPhaseProjectionUnknown:
        raise
    except (
        device_module.PhaseProjectionDeviceProgramError,
        owner_module.HighsOwnerUnknown,
        repair_module.IncrementalRepairUnknown,
        _live.ExactReLULiveRowStreamError,
    ) as exc:
        raise ExactReLUPhaseProjectionUnknown(
            "phase-projection request-local module failed closed"
        ) from exc
    except Exception as exc:
        raise ExactReLUPhaseProjectionUnknown(
            "phase-projection request-local transaction failed closed"
        ) from exc
