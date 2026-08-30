"""Synthetic-only census for the forward compact exact-ReLU sparse HZ path.

This module is deliberately disconnected from verifier dispatch.  It records
the structural cost of the existing exact sparse-HZ forward primitives; it
does not certify a property and it has no production authority.  In
particular, the only ReLU call made here fixes ``compressed=True`` and
``valid_cuts=False``.

The reported payload is the owned numeric payload of one ``SparseHZono``:
all reachable NumPy storage plus the ``data``, ``indices``, and ``indptr``
buffers of every CSR matrix, with aliased storage counted once.  Python-object
and allocator overhead are intentionally outside that reproducible count.
"""

from __future__ import annotations

from copy import deepcopy
import math
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_mlp import (
    sparse_hz_add_same_frame,
    sparse_hz_apply_relu_exact,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


SCHEMA = "forward_compact_exact_relu_census_v1"
MAX_SYNTHETIC_WIDTH = 512
PAYLOAD_TRAVERSAL_MAX_DEPTH = 4096
PAYLOAD_TRAVERSAL_MAX_NODES = 100_000
PAYLOAD_TRAVERSAL_CONTRACT = (
    "exact leaves: None, bool, int, finite float, str, bytes",
    "exact containers: dict, list, tuple, set, frozenset",
    "attribute objects: exact SimpleNamespace or non-callable plain instances "
    "with an exact built-in __dict__ and no declared slots",
    "numeric buffers: contiguous float64/int32/int64 ndarray or finite "
    "float64 scipy.sparse.csr_matrix buffers",
    "opaque or over-cap objects fail closed with ValueError",
)
_PAYLOAD_EXACT_LEAF_TYPES = (type(None), bool, int, float, str, bytes)
_PAYLOAD_EXACT_CONTAINER_TYPES = (dict, list, tuple, set, frozenset)


def _strict_float64_vector(value: Any, *, name: str) -> np.ndarray:
    """Accept one finite, contiguous, real binary64 vector without coercion."""

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an exact float64 vector") from exc
    if (
        np.iscomplexobj(array)
        or array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or not array.flags.c_contiguous
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite contiguous float64 vector")
    return array


def _strict_affine_weight(value: Any, *, name: str) -> Any:
    """Validate an affine matrix before production code can coerce its dtype."""

    if sp.issparse(value):
        if (
            np.iscomplexobj(value.data)
            or value.dtype != np.dtype(np.float64)
            or value.ndim != 2
            or not np.all(np.isfinite(value.data))
        ):
            raise ValueError(f"{name} must be a finite real float64 matrix")
        return sp.csr_matrix(value, dtype=np.float64, copy=False)
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an exact float64 matrix") from exc
    if (
        np.iscomplexobj(array)
        or array.dtype != np.dtype(np.float64)
        or array.ndim != 2
        or not array.flags.c_contiguous
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite contiguous float64 matrix")
    return array


def _strict_csr(
    value: Any,
    *,
    name: str,
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    if (
        type(value) is not sp.csr_matrix
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not np.all(np.isfinite(value.data))
    ):
        raise ValueError(f"{name} must be a finite float64 CSR matrix")
    if (
        type(value.data) is not np.ndarray
        or value.data.dtype != np.dtype(np.float64)
        or value.data.ndim != 1
        or not value.data.flags.c_contiguous
        or type(value.indices) is not np.ndarray
        or value.indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or value.indices.ndim != 1
        or not value.indices.flags.c_contiguous
        or type(value.indptr) is not np.ndarray
        or value.indptr.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or value.indptr.ndim != 1
        or not value.indptr.flags.c_contiguous
    ):
        raise ValueError(f"{name} CSR buffers are malformed")
    if (
        int(value.indptr.size) != int(shape[0]) + 1
        or int(value.indptr[0]) != 0
        or int(value.indptr[-1]) != int(value.indices.size)
        or int(value.indices.size) != int(value.data.size)
        or np.any(value.indptr[1:] < value.indptr[:-1])
        or (
            value.indices.size
            and (np.any(value.indices < 0) or np.any(value.indices >= shape[1]))
        )
    ):
        raise ValueError(f"{name} CSR index structure is malformed")
    return value


def _strict_hz(hz: Any, *, name: str) -> SparseHZono:
    """Fail closed on malformed or non-finite public sparse-HZ inputs."""

    if type(hz) is not SparseHZono:
        raise ValueError(f"{name} must be an exact SparseHZono")
    n_out = int(hz.c.size) if type(hz.c) is np.ndarray else -1
    n_cont = int(hz.Gc.shape[1]) if type(hz.Gc) is sp.csr_matrix else -1
    n_bin = int(hz.Gb.shape[1]) if type(hz.Gb) is sp.csr_matrix else -1
    n_eq = int(hz.Ac.shape[0]) if type(hz.Ac) is sp.csr_matrix else -1
    for field_name, value, length in (
        ("c", hz.c, n_out),
        ("b", hz.b, n_eq),
    ):
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.float64)
            or value.ndim != 1
            or int(value.size) != length
            or not value.flags.c_contiguous
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"{name}.{field_name} must be finite contiguous float64")
    _strict_csr(hz.Gc, name=f"{name}.Gc", shape=(n_out, n_cont))
    _strict_csr(hz.Gb, name=f"{name}.Gb", shape=(n_out, n_bin))
    _strict_csr(hz.Ac, name=f"{name}.Ac", shape=(n_eq, n_cont))
    _strict_csr(hz.Ab, name=f"{name}.Ab", shape=(n_eq, n_bin))

    upper_present = (hz.Auc is not None, hz.Aub is not None, hz.ub is not None)
    if any(upper_present) and not all(upper_present):
        raise ValueError(f"{name} has a partial upper-constraint triple")
    if all(upper_present):
        n_ub = int(hz.Auc.shape[0]) if type(hz.Auc) is sp.csr_matrix else -1
        _strict_csr(hz.Auc, name=f"{name}.Auc", shape=(n_ub, n_cont))
        _strict_csr(hz.Aub, name=f"{name}.Aub", shape=(n_ub, n_bin))
        if (
            type(hz.ub) is not np.ndarray
            or hz.ub.dtype != np.dtype(np.float64)
            or hz.ub.ndim != 1
            or int(hz.ub.size) != n_ub
            or not hz.ub.flags.c_contiguous
            or not np.all(np.isfinite(hz.ub))
        ):
            raise ValueError(f"{name}.ub must be finite contiguous float64")

    for field_name, value, length in (
        ("col_ids", hz.col_ids, n_cont),
        ("bcol_ids", hz.bcol_ids, n_bin),
    ):
        if value is None:
            continue
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.int64)
            or value.ndim != 1
            or int(value.size) != length
            or not value.flags.c_contiguous
            or np.unique(value).size != value.size
        ):
            raise ValueError(f"{name}.{field_name} must be unique contiguous int64")
    full_col_ids = vars(hz).get("full_col_ids")
    if full_col_ids is not None and (
        type(full_col_ids) is not np.ndarray
        or full_col_ids.dtype != np.dtype(np.int64)
        or full_col_ids.ndim != 1
        or not full_col_ids.flags.c_contiguous
        or np.unique(full_col_ids).size != full_col_ids.size
    ):
        raise ValueError(f"{name}.full_col_ids must be unique contiguous int64")
    return hz


def _strict_pre_bounds(value: Any, *, n_out: int) -> Bounds:
    if type(value) is not Bounds:
        raise ValueError("pre_bounds must be an exact Bounds instance")
    for name, tensor in (("lb", value.lb), ("ub", value.ub)):
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.float64
            or tensor.is_complex()
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or int(tensor.numel()) != int(n_out)
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"pre_bounds.{name} must be finite CPU float64")
    if value.lb.shape != value.ub.shape or bool(torch.any(value.lb > value.ub).item()):
        raise ValueError("pre_bounds must be shape matched with lb <= ub")
    return value


def _merge_byte_intervals(
    intervals: Iterable[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    ordered = sorted((int(lo), int(hi)) for lo, hi in intervals if hi > lo)
    merged: List[Tuple[int, int]] = []
    for lo, hi in ordered:
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    return merged


def _byte_union_size(intervals: Iterable[Tuple[int, int]]) -> int:
    return int(sum(hi - lo for lo, hi in _merge_byte_intervals(intervals)))


def _plain_attribute_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Return stored attributes only for the frozen plain-object contract."""

    if callable(value) or type(value).__module__ == "builtins":
        return None
    for cls in type(value).__mro__[:-1]:
        if "__slots__" in vars(cls):
            return None
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except Exception:
        return None
    return attributes if type(attributes) is dict else None


def sparse_hz_payload_breakdown(hz: SparseHZono) -> Dict[str, int]:
    """Count every ndarray/CSR buffer reachable from this exact sparse HZ.

    Traversal follows only :data:`PAYLOAD_TRAVERSAL_CONTRACT`; it does not
    inspect descriptors, slots, callables, or arbitrary opaque Python
    objects.  Dynamic payload such as ``full_col_ids`` and arrays stored in
    supported nested attribute objects is included.  Overlapping live byte
    intervals are unioned, including distinct ndarray wrappers over the same
    ultimate backing store.  Bytes reachable as both dense arrays and CSR
    buffers are assigned to the CSR breakdown exactly once.
    """

    _strict_hz(hz, name="hz")
    seen_objects = set()
    queued_objects = set()
    dense_intervals: List[Tuple[int, int]] = []
    csr_intervals: List[Tuple[int, int]] = []
    stack: List[Tuple[Any, int]] = []
    discovered_nodes = 0

    def count_array(value: np.ndarray, *, csr_buffer: bool) -> None:
        if (
            np.iscomplexobj(value)
            or value.dtype
            not in (np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int64))
            or (
                value.dtype == np.dtype(np.float64)
                and not np.all(np.isfinite(value))
            )
        ):
            raise ValueError("hz-owned numeric buffers must have exact finite dtypes")
        if any(int(stride) < 0 for stride in value.strides):
            raise ValueError("hz-owned ndarray buffers must not have negative strides")
        if not (value.flags.c_contiguous or value.flags.f_contiguous):
            raise ValueError("hz-owned ndarray buffers must be contiguous")
        if int(value.nbytes) == 0:
            return
        pointer = int(value.__array_interface__["data"][0])
        interval = (pointer, pointer + int(value.nbytes))
        if csr_buffer:
            csr_intervals.append(interval)
        else:
            dense_intervals.append(interval)

    def schedule(value: Any, depth: int) -> None:
        nonlocal discovered_nodes
        discovered_nodes += 1
        if discovered_nodes > PAYLOAD_TRAVERSAL_MAX_NODES:
            raise ValueError("hz payload traversal node cap exceeded")
        if depth > PAYLOAD_TRAVERSAL_MAX_DEPTH:
            raise ValueError("hz payload traversal depth cap exceeded")
        if type(value) in (type(None), bool, int, str, bytes):
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("hz payload contains a non-finite float leaf")
            return
        object_id = id(value)
        if object_id in seen_objects or object_id in queued_objects:
            return
        queued_objects.add(object_id)
        stack.append((value, depth))

    for attribute in vars(hz).values():
        schedule(attribute, 0)

    while stack:
        value, depth = stack.pop()
        object_id = id(value)
        queued_objects.discard(object_id)
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)

        if type(value) is np.ndarray:
            count_array(value, csr_buffer=False)
            continue
        if type(value) is sp.csr_matrix:
            _strict_csr(value, name="reachable_csr", shape=value.shape)
            count_array(value.data, csr_buffer=True)
            count_array(value.indices, csr_buffer=True)
            count_array(value.indptr, csr_buffer=True)
            continue

        children: Iterable[Any]
        if type(value) is dict:
            children = (
                child
                for item in value.items()
                for child in item
            )
        elif type(value) in (list, tuple, set, frozenset):
            children = value
        elif type(value) is SimpleNamespace:
            children = vars(value).values()
        else:
            if (
                isinstance(value, _PAYLOAD_EXACT_LEAF_TYPES)
                or isinstance(value, _PAYLOAD_EXACT_CONTAINER_TYPES)
                or isinstance(value, (np.ndarray, SimpleNamespace))
                or sp.issparse(value)
            ):
                raise ValueError(
                    "subclass of an exact payload type is not traversable"
                )
            attributes = _plain_attribute_dict(value)
            if attributes is None:
                raise ValueError(
                    f"opaque hz payload object rejected: {type(value).__name__}"
                )
            children = attributes.values()
        for child in children:
            schedule(child, depth + 1)

    csr_bytes = _byte_union_size(csr_intervals)
    total_bytes = _byte_union_size([*csr_intervals, *dense_intervals])
    dense_bytes = total_bytes - csr_bytes
    return {
        "dense_bytes": int(dense_bytes),
        "csr_bytes": int(csr_bytes),
        "payload_bytes": int(total_bytes),
    }


def _state_counts(hz: SparseHZono) -> Dict[str, int]:
    payload = sparse_hz_payload_breakdown(hz)
    return {
        "C": int(hz.n_cont),
        "B": int(hz.n_bin),
        "E": int(hz.n_eq),
        "U": int(hz.n_ub),
        "constraint_rows": int(hz.n_eq + hz.n_ub),
        "constraint_nnz": int(hz.constraint_nnz),
        "value_nnz": int(hz.value_nnz),
        **payload,
    }


def _count_delta(after: Mapping[str, int], before: Mapping[str, int]) -> Dict[str, int]:
    keys = ("C", "B", "E", "U", "constraint_nnz")
    return {key: int(after[key] - before[key]) for key in keys}


def _as_bounds(lower: Sequence[float], upper: Sequence[float]) -> Bounds:
    lb = _strict_float64_vector(lower, name="lower")
    ub = _strict_float64_vector(upper, name="upper")
    if lb.size == 0 or lb.size != ub.size:
        raise ValueError("synthetic bounds must be nonempty and shape matched")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("synthetic bounds must be finite")
    if np.any(lb > ub):
        raise ValueError("synthetic lower bound exceeds upper bound")
    return Bounds(
        torch.from_numpy(lb.copy()).reshape(1, -1).double(),
        torch.from_numpy(ub.copy()).reshape(1, -1).double(),
    )


class ForwardExactReLUCensus:
    """Timed forward-only recorder around exact sparse-HZ toy operations."""

    def __init__(self, scenario: str):
        scenario = str(scenario).strip()
        if not scenario:
            raise ValueError("scenario must be nonempty")
        self.scenario = scenario
        self._records: List[Dict[str, Any]] = []

    @property
    def records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(self._records))

    def _record(
        self,
        *,
        label: str,
        operation: str,
        hz: SparseHZono,
        wall_ns: int,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        row: Dict[str, Any] = {
            "scenario": self.scenario,
            "layer_index": int(len(self._records)),
            "label": str(label),
            "operation": str(operation),
            "wall_ns": int(wall_ns),
            "wall_seconds": float(wall_ns / 1_000_000_000.0),
            **_state_counts(hz),
        }
        if details is not None:
            row["details"] = deepcopy(dict(details))
        self._records.append(row)

    def box(
        self,
        label: str,
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> SparseHZono:
        """Create and record a sparse exact-HZ input box."""

        bounds = _as_bounds(lower, upper)
        started = time.perf_counter_ns()
        hz = sparse_hz_from_bounds(bounds)
        elapsed = time.perf_counter_ns() - started
        self._record(
            label=label,
            operation="input_box",
            hz=hz,
            wall_ns=elapsed,
            details={"forward_only": True},
        )
        return hz

    def affine(
        self,
        label: str,
        hz: SparseHZono,
        weight: Any,
        bias: Optional[Sequence[float]] = None,
    ) -> SparseHZono:
        """Apply and record one exact sparse affine map."""

        _strict_hz(hz, name="hz")
        operator = _strict_affine_weight(weight, name="weight")
        bias_array = None
        if bias is not None:
            bias_array = _strict_float64_vector(bias, name="bias")
            if int(bias_array.size) != int(operator.shape[0]):
                raise ValueError("bias length must equal affine output width")
        started = time.perf_counter_ns()
        out = sparse_hz_linear(hz, operator, bias_array)
        elapsed = time.perf_counter_ns() - started
        operator_nnz = (
            int(operator.nnz)
            if sp.issparse(operator)
            else int(np.count_nonzero(operator))
        )
        self._record(
            label=label,
            operation="sparse_affine",
            hz=out,
            wall_ns=elapsed,
            details={"operator_nnz": operator_nnz, "forward_only": True},
        )
        return out

    def relu(
        self,
        label: str,
        hz: SparseHZono,
        pre_bounds: Optional[Bounds] = None,
    ) -> SparseHZono:
        """Apply compact exact ReLU and enforce its per-unstable cost law."""

        _strict_hz(hz, name="hz")
        if pre_bounds is not None:
            _strict_pre_bounds(pre_bounds, n_out=hz.n_out)
        before = _state_counts(hz)
        started = time.perf_counter_ns()
        out, phase_counts, info = sparse_hz_apply_relu_exact(
            hz,
            pre_bounds=pre_bounds,
            compressed=True,
            valid_cuts=False,
            return_info=True,
        )
        elapsed = time.perf_counter_ns() - started
        after = _state_counts(out)

        unstable_idx = np.asarray(info["unstable_idx"], dtype=np.int64)
        cont_support = np.diff(hz.Gc.indptr)[unstable_idx]
        if hz.n_bin:
            bin_support = np.diff(hz.Gb.indptr)[unstable_idx]
        else:
            bin_support = np.zeros(unstable_idx.size, dtype=np.int64)
        support_nnz = int(np.sum(cont_support + bin_support, dtype=np.int64))
        unstable = int(unstable_idx.size)
        expected = {
            "C": int(2 * unstable),
            "B": int(unstable),
            "E": int(unstable),
            "U": int(2 * unstable),
            "constraint_nnz": int(support_nnz + 7 * unstable),
        }
        actual = _count_delta(after, before)
        if actual != expected:
            raise AssertionError(
                "compact exact-ReLU structural invariant failed: "
                f"expected={expected}, actual={actual}"
            )

        details = {
            "exact_binary_relu": True,
            "compressed": True,
            "valid_cuts": False,
            "active": int(phase_counts[0]),
            "inactive": int(phase_counts[1]),
            "unstable": unstable,
            "preactivation_support_nnz": support_nnz,
            "per_unstable_cell": {"C": 2, "B": 1, "E": 1, "U": 2},
            "constraint_nnz_law": "sum(p_i + 7)",
            "expected_delta": expected,
            "actual_delta": actual,
            "invariant_passed": True,
        }
        self._record(
            label=label,
            operation="compact_exact_relu",
            hz=out,
            wall_ns=elapsed,
            details=details,
        )
        return out

    def add(
        self,
        label: str,
        left: SparseHZono,
        right: SparseHZono,
        *,
        residual_kind: str,
    ) -> SparseHZono:
        """Apply and record an exact stable-factor-aligned residual add."""

        _strict_hz(left, name="left")
        _strict_hz(right, name="right")
        started = time.perf_counter_ns()
        out = sparse_hz_add_same_frame(left, right)
        elapsed = time.perf_counter_ns() - started
        self._record(
            label=label,
            operation="exact_residual_add",
            hz=out,
            wall_ns=elapsed,
            details={"residual_kind": str(residual_kind), "forward_only": True},
        )
        return out

    def receipt(self) -> Dict[str, Any]:
        """Return a JSON-safe, explicitly non-authoritative receipt."""

        records = deepcopy(self._records)
        return {
            "schema": SCHEMA,
            "authoritative": False,
            "production_integration": False,
            "scope": "synthetic_only",
            "scenario": self.scenario,
            "execution_contract": {
                "direction": "forward_only",
                "relu_semantics": "compact_exact_binary_hz",
                "compressed": True,
                "valid_cuts": False,
                "solver_called": False,
                "real_dataset_loaded": False,
                "prohibited_mechanisms_called": {
                    "triangle_relaxation": False,
                    "branch_and_bound": False,
                    "backward_propagation": False,
                    "dual_tightening": False,
                },
            },
            "records": records,
            "summary": _summarize_records(records),
        }


def _summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    return {
        "layer_records": int(len(rows)),
        "total_wall_ns": int(sum(int(row["wall_ns"]) for row in rows)),
        "total_wall_seconds": float(sum(float(row["wall_seconds"]) for row in rows)),
        "peak_payload_bytes": int(max((int(row["payload_bytes"]) for row in rows), default=0)),
        "peak_constraint_nnz": int(max((int(row["constraint_nnz"]) for row in rows), default=0)),
        "peak_value_nnz": int(max((int(row["value_nnz"]) for row in rows), default=0)),
    }


def _ring_operator(width: int, sign: float = 1.0) -> sp.csr_matrix:
    rows = np.repeat(np.arange(width, dtype=np.int32), 2)
    cols = np.column_stack(
        [np.arange(width, dtype=np.int32), (np.arange(width, dtype=np.int32) + 1) % width]
    ).reshape(-1)
    data = np.tile(np.asarray([sign, -0.5 * sign], dtype=np.float64), width)
    operator = sp.coo_matrix((data, (rows, cols)), shape=(width, width)).tocsr()
    operator.eliminate_zeros()
    return operator


def run_synthetic_exact_relu_census(width: int = 64) -> Dict[str, Any]:
    """Run stable, affine, identity-residual, and fork/fork toy censuses.

    ``width`` is intentionally capped so this helper cannot silently turn
    into a real CIFAR/TinyImageNet or large-model run.
    """

    if type(width) is not int:
        raise ValueError("synthetic width must be an integer")
    if width < 2 or width > MAX_SYNTHETIC_WIDTH:
        raise ValueError(
            f"synthetic width must be in [2, {MAX_SYNTHETIC_WIDTH}]"
        )

    traces: List[ForwardExactReLUCensus] = []

    mixed = ForwardExactReLUCensus("mixed_stable_unstable")
    mixed_hz = mixed.box("mixed_input", [0.25, -2.0, -1.0], [1.0, -0.25, 1.0])
    mixed.relu(
        "mixed_relu",
        mixed_hz,
        _as_bounds([0.25, -2.0, -1.0], [1.0, -0.25, 1.0]),
    )
    traces.append(mixed)

    affine = ForwardExactReLUCensus("sparse_affine_then_relu")
    affine_hz = affine.box("affine_input", -np.ones(width), np.ones(width))
    affine_hz = affine.affine("ring_affine", affine_hz, _ring_operator(width))
    affine.relu("ring_relu", affine_hz)
    traces.append(affine)

    identity = ForwardExactReLUCensus("identity_residual")
    identity_root = identity.box("identity_input", -np.ones(width), np.ones(width))
    identity_branch = identity.relu("identity_relu_branch", identity_root)
    identity.add(
        "identity_join",
        identity_root,
        identity_branch,
        residual_kind="identity",
    )
    traces.append(identity)

    fork = ForwardExactReLUCensus("fork_fork_residual")
    fork_root = fork.box("fork_input", -np.ones(width), np.ones(width))
    common = fork.relu("fork_common_relu", fork_root)
    eye = sp.eye(width, format="csr", dtype=np.float64)
    left_pre = fork.affine("fork_left_affine", common, eye, -0.5 * np.ones(width))
    left = fork.relu("fork_left_relu", left_pre)
    right_pre = fork.affine("fork_right_affine", common, -eye, 0.5 * np.ones(width))
    right = fork.relu("fork_right_relu", right_pre)
    fork.add("fork_join", left, right, residual_kind="fork_fork")
    traces.append(fork)

    records = [dict(row) for trace in traces for row in trace.records]
    return {
        "schema": SCHEMA,
        "authoritative": False,
        "production_integration": False,
        "scope": "synthetic_only",
        "synthetic_width": width,
        "execution_contract": {
            "direction": "forward_only",
            "relu_semantics": "compact_exact_binary_hz",
            "compressed": True,
            "valid_cuts": False,
            "solver_called": False,
            "real_dataset_loaded": False,
            "prohibited_mechanisms_called": {
                "triangle_relaxation": False,
                "branch_and_bound": False,
                "backward_propagation": False,
                "dual_tightening": False,
            },
        },
        "scenarios": [trace.scenario for trace in traces],
        "records": records,
        "summary": _summarize_records(records),
    }


__all__ = [
    "ForwardExactReLUCensus",
    "MAX_SYNTHETIC_WIDTH",
    "PAYLOAD_TRAVERSAL_CONTRACT",
    "PAYLOAD_TRAVERSAL_MAX_DEPTH",
    "PAYLOAD_TRAVERSAL_MAX_NODES",
    "SCHEMA",
    "run_synthetic_exact_relu_census",
    "sparse_hz_payload_breakdown",
]
