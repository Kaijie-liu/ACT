"""Operator-backed HybridZ construction for large residual classifiers.

This module builds one :class:`~act.back_end.solver.solver_hz.SparseHZono`
without propagating a global generator matrix through every convolution.
Instead, it keeps affine operators as short-lived expressions and introduces
local, normalized variables at the input, every ReLU, every residual ADD, and
when two affine operators would otherwise be composed.

The represented variables use the standard ACT convention::

    value = center + generators @ xi_c,       xi_c in [-1, 1]
    Ac @ xi_c + Ab @ xi_b == b,               xi_b in {-1, 1}
    Auc @ xi_c + Aub @ xi_b <= ub

For an unstable ReLU with a valid pre-activation interval ``l < 0 < u``, the
continuous relaxation uses::

    y >= x
    y <= slope * x + intercept
    0 <= y <= u

``slope`` is the stored binary64 quotient and ``intercept`` is chosen by an
exact dyadic check of the ``l,0,u`` endpoint requirements, then rounded toward
``+inf``.  Thus the familiar triangle edge is retained without assuming that
rounded division/multiplication cancel.  The exact encoding adds a phase
``z = (xi_b + 1) / 2`` and uses::

    y >= x
    y <= x - l * (1 - z)
    y <= u * z
    0 <= y <= u

All affine equalities and ReLU inequalities are kept sparse and local.  The
only generator matrix exposed at the output is the final affine expression.
Unsupported topology or operators raise :class:`OperatorHZBuildError`; there
is deliberately no interval-box fallback in this strict builder.

Numerical scope
---------------
The builder reasons in real arithmetic over the floating-point parameters
stored in the ACT layers.  Every rounded affine/ADD expression carries an
explicit row-wise semantic allowance.  It normally represents numerical
roundoff; a guarded fusion may also collapse a complete generator row into an
independent interval of the same semantics.  Local relations consume this
allowance as outward bands and the final logits materialize it as independent
generators.  Cube bounds used for local normalization are computed
independently from the supplied interval facts.  The interval facts are used
for shape checks and audit metadata, not as unproved big-M constants.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import heapq
import hmac
import json
import math
import os
import resource
import secrets
import threading
import time
from fractions import Fraction
from itertools import product
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, Fact, Net
from act.back_end.hybridz_tf.tf_cnn import (
    sparse_conv2d_matrix_from_layer as _legacy_sparse_conv2d_matrix_from_layer,
)
from act.back_end.hybridz_tf.tf_mlp import sparse_dense_matrix_from_layer
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _highs_candidate_csr,
    _highs_process_threads,
    _hz_attach_exact_phase_conditional_property_rows_from_operator,
    _hz_independent_lp_lagrangian_upper,
    hz_constructively_nonempty,
    hz_fresh_col_ids,
    hz_mark_constructively_nonempty,
)

try:
    import highspy as _highspy
except Exception:  # pragma: no cover - optional solver dependency
    _highspy = None


_SUPPORTED_KINDS = frozenset(
    {
        "INPUT",
        "INPUT_SPEC",
        "CONV2D",
        "DENSE",
        "ADD",
        "RELU",
        "FLATTEN",
        "ASSERT",
    }
)
_AFFINE_KINDS = frozenset({"CONV2D", "DENSE"})

# Bound transient vectorized CONV2D index work.  This is a construction-only
# memory stop-loss: it changes neither the emitted coefficient order nor the
# exact stored binary64 coefficients.
_OPERATOR_CONV_TRIPLET_CHUNK = 1_000_000

# The cached direct constructor remains an isolated experiment until a
# production-authoritative core, implementation-integrity binding, consumer
# schema migration, and a complete Operator-stage >=1.50x gate all pass.
# Keeping the default false prevents a disconnected candidate from entering a
# proof-bearing build merely because its operator coefficients are bit exact.
_EXPERIMENTAL_CACHED_DIRECT_CONV = False

# Phase-B source-program integration remains an internal, default-off
# experiment.  In particular, keep the representation core out of this
# module's import graph unless a focused caller deliberately enables the
# sink.  Config/verifier/solver promotion is a separate phase.
_EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK = False


class _VectorizedOperatorConvUnsupported(Exception):
    """Request the established scalar constructor for an unsupported shape."""


def _operator_conv_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        return int(value), int(value)
    try:
        if len(value) != 2:
            raise _VectorizedOperatorConvUnsupported
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise _VectorizedOperatorConvUnsupported from exc


def _operator_conv_shape4(value: Any) -> Tuple[int, int, int, int]:
    try:
        dims = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise _VectorizedOperatorConvUnsupported from exc
    if len(dims) == 4:
        return dims
    if len(dims) == 3:
        return (1, *dims)
    raise _VectorizedOperatorConvUnsupported


def _operator_conv_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _vectorized_sparse_conv2d_matrix_from_layer(
    layer: Any,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW CONV2D CSR without a Python loop per nonzero.

    The legacy constructor emits one COO triplet at a time inside the complete
    output/channel/kernel nest.  Here each fixed kernel offset is expanded by
    NumPy in bounded chunks.  Every data entry is copied directly from the
    stored binary64 weight; there is no coefficient arithmetic or reduction.
    COO canonicalization therefore produces the identical sparse linear map.

    Only conventional, positive NCHW geometry whose complete COO fits signed
    int32 is accepted.  The wrapper below delegates all other requests to the
    established constructor.  Exceptions after this explicit admission check
    are deliberately *not* caught, so a vectorized bug cannot silently select
    a different implementation.
    """

    try:
        weight = _operator_conv_numpy(layer.params["weight"]).astype(
            np.float64, copy=False
        )
        if weight.ndim != 4:
            raise _VectorizedOperatorConvUnsupported
        out_ch, in_ch_per_group, kh, kw = (
            int(value) for value in weight.shape
        )
        groups = int(layer.params.get("groups", 1))
        stride = _operator_conv_pair(layer.params.get("stride", 1))
        padding = _operator_conv_pair(layer.params.get("padding", 0))
        dilation = _operator_conv_pair(layer.params.get("dilation", 1))
        bsz, in_ch, in_h, in_w = _operator_conv_shape4(
            layer.params["input_shape"]
        )
        out_bsz, out_ch_shape, out_h, out_w = _operator_conv_shape4(
            layer.params["output_shape"]
        )
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise _VectorizedOperatorConvUnsupported from exc

    if (
        min(
            bsz,
            in_ch,
            in_h,
            in_w,
            out_bsz,
            out_ch_shape,
            out_h,
            out_w,
            out_ch,
            in_ch_per_group,
            kh,
            kw,
            groups,
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
        )
        <= 0
        or min(padding) < 0
        or bsz != out_bsz
        or out_ch != out_ch_shape
        or out_ch % groups != 0
        or in_ch_per_group * groups != in_ch
    ):
        raise _VectorizedOperatorConvUnsupported

    n_rows = bsz * out_ch * out_h * out_w
    n_cols = bsz * in_ch * in_h * in_w
    int32_max = int(np.iinfo(np.int32).max)
    if max(n_rows, n_cols) > int32_max:
        raise _VectorizedOperatorConvUnsupported

    bias = layer.params.get("bias")
    if bias is None:
        bias_array = None
    else:
        try:
            bias_array = _operator_conv_numpy(bias).astype(
                np.float64, copy=False
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _VectorizedOperatorConvUnsupported from exc
        if bias_array.size != out_ch:
            raise _VectorizedOperatorConvUnsupported

    output_area = out_h * out_w
    input_area = in_h * in_w
    out_ch_per_group = out_ch // groups
    offset_frames: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    total_nnz = 0
    output_h = np.arange(out_h, dtype=np.int64)
    output_w = np.arange(out_w, dtype=np.int64)
    for rr in range(kh):
        input_h = output_h * stride[0] - padding[0] + rr * dilation[0]
        valid_h = (input_h >= 0) & (input_h < in_h)
        if not np.any(valid_h):
            continue
        oh = output_h[valid_h]
        ih = input_h[valid_h]
        for cc in range(kw):
            input_w_values = (
                output_w * stride[1] - padding[1] + cc * dilation[1]
            )
            valid_w = (input_w_values >= 0) & (input_w_values < in_w)
            if not np.any(valid_w):
                continue
            ow = output_w[valid_w]
            iw = input_w_values[valid_w]
            output_spatial = (
                oh[:, None] * out_w + ow[None, :]
            ).reshape(-1)
            input_spatial = (
                ih[:, None] * in_w + iw[None, :]
            ).reshape(-1)
            kernel_plane = weight[:, :, rr, cc]
            co, ci_local = np.nonzero(kernel_plane != 0.0)
            if co.size == 0:
                continue
            values = np.asarray(
                kernel_plane[co, ci_local], dtype=np.float64
            )
            ci = (
                (co // out_ch_per_group) * in_ch_per_group + ci_local
            ).astype(np.int64, copy=False)
            plane_nnz = int(bsz * co.size * output_spatial.size)
            total_nnz += plane_nnz
            if total_nnz > int32_max:
                raise _VectorizedOperatorConvUnsupported
            offset_frames.append(
                (
                    co.astype(np.int64, copy=False),
                    ci,
                    values,
                    output_spatial,
                    input_spatial,
                )
            )

    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float64)
    cursor = 0
    for co, ci, values, output_spatial, input_spatial in offset_frames:
        spatial_count = int(output_spatial.size)
        weight_chunk = max(
            1, _OPERATOR_CONV_TRIPLET_CHUNK // max(1, spatial_count)
        )
        for n in range(bsz):
            for start in range(0, int(co.size), weight_chunk):
                stop = min(int(co.size), start + weight_chunk)
                expanded = (stop - start) * spatial_count
                destination = slice(cursor, cursor + expanded)
                rows[destination] = (
                    (n * out_ch + co[start:stop, None]) * output_area
                    + output_spatial[None, :]
                ).reshape(-1)
                cols[destination] = (
                    (n * in_ch + ci[start:stop, None]) * input_area
                    + input_spatial[None, :]
                ).reshape(-1)
                data[destination] = np.repeat(
                    values[start:stop], spatial_count
                )
                cursor += expanded
    if cursor != total_nnz:
        raise OperatorHZBuildError(
            "vectorized CONV2D triplet count changed during construction"
        )

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_rows, n_cols),
        dtype=np.float64,
    )
    matrix.eliminate_zeros()
    if bias_array is None:
        bias_vector = np.zeros(n_rows, dtype=np.float64)
    else:
        bias_vector = np.tile(
            np.repeat(bias_array, output_area), bsz
        )
    return matrix, bias_vector


def _vectorized_sparse_conv2d_matrix_from_layer_with_legacy_fallback(
    layer: Any,
) -> Tuple[sp.csr_matrix, np.ndarray, str]:
    """Established production constructor with an explicit legacy fallback.

    This remains the default strict Operator-HZ path while the cached-direct
    experiment is disabled.  It also serves as the experiment's independent
    oracle.  Only the vectorized implementation's explicit admission exception
    selects the legacy implementation; internal defects propagate.
    """

    try:
        matrix, bias = _vectorized_sparse_conv2d_matrix_from_layer(layer)
        return matrix, bias, "vectorized_exact_csr_v1"
    except _VectorizedOperatorConvUnsupported:
        matrix, bias = _legacy_sparse_conv2d_matrix_from_layer(layer)
        return matrix, bias, "legacy_explicit_unsupported_fallback_v1"


def _sparse_conv2d_matrix_from_layer_strict(
    layer: Any,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    matrix, bias, _mode = _sparse_conv2d_matrix_from_layer_strict_with_mode(
        layer
    )
    return matrix, bias


def _sparse_conv2d_matrix_from_layer_strict_with_mode(
    layer: Any,
) -> Tuple[sp.csr_matrix, np.ndarray, str]:
    """Build an admitted CONV, optionally exercising the direct candidate.

    Production defaults to the established vectorized/explicit-fallback path.
    Tests may opt into the disconnected cached-direct constructor, whose
    rejection is then fail-closed rather than a reason to silently fall back.
    """

    if not _EXPERIMENTAL_CACHED_DIRECT_CONV:
        return _vectorized_sparse_conv2d_matrix_from_layer_with_legacy_fallback(
            layer
        )

    # Lazy import is intentional.  The default proof-bearing path neither
    # imports nor executes the disconnected candidate dependency.
    from act.back_end.hybridz_tf import (
        exact_sparse_conv_csr_candidate as _exact_conv_csr,
    )

    try:
        matrix, bias = (
            _exact_conv_csr.exact_sparse_conv2d_matrix_from_layer_candidate(
                layer
            )
        )
    except _exact_conv_csr.ExactSparseConvCandidateError as exc:
        layer_id = getattr(layer, "id", "<unknown>")
        raise OperatorHZBuildError(
            f"cached direct CONV2D CSR rejected layer {layer_id}: {exc}"
        ) from exc
    return matrix, bias, "cached_direct_exact_csr_v2"

# Candidate-only ADD -> affine -> ReLU fusion controls.  The public
# ``materialize_add=True`` default never enters this path.  A small absolute
# budget and row chunks bound both the transient sparse product and the blast
# radius of an unsuccessful experiment.
_LIVE_AFFINE_CHUNK_ROWS = 256
_LIVE_AFFINE_TOTAL_SECONDS = 8.0
_LIVE_AFFINE_MAX_STORED_NNZ = 20_000_000
_PROPERTY_MICRO_RLT_PRODUCT_FACTOR_CAP_MAX = 4096
_PROPERTY_MICRO_RLT_SELECTED_ROW_NNZ_CAP = 16384
_PROPERTY_MICRO_RLT_REQUIREMENT_SCAN_NNZ_CAP = 65536


class OperatorHZBuildError(ValueError):
    """Strict operator-HZ construction failure.

    Callers must map this exception to ``UNKNOWN``.  Falling back to a box and
    using that box as a HybridZ proof object would erase graph constraints.
    """


class OperatorHZBuildTimeout(TimeoutError):
    """The shared HybridZ wall deadline expired during construction."""


_PREACTIVATION_FRAME_SCHEMA = (
    "operator_hz_verified_preactivation_frame_v1"
)
_PREACTIVATION_FRAME_LOCK = threading.Lock()
_PREACTIVATION_FRAME_AUTHORITIES: Dict[
    int,
    Tuple[
        "weakref.ReferenceType[OperatorHZPreactivationFrame]",
        str,
        str,
    ],
] = {}

_CONSTRUCTIVE_NONEMPTY_SEAL_PRODUCER = object()
_CONSTRUCTIVE_NONEMPTY_SEAL_LOCK = threading.Lock()
_CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS: Dict[
    str, "_OperatorHZConstructiveNonemptySealRecord"
] = {}


class OperatorHZConstructiveNonemptySeal:
    """Opaque process-local authority for one completed Operator-HZ build."""

    __slots__ = (
        "_token",
        "_semantic_digest",
        "_process_id",
        "_reason",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        token: str,
        semantic_digest: str,
        process_id: int,
        reason: str,
        _producer_capability: Any,
    ) -> None:
        if (
            _producer_capability
            is not _CONSTRUCTIVE_NONEMPTY_SEAL_PRODUCER
        ):
            raise PermissionError(
                "constructive-nonempty seal requires its producer"
            )
        object.__setattr__(self, "_token", token)
        object.__setattr__(
            self, "_semantic_digest", semantic_digest
        )
        object.__setattr__(self, "_process_id", process_id)
        object.__setattr__(self, "_reason", reason)

    @property
    def token(self) -> str:
        return self._token

    @property
    def semantic_digest(self) -> str:
        return self._semantic_digest

    @property
    def process_id(self) -> int:
        return self._process_id

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def proof_authority(self) -> bool:
        return True

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("constructive-nonempty seals are immutable")

    def __copy__(self):
        copied = object.__new__(type(self))
        for name in (
            "_token",
            "_semantic_digest",
            "_process_id",
            "_reason",
        ):
            object.__setattr__(
                copied, name, object.__getattribute__(self, name)
            )
        return copied

    def __deepcopy__(self, _memo):
        return self.__copy__()


@dataclass(frozen=True)
class _OperatorHZConstructiveNonemptySealRecord:
    seal_ref: "weakref.ReferenceType[OperatorHZConstructiveNonemptySeal]"
    build_ref: "weakref.ReferenceType[OperatorHZBuild]"
    hz_ref: "weakref.ReferenceType[SparseHZono]"
    core_refs: Tuple["weakref.ReferenceType[Any]", ...]
    core_identity: Tuple[int, ...]
    semantic_digest: str
    process_id: int
    reason: str


@dataclass(frozen=True)
class OperatorHZPreactivationFrame:
    """Process-local sound ReLU bounds exported by one Operator-HZ build.

    Every pair follows query-dual's ReLU-key convention and encloses the
    *preactivation*.  A receipt is only an audit record; consumers must
    validate the exact live object through
    :func:`validate_operator_hz_preactivation_frame`.
    """

    bounds: Mapping[int, Tuple[np.ndarray, np.ndarray]]
    receipt: Mapping[str, Any]
    provenance_nonce: str
    proof_authority: bool = True

    def __post_init__(self) -> None:
        if self.proof_authority is not True:
            raise ValueError(
                "operator-HZ preactivation frame must be authoritative"
            )
        frozen: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for raw_lid, raw_pair in self.bounds.items():
            lid = int(raw_lid)
            if lid in frozen or len(raw_pair) != 2:
                raise ValueError("malformed preactivation frame")
            lower = np.ascontiguousarray(
                np.asarray(raw_pair[0], dtype=np.float64).reshape(-1)
            ).copy()
            upper = np.ascontiguousarray(
                np.asarray(raw_pair[1], dtype=np.float64).reshape(-1)
            ).copy()
            if (
                lower.shape != upper.shape
                or lower.size == 0
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > upper)
            ):
                raise ValueError(
                    f"invalid preactivation frame bounds at ReLU {lid}"
                )
            lower.setflags(write=False)
            upper.setflags(write=False)
            frozen[lid] = (lower, upper)
        if not frozen:
            raise ValueError("preactivation frame must not be empty")
        object.__setattr__(
            self,
            "bounds",
            MappingProxyType(dict(sorted(frozen.items()))),
        )
        object.__setattr__(
            self,
            "receipt",
            MappingProxyType(copy.deepcopy(dict(self.receipt))),
        )


class _PreactivationLPDeadline(TimeoutError):
    """The optional local tightening budget expired (never a proof result)."""


@dataclass(frozen=True)
class OperatorHZBuild:
    """Result of :func:`build_operator_hz`.

    ``input_col_ids`` has one stable id per flattened input coordinate,
    including point coordinates that do not allocate a continuous column.
    This is the provenance map required by the independent witness replay gate.
    """

    hz: SparseHZono
    input_col_ids: np.ndarray
    input_layer_id: int
    output_layer_id: int
    assert_layer_id: int
    metadata: Dict[str, Any]
    property_upper_output: bool = False
    property_upper_row_groups: Tuple[Tuple[int, ...], ...] = ()
    verified_preactivation_frame: Optional[
        OperatorHZPreactivationFrame
    ] = None
    constructive_nonempty_seal: Optional[
        OperatorHZConstructiveNonemptySeal
    ] = None
    # Candidate-only runtime telemetry.  This deliberately lives outside
    # ``metadata`` (and therefore outside ``hz.operator_hz_metadata``): wall
    # clocks, process counters and page-fault counts are non-semantic and must
    # never perturb proof/bit-compatibility receipts.
    performance_diagnostic: Optional[Mapping[str, Any]] = None
    # Optional Phase-B representation/replay authority.  It deliberately
    # lives outside ``metadata`` so the default path and the legacy semantic
    # receipt remain byte-for-byte unchanged.  The object is not solver or
    # verdict authority; Phase C owns native consumption.
    constraint_program: Optional[Any] = None


def _constructive_nonempty_core_objects(
    hz: Any,
) -> Optional[Tuple[Any, ...]]:
    """Capture the exact core objects whose identities a seal owns."""

    if type(hz) is not SparseHZono:
        return None
    live = vars(hz)
    objects = [hz]
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        value = live.get(name)
        if type(value) is not np.ndarray:
            return None
        objects.append(value)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = live.get(name)
        if type(matrix) is not sp.csr_matrix:
            return None
        matrix_vars = vars(matrix)
        data = matrix_vars.get("data")
        indices = matrix_vars.get("indices")
        indptr = matrix_vars.get("indptr")
        if (
            type(data) is not np.ndarray
            or type(indices) is not np.ndarray
            or type(indptr) is not np.ndarray
        ):
            return None
        objects.extend((matrix, data, indices, indptr))
    return tuple(objects)


def _sweep_constructive_nonempty_seals_locked() -> None:
    process_id = os.getpid()
    stale = tuple(
        token
        for token, record in (
            _CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS.items()
        )
        if (
            record.process_id != process_id
            or record.seal_ref() is None
            or record.build_ref() is None
            or record.hz_ref() is None
            or any(ref() is None for ref in record.core_refs)
        )
    )
    for token in stale:
        _CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS.pop(
            token, None
        )


def _make_operator_hz_constructive_nonempty_seal(
    *,
    semantic_digest: str,
    reason: str,
) -> OperatorHZConstructiveNonemptySeal:
    if (
        type(semantic_digest) is not str
        or len(semantic_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in semantic_digest
        )
        or type(reason) is not str
        or not reason
        or len(reason.encode("utf-8")) > 4096
    ):
        raise OperatorHZBuildError(
            "constructive-nonempty seal payload is malformed"
        )
    return OperatorHZConstructiveNonemptySeal(
        token=secrets.token_hex(32),
        semantic_digest=semantic_digest,
        process_id=os.getpid(),
        reason=reason,
        _producer_capability=(
            _CONSTRUCTIVE_NONEMPTY_SEAL_PRODUCER
        ),
    )


def _register_operator_hz_constructive_nonempty_seal(
    seal: OperatorHZConstructiveNonemptySeal,
    build: OperatorHZBuild,
) -> None:
    if (
        type(seal) is not OperatorHZConstructiveNonemptySeal
        or type(build) is not OperatorHZBuild
        or build.constructive_nonempty_seal is not seal
        or type(build.hz) is not SparseHZono
    ):
        raise OperatorHZBuildError(
            "constructive-nonempty seal owner is malformed"
        )
    core_objects = _constructive_nonempty_core_objects(
        build.hz
    )
    if core_objects is None:
        raise OperatorHZBuildError(
            "constructive-nonempty seal core is malformed"
        )
    try:
        core_refs = tuple(
            weakref.ref(value) for value in core_objects
        )
        seal_ref = weakref.ref(seal)
        build_ref = weakref.ref(build)
        hz_ref = weakref.ref(build.hz)
    except TypeError as exc:
        raise OperatorHZBuildError(
            "constructive-nonempty seal owner is not weak-referenceable"
        ) from exc
    record = _OperatorHZConstructiveNonemptySealRecord(
        seal_ref=seal_ref,
        build_ref=build_ref,
        hz_ref=hz_ref,
        core_refs=core_refs,
        core_identity=tuple(id(value) for value in core_objects),
        semantic_digest=seal.semantic_digest,
        process_id=seal.process_id,
        reason=seal.reason,
    )
    with _CONSTRUCTIVE_NONEMPTY_SEAL_LOCK:
        _sweep_constructive_nonempty_seals_locked()
        if seal.token in _CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS:
            raise OperatorHZBuildError(
                "constructive-nonempty seal token collision"
            )
        _CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS[
            seal.token
        ] = record


def validate_operator_hz_constructive_nonempty_seal(
    seal: Any,
    *,
    owner_build: Any,
    owner_hz: Any,
    owner_core_identity: Any,
    private_parent_semantic_digest: Any,
) -> bool:
    """Validate one producer seal without reading mutable live metadata."""

    if (
        type(seal) is not OperatorHZConstructiveNonemptySeal
        or type(owner_build) is not OperatorHZBuild
        or type(owner_hz) is not SparseHZono
        or type(owner_core_identity) is not tuple
        or not owner_core_identity
        or any(type(value) is not int for value in owner_core_identity)
        or type(private_parent_semantic_digest) is not str
        or len(private_parent_semantic_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in private_parent_semantic_digest
        )
        or seal.proof_authority is not True
        or type(seal.token) is not str
        or len(seal.token) != 64
        or any(
            character not in "0123456789abcdef"
            for character in seal.token
        )
        or type(seal.process_id) is not int
        or type(seal.reason) is not str
        or not seal.reason
        or len(seal.reason.encode("utf-8")) > 4096
        or type(seal.semantic_digest) is not str
        or len(seal.semantic_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in seal.semantic_digest
        )
    ):
        return False
    process_id = os.getpid()
    with _CONSTRUCTIVE_NONEMPTY_SEAL_LOCK:
        _sweep_constructive_nonempty_seals_locked()
        record = _CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS.get(
            seal.token
        )
        if record is None:
            return False
        live_core = tuple(ref() for ref in record.core_refs)
        return bool(
            record.process_id == process_id
            and seal.process_id == process_id
            and record.seal_ref() is seal
            and record.build_ref() is owner_build
            and record.hz_ref() is owner_hz
            and all(value is not None for value in live_core)
            and record.core_identity == owner_core_identity
            and tuple(id(value) for value in live_core)
            == owner_core_identity
            and hmac.compare_digest(
                record.semantic_digest,
                private_parent_semantic_digest,
            )
            and hmac.compare_digest(
                seal.semantic_digest,
                record.semantic_digest,
            )
            and seal.reason == record.reason
        )


@dataclass(frozen=True)
class _AffineExpr:
    """A flattened affine expression over the current global continuous frame.

    The semantic value is enclosed by

    ``c + G @ xi + delta`` with ``|delta[row]| <= err[row]``.

    Keeping the semantic allowance separate avoids allocating one global
    factor per intermediate activation.  It contains numerical roundoff and
    may conservatively contain a deliberately box-collapsed generator row.
    Relations which consume an expression widen their right-hand side by
    ``err``; only the final logits materialize the remaining allowance as
    independent generators.
    """

    c: np.ndarray
    G: sp.csr_matrix
    err: np.ndarray
    affine_depth: int = 0

    @property
    def size(self) -> int:
        return int(self.c.size)


@dataclass(frozen=True)
class _ConstraintBlock:
    Ac: sp.csr_matrix
    Ab: sp.csr_matrix
    rhs: np.ndarray
    tag: str


class _OperatorConstraintFactorAllocatorBridge:
    """Append-only typed view of the established global HZ ID allocator.

    INPUT reserves one stable ID for every flattened coordinate, including
    point coordinates which never become factor columns.  The constraint
    program must see only actual factor columns, so that full reservation is
    held as one pending capability until the owner claims exactly the active
    subset.  Every reserved ID remains burned on failure.
    """

    __slots__ = (
        "_continuous_ids",
        "_binary_ids",
        "_pending_input_continuous_ids",
    )

    def __init__(self) -> None:
        self._continuous_ids: Tuple[int, ...] = ()
        self._binary_ids: Tuple[int, ...] = ()
        self._pending_input_continuous_ids: Optional[Tuple[int, ...]] = None

    @staticmethod
    def _reserve_raw_ids(count: int) -> Tuple[int, ...]:
        if type(count) is not int or count < 0:
            raise OperatorHZBuildError(
                "constraint-program factor count must be a nonnegative "
                "builtin int"
            )
        values = hz_fresh_col_ids(count, device="cpu")
        return tuple(int(value) for value in values.detach().cpu().tolist())

    def reserve_input(
        self,
        count: int,
        active_positions: np.ndarray,
    ) -> np.ndarray:
        if (
            self._continuous_ids
            or self._binary_ids
            or self._pending_input_continuous_ids is not None
        ):
            raise OperatorHZBuildError(
                "constraint-program INPUT reservation is not first or unique"
            )
        positions = np.asarray(active_positions, dtype=np.int64).reshape(-1)
        if positions.size and (
            int(positions[0]) < 0
            or int(positions[-1]) >= int(count)
            or np.any(positions[1:] <= positions[:-1])
        ):
            raise OperatorHZBuildError(
                "constraint-program INPUT active positions are malformed"
            )
        full = self._reserve_raw_ids(int(count))
        self._pending_input_continuous_ids = tuple(
            full[int(position)] for position in positions.tolist()
        )
        return np.asarray(full, dtype=np.int64)

    def allocate_continuous(self, count: int) -> Tuple[int, ...]:
        if type(count) is not int or count < 0:
            raise OperatorHZBuildError(
                "constraint-program continuous count is malformed"
            )
        if self._pending_input_continuous_ids is not None:
            values = self._pending_input_continuous_ids
            if len(values) != count:
                raise OperatorHZBuildError(
                    "constraint-program INPUT pending reservation count "
                    "changed"
                )
            self._pending_input_continuous_ids = None
        else:
            values = self._reserve_raw_ids(count)
        self._continuous_ids = self._continuous_ids + values
        return values

    def allocate_binary(self, count: int) -> Tuple[int, ...]:
        if self._pending_input_continuous_ids is not None:
            raise OperatorHZBuildError(
                "constraint-program INPUT reservation was not claimed before "
                "binary allocation"
            )
        values = self._reserve_raw_ids(count)
        self._binary_ids = self._binary_ids + values
        return values

    def snapshot(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self._continuous_ids, self._binary_ids


class _OperatorConstraintProgramSink:
    """One build-local Phase-B source-program transaction.

    The class contains no eager reference to ``constraint_program``.  Its
    :meth:`initialize` method is called only after the complete sink preflight
    has passed.
    """

    __slots__ = (
        "core",
        "bridge",
        "adapter",
        "owner",
        "arena",
        "view",
        "program",
        "phase",
        "virtual_rows",
        "source_rows",
        "virtual_nnz",
        "source_nnz",
        "legacy_cont_nnz",
        "legacy_tag_rows",
    )

    def __init__(self) -> None:
        self.core: Any = None
        self.bridge: Optional[_OperatorConstraintFactorAllocatorBridge] = None
        self.adapter: Any = None
        self.owner: Any = None
        self.arena: Any = None
        self.view: Any = None
        self.program: Any = None
        self.phase = "new"
        self.virtual_rows = 0
        self.source_rows = 0
        self.virtual_nnz = 0
        self.source_nnz = 0
        self.legacy_cont_nnz = 0
        self.legacy_tag_rows: List[Tuple[str, int]] = []

    def initialize(self) -> None:
        if self.phase != "new":
            raise OperatorHZBuildError(
                "constraint-program sink was initialized twice"
            )
        # Deliberately lazy: the default-false legacy path must not import the
        # production representation core.
        from act.back_end.solver import constraint_program as core

        self.core = core
        self.bridge = _OperatorConstraintFactorAllocatorBridge()
        self.adapter = core.ExternalFactorAllocatorAdapter.reserve()
        self.phase = "binding"
        self.adapter.initialize(
            self.bridge,
            allocate_continuous=self.bridge.allocate_continuous,
            allocate_binary=self.bridge.allocate_binary,
            live_ids_snapshot=self.bridge.snapshot,
        )
        self.phase = "bound"
        self.owner = core.ConstraintProgramOwner.reserve()
        # ``new_arena`` is recoverable for the same owner if an asynchronous
        # exception crosses either owner initialization or arena creation.
        # Publish this phase first so outer cleanup always retains the exact
        # reserved/initializing/complete owner handle.
        self.phase = "owner"
        self.owner.initialize(self.adapter)
        self.arena = self.owner.new_arena()
        # Once the arena exists, every later pre-seal failure is required to
        # terminally discard the owner/arena pair.
        self.phase = "open"
        self.view = self.arena.empty_view

    @staticmethod
    def _raw_ids(values: Sequence[Any]) -> Tuple[int, ...]:
        return tuple(int(value.raw_id) for value in values)

    def reserve_input(
        self,
        count: int,
        active_positions: np.ndarray,
    ) -> np.ndarray:
        if self.phase != "open" or self.bridge is None:
            raise OperatorHZBuildError(
                "constraint-program INPUT reservation requires an open sink"
            )
        full = self.bridge.reserve_input(count, active_positions)
        expected = tuple(
            int(full[int(position)])
            for position in np.asarray(
                active_positions, dtype=np.int64
            ).reshape(-1)
        )
        issued = self._raw_ids(
            self.owner.allocate_continuous(len(expected))
        )
        if issued != expected:
            raise OperatorHZBuildError(
                "constraint-program INPUT factor IDs changed while claimed"
            )
        return full

    def allocate_continuous(self, count: int) -> np.ndarray:
        if self.phase != "open":
            raise OperatorHZBuildError(
                "constraint-program continuous allocation requires an open "
                "sink"
            )
        return np.asarray(
            self._raw_ids(self.owner.allocate_continuous(int(count))),
            dtype=np.int64,
        )

    def allocate_binary(self, count: int) -> np.ndarray:
        if self.phase != "open":
            raise OperatorHZBuildError(
                "constraint-program binary allocation requires an open sink"
            )
        return np.asarray(
            self._raw_ids(self.owner.allocate_binary(int(count))),
            dtype=np.int64,
        )

    def append_le(self, block: _ConstraintBlock, *, layer_id: int) -> int:
        if self.phase != "open":
            raise OperatorHZBuildError(
                "constraint-program LE append requires an open sink"
            )
        rows = int(block.Ac.shape[0])
        nnz = int(block.Ac.nnz + block.Ab.nnz)
        result = self.arena.append_le_exact_tag(
            self.view,
            frame=self.owner.frame(),
            A_cont=block.Ac,
            A_bin=block.Ab,
            upper=block.rhs,
            tag=block.tag,
            layer_id=int(layer_id),
        )
        if (
            result.source_rows != rows
            or result.virtual_rows != rows
            or result.source_nnz != nnz
            or result.virtual_nnz != nnz
            or result.ranged_rows != 0
            or result.fallback_pairs != 0
        ):
            raise OperatorHZBuildError(
                "constraint-program LE accounting changed after commit"
            )
        self.view = result.view
        self.virtual_rows += rows
        self.source_rows += rows
        self.virtual_nnz += nnz
        self.source_nnz += nnz
        self.legacy_cont_nnz += int(block.Ac.nnz)
        self.legacy_tag_rows.append((str(block.tag), rows))
        return rows

    def append_add_materialize_range(
        self,
        forward: _ConstraintBlock,
        reverse: _ConstraintBlock,
        *,
        layer_id: int,
    ) -> int:
        if self.phase != "open":
            raise OperatorHZBuildError(
                "constraint-program RANGE append requires an open sink"
            )
        rows = int(forward.Ac.shape[0])
        if rows <= 0 or reverse.Ac.shape[0] != rows:
            raise OperatorHZBuildError(
                "ADD materialization RANGE sides have different row counts"
            )
        result = self.arena.append_guarded_band(
            self.view,
            frame=self.owner.frame(),
            forward_cont=forward.Ac,
            forward_bin=forward.Ab,
            forward_upper=forward.rhs,
            reverse_cont=reverse.Ac,
            reverse_bin=reverse.Ab,
            reverse_upper=reverse.rhs,
            layer_id=int(layer_id),
            family=self.core.ConstraintFamily.ADD_MATERIALIZE,
        )
        virtual_nnz = int(
            forward.Ac.nnz
            + forward.Ab.nnz
            + reverse.Ac.nnz
            + reverse.Ab.nnz
        )
        source_nnz = int(forward.Ac.nnz + forward.Ab.nnz)
        if (
            result.source_rows != rows
            or result.virtual_rows != 2 * rows
            or result.source_nnz != source_nnz
            or result.virtual_nnz != virtual_nnz
            or result.ranged_rows != rows
            or result.fallback_pairs != 0
        ):
            # The core supports an exact per-row fallback, but this first
            # Operator integration deliberately does not.  Whole-build
            # discard is the only admissible outcome.
            raise OperatorHZBuildError(
                "ADD materialization did not commit as an all-RANGE block"
            )
        self.view = result.view
        self.virtual_rows += 2 * rows
        self.source_rows += rows
        self.virtual_nnz += virtual_nnz
        self.source_nnz += source_nnz
        self.legacy_cont_nnz += int(
            forward.Ac.nnz + reverse.Ac.nnz
        )
        self.legacy_tag_rows.extend(
            ((str(forward.tag), rows), (str(reverse.tag), rows))
        )
        return 2 * rows

    def discard_open(self) -> Optional[BaseException]:
        """Boundedly converge an open pre-seal sink to terminal discard."""

        first_error: Optional[BaseException] = None
        if self.phase == "owner" and self.owner is not None:
            # Recover a complete owner/arena whose initialization or arena
            # creation may have completed before a public-return
            # interruption.  A terminally poisoned owner rejects every
            # attempt and is retained only long enough to annotate the
            # primary build exception; it cannot create program authority.
            for _attempt in range(4):
                try:
                    self.arena = self.owner.new_arena()
                    self.phase = "open"
                    break
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if self.phase != "open" or self.arena is None or self.owner is None:
            if self.phase == "owner":
                return first_error or OperatorHZBuildError(
                    "constraint-program arena recovery did not converge"
                )
            return None
        for _attempt in range(4):
            try:
                self.arena.discard()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            try:
                if self.arena.discarded and self.owner.discarded:
                    self.phase = "discarded"
                    return None
            except BaseException as error:
                if first_error is None:
                    first_error = error
        return first_error or OperatorHZBuildError(
            "constraint-program pre-seal discard did not reach terminal state"
        )

    def seal_and_replay(
        self,
        *,
        expected_continuous_ids: Sequence[int],
        expected_binary_ids: Sequence[int],
    ) -> Tuple[Any, sp.csr_matrix, sp.csr_matrix, np.ndarray, Tuple[str, ...]]:
        if self.phase != "open":
            raise OperatorHZBuildError(
                "constraint-program final seal requires an open sink"
            )
        expected_continuous = tuple(
            int(value) for value in expected_continuous_ids
        )
        expected_binary = tuple(int(value) for value in expected_binary_ids)
        final_frame = self.owner.frame()
        if (
            self._raw_ids(final_frame.continuous_ids)
            != expected_continuous
            or self._raw_ids(final_frame.binary_ids) != expected_binary
        ):
            raise OperatorHZBuildError(
                "constraint-program final factor frame differs from Operator "
                "stable IDs"
            )

        # This is the irrevocable publication boundary.  The frozen core
        # guarantees OLD-or-complete-NEW state; the exception path below
        # terminally discards OLD, while a sealed discard rejection selects
        # recovery of the already-published complete program.
        self.phase = "sealing"
        try:
            program = self.arena.seal(
                self.view, final_frame=final_frame
            )
        except BaseException:
            # First converge a still-OLD arena to terminal discard.  If seal
            # crossed complete-NEW, discard is guaranteed to reject without
            # revoking authority; the same authenticated view/frame call can
            # then recover that exact sealed program.
            for _attempt in range(4):
                try:
                    self.arena.discard()
                except BaseException:
                    pass
                try:
                    terminal_discard = bool(
                        self.arena.discarded and self.owner.discarded
                    )
                except BaseException:
                    terminal_discard = False
                if terminal_discard:
                    self.phase = "discarded"
                    raise
            recovered = None
            for _attempt in range(4):
                try:
                    recovered = self.arena.seal(
                        self.view, final_frame=final_frame
                    )
                    break
                except BaseException:
                    continue
            if recovered is None:
                # Do not call discard from the ambiguous publication state.
                # A persistent fault can hide a complete sealed program, and
                # sealed authority is irrevocable by contract.
                self.phase = "sealing"
            else:
                self.program = recovered
                self.phase = "sealed"
            raise
        self.program = program
        self.phase = "sealed"

        if (
            self._raw_ids(program.continuous_ids) != expected_continuous
            or self._raw_ids(program.binary_ids) != expected_binary
            or program.virtual_facet_rows != self.virtual_rows
            or program.source_rows != self.source_rows
            or program.virtual_facet_nnz != self.virtual_nnz
            or program.source_nnz != self.source_nnz
            or program.fallback_pairs != 0
        ):
            raise OperatorHZBuildError(
                "sealed constraint program disagrees with pre-seal accounting"
            )

        continuous_batches: List[sp.csr_matrix] = []
        binary_batches: List[sp.csr_matrix] = []
        upper_batches: List[np.ndarray] = []
        row_tags: List[str] = []
        offset = 0
        cursor = program.iter_legacy_facet_batches(max_rows=256)
        try:
            for batch in cursor:
                if (
                    batch.row_offset != offset
                    or not 0 < batch.row_count <= 256
                    or batch.total_rows != self.virtual_rows
                ):
                    raise OperatorHZBuildError(
                        "constraint-program legacy replay batch offsets "
                        "changed"
                    )
                continuous_batches.append(batch.A_cont)
                binary_batches.append(batch.A_bin)
                upper_batches.append(batch.upper)
                row_tags.extend(str(value) for value in batch.row_tags)
                offset += int(batch.row_count)
        except BaseException as replay_error:
            close_error = self._close_legacy_replay_cursor(cursor)
            if close_error is not None:
                try:
                    replay_error.add_note(
                        "constraint-program replay cursor cleanup also "
                        "failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                except BaseException:
                    pass
            raise
        close_error = self._close_legacy_replay_cursor(cursor)
        if close_error is not None:
            raise close_error
        if offset != self.virtual_rows:
            raise OperatorHZBuildError(
                "constraint-program legacy replay ended before all virtual "
                "facets"
            )
        expected_tags = tuple(
            tag
            for tag, rows in self.legacy_tag_rows
            for _row in range(int(rows))
        )
        if tuple(row_tags) != expected_tags:
            raise OperatorHZBuildError(
                "constraint-program legacy replay changed Operator row tags"
            )
        A_cont = _stack_padded(
            continuous_batches, width=len(expected_continuous)
        )
        A_bin = _stack_padded(
            binary_batches, width=len(expected_binary)
        )
        upper = (
            np.concatenate(upper_batches)
            if upper_batches
            else np.zeros(0, dtype=np.float64)
        )
        return program, A_cont, A_bin, upper, tuple(row_tags)

    @staticmethod
    def _close_legacy_replay_cursor(
        cursor: Any,
    ) -> Optional[BaseException]:
        """Boundedly close a replay cursor without hiding a primary error."""

        first_error: Optional[BaseException] = None
        for _attempt in range(4):
            try:
                cursor.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                continue
            return first_error
        return first_error or OperatorHZBuildError(
            "constraint-program replay cursor cleanup did not converge"
        )


@dataclass(frozen=True)
class _PreactivationLPBase:
    """Immutable snapshot of the original HZ frame used by a ReLU audit.

    Candidate LPs may relax binary factors to ``[-1, 1]``.  This only enlarges
    the feasible set.  The authoritative bound is subsequently reconstructed
    by the independent long-double Lagrangian checker, never by trusting the
    LP status or objective value.
    """

    A: sp.csr_matrix
    rl: np.ndarray
    ru: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    n_eq: int
    n_ub: int
    csr_sha256: str = ""


@dataclass
class _PersistentPreactivationHighs:
    """One candidate-only HiGHS model for one immutable constraint snapshot."""

    highs: Any
    base: _PreactivationLPBase
    all_cols: np.ndarray
    receipt: Dict[str, Any]
    basis: Optional[Any] = None
    solve_count: int = 0


@dataclass(frozen=True)
class _PropertyTailSnapshot:
    """Frame prefix immediately before a candidate final ReLU."""

    relu_layer_id: int
    preactivation: _AffineExpr
    lower: np.ndarray
    upper: np.ndarray
    n_cont: int
    n_bin: int
    eq_block_count: int
    ub_block_count: int
    exact_used: int


@dataclass(frozen=True)
class _LayerFrameSnapshot:
    """Exact builder frame immediately after one topological layer.

    A property suffix plane replayed to an earlier layer only depends on the
    state available at that layer (plus the final explicit roundoff factor).
    The solver may therefore *relax* its candidate LP by retaining only these
    exact stored constraint prefixes.  Dropping later constraints enlarges the
    feasible set, so an independently checked upper bound over that relaxation
    is also an upper bound over the full HybridZ state.
    """

    n_cont: int
    n_bin: int
    eq_rows: int
    ub_rows: int
    eq_block_count: int
    ub_block_count: int


@dataclass(frozen=True)
class _PropertySuffixAddSourceSnapshot:
    """Correlated ADD expression and exact frame before materialization."""

    add_layer_id: int
    expression: _AffineExpr
    n_cont: int
    n_bin: int
    eq_rows: int
    ub_rows: int
    eq_block_count: int
    ub_block_count: int


@dataclass(frozen=True)
class _MaterializedAddSourceSnapshot:
    """Immutable source of the materialized ADD feeding the final ReLU.

    The ordinary ADD output remains in the frame together with both equality
    bands.  This snapshot is used only to construct additional safe upper
    rows over the older, correlated source expression; it never authorizes
    pruning the materialized variables or their constraints.
    """

    add_layer_id: int
    expression: _AffineExpr
    n_cont_before: int
    n_cont_after: int
    n_bin: int
    eq_block_count_before: int
    eq_block_count_after: int
    ub_block_count_before: int
    ub_block_count_after: int
    relation_block_rows: Tuple[int, ...]
    relation_block_tags: Tuple[str, ...]
    relation_blocks_sha256: str
    new_cont: int
    new_ub: int


def _csr_sha256(matrix: sp.csr_matrix) -> str:
    """Hash the exact CSR structure and stored binary64 values."""

    csr = sp.csr_matrix(matrix, dtype=np.float64, copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _f64_array_sha256(value: Any) -> str:
    """Hash one binary64 array including its exact shape and stored bits."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def operator_hz_property_suffix_dominating_add_candidates(
    net: Net,
    *,
    output_layer_id: int,
) -> Tuple[int, ...]:
    """Return dominating suffix ADDs in nearest-to-farthest order.

    This public, read-only topology helper is also used by the C22
    property-conditioned phase selector.  Restricting candidate exact ReLUs
    to layers before the selected stop ADD ensures their fixed-phase rows are
    present in the C17 constraint-prefix relaxation.
    """

    layers = list(net.layers)
    layer_by_id = {int(layer.id): layer for layer in layers}
    if len(layer_by_id) != len(layers):
        raise OperatorHZBuildError("operator-HZ topology has duplicate layer ids")
    output = int(output_layer_id)
    if output not in layer_by_id:
        raise OperatorHZBuildError(
            f"property suffix output layer {output} is missing"
        )
    position = {
        int(layer.id): index for index, layer in enumerate(layers)
    }
    ancestors: set[int] = set()
    stack = [output]
    while stack:
        lid = int(stack.pop())
        if lid in ancestors:
            continue
        if lid not in layer_by_id:
            raise OperatorHZBuildError(
                f"property suffix ancestor {lid} is missing"
            )
        ancestors.add(lid)
        stack.extend(int(value) for value in net.preds.get(lid, ()))

    def dominates(candidate: int) -> bool:
        pending = [output]
        seen: set[int] = set()
        while pending:
            lid = int(pending.pop())
            if lid == candidate or lid in seen:
                continue
            seen.add(lid)
            preds = [int(value) for value in net.preds.get(lid, ())]
            if not preds:
                return False
            pending.extend(preds)
        return True

    return tuple(
        sorted(
            (
                lid
                for lid in ancestors
                if _kind(layer_by_id[lid].kind) == "ADD"
                and dominates(lid)
            ),
            key=position.__getitem__,
            reverse=True,
        )
    )


def operator_hz_property_suffix_stop_layer_id(
    net: Net,
    *,
    output_layer_id: int,
    suffix_blocks: int,
) -> Tuple[int, Tuple[int, ...]]:
    """Resolve the exact C12/C17 stop ADD without constructing an HZ."""

    if isinstance(suffix_blocks, (bool, np.bool_)) or not isinstance(
        suffix_blocks, (int, np.integer)
    ):
        raise OperatorHZBuildError("suffix_blocks must be an integer")
    suffix_blocks = int(suffix_blocks)
    if suffix_blocks <= 0:
        raise OperatorHZBuildError(
            "property suffix stop selection was requested while disabled"
        )
    candidates = operator_hz_property_suffix_dominating_add_candidates(
        net,
        output_layer_id=int(output_layer_id),
    )
    if suffix_blocks >= len(candidates):
        raise OperatorHZBuildError(
            "property suffix replay found too few dominating ADDs: "
            f"requested earlier-block depth {suffix_blocks}, "
            f"candidates={candidates}"
        )
    return int(candidates[suffix_blocks]), candidates


def _preactivation_frame_records(
    bounds: Mapping[int, Tuple[Any, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_lid in sorted(bounds):
        lid = int(raw_lid)
        lower = np.ascontiguousarray(
            np.asarray(bounds[raw_lid][0], dtype=np.float64).reshape(-1)
        )
        upper = np.ascontiguousarray(
            np.asarray(bounds[raw_lid][1], dtype=np.float64).reshape(-1)
        )
        if (
            lower.shape != upper.shape
            or lower.size == 0
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise OperatorHZBuildError(
                f"invalid exported preactivation bounds at ReLU {lid}"
            )
        records.append(
            {
                "layer_id": lid,
                "semantics": "preactivation",
                "width": int(lower.size),
                "lower_sha256": _f64_array_sha256(lower),
                "upper_sha256": _f64_array_sha256(upper),
            }
        )
    if not records:
        raise OperatorHZBuildError(
            "cannot export an empty preactivation frame"
        )
    return records


def _register_preactivation_frame(
    frame: OperatorHZPreactivationFrame,
) -> None:
    object_id = id(frame)

    def cleanup(
        _reference: "weakref.ReferenceType[OperatorHZPreactivationFrame]",
    ) -> None:
        with _PREACTIVATION_FRAME_LOCK:
            _PREACTIVATION_FRAME_AUTHORITIES.pop(object_id, None)

    reference = weakref.ref(frame, cleanup)
    with _PREACTIVATION_FRAME_LOCK:
        _PREACTIVATION_FRAME_AUTHORITIES[object_id] = (
            reference,
            frame.provenance_nonce,
            str(frame.receipt["receipt_sha256"]),
        )


def _make_operator_hz_preactivation_frame(
    *,
    net: Any,
    bounds: Mapping[int, Tuple[Any, Any]],
    residual_rows_tightened: int,
) -> OperatorHZPreactivationFrame:
    from act.back_end.hybridz_tf.query_dual_box_certifier import (
        query_dual_network_sha256,
    )

    records = _preactivation_frame_records(bounds)
    nonce = secrets.token_hex(32)
    body: Dict[str, Any] = {
        "schema": _PREACTIVATION_FRAME_SCHEMA,
        "status": "verified",
        "proof_authority": True,
        "authority_source": (
            "operator_hz_cube_plus_recursive_residual_shadow_intersections"
        ),
        "semantics": "relu_preactivation_outward_boxes",
        "network_sha256": query_dual_network_sha256(net),
        "bounds_records": records,
        "bounds_sha256": _canonical_json_sha256(records),
        "relu_layer_count": int(len(records)),
        "residual_bound_screen_requested": True,
        "residual_rows_tightened": int(residual_rows_tightened),
        "ordinary_internal_interval_facts_consumed": False,
        "input_fact_must_enclose_raw_box": True,
        "process_local_identity_capability_required": True,
        "provenance_nonce_sha256": hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest(),
    }
    body["receipt_sha256"] = _canonical_json_sha256(body)
    frame = OperatorHZPreactivationFrame(
        bounds=bounds,
        receipt=body,
        provenance_nonce=nonce,
    )
    _register_preactivation_frame(frame)
    return frame


def validate_operator_hz_preactivation_frame(
    frame: OperatorHZPreactivationFrame,
    *,
    net: Any,
    expected_network_sha256: Optional[str] = None,
    require_live_provenance: bool = True,
) -> bool:
    """Validate the exact live Operator-HZ bound frame before proof use."""

    try:
        if (
            not isinstance(frame, OperatorHZPreactivationFrame)
            or frame.proof_authority is not True
        ):
            return False
        receipt = frame.receipt
        body = copy.deepcopy(dict(receipt))
        claimed = str(body.pop("receipt_sha256"))
        if (
            body.get("schema") != _PREACTIVATION_FRAME_SCHEMA
            or body.get("status") != "verified"
            or body.get("proof_authority") is not True
            or body.get("authority_source")
            != (
                "operator_hz_cube_plus_recursive_residual_"
                "shadow_intersections"
            )
            or body.get("semantics")
            != "relu_preactivation_outward_boxes"
            or body.get("residual_bound_screen_requested") is not True
            or body.get("ordinary_internal_interval_facts_consumed")
            is not False
            or body.get("input_fact_must_enclose_raw_box") is not True
            or body.get("process_local_identity_capability_required")
            is not True
            or body.get("provenance_nonce_sha256")
            != hashlib.sha256(
                frame.provenance_nonce.encode("ascii")
            ).hexdigest()
            or not hmac.compare_digest(
                _canonical_json_sha256(body), claimed
            )
        ):
            return False
        if require_live_provenance:
            with _PREACTIVATION_FRAME_LOCK:
                entry = _PREACTIVATION_FRAME_AUTHORITIES.get(id(frame))
            if (
                entry is None
                or entry[0]() is not frame
                or not hmac.compare_digest(
                    entry[1], frame.provenance_nonce
                )
                or not hmac.compare_digest(entry[2], claimed)
            ):
                return False
        from act.back_end.hybridz_tf.query_dual_box_certifier import (
            query_dual_network_sha256,
        )

        live_network_sha = query_dual_network_sha256(net)
        if (
            not hmac.compare_digest(
                live_network_sha, str(body["network_sha256"])
            )
            or (
                expected_network_sha256 is not None
                and not hmac.compare_digest(
                    live_network_sha, str(expected_network_sha256)
                )
            )
        ):
            return False
        records = _preactivation_frame_records(frame.bounds)
        return bool(
            records == body.get("bounds_records")
            and body.get("bounds_sha256")
            == _canonical_json_sha256(records)
            and body.get("relu_layer_count") == len(records)
            and isinstance(
                body.get("residual_rows_tightened"), int
            )
            and not isinstance(
                body.get("residual_rows_tightened"), bool
            )
            and int(body["residual_rows_tightened"]) >= 0
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OperatorHZBuildError,
    ):
        return False


def _intersect_verified_query_dual_box(
    local_lower: Any,
    local_upper: Any,
    verified_lower: Any,
    verified_upper: Any,
    *,
    layer_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Atomically intersect one independently verified ReLU box.

    This helper deliberately has no repair path.  Shape, finiteness, ordering,
    or cross failures invalidate the complete operator build rather than
    clamping one side or restoring a baseline row after a partial mutation.
    """

    local_l = np.ascontiguousarray(
        np.asarray(local_lower, dtype=np.float64).reshape(-1)
    )
    local_u = np.ascontiguousarray(
        np.asarray(local_upper, dtype=np.float64).reshape(-1)
    )
    query_l = np.ascontiguousarray(
        np.asarray(verified_lower, dtype=np.float64).reshape(-1)
    )
    query_u = np.ascontiguousarray(
        np.asarray(verified_upper, dtype=np.float64).reshape(-1)
    )
    if (
        local_l.shape != local_u.shape
        or query_l.shape != query_u.shape
        or query_l.shape != local_l.shape
        or local_l.size == 0
    ):
        raise OperatorHZBuildError(
            f"verified query-dual ReLU {int(layer_id)} bound shape mismatch: "
            f"local={local_l.shape}/{local_u.shape}, "
            f"verified={query_l.shape}/{query_u.shape}"
        )
    if (
        not np.all(np.isfinite(local_l))
        or not np.all(np.isfinite(local_u))
        or not np.all(np.isfinite(query_l))
        or not np.all(np.isfinite(query_u))
    ):
        raise OperatorHZBuildError(
            f"verified query-dual ReLU {int(layer_id)} bounds are non-finite"
        )
    if np.any(local_l > local_u) or np.any(query_l > query_u):
        raise OperatorHZBuildError(
            f"verified query-dual ReLU {int(layer_id)} has reversed bounds"
        )

    # Compute both sides before publishing either one.  A cross is a hard
    # soundness failure, not a reason to clip one side to the other.
    combined_l = np.maximum(local_l, query_l)
    combined_u = np.minimum(local_u, query_u)
    crossed = np.flatnonzero(combined_l > combined_u)
    if crossed.size:
        row = int(crossed[0])
        raise OperatorHZBuildError(
            f"verified query-dual ReLU {int(layer_id)} conflicts at row "
            f"{row}: lower={combined_l[row]} > upper={combined_u[row]}"
        )
    return (
        np.ascontiguousarray(combined_l, dtype=np.float64),
        np.ascontiguousarray(combined_u, dtype=np.float64),
    )


def _constraint_blocks_sha256(
    blocks: Sequence[_ConstraintBlock],
) -> str:
    """Hash an ordered local-constraint block slice."""

    digest = hashlib.sha256()
    digest.update(np.asarray([len(blocks)], dtype=np.int64).tobytes())
    for block in blocks:
        encoded_tag = str(block.tag).encode("utf-8")
        digest.update(
            np.asarray([len(encoded_tag)], dtype=np.int64).tobytes()
        )
        digest.update(encoded_tag)
        digest.update(bytes.fromhex(_csr_sha256(block.Ac)))
        digest.update(bytes.fromhex(_csr_sha256(block.Ab)))
        rhs = np.ascontiguousarray(block.rhs, dtype=np.float64)
        digest.update(
            np.asarray(rhs.shape, dtype=np.int64).tobytes()
        )
        digest.update(rhs.tobytes())
    return digest.hexdigest()


def _normalize_preactivation_targets(
    targets: Optional[Any],
) -> Optional[Dict[int, Tuple[int, ...]]]:
    """Normalize an explicit ``(layer_id, row)`` schedule.

    Accepted forms are an ordered sequence of pairs, ``{layer: rows}``, or an
    ordered mapping whose keys themselves are ``(layer, row)`` pairs.  Mapping
    insertion order is retained.  Duplicate targets are removed without
    changing the first occurrence.  ``None`` preserves the legacy
    deterministic first-unstable-row policy.
    """

    if targets is None:
        return None
    pairs: List[Tuple[int, int]] = []
    if isinstance(targets, Mapping):
        for key, value in targets.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                pairs.append((int(key[0]), int(key[1])))
                continue
            layer_id = int(key)
            if isinstance(value, (str, bytes)):
                raise OperatorHZBuildError(
                    "preactivation target rows cannot be strings"
                )
            if np.isscalar(value):
                rows = (int(value),)
            else:
                try:
                    rows = tuple(int(row) for row in value)
                except TypeError as exc:
                    raise OperatorHZBuildError(
                        "preactivation target mapping values must be rows"
                    ) from exc
            pairs.extend((layer_id, row) for row in rows)
    else:
        if isinstance(targets, (str, bytes)):
            raise OperatorHZBuildError(
                "preactivation_targets must contain (layer_id, row) pairs"
            )
        try:
            entries = list(targets)
        except TypeError as exc:
            raise OperatorHZBuildError(
                "preactivation_targets must be a sequence or mapping"
            ) from exc
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise OperatorHZBuildError(
                    "each preactivation target must be (layer_id, row)"
                )
            pairs.append((int(entry[0]), int(entry[1])))

    normalized: Dict[int, List[int]] = {}
    seen: set[Tuple[int, int]] = set()
    for layer_id, row in pairs:
        if layer_id < 0 or row < 0:
            raise OperatorHZBuildError(
                "preactivation target layer ids and rows must be nonnegative"
            )
        pair = (layer_id, row)
        if pair in seen:
            continue
        seen.add(pair)
        normalized.setdefault(layer_id, []).append(row)
    return {
        layer_id: tuple(rows)
        for layer_id, rows in normalized.items()
    }


_RESIDUAL_GUARDS = frozenset({"none", "zero", "identity", "both"})


def _normalize_residual_targets(
    targets: Optional[Any],
) -> Optional[Dict[int, Dict[int, str]]]:
    """Normalize explicit ``(ReLU layer, row, guard)`` targets.

    Residualization is never selected by topological position.  Callers must
    name every candidate and its retained lower guard.  Accepted forms are an
    ordered sequence of triples, ``{(layer, row): guard}``, or
    ``{layer: {row: guard}}``.  Duplicate coordinates are rejected rather
    than silently changing a property-derived schedule.
    """

    if targets is None:
        return None
    triples: List[Tuple[int, int, str]] = []
    if isinstance(targets, Mapping):
        for key, value in targets.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                triples.append((int(key[0]), int(key[1]), str(value)))
                continue
            layer_id = int(key)
            if not isinstance(value, Mapping):
                raise OperatorHZBuildError(
                    "residual target mapping values must be {row: guard}"
                )
            triples.extend(
                (layer_id, int(row), str(guard))
                for row, guard in value.items()
            )
    else:
        if isinstance(targets, (str, bytes)):
            raise OperatorHZBuildError(
                "residual_targets must contain (layer_id, row, guard) triples"
            )
        try:
            entries = list(targets)
        except TypeError as exc:
            raise OperatorHZBuildError(
                "residual_targets must be a sequence or mapping"
            ) from exc
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                raise OperatorHZBuildError(
                    "each residual target must be (layer_id, row, guard)"
                )
            triples.append((int(entry[0]), int(entry[1]), str(entry[2])))

    normalized: Dict[int, Dict[int, str]] = {}
    seen: set[Tuple[int, int]] = set()
    for layer_id, row, raw_guard in triples:
        if layer_id < 0 or row < 0:
            raise OperatorHZBuildError(
                "residual target layer ids and rows must be nonnegative"
            )
        guard = raw_guard.strip().lower()
        if guard not in _RESIDUAL_GUARDS:
            raise OperatorHZBuildError(
                f"invalid residual guard {raw_guard!r}; "
                f"expected one of {sorted(_RESIDUAL_GUARDS)}"
            )
        coordinate = (layer_id, row)
        if coordinate in seen:
            raise OperatorHZBuildError(
                f"duplicate residual target {coordinate}"
            )
        seen.add(coordinate)
        normalized.setdefault(layer_id, {})[row] = guard
    return normalized


def _normalize_exact_target_reservoir(
    targets: Optional[Any],
) -> Optional[Dict[int, Tuple[int, ...]]]:
    """Normalize same-layer backup coordinates for post-screen exact bits.

    The reservoir is deliberately separate from ``residual_targets``.  A
    backup row is an exact-bit scheduling candidate only: if it is not used,
    it receives the ordinary triangle relaxation and is never silently
    residualized.  Accepted forms are an ordered sequence of ``(layer,row)``
    pairs or ``{layer: [row, ...]}``.  Duplicate coordinates are rejected so
    caller order remains an auditable deterministic tie break.
    """

    if targets is None:
        return None
    pairs: List[Tuple[int, int]] = []

    def strict_nonnegative_int(value: Any, *, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise OperatorHZBuildError(f"{name} must be an integer")
        result = int(value)
        if result < 0:
            raise OperatorHZBuildError(f"{name} must be nonnegative")
        return result

    if isinstance(targets, Mapping):
        for raw_layer, raw_rows in targets.items():
            layer_id = strict_nonnegative_int(
                raw_layer, name="exact reservoir layer id"
            )
            if isinstance(raw_rows, (str, bytes)):
                raise OperatorHZBuildError(
                    "exact reservoir mapping values must be row sequences"
                )
            try:
                rows = list(raw_rows)
            except TypeError as exc:
                raise OperatorHZBuildError(
                    "exact reservoir mapping values must be row sequences"
                ) from exc
            pairs.extend(
                (
                    layer_id,
                    strict_nonnegative_int(
                        raw_row, name="exact reservoir row"
                    ),
                )
                for raw_row in rows
            )
    else:
        if isinstance(targets, (str, bytes)):
            raise OperatorHZBuildError(
                "exact_target_reservoir must contain (layer_id, row) pairs"
            )
        try:
            entries = list(targets)
        except TypeError as exc:
            raise OperatorHZBuildError(
                "exact_target_reservoir must be a sequence or mapping"
            ) from exc
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise OperatorHZBuildError(
                    "each exact reservoir target must be (layer_id, row)"
                )
            pairs.append(
                (
                    strict_nonnegative_int(
                        entry[0], name="exact reservoir layer id"
                    ),
                    strict_nonnegative_int(
                        entry[1], name="exact reservoir row"
                    ),
                )
            )

    normalized: Dict[int, List[int]] = {}
    seen: set[Tuple[int, int]] = set()
    for layer_id, row in pairs:
        coordinate = (int(layer_id), int(row))
        if coordinate in seen:
            raise OperatorHZBuildError(
                f"duplicate exact reservoir target {coordinate}"
            )
        seen.add(coordinate)
        normalized.setdefault(int(layer_id), []).append(int(row))
    return {
        layer_id: tuple(normalized[layer_id])
        for layer_id in sorted(normalized)
    }


_F64_EPS = np.finfo(np.float64).eps
_F64_TINY = np.finfo(np.float64).tiny


def _gamma_ops(op_count: np.ndarray | int, *, name: str) -> np.ndarray:
    """Return Higham's ``gamma_k`` and fail closed outside its finite regime."""

    count = np.asarray(op_count, dtype=np.float64)
    if count.ndim == 0:
        # Most exact-ReLU roundoff guards use one fixed operation count for
        # the complete row block.  Validating a broadcast row-sized view here
        # repeated two NumPy reductions for every guard.  Keep the identical
        # binary64 formula while validating the scalar before broadcasting is
        # needed by the caller.
        scalar = np.float64(count)
        if not np.isfinite(scalar) or scalar < 0.0:
            raise OperatorHZBuildError(
                f"{name} has an invalid operation count"
            )
        product = scalar * _F64_EPS
        if product >= 0.5:
            raise OperatorHZBuildError(
                f"{name} is too large for a finite binary64 roundoff allowance"
            )
        return product / (1.0 - product)
    if np.any(count < 0.0) or not np.all(np.isfinite(count)):
        raise OperatorHZBuildError(f"{name} has an invalid operation count")
    product = count * _F64_EPS
    if np.any(product >= 0.5):
        raise OperatorHZBuildError(
            f"{name} is too large for a finite binary64 roundoff allowance"
        )
    return product / (1.0 - product)


def _inflate_nonnegative(
    rounded: np.ndarray,
    op_count: np.ndarray | int,
    *,
    active: Optional[np.ndarray] = None,
    name: str,
) -> np.ndarray:
    """Outward-inflate a rounded nonnegative reduction.

    ``rounded`` is a binary64 result of nonnegative products/additions.
    Dividing by ``1-gamma_k`` bounds its exact-real counterpart.  A deliberately
    conservative normal-minimum term covers gradual underflow.  Structurally
    zero rows stay exactly zero so zero networks retain point consistency.
    """

    value = np.asarray(rounded, dtype=np.float64)
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise OperatorHZBuildError(f"{name} is non-finite or negative")
    count_raw = np.asarray(op_count, dtype=np.float64)
    scalar_count = count_raw.ndim == 0
    if scalar_count:
        count = np.float64(count_raw)
        gamma = _gamma_ops(count, name=name)
        underflow = _F64_TINY * np.maximum(1.0, count)
    else:
        count = np.broadcast_to(count_raw, value.shape)
        gamma = _gamma_ops(count, name=name)
        underflow = _F64_TINY * np.maximum(1.0, count)
    if active is None:
        active_mask = value > 0.0
    else:
        active_mask = np.broadcast_to(np.asarray(active, dtype=bool), value.shape)

    if np.all(active_mask):
        # The common Conv/ADD/exact-ReLU path has no structurally zero rows.
        # Avoid three boolean gather/scatter copies while preserving the same
        # binary64 operation order used by the mixed-row path below.
        out = value / (1.0 - gamma)
        out = out + underflow
        out = np.nextafter(out, np.inf)
    else:
        out = np.zeros_like(value)
        if np.any(active_mask):
            idx = active_mask
            if scalar_count:
                out[idx] = value[idx] / (1.0 - gamma)
                out[idx] = out[idx] + underflow
            else:
                out[idx] = value[idx] / (1.0 - gamma[idx])
                out[idx] = out[idx] + underflow[idx]
            out[idx] = np.nextafter(out[idx], np.inf)
    if not np.all(np.isfinite(out)):
        raise OperatorHZBuildError(f"{name} outward inflation overflowed")
    return out


def _nonnegative_sum_upper(*terms: np.ndarray, name: str) -> np.ndarray:
    """Return an outward upper bound on a short sum of nonnegative arrays."""

    if not terms:
        raise OperatorHZBuildError(f"{name} requires at least one term")
    arrays = [np.asarray(term, dtype=np.float64) for term in terms]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise OperatorHZBuildError(f"{name} shape mismatch")
    if any(not np.all(np.isfinite(array)) or np.any(array < 0.0)
           for array in arrays):
        raise OperatorHZBuildError(f"{name} has a non-finite/negative term")
    rounded = np.zeros(shape, dtype=np.float64)
    for array in arrays:
        rounded = rounded + array
    active = np.zeros(shape, dtype=bool)
    for array in arrays:
        active |= array > 0.0
    return _inflate_nonnegative(
        rounded,
        max(1, 2 * len(arrays)),
        active=active,
        name=name,
    )


def _row_l1_upper(matrix: sp.csr_matrix, *, name: str) -> np.ndarray:
    """Outward row-wise L1 mass of a binary64 sparse matrix."""

    matrix = matrix.tocsr().astype(np.float64, copy=False)
    # ``abs(matrix).sum(axis=1)`` duplicates data, indices and indptr even
    # though only the absolute values participate in the reduction.  Reduce
    # the canonical CSR data in the same stored row order and allocate no
    # second sparse matrix.
    starts = np.asarray(matrix.indptr[:-1], dtype=np.int64)
    ends = np.asarray(matrix.indptr[1:], dtype=np.int64)
    nonempty = ends > starts
    raw = np.zeros(matrix.shape[0], dtype=np.float64)
    if matrix.data.size:
        raw[nonempty] = np.add.reduceat(
            np.abs(matrix.data), starts[nonempty]
        )
    nnz = np.diff(matrix.indptr).astype(np.float64)
    return _inflate_nonnegative(
        raw,
        2.0 * nnz + 2.0,
        active=nnz > 0.0,
        name=name,
    )


def _positive_spmv_upper(
    matrix: sp.csr_matrix,
    vector: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Outward bound ``matrix @ vector`` for nonnegative operands."""

    matrix = matrix.tocsr().astype(np.float64, copy=False)
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if (
        matrix.shape[1] != vector.size
        or np.any(matrix.data < 0.0)
        or np.any(vector < 0.0)
        or not np.all(np.isfinite(matrix.data))
        or not np.all(np.isfinite(vector))
    ):
        raise OperatorHZBuildError(f"{name} requires finite nonnegative operands")
    raw = np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)
    nnz = np.diff(matrix.indptr).astype(np.float64)
    # A row is structurally active only when it contains a coefficient whose
    # corresponding upper mass is nonzero.  Source mass and propagated error
    # are normally positive in every live column, so a second boolean sparse
    # matrix and SpMV are redundant on the common path.  Preserve the exact
    # mixed/zero-column predicate when that invariant does not hold.
    positive = vector > 0.0
    if np.all(positive):
        active = nnz > 0.0
    elif not np.any(positive):
        active = np.zeros(matrix.shape[0], dtype=bool)
    else:
        active = np.asarray(
            matrix.astype(bool) @ positive, dtype=bool
        ).reshape(-1)
    return _inflate_nonnegative(
        raw,
        2.0 * nnz + 2.0,
        active=active,
        name=name,
    )


def _relu_triangle_parameters(
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Fraction-audited outer line ``relu(x) <= slope*x+intercept``.

    For any fixed stored-float slope, ``relu(x)-slope*x`` is linear on
    ``[l,0]`` and ``[0,u]``.  Its maximum is therefore attained at one of
    ``l, 0, u``.  We evaluate those three endpoint requirements exactly over
    the stored binary64 values and round the chosen intercept toward ``+inf``.
    This avoids relying on the rounded identity
    ``u/(u-l) * (u-l) == u``, which is false for ordinary decimal bounds.
    """

    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if (
        lower.size != upper.size
        or np.any(lower >= 0.0)
        or np.any(upper <= 0.0)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise OperatorHZBuildError("triangle parameters require finite l<0<u")
    denominator = upper - lower
    slope = upper / denominator
    if (
        not np.all(np.isfinite(denominator))
        or np.any(denominator <= 0.0)
        or not np.all(np.isfinite(slope))
    ):
        raise OperatorHZBuildError("triangle slope overflowed")

    intercept = np.empty(lower.size, dtype=np.float64)
    inflation = np.empty(lower.size, dtype=np.float64)
    for row in range(lower.size):
        lf = Fraction.from_float(float(lower[row]))
        uf = Fraction.from_float(float(upper[row]))
        sf = Fraction.from_float(float(slope[row]))
        required = max(
            Fraction(0),
            -sf * lf,
            uf * (Fraction(1) - sf),
        )
        try:
            rounded = float(required)
        except OverflowError as exc:
            raise OperatorHZBuildError(
                f"triangle intercept overflowed at row {row}"
            ) from exc
        if not np.isfinite(rounded):
            raise OperatorHZBuildError(
                f"triangle intercept is non-finite at row {row}"
            )
        if Fraction.from_float(rounded) < required:
            rounded = float(np.nextafter(rounded, np.inf))
        if (
            not np.isfinite(rounded)
            or Fraction.from_float(rounded) < required
        ):
            raise OperatorHZBuildError(
                f"triangle intercept could not be rounded outward at row {row}"
            )
        intercept[row] = rounded
        nominal = -float(slope[row]) * float(lower[row])
        inflation[row] = max(0.0, rounded - nominal)
    return slope, intercept, inflation


def _fraction_to_f64_upper(value: Fraction, *, name: str) -> float:
    """Round an exact rational upper bound toward ``+inf``."""

    try:
        rounded = float(value)
    except OverflowError as exc:
        raise OperatorHZBuildError(f"{name} overflows binary64") from exc
    if not np.isfinite(rounded):
        raise OperatorHZBuildError(f"{name} is non-finite")
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    if not np.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise OperatorHZBuildError(
            f"{name} could not be rounded outward"
        )
    return rounded


def _property_relu_upper_planes(
    C: np.ndarray,
    thresholds: np.ndarray,
    weight: sp.csr_matrix,
    bias: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    negative_alpha: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fold ``C @ (W @ relu(x) + b) - t`` into affine upper planes.

    Every coefficient of ``q = C @ W`` is accumulated with exact dyadic
    :class:`Fraction` arithmetic over the stored binary64 inputs.  For the
    chosen stored plane coefficient ``d``, the required intercept is the exact
    maximum of ``q*relu(x)-d*x`` at ``l,0,u``.  A final exact accumulation of
    all per-neuron requirements and ``C@b-t`` is rounded once toward
    ``+inf``.  The returned rows therefore dominate the original property on
    the supplied outer preactivation box without relying on a rounded matrix
    multiplication identity.

    For unstable rows, positive ``q`` uses the stored secant slope while
    non-positive ``q`` uses the sound lower facet ``relu(x)>=0``.  This is a
    fixed, optimization-free DeepPoly/CROWN tail policy; its output is later
    optimized over the existing HybridZ prefix constraints.
    """

    C = np.asarray(C, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    weight = sp.csr_matrix(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if (
        C.ndim != 2
        or C.shape[0] != thresholds.size
        or C.shape[1] != weight.shape[0]
        or bias.size != weight.shape[0]
        or lower.size != weight.shape[1]
        or upper.size != lower.size
        or np.any(lower > upper)
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(thresholds))
        or not np.all(np.isfinite(weight.data))
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise OperatorHZBuildError(
            "property-tail plane inputs are malformed or non-finite"
        )
    if negative_alpha is None:
        negative_alpha_array = np.zeros(
            (C.shape[0], lower.size), dtype=np.float64
        )
    else:
        negative_alpha_array = np.asarray(
            negative_alpha, dtype=np.float64
        )
    if (
        negative_alpha_array.shape != (C.shape[0], lower.size)
        or not np.all(np.isfinite(negative_alpha_array))
        or np.any(negative_alpha_array < 0.0)
        or np.any(negative_alpha_array > 1.0)
    ):
        raise OperatorHZBuildError(
            "property-tail negative alpha is malformed or outside [0,1]"
        )

    inactive = upper <= 0.0
    active = (lower >= 0.0) & (upper > 0.0)
    unstable = (lower < 0.0) & (upper > 0.0)
    if int(inactive.sum() + active.sum() + unstable.sum()) != lower.size:
        raise OperatorHZBuildError(
            "property-tail ReLU phase partition is incomplete"
        )
    secant = np.zeros(lower.size, dtype=np.float64)
    if np.any(unstable):
        secant_values, _intercept, _inflation = _relu_triangle_parameters(
            lower[unstable], upper[unstable]
        )
        secant[unstable] = secant_values

    dense_weight = weight.toarray()
    planes = np.zeros((C.shape[0], lower.size), dtype=np.float64)
    intercepts = np.zeros(C.shape[0], dtype=np.float64)
    positive_unstable = 0
    negative_unstable = 0
    negative_unstable_zero_facets = 0
    nonzero_negative_alpha = 0
    requested_nonzero_negative_alpha = int(
        np.count_nonzero(negative_alpha_array)
    )
    negative_d_below_q = 0
    max_negative_requirement = Fraction(0)
    exact_q_terms = 0
    max_requirement = Fraction(0)

    for rival in range(C.shape[0]):
        c_nonzero = np.flatnonzero(C[rival] != 0.0)
        exact_q: List[Fraction] = []
        for neuron in range(lower.size):
            q = sum(
                (
                    Fraction.from_float(float(C[rival, output]))
                    * Fraction.from_float(
                        float(dense_weight[output, neuron])
                    )
                    for output in c_nonzero
                    if dense_weight[output, neuron] != 0.0
                ),
                Fraction(0),
            )
            exact_q.append(q)
            exact_q_terms += int(c_nonzero.size)

        bias_exact = (
            sum(
                (
                    Fraction.from_float(float(C[rival, output]))
                    * Fraction.from_float(float(bias[output]))
                    for output in c_nonzero
                    if bias[output] != 0.0
                ),
                Fraction(0),
            )
            - Fraction.from_float(float(thresholds[rival]))
        )
        required_total = bias_exact
        for neuron, q in enumerate(exact_q):
            if q == 0 or inactive[neuron]:
                stored_d = 0.0
            elif active[neuron]:
                try:
                    stored_d = float(q)
                except OverflowError as exc:
                    raise OperatorHZBuildError(
                        "property-tail active coefficient overflowed"
                    ) from exc
            elif q > 0:
                try:
                    stored_d = float(
                        q * Fraction.from_float(float(secant[neuron]))
                    )
                except OverflowError as exc:
                    raise OperatorHZBuildError(
                        "property-tail positive unstable coefficient "
                        "overflowed"
                    ) from exc
                positive_unstable += 1
            else:
                # relu(x) >= alpha*x for every alpha in [0,1].  Multiplying
                # by q<0 gives a valid upper plane.  The optimizer choosing
                # alpha is heuristic-only; the exact endpoint requirement
                # below remains the proof authority for the stored value.
                alpha_fraction = Fraction.from_float(
                    float(negative_alpha_array[rival, neuron])
                )
                try:
                    stored_d = float(q * alpha_fraction)
                except OverflowError as exc:
                    raise OperatorHZBuildError(
                        "property-tail negative unstable coefficient "
                        "overflowed"
                    ) from exc
                negative_unstable += 1
                negative_unstable_zero_facets += int(alpha_fraction == 0)
                nonzero_negative_alpha += int(alpha_fraction != 0)
                negative_d_below_q += int(
                    Fraction.from_float(float(stored_d)) < q
                )
            if not np.isfinite(stored_d):
                raise OperatorHZBuildError(
                    "property-tail plane coefficient is non-finite"
                )
            planes[rival, neuron] = stored_d
            q_fraction = q
            d_fraction = Fraction.from_float(float(stored_d))
            endpoints = [
                Fraction.from_float(float(lower[neuron])),
                Fraction.from_float(float(upper[neuron])),
            ]
            if lower[neuron] < 0.0 < upper[neuron]:
                endpoints.append(Fraction(0))
            requirement = max(
                (
                    q_fraction * max(Fraction(0), point)
                    - d_fraction * point
                    for point in endpoints
                ),
                default=Fraction(0),
            )
            required_total += requirement
            max_requirement = max(max_requirement, requirement)
            if q < 0 and unstable[neuron]:
                max_negative_requirement = max(
                    max_negative_requirement, requirement
                )
        intercepts[rival] = _fraction_to_f64_upper(
            required_total,
            name=f"property-tail intercept[{rival}]",
        )

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(planes).tobytes())
    digest.update(np.ascontiguousarray(intercepts).tobytes())
    return planes, intercepts, {
        "schema": "operator_hz_property_tail_fraction_v1",
        "proof_authority": True,
        "rivals": int(C.shape[0]),
        "preactivation_rows": int(lower.size),
        "phase_inactive": int(inactive.sum()),
        "phase_active": int(active.sum()),
        "phase_unstable": int(unstable.sum()),
        "positive_unstable_planes": int(positive_unstable),
        "negative_unstable_planes": int(negative_unstable),
        "nonpositive_unstable_zero_facets": int(
            negative_unstable_zero_facets
        ),
        "nonzero_negative_alpha": int(nonzero_negative_alpha),
        "requested_nonzero_negative_alpha": int(
            requested_nonzero_negative_alpha
        ),
        "ignored_nonzero_negative_alpha": int(
            requested_nonzero_negative_alpha - nonzero_negative_alpha
        ),
        "negative_d_below_exact_q": int(negative_d_below_q),
        "negative_alpha_max": float(
            np.max(negative_alpha_array)
            if negative_alpha_array.size else 0.0
        ),
        "exact_q_term_visits": int(exact_q_terms),
        "max_exact_endpoint_requirement": _fraction_to_f64_upper(
            max_requirement,
            name="property-tail max endpoint requirement",
        ),
        "max_negative_exact_endpoint_requirement": _fraction_to_f64_upper(
            max_negative_requirement,
            name="property-tail max negative endpoint requirement",
        ),
        "planes_sha256": digest.hexdigest(),
    }


def _strict_pair(value: Any, *, name: str, positive: bool) -> Tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise OperatorHZBuildError(f"{name} is not an integer pair") from exc
        if len(raw) != 2:
            raise OperatorHZBuildError(
                f"{name} must have exactly two entries, got {raw!r}"
            )
        pair = (int(raw[0]), int(raw[1]))
    if positive and (pair[0] <= 0 or pair[1] <= 0):
        raise OperatorHZBuildError(f"{name} must be positive, got {pair}")
    if not positive and (pair[0] < 0 or pair[1] < 0):
        raise OperatorHZBuildError(f"{name} must be nonnegative, got {pair}")
    return pair


def _validate_strict_conv2d_layer(layer: Any) -> None:
    """Validate the exact NCHW/zero-padding operator contract before expansion."""

    params = layer.params
    data_format = str(params.get("data_format", "NCHW")).upper()
    padding_mode = str(params.get("padding_mode", "zeros")).lower()
    auto_pad = str(params.get("auto_pad", "NOTSET")).upper()
    if data_format != "NCHW":
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} data_format={data_format!r}, expected NCHW"
        )
    if padding_mode not in {"zeros", "zero", "constant"}:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} padding_mode={padding_mode!r} is unsupported"
        )
    if auto_pad not in {"NOTSET", "NONE", ""}:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} auto_pad={auto_pad!r} is unsupported"
        )

    try:
        input_shape = tuple(int(v) for v in params["input_shape"])
        output_shape = tuple(int(v) for v in params["output_shape"])
    except Exception as exc:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} lacks valid input/output shapes"
        ) from exc
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} requires explicit 4D NCHW shapes, got "
            f"{input_shape}/{output_shape}"
        )
    batch, in_ch, in_h, in_w = input_shape
    out_batch, out_ch_declared, out_h, out_w = output_shape
    if batch != 1 or out_batch != 1:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} strict operator mode requires batch 1"
        )

    weight = params.get("weight")
    weight_shape = tuple(int(v) for v in getattr(weight, "shape", ()))
    if len(weight_shape) != 4:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} weight is not OIHW: {weight_shape}"
        )
    out_ch, in_ch_per_group, kh, kw = weight_shape
    groups = int(params.get("groups", 1))
    stride = _strict_pair(
        params.get("stride", 1), name=f"CONV2D[{layer.id}].stride", positive=True
    )
    padding = _strict_pair(
        params.get("padding", 0),
        name=f"CONV2D[{layer.id}].padding",
        positive=False,
    )
    dilation = _strict_pair(
        params.get("dilation", 1),
        name=f"CONV2D[{layer.id}].dilation",
        positive=True,
    )
    if groups <= 0 or in_ch_per_group * groups != in_ch or out_ch % groups:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} has invalid groups/channels: "
            f"in={in_ch}, out={out_ch}, per_group={in_ch_per_group}, groups={groups}"
        )
    if out_ch != out_ch_declared:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} output channels disagree: "
            f"weight={out_ch}, shape={out_ch_declared}"
        )
    if "in_channels" in params and int(params["in_channels"]) != in_ch:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} declared in_channels mismatch"
        )
    if "out_channels" in params and int(params["out_channels"]) != out_ch:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} declared out_channels mismatch"
        )
    bias = params.get("bias")
    if bias is not None and int(np.prod(tuple(int(v) for v in bias.shape))) != out_ch:
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} bias length does not equal out_channels"
        )

    expected_h = (
        in_h
        + 2 * padding[0]
        - dilation[0] * (kh - 1)
        - 1
    ) // stride[0] + 1
    expected_w = (
        in_w
        + 2 * padding[1]
        - dilation[1] * (kw - 1)
        - 1
    ) // stride[1] + 1
    if expected_h <= 0 or expected_w <= 0 or (out_h, out_w) != (
        expected_h,
        expected_w,
    ):
        raise OperatorHZBuildError(
            f"CONV2D layer {layer.id} output geometry mismatch: "
            f"declared={(out_h, out_w)}, expected={(expected_h, expected_w)}"
        )


def _kind(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _as_finite_vector(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu().double().numpy().reshape(-1)
    else:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
    out = out.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        raise OperatorHZBuildError(f"{name} contains NaN or infinity")
    return out


def _pad_cols(mat: sp.csr_matrix, width: int) -> sp.csr_matrix:
    mat = mat.tocsr().astype(np.float64, copy=False)
    width = int(width)
    if mat.shape[1] > width:
        raise OperatorHZBuildError(
            f"cannot shrink sparse frame from {mat.shape[1]} to {width} columns"
        )
    if mat.shape[1] == width:
        return mat
    return sp.hstack(
        [mat, sp.csr_matrix((mat.shape[0], width - mat.shape[1]), dtype=np.float64)],
        format="csr",
    )


def _require_canonical_csr(matrix: sp.csr_matrix, *, name: str) -> sp.csr_matrix:
    """Return sorted canonical CSR, failing closed on duplicate coefficients.

    The roundoff operation counts assume one stored coefficient per
    row/column pair.  Sorting indices is an exact permutation and is allowed;
    silently summing duplicates would itself be a rounded transformation of
    the exact-real operator, so strict mode rejects duplicates.
    """

    matrix = matrix.tocsr().astype(np.float64, copy=False)
    if not bool(matrix.has_sorted_indices):
        matrix.sort_indices()
    if not bool(matrix.has_canonical_format):
        raise OperatorHZBuildError(
            f"{name} contains duplicate sparse coefficients"
        )
    return matrix


def _absolute_csr_topology_view(
    matrix: sp.csr_matrix,
    *,
    name: str,
) -> sp.csr_matrix:
    """Return ``abs(matrix)`` while reusing its immutable CSR topology.

    Affine error propagation reads the absolute operator twice but never
    mutates it.  SciPy's generic ``abs(CSR)`` duplicates data, indices and
    indptr.  The canonical topology is already final here, so only negative
    data require a new buffer; indices and indptr remain shared read-only
    inputs to the two SpMVs.  A negative signed zero also takes this path so
    stored bits match generic absolute value exactly.
    """

    matrix = _require_canonical_csr(matrix, name=name)
    if not np.all(np.isfinite(matrix.data)):
        raise OperatorHZBuildError(f"{name} contains NaN or infinity")
    if not np.any(np.signbit(matrix.data)):
        return matrix
    return sp.csr_matrix(
        (np.abs(matrix.data), matrix.indices, matrix.indptr),
        shape=matrix.shape,
        dtype=np.float64,
        copy=False,
    )


def _stack_padded(
    blocks: Iterable[sp.csr_matrix],
    *,
    width: int,
) -> sp.csr_matrix:
    padded: List[sp.csr_matrix] = []
    for block in blocks:
        matrix = block.tocsr().astype(np.float64, copy=False)
        if matrix.shape[1] > int(width):
            raise OperatorHZBuildError(
                "cannot shrink sparse assembly block from "
                f"{matrix.shape[1]} to {int(width)} columns"
            )
        if matrix.shape[1] < int(width):
            # A wider CSR shape with unchanged indptr/indices/data is exactly
            # zero-column padding.  The read-only assembly view shares the
            # three existing arrays, avoiding one full constraint copy per
            # block before ``vstack`` creates the owned final CSR.
            matrix = sp.csr_matrix(
                (matrix.data, matrix.indices, matrix.indptr),
                shape=(int(matrix.shape[0]), int(width)),
                dtype=np.float64,
                copy=False,
            )
        padded.append(matrix)
    if not padded:
        return sp.csr_matrix((0, int(width)), dtype=np.float64)
    out = sp.vstack(padded, format="csr")
    out.eliminate_zeros()
    return out


def _assemble_owned_operator_sparse_hz(
    *,
    c: np.ndarray,
    Gc: sp.csr_matrix,
    Gb: sp.csr_matrix,
    Ac: sp.csr_matrix,
    Ab: sp.csr_matrix,
    b: np.ndarray,
    Auc: sp.csr_matrix,
    Aub: sp.csr_matrix,
    ub: np.ndarray,
    col_ids: np.ndarray,
    bcol_ids: np.ndarray,
) -> SparseHZono:
    """Assemble the builder's already-owned canonical core without recopy.

    Final sparse assembly has just created detached CSR matrices and all
    traversal/constraint aliases have been released.  Passing those matrices
    through ``SparseHZono.__post_init__`` would nevertheless call same-dtype
    ``astype`` and copy the complete constraint core once more.  Validate the
    exact owned buffers here, then install them directly; this changes only
    object lifetime, never a stored coefficient.
    """

    dense = {
        "c": (c, np.dtype(np.float64)),
        "b": (b, np.dtype(np.float64)),
        "ub": (ub, np.dtype(np.float64)),
        "col_ids": (col_ids, np.dtype(np.int64)),
        "bcol_ids": (bcol_ids, np.dtype(np.int64)),
    }
    normalized_dense: Dict[str, np.ndarray] = {}
    for name, (value, dtype) in dense.items():
        array = np.asarray(value, dtype=dtype).reshape(-1)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if dtype == np.dtype(np.float64) and not np.all(
            np.isfinite(array)
        ):
            raise OperatorHZBuildError(
                f"owned SparseHZ {name} contains NaN or infinity"
            )
        normalized_dense[name] = array

    n_out = int(normalized_dense["c"].size)
    n_cont = int(normalized_dense["col_ids"].size)
    n_bin = int(normalized_dense["bcol_ids"].size)
    n_eq = int(normalized_dense["b"].size)
    n_ub = int(normalized_dense["ub"].size)
    expected_shapes = {
        "Gc": (n_out, n_cont),
        "Gb": (n_out, n_bin),
        "Ac": (n_eq, n_cont),
        "Ab": (n_eq, n_bin),
        "Auc": (n_ub, n_cont),
        "Aub": (n_ub, n_bin),
    }
    sparse = {
        "Gc": Gc,
        "Gb": Gb,
        "Ac": Ac,
        "Ab": Ab,
        "Auc": Auc,
        "Aub": Aub,
    }
    for name, matrix in sparse.items():
        rows, columns = expected_shapes[name]
        if (
            type(matrix) is not sp.csr_matrix
            or matrix.dtype != np.dtype(np.float64)
            or matrix.shape != (rows, columns)
            or matrix.data.ndim != 1
            or matrix.indices.ndim != 1
            or matrix.indptr.ndim != 1
            or not matrix.data.flags.c_contiguous
            or not matrix.indices.flags.c_contiguous
            or not matrix.indptr.flags.c_contiguous
            or matrix.data.size != matrix.indices.size
            or matrix.indptr.size != rows + 1
            or int(matrix.indptr[0]) != 0
            or int(matrix.indptr[-1]) != int(matrix.data.size)
            or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
            or np.any(matrix.indices < 0)
            or np.any(matrix.indices >= columns)
            or not np.all(np.isfinite(matrix.data))
        ):
            raise OperatorHZBuildError(
                f"owned SparseHZ {name} is not canonical CSR"
            )
        if matrix.indices.size > 1:
            adjacent = matrix.indices[1:] <= matrix.indices[:-1]
            row_boundaries = matrix.indptr[1:-1]
            if row_boundaries.size:
                real_boundaries = row_boundaries[
                    (row_boundaries > 0)
                    & (row_boundaries < matrix.indices.size)
                ]
                adjacent[real_boundaries - 1] = False
            if np.any(adjacent):
                raise OperatorHZBuildError(
                    f"owned SparseHZ {name} has unsorted/duplicate columns"
                )
    for name in ("col_ids", "bcol_ids"):
        values = normalized_dense[name]
        if np.unique(values).size != values.size:
            raise OperatorHZBuildError(
                f"owned SparseHZ {name} are not unique"
            )

    result = object.__new__(SparseHZono)
    for name, value in normalized_dense.items():
        object.__setattr__(result, name, value)
    for name, value in sparse.items():
        object.__setattr__(result, name, value)
    return result


def _enclosing_center_radius(
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stored-float ``c, r`` whose real interval encloses ``[lb, ub]``.

    Computing ``r = (ub - lb) / 2`` independently from a rounded center can
    make ``c-r`` or ``c+r`` fall just inside an endpoint.  We compute the
    radius required by the *stored* center in extended precision and round it
    upward until both endpoint inequalities hold.  Point dimensions remain
    exact and allocate no generator.
    """

    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    if lb.size != ub.size or np.any(lb > ub):
        raise OperatorHZBuildError(f"{name} has malformed bounds")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise OperatorHZBuildError(f"{name} has NaN or infinity")

    center = 0.5 * lb + 0.5 * ub
    point = lb == ub
    center[point] = lb[point]
    if not np.all(np.isfinite(center)):
        raise OperatorHZBuildError(f"{name} center overflowed")

    # Binary64 addition/subtraction are correctly rounded.  One successor of
    # each rounded distance is therefore an upper bound on the corresponding
    # exact-real distance between the stored endpoints and stored center.  This
    # avoids depending on platform-specific ``longdouble`` precision.
    lower_distance = center - lb
    upper_distance = ub - center
    nonpoint = ~point
    lower_distance[nonpoint] = np.nextafter(
        lower_distance[nonpoint], np.inf
    )
    upper_distance[nonpoint] = np.nextafter(
        upper_distance[nonpoint], np.inf
    )
    radius = np.maximum(lower_distance, upper_distance)
    radius = np.maximum(radius, 0.0)
    radius[point] = 0.0
    if not np.all(np.isfinite(radius)):
        raise OperatorHZBuildError(f"{name} radius overflowed")
    if np.any(radius < 0.0):
        raise OperatorHZBuildError(f"{name} has a negative radius")
    return center, radius


def _longdouble_to_f64_upper(value: np.longdouble, *, name: str) -> float:
    """Round a finite authoritative long-double upper bound toward ``+inf``."""

    exact_upper = np.asarray(value, dtype=np.longdouble).reshape(())
    if not np.isfinite(exact_upper):
        raise OperatorHZBuildError(f"{name} is not finite")
    rounded = float(exact_upper)
    if not np.isfinite(rounded):
        raise OperatorHZBuildError(f"{name} overflows binary64")
    if np.longdouble(rounded) < exact_upper:
        rounded = float(np.nextafter(rounded, np.inf))
    if not np.isfinite(rounded) or np.longdouble(rounded) < exact_upper:
        raise OperatorHZBuildError(f"{name} could not be rounded outward")
    return rounded


def _independent_preactivation_lagrangian_upper(
    expr: _AffineExpr,
    row: int,
    *,
    sign: float,
    base: _PreactivationLPBase,
    row_dual: np.ndarray,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Check a candidate upper bound for ``sign * expr[row]`` from scratch.

    ``expr`` denotes a semantic row

    ``c + G @ xi_c + delta, |delta| <= err``.

    The objective therefore includes ``+err`` for either sign.  Binary frame
    columns are appended with zero objective coefficients and relaxed to their
    enclosing box.  This is sound because the true ``{-1,+1}`` set is a
    subset of that box.  The solver is used only to suggest ``row_dual``; the
    imported checker reconstructs a Lagrangian support bound in the original
    stored-float coordinates with explicit long-double roundoff guards.
    """

    row = int(row)
    direction = float(sign)
    receipt: Dict[str, Any] = {
        "schema": "operator_hz_preactivation_lagrangian_v1",
        "row": row,
        "sign": direction,
        "status": "not_started",
        "bound": None,
        "semantic_error": None,
        "binary_relaxation": bool(base.lb.size > expr.G.shape[1]),
        "proof_authority": False,
    }
    if row < 0 or row >= expr.size:
        receipt["status"] = "invalid:row_out_of_range"
        return None, receipt
    if direction not in {-1.0, 1.0}:
        receipt["status"] = "invalid:sign"
        return None, receipt
    try:
        Gc = _require_canonical_csr(
            expr.G.getrow(row), name="preactivation objective"
        )
        n_frame = int(base.lb.size)
        if Gc.shape[1] > n_frame:
            raise OperatorHZBuildError(
                "preactivation objective is wider than its constraint frame"
            )
        if Gc.shape[1] < n_frame:
            Gc = _pad_cols(Gc, n_frame)
        if direction < 0.0:
            Gc = (-Gc).tocsr()
            Gc.eliminate_zeros()
        semantic_error = float(expr.err[row])
        if not np.isfinite(semantic_error) or semantic_error < 0.0:
            raise OperatorHZBuildError(
                "preactivation semantic error is invalid"
            )
        receipt["semantic_error"] = semantic_error
        checked, checked_receipt = _hz_independent_lp_lagrangian_upper(
            c=np.asarray([direction * float(expr.c[row])], dtype=np.float64),
            Gc=Gc,
            C_row=np.asarray([1.0], dtype=np.float64),
            # c + q@v - threshold = c + q@v + err.
            threshold=-semantic_error,
            A=base.A,
            rl=base.rl,
            ru=base.ru,
            lb=base.lb,
            ub=base.ub,
            row_dual=np.asarray(row_dual, dtype=np.float64).reshape(-1),
        )
        receipt["certificate"] = checked_receipt
        if (
            checked is None
            or checked_receipt.get("status") != "verified_upper"
        ):
            receipt["status"] = "certificate_rejected"
            return None, receipt
        outward = _longdouble_to_f64_upper(
            checked, name="certified preactivation upper"
        )
        receipt.update(
            {
                "status": "verified_upper",
                "bound": outward,
                "proof_authority": True,
            }
        )
        return outward, receipt
    except Exception as exc:
        receipt["status"] = (
            f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        )
        return None, receipt


class _OperatorHZBuilder:
    def __init__(
        self,
        net: Net,
        before: Mapping[int, Fact],
        after: Mapping[int, Fact],
        *,
        exact_budget: int,
        materialize_add: bool,
        preactivation_lp_budget: int,
        preactivation_lp_time_limit: float,
        preactivation_targets: Optional[Any] = None,
        correlation_targets: Optional[Any] = None,
        residual_phase_screen: bool = False,
        residual_bound_screen: bool = False,
        residual_targets: Optional[Any] = None,
        exact_target_reservoir: Optional[Any] = None,
        export_verified_preactivation_frame: bool = True,
        property_phase_focus_rivals: Optional[Any] = None,
        property_micro_rlt_product_cap: int = 0,
        property_micro_rlt_packet_mode: str = "both",
        property_upper_C: Optional[Any] = None,
        property_upper_thresholds: Optional[Any] = None,
        property_tail_add_source_planes: bool = False,
        property_tail_alpha_steps: int = 0,
        property_tail_alpha_time_limit: float = 0.0,
        property_tail_alpha_learning_rate: float = 0.08,
        property_tail_alpha_max_cells: int = 50_000_000,
        property_tail_alpha_device: str = "auto",
        property_tail_pairhull_budget: int = 0,
        property_tail_pairhull_time_limit: float = 0.0,
        property_tail_suffix_blocks: int = 0,
        property_tail_suffix_alpha_steps: int = 0,
        property_tail_suffix_alpha_time_limit: float = 0.0,
        property_tail_suffix_alpha_device: str = "auto",
        verified_query_dual_feedback: Optional[Any] = None,
        issue_constructive_nonempty_seal: bool = False,
        deadline: Optional[float],
    ) -> None:
        if deadline is not None and not math.isfinite(float(deadline)):
            raise OperatorHZBuildError("operator-HZ deadline must be finite")
        self.deadline = None if deadline is None else float(deadline)
        self._check_deadline("constructor_entry")
        if type(issue_constructive_nonempty_seal) is not bool:
            raise OperatorHZBuildError(
                "issue_constructive_nonempty_seal must be a bool"
            )
        self.issue_constructive_nonempty_seal = (
            issue_constructive_nonempty_seal
        )
        if int(exact_budget) < -1:
            raise OperatorHZBuildError(
                f"exact_budget must be -1, 0, or a positive integer; got {exact_budget}"
            )
        self.net = net
        self.before = before
        self.after = after
        self.exact_budget = int(exact_budget)
        self.materialize_add = bool(materialize_add)
        if int(preactivation_lp_budget) < 0:
            raise OperatorHZBuildError(
                "preactivation_lp_budget must be a nonnegative integer"
            )
        if (
            not math.isfinite(float(preactivation_lp_time_limit))
            or float(preactivation_lp_time_limit) < 0.0
        ):
            raise OperatorHZBuildError(
                "preactivation_lp_time_limit must be finite and nonnegative"
            )
        self.preactivation_lp_budget = int(preactivation_lp_budget)
        self.preactivation_lp_used = 0
        self.preactivation_lp_time_limit = float(
            preactivation_lp_time_limit
        )
        self.preactivation_targets = _normalize_preactivation_targets(
            preactivation_targets
        )
        self.correlation_targets = _normalize_preactivation_targets(
            correlation_targets
        )
        self.residual_phase_screen = bool(residual_phase_screen)
        self.residual_bound_screen = bool(residual_bound_screen)
        self.residual_targets = _normalize_residual_targets(residual_targets)
        self.exact_target_reservoir = _normalize_exact_target_reservoir(
            exact_target_reservoir
        )
        if type(export_verified_preactivation_frame) is not bool:
            raise OperatorHZBuildError(
                "export_verified_preactivation_frame must be a bool"
            )
        self.export_verified_preactivation_frame = bool(
            export_verified_preactivation_frame
        )
        if self.exact_target_reservoir is not None:
            if not self.residual_bound_screen:
                raise OperatorHZBuildError(
                    "exact_target_reservoir requires residual_bound_screen"
                )
            if self.residual_phase_screen:
                raise OperatorHZBuildError(
                    "exact_target_reservoir forbids simultaneous phase-only "
                    "screen mode"
                )
            if self.exact_budget <= 0:
                raise OperatorHZBuildError(
                    "exact_target_reservoir requires a positive exact_budget"
                )
            if not self.residual_targets:
                raise OperatorHZBuildError(
                    "exact_target_reservoir requires primary residual_targets"
                )
            primary_count = sum(
                len(rows) for rows in self.residual_targets.values()
            )
            if primary_count != self.exact_budget:
                raise OperatorHZBuildError(
                    "exact_target_reservoir requires exactly exact_budget "
                    "primary residual targets"
                )
            primary_coordinates = {
                (int(layer_id), int(row))
                for layer_id, rows in self.residual_targets.items()
                for row in rows
            }
            reserve_coordinates = {
                (int(layer_id), int(row))
                for layer_id, rows in self.exact_target_reservoir.items()
                for row in rows
            }
            if not reserve_coordinates:
                raise OperatorHZBuildError(
                    "exact_target_reservoir must contain at least one backup"
                )
            if primary_coordinates.intersection(reserve_coordinates):
                raise OperatorHZBuildError(
                    "exact reservoir backups must be disjoint from primaries"
                )
            primary_layers = set(self.residual_targets)
            if not set(self.exact_target_reservoir).issubset(primary_layers):
                raise OperatorHZBuildError(
                    "exact reservoir backups must stay in primary layers"
                )
            for layer_id, reserve_rows in self.exact_target_reservoir.items():
                layer_primary_count = len(self.residual_targets[layer_id])
                if len(reserve_rows) > 3 * layer_primary_count:
                    raise OperatorHZBuildError(
                        "exact reservoir exceeds the per-layer "
                        "three-per-primary cap"
                    )
        if (
            isinstance(property_micro_rlt_product_cap, (bool, np.bool_))
            or not isinstance(
                property_micro_rlt_product_cap, (int, np.integer)
            )
            or not 0
            <= int(property_micro_rlt_product_cap)
            <= _PROPERTY_MICRO_RLT_PRODUCT_FACTOR_CAP_MAX
        ):
            raise OperatorHZBuildError(
                "property_micro_rlt_product_cap must be an integer in "
                "[0, 4096]"
            )
        self.property_micro_rlt_product_cap = int(
            property_micro_rlt_product_cap
        )
        if (
            not isinstance(property_micro_rlt_packet_mode, str)
            or property_micro_rlt_packet_mode
            not in {"both", "first", "second"}
        ):
            raise OperatorHZBuildError(
                "property_micro_rlt_packet_mode must be one of "
                "both|first|second"
            )
        if (
            self.property_micro_rlt_product_cap <= 0
            and property_micro_rlt_packet_mode != "both"
        ):
            raise OperatorHZBuildError(
                "property_micro_rlt_packet_mode first/second requires a "
                "positive product cap"
            )
        self.property_micro_rlt_packet_mode = str(
            property_micro_rlt_packet_mode
        )
        self.property_phase_focus_rivals: Dict[
            Tuple[int, int], Tuple[int, ...]
        ] = {}
        self.residual_target_layers_seen: set[int] = set()
        self.residual_target_receipts: List[Dict[str, Any]] = []
        self.exact_target_reservoir_receipts: List[Dict[str, Any]] = []
        if int(property_tail_alpha_steps) < 0:
            raise OperatorHZBuildError(
                "property_tail_alpha_steps must be nonnegative"
            )
        if (
            not math.isfinite(float(property_tail_alpha_time_limit))
            or float(property_tail_alpha_time_limit) < 0.0
        ):
            raise OperatorHZBuildError(
                "property_tail_alpha_time_limit must be finite and "
                "nonnegative"
            )
        if (
            not math.isfinite(float(property_tail_alpha_learning_rate))
            or float(property_tail_alpha_learning_rate) <= 0.0
        ):
            raise OperatorHZBuildError(
                "property_tail_alpha_learning_rate must be finite and "
                "positive"
            )
        if int(property_tail_alpha_max_cells) <= 0:
            raise OperatorHZBuildError(
                "property_tail_alpha_max_cells must be positive"
            )
        self.property_tail_alpha_steps = int(property_tail_alpha_steps)
        self.property_tail_alpha_time_limit = float(
            property_tail_alpha_time_limit
        )
        if (
            self.property_tail_alpha_steps > 0
        ) != (self.property_tail_alpha_time_limit > 0.0):
            raise OperatorHZBuildError(
                "property-tail alpha steps and time limit must be enabled "
                "together"
            )
        self.property_tail_alpha_learning_rate = float(
            property_tail_alpha_learning_rate
        )
        self.property_tail_alpha_max_cells = int(
            property_tail_alpha_max_cells
        )
        self.property_tail_alpha_device = str(
            property_tail_alpha_device
        ).lower()
        if self.property_tail_alpha_device not in {
            "auto",
            "cpu",
            "cuda",
        }:
            raise OperatorHZBuildError(
                "property_tail_alpha_device must be auto, cpu, or cuda"
            )
        if (property_upper_C is None) != (property_upper_thresholds is None):
            raise OperatorHZBuildError(
                "property upper C and thresholds must be supplied together"
            )
        if property_upper_C is None:
            self.property_upper_C = None
            self.property_upper_thresholds = None
        else:
            self.property_upper_C = np.asarray(
                property_upper_C, dtype=np.float64
            )
            self.property_upper_thresholds = np.asarray(
                property_upper_thresholds, dtype=np.float64
            ).reshape(-1)
            if (
                self.property_upper_C.ndim != 2
                or self.property_upper_C.shape[0]
                != self.property_upper_thresholds.size
                or self.property_upper_C.shape[0] == 0
                or not np.all(np.isfinite(self.property_upper_C))
                or not np.all(np.isfinite(self.property_upper_thresholds))
            ):
                raise OperatorHZBuildError(
                    "property upper rows are malformed or non-finite"
                )
            self.property_upper_C = np.ascontiguousarray(
                self.property_upper_C, dtype=np.float64
            )
            self.property_upper_thresholds = np.ascontiguousarray(
                self.property_upper_thresholds, dtype=np.float64
            )
            if self.residual_targets is not None:
                target_count = sum(
                    len(rows) for rows in self.residual_targets.values()
                )
                if (
                    self.exact_budget <= 0
                    or target_count <= 0
                    or target_count > self.exact_budget
                ):
                    raise OperatorHZBuildError(
                        "property upper tail accepts residual targets only "
                        "as a bounded exact phase-cover schedule with "
                        "0 < target_count <= exact_budget"
                    )
        if (
            property_phase_focus_rivals is not None
            and (
                self.property_upper_C is not None
                or self.property_micro_rlt_product_cap > 0
            )
        ):
            if not isinstance(property_phase_focus_rivals, Mapping):
                raise OperatorHZBuildError(
                    "property phase focus rivals must be a mapping"
                )
            for raw_key, raw_rivals in (
                property_phase_focus_rivals.items()
            ):
                if (
                    not isinstance(raw_key, (tuple, list))
                    or len(raw_key) != 2
                ):
                    raise OperatorHZBuildError(
                        "property phase focus key must be (layer, row)"
                    )
                key = (int(raw_key[0]), int(raw_key[1]))
                if (
                    self.residual_targets is None
                    or key[0] not in self.residual_targets
                    or key[1] not in self.residual_targets[key[0]]
                ):
                    raise OperatorHZBuildError(
                        "property phase focus must bind an exact target"
                    )
                values = (
                    (raw_rivals,)
                    if isinstance(raw_rivals, (int, np.integer))
                    else tuple(raw_rivals)
                )
                rivals = tuple(
                    dict.fromkeys(int(value) for value in values)
                )
                if not rivals or min(rivals) < 0:
                    raise OperatorHZBuildError(
                        "property phase focus rival is out of range"
                    )
                if (
                    self.property_upper_C is not None
                    and max(rivals) >= self.property_upper_C.shape[0]
                ):
                    raise OperatorHZBuildError(
                        "property phase focus rival is out of range"
                    )
                self.property_phase_focus_rivals[key] = rivals
        self.verified_query_dual_feedback = None
        self.verified_query_dual_target_ids: Tuple[int, ...] = ()
        self.verified_query_dual_bounds: Dict[
            int, Tuple[np.ndarray, np.ndarray]
        ] = {}
        self.verified_query_dual_property_upper: Optional[np.ndarray] = None
        self.verified_query_dual_receipt: Optional[Dict[str, str]] = None
        self.verified_query_dual_consume_seconds = 0.0
        if verified_query_dual_feedback is not None:
            query_dual_consume_started = time.monotonic()
            self._check_deadline("query_dual_validation_before")
            if (
                self.property_upper_C is None
                or self.property_upper_thresholds is None
            ):
                raise OperatorHZBuildError(
                    "verified query-dual feedback requires property upper C "
                    "and thresholds"
                )
            # The authority validator and every later consumer must see the
            # same property operands even if a caller retains writable NumPy
            # aliases.  Only the authority-bearing path pays for this copy.
            self.property_upper_C = np.ascontiguousarray(
                self.property_upper_C, dtype=np.float64
            ).copy()
            self.property_upper_thresholds = np.ascontiguousarray(
                self.property_upper_thresholds, dtype=np.float64
            ).copy()
            self.property_upper_C.setflags(write=False)
            self.property_upper_thresholds.setflags(write=False)
            try:
                from act.back_end.hybridz_tf.query_dual_pipeline import (
                    validate_verified_query_dual_feedback,
                )

                requested_targets = tuple(
                    int(value)
                    for value in verified_query_dual_feedback.target_relu_ids
                )
                valid_feedback = validate_verified_query_dual_feedback(
                    verified_query_dual_feedback,
                    net=self.net,
                    property_rows=self.property_upper_C,
                    thresholds=self.property_upper_thresholds,
                    expected_target_relu_ids=requested_targets,
                )
            except Exception as exc:
                raise OperatorHZBuildError(
                    "verified query-dual feedback validation raised "
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                ) from exc
            if valid_feedback is not True:
                raise OperatorHZBuildError(
                    "verified query-dual feedback failed its process-local "
                    "transaction validator"
                )
            self._check_deadline("query_dual_validation_after")
            if (
                verified_query_dual_feedback.proof_authority is not True
                or len(set(requested_targets)) != len(requested_targets)
            ):
                raise OperatorHZBuildError(
                    "verified query-dual feedback has invalid authority or "
                    "target ordering"
                )
            receipt = verified_query_dual_feedback.receipt
            if not isinstance(receipt, Mapping):
                raise OperatorHZBuildError(
                    "verified query-dual feedback receipt is not a mapping"
                )
            required_hashes = (
                "receipt_sha256",
                "root_boxes_sha256",
                "final_boxes_sha256",
                "property_spec_sha256",
                "property_upper_sha256",
            )
            if any(
                not isinstance(receipt.get(key), str)
                or len(receipt[key]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in receipt[key]
                )
                for key in required_hashes
            ):
                raise OperatorHZBuildError(
                    "verified query-dual feedback receipt hashes are malformed"
                )
            try:
                frozen_receipt = {
                    key: str(receipt[key]) for key in required_hashes
                }
            except (KeyError, TypeError) as exc:
                raise OperatorHZBuildError(
                    "verified query-dual feedback receipt changed during "
                    "validation"
                ) from exc
            frozen_bounds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            try:
                for layer_id in requested_targets:
                    self._check_deadline(
                        f"query_dual_snapshot_layer_{int(layer_id)}"
                    )
                    certified = (
                        verified_query_dual_feedback.certified_bounds[
                            int(layer_id)
                        ]
                    )
                    lower = np.ascontiguousarray(
                        _as_finite_vector(
                            certified.lb,
                            name=(
                                "verified_query_dual"
                                f"[{int(layer_id)}].lb"
                            ),
                        ),
                        dtype=np.float64,
                    ).copy()
                    upper = np.ascontiguousarray(
                        _as_finite_vector(
                            certified.ub,
                            name=(
                                "verified_query_dual"
                                f"[{int(layer_id)}].ub"
                            ),
                        ),
                        dtype=np.float64,
                    ).copy()
                    lower.setflags(write=False)
                    upper.setflags(write=False)
                    frozen_bounds[int(layer_id)] = (lower, upper)
                frozen_property_upper = np.ascontiguousarray(
                    np.asarray(
                        verified_query_dual_feedback.property_upper,
                        dtype=np.float64,
                    ).reshape(-1)
                ).copy()
            except Exception as exc:
                if isinstance(exc, OperatorHZBuildTimeout):
                    raise
                raise OperatorHZBuildError(
                    "verified query-dual feedback snapshot failed: "
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                ) from exc
            if (
                frozen_property_upper.shape
                != (self.property_upper_C.shape[0],)
                or not np.all(np.isfinite(frozen_property_upper))
            ):
                raise OperatorHZBuildError(
                    "verified query-dual property snapshot is malformed"
                )
            frozen_property_upper.setflags(write=False)
            # Validate once more after taking the private snapshot, then
            # compare its exact binary64 bits with the still-live authority.
            # This closes the public builder's validate/snapshot TOCTOU gap:
            # mutation during either validation or copying invalidates the
            # complete build, while mutation after this comparison is
            # irrelevant because construction consumes only private arrays.
            self._check_deadline("query_dual_snapshot_revalidation_before")
            try:
                valid_feedback_after_snapshot = (
                    validate_verified_query_dual_feedback(
                        verified_query_dual_feedback,
                        net=self.net,
                        property_rows=self.property_upper_C,
                        thresholds=self.property_upper_thresholds,
                        expected_target_relu_ids=requested_targets,
                    )
                )
            except Exception as exc:
                raise OperatorHZBuildError(
                    "verified query-dual feedback post-snapshot validation "
                    f"raised {type(exc).__name__}: {str(exc)[:500]}"
                ) from exc
            if valid_feedback_after_snapshot is not True:
                raise OperatorHZBuildError(
                    "verified query-dual feedback changed while taking the "
                    "private snapshot"
                )
            for layer_id in requested_targets:
                live_certified = (
                    verified_query_dual_feedback.certified_bounds[
                        int(layer_id)
                    ]
                )
                live_lower = _as_finite_vector(
                    live_certified.lb,
                    name=f"verified_query_dual[{int(layer_id)}].live_lb",
                )
                live_upper = _as_finite_vector(
                    live_certified.ub,
                    name=f"verified_query_dual[{int(layer_id)}].live_ub",
                )
                frozen_lower, frozen_upper = frozen_bounds[int(layer_id)]
                if (
                    _f64_array_sha256(live_lower)
                    != _f64_array_sha256(frozen_lower)
                    or _f64_array_sha256(live_upper)
                    != _f64_array_sha256(frozen_upper)
                ):
                    raise OperatorHZBuildError(
                        "verified query-dual feedback bounds changed while "
                        f"taking the private snapshot at layer {int(layer_id)}"
                    )
            live_property_upper = _as_finite_vector(
                verified_query_dual_feedback.property_upper,
                name="verified_query_dual.live_property_upper",
            )
            if _f64_array_sha256(
                live_property_upper
            ) != _f64_array_sha256(frozen_property_upper):
                raise OperatorHZBuildError(
                    "verified query-dual property changed while taking the "
                    "private snapshot"
                )
            if any(
                not hmac.compare_digest(
                    str(verified_query_dual_feedback.receipt.get(key, "")),
                    frozen_receipt[key],
                )
                for key in required_hashes
            ):
                raise OperatorHZBuildError(
                    "verified query-dual receipt changed while taking the "
                    "private snapshot"
                )
            self._check_deadline("query_dual_snapshot_revalidation_after")
            self.verified_query_dual_feedback = verified_query_dual_feedback
            self.verified_query_dual_target_ids = requested_targets
            self.verified_query_dual_bounds = frozen_bounds
            self.verified_query_dual_property_upper = (
                frozen_property_upper
            )
            self.verified_query_dual_receipt = {
                key: frozen_receipt[key] for key in required_hashes
            }
            self._check_deadline("query_dual_snapshot_after")
            self.verified_query_dual_consume_seconds = max(
                0.0, time.monotonic() - query_dual_consume_started
            )
        self.property_tail_add_source_planes = bool(
            property_tail_add_source_planes
        )
        if (
            self.property_tail_add_source_planes
            and self.property_upper_C is None
        ):
            raise OperatorHZBuildError(
                "property-tail ADD source planes require property upper rows"
            )
        if (
            self.property_tail_add_source_planes
            and not self.materialize_add
        ):
            raise OperatorHZBuildError(
                "property-tail ADD source planes require materialize_add=True"
            )
        if (
            (
                self.property_tail_alpha_steps > 0
                or self.property_tail_alpha_time_limit > 0.0
            )
            and self.property_upper_C is None
        ):
            raise OperatorHZBuildError(
                "property-tail alpha candidates require property upper rows"
            )
        if (
            self.property_tail_alpha_steps > 0
            and self.exact_budget != 0
        ):
            raise OperatorHZBuildError(
                "property-tail alpha candidates currently require "
                "exact_budget=0"
            )
        if (
            isinstance(property_tail_pairhull_budget, (bool, np.bool_))
            or not isinstance(
                property_tail_pairhull_budget, (int, np.integer)
            )
        ):
            raise OperatorHZBuildError(
                "property_tail_pairhull_budget must be an integer"
            )
        if not 0 <= int(property_tail_pairhull_budget) <= 8:
            raise OperatorHZBuildError(
                "property_tail_pairhull_budget must lie in [0, 8]"
            )
        if (
            isinstance(
                property_tail_pairhull_time_limit, (bool, np.bool_)
            )
            or not isinstance(
                property_tail_pairhull_time_limit,
                (int, float, np.integer, np.floating),
            )
        ):
            raise OperatorHZBuildError(
                "property_tail_pairhull_time_limit must be numeric"
            )
        pairhull_seconds = float(property_tail_pairhull_time_limit)
        if (
            not math.isfinite(pairhull_seconds)
            or not 0.0 <= pairhull_seconds <= 1.5
        ):
            raise OperatorHZBuildError(
                "property_tail_pairhull_time_limit must be finite and "
                "lie in [0, 1.5]"
            )
        self.property_tail_pairhull_budget = int(
            property_tail_pairhull_budget
        )
        self.property_tail_pairhull_time_limit = pairhull_seconds
        if (
            self.property_tail_pairhull_budget > 0
        ) != (self.property_tail_pairhull_time_limit > 0.0):
            raise OperatorHZBuildError(
                "property-tail PairHull budget and time limit must be "
                "enabled together"
            )
        if (
            self.property_tail_pairhull_budget > 0
            and self.property_upper_C is None
        ):
            raise OperatorHZBuildError(
                "property-tail PairHull candidates require property upper "
                "rows"
            )
        if (
            self.property_tail_pairhull_budget > 0
            and self.exact_budget != 0
        ):
            raise OperatorHZBuildError(
                "property-tail PairHull candidates currently require "
                "exact_budget=0"
            )
        if (
            isinstance(property_tail_suffix_blocks, (bool, np.bool_))
            or not isinstance(
                property_tail_suffix_blocks, (int, np.integer)
            )
            or not 0 <= int(property_tail_suffix_blocks) <= 8
        ):
            raise OperatorHZBuildError(
                "property_tail_suffix_blocks must be an integer in [0, 8]"
            )
        self.property_tail_suffix_blocks = int(
            property_tail_suffix_blocks
        )
        if (
            self.property_tail_suffix_blocks > 0
            and self.property_upper_C is None
        ):
            raise OperatorHZBuildError(
                "property-tail suffix replay requires property upper rows"
            )
        if (
            isinstance(property_tail_suffix_alpha_steps, (bool, np.bool_))
            or not isinstance(
                property_tail_suffix_alpha_steps, (int, np.integer)
            )
            or not 0 <= int(property_tail_suffix_alpha_steps) <= 64
        ):
            raise OperatorHZBuildError(
                "property_tail_suffix_alpha_steps must be an integer "
                "in [0, 64]"
            )
        suffix_alpha_seconds = float(
            property_tail_suffix_alpha_time_limit
        )
        if (
            not math.isfinite(suffix_alpha_seconds)
            or not 0.0 <= suffix_alpha_seconds <= 20.0
        ):
            raise OperatorHZBuildError(
                "property_tail_suffix_alpha_time_limit must be finite and "
                "lie in [0, 20]"
            )
        self.property_tail_suffix_alpha_steps = int(
            property_tail_suffix_alpha_steps
        )
        self.property_tail_suffix_alpha_time_limit = (
            suffix_alpha_seconds
        )
        if (
            self.property_tail_suffix_alpha_steps > 0
        ) != (suffix_alpha_seconds > 0.0):
            raise OperatorHZBuildError(
                "property-tail suffix alpha steps and time limit must be "
                "enabled together"
            )
        if (
            self.property_tail_suffix_alpha_steps > 0
            and self.property_tail_suffix_blocks <= 0
        ):
            raise OperatorHZBuildError(
                "property-tail suffix alpha requires suffix replay"
            )
        self.property_tail_suffix_alpha_device = str(
            property_tail_suffix_alpha_device
        ).lower()
        if self.property_tail_suffix_alpha_device not in {
            "auto",
            "cpu",
            "cuda",
        }:
            raise OperatorHZBuildError(
                "property_tail_suffix_alpha_device must be auto, cpu, or cuda"
            )
        self.property_tail_relu_layer_id: Optional[int] = None
        self.property_tail_output_layer_id: Optional[int] = None
        self.property_tail_snapshot: Optional[_PropertyTailSnapshot] = None
        self.property_tail_add_source_layer_id: Optional[int] = None
        self.property_tail_add_source_bridge_layer_ids: Tuple[int, ...] = ()
        self.property_tail_add_source_snapshot: Optional[
            _MaterializedAddSourceSnapshot
        ] = None
        self.property_suffix_stop_layer_id: Optional[int] = None
        self.property_suffix_add_source_snapshot: Optional[
            _PropertySuffixAddSourceSnapshot
        ] = None
        self.property_full_input_replay_result: Optional[Any] = None
        self.property_tail_receipt: Optional[Dict[str, Any]] = None
        self.property_tail_row_groups: Tuple[Tuple[int, ...], ...] = ()
        target_count = (
            sum(len(rows) for rows in self.preactivation_targets.values())
            if self.preactivation_targets is not None
            else self.preactivation_lp_budget
        )
        self.preactivation_lp_direction_capacity = 2 * min(
            self.preactivation_lp_budget,
            int(target_count),
        )
        self.preactivation_lp_directions_used = 0
        self.preactivation_lp_started_at: Optional[float] = None
        self.preactivation_lp_deadline: Optional[float] = None
        self.preactivation_lp_time_spent = 0.0
        self.preactivation_lp_snapshot_seconds = 0.0
        self.preactivation_lp_candidate_seconds = 0.0
        self.preactivation_lp_certificate_seconds = 0.0
        self.preactivation_lp_model_builds = 0
        self.preactivation_lp_deadline_stage: Optional[str] = None
        self.preactivation_layers_attempted: set[int] = set()
        self.exprs: Dict[int, _AffineExpr] = {}
        self.eq_blocks: List[_ConstraintBlock] = []
        self.ub_blocks: List[_ConstraintBlock] = []
        self.layer_frame_snapshots: Dict[int, _LayerFrameSnapshot] = {}
        self.layer_metadata: List[Dict[str, Any]] = []
        self.materialization_events: List[Dict[str, Any]] = []
        self._layer_by_id: Dict[int, Any] = {}
        self._successors: Dict[int, List[int]] = {}
        self._topological_ids: Tuple[int, ...] = ()
        self.live_affine_fusion_started_at: Optional[float] = None
        self.live_affine_fusion_elapsed_seconds = 0.0
        self.live_affine_fusion_attempts: List[Dict[str, Any]] = []
        self.projection_skip_chain_preservations: List[
            Dict[str, Any]
        ] = []
        self.projection_skip_required_downstream: Dict[int, int] = {}
        # Property-conditioned row-local correlation shadows.  A shadow keeps
        # the pre-materialization ADD expression only for an explicitly
        # targeted downstream ReLU row.  It supplies a second independently
        # outward enclosure for that row; the ordinary materialized graph and
        # all of its equality bands remain unchanged.
        self.correlation_add_sources: Dict[int, _AffineExpr] = {}
        self.correlation_relu_shadows: Dict[
            int, Tuple[Tuple[int, ...], _AffineExpr]
        ] = {}
        self.correlation_shadow_receipts: List[Dict[str, Any]] = []
        self.residual_phase_screen_bounds: Dict[
            int, Tuple[Tuple[int, ...], np.ndarray, np.ndarray]
        ] = {}
        self.residual_phase_screen_receipts: List[Dict[str, Any]] = []
        self.verified_preactivation_bounds: Dict[
            int, Tuple[np.ndarray, np.ndarray]
        ] = {}
        # Alternative pre-materialization expressions for residual skip
        # fanout.  Nonlinear main branches continue to use their ordinary
        # materialized ReLU variables; only a later ADD's skip operand reuses
        # the prior ADD shadow, so correlation can cross residual depths
        # without replacing any authoritative graph node.
        self.residual_skip_shadows: Dict[int, _AffineExpr] = {}

        self.n_cont = 0
        self.n_bin = 0
        self.col_ids: List[int] = []
        self.bcol_ids: List[int] = []
        self.cont_column_layer_by_id: Dict[int, int] = {}
        self._allocation_layer_id: Optional[int] = None
        self.input_col_ids: Optional[np.ndarray] = None
        self.input_lb: Optional[np.ndarray] = None
        self.input_ub: Optional[np.ndarray] = None
        self.input_center: Optional[np.ndarray] = None
        self.input_radius: Optional[np.ndarray] = None
        self.input_layer_id: Optional[int] = None
        self.exact_used = 0
        # Exact property-selected ReLUs may later support branch-conditioned
        # suffix planes.  Records are private builder state until both the
        # exact binary graph and an independently replayed conditional plane
        # have been constructed.
        # Detailed per-row exact-phase dictionaries are consumed only by the
        # optional property suffix and micro-RLT paths.  The ordinary exact
        # forward path previously built one dictionary per unstable ReLU and
        # then used only ``len(...)``.  Keep the scalar accounting universally,
        # but do not allocate diagnostic/experimental records on the common
        # CIFAR/TinyImageNet forward path.
        self.exact_phase_record_count = 0
        self.collect_property_exact_phase_records = bool(
            self.property_upper_C is not None
            or self.property_micro_rlt_product_cap > 0
        )
        self.property_exact_phase_records: List[Dict[str, Any]] = []
        self.property_conditional_suffix_rows: List[Dict[str, Any]] = []
        self._constraint_program_sink: Optional[
            _OperatorConstraintProgramSink
        ] = None

    def _check_deadline(self, stage: str) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise OperatorHZBuildTimeout(
                f"shared HybridZ deadline expired during {stage}"
            )

    def _initialize_constraint_program_sink(
        self,
        order: Sequence[Any],
    ) -> None:
        """Run the complete Phase-B allowlist before creating an owner."""

        enabled = _EXPERIMENTAL_CONSTRAINT_PROGRAM_SINK
        if type(enabled) is not bool:
            raise OperatorHZBuildError(
                "internal constraint-program sink switch must be a bool"
            )
        if not enabled:
            return

        unsupported_layers = tuple(
            (int(layer.id), _kind(layer.kind))
            for layer in order
            if _kind(layer.kind) not in _SUPPORTED_KINDS
        )
        input_count = sum(_kind(layer.kind) == "INPUT" for layer in order)
        assert_count = sum(_kind(layer.kind) == "ASSERT" for layer in order)
        reasons: List[str] = []
        if unsupported_layers:
            reasons.append(f"unsupported_layers={unsupported_layers}")
        if input_count != 1 or assert_count != 1:
            reasons.append(
                f"INPUT/ASSERT counts are {input_count}/{assert_count}"
            )
        if self.exact_budget != -1:
            reasons.append("exact_budget must be exactly -1")
        # These paths consume, slice, prune, or extend the mutable legacy UB
        # block list.  Phase B replaces that list with a streamed source
        # program, so every such mode must fail before adapter/owner creation.
        if self.preactivation_lp_budget != 0:
            reasons.append("preactivation LP constraint snapshots are unsupported")
        if (
            self.property_upper_C is not None
            or self.property_upper_thresholds is not None
            or self.property_tail_add_source_planes
            or self.property_tail_alpha_steps != 0
            or self.property_tail_alpha_time_limit != 0.0
            or self.property_tail_pairhull_budget != 0
            or self.property_tail_pairhull_time_limit != 0.0
            or self.property_tail_suffix_blocks != 0
            or self.property_tail_suffix_alpha_steps != 0
            or self.property_tail_suffix_alpha_time_limit != 0.0
            or self.verified_query_dual_feedback is not None
        ):
            reasons.append("property-tail/pruning consumers are unsupported")
        if self.property_micro_rlt_product_cap != 0:
            reasons.append("property micro-RLT block consumption is unsupported")
        if self.eq_blocks or self.ub_blocks:
            reasons.append("constraint sink was initialized after row mutation")
        if reasons:
            raise OperatorHZBuildError(
                "constraint-program sink preflight rejected the build: "
                + "; ".join(reasons)
            )

        sink = _OperatorConstraintProgramSink()
        # Publish the cleanup root before lazy import/bind/owner/arena work.
        self._constraint_program_sink = sink
        sink.initialize()

    def _discard_open_constraint_program_sink(
        self,
    ) -> Optional[BaseException]:
        sink = self._constraint_program_sink
        if sink is None:
            return None
        return sink.discard_open()

    def _eq_row_count(self) -> int:
        return int(sum(block.Ac.shape[0] for block in self.eq_blocks))

    def _ub_row_count(self) -> int:
        sink = self._constraint_program_sink
        if sink is not None:
            return int(sink.virtual_rows)
        return int(sum(block.Ac.shape[0] for block in self.ub_blocks))

    def _eq_block_count(self) -> int:
        return int(len(self.eq_blocks))

    def _ub_block_count(self) -> int:
        sink = self._constraint_program_sink
        if sink is not None:
            return int(len(sink.legacy_tag_rows))
        return int(len(self.ub_blocks))

    # ------------------------------------------------------------------
    # Graph and fact validation
    # ------------------------------------------------------------------

    def _topological_layers(self) -> List[Any]:
        layers = list(self.net.layers)
        if not layers:
            raise OperatorHZBuildError("network has no layers")
        by_id = {int(layer.id): layer for layer in layers}
        if len(by_id) != len(layers):
            raise OperatorHZBuildError("network layer ids are not unique")
        position = {int(layer.id): i for i, layer in enumerate(layers)}

        successors: Dict[int, List[int]] = {lid: [] for lid in by_id}
        indegree: Dict[int, int] = {}
        for layer in layers:
            lid = int(layer.id)
            preds = [int(pid) for pid in self.net.preds.get(lid, [])]
            if len(set(preds)) != len(preds):
                raise OperatorHZBuildError(f"layer {lid} repeats a predecessor")
            for pid in preds:
                if pid not in by_id:
                    raise OperatorHZBuildError(
                        f"layer {lid} references missing predecessor {pid}"
                    )
                successors[pid].append(lid)
            indegree[lid] = len(preds)

        ready = [
            (position[lid], lid)
            for lid, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        order: List[Any] = []
        while ready:
            _, lid = heapq.heappop(ready)
            order.append(by_id[lid])
            for sid in sorted(successors[lid], key=position.__getitem__):
                indegree[sid] -= 1
                if indegree[sid] == 0:
                    heapq.heappush(ready, (position[sid], sid))
        if len(order) != len(layers):
            cyclic = sorted(lid for lid, degree in indegree.items() if degree)
            raise OperatorHZBuildError(
                f"operator-HZ requires a DAG; cyclic/unresolved layers={cyclic}"
            )
        if self.verified_query_dual_feedback is not None:
            for layer_id in self.verified_query_dual_target_ids:
                layer = by_id.get(int(layer_id))
                if layer is None or _kind(layer.kind) != "RELU":
                    raise OperatorHZBuildError(
                        "verified query-dual target references a missing or "
                        f"non-ReLU layer {int(layer_id)}"
                    )
                if int(layer_id) not in self.verified_query_dual_bounds:
                    raise OperatorHZBuildError(
                        "verified query-dual target has no certified box at "
                        f"layer {int(layer_id)}"
                    )
        if self.residual_targets is not None:
            for layer_id, rows in self.residual_targets.items():
                layer = by_id.get(int(layer_id))
                if layer is None:
                    raise OperatorHZBuildError(
                        f"residual target references missing layer {layer_id}"
                    )
                if _kind(layer.kind) != "RELU":
                    raise OperatorHZBuildError(
                        f"residual target layer {layer_id} is "
                        f"{_kind(layer.kind)}, not RELU"
                    )
                width = len(layer.out_vars)
                bad = [int(row) for row in rows if int(row) >= width]
                if bad:
                    raise OperatorHZBuildError(
                        f"residual target layer {layer_id} rows {bad[:8]} "
                        f"exceed width {width}"
                    )
        if self.exact_target_reservoir is not None:
            for layer_id, rows in self.exact_target_reservoir.items():
                layer = by_id.get(int(layer_id))
                if layer is None:
                    raise OperatorHZBuildError(
                        "exact reservoir references missing layer "
                        f"{layer_id}"
                    )
                if _kind(layer.kind) != "RELU":
                    raise OperatorHZBuildError(
                        f"exact reservoir layer {layer_id} is "
                        f"{_kind(layer.kind)}, not RELU"
                    )
                width = len(layer.out_vars)
                bad = [int(row) for row in rows if int(row) >= width]
                if bad:
                    raise OperatorHZBuildError(
                        f"exact reservoir layer {layer_id} rows {bad[:8]} "
                        f"exceed width {width}"
                    )
        if self.correlation_targets is not None:
            for layer_id, rows in self.correlation_targets.items():
                layer = by_id.get(int(layer_id))
                if layer is None:
                    raise OperatorHZBuildError(
                        "correlation target references missing layer "
                        f"{layer_id}"
                    )
                if _kind(layer.kind) != "RELU":
                    raise OperatorHZBuildError(
                        f"correlation target layer {layer_id} is "
                        f"{_kind(layer.kind)}, not RELU"
                    )
                width = len(layer.out_vars)
                bad = [int(row) for row in rows if int(row) >= width]
                if bad:
                    raise OperatorHZBuildError(
                        f"correlation target layer {layer_id} rows {bad[:8]} "
                        f"exceed width {width}"
                    )
        if self.property_upper_C is not None:
            asserts = [
                layer for layer in order if _kind(layer.kind) == "ASSERT"
            ]
            if len(asserts) != 1:
                raise OperatorHZBuildError(
                    "property upper tail requires exactly one ASSERT"
                )
            assert_layer = asserts[0]
            assert_preds = [
                int(value)
                for value in self.net.preds.get(int(assert_layer.id), [])
            ]
            if len(assert_preds) != 1:
                raise OperatorHZBuildError(
                    "property upper ASSERT must have one predecessor"
                )
            output_layer = by_id[assert_preds[0]]
            output_preds = [
                int(value)
                for value in self.net.preds.get(int(output_layer.id), [])
            ]
            if _kind(output_layer.kind) != "DENSE" or len(output_preds) != 1:
                raise OperatorHZBuildError(
                    "property upper tail requires ASSERT <- DENSE <- RELU"
                )
            relu_layer = by_id[output_preds[0]]
            if _kind(relu_layer.kind) != "RELU":
                raise OperatorHZBuildError(
                    "property upper tail requires ASSERT <- DENSE <- RELU"
                )
            if (
                successors[int(relu_layer.id)] != [int(output_layer.id)]
                or successors[int(output_layer.id)] != [int(assert_layer.id)]
            ):
                raise OperatorHZBuildError(
                    "property upper tail requires an exclusive final "
                    "RELU -> DENSE -> ASSERT chain"
                )
            if self.property_upper_C.shape[1] != len(output_layer.out_vars):
                raise OperatorHZBuildError(
                    "property upper C width does not match final DENSE output"
                )
            self.property_tail_relu_layer_id = int(relu_layer.id)
            self.property_tail_output_layer_id = int(output_layer.id)
            if self.property_tail_add_source_planes:
                relu_preds = [
                    int(value)
                    for value in self.net.preds.get(int(relu_layer.id), [])
                ]
                if len(relu_preds) != 1:
                    raise OperatorHZBuildError(
                        "property-tail ADD source planes require the final "
                        "RELU to have exactly one predecessor"
                    )
                relu_pred = by_id[relu_preds[0]]
                bridge_layer_ids: Tuple[int, ...]
                if _kind(relu_pred.kind) == "ADD":
                    add_layer = relu_pred
                    bridge_layer_ids = ()
                elif _kind(relu_pred.kind) == "DENSE":
                    dense_layer = relu_pred
                    dense_preds = [
                        int(value)
                        for value in self.net.preds.get(
                            int(dense_layer.id), []
                        )
                    ]
                    if len(dense_preds) != 1:
                        raise OperatorHZBuildError(
                            "property-tail ADD source bridge DENSE must "
                            "have one predecessor"
                        )
                    flatten_layer = by_id[dense_preds[0]]
                    flatten_preds = [
                        int(value)
                        for value in self.net.preds.get(
                            int(flatten_layer.id), []
                        )
                    ]
                    if (
                        _kind(flatten_layer.kind) != "FLATTEN"
                        or len(flatten_preds) != 1
                    ):
                        raise OperatorHZBuildError(
                            "property-tail ADD source planes require "
                            "ADD -> FLATTEN -> DENSE -> final RELU"
                        )
                    add_layer = by_id[flatten_preds[0]]
                    bridge_layer_ids = (
                        int(flatten_layer.id),
                        int(dense_layer.id),
                    )
                    if (
                        len(flatten_layer.out_vars)
                        != len(add_layer.out_vars)
                        or len(dense_layer.out_vars)
                        != len(relu_layer.out_vars)
                    ):
                        raise OperatorHZBuildError(
                            "property-tail ADD source bridge width mismatch"
                        )
                else:
                    add_layer = relu_pred
                    bridge_layer_ids = ()
                bridge_chain = (
                    (int(add_layer.id), *bridge_layer_ids, int(relu_layer.id))
                )
                exclusive = all(
                    successors[int(left)] == [int(right)]
                    for left, right in zip(
                        bridge_chain[:-1], bridge_chain[1:]
                    )
                )
                if _kind(add_layer.kind) != "ADD" or not exclusive:
                    raise OperatorHZBuildError(
                        "property-tail ADD source planes require an exclusive "
                        "ADD -> final RELU edge or ADD -> FLATTEN -> DENSE -> "
                        "final RELU bridge"
                    )
                self.property_tail_add_source_layer_id = int(add_layer.id)
                self.property_tail_add_source_bridge_layer_ids = (
                    bridge_layer_ids
                )
        self._layer_by_id = by_id
        self._successors = successors
        self._topological_ids = tuple(int(layer.id) for layer in order)
        return order

    def _preds(self, layer: Any, expected: int) -> List[int]:
        preds = [int(pid) for pid in self.net.preds.get(int(layer.id), [])]
        if len(preds) != int(expected):
            raise OperatorHZBuildError(
                f"{_kind(layer.kind)} layer {layer.id} requires {expected} "
                f"predecessor(s), got {preds}"
            )
        missing = [pid for pid in preds if pid not in self.exprs]
        if missing:
            raise OperatorHZBuildError(
                f"layer {layer.id} predecessors have no operator expression: {missing}"
            )
        return preds


    def _fact_box(
        self,
        facts: Mapping[int, Fact],
        layer_id: int,
        expected_size: int,
        *,
        label: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if int(layer_id) not in facts:
            raise OperatorHZBuildError(f"missing {label} fact for layer {layer_id}")
        fact = facts[int(layer_id)]
        lb = _as_finite_vector(fact.bounds.lb, name=f"{label}[{layer_id}].lb")
        ub = _as_finite_vector(fact.bounds.ub, name=f"{label}[{layer_id}].ub")
        if lb.size != int(expected_size) or ub.size != int(expected_size):
            raise OperatorHZBuildError(
                f"{label} fact size mismatch at layer {layer_id}: "
                f"lb={lb.size}, ub={ub.size}, expected={expected_size}. "
                "The strict operator-HZ path currently supports one batch lane."
            )
        if np.any(lb > ub):
            bad = int(np.flatnonzero(lb > ub)[0])
            raise OperatorHZBuildError(
                f"invalid {label} bounds at layer {layer_id}, row {bad}: "
                f"{lb[bad]} > {ub[bad]}"
            )
        return lb, ub

    # ------------------------------------------------------------------
    # Sparse frame and expression algebra
    # ------------------------------------------------------------------

    @staticmethod
    def _fresh_ids(count: int) -> np.ndarray:
        ids = hz_fresh_col_ids(int(count), device="cpu")
        return ids.detach().cpu().numpy().astype(np.int64, copy=False)

    def _allocate_cont(
        self,
        count: int,
        *,
        layer_id: Optional[int] = None,
    ) -> np.ndarray:
        count = int(count)
        start = self.n_cont
        if count:
            sink = self._constraint_program_sink
            fresh = (
                self._fresh_ids(count)
                if sink is None
                else sink.allocate_continuous(count)
            )
            self.col_ids.extend(fresh.tolist())
            origin = (
                self._allocation_layer_id
                if layer_id is None
                else int(layer_id)
            )
            origin = -1 if origin is None else int(origin)
            for stable_id in fresh:
                self.cont_column_layer_by_id[int(stable_id)] = origin
            self.n_cont += count
        return np.arange(start, start + count, dtype=np.int64)

    def _allocate_bin(self, count: int) -> np.ndarray:
        count = int(count)
        start = self.n_bin
        if count:
            sink = self._constraint_program_sink
            fresh = (
                self._fresh_ids(count)
                if sink is None
                else sink.allocate_binary(count)
            )
            self.bcol_ids.extend(fresh.tolist())
            self.n_bin += count
        return np.arange(start, start + count, dtype=np.int64)

    def _align(self, expr: _AffineExpr) -> _AffineExpr:
        return _AffineExpr(
            c=expr.c,
            G=_pad_cols(expr.G, self.n_cont),
            err=expr.err,
            affine_depth=expr.affine_depth,
        )

    def _rows(self, expr: _AffineExpr, rows: Sequence[int]) -> _AffineExpr:
        idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= expr.size):
            raise OperatorHZBuildError("affine-expression row selection is out of range")
        return _AffineExpr(
            c=expr.c[idx],
            G=expr.G[idx, :].tocsr(),
            err=expr.err[idx],
            affine_depth=expr.affine_depth,
        )

    def _replace_rows(
        self,
        base: _AffineExpr,
        rows: Sequence[int],
        replacement: _AffineExpr,
    ) -> _AffineExpr:
        """Return ``base`` with named rows replaced in the current frame."""

        idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        if (
            idx.size != replacement.size
            or (idx.size and (
                int(idx.min()) < 0
                or int(idx.max()) >= base.size
                or np.unique(idx).size != idx.size
            ))
        ):
            raise OperatorHZBuildError(
                "affine-expression row replacement is malformed"
            )
        if not idx.size:
            return self._align(base)
        base_aligned = self._align(base)
        replacement_aligned = self._align(replacement)
        # Callers deliberately zero the base rows before replacement.  Assert
        # that invariant so an accidental additive overlay cannot duplicate a
        # local variable or silently change the represented graph.
        if (
            np.any(base_aligned.c[idx] != 0.0)
            or np.any(base_aligned.err[idx] != 0.0)
            or base_aligned.G[idx, :].nnz != 0
        ):
            raise OperatorHZBuildError(
                "affine-expression replacement base rows are not zero"
            )
        coo = replacement_aligned.G.tocoo(copy=False)
        embedded = sp.csr_matrix(
            (
                coo.data,
                (idx[np.asarray(coo.row, dtype=np.int64)], coo.col),
            ),
            shape=(base.size, self.n_cont),
            dtype=np.float64,
        )
        embedded.eliminate_zeros()
        c = base_aligned.c.copy()
        err = base_aligned.err.copy()
        c[idx] = replacement_aligned.c
        err[idx] = replacement_aligned.err
        G = (base_aligned.G + embedded).tocsr()
        G.eliminate_zeros()
        # Sparse addition does not promise canonical column order for a row
        # assembled from disjoint active/unstable phase slices.  The owned HZ
        # contract requires strictly increasing columns, so canonicalize once
        # at the row-replacement boundary where the mixed matrix is created.
        G.sort_indices()
        return _AffineExpr(
            c=c,
            G=G,
            err=err,
            affine_depth=max(
                base_aligned.affine_depth,
                replacement_aligned.affine_depth,
            ),
        )

    def _unmaterialized_add_origin(self, layer_id: int) -> Optional[int]:
        """Trace identity-only provenance back to an ADD.

        The candidate fusion is intentionally narrow: an unmaterialized ADD
        may pass through FLATTEN before the next affine operator, but no other
        kind is silently folded into this experiment.
        """

        current = int(layer_id)
        seen: set[int] = set()
        while current not in seen:
            seen.add(current)
            layer = self._layer_by_id.get(current)
            if layer is None:
                return None
            kind = _kind(layer.kind)
            if kind == "ADD":
                return current
            if kind != "FLATTEN":
                return None
            preds = [int(value) for value in self.net.preds.get(current, [])]
            if len(preds) != 1:
                return None
            current = preds[0]
        return None

    def _projection_skip_chain(
        self,
        *,
        pred: int,
        layer: Any,
    ) -> Optional[Dict[str, Any]]:
        """Recognize the common ResNet downsample projection exactly once.

        The source ADD feeds both ``Conv(stride=2) -> ReLU -> Conv`` and a
        1x1 stride-2 projection.  Both meet at the next ADD, whose ordinary
        main route is already consumed by the established live
        ``ADD -> Conv -> ReLU`` builder.  Keeping the source expression live
        through the projection removes the intervening equality band; it
        does not create another representation or perform a speculative
        build followed by fallback.
        """

        layer_id = int(layer.id)
        source = self._layer_by_id.get(int(pred))
        if (
            self.materialize_add
            or self.exact_budget != -1
            or source is None
            or _kind(source.kind) != "ADD"
            or _kind(layer.kind) != "CONV2D"
        ):
            return None

        try:
            weight_shape = tuple(
                int(value) for value in layer.params["weight"].shape
            )
            stride = _strict_pair(
                layer.params.get("stride", 1),
                name=f"projection_skip[{layer_id}].stride",
                positive=True,
            )
            padding = _strict_pair(
                layer.params.get("padding", 0),
                name=f"projection_skip[{layer_id}].padding",
                positive=False,
            )
            dilation = _strict_pair(
                layer.params.get("dilation", 1),
                name=f"projection_skip[{layer_id}].dilation",
                positive=True,
            )
            groups = int(layer.params.get("groups", 1))
        except (KeyError, AttributeError, TypeError, ValueError, OverflowError):
            return None
        if (
            len(weight_shape) != 4
            or weight_shape[2:] != (1, 1)
            or stride != (2, 2)
            or padding != (0, 0)
            or dilation != (1, 1)
            or groups != 1
        ):
            return None

        source_successors = list(self._successors.get(int(pred), ()))
        projection_successors = list(
            self._successors.get(layer_id, ())
        )
        if len(source_successors) != 2 or len(projection_successors) != 1:
            return None
        main_first_ids = [
            successor
            for successor in source_successors
            if int(successor) != layer_id
        ]
        if len(main_first_ids) != 1:
            return None
        main_first_id = int(main_first_ids[0])
        target_add_id = int(projection_successors[0])
        main_first = self._layer_by_id.get(main_first_id)
        target_add = self._layer_by_id.get(target_add_id)
        if (
            main_first is None
            or target_add is None
            or _kind(main_first.kind) != "CONV2D"
            or _kind(target_add.kind) != "ADD"
        ):
            return None

        first_successors = list(
            self._successors.get(main_first_id, ())
        )
        if len(first_successors) != 1:
            return None
        main_relu_id = int(first_successors[0])
        main_relu = self._layer_by_id.get(main_relu_id)
        if main_relu is None or _kind(main_relu.kind) != "RELU":
            return None
        relu_successors = list(self._successors.get(main_relu_id, ()))
        if len(relu_successors) != 1:
            return None
        main_second_id = int(relu_successors[0])
        main_second = self._layer_by_id.get(main_second_id)
        if (
            main_second is None
            or _kind(main_second.kind) != "CONV2D"
            or list(self._successors.get(main_second_id, ()))
            != [target_add_id]
        ):
            return None
        target_preds = [
            int(value) for value in self.net.preds.get(target_add_id, ())
        ]
        if len(target_preds) != 2 or set(target_preds) != {
            layer_id,
            main_second_id,
        }:
            return None

        downstream_routes: List[Tuple[int, int]] = []
        for successor in self._successors.get(target_add_id, ()):
            affine_id = int(successor)
            affine = self._layer_by_id.get(affine_id)
            affine_successors = list(
                self._successors.get(affine_id, ())
            )
            if (
                affine is None
                or _kind(affine.kind) != "CONV2D"
                or len(affine_successors) != 1
            ):
                continue
            relu_id = int(affine_successors[0])
            relu = self._layer_by_id.get(relu_id)
            if relu is not None and _kind(relu.kind) == "RELU":
                downstream_routes.append((affine_id, relu_id))
        if len(downstream_routes) != 1:
            return None

        downstream_affine_id, downstream_relu_id = downstream_routes[0]
        return {
            "schema": "operator_hz_projection_skip_chain_v1",
            "status": "applied",
            "source_add_layer_id": int(pred),
            "projection_layer_id": layer_id,
            "main_relu_layer_id": main_relu_id,
            "main_tail_layer_id": main_second_id,
            "target_add_layer_id": target_add_id,
            "downstream_affine_layer_id": downstream_affine_id,
            "downstream_relu_layer_id": downstream_relu_id,
            "kernel": [1, 1],
            "stride": [2, 2],
            "runtime_fallback": False,
            "second_representation_built": False,
            "soundness_contract": "existing_outward_affine_composition",
        }

    def _correlation_target_after_add(
        self, layer_id: int
    ) -> Optional[Tuple[Optional[int], int]]:
        """Return one targeted ``ADD -> [FLATTEN] -> affine -> RELU`` route.

        Other ADD successors may preserve residual skip fanout.  Exactly one
        targeted nonlinear route is required so a row cannot be silently
        redirected to another predicate.
        A direct ``ADD -> RELU`` route is represented by ``(None, relu_id)``.
        """

        if (
            self.correlation_targets is None
            and not self.residual_phase_screen
            and not self.residual_bound_screen
        ):
            return None
        routes: List[Tuple[Optional[int], int]] = []
        for raw_successor in self._successors.get(int(layer_id), ()):
            current = int(raw_successor)
            layer = self._layer_by_id.get(current)
            if layer is None:
                continue
            if _kind(layer.kind) == "RELU":
                if (
                    self.residual_phase_screen
                    or self.residual_bound_screen
                    or (
                        self.correlation_targets is not None
                        and self.correlation_targets.get(current, ())
                    )
                ):
                    routes.append((None, current))
                continue
            if _kind(layer.kind) == "FLATTEN":
                successors = list(self._successors.get(current, ()))
                if len(successors) != 1:
                    continue
                current = int(successors[0])
                layer = self._layer_by_id.get(current)
                if layer is None:
                    continue
            if _kind(layer.kind) not in {"DENSE", "CONV2D"}:
                continue
            affine_id = current
            successors = list(self._successors.get(affine_id, ()))
            if len(successors) != 1:
                continue
            relu_id = int(successors[0])
            relu = self._layer_by_id.get(relu_id)
            if (
                relu is not None
                and _kind(relu.kind) == "RELU"
                and (
                    self.residual_phase_screen
                    or self.residual_bound_screen
                    or (
                        self.correlation_targets is not None
                        and self.correlation_targets.get(relu_id, ())
                    )
                )
            ):
                routes.append((affine_id, relu_id))
        # Additional ADD fanout to a later skip join is expected and retained.
        # More than one targeted nonlinear route is ambiguous in C2 and fails
        # closed instead of assigning one source to the wrong predicate.
        return routes[0] if len(routes) == 1 else None

    def _capture_correlation_add_source(
        self,
        *,
        layer_id: int,
        expression: _AffineExpr,
    ) -> bool:
        """Freeze one pre-materialization ADD expression when it is targeted."""

        route = self._correlation_target_after_add(int(layer_id))
        if route is None:
            return False
        if int(layer_id) in self.correlation_add_sources:
            raise OperatorHZBuildError(
                f"correlation ADD source {layer_id} was captured twice"
            )
        frozen = _AffineExpr(
            c=np.ascontiguousarray(expression.c, dtype=np.float64).copy(),
            G=expression.G.copy().tocsr(),
            err=np.ascontiguousarray(expression.err, dtype=np.float64).copy(),
            affine_depth=int(expression.affine_depth),
        )
        frozen.G.sort_indices()
        self.correlation_add_sources[int(layer_id)] = frozen
        return True

    def _prepare_affine_correlation_shadow(
        self,
        *,
        pred: int,
        layer: Any,
        matrix: sp.csr_matrix,
        bias: np.ndarray,
    ) -> Dict[str, Any]:
        """Compose only property-selected affine rows through an ADD source."""

        layer_id = int(layer.id)
        receipt: Dict[str, Any] = {
            "schema": "operator_hz_property_correlation_shadow_v1",
            "affine_layer_id": layer_id,
            "proof_authority": False,
            "status": "not_targeted",
        }
        if self.correlation_targets is None or not self.materialize_add:
            return receipt
        successors = list(self._successors.get(layer_id, ()))
        if len(successors) != 1:
            return receipt
        relu_id = int(successors[0])
        relu = self._layer_by_id.get(relu_id)
        rows = tuple(self.correlation_targets.get(relu_id, ()))
        if relu is None or _kind(relu.kind) != "RELU" or not rows:
            return receipt
        origin = self._unmaterialized_add_origin(int(pred))
        if origin is None or origin not in self.correlation_add_sources:
            receipt.update(
                {
                    "relu_layer_id": relu_id,
                    "rows": list(rows),
                    "status": "unsupported_source_route",
                }
            )
            self.correlation_shadow_receipts.append(receipt)
            return receipt
        if relu_id in self.correlation_relu_shadows:
            raise OperatorHZBuildError(
                f"correlation shadow for ReLU {relu_id} was built twice"
            )
        matrix = _require_canonical_csr(
            matrix, name=f"correlation_shadow[{layer_id}].matrix"
        )
        bias = np.asarray(bias, dtype=np.float64).reshape(-1)
        row_ids = np.asarray(rows, dtype=np.int64)
        if (
            row_ids.size == 0
            or int(row_ids.min()) < 0
            or int(row_ids.max()) >= matrix.shape[0]
            or np.unique(row_ids).size != row_ids.size
        ):
            raise OperatorHZBuildError(
                f"correlation rows for ReLU {relu_id} are malformed"
            )
        source = self.correlation_add_sources[origin]
        shadow = self._affine(
            source,
            matrix[row_ids, :].tocsr(),
            bias[row_ids],
            layer_id=layer_id,
        )
        self.correlation_relu_shadows[relu_id] = (rows, shadow)
        lower, upper = self._cube_bounds(shadow)
        receipt.update(
            {
                "status": "prepared",
                "proof_authority": True,
                "add_origin_layer_id": int(origin),
                "relu_layer_id": relu_id,
                "rows": list(rows),
                "row_count": int(row_ids.size),
                "source_generator_nnz": int(source.G.nnz),
                "shadow_generator_nnz": int(shadow.G.nnz),
                "shadow_lower_min": float(np.min(lower)),
                "shadow_upper_max": float(np.max(upper)),
                "shadow_generator_sha256": _csr_sha256(shadow.G),
            }
        )
        self.correlation_shadow_receipts.append(receipt)
        return receipt

    def _prepare_residual_phase_screen(
        self,
        *,
        pred: int,
        layer: Any,
        ordinary: _AffineExpr,
        matrix: sp.csr_matrix,
        bias: np.ndarray,
    ) -> Dict[str, Any]:
        """Screen only cube-unstable rows for provable residual phases.

        Transient row chunks are recomposed over the pre-materialization ADD
        source.  Phase mode retains only rows proving ``u <= 0`` or
        ``l >= 0``.  Bound mode retains every strict outward improvement.
        """

        layer_id = int(layer.id)
        receipt: Dict[str, Any] = {
            "schema": "operator_hz_residual_phase_screen_v1",
            "affine_layer_id": layer_id,
            "enabled": bool(
                self.residual_phase_screen
                or self.residual_bound_screen
            ),
            "proof_authority": False,
            "status": "disabled",
            "unstable_rows_scanned": 0,
            "stabilized_active": 0,
            "stabilized_inactive": 0,
        }
        if not (
            self.residual_phase_screen or self.residual_bound_screen
        ):
            return receipt
        if not self.materialize_add:
            raise OperatorHZBuildError(
                "residual phase screen requires materialized ADD frames"
            )
        successors = list(self._successors.get(layer_id, ()))
        if len(successors) != 1:
            receipt["status"] = "unsupported_affine_fanout"
            return receipt
        relu_id = int(successors[0])
        relu = self._layer_by_id.get(relu_id)
        if relu is None or _kind(relu.kind) != "RELU":
            receipt["status"] = "not_relu_predecessor"
            return receipt
        origin = self._unmaterialized_add_origin(int(pred))
        if origin is None or origin not in self.correlation_add_sources:
            receipt.update(
                {
                    "relu_layer_id": relu_id,
                    "status": "unsupported_source_route",
                }
            )
            self.residual_phase_screen_receipts.append(receipt)
            return receipt
        if relu_id in self.residual_phase_screen_bounds:
            raise OperatorHZBuildError(
                f"residual phase screen for ReLU {relu_id} was built twice"
            )

        ordinary_lower, ordinary_upper = self._cube_bounds(ordinary)
        unstable = np.flatnonzero(
            (ordinary_lower < 0.0) & (ordinary_upper > 0.0)
        ).astype(np.int64, copy=False)
        receipt.update(
            {
                "status": "no_cube_unstable_rows",
                "add_origin_layer_id": int(origin),
                "relu_layer_id": relu_id,
                "unstable_rows_scanned": int(unstable.size),
            }
        )
        if not unstable.size:
            self.residual_phase_screen_receipts.append(receipt)
            return receipt

        matrix = _require_canonical_csr(
            matrix, name=f"phase_screen[{layer_id}].matrix"
        )
        bias = np.asarray(bias, dtype=np.float64).reshape(-1)
        source = self.correlation_add_sources[origin]
        retained_rows: List[int] = []
        retained_lower: List[float] = []
        retained_upper: List[float] = []
        transient_nnz = 0
        started = time.monotonic()
        for start in range(0, int(unstable.size), _LIVE_AFFINE_CHUNK_ROWS):
            self._check_deadline(
                f"residual_phase_screen_{layer_id}_{start}"
            )
            rows = unstable[
                start : start + int(_LIVE_AFFINE_CHUNK_ROWS)
            ]
            shadow = self._affine(
                source,
                matrix[rows, :].tocsr(),
                bias[rows],
                layer_id=layer_id,
            )
            lower, upper = self._cube_bounds(shadow)
            transient_nnz += int(shadow.G.nnz)
            stable = (upper <= 0.0) | (lower >= 0.0)
            improved = (
                (lower > ordinary_lower[rows])
                | (upper < ordinary_upper[rows])
            )
            retain = improved if self.residual_bound_screen else stable
            for position in np.flatnonzero(retain):
                retained_rows.append(int(rows[int(position)]))
                retained_lower.append(float(lower[int(position)]))
                retained_upper.append(float(upper[int(position)]))

        retained = tuple(retained_rows)
        lower_array = np.ascontiguousarray(
            retained_lower, dtype=np.float64
        )
        upper_array = np.ascontiguousarray(
            retained_upper, dtype=np.float64
        )
        self.residual_phase_screen_bounds[relu_id] = (
            retained,
            lower_array,
            upper_array,
        )
        inactive = int(np.count_nonzero(upper_array <= 0.0))
        active = int(np.count_nonzero(lower_array >= 0.0))
        receipt.update(
            {
                "status": "prepared",
                "proof_authority": bool(retained),
                "mode": (
                    "strict_bound_improvement"
                    if self.residual_bound_screen
                    else "stable_phase_only"
                ),
                "retained_rows": list(retained),
                "retained_count": int(len(retained)),
                "stabilized_active": active,
                "stabilized_inactive": inactive,
                "transient_shadow_nnz": int(transient_nnz),
                "elapsed_seconds": float(
                    max(0.0, time.monotonic() - started)
                ),
                "retained_bounds_sha256": _f64_array_sha256(
                    np.stack((lower_array, upper_array), axis=0)
                    if retained
                    else np.zeros((2, 0), dtype=np.float64)
                ),
            }
        )
        self.residual_phase_screen_receipts.append(receipt)
        return receipt

    def _apply_residual_phase_screen(
        self,
        *,
        layer_id: int,
        cube_lower: np.ndarray,
        cube_upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        lower = np.asarray(cube_lower, dtype=np.float64).copy()
        upper = np.asarray(cube_upper, dtype=np.float64).copy()
        receipt: Dict[str, Any] = {
            "schema": "operator_hz_residual_phase_screen_v1",
            "relu_layer_id": int(layer_id),
            "enabled": bool(
                self.residual_phase_screen
                or self.residual_bound_screen
            ),
            "proof_authority": False,
            "status": "not_prepared",
            "rows_applied": 0,
            "stabilized_active": 0,
            "stabilized_inactive": 0,
        }
        prepared = self.residual_phase_screen_bounds.get(int(layer_id))
        if prepared is None:
            receipt["status"] = (
                "not_targeted"
                if (
                    self.residual_phase_screen
                    or self.residual_bound_screen
                )
                else "disabled"
            )
            return lower, upper, receipt
        rows, shadow_lower, shadow_upper = prepared
        if not rows:
            receipt["status"] = "no_stable_shadow_rows"
            return lower, upper, receipt
        row_ids = np.asarray(rows, dtype=np.int64)
        lower[row_ids] = np.maximum(lower[row_ids], shadow_lower)
        upper[row_ids] = np.minimum(upper[row_ids], shadow_upper)
        if np.any(lower[row_ids] > upper[row_ids]):
            raise OperatorHZBuildError(
                f"residual phase screen crossed cube at ReLU {layer_id}"
            )
        inactive = upper[row_ids] <= 0.0
        active = lower[row_ids] >= 0.0
        if (
            not self.residual_bound_screen
            and np.any(~(inactive | active))
        ):
            raise OperatorHZBuildError(
                f"residual phase screen retained an unstable row at "
                f"ReLU {layer_id}"
            )
        receipt.update(
            {
                "status": "applied",
                "proof_authority": True,
                "rows": list(rows),
                "rows_applied": int(row_ids.size),
                "rows_tightened": int(row_ids.size),
                "stabilized_active": int(np.count_nonzero(active)),
                "stabilized_inactive": int(np.count_nonzero(inactive)),
            }
        )
        return lower, upper, receipt

    def _correlation_shadow_bounds(
        self,
        *,
        layer_id: int,
        pred: int,
        cube_lower: np.ndarray,
        cube_upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Intersect the ordinary cube with a row-local correlation shadow."""

        lower = np.asarray(cube_lower, dtype=np.float64).copy()
        upper = np.asarray(cube_upper, dtype=np.float64).copy()
        receipt: Dict[str, Any] = {
            "schema": "operator_hz_property_correlation_shadow_v1",
            "relu_layer_id": int(layer_id),
            "enabled": bool(self.correlation_targets is not None),
            "proof_authority": False,
            "status": "not_targeted",
            "rows_tightened": 0,
            "stabilized_active": 0,
            "stabilized_inactive": 0,
        }
        rows = (
            ()
            if self.correlation_targets is None
            else tuple(self.correlation_targets.get(int(layer_id), ()))
        )
        if not rows:
            return lower, upper, receipt

        prepared = self.correlation_relu_shadows.get(int(layer_id))
        if prepared is None:
            origin = self._unmaterialized_add_origin(int(pred))
            if origin is not None and origin in self.correlation_add_sources:
                source = self.correlation_add_sources[origin]
                row_ids = np.asarray(rows, dtype=np.int64)
                shadow = self._rows(source, row_ids)
                prepared = (rows, shadow)
            else:
                receipt.update(
                    {"status": "unsupported_source_route", "rows": list(rows)}
                )
                self.correlation_shadow_receipts.append(dict(receipt))
                return lower, upper, receipt

        prepared_rows, shadow = prepared
        if tuple(prepared_rows) != rows or shadow.size != len(rows):
            raise OperatorHZBuildError(
                f"correlation shadow row binding mismatch at ReLU {layer_id}"
            )
        row_ids = np.asarray(rows, dtype=np.int64)
        shadow_lower, shadow_upper = self._cube_bounds(shadow)
        old_lower = lower[row_ids].copy()
        old_upper = upper[row_ids].copy()
        lower[row_ids] = np.maximum(old_lower, shadow_lower)
        upper[row_ids] = np.minimum(old_upper, shadow_upper)
        if np.any(lower[row_ids] > upper[row_ids]):
            raise OperatorHZBuildError(
                f"correlation shadow crossed ordinary cube at ReLU {layer_id}"
            )
        lower_gain = lower[row_ids] - old_lower
        upper_gain = old_upper - upper[row_ids]
        tightened = (lower_gain > 0.0) | (upper_gain > 0.0)
        receipt.update(
            {
                "status": "applied",
                "proof_authority": True,
                "rows": list(rows),
                "rows_attempted": int(row_ids.size),
                "rows_tightened": int(np.count_nonzero(tightened)),
                "stabilized_active": int(
                    np.count_nonzero(
                        (old_lower < 0.0)
                        & (lower[row_ids] >= 0.0)
                        & (upper[row_ids] > 0.0)
                    )
                ),
                "stabilized_inactive": int(
                    np.count_nonzero(
                        (old_upper > 0.0) & (upper[row_ids] <= 0.0)
                    )
                ),
                "max_lower_improvement": float(
                    np.max(lower_gain) if lower_gain.size else 0.0
                ),
                "max_upper_improvement": float(
                    np.max(upper_gain) if upper_gain.size else 0.0
                ),
                "ordinary_lower_min": float(np.min(old_lower)),
                "ordinary_upper_max": float(np.max(old_upper)),
                "shadow_lower_min": float(np.min(shadow_lower)),
                "shadow_upper_max": float(np.max(shadow_upper)),
                "shadow_generator_nnz": int(shadow.G.nnz),
                "shadow_generator_sha256": _csr_sha256(shadow.G),
            }
        )
        self.correlation_shadow_receipts.append(dict(receipt))
        return lower, upper, receipt

    def _affine_collapse_prescreen(
        self,
        source: _AffineExpr,
        matrix: sp.csr_matrix,
        bias: np.ndarray,
        *,
        source_radius: np.ndarray,
        source_mass_upper: np.ndarray,
        layer_id: int,
    ) -> _AffineExpr:
        """Cheaply enclose rows that can be collapsed before ``W @ G``.

        ``source_radius`` encloses ``row_l1(source.G) + source.err``.  Hence
        ``abs(W) @ source_radius`` encloses both the composed generator mass
        and propagated source error without constructing the sparse product.
        The ordinary affine arithmetic allowance is then added unchanged.
        A row whose resulting cube upper bound is nonpositive can therefore
        skip ``W @ G`` entirely; its sole successor ReLU maps it to exact zero.

        This is only a one-sided screen.  Rows not proved inactive here retain
        the established sparse affine path and its exact stored-float order.
        """

        center = (
            np.asarray(matrix @ source.c, dtype=np.float64).reshape(-1)
            + bias
        )
        abs_matrix = _absolute_csr_topology_view(
            matrix, name=f"live_affine[{layer_id}].prescreen_abs_matrix"
        )
        projected_radius = _positive_spmv_upper(
            abs_matrix,
            source_radius,
            name=f"live_affine[{layer_id}].prescreen_radius",
        )
        transformed_mass = _positive_spmv_upper(
            abs_matrix,
            source_mass_upper,
            name=f"live_affine[{layer_id}].prescreen_mass",
        )
        arithmetic_mass = _nonnegative_sum_upper(
            transformed_mass,
            np.abs(bias),
            name=f"live_affine[{layer_id}].prescreen_arithmetic_mass",
        )
        fanin = np.diff(matrix.indptr).astype(np.float64)
        arithmetic_error = _inflate_nonnegative(
            _gamma_ops(
                2.0 * fanin + 2.0,
                name=f"live_affine[{layer_id}].prescreen_gamma",
            )
            * arithmetic_mass,
            4,
            active=arithmetic_mass > 0.0,
            name=f"live_affine[{layer_id}].prescreen_arithmetic_error",
        )
        radius = _nonnegative_sum_upper(
            projected_radius,
            arithmetic_error,
            name=f"live_affine[{layer_id}].prescreen_total_radius",
        )
        if not np.all(np.isfinite(center)):
            raise OperatorHZBuildError(
                f"live affine prescreen overflow/NaN at layer {layer_id}"
            )
        return _AffineExpr(
            c=center,
            G=sp.csr_matrix(
                (center.size, source.G.shape[1]), dtype=np.float64
            ),
            err=radius,
            affine_depth=source.affine_depth + 1,
        )

    def _try_fuse_affine_into_relu(
        self,
        *,
        pred: int,
        layer: Any,
        source: _AffineExpr,
        matrix: sp.csr_matrix,
        bias: np.ndarray,
    ) -> Tuple[Optional[_AffineExpr], Dict[str, Any]]:
        """Try a bounded ADD -> affine -> single-consumer ReLU fusion.

        Each sparse product is formed in a small row chunk with the ordinary
        :meth:`_affine` numerical allowance.  Rows whose *box-collapsed*
        expression still has upper bound ``<= 0`` discard their composed
        generator row and carry the entire row radius in ``err``.  This remains
        a complete affine enclosure; the sole successor ReLU then maps that
        independently certified nonpositive row to exact zero.

        The candidate mutates no frame state.  If its absolute time/size or
        estimated downstream-nnz stop-loss fires, the caller performs the
        existing full materialization instead.
        """

        layer_id = int(layer.id)
        successor_ids = list(self._successors.get(layer_id, []))
        origin = self._unmaterialized_add_origin(pred)
        audit: Dict[str, Any] = {
            "schema": "operator_hz_live_affine_relu_v1",
            "layer_id": layer_id,
            "source_layer_id": int(pred),
            "add_origin_layer_id": origin,
            "chunk_rows": int(_LIVE_AFFINE_CHUNK_ROWS),
            "candidate_only": True,
            "proof_authority": False,
            "status": "not_eligible",
        }
        if (
            self.materialize_add
            or source.affine_depth < 1
            or origin is None
            or len(successor_ids) != 1
            or _kind(self._layer_by_id[successor_ids[0]].kind) != "RELU"
        ):
            return None, audit

        matrix = _require_canonical_csr(
            matrix, name=f"live_affine[{layer_id}].matrix"
        )
        _require_canonical_csr(
            source.G, name=f"live_affine[{layer_id}].source_generators"
        )
        source_generator_sha256_before = _csr_sha256(source.G)
        bias = np.asarray(bias, dtype=np.float64).reshape(-1)
        if matrix.shape != (bias.size, source.size):
            raise OperatorHZBuildError(
                f"live affine shape mismatch at layer {layer_id}: "
                f"W={matrix.shape}, source={source.size}, bias={bias.size}"
            )
        if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(bias)):
            raise OperatorHZBuildError(
                f"live affine parameters are non-finite at layer {layer_id}"
            )
        source_lower, source_upper = self._cube_bounds(source)
        _, source_midpoint_radius = _enclosing_center_radius(
            source_lower,
            source_upper,
            name=f"live affine source box {layer_id}",
        )
        # The prescreen propagates a radius around the *stored affine center*,
        # not around the midpoint of its already-outward cube.  Those centers
        # can differ by several ulps on asymmetric rows; midpoint radius would
        # then be an under-approximation and could falsely erase a live row.
        lower_distance = source.c - source_lower
        upper_distance = source_upper - source.c
        variable_distance = (lower_distance > 0.0) | (upper_distance > 0.0)
        lower_distance[variable_distance] = np.nextafter(
            lower_distance[variable_distance], np.inf
        )
        upper_distance[variable_distance] = np.nextafter(
            upper_distance[variable_distance], np.inf
        )
        source_radius = np.maximum(lower_distance, upper_distance)
        if (
            not np.all(np.isfinite(source_radius))
            or np.any(source_radius < 0.0)
        ):
            raise OperatorHZBuildError(
                f"live affine source radius is invalid at layer {layer_id}"
            )
        # Preserve the established structural stop-loss population.  The
        # midpoint radius is sufficient for its old nonpoint predicate, while
        # only the numerical prescreen consumes the stricter center-relative
        # radius above.
        source_variable_rows = np.flatnonzero(
            source_midpoint_radius > 0.0
        ).astype(np.int64, copy=False)
        source_mass_upper = _nonnegative_sum_upper(
            np.abs(source.c),
            _row_l1_upper(
                source.G,
                name=f"live_affine[{layer_id}].source_G_l1",
            ),
            source.err,
            name=f"live_affine[{layer_id}].source_mass",
        )
        source_mass_upper.setflags(write=False)

        attempt_started = time.monotonic()
        if self.live_affine_fusion_started_at is None:
            self.live_affine_fusion_started_at = attempt_started
        remaining_candidate_seconds = max(
            0.0,
            _LIVE_AFFINE_TOTAL_SECONDS
            - self.live_affine_fusion_elapsed_seconds,
        )
        local_deadline = attempt_started + remaining_candidate_seconds
        if self.deadline is not None:
            local_deadline = min(local_deadline, self.deadline)
        if attempt_started >= local_deadline:
            if self.deadline is not None and attempt_started >= self.deadline:
                self._check_deadline(f"live_affine_{layer_id}_before")
            audit.update(
                {
                    "status": "fallback:local_time_budget_exhausted",
                    "elapsed_seconds": 0.0,
                }
            )
            return None, audit

        prescreen = self._affine_collapse_prescreen(
            source,
            matrix,
            bias,
            source_radius=source_radius,
            source_mass_upper=source_mass_upper,
            layer_id=layer_id,
        )
        _, prescreen_upper = self._cube_bounds(prescreen)
        prescreen_drop = prescreen_upper <= 0.0
        if time.monotonic() >= local_deadline:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self._check_deadline(f"live_affine_{layer_id}_prescreen")
            elapsed = max(0.0, time.monotonic() - attempt_started)
            audit.update(
                {
                    "status": "fallback:local_time_budget_exhausted",
                    "elapsed_seconds": float(elapsed),
                    "prescreen_inactive_rows": int(
                        np.count_nonzero(prescreen_drop)
                    ),
                }
            )
            self.live_affine_fusion_elapsed_seconds += float(elapsed)
            return None, audit

        c_parts: List[np.ndarray] = []
        err_parts: List[np.ndarray] = []
        generator_parts: List[sp.csr_matrix] = []
        dropped_parts: List[np.ndarray] = []
        full_composed_nnz = 0
        composed_rows_evaluated = 0
        stored_nnz = 0
        exact_inactive_rows = 0
        box_inactive_rows = 0
        closest_box_inactive_upper = -math.inf

        for start in range(0, int(matrix.shape[0]), _LIVE_AFFINE_CHUNK_ROWS):
            now = time.monotonic()
            if now >= local_deadline:
                if self.deadline is not None and now >= self.deadline:
                    self._check_deadline(
                        f"live_affine_{layer_id}_chunk_{start}"
                    )
                audit.update(
                    {
                        "status": "fallback:local_time_budget_exhausted",
                        "elapsed_seconds": max(
                            0.0, time.monotonic() - attempt_started
                        ),
                        "full_composed_nnz": int(full_composed_nnz),
                        "full_composed_nnz_scope": "post_prescreen_rows_only",
                        "composed_rows_evaluated": int(
                            composed_rows_evaluated
                        ),
                        "prescreen_inactive_rows": int(
                            np.count_nonzero(prescreen_drop)
                        ),
                        "stored_nnz": int(stored_nnz),
                    }
                )
                self.live_affine_fusion_elapsed_seconds += float(
                    audit["elapsed_seconds"]
                )
                return None, audit
            stop = min(
                int(matrix.shape[0]),
                start + int(_LIVE_AFFINE_CHUNK_ROWS),
            )
            chunk_size = stop - start
            chunk_c = prescreen.c[start:stop].copy()
            chunk_err = prescreen.err[start:stop].copy()
            drop = prescreen_drop[start:stop].copy()
            prescreen_count = int(np.count_nonzero(drop))
            exact_inactive_rows += prescreen_count
            box_inactive_rows += prescreen_count
            if prescreen_count:
                closest_box_inactive_upper = max(
                    closest_box_inactive_upper,
                    float(np.max(prescreen_upper[start:stop][drop])),
                )

            local_keep = np.flatnonzero(~drop).astype(np.int64, copy=False)
            if local_keep.size:
                composed_rows_evaluated += int(local_keep.size)
                selected = start + local_keep
                exact = self._affine(
                    source,
                    matrix[selected, :].tocsr(),
                    bias[selected],
                    layer_id=layer_id,
                    _source_mass_upper=source_mass_upper,
                )
                chunk_c[local_keep] = exact.c
                chunk_err[local_keep] = exact.err
                _, exact_upper = self._cube_bounds(exact)
                exact_inactive_rows += int(
                    np.count_nonzero(exact_upper <= 0.0)
                )

                # The cheap screen is intentionally one-sided.  Retained
                # rows still receive the established composed-row recheck so
                # its additional exact inactive rows and final sparse frame
                # remain available without composing already-proven rows.
                collapsed_radius = _nonnegative_sum_upper(
                    _row_l1_upper(
                        exact.G,
                        name=f"live_affine[{layer_id}].chunk_G_l1",
                    ),
                    exact.err,
                    name=f"live_affine[{layer_id}].collapsed_radius",
                )
                collapsed = _AffineExpr(
                    c=exact.c,
                    G=sp.csr_matrix(
                        (exact.size, exact.G.shape[1]), dtype=np.float64
                    ),
                    err=collapsed_radius,
                    affine_depth=exact.affine_depth,
                )
                _, collapsed_upper = self._cube_bounds(collapsed)
                exact_drop = collapsed_upper <= 0.0
                box_inactive_rows += int(np.count_nonzero(exact_drop))
                if np.any(exact_drop):
                    closest_box_inactive_upper = max(
                        closest_box_inactive_upper,
                        float(np.max(collapsed_upper[exact_drop])),
                    )
                    chunk_err[local_keep[exact_drop]] = collapsed_radius[
                        exact_drop
                    ]
                    drop[local_keep[exact_drop]] = True

                keep_scale = sp.diags(
                    (~exact_drop).astype(np.float64), format="csr"
                )
                compact_G = (keep_scale @ exact.G).tocsr()
                compact_G.eliminate_zeros()
                compact_G.sort_indices()
                row_nnz = np.zeros(
                    chunk_size, dtype=compact_G.indptr.dtype
                )
                row_nnz[local_keep] = np.diff(compact_G.indptr)
                expanded_indptr = np.empty(
                    chunk_size + 1, dtype=compact_G.indptr.dtype
                )
                expanded_indptr[0] = 0
                np.cumsum(row_nnz, out=expanded_indptr[1:])
                kept_G = sp.csr_matrix(
                    (
                        compact_G.data,
                        compact_G.indices,
                        expanded_indptr,
                    ),
                    shape=(chunk_size, source.G.shape[1]),
                    copy=False,
                )
                full_composed_nnz += int(exact.G.nnz)
            else:
                kept_G = sp.csr_matrix(
                    (chunk_size, source.G.shape[1]), dtype=np.float64
                )

            stored_nnz += int(kept_G.nnz)
            if stored_nnz > _LIVE_AFFINE_MAX_STORED_NNZ:
                audit.update(
                    {
                        "status": "fallback:stored_nnz_stoploss",
                        "elapsed_seconds": max(
                            0.0, time.monotonic() - attempt_started
                        ),
                        "full_composed_nnz": int(full_composed_nnz),
                        "full_composed_nnz_scope": "post_prescreen_rows_only",
                        "composed_rows_evaluated": int(
                            composed_rows_evaluated
                        ),
                        "prescreen_inactive_rows": int(
                            np.count_nonzero(prescreen_drop)
                        ),
                        "stored_nnz": int(stored_nnz),
                    }
                )
                self.live_affine_fusion_elapsed_seconds += float(
                    audit["elapsed_seconds"]
                )
                return None, audit
            c_parts.append(chunk_c)
            err_parts.append(chunk_err)
            generator_parts.append(kept_G)
            dropped_parts.append(drop)

        c = np.concatenate(c_parts) if c_parts else np.zeros(0, dtype=np.float64)
        err = (
            np.concatenate(err_parts)
            if err_parts
            else np.zeros(0, dtype=np.float64)
        )
        G = (
            sp.vstack(generator_parts, format="csr")
            if generator_parts
            else sp.csr_matrix((0, source.G.shape[1]), dtype=np.float64)
        )
        G.eliminate_zeros()
        G.sort_indices()
        dropped = (
            np.concatenate(dropped_parts)
            if dropped_parts
            else np.zeros(0, dtype=bool)
        )
        out = _AffineExpr(
            c=c,
            G=G,
            err=err,
            affine_depth=source.affine_depth + 1,
        )
        final_lower, final_upper = self._cube_bounds(out)
        if dropped.size != out.size or (
            np.any(dropped) and np.any(final_upper[dropped] > 0.0)
        ):
            raise OperatorHZBuildError(
                f"live affine collapsed-row postcondition failed at layer "
                f"{layer_id}"
            )
        source_generator_sha256_after = _csr_sha256(source.G)
        if (
            source_generator_sha256_after
            != source_generator_sha256_before
        ):
            raise OperatorHZBuildError(
                f"live affine candidate mutated its source CSR at layer "
                f"{layer_id}"
            )

        downstream_rows = np.flatnonzero(final_upper > 0.0).astype(
            np.int64, copy=False
        )
        downstream_direct_nnz = int(G[downstream_rows, :].nnz)
        downstream_local_nnz = int(
            matrix[downstream_rows, :][:, source_variable_rows].nnz
        )
        # Account for the two equality-band directions eliminated by avoiding
        # the old full source cut.  Fresh materialization columns cannot
        # cancel source columns, so this is the exact structural nnz of those
        # two bands before constant-true row removal.
        eliminated_cut_nnz = 2 * (
            int(source.G.nnz) + int(source_variable_rows.size)
        )
        estimated_nnz_delta = (
            2 * (downstream_direct_nnz - downstream_local_nnz)
            - eliminated_cut_nnz
        )
        elapsed = max(0.0, time.monotonic() - attempt_started)
        audit.update(
            {
                "status": (
                    "applied"
                    if estimated_nnz_delta <= 0
                    else "fallback:estimated_nnz_regression"
                ),
                "elapsed_seconds": float(elapsed),
                "output_rows": int(out.size),
                "exact_inactive_rows": int(exact_inactive_rows),
                "box_inactive_rows": int(box_inactive_rows),
                "downstream_rows": int(downstream_rows.size),
                "full_composed_nnz": int(full_composed_nnz),
                "full_composed_nnz_scope": "post_prescreen_rows_only",
                "composed_rows_evaluated": int(composed_rows_evaluated),
                "prescreen_inactive_rows": int(
                    np.count_nonzero(prescreen_drop)
                ),
                "stored_nnz": int(G.nnz),
                "downstream_direct_nnz": int(downstream_direct_nnz),
                "downstream_local_nnz": int(downstream_local_nnz),
                "source_variable_rows": int(source_variable_rows.size),
                "eliminated_cut_nnz_estimate": int(eliminated_cut_nnz),
                "estimated_nnz_delta": int(estimated_nnz_delta),
                "closest_box_inactive_upper": (
                    None
                    if not np.isfinite(closest_box_inactive_upper)
                    else float(closest_box_inactive_upper)
                ),
                "final_lower_min": (
                    float(np.min(final_lower)) if final_lower.size else None
                ),
                "final_upper_max": (
                    float(np.max(final_upper)) if final_upper.size else None
                ),
                "source_generator_sha256_before": (
                    source_generator_sha256_before
                ),
                "source_generator_sha256_after": (
                    source_generator_sha256_after
                ),
                "fused_generator_sha256": _csr_sha256(G),
                "proof_authority": bool(box_inactive_rows),
            }
        )
        self.live_affine_fusion_elapsed_seconds += float(elapsed)
        if estimated_nnz_delta > 0:
            return None, audit
        return out, audit

    def _affine(
        self,
        expr: _AffineExpr,
        matrix: sp.csr_matrix,
        bias: np.ndarray,
        *,
        layer_id: int,
        _source_mass_upper: Optional[np.ndarray] = None,
    ) -> _AffineExpr:
        matrix = _require_canonical_csr(
            matrix, name=f"affine[{layer_id}].matrix"
        )
        _require_canonical_csr(
            expr.G, name=f"affine[{layer_id}].source_generators"
        )
        bias = np.asarray(bias, dtype=np.float64).reshape(-1)
        if matrix.shape[1] != expr.size or matrix.shape[0] != bias.size:
            raise OperatorHZBuildError(
                f"affine shape mismatch at layer {layer_id}: W={matrix.shape}, "
                f"input={expr.size}, bias={bias.size}"
            )
        if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(bias)):
            raise OperatorHZBuildError(
                f"affine parameters contain NaN or infinity at layer {layer_id}"
            )
        c = np.asarray(matrix @ expr.c, dtype=np.float64).reshape(-1) + bias
        G = (matrix @ expr.G).tocsr()
        G.eliminate_zeros()
        # SciPy sparse multiplication may emit a valid CSR with reverse
        # column order.  Normalize at the affine producer boundary so an
        # ASSERT immediately following a mixed-phase affine cannot publish a
        # non-canonical owned HZ.
        G.sort_indices()
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(G.data)):
            raise OperatorHZBuildError(
                f"affine expression overflow/NaN at layer {layer_id}"
            )

        abs_matrix = _absolute_csr_topology_view(
            matrix, name=f"affine[{layer_id}].absolute_matrix"
        )
        if _source_mass_upper is None:
            source_mass = _nonnegative_sum_upper(
                np.abs(expr.c),
                _row_l1_upper(
                    expr.G, name=f"affine[{layer_id}].source_G_l1"
                ),
                expr.err,
                name=f"affine[{layer_id}].source_mass",
            )
        else:
            source_mass = np.asarray(
                _source_mass_upper, dtype=np.float64
            ).reshape(-1)
            if (
                source_mass.size != expr.size
                or not np.all(np.isfinite(source_mass))
                or np.any(source_mass < 0.0)
            ):
                raise OperatorHZBuildError(
                    f"affine[{layer_id}] precomputed source mass is invalid"
                )
        transformed_mass = _positive_spmv_upper(
            abs_matrix,
            source_mass,
            name=f"affine[{layer_id}].transformed_mass",
        )
        arithmetic_mass = _nonnegative_sum_upper(
            transformed_mass,
            np.abs(bias),
            name=f"affine[{layer_id}].arithmetic_mass",
        )
        propagated_error = _positive_spmv_upper(
            abs_matrix,
            expr.err,
            name=f"affine[{layer_id}].propagated_error",
        )
        fanin = np.diff(matrix.indptr).astype(np.float64)
        arithmetic_error = (
            _gamma_ops(2.0 * fanin + 2.0, name=f"affine[{layer_id}].gamma")
            * arithmetic_mass
        )
        arithmetic_error = _inflate_nonnegative(
            arithmetic_error,
            4,
            active=arithmetic_mass > 0.0,
            name=f"affine[{layer_id}].arithmetic_error",
        )
        err = _nonnegative_sum_upper(
            propagated_error,
            arithmetic_error,
            name=f"affine[{layer_id}].total_error",
        )
        return _AffineExpr(
            c=c,
            G=G,
            err=err,
            affine_depth=expr.affine_depth + 1,
        )

    def _add_expr(
        self,
        left: _AffineExpr,
        right: _AffineExpr,
        *,
        layer_id: int,
    ) -> _AffineExpr:
        if left.size != right.size:
            raise OperatorHZBuildError(
                f"ADD layer {layer_id} does not support broadcasting in strict "
                f"operator-HZ mode: {left.size} vs {right.size}"
            )
        _require_canonical_csr(left.G, name=f"add[{layer_id}].left")
        _require_canonical_csr(right.G, name=f"add[{layer_id}].right")
        width = max(left.G.shape[1], right.G.shape[1], self.n_cont)
        G = (_pad_cols(left.G, width) + _pad_cols(right.G, width)).tocsr()
        G.eliminate_zeros()
        c = left.c + right.c
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(G.data)):
            raise OperatorHZBuildError(f"ADD expression overflow/NaN at layer {layer_id}")

        nominal_mass = _nonnegative_sum_upper(
            np.abs(left.c),
            np.abs(right.c),
            _row_l1_upper(left.G, name=f"add[{layer_id}].left_G_l1"),
            _row_l1_upper(right.G, name=f"add[{layer_id}].right_G_l1"),
            name=f"add[{layer_id}].nominal_mass",
        )
        propagated_error = _nonnegative_sum_upper(
            left.err,
            right.err,
            name=f"add[{layer_id}].propagated_error",
        )
        arithmetic_error = _inflate_nonnegative(
            _gamma_ops(8, name=f"add[{layer_id}].gamma") * nominal_mass,
            4,
            active=nominal_mass > 0.0,
            name=f"add[{layer_id}].arithmetic_error",
        )
        err = _nonnegative_sum_upper(
            propagated_error,
            arithmetic_error,
            name=f"add[{layer_id}].total_error",
        )
        return _AffineExpr(
            c=c,
            G=G,
            err=err,
            affine_depth=max(left.affine_depth, right.affine_depth),
        )

    @staticmethod
    def _cube_bounds(expr: _AffineExpr) -> Tuple[np.ndarray, np.ndarray]:
        """Outward-rounded cube bounds independent of interval-analysis facts."""

        G = _require_canonical_csr(expr.G, name="cube.generators")
        mass = _nonnegative_sum_upper(
            _row_l1_upper(G, name="cube.G_l1"),
            expr.err,
            name="cube.total_radius",
        )
        if not np.all(np.isfinite(mass)):
            raise OperatorHZBuildError("generator row mass overflowed")

        # Higham-style gamma_k guard for each sparse row sum, plus the final
        # center +/- mass operation.  For impossible k*eps >= 1, fail closed.
        row_nnz = np.diff(G.indptr).astype(np.float64)
        eps = np.finfo(np.float64).eps
        k_eps = (row_nnz + 2.0) * eps
        if np.any(k_eps >= 1.0):
            raise OperatorHZBuildError("cube-bound row is too long for a finite guard")
        gamma = k_eps / (1.0 - k_eps)
        guard = gamma * (np.abs(expr.c) + mass)
        guard += np.finfo(np.float64).tiny

        # A structurally constant row is an exact point in the stored-float
        # model.  Widening it would violate the point-consistency audit and can
        # turn ReLU(0) into a spurious unstable neuron.
        variable = (row_nnz > 0) | (expr.err > 0.0)
        lb = expr.c.copy()
        ub = expr.c.copy()
        lb[variable] = np.nextafter(
            expr.c[variable] - mass[variable] - guard[variable], -np.inf
        )
        ub[variable] = np.nextafter(
            expr.c[variable] + mass[variable] + guard[variable], np.inf
        )
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise OperatorHZBuildError("cube bounds contain NaN or infinity")
        if np.any(lb > ub):
            raise OperatorHZBuildError("cube bound construction produced lb > ub")
        return lb, ub

    def _box_expr(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        *,
        affine_depth: int = 0,
    ) -> Tuple[_AffineExpr, int]:
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        if lb.size != ub.size or np.any(lb > ub):
            raise OperatorHZBuildError("invalid local materialization box")
        center, radius = _enclosing_center_radius(
            lb, ub, name="local materialization box"
        )
        rows = np.flatnonzero(radius > 0.0).astype(np.int64, copy=False)
        cols = self._allocate_cont(int(rows.size))
        G = sp.csr_matrix(
            (radius[rows], (rows, cols)),
            shape=(lb.size, self.n_cont),
            dtype=np.float64,
        )
        G.eliminate_zeros()
        return _AffineExpr(
            center,
            G,
            np.zeros(lb.size, dtype=np.float64),
            affine_depth=affine_depth,
        ), int(rows.size)

    # ------------------------------------------------------------------
    # Constraint assembly
    # ------------------------------------------------------------------

    def _append_equality(
        self,
        lhs: _AffineExpr,
        rhs: _AffineExpr,
        *,
        tag: str,
        layer_id: int,
        add_materialize_range: bool = False,
    ) -> int:
        """Append a sound equality *band* and return its upper-row count.

        A literal equality between rounded affine coefficients can exclude the
        exact-real graph.  Each direction below is widened by the semantic and
        assembly allowance carried by :meth:`_append_upper`.
        """

        if lhs.size != rhs.size:
            raise OperatorHZBuildError(
                f"equality-band {tag} shape mismatch: {lhs.size} vs {rhs.size}"
            )
        if type(add_materialize_range) is not bool:
            raise OperatorHZBuildError(
                "equality RANGE selector must be a builtin bool"
            )
        zero = np.zeros(lhs.size, dtype=np.float64)
        forward_expr, reverse_expr = self._opposite_differences(lhs, rhs)
        forward, forward_keep = self._prepare_upper_block(
            forward_expr,
            zero,
            tag=f"{tag}:forward",
        )
        reverse, reverse_keep = self._prepare_upper_block(
            reverse_expr,
            zero,
            tag=f"{tag}:reverse",
        )

        sink = self._constraint_program_sink
        if sink is not None and add_materialize_range:
            if not np.array_equal(forward_keep, reverse_keep):
                raise OperatorHZBuildError(
                    "ADD materialization equality directions retained "
                    "different rows"
                )
            if forward is None or reverse is None:
                if forward is None and reverse is None:
                    return 0
                raise OperatorHZBuildError(
                    "ADD materialization equality has only one nonempty side"
                )
            return sink.append_add_materialize_range(
                forward,
                reverse,
                layer_id=int(layer_id),
            )

        count = 0
        if forward is not None:
            count += self._publish_upper_block(
                forward, layer_id=int(layer_id)
            )
        if reverse is not None:
            count += self._publish_upper_block(
                reverse, layer_id=int(layer_id)
            )
        return int(count)

    def _prepare_upper_block(
        self,
        expr: _AffineExpr,
        rhs: np.ndarray,
        *,
        tag: str,
        binary_rows: Optional[np.ndarray] = None,
        binary_cols: Optional[np.ndarray] = None,
        binary_data: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[_ConstraintBlock], np.ndarray]:
        _require_canonical_csr(expr.G, name=f"upper[{tag}].generators")
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs.size != expr.size:
            raise OperatorHZBuildError(
                f"upper constraint {tag} rhs={rhs.size}, rows={expr.size}"
            )
        Ac = _pad_cols(expr.G, self.n_cont)
        # For a semantic lhs ``nominal + delta`` with ``|delta|<=err``, every
        # true point satisfying lhs<=rhs obeys nominal<=rhs+err.  Inflate the
        # short rhs-c+err sum so binary64 assembly can only relax the row.
        rounded = rhs - expr.c + expr.err
        assembly_mass = (
            np.abs(rhs) + np.abs(expr.c) + np.asarray(expr.err, dtype=np.float64)
        )
        assembly_guard = (
            _gamma_ops(6, name=f"upper[{tag}].gamma") * assembly_mass
        )
        assembly_guard = _inflate_nonnegative(
            assembly_guard,
            4,
            active=assembly_mass > 0.0,
            name=f"upper[{tag}].assembly_guard",
        )
        ub = rounded + assembly_guard
        ub = np.where(
            assembly_mass > 0.0,
            np.nextafter(ub, np.inf),
            ub,
        )
        if not np.all(np.isfinite(ub)):
            raise OperatorHZBuildError(f"upper constraint {tag} has non-finite rhs")

        if binary_data is None:
            Ab = sp.csr_matrix((expr.size, self.n_bin), dtype=np.float64)
        else:
            rr = np.asarray(binary_rows, dtype=np.int64).reshape(-1)
            cc = np.asarray(binary_cols, dtype=np.int64).reshape(-1)
            dd = np.asarray(binary_data, dtype=np.float64).reshape(-1)
            if rr.size != cc.size or rr.size != dd.size:
                raise OperatorHZBuildError(f"binary triplet mismatch in {tag}")
            if rr.size and (
                int(rr.min()) < 0
                or int(rr.max()) >= expr.size
                or int(cc.min()) < 0
                or int(cc.max()) >= self.n_bin
            ):
                raise OperatorHZBuildError(f"binary triplet out of range in {tag}")
            Ab = sp.csr_matrix(
                (dd, (rr, cc)),
                shape=(expr.size, self.n_bin),
                dtype=np.float64,
            )
            Ab.eliminate_zeros()

        row_nnz = np.diff(Ac.indptr) + np.diff(Ab.indptr)
        constant_bad = (row_nnz == 0) & (ub < 0.0)
        if np.any(constant_bad):
            row = int(np.flatnonzero(constant_bad)[0])
            raise OperatorHZBuildError(
                f"constant upper contradiction in {tag}, row {row}: 0<={ub[row]}"
            )
        keep = (row_nnz != 0) | (ub < 0.0)
        if not np.any(keep):
            return None, np.asarray(keep, dtype=bool)
        # Exact-ReLU and ordinary affine relations almost always retain the
        # complete row block.  Boolean CSR slicing in that common case makes
        # two additional sparse copies per relation while changing nothing.
        # Only compact genuinely dropped constant rows.
        if not np.all(keep):
            Ac = Ac[keep, :].tocsr()
            Ab = Ab[keep, :].tocsr()
            ub = ub[keep]
        return (
            _ConstraintBlock(Ac, Ab, ub, tag),
            np.asarray(keep, dtype=bool),
        )

    def _publish_upper_block(
        self,
        block: _ConstraintBlock,
        *,
        layer_id: int,
    ) -> int:
        sink = self._constraint_program_sink
        if sink is None:
            self.ub_blocks.append(block)
            return int(block.Ac.shape[0])
        return sink.append_le(block, layer_id=int(layer_id))

    def _append_upper(
        self,
        expr: _AffineExpr,
        rhs: np.ndarray,
        *,
        tag: str,
        layer_id: int,
        binary_rows: Optional[np.ndarray] = None,
        binary_cols: Optional[np.ndarray] = None,
        binary_data: Optional[np.ndarray] = None,
    ) -> int:
        block, _keep = self._prepare_upper_block(
            expr,
            rhs,
            tag=tag,
            binary_rows=binary_rows,
            binary_cols=binary_cols,
            binary_data=binary_data,
        )
        if block is None:
            return 0
        return self._publish_upper_block(block, layer_id=int(layer_id))

    def _difference(
        self,
        left: _AffineExpr,
        right: _AffineExpr,
        *,
        left_scale: np.ndarray | float = 1.0,
        right_scale: np.ndarray | float = 1.0,
    ) -> _AffineExpr:
        if left.size != right.size:
            raise OperatorHZBuildError("affine difference shape mismatch")
        _require_canonical_csr(left.G, name="difference.left")
        _require_canonical_csr(right.G, name="difference.right")
        ls_raw = np.asarray(left_scale, dtype=np.float64)
        rs_raw = np.asarray(right_scale, dtype=np.float64)
        ls = np.broadcast_to(ls_raw, (left.size,))
        rs = np.broadcast_to(rs_raw, (right.size,))
        if not np.all(np.isfinite(ls)) or not np.all(np.isfinite(rs)):
            raise OperatorHZBuildError("affine difference has non-finite scale")
        left_G = _pad_cols(left.G, self.n_cont)
        right_G = _pad_cols(right.G, self.n_cont)
        if (
            ls_raw.ndim == 0
            and rs_raw.ndim == 0
            and float(ls_raw) == 1.0
            and float(rs_raw) == 1.0
        ):
            # The dominant exact-ReLU/equality path is an unscaled
            # difference.  Building two explicit identity diagonals here was
            # pure allocation and sparse-matmul overhead.
            G = (left_G - right_G).tocsr()
        else:
            LG = sp.diags(ls, format="csr") @ left_G
            RG = sp.diags(rs, format="csr") @ right_G
            G = (LG - RG).tocsr()
        G.eliminate_zeros()
        G.sort_indices()
        left_mass = _nonnegative_sum_upper(
            np.abs(left.c),
            _row_l1_upper(left.G, name="difference.left_G_l1"),
            name="difference.left_mass",
        )
        right_mass = _nonnegative_sum_upper(
            np.abs(right.c),
            _row_l1_upper(right.G, name="difference.right_G_l1"),
            name="difference.right_mass",
        )
        scaled_left_mass = _inflate_nonnegative(
            np.abs(ls) * left_mass,
            4,
            active=(np.abs(ls) > 0.0) & (left_mass > 0.0),
            name="difference.scaled_left_mass",
        )
        scaled_right_mass = _inflate_nonnegative(
            np.abs(rs) * right_mass,
            4,
            active=(np.abs(rs) > 0.0) & (right_mass > 0.0),
            name="difference.scaled_right_mass",
        )
        nominal_mass = _nonnegative_sum_upper(
            scaled_left_mass,
            scaled_right_mass,
            name="difference.nominal_mass",
        )
        propagated_left = _inflate_nonnegative(
            np.abs(ls) * left.err,
            4,
            active=(np.abs(ls) > 0.0) & (left.err > 0.0),
            name="difference.left_error",
        )
        propagated_right = _inflate_nonnegative(
            np.abs(rs) * right.err,
            4,
            active=(np.abs(rs) > 0.0) & (right.err > 0.0),
            name="difference.right_error",
        )
        arithmetic_error = _inflate_nonnegative(
            _gamma_ops(8, name="difference.gamma") * nominal_mass,
            4,
            active=nominal_mass > 0.0,
            name="difference.arithmetic_error",
        )
        err = _nonnegative_sum_upper(
            propagated_left,
            propagated_right,
            arithmetic_error,
            name="difference.total_error",
        )
        return _AffineExpr(
            c=ls * left.c - rs * right.c,
            G=G,
            err=err,
            affine_depth=max(left.affine_depth, right.affine_depth),
        )

    def _opposite_differences(
        self,
        left: _AffineExpr,
        right: _AffineExpr,
    ) -> Tuple[_AffineExpr, _AffineExpr]:
        """Build ``left-right`` and ``right-left`` with one error audit.

        Binary64 negation is exact.  The two relations have identical
        absolute coefficient mass, propagated input allowance, operation
        count, and therefore the same outward error bound.  Compute that
        bound once, while retaining the established subtraction order for the
        reverse center so signed-zero behavior is unchanged.
        """

        forward = self._difference(left, right)
        reverse_G = (-forward.G).tocsr()
        reverse_G.sort_indices()
        reverse = _AffineExpr(
            c=right.c - left.c,
            G=reverse_G,
            err=forward.err,
            affine_depth=forward.affine_depth,
        )
        return forward, reverse

    def _materialize(
        self,
        expr: _AffineExpr,
        *,
        layer_id: int,
        reason: str,
    ) -> Tuple[_AffineExpr, int, int, int]:
        lb, ub = self._cube_bounds(expr)
        y, new_cont = self._box_expr(lb, ub)
        if reason not in {"add_materialize", "affine_chain_cut"}:
            raise OperatorHZBuildError(
                f"unknown materialization constraint family {reason!r}"
            )
        new_ub = self._append_equality(
            y,
            expr,
            tag=f"{reason}:{layer_id}",
            layer_id=int(layer_id),
            add_materialize_range=(reason == "add_materialize"),
        )
        self.materialization_events.append(
            {
                "layer_id": int(layer_id),
                "reason": str(reason),
                "size": int(expr.size),
                "new_cont": int(new_cont),
                "new_eq": 0,
                "new_ub": int(new_ub),
                "source_value_nnz": int(expr.G.nnz),
                "source_error_max": (
                    float(np.max(expr.err)) if expr.err.size else 0.0
                ),
            }
        )
        return y, new_cont, 0, new_ub

    # ------------------------------------------------------------------
    # Proof-carrying constrained preactivation bounds
    # ------------------------------------------------------------------

    def _start_preactivation_clock(self) -> None:
        """Start one absolute local clock shared by every targeted ReLU."""

        if self.preactivation_lp_started_at is not None:
            return
        started = time.monotonic()
        local_deadline = started + self.preactivation_lp_time_limit
        if self.deadline is not None:
            local_deadline = min(local_deadline, self.deadline)
        self.preactivation_lp_started_at = started
        self.preactivation_lp_deadline = local_deadline

    def _preactivation_time_remaining(self) -> float:
        if self.preactivation_lp_started_at is None:
            local = self.preactivation_lp_time_limit
            if self.deadline is None:
                return max(0.0, local)
            return max(0.0, min(local, self.deadline - time.monotonic()))
        assert self.preactivation_lp_deadline is not None
        return max(0.0, self.preactivation_lp_deadline - time.monotonic())

    def _preactivation_require_time(self, stage: str) -> None:
        if self._preactivation_time_remaining() <= 0.0:
            if self.preactivation_lp_deadline_stage is None:
                self.preactivation_lp_deadline_stage = str(stage)
            raise _PreactivationLPDeadline(str(stage))

    @staticmethod
    def _preactivation_original_hash(base: _PreactivationLPBase) -> str:
        current = _csr_sha256(base.A)
        expected = base.csr_sha256 or current
        if current != expected:
            raise OperatorHZBuildError(
                "original preactivation CSR changed after snapshot"
            )
        return current

    def _preactivation_lp_base(self) -> _PreactivationLPBase:
        """Freeze the already-built HZ constraints in their original frame."""

        self._preactivation_require_time("snapshot_before")
        snapshot_started = time.monotonic()
        try:
            eq_c = _stack_padded(
                (block.Ac for block in self.eq_blocks), width=self.n_cont
            )
            eq_b = _stack_padded(
                (block.Ab for block in self.eq_blocks), width=self.n_bin
            )
            ub_c = _stack_padded(
                (block.Ac for block in self.ub_blocks), width=self.n_cont
            )
            ub_b = _stack_padded(
                (block.Ab for block in self.ub_blocks), width=self.n_bin
            )
            eq_A = sp.hstack((eq_c, eq_b), format="csr")
            ub_A = sp.hstack((ub_c, ub_b), format="csr")
            A = sp.vstack((eq_A, ub_A), format="csr")
            A.eliminate_zeros()
            A.sort_indices()
            if (
                not A.has_canonical_format
                or not A.has_sorted_indices
                or (A.nnz and not np.all(np.isfinite(A.data)))
            ):
                raise OperatorHZBuildError(
                    "preactivation LP constraint snapshot is invalid CSR"
                )
            eq_rhs = (
                np.concatenate([block.rhs for block in self.eq_blocks])
                if self.eq_blocks
                else np.zeros(0, dtype=np.float64)
            )
            upper_rhs = (
                np.concatenate([block.rhs for block in self.ub_blocks])
                if self.ub_blocks
                else np.zeros(0, dtype=np.float64)
            )
            rl = np.concatenate(
                (
                    eq_rhs,
                    np.full(upper_rhs.size, -np.inf, dtype=np.float64),
                )
            )
            ru = np.concatenate((eq_rhs, upper_rhs))
            n_frame = self.n_cont + self.n_bin
            lb = np.full(n_frame, -1.0, dtype=np.float64)
            ub = np.full(n_frame, 1.0, dtype=np.float64)
            csr_sha256 = _csr_sha256(A)
            # Candidate copies may be perturbed, but the certificate authority
            # is structurally immutable for the lifetime of this snapshot.
            for values in (A.data, A.indices, A.indptr, rl, ru, lb, ub):
                values.setflags(write=False)
            base = _PreactivationLPBase(
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                n_eq=int(eq_rhs.size),
                n_ub=int(upper_rhs.size),
                csr_sha256=csr_sha256,
            )
        finally:
            self.preactivation_lp_snapshot_seconds += max(
                0.0, time.monotonic() - snapshot_started
            )
        self._preactivation_require_time("snapshot_after")
        self._preactivation_original_hash(base)
        return base

    @staticmethod
    def _require_highs_ok(status: Any, operation: str) -> None:
        if _highspy is None or status != _highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS {operation} returned {status}")

    def _build_preactivation_candidate_model(
        self,
        base: _PreactivationLPBase,
    ) -> Tuple[Optional[_PersistentPreactivationHighs], Dict[str, Any]]:
        """Build one candidate-only model from a filtered copy of ``base.A``."""

        receipt: Dict[str, Any] = {
            "schema": "operator_hz_preactivation_highs_model_v1",
            "status": "not_started",
            "candidate_only": True,
            "proof_authority": False,
            "model_constructions": 0,
            "threads": int(_highs_process_threads()),
            "original_csr_sha256": None,
        }
        started = time.monotonic()
        try:
            self._preactivation_require_time("candidate_model_before")
            if _highspy is None:
                receipt["status"] = "highspy_unavailable"
                return None, receipt
            original_hash = self._preactivation_original_hash(base)
            receipt["original_csr_sha256"] = original_hash
            if (
                base.A.shape != (base.rl.size, base.lb.size)
                or base.ru.size != base.rl.size
                or base.ub.size != base.lb.size
                or base.n_eq + base.n_ub != base.A.shape[0]
                or np.any(np.isnan(base.rl))
                or np.any(np.isnan(base.ru))
                or np.any(np.isposinf(base.rl))
                or np.any(np.isneginf(base.ru))
                or not np.all(np.isfinite(base.lb))
                or not np.all(np.isfinite(base.ub))
                or np.any(base.lb > base.ub)
            ):
                raise OperatorHZBuildError(
                    "preactivation candidate base has invalid numerical data"
                )
            candidate_A, matrix_stats = _highs_candidate_csr(
                base.A,
                small_matrix_value=1.0e-12,
            )
            receipt["matrix"] = dict(matrix_stats)
            self._preactivation_require_time("candidate_model_filtered")

            h = _highspy.Highs()
            HS = _highspy.HighsStatus

            def require(status: Any, operation: str) -> None:
                if status != HS.kOk:
                    raise RuntimeError(
                        f"HiGHS {operation} returned {status}"
                    )

            require(h.setOptionValue("output_flag", False), "set output_flag")
            require(h.setOptionValue("presolve", "on"), "set presolve")
            require(
                h.setOptionValue("small_matrix_value", 1.0e-12),
                "set small_matrix_value",
            )
            require(
                h.setOptionValue("threads", _highs_process_threads()),
                "set threads",
            )
            ncol = int(base.A.shape[1])
            all_cols = np.arange(ncol, dtype=np.int32)
            require(
                h.addCols(
                    ncol,
                    np.zeros(ncol, dtype=np.float64),
                    np.asarray(base.lb, dtype=np.float64),
                    np.asarray(base.ub, dtype=np.float64),
                    0,
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.float64),
                ),
                "add columns",
            )
            if candidate_A.shape[0]:
                require(
                    h.addRows(
                        int(candidate_A.shape[0]),
                        np.asarray(base.rl, dtype=np.float64),
                        np.asarray(base.ru, dtype=np.float64),
                        int(candidate_A.nnz),
                        candidate_A.indptr.astype(np.int32),
                        candidate_A.indices.astype(np.int32),
                        candidate_A.data.astype(np.float64),
                    ),
                    "add rows",
                )
            if (
                int(h.getNumRow()) != int(candidate_A.shape[0])
                or int(h.getNumCol()) != int(candidate_A.shape[1])
                or int(h.getNumNz()) != int(candidate_A.nnz)
            ):
                raise RuntimeError(
                    "preactivation candidate model postcondition failed"
                )
            self._preactivation_original_hash(base)
            self._preactivation_require_time("candidate_model_after")
            receipt.update(
                {
                    "status": "ready",
                    "model_constructions": 1,
                    "loaded_rows": int(candidate_A.shape[0]),
                    "loaded_cols": int(candidate_A.shape[1]),
                    "loaded_nnz": int(candidate_A.nnz),
                }
            )
            self.preactivation_lp_model_builds += 1
            return (
                _PersistentPreactivationHighs(
                    highs=h,
                    base=base,
                    all_cols=all_cols,
                    receipt=receipt,
                ),
                receipt,
            )
        except _PreactivationLPDeadline:
            receipt["status"] = (
                f"deadline:{self.preactivation_lp_deadline_stage}"
            )
            return None, receipt
        except Exception as exc:
            receipt["status"] = (
                f"build_error:{type(exc).__name__}:{str(exc)[:120]}"
            )
            return None, receipt
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            self.preactivation_lp_candidate_seconds += elapsed
            receipt["build_seconds"] = elapsed

    def _preactivation_candidate_dual(
        self,
        model: _PersistentPreactivationHighs,
        objective: np.ndarray,
        *,
        time_slice: float,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Change only the objective and ask the persistent model for a dual."""

        receipt: Dict[str, Any] = {
            "solver": "highspy_persistent",
            "status": "not_started",
            "success": False,
            "candidate_only": True,
            "proof_authority": False,
            "time_slice": float(time_slice),
            "basis_reused": False,
        }
        started = time.monotonic()
        q = np.asarray(objective, dtype=np.float64).reshape(-1)
        try:
            if q.size != model.base.lb.size or not np.all(np.isfinite(q)):
                receipt["status"] = "invalid_objective"
                return None, receipt
            self._preactivation_require_time("candidate_before")
            self._preactivation_original_hash(model.base)
            remaining = self._preactivation_time_remaining()
            allowed = min(float(time_slice), remaining)
            if not np.isfinite(allowed) or allowed <= 0.0:
                raise _PreactivationLPDeadline("candidate_no_slice")
            solve_deadline = time.monotonic() + allowed
            h = model.highs
            HS = _highspy.HighsStatus
            require = self._require_highs_ok
            require(
                h.setOptionValue("time_limit", max(1.0e-6, allowed)),
                "set candidate time_limit",
            )
            require(
                h.changeColsCost(
                    int(model.all_cols.size),
                    model.all_cols,
                    -q,
                ),
                "change candidate objective",
            )
            if model.basis is not None:
                require(h.setBasis(model.basis), "restore candidate basis")
                receipt["basis_reused"] = True
            run_started = time.monotonic()
            run_status = h.run()
            run_finished = time.monotonic()
            model_status = h.getModelStatus()
            model.solve_count += 1
            receipt.update(
                {
                    "run_status": str(run_status),
                    "model_status": str(model_status),
                    "solve_seconds": max(0.0, run_finished - run_started),
                    "solve_count": int(model.solve_count),
                }
            )
            if (
                run_finished >= solve_deadline
                or self._preactivation_time_remaining() <= 0.0
            ):
                if self.preactivation_lp_deadline_stage is None:
                    self.preactivation_lp_deadline_stage = (
                        "candidate_solver_overrun"
                    )
                receipt["status"] = "discarded:solver_overrun"
                return None, receipt
            if run_status != HS.kOk:
                receipt["status"] = "rejected:run_non_ok"
                return None, receipt
            if model_status != _highspy.HighsModelStatus.kOptimal:
                receipt["status"] = "rejected:model_nonoptimal"
                return None, receipt
            solution = h.getSolution()
            if not bool(getattr(solution, "dual_valid", False)):
                receipt["status"] = "rejected:dual_invalid"
                return None, receipt
            row_dual = np.asarray(
                solution.row_dual, dtype=np.float64
            ).reshape(-1)
            if (
                row_dual.size != model.base.A.shape[0]
                or not np.all(np.isfinite(row_dual))
            ):
                receipt["status"] = "rejected:invalid_row_dual"
                return None, receipt
            basis = h.getBasis()
            if bool(getattr(basis, "valid", False)):
                model.basis = basis
            self._preactivation_original_hash(model.base)
            receipt["dual_nnz"] = int(np.count_nonzero(row_dual))
            receipt["status"] = "optimal_candidate"
            receipt["success"] = True
            return row_dual, receipt
        except _PreactivationLPDeadline as exc:
            receipt["status"] = f"deadline:{exc}"
            return None, receipt
        except Exception as exc:
            receipt["status"] = (
                f"candidate_error:{type(exc).__name__}:{str(exc)[:120]}"
            )
            return None, receipt
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            self.preactivation_lp_candidate_seconds += elapsed
            receipt["total_seconds"] = elapsed

    def _tighten_relu_bounds(
        self,
        layer_id: int,
        expr: _AffineExpr,
        cube_lower: np.ndarray,
        cube_upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Tighten selected unstable rows only through verified certificates."""

        lower = np.asarray(cube_lower, dtype=np.float64).copy()
        upper = np.asarray(cube_upper, dtype=np.float64).copy()
        summary: Dict[str, Any] = {
            "schema": "operator_hz_constrained_preactivation_v1",
            "enabled": bool(
                self.preactivation_lp_budget > 0
                and self.preactivation_lp_time_limit > 0.0
            ),
            "budget_total": int(self.preactivation_lp_budget),
            "budget_used_before": int(self.preactivation_lp_used),
            "rows_attempted": 0,
            "directions_attempted": 0,
            "directions_certified": 0,
            "lower_tightened": 0,
            "upper_tightened": 0,
            "rows_tightened": 0,
            "stabilized_active": 0,
            "stabilized_inactive": 0,
            "max_lower_improvement": 0.0,
            "max_upper_improvement": 0.0,
            "proof_authority": False,
            "candidate_solver_authority": False,
            "target_schedule_explicit": bool(
                self.preactivation_targets is not None
            ),
            "snapshot_seconds": 0.0,
            "candidate_seconds": 0.0,
            "certificate_seconds": 0.0,
            "deadline_stage": None,
            "receipts": [],
        }
        if not summary["enabled"]:
            summary["status"] = "disabled"
            return lower, upper, summary
        remaining_budget = (
            self.preactivation_lp_budget - self.preactivation_lp_used
        )
        if remaining_budget <= 0:
            summary["status"] = "budget_exhausted"
            return lower, upper, summary

        cube_unstable = np.flatnonzero(
            (cube_lower < 0.0) & (cube_upper > 0.0)
        ).astype(np.int64, copy=False)
        if not cube_unstable.size:
            summary["status"] = "no_cube_unstable_rows"
            return lower, upper, summary
        if self.preactivation_targets is None:
            eligible = cube_unstable
            requested_rows = None
        else:
            requested_rows = tuple(
                self.preactivation_targets.get(int(layer_id), ())
            )
            summary["requested_rows"] = list(requested_rows)
            unstable_set = set(int(row) for row in cube_unstable)
            eligible = np.asarray(
                [
                    int(row) for row in requested_rows
                    if 0 <= int(row) < expr.size
                    and int(row) in unstable_set
                ],
                dtype=np.int64,
            )
            summary["targets_not_cube_unstable"] = [
                int(row) for row in requested_rows
                if not (
                    0 <= int(row) < expr.size
                    and int(row) in unstable_set
                )
            ]
            if not eligible.size:
                summary["status"] = "no_eligible_target_rows"
                return lower, upper, summary

        if int(layer_id) in self.preactivation_layers_attempted:
            summary["status"] = "layer_already_attempted"
            return lower, upper, summary
        self.preactivation_layers_attempted.add(int(layer_id))
        self._start_preactivation_clock()
        if self._preactivation_time_remaining() <= 0.0:
            summary["status"] = "deadline"
            return lower, upper, summary

        snapshot_before = self.preactivation_lp_snapshot_seconds
        candidate_before = self.preactivation_lp_candidate_seconds
        certificate_before = self.preactivation_lp_certificate_seconds
        try:
            base = self._preactivation_lp_base()
        except _PreactivationLPDeadline:
            summary["status"] = "deadline"
            summary["deadline_stage"] = self.preactivation_lp_deadline_stage
            summary["snapshot_seconds"] = float(
                self.preactivation_lp_snapshot_seconds - snapshot_before
            )
            return lower, upper, summary
        summary["constraint_rows"] = int(base.A.shape[0])
        summary["constraint_nnz"] = int(base.A.nnz)
        summary["original_csr_sha256"] = base.csr_sha256
        summary["frame_variables"] = int(base.A.shape[1])
        summary["binary_variables_relaxed"] = int(self.n_bin)
        if base.A.shape[0] == 0:
            summary["status"] = "no_original_constraints"
            return lower, upper, summary

        selected = eligible[:remaining_budget]
        model, model_receipt = self._build_preactivation_candidate_model(base)
        summary["candidate_model"] = model_receipt
        if model is None:
            summary["status"] = "candidate_model_unavailable"
            summary["deadline_stage"] = self.preactivation_lp_deadline_stage
            summary["snapshot_seconds"] = float(
                self.preactivation_lp_snapshot_seconds - snapshot_before
            )
            summary["candidate_seconds"] = float(
                self.preactivation_lp_candidate_seconds - candidate_before
            )
            return lower, upper, summary

        for row_value in selected:
            if self._preactivation_time_remaining() <= 0.0:
                summary["status"] = "deadline"
                break
            row = int(row_value)
            self.preactivation_lp_used += 1
            summary["rows_attempted"] += 1
            old_lower = float(lower[row])
            old_upper = float(upper[row])
            row_receipt: Dict[str, Any] = {
                "row": row,
                "cube_lower": old_lower,
                "cube_upper": old_upper,
                "directions": {},
            }

            objective = np.zeros(base.lb.size, dtype=np.float64)
            local = expr.G.getrow(row)
            objective[local.indices] = local.data
            for label, sign in (("upper", 1.0), ("negated_lower", -1.0)):
                summary["directions_attempted"] += 1
                direction_slots = max(
                    1,
                    self.preactivation_lp_direction_capacity
                    - self.preactivation_lp_directions_used,
                )
                remaining = self._preactivation_time_remaining()
                fair_slice = remaining / float(direction_slots)
                self.preactivation_lp_directions_used += 1
                dual, candidate_receipt = self._preactivation_candidate_dual(
                    model,
                    sign * objective,
                    time_slice=fair_slice,
                )
                direction_receipt: Dict[str, Any] = {
                    "candidate": candidate_receipt,
                    "certificate": None,
                    "used": False,
                    "fair_slice": float(fair_slice),
                    "directions_left_including_this": int(direction_slots),
                }
                if (
                    dual is not None
                    and self._preactivation_time_remaining() > 0.0
                ):
                    try:
                        self._preactivation_require_time(
                            "certificate_before"
                        )
                        self._preactivation_original_hash(base)
                    except _PreactivationLPDeadline:
                        direction_receipt["certificate_discarded"] = (
                            "deadline_before"
                        )
                        row_receipt["directions"][label] = direction_receipt
                        summary["status"] = "deadline"
                        break
                    certificate_started = time.monotonic()
                    bound, cert_receipt = (
                        _independent_preactivation_lagrangian_upper(
                            expr,
                            row,
                            sign=sign,
                            base=base,
                            row_dual=dual,
                        )
                    )
                    certificate_elapsed = max(
                        0.0, time.monotonic() - certificate_started
                    )
                    self.preactivation_lp_certificate_seconds += (
                        certificate_elapsed
                    )
                    cert_receipt["certificate_seconds"] = certificate_elapsed
                    direction_receipt["certificate"] = cert_receipt
                    self._preactivation_original_hash(base)
                    if self._preactivation_time_remaining() <= 0.0:
                        if self.preactivation_lp_deadline_stage is None:
                            self.preactivation_lp_deadline_stage = (
                                "certificate_after"
                            )
                        direction_receipt["certificate_discarded"] = (
                            "deadline"
                        )
                        bound = None
                        summary["status"] = "deadline"
                    if bound is not None:
                        summary["directions_certified"] += 1
                        if label == "upper" and bound < upper[row]:
                            upper[row] = bound
                            summary["upper_tightened"] += 1
                            direction_receipt["used"] = True
                        elif label == "negated_lower":
                            certified_lower = -bound
                            if certified_lower > lower[row]:
                                lower[row] = certified_lower
                                summary["lower_tightened"] += 1
                                direction_receipt["used"] = True
                row_receipt["directions"][label] = direction_receipt
                if self._preactivation_time_remaining() <= 0.0:
                    break

            # A verified outer upper/lower pair cannot cross on a nonempty
            # graph.  Preserve the cube row if numerical diagnostics ever
            # contradict that invariant.
            if lower[row] > upper[row]:
                lower[row] = old_lower
                upper[row] = old_upper
                row_receipt["conflict"] = "certified_bounds_crossed_cube_restored"
            else:
                lower_gain = max(0.0, float(lower[row]) - old_lower)
                upper_gain = max(0.0, old_upper - float(upper[row]))
                if lower_gain > 0.0 or upper_gain > 0.0:
                    summary["rows_tightened"] += 1
                    summary["max_lower_improvement"] = max(
                        float(summary["max_lower_improvement"]), lower_gain
                    )
                    summary["max_upper_improvement"] = max(
                        float(summary["max_upper_improvement"]), upper_gain
                    )
            row_receipt["certified_lower"] = float(lower[row])
            row_receipt["certified_upper"] = float(upper[row])
            summary["receipts"].append(row_receipt)
            if self._preactivation_time_remaining() <= 0.0:
                break

        selected_rows = np.asarray(
            [int(item["row"]) for item in summary["receipts"]],
            dtype=np.int64,
        )
        if selected_rows.size:
            summary["stabilized_active"] = int(np.count_nonzero(
                (lower[selected_rows] >= 0.0)
                & (upper[selected_rows] > 0.0)
            ))
            summary["stabilized_inactive"] = int(np.count_nonzero(
                upper[selected_rows] <= 0.0
            ))
        summary["budget_used_after"] = int(self.preactivation_lp_used)
        self.preactivation_lp_time_spent = (
            0.0
            if self.preactivation_lp_started_at is None
            else max(
                0.0,
                time.monotonic() - self.preactivation_lp_started_at,
            )
        )
        summary["local_elapsed_seconds"] = float(
            self.preactivation_lp_time_spent
        )
        summary["snapshot_seconds"] = float(
            self.preactivation_lp_snapshot_seconds - snapshot_before
        )
        summary["candidate_seconds"] = float(
            self.preactivation_lp_candidate_seconds - candidate_before
        )
        summary["certificate_seconds"] = float(
            self.preactivation_lp_certificate_seconds - certificate_before
        )
        summary["persistent_model_constructions"] = int(
            model_receipt.get("model_constructions", 0)
        )
        summary["deadline_stage"] = self.preactivation_lp_deadline_stage
        summary["proof_authority"] = bool(summary["rows_tightened"])
        summary.setdefault("status", "completed")
        return lower, upper, summary

    # ------------------------------------------------------------------
    # Layer builders
    # ------------------------------------------------------------------

    def _build_input(self, layer: Any) -> Dict[str, Any]:
        if self.input_layer_id is not None:
            raise OperatorHZBuildError("strict operator-HZ supports exactly one INPUT")
        if self.net.preds.get(int(layer.id), []):
            raise OperatorHZBuildError(f"INPUT layer {layer.id} must have no predecessor")
        n = len(layer.out_vars)
        lb, ub = self._fact_box(
            self.after, int(layer.id), n, label="after"
        )
        # The input parametrization must match verify_once's seed exactly;
        # unlike internal cube bounds, it is not numerically widened.
        sink = self._constraint_program_sink
        if sink is None:
            # Preserve the established allocation/numeric operation order on
            # the default legacy path.
            full_ids = self._fresh_ids(n)
            center, radius = _enclosing_center_radius(
                lb, ub, name=f"input layer {layer.id}"
            )
            rows = np.flatnonzero(radius > 0.0).astype(
                np.int64, copy=False
            )
        else:
            center, radius = _enclosing_center_radius(
                lb, ub, name=f"input layer {layer.id}"
            )
            rows = np.flatnonzero(radius > 0.0).astype(
                np.int64, copy=False
            )
            full_ids = sink.reserve_input(n, rows)
        active_ids = full_ids[rows]
        start = self.n_cont
        self.col_ids.extend(active_ids.tolist())
        for stable_id in active_ids:
            self.cont_column_layer_by_id[int(stable_id)] = int(layer.id)
        self.n_cont += int(rows.size)
        cols = np.arange(start, self.n_cont, dtype=np.int64)
        G = sp.csr_matrix(
            (radius[rows], (rows, cols)),
            shape=(n, self.n_cont),
            dtype=np.float64,
        )
        self.exprs[int(layer.id)] = _AffineExpr(
            center,
            G,
            np.zeros(n, dtype=np.float64),
            affine_depth=0,
        )
        self.input_col_ids = full_ids
        self.input_lb = lb.copy()
        self.input_ub = ub.copy()
        self.input_center = center.copy()
        self.input_radius = radius.copy()
        self.input_layer_id = int(layer.id)
        return {
            "materialized": True,
            "new_cont": int(rows.size),
            "new_bin": 0,
            "new_eq": 0,
            "new_ub": 0,
        }

    def _validate_input_spec_enclosure(self, order: Sequence[Any]) -> None:
        """Check that the represented input box contains the BOX intersection."""

        if (
            self.input_lb is None
            or self.input_ub is None
            or self.input_col_ids is None
        ):
            raise OperatorHZBuildError("input box is unavailable for spec validation")
        spec_layers = [layer for layer in order if _kind(layer.kind) == "INPUT_SPEC"]
        if not spec_layers:
            raise OperatorHZBuildError("operator-HZ requires at least one INPUT_SPEC")
        lowers: List[np.ndarray] = []
        uppers: List[np.ndarray] = []
        for layer in spec_layers:
            if _kind(layer.params.get("kind", "")) != "BOX":
                raise OperatorHZBuildError(
                    f"INPUT_SPEC layer {layer.id} is not a BOX"
                )
            if "lb" not in layer.params or "ub" not in layer.params:
                raise OperatorHZBuildError(
                    f"BOX INPUT_SPEC layer {layer.id} lacks lb/ub"
                )
            lb = _as_finite_vector(
                layer.params["lb"], name=f"INPUT_SPEC[{layer.id}].lb"
            )
            ub = _as_finite_vector(
                layer.params["ub"], name=f"INPUT_SPEC[{layer.id}].ub"
            )
            if lb.size != self.input_col_ids.size or ub.size != self.input_col_ids.size:
                raise OperatorHZBuildError(
                    f"INPUT_SPEC layer {layer.id} has size {lb.size}/{ub.size}, "
                    f"expected {self.input_col_ids.size}"
                )
            if np.any(lb > ub):
                raise OperatorHZBuildError(
                    f"BOX INPUT_SPEC layer {layer.id} has lb > ub"
                )
            lowers.append(lb)
            uppers.append(ub)
        effective_lb = np.maximum.reduce(lowers)
        effective_ub = np.minimum.reduce(uppers)
        if np.any(effective_lb > effective_ub):
            raise OperatorHZBuildError(
                "BOX INPUT_SPEC intersection is empty; do not mark the graph known-nonempty"
            )
        misses = (self.input_lb > effective_lb) | (self.input_ub < effective_ub)
        if np.any(misses):
            row = int(np.flatnonzero(misses)[0])
            raise OperatorHZBuildError(
                f"represented input box does not enclose raw BOX intersection at "
                f"row {row}: represented=[{self.input_lb[row]}, {self.input_ub[row]}], "
                f"required=[{effective_lb[row]}, {effective_ub[row]}]"
            )

    def _build_identity(self, layer: Any, *, require_box_spec: bool) -> Dict[str, Any]:
        pred = self._preds(layer, 1)[0]
        if require_box_spec:
            spec_kind = _kind(layer.params.get("kind", ""))
            if spec_kind != "BOX":
                raise OperatorHZBuildError(
                    f"INPUT_SPEC layer {layer.id} kind {spec_kind!r} is unsupported; "
                    "strict operator-HZ currently accepts BOX only"
                )
        expr = self.exprs[pred]
        if len(layer.out_vars) != expr.size:
            raise OperatorHZBuildError(
                f"identity layer {layer.id} output size {len(layer.out_vars)} "
                f"does not match predecessor size {expr.size}"
            )
        self.exprs[int(layer.id)] = expr
        return {
            "materialized": False,
            "new_cont": 0,
            "new_bin": 0,
            "new_eq": 0,
            "new_ub": 0,
        }

    def _build_affine(self, layer: Any) -> Dict[str, Any]:
        pred = self._preds(layer, 1)[0]
        source = self.exprs[pred]
        implicit_cont = implicit_eq = implicit_ub = 0
        operator_csr_builder = "dense_sparse_exact_v1"

        try:
            if _kind(layer.kind) == "CONV2D":
                _validate_strict_conv2d_layer(layer)
                # Production currently uses the established vectorized path
                # with its explicit legacy admission fallback.  A cached
                # direct constructor is available only behind an internal,
                # default-false experiment switch until the authoritative
                # fused full-affine path and complete-stage speed gate pass.
                (
                    matrix,
                    bias,
                    operator_csr_builder,
                ) = _sparse_conv2d_matrix_from_layer_strict_with_mode(layer)
            else:
                matrix, bias = sparse_dense_matrix_from_layer(layer)
        except Exception as exc:
            raise OperatorHZBuildError(
                f"failed to build sparse {_kind(layer.kind)} operator at layer "
                f"{layer.id}: {type(exc).__name__}: {exc}"
            ) from exc

        live_affine_receipt: Optional[Dict[str, Any]] = None
        out: Optional[_AffineExpr] = None
        projection_skip_receipt = (
            self._projection_skip_chain(pred=pred, layer=layer)
            if source.affine_depth >= 1
            else None
        )
        if projection_skip_receipt is not None:
            self.projection_skip_chain_preservations.append(
                projection_skip_receipt
            )
            target_add_id = int(
                projection_skip_receipt["target_add_layer_id"]
            )
            downstream_affine_id = int(
                projection_skip_receipt["downstream_affine_layer_id"]
            )
            prior = self.projection_skip_required_downstream.get(
                target_add_id
            )
            if prior is not None and prior != downstream_affine_id:
                raise OperatorHZBuildError(
                    "projection skip chain assigned two downstream routes"
                )
            self.projection_skip_required_downstream[target_add_id] = (
                downstream_affine_id
            )
        if (
            source.affine_depth >= 1
            and not self.materialize_add
            and projection_skip_receipt is None
        ):
            out, live_affine_receipt = self._try_fuse_affine_into_relu(
                pred=pred,
                layer=layer,
                source=source,
                matrix=matrix,
                bias=bias,
            )
            self.live_affine_fusion_attempts.append(
                dict(live_affine_receipt)
            )
            required_affine = self.projection_skip_required_downstream.get(
                int(pred)
            )
            if required_affine == int(layer.id) and out is None:
                raise OperatorHZBuildError(
                    "projection skip chain downstream affine did not satisfy "
                    "the single live exact-ReLU path"
                )
        if (
            source.affine_depth >= 1
            and out is None
            and projection_skip_receipt is None
        ):
            source, implicit_cont, implicit_eq, implicit_ub = self._materialize(
                source,
                layer_id=pred,
                reason="affine_chain_cut",
            )
            self.exprs[pred] = source
        if out is None:
            out = self._affine(source, matrix, bias, layer_id=int(layer.id))
        correlation_shadow = self._prepare_affine_correlation_shadow(
            pred=pred,
            layer=layer,
            matrix=matrix,
            bias=bias,
        )
        phase_screen = self._prepare_residual_phase_screen(
            pred=pred,
            layer=layer,
            ordinary=out,
            matrix=matrix,
            bias=bias,
        )
        if len(layer.out_vars) != out.size:
            raise OperatorHZBuildError(
                f"{_kind(layer.kind)} layer {layer.id} output size mismatch: "
                f"operator={out.size}, out_vars={len(layer.out_vars)}"
            )
        self.exprs[int(layer.id)] = out
        return {
            "materialized": bool(implicit_cont or implicit_eq),
            "implicit_materialization": bool(implicit_cont or implicit_eq),
            "new_cont": int(implicit_cont),
            "new_bin": 0,
            "new_eq": int(implicit_eq),
            "new_ub": int(implicit_ub),
            "operator_nnz": int(matrix.nnz),
            "operator_csr_builder": operator_csr_builder,
            "projection_skip_chain": (
                projection_skip_receipt
                if projection_skip_receipt is not None
                else {
                    "schema": "operator_hz_projection_skip_chain_v1",
                    "status": "not_eligible",
                }
            ),
            "live_affine_relu": (
                live_affine_receipt
                if live_affine_receipt is not None
                else {
                    "schema": "operator_hz_live_affine_relu_v1",
                    "status": "not_requested",
                    "candidate_only": True,
                    "proof_authority": False,
                }
            ),
            "property_correlation_shadow": correlation_shadow,
            "residual_phase_screen": phase_screen,
            "roundoff_error_max": (
                float(np.max(out.err)) if out.err.size else 0.0
            ),
        }

    def _build_add(self, layer: Any) -> Dict[str, Any]:
        left_id, right_id = self._preds(layer, 2)
        summed = self._add_expr(
            self.exprs[left_id],
            self.exprs[right_id],
            layer_id=int(layer.id),
        )
        shadow_summed = summed
        if (
            self.residual_phase_screen
            or self.residual_bound_screen
            or self.correlation_targets is not None
        ):
            shadow_left = self.residual_skip_shadows.get(
                int(left_id), self.exprs[left_id]
            )
            shadow_right = self.residual_skip_shadows.get(
                int(right_id), self.exprs[right_id]
            )
            if (
                shadow_left is not self.exprs[left_id]
                or shadow_right is not self.exprs[right_id]
            ):
                shadow_summed = self._add_expr(
                    shadow_left,
                    shadow_right,
                    layer_id=int(layer.id),
                )
            self.residual_skip_shadows[int(layer.id)] = shadow_summed
        if len(layer.out_vars) != summed.size:
            raise OperatorHZBuildError(
                f"ADD layer {layer.id} output size {len(layer.out_vars)} "
                f"does not match operands {summed.size}"
            )
        capture_source = (
            int(layer.id) == self.property_tail_add_source_layer_id
        )
        if capture_source and self.property_tail_add_source_snapshot is not None:
            raise OperatorHZBuildError(
                "property-tail ADD source snapshot was constructed twice"
            )
        n_cont_before = int(self.n_cont)
        eq_blocks_before = self._eq_block_count()
        ub_blocks_before = self._ub_block_count()
        if int(layer.id) == self.property_suffix_stop_layer_id:
            if self.property_suffix_add_source_snapshot is not None:
                raise OperatorHZBuildError(
                    "property suffix ADD source snapshot was constructed twice"
                )
            if shadow_summed.G.shape[1] > n_cont_before:
                raise OperatorHZBuildError(
                    "property suffix ADD source depends on future columns"
                )
            self.property_suffix_add_source_snapshot = (
                _PropertySuffixAddSourceSnapshot(
                    add_layer_id=int(layer.id),
                    expression=_AffineExpr(
                        c=shadow_summed.c.copy(),
                        G=shadow_summed.G.copy().tocsr(),
                        err=shadow_summed.err.copy(),
                        affine_depth=int(shadow_summed.affine_depth),
                    ),
                    n_cont=int(n_cont_before),
                    n_bin=int(self.n_bin),
                    eq_rows=self._eq_row_count(),
                    ub_rows=self._ub_row_count(),
                    eq_block_count=int(eq_blocks_before),
                    ub_block_count=int(ub_blocks_before),
                )
            )
        if self.materialize_add:
            correlation_source_captured = (
                self._capture_correlation_add_source(
                    layer_id=int(layer.id),
                    expression=shadow_summed,
                )
            )
            out, new_cont, new_eq, new_ub = self._materialize(
                summed,
                layer_id=int(layer.id),
                reason="add_materialize",
            )
        else:
            correlation_source_captured = False
            out, new_cont, new_eq, new_ub = summed, 0, 0, 0
        if capture_source:
            if not self.materialize_add:
                raise OperatorHZBuildError(
                    "property-tail ADD source snapshot requires a "
                    "materialized ADD"
                )
            if summed.G.shape[1] > n_cont_before:
                raise OperatorHZBuildError(
                    "property-tail ADD source depends on future frame columns"
                )
            relation_blocks = tuple(self.ub_blocks[ub_blocks_before:])
            relation_block_rows = tuple(
                int(block.Ac.shape[0]) for block in relation_blocks
            )
            self.property_tail_add_source_snapshot = (
                _MaterializedAddSourceSnapshot(
                    add_layer_id=int(layer.id),
                    expression=_AffineExpr(
                        c=summed.c.copy(),
                        G=summed.G.copy().tocsr(),
                        err=summed.err.copy(),
                        affine_depth=int(summed.affine_depth),
                    ),
                    n_cont_before=n_cont_before,
                    n_cont_after=int(self.n_cont),
                    n_bin=int(self.n_bin),
                    eq_block_count_before=eq_blocks_before,
                    eq_block_count_after=self._eq_block_count(),
                    ub_block_count_before=ub_blocks_before,
                    ub_block_count_after=self._ub_block_count(),
                    relation_block_rows=relation_block_rows,
                    relation_block_tags=tuple(
                        str(block.tag) for block in relation_blocks
                    ),
                    relation_blocks_sha256=(
                        _constraint_blocks_sha256(relation_blocks)
                    ),
                    new_cont=int(new_cont),
                    new_ub=int(new_ub),
                )
            )
        self.exprs[int(layer.id)] = out
        return {
            "materialized": bool(self.materialize_add),
            "property_tail_source_captured": bool(capture_source),
            "property_correlation_source_captured": bool(
                correlation_source_captured
            ),
            "residual_skip_shadow_recursive": bool(
                shadow_summed is not summed
            ),
            "new_cont": int(new_cont),
            "new_bin": 0,
            "new_eq": int(new_eq),
            "new_ub": int(new_ub),
            "roundoff_error_max": (
                float(np.max(summed.err)) if summed.err.size else 0.0
            ),
        }

    def _take_exact(self, count: int) -> int:
        count = int(count)
        if self.exact_budget == -1:
            take = count
        elif self.exact_budget == 0:
            take = 0
        else:
            take = min(count, max(0, self.exact_budget - self.exact_used))
        self.exact_used += int(take)
        return int(take)

    def _build_relu(self, layer: Any) -> Dict[str, Any]:
        pred = self._preds(layer, 1)[0]
        x = self._align(self.exprs[pred])
        if len(layer.out_vars) != x.size:
            raise OperatorHZBuildError(
                f"RELU layer {layer.id} output size {len(layer.out_vars)} "
                f"does not match input {x.size}"
            )

        cube_l, cube_u = self._cube_bounds(x)
        phase_l, phase_u, phase_screen_receipt = (
            self._apply_residual_phase_screen(
                layer_id=int(layer.id),
                cube_lower=cube_l,
                cube_upper=cube_u,
            )
        )
        correlation_l, correlation_u, correlation_receipt = (
            self._correlation_shadow_bounds(
                layer_id=int(layer.id),
                pred=pred,
                cube_lower=phase_l,
                cube_upper=phase_u,
            )
        )
        l, u, preactivation_receipt = self._tighten_relu_bounds(
            int(layer.id), x, correlation_l, correlation_u
        )
        verified_query_dual_bound: Optional[Dict[str, Any]] = None
        if (
            self.verified_query_dual_feedback is not None
            and int(layer.id) in self.verified_query_dual_target_ids
        ):
            query_l, query_u = self.verified_query_dual_bounds[
                int(layer.id)
            ]
            local_l = np.ascontiguousarray(l, dtype=np.float64)
            local_u = np.ascontiguousarray(u, dtype=np.float64)
            l, u = _intersect_verified_query_dual_box(
                local_l,
                local_u,
                query_l,
                query_u,
                layer_id=int(layer.id),
            )
            feedback_receipt = self.verified_query_dual_receipt
            if feedback_receipt is None:
                raise OperatorHZBuildError(
                    "verified query-dual receipt snapshot is unavailable"
                )
            verified_query_dual_bound = {
                "schema": "operator_hz_verified_query_dual_relu_bound_v1",
                "proof_authority": True,
                "layer_id": int(layer.id),
                "bound_sha256": _f64_array_sha256(
                    np.stack(
                        [
                            np.ascontiguousarray(query_l),
                            np.ascontiguousarray(query_u),
                        ],
                        axis=0,
                    )
                ),
                "root_boxes_sha256": feedback_receipt[
                    "root_boxes_sha256"
                ],
                "final_boxes_sha256": feedback_receipt[
                    "final_boxes_sha256"
                ],
                "property_spec_sha256": feedback_receipt[
                    "property_spec_sha256"
                ],
                "property_upper_sha256": feedback_receipt[
                    "property_upper_sha256"
                ],
                "transaction_receipt_sha256": feedback_receipt[
                    "receipt_sha256"
                ],
                "lower_improved_rows": int(np.count_nonzero(l > local_l)),
                "upper_improved_rows": int(np.count_nonzero(u < local_u)),
                "local_lower_sha256": _f64_array_sha256(local_l),
                "local_upper_sha256": _f64_array_sha256(local_u),
                "intersected_lower_sha256": _f64_array_sha256(l),
                "intersected_upper_sha256": _f64_array_sha256(u),
            }
        self.verified_preactivation_bounds[int(layer.id)] = (
            np.ascontiguousarray(l, dtype=np.float64).copy(),
            np.ascontiguousarray(u, dtype=np.float64).copy(),
        )
        inactive = np.flatnonzero(u <= 0.0).astype(np.int64, copy=False)
        # At the degenerate point l == u == 0 both conventional stable tests
        # hold.  Assign it to the inactive phase only so the partition remains
        # disjoint and ReLU(0) stays an exact point.
        active = np.flatnonzero((l >= 0.0) & (u > 0.0)).astype(
            np.int64, copy=False
        )
        unstable = np.flatnonzero((l < 0.0) & (u > 0.0)).astype(
            np.int64, copy=False
        )
        if active.size + inactive.size + unstable.size != x.size:
            raise OperatorHZBuildError(
                f"RELU phase partition is incomplete at layer {layer.id}"
            )
        target_modes = (
            {}
            if self.residual_targets is None
            else self.residual_targets.get(int(layer.id), {})
        )
        retain_property_tail_exact_graph = bool(
            int(layer.id) == self.property_tail_relu_layer_id
            and self.exact_budget > 0
            and target_modes
        )
        if (
            int(layer.id) == self.property_tail_relu_layer_id
            and not retain_property_tail_exact_graph
        ):
            if self.property_tail_snapshot is not None:
                raise OperatorHZBuildError(
                    "property upper tail snapshot was constructed twice"
                )
            self.property_tail_snapshot = _PropertyTailSnapshot(
                relu_layer_id=int(layer.id),
                preactivation=x,
                lower=l.copy(),
                upper=u.copy(),
                n_cont=int(self.n_cont),
                n_bin=int(self.n_bin),
                eq_block_count=self._eq_block_count(),
                ub_block_count=self._ub_block_count(),
                exact_used=int(self.exact_used),
            )

        if self.residual_targets is not None and int(layer.id) in self.residual_targets:
            self.residual_target_layers_seen.add(int(layer.id))
        for row in target_modes:
            if int(row) >= x.size:
                raise OperatorHZBuildError(
                    f"residual target ({layer.id}, {row}) is out of range "
                    f"for ReLU width {x.size}"
                )
        property_guided_exact = bool(
            self.exact_budget > 0
            and self.residual_targets is not None
        )
        exact_reservoir_receipt: Dict[str, Any] = {
            "schema": "operator_hz_exact_target_reservoir_v1",
            "enabled": False,
            "status": "not_requested",
            "candidate_only": True,
            "proof_authority": False,
        }
        selected_reserve: Tuple[int, ...] = ()
        reserve_replacement: Dict[int, int] = {}
        if property_guided_exact and self.exact_target_reservoir is not None:
            unstable_set = set(int(row) for row in unstable.tolist())
            remaining_exact = max(
                0, self.exact_budget - self.exact_used
            )
            primary_rows = tuple(int(row) for row in target_modes)
            reserve_rows = self.exact_target_reservoir.get(
                int(layer.id), ()
            )
            layer_quota = len(primary_rows)
            candidate_rows = (*primary_rows, *reserve_rows)
            tightened_rows = set(
                int(row)
                for row in phase_screen_receipt.get("rows", ())
            )
            cube_unstable_set = set(
                int(row)
                for row in np.flatnonzero(
                    (cube_l < 0.0) & (cube_u > 0.0)
                ).tolist()
            )
            phase_stable_set = set(
                int(row)
                for row in np.flatnonzero(
                    (phase_u <= 0.0)
                    | ((phase_l >= 0.0) & (phase_u > 0.0))
                ).tolist()
            )
            post_screen_stable_primary = tuple(
                row for row in primary_rows if row in phase_stable_set
            )
            rbs_stabilized_primary = tuple(
                row for row in post_screen_stable_primary
                if row in cube_unstable_set and row in tightened_rows
            )
            non_rbs_stable_primary = tuple(
                row for row in primary_rows
                if row not in unstable_set
                and row not in rbs_stabilized_primary
            )
            selected_primary_rows = [
                int(row) for row in primary_rows
                if int(row) in unstable_set
            ][:remaining_exact]
            remaining_after_primary = max(
                0, remaining_exact - len(selected_primary_rows)
            )
            reserve_take = min(
                len(rbs_stabilized_primary), remaining_after_primary
            )
            selected_reserve_rows = [
                int(row) for row in reserve_rows
                if int(row) in unstable_set
            ][:reserve_take]
            exact_rows = np.asarray(
                [*selected_primary_rows, *selected_reserve_rows],
                dtype=np.int64,
            )
            self.exact_used += int(exact_rows.size)
            exact_mask = np.zeros(x.size, dtype=bool)
            exact_mask[exact_rows] = True
            relaxed_rows = unstable[~exact_mask[unstable]]
            selected_set = set(int(row) for row in exact_rows.tolist())
            selected_primary = tuple(
                row for row in primary_rows if row in selected_set
            )
            selected_reserve = tuple(
                row for row in reserve_rows if row in selected_set
            )
            reserve_replacement = {
                int(reserve): int(primary)
                for primary, reserve in zip(
                    rbs_stabilized_primary, selected_reserve
                )
            }
            stabilized_active = set(int(row) for row in active.tolist())
            stabilized_inactive = set(int(row) for row in inactive.tolist())
            shortfall = int(layer_quota - exact_rows.size)
            exact_reservoir_receipt = {
                "schema": "operator_hz_exact_target_reservoir_v1",
                "enabled": True,
                "status": (
                    "filled"
                    if shortfall == 0
                    else "post_screen_reservoir_exhausted"
                ),
                "candidate_only": True,
                "proof_authority": False,
                "relu_layer_id": int(layer.id),
                "layer_quota": int(layer_quota),
                "global_remaining_before": int(remaining_exact),
                "primary_rows": list(primary_rows),
                "reserve_rows": [int(row) for row in reserve_rows],
                "candidate_rows_sha256": _canonical_json_sha256(
                    [int(row) for row in candidate_rows]
                ),
                "pre_screen_cube_unstable_primary": [
                    row for row in primary_rows if row in cube_unstable_set
                ],
                "primary_rows_rbs_tightened": [
                    row for row in primary_rows if row in tightened_rows
                ],
                "all_primary_rows_rbs_tightened": bool(
                    primary_rows
                    and all(row in tightened_rows for row in primary_rows)
                ),
                "post_screen_unstable_primary": [
                    row for row in primary_rows
                    if row in cube_unstable_set
                    and row not in phase_stable_set
                ],
                "post_screen_stabilized_active_primary": [
                    row for row in primary_rows if row in stabilized_active
                ],
                "post_screen_stabilized_inactive_primary": [
                    row for row in primary_rows if row in stabilized_inactive
                ],
                "rbs_newly_stabilized_primary": [
                    int(row) for row in rbs_stabilized_primary
                ],
                "non_rbs_stable_primary_not_replaced": [
                    int(row) for row in non_rbs_stable_primary
                ],
                "post_screen_unstable_reserve": [
                    int(row) for row in reserve_rows
                    if int(row) in unstable_set
                ],
                "post_screen_stable_reserve_skipped": [
                    int(row) for row in reserve_rows
                    if int(row) not in unstable_set
                ],
                "selected_primary_rows": list(selected_primary),
                "selected_reserve_rows": list(selected_reserve),
                "selected_rows": [
                    int(row) for row in exact_rows.tolist()
                ],
                "selected_rows_sha256": _canonical_json_sha256(
                    [int(row) for row in exact_rows.tolist()]
                ),
                "replacement_count": int(len(selected_reserve)),
                "replacement_slots": [
                    {
                        "stabilized_primary_row": int(primary),
                        "selected_reserve_row": int(reserve),
                    }
                    for primary, reserve in zip(
                        rbs_stabilized_primary, selected_reserve
                    )
                ],
                "reservoir_consulted": bool(rbs_stabilized_primary),
                "selected_rows_rbs_tightened": [
                    int(row) for row in exact_rows.tolist()
                    if int(row) in tightened_rows
                ],
                "all_selected_rows_rbs_tightened": bool(
                    exact_rows.size > 0
                    and all(
                        int(row) in tightened_rows
                        for row in exact_rows.tolist()
                    )
                ),
                "shortfall": shortfall,
                "same_layer_only": True,
                "unselected_reserves_use_ordinary_triangle": True,
                "selected_rows_use_existing_exact_big_m": True,
                "selection_has_no_bound_or_verdict_authority": True,
            }
            if primary_rows or reserve_rows:
                self.exact_target_reservoir_receipts.append(
                    dict(exact_reservoir_receipt)
                )
        elif property_guided_exact:
            unstable_set = set(int(row) for row in unstable.tolist())
            remaining_exact = max(
                0, self.exact_budget - self.exact_used
            )
            exact_rows = np.asarray(
                [
                    int(row)
                    for row in target_modes
                    if int(row) in unstable_set
                ][:remaining_exact],
                dtype=np.int64,
            )
            self.exact_used += int(exact_rows.size)
            exact_mask = np.zeros(x.size, dtype=bool)
            exact_mask[exact_rows] = True
            relaxed_rows = unstable[~exact_mask[unstable]]
        else:
            exact_count = self._take_exact(int(unstable.size))
            exact_rows = unstable[:exact_count]
            relaxed_rows = unstable[exact_count:]

        relaxed_mask = np.zeros(x.size, dtype=bool)
        relaxed_mask[relaxed_rows] = True
        residual_rows = np.asarray(
            [int(row) for row in target_modes if relaxed_mask[int(row)]],
            dtype=np.int64,
        )
        residual_mask = np.zeros(x.size, dtype=bool)
        residual_mask[residual_rows] = True
        ordinary_relaxed_rows = relaxed_rows[~residual_mask[relaxed_rows]]
        if (
            active.size
            + inactive.size
            + exact_rows.size
            + ordinary_relaxed_rows.size
            + residual_rows.size
            != x.size
        ):
            raise OperatorHZBuildError(
                f"ReLU {layer.id} residual partition is incomplete"
            )

        triangle_slope = np.zeros(relaxed_rows.size, dtype=np.float64)
        triangle_intercept = np.zeros(relaxed_rows.size, dtype=np.float64)
        triangle_inflation = np.zeros(relaxed_rows.size, dtype=np.float64)
        if relaxed_rows.size:
            triangle_slope, triangle_intercept, triangle_inflation = (
                _relu_triangle_parameters(l[relaxed_rows], u[relaxed_rows])
            )
        triangle_inflation_max = (
            float(np.max(triangle_inflation))
            if triangle_inflation.size else 0.0
        )

        y_lb = np.zeros(x.size, dtype=np.float64)
        y_ub = np.zeros(x.size, dtype=np.float64)
        y_lb[active] = l[active]
        y_ub[active] = u[active]
        y_ub[unstable] = u[unstable]
        # A residual row is represented by rho, not by an independent y
        # factor.  Zeroing it here ensures there is exactly one fresh local
        # factor for the node and no disconnected column.
        y_lb[residual_rows] = 0.0
        y_ub[residual_rows] = 0.0
        y, new_cont = self._box_expr(y_lb, y_ub)

        residual_slope = np.zeros(residual_rows.size, dtype=np.float64)
        residual_intercept = np.zeros(residual_rows.size, dtype=np.float64)
        residual_factor_columns = np.zeros(residual_rows.size, dtype=np.int64)
        if residual_rows.size:
            positions = np.searchsorted(relaxed_rows, residual_rows)
            if (
                np.any(positions < 0)
                or np.any(positions >= relaxed_rows.size)
                or np.any(relaxed_rows[positions] != residual_rows)
            ):
                raise OperatorHZBuildError(
                    f"residual target partition failed at ReLU {layer.id}"
                )
            residual_slope = triangle_slope[positions]
            residual_intercept = triangle_intercept[positions]
            if (
                not np.all(np.isfinite(residual_slope))
                or not np.all(np.isfinite(residual_intercept))
                or np.any(residual_slope < 0.0)
                or np.any(residual_slope > 1.0)
                or np.any(residual_intercept <= 0.0)
            ):
                raise OperatorHZBuildError(
                    f"invalid residual envelope at ReLU {layer.id}"
                )
            rho, rho_cont = self._box_expr(
                np.zeros(residual_rows.size, dtype=np.float64),
                residual_intercept,
            )
            if int(rho_cont) != int(residual_rows.size):
                raise OperatorHZBuildError(
                    f"ReLU {layer.id} did not allocate exactly one residual "
                    "factor per targeted row"
                )
            for local_row in range(residual_rows.size):
                columns = rho.G.getrow(local_row).indices
                if columns.size != 1:
                    raise OperatorHZBuildError(
                        f"ReLU {layer.id} residual row {residual_rows[local_row]} "
                        "does not have exactly one factor"
                    )
                residual_factor_columns[local_row] = int(columns[0])
            xr = self._rows(self._align(x), residual_rows)
            residual_expr = self._difference(
                rho,
                xr,
                right_scale=-residual_slope,
            )
            # ``affine_depth`` is a sparsity-control counter, not semantic
            # provenance.  A bounded number of explicitly targeted rows may
            # flow through the next affine operator so their shared rho
            # correlation can reach the next ReLU/property.  Marking the
            # entire mixed layer as depth one would force a full-width chain
            # cut and erase the intended row-local saving.  The next affine
            # itself sets depth one, so a second consecutive affine still
            # triggers the ordinary stop-loss materialization.
            residual_expr = _AffineExpr(
                residual_expr.c,
                residual_expr.G,
                residual_expr.err,
                affine_depth=0,
            )
            y = self._replace_rows(y, residual_rows, residual_expr)
            new_cont += int(rho_cont)

        new_eq = 0
        new_ub = 0
        if active.size:
            new_ub += self._append_equality(
                self._rows(y, active),
                self._rows(x, active),
                tag=f"relu_active:{layer.id}",
                layer_id=int(layer.id),
            )

        if ordinary_relaxed_rows.size:
            positions = np.searchsorted(relaxed_rows, ordinary_relaxed_rows)
            xr = self._rows(x, ordinary_relaxed_rows)
            yr = self._rows(y, ordinary_relaxed_rows)
            # x - y <= 0
            new_ub += self._append_upper(
                self._difference(xr, yr),
                np.zeros(ordinary_relaxed_rows.size, dtype=np.float64),
                tag=f"relu_relaxed_lower:{layer.id}",
                layer_id=int(layer.id),
            )
            # y - slope*x <= intercept.  The intercept is the smallest
            # binary64 value proven (with exact dyadic arithmetic) to dominate
            # the l,0,u endpoint requirements for the stored slope.
            new_ub += self._append_upper(
                self._difference(
                    yr,
                    xr,
                    right_scale=triangle_slope[positions],
                ),
                triangle_intercept[positions],
                tag=f"relu_relaxed_upper_fraction:{layer.id}",
                layer_id=int(layer.id),
            )

        zero_guard_rows = np.asarray(
            [
                int(row)
                for row in residual_rows
                if target_modes[int(row)] in {"zero", "both"}
            ],
            dtype=np.int64,
        )
        identity_guard_rows = np.asarray(
            [
                int(row)
                for row in residual_rows
                if target_modes[int(row)] in {"identity", "both"}
            ],
            dtype=np.int64,
        )
        if zero_guard_rows.size:
            yr = self._rows(y, zero_guard_rows)
            zero = _AffineExpr(
                c=np.zeros(zero_guard_rows.size, dtype=np.float64),
                G=sp.csr_matrix(
                    (zero_guard_rows.size, self.n_cont),
                    dtype=np.float64,
                ),
                err=np.zeros(zero_guard_rows.size, dtype=np.float64),
                affine_depth=0,
            )
            new_ub += self._append_upper(
                self._difference(zero, yr),
                np.zeros(zero_guard_rows.size, dtype=np.float64),
                tag=f"relu_residual_zero_guard:{layer.id}",
                layer_id=int(layer.id),
            )
        if identity_guard_rows.size:
            new_ub += self._append_upper(
                self._difference(
                    self._rows(x, identity_guard_rows),
                    self._rows(y, identity_guard_rows),
                ),
                np.zeros(identity_guard_rows.size, dtype=np.float64),
                tag=f"relu_residual_identity_guard:{layer.id}",
                layer_id=int(layer.id),
            )

        residual_position = {
            int(row): position
            for position, row in enumerate(residual_rows.tolist())
        }
        active_set = set(int(row) for row in active.tolist())
        inactive_set = set(int(row) for row in inactive.tolist())
        exact_set = set(int(row) for row in exact_rows.tolist())
        for row, guard in target_modes.items():
            if int(row) in residual_position:
                position = residual_position[int(row)]
                receipt = {
                    "layer_id": int(layer.id),
                    "row": int(row),
                    "guard": str(guard),
                    "status": "applied",
                    "lower": float(l[int(row)]),
                    "upper": float(u[int(row)]),
                    "slope": float(residual_slope[position]),
                    "intercept": float(residual_intercept[position]),
                    "factor_column": int(residual_factor_columns[position]),
                    "proof_authority": (
                        "independent_cube_or_certified_preactivation+"
                        "fraction_endpoint_envelope"
                    ),
                }
            elif int(row) in exact_set:
                receipt = {
                    "layer_id": int(layer.id),
                    "row": int(row),
                    "guard": str(guard),
                    "status": "skipped_exact",
                }
            elif int(row) in active_set:
                receipt = {
                    "layer_id": int(layer.id),
                    "row": int(row),
                    "guard": str(guard),
                    "status": "skipped_active",
                }
            elif int(row) in inactive_set:
                receipt = {
                    "layer_id": int(layer.id),
                    "row": int(row),
                    "guard": str(guard),
                    "status": "skipped_inactive",
                }
            else:
                raise OperatorHZBuildError(
                    f"residual target ({layer.id}, {row}) was not classified"
                )
            self.residual_target_receipts.append(receipt)

        exact_binary_positions = np.zeros(0, dtype=np.int64)
        if exact_rows.size:
            self.exact_phase_record_count += int(exact_rows.size)
            exact_binary_positions = self._allocate_bin(int(exact_rows.size))
            pending_exact_records: List[Dict[str, Any]] = []
            if self.collect_property_exact_phase_records:
                for local_index, (exact_row, binary_position) in enumerate(
                    zip(exact_rows.tolist(), exact_binary_positions.tolist())
                ):
                    exact_key = (int(layer.id), int(exact_row))
                    pending_exact_records.append(
                        {
                            "layer_id": int(layer.id),
                            "row": int(exact_row),
                            "rival_ids": self.property_phase_focus_rivals.get(
                                exact_key,
                                tuple(
                                    range(
                                        0
                                        if self.property_upper_C is None
                                        else int(
                                            self.property_upper_C.shape[0]
                                        )
                                    )
                                ),
                            ),
                            "binary_position": int(binary_position),
                            "binary_col_id": int(
                                self.bcol_ids[int(binary_position)]
                            ),
                            "exact_order": int(
                                self.exact_used - exact_rows.size + local_index
                            ),
                            "property_selected": bool(
                                property_guided_exact
                                and (
                                    int(exact_row) in target_modes
                                    or int(exact_row) in selected_reserve
                                )
                            ),
                            "exact_selection_role": (
                                "same_layer_rbs_reserve"
                                if int(exact_row) in selected_reserve
                                else (
                                    "property_primary"
                                    if property_guided_exact
                                    else "topological_prefix"
                                )
                            ),
                            "replaces_stabilized_primary_row": (
                                reserve_replacement.get(int(exact_row))
                            ),
                            "focused_rivals_explicit": bool(
                                exact_key in self.property_phase_focus_rivals
                            ),
                        }
                    )
            xe = self._rows(x, exact_rows)
            ye = self._rows(y, exact_rows)
            le = l[exact_rows]
            ue = u[exact_rows]
            lower_half = -0.5 * le
            upper_half = 0.5 * ue
            if (
                not np.all(np.isfinite(lower_half))
                or not np.all(np.isfinite(upper_half))
                or np.any(2.0 * lower_half != -le)
                or np.any(2.0 * upper_half != ue)
            ):
                raise OperatorHZBuildError(
                    f"exact ReLU layer {layer.id} has a Big-M half coefficient "
                    "which is not exactly representable"
                )
            local_rows = np.arange(exact_rows.size, dtype=np.int64)
            lower_expr, x_branch_expr = self._opposite_differences(xe, ye)

            # x - y <= 0
            lower_row_start = self._ub_row_count()
            lower_row_count = self._append_upper(
                lower_expr,
                np.zeros(exact_rows.size, dtype=np.float64),
                tag=f"relu_exact_lower:{layer.id}",
                layer_id=int(layer.id),
            )
            if lower_row_count != int(exact_rows.size):
                raise OperatorHZBuildError(
                    f"exact ReLU layer {layer.id} lower-row mapping is not "
                    "one-to-one"
                )
            new_ub += lower_row_count
            # y - x - (l/2) xi_b <= -l/2
            x_branch_row_start = self._ub_row_count()
            x_branch_row_count = self._append_upper(
                x_branch_expr,
                lower_half,
                tag=f"relu_exact_x_branch:{layer.id}",
                layer_id=int(layer.id),
                binary_rows=local_rows,
                binary_cols=exact_binary_positions,
                binary_data=lower_half,
            )
            if x_branch_row_count != int(exact_rows.size):
                raise OperatorHZBuildError(
                    f"exact ReLU layer {layer.id} x-branch row mapping is "
                    "not one-to-one"
                )
            new_ub += x_branch_row_count
            # y - (u/2) xi_b <= u/2
            zero_branch_row_start = self._ub_row_count()
            zero_branch_row_count = self._append_upper(
                ye,
                upper_half,
                tag=f"relu_exact_zero_branch:{layer.id}",
                layer_id=int(layer.id),
                binary_rows=local_rows,
                binary_cols=exact_binary_positions,
                binary_data=-upper_half,
            )
            if zero_branch_row_count != int(exact_rows.size):
                raise OperatorHZBuildError(
                    f"exact ReLU layer {layer.id} zero-branch row mapping is "
                    "not one-to-one"
                )
            new_ub += zero_branch_row_count
            for local_index, record in enumerate(pending_exact_records):
                record["exact_upper_rows"] = {
                    "lower": int(lower_row_start + local_index),
                    "x_branch": int(x_branch_row_start + local_index),
                    "zero_branch": int(
                        zero_branch_row_start + local_index
                    ),
                }
                self.property_exact_phase_records.append(record)

        if retain_property_tail_exact_graph:
            if self.property_tail_snapshot is not None:
                raise OperatorHZBuildError(
                    "property upper tail snapshot was constructed twice"
                )
            self.property_tail_snapshot = _PropertyTailSnapshot(
                relu_layer_id=int(layer.id),
                preactivation=x,
                lower=l.copy(),
                upper=u.copy(),
                n_cont=int(self.n_cont),
                n_bin=int(self.n_bin),
                eq_block_count=self._eq_block_count(),
                ub_block_count=self._ub_block_count(),
                exact_used=int(self.exact_used),
            )

        exact_key = ",".join(
            f"{int(layer.id)}:{int(row)}" for row in exact_rows.tolist()
        ).encode("ascii")
        if verified_query_dual_bound is not None:
            preactivation_bound_source = (
                "verified_query_dual_replay_intersection"
            )
        elif preactivation_receipt.get("proof_authority"):
            preactivation_bound_source = (
                "verified_original_constraint_lagrangian"
            )
        elif correlation_receipt.get("proof_authority"):
            preactivation_bound_source = (
                "property_conditioned_add_affine_correlation_shadow"
            )
        elif phase_screen_receipt.get("proof_authority"):
            preactivation_bound_source = (
                "residual_add_affine_phase_screen"
            )
        else:
            preactivation_bound_source = "independent_cube"
        self.exprs[int(layer.id)] = y
        return {
            "materialized": True,
            "new_cont": int(new_cont),
            "new_bin": int(exact_rows.size),
            "new_eq": int(new_eq),
            "new_ub": int(new_ub),
            "relu_active": int(active.size),
            "relu_inactive": int(inactive.size),
            "relu_unstable": int(unstable.size),
            "relu_exact": int(exact_rows.size),
            "relu_exact_selection": (
                "property_gap_adjoint_same_layer_rbs_reservoir_v1"
                if self.exact_target_reservoir is not None
                else (
                    "property_gap_adjoint_facility_targets_v1"
                    if property_guided_exact
                    else "topological_prefix_v1"
                )
            ),
            "relu_relaxed": int(relaxed_rows.size),
            "relu_triangle_rows": int(ordinary_relaxed_rows.size),
            "relu_residual_rows": int(residual_rows.size),
            "relu_residual_guard_none": int(
                sum(target_modes[int(row)] == "none" for row in residual_rows)
            ),
            "relu_residual_guard_zero": int(
                sum(target_modes[int(row)] == "zero" for row in residual_rows)
            ),
            "relu_residual_guard_identity": int(
                sum(
                    target_modes[int(row)] == "identity"
                    for row in residual_rows
                )
            ),
            "relu_residual_guard_both": int(
                sum(target_modes[int(row)] == "both" for row in residual_rows)
            ),
            "relu_residual_expected_rows_saved": int(
                sum(
                    2
                    - (
                        0
                        if target_modes[int(row)] == "none"
                        else 2
                        if target_modes[int(row)] == "both"
                        else 1
                    )
                    for row in residual_rows
                )
            ),
            "relu_residual_index_preview": [
                int(value) for value in residual_rows[:16]
            ],
            "relu_triangle_upper": "fraction_endpoint_envelope_v1",
            "relu_triangle_intercept_inflation_max": triangle_inflation_max,
            "preactivation_bound_source": preactivation_bound_source,
            "preactivation_constrained_lp": preactivation_receipt,
            "property_correlation_shadow": correlation_receipt,
            "residual_phase_screen": phase_screen_receipt,
            "exact_target_reservoir": exact_reservoir_receipt,
            "preactivation_cube_lb_min": float(np.min(cube_l)),
            "preactivation_cube_ub_max": float(np.max(cube_u)),
            "preactivation_correlation_lb_min": float(
                np.min(correlation_l)
            ),
            "preactivation_correlation_ub_max": float(
                np.max(correlation_u)
            ),
            "preactivation_phase_screen_lb_min": float(
                np.min(phase_l)
            ),
            "preactivation_phase_screen_ub_max": float(
                np.max(phase_u)
            ),
            "preactivation_certified_lb_min": float(np.min(l)),
            "preactivation_certified_ub_max": float(np.max(u)),
            "exact_index_preview": [int(v) for v in exact_rows[:16]],
            "exact_index_sha256": hashlib.sha256(exact_key).hexdigest(),
            **(
                {
                    "verified_query_dual_bound": (
                        verified_query_dual_bound
                    )
                }
                if verified_query_dual_bound is not None
                else {}
            ),
        }

    def _property_suffix_dominating_add_candidates(
        self,
        *,
        output_layer_id: int,
    ) -> Tuple[int, ...]:
        """Return every dominating ADD, ordered nearest to farthest."""
        return operator_hz_property_suffix_dominating_add_candidates(
            self.net,
            output_layer_id=int(output_layer_id),
        )

    def _property_suffix_stop_layer(
        self,
        *,
        output_layer_id: int,
    ) -> Tuple[int, Tuple[int, ...]]:
        """Select an earlier ADD which dominates the complete property suffix.

        The nearest dominating ADD is deliberately skipped: folding only the
        final residual join largely duplicates the ordinary final-ReLU tail.
        ``property_tail_suffix_blocks=1`` therefore crosses one complete
        residual block and selects the second-nearest dominating ADD.
        """

        return operator_hz_property_suffix_stop_layer_id(
            self.net,
            output_layer_id=int(output_layer_id),
            suffix_blocks=int(self.property_tail_suffix_blocks),
        )

    def _build_property_suffix_candidate(
        self,
        *,
        output_layer_id: int,
    ) -> Tuple[Optional[_AffineExpr], Dict[str, Any]]:
        """Build ``violation <= affine(prefix_ADD)`` by suffix dual replay."""

        receipt: Dict[str, Any] = {
            "schema": "operator_hz_property_suffix_replay_v1",
            "status": "disabled",
            "proof_authority": False,
            "safe_only": True,
            "requested_earlier_blocks": int(
                self.property_tail_suffix_blocks
            ),
            "requested_alpha_steps": int(
                self.property_tail_suffix_alpha_steps
            ),
            "requested_alpha_time_limit": float(
                self.property_tail_suffix_alpha_time_limit
            ),
            "requested_alpha_device": str(
                self.property_tail_suffix_alpha_device
            ),
            "baseline_fallback_retained_per_rival": True,
        }
        if self.property_tail_suffix_blocks <= 0:
            return None, receipt
        started = time.monotonic()
        try:
            from act.back_end.hybridz_tf.query_dual_replay import (
                replay_query_affine_lower_to_layer,
                validate_query_dual_affine_lower_plane,
            )

            if (
                self.property_upper_C is None
                or self.property_upper_thresholds is None
            ):
                raise OperatorHZBuildError(
                    "property suffix replay has no property rows"
                )
            full_input_mode = self.property_tail_suffix_blocks == 8
            candidates = self._property_suffix_dominating_add_candidates(
                output_layer_id=output_layer_id
            )
            if not candidates:
                raise OperatorHZBuildError(
                    "property suffix replay found no dominating ADD"
                )
            stop_lid: Optional[int] = None
            source_expr: Optional[_AffineExpr] = None
            if full_input_mode:
                receipt.update(
                    {
                        "stop_expression_source": (
                            "full_input_box_support"
                        ),
                        "dominating_add_candidates_nearest_first": [
                            int(value) for value in candidates
                        ],
                        "crosses_all_dominating_adds": True,
                    }
                )
            else:
                stop_lid, _selected_candidates = (
                    self._property_suffix_stop_layer(
                        output_layer_id=output_layer_id
                    )
                )
                if _selected_candidates != candidates:
                    raise OperatorHZBuildError(
                        "property suffix candidate ordering changed"
                    )
                source_snapshot = (
                    self.property_suffix_add_source_snapshot
                )
                if (
                    source_snapshot is None
                    or source_snapshot.add_layer_id != int(stop_lid)
                ):
                    raise OperatorHZBuildError(
                        "property suffix replay has no pre-materialization "
                        "correlated ADD source"
                    )
                source_expr = source_snapshot.expression
                if (
                    source_expr.size
                    != len(self._layer_by_id[int(stop_lid)].out_vars)
                    or source_expr.G.shape[1] > source_snapshot.n_cont
                ):
                    raise OperatorHZBuildError(
                        "property suffix correlated ADD source is malformed"
                    )
                receipt.update(
                    {
                        "stop_expression_source": (
                            "pre_materialization_correlated_add_sum"
                        ),
                        "stop_expression_center_sha256": (
                            _f64_array_sha256(source_expr.c)
                        ),
                        "stop_expression_generator_sha256": _csr_sha256(
                            source_expr.G
                        ),
                        "stop_expression_error_sha256": _f64_array_sha256(
                            source_expr.err
                        ),
                        "stop_expression_generator_nnz": int(
                            source_expr.G.nnz
                        ),
                        "stop_expression_error_max": float(
                            np.max(source_expr.err)
                            if source_expr.err.size
                            else 0.0
                        ),
                        "composition_prefix_n_cont": int(
                            source_snapshot.n_cont
                        ),
                        "composition_prefix_n_bin": int(
                            source_snapshot.n_bin
                        ),
                        "composition_prefix_eq_rows": int(
                            source_snapshot.eq_rows
                        ),
                        "composition_prefix_ub_rows": int(
                            source_snapshot.ub_rows
                        ),
                        "composition_prefix_eq_block_count": int(
                            source_snapshot.eq_block_count
                        ),
                        "composition_prefix_ub_block_count": int(
                            source_snapshot.ub_block_count
                        ),
                    }
                )
            cone: set[int] = set()
            stack = [int(output_layer_id)]
            while stack:
                lid = int(stack.pop())
                if lid in cone:
                    continue
                cone.add(lid)
                stack.extend(
                    int(value)
                    for value in self.net.preds.get(lid, ())
                )
            certified_bounds: Dict[int, Mapping[str, np.ndarray]] = {}
            for lid in sorted(cone):
                layer = self._layer_by_id[lid]
                kind = _kind(layer.kind)
                if kind == "INPUT":
                    continue
                if kind == "RELU":
                    pair = self.verified_preactivation_bounds.get(lid)
                    if pair is None:
                        raise OperatorHZBuildError(
                            "property suffix replay has no verified "
                            f"preactivation box for ReLU {lid}"
                        )
                    lower, upper = pair
                else:
                    expr = self.exprs.get(lid)
                    if expr is None:
                        raise OperatorHZBuildError(
                            "property suffix replay has no prefix expression "
                            f"for layer {lid}:{kind}"
                        )
                    lower, upper = self._cube_bounds(expr)
                certified_bounds[lid] = {
                    "lb": np.ascontiguousarray(
                        lower, dtype=np.float64
                    ),
                    "ub": np.ascontiguousarray(
                        upper, dtype=np.float64
                    ),
                }
            if full_input_mode:
                return self._build_full_input_property_candidate(
                    candidates=candidates,
                    certified_bounds=certified_bounds,
                    receipt=receipt,
                    started=started,
                )
            if stop_lid is None or source_expr is None:
                raise OperatorHZBuildError(
                    "property suffix ADD source is unavailable"
                )
            if (
                self.property_tail_suffix_blocks >= 1
                and self.property_tail_suffix_alpha_steps > 0
            ):
                return self._build_deep_property_suffix_candidate(
                    output_layer_id=int(output_layer_id),
                    stop_lid=int(stop_lid),
                    candidates=candidates,
                    source_expr=source_expr,
                    certified_bounds=certified_bounds,
                    receipt=receipt,
                    started=started,
                )
            plane_zero = replay_query_affine_lower_to_layer(
                self.net,
                certified_bounds,
                stop_lid=int(stop_lid),
                query_rows=-self.property_upper_C,
                query_bias=self.property_upper_thresholds,
                chunk_size=128,
                deadline=self.deadline,
            )
            relu_ids = tuple(
                lid
                for lid in sorted(cone)
                if _kind(self._layer_by_id[lid].kind) == "RELU"
            )
            plane_one = replay_query_affine_lower_to_layer(
                self.net,
                certified_bounds,
                stop_lid=int(stop_lid),
                query_rows=-self.property_upper_C,
                query_bias=self.property_upper_thresholds,
                alpha_by_relu={lid: 1.0 for lid in relu_ids},
                chunk_size=128,
                deadline=self.deadline,
            )
            if (
                not validate_query_dual_affine_lower_plane(plane_zero)
                or not validate_query_dual_affine_lower_plane(plane_one)
            ):
                raise OperatorHZBuildError(
                    "property suffix affine replay failed live validation"
                )
            if any(
                plane_zero.receipt["hashes"][key]
                != plane_one.receipt["hashes"][key]
                for key in (
                    "net_sha256",
                    "bounds_sha256",
                    "query_sha256",
                )
            ):
                raise OperatorHZBuildError(
                    "property suffix alpha extremes are not bound to the "
                    "same network, boxes, and property"
                )
            # Replay proves s + a*y_stop <= -violation.  Negating it gives
            # the required safe-only upper plane over the shared HZ prefix.
            candidate_zero = self._affine(
                source_expr,
                sp.csr_matrix(
                    -plane_zero.coefficients, dtype=np.float64
                ),
                -plane_zero.scalar,
                layer_id=int(output_layer_id),
            )
            candidate_one = self._affine(
                source_expr,
                sp.csr_matrix(
                    -plane_one.coefficients, dtype=np.float64
                ),
                -plane_one.scalar,
                layer_id=int(output_layer_id),
            )
            _zero_lb, zero_ub = self._cube_bounds(candidate_zero)
            _one_lb, one_ub = self._cube_bounds(candidate_one)
            plane_options = [plane_zero, plane_one]
            upper_options = [zero_ub, one_ub]
            optimizer_receipt: Dict[str, Any] = {
                "status": "disabled",
                "candidate_only": True,
                "proof_authority": False,
                "steps": int(self.property_tail_suffix_alpha_steps),
                "time_limit_seconds": float(
                    self.property_tail_suffix_alpha_time_limit
                ),
                "device": str(
                    self.property_tail_suffix_alpha_device
                ),
            }
            if self.property_tail_suffix_alpha_steps > 0:
                try:
                    from act.back_end.hybridz_tf.query_dual_candidates import (
                        generate_query_dual_candidates,
                        validate_query_dual_candidates,
                    )
                    from act.back_end.hybridz_tf.query_dual_pipeline import (
                        _flat_alpha_tree,
                    )

                    requested_device = (
                        "cuda"
                        if self.property_tail_suffix_alpha_device == "auto"
                        and torch.cuda.is_available()
                        else (
                            "cpu"
                            if self.property_tail_suffix_alpha_device == "auto"
                            else self.property_tail_suffix_alpha_device
                        )
                    )
                    alpha_device = torch.device(requested_device)
                    torch_bounds: Dict[int, Bounds] = {
                        lid: Bounds(
                            lb=torch.as_tensor(
                                value["lb"],
                                device=alpha_device,
                                dtype=torch.float64,
                            ).reshape(1, -1),
                            ub=torch.as_tensor(
                                value["ub"],
                                device=alpha_device,
                                dtype=torch.float64,
                            ).reshape(1, -1),
                        )
                        for lid, value in certified_bounds.items()
                    }
                    local_deadline = time.monotonic() + float(
                        self.property_tail_suffix_alpha_time_limit
                    )
                    if self.deadline is not None:
                        local_deadline = min(
                            local_deadline, float(self.deadline)
                        )
                    with torch.device(alpha_device):
                        optimized = generate_query_dual_candidates(
                            net=self.net,
                            bounds_dict=torch_bounds,
                            property_rows=self.property_upper_C,
                            property_upper_only=True,
                            steps=self.property_tail_suffix_alpha_steps,
                            block_size=max(
                                1, int(self.property_upper_C.shape[0])
                            ),
                            deadline=local_deadline,
                            descriptor_only=True,
                            selected_target_rows=(),
                        )
                    if (
                        not validate_query_dual_candidates(optimized)
                        or optimized.status != "descriptors_generated"
                        or len(optimized.query_descriptors) != 1
                        or len(optimized.alpha_trees) != 1
                    ):
                        raise OperatorHZBuildError(
                            "suffix alpha optimizer did not return one "
                            "complete property descriptor"
                        )
                    optimized_alpha = _flat_alpha_tree(
                        optimized.alpha_trees[0],
                        net=self.net,
                        start_lid=None,
                    )
                    plane_optimized = (
                        replay_query_affine_lower_to_layer(
                            self.net,
                            certified_bounds,
                            stop_lid=stop_lid,
                            query_rows=-self.property_upper_C,
                            query_bias=self.property_upper_thresholds,
                            alpha_by_relu=optimized_alpha,
                            chunk_size=128,
                            deadline=self.deadline,
                        )
                    )
                    if not validate_query_dual_affine_lower_plane(
                        plane_optimized
                    ):
                        raise OperatorHZBuildError(
                            "optimized suffix alpha replay failed validation"
                        )
                    if any(
                        plane_zero.receipt["hashes"][key]
                        != plane_optimized.receipt["hashes"][key]
                        for key in (
                            "net_sha256",
                            "bounds_sha256",
                            "query_sha256",
                        )
                    ):
                        raise OperatorHZBuildError(
                            "optimized suffix alpha replay changed the "
                            "network, boxes, or property"
                        )
                    candidate_optimized = self._affine(
                        source_expr,
                        sp.csr_matrix(
                            -plane_optimized.coefficients,
                            dtype=np.float64,
                        ),
                        -plane_optimized.scalar,
                        layer_id=int(output_layer_id),
                    )
                    _optimized_lb, optimized_ub = self._cube_bounds(
                        candidate_optimized
                    )
                    plane_options.append(plane_optimized)
                    upper_options.append(optimized_ub)
                    optimizer_receipt.update(
                        {
                            "status": "replayed",
                            "candidate_receipt_sha256": (
                                optimized.receipt["receipt_sha256"]
                            ),
                            "candidate_alpha_sha256": (
                                optimized.query_descriptors[
                                    0
                                ].alpha_sha256
                            ),
                            "replay_receipt_sha256": (
                                plane_optimized.receipt[
                                    "receipt_sha256"
                                ]
                            ),
                            "optimizer_seconds": float(
                                optimized.timings[0][
                                    "optimize_seconds"
                                ]
                            ),
                            "effective_device": requested_device,
                        }
                    )
                except Exception as exc:
                    optimizer_receipt.update(
                        {
                            "status": "error_fallback_extremes",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                    )
            stacked_upper = np.stack(upper_options, axis=0)
            selected_option = np.argmin(stacked_upper, axis=0)
            selected_coefficients = np.ascontiguousarray(
                np.stack(
                    [
                        plane_options[int(option)].coefficients[row]
                        for row, option in enumerate(selected_option)
                    ],
                    axis=0,
                ),
                dtype=np.float64,
            )
            selected_scalar = np.ascontiguousarray(
                np.asarray(
                    [
                        plane_options[int(option)].scalar[row]
                        for row, option in enumerate(selected_option)
                    ],
                    dtype=np.float64,
                )
            )
            candidate = self._affine(
                source_expr,
                sp.csr_matrix(
                    -selected_coefficients, dtype=np.float64
                ),
                -selected_scalar,
                layer_id=int(output_layer_id),
            )
            if candidate.size != self.property_upper_C.shape[0]:
                raise OperatorHZBuildError(
                    "property suffix replay returned the wrong rival count"
                )
            zero_receipt = plane_zero.receipt
            one_receipt = plane_one.receipt
            receipt.update(
                {
                    "status": "verified_affine_suffix",
                    "proof_authority": True,
                    "stop_layer_id": int(stop_lid),
                    "stop_layer_kind": "ADD",
                    "dominating_add_candidates_nearest_first": [
                        int(value) for value in candidates
                    ],
                    "nearest_add_skipped": True,
                    "query_count": int(candidate.size),
                    "stop_width": int(
                        selected_coefficients.shape[1]
                    ),
                    "coefficient_bytes": int(
                        selected_coefficients.nbytes
                    ),
                    "coefficient_sha256": _f64_array_sha256(
                        selected_coefficients
                    ),
                    "scalar_sha256": _f64_array_sha256(selected_scalar),
                    "alpha_extremes": [0.0, 1.0],
                    "alpha_one_selected_rows": int(
                        np.count_nonzero(selected_option == 1)
                    ),
                    "alpha_one_selected_mask_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            selected_option == 1, dtype=np.uint8
                        ).tobytes()
                    ).hexdigest(),
                    "optimized_alpha_selected_rows": int(
                        np.count_nonzero(selected_option == 2)
                    ),
                    "selected_option_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            selected_option, dtype=np.int8
                        ).tobytes()
                    ).hexdigest(),
                    "optimized_alpha": optimizer_receipt,
                    "alpha_zero_replay_receipt_sha256": zero_receipt[
                        "receipt_sha256"
                    ],
                    "alpha_one_replay_receipt_sha256": one_receipt[
                        "receipt_sha256"
                    ],
                    "replay_net_sha256": zero_receipt["hashes"][
                        "net_sha256"
                    ],
                    "replay_bounds_sha256": zero_receipt["hashes"][
                        "bounds_sha256"
                    ],
                    "replay_query_sha256": zero_receipt["hashes"][
                        "query_sha256"
                    ],
                    "bound_layer_count": int(len(certified_bounds)),
                    "composition_rule": (
                        "s+a@ADD<=(-C@output+threshold) implies "
                        "C@output-threshold<=-s-a@ADD"
                    ),
                    "elapsed_seconds": float(
                        time.monotonic() - started
                    ),
                }
            )
            return candidate, receipt
        except Exception as exc:
            receipt.update(
                {
                    "status": "error_fallback_baseline",
                    "proof_authority": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "elapsed_seconds": float(
                        time.monotonic() - started
                    ),
                }
            )
            return None, receipt

    def _build_full_input_property_candidate(
        self,
        *,
        candidates: Tuple[int, ...],
        certified_bounds: Mapping[
            int, Mapping[str, np.ndarray]
        ],
        receipt: Dict[str, Any],
        started: float,
    ) -> Tuple[Optional[_AffineExpr], Dict[str, Any]]:
        """Replay one optimized property batch through the complete network."""

        try:
            from act.back_end.hybridz_tf.query_dual_candidates import (
                generate_query_dual_candidates,
                validate_query_dual_candidates,
            )
            from act.back_end.hybridz_tf.query_dual_pipeline import (
                _flat_alpha_tree,
            )
            from act.back_end.hybridz_tf.query_dual_replay import (
                replay_query_lower_bounds,
                validate_query_dual_replay_result,
            )

            if (
                self.property_upper_C is None
                or self.property_upper_thresholds is None
                or self.property_tail_suffix_alpha_steps <= 0
                or self.property_tail_suffix_alpha_time_limit <= 0.0
            ):
                raise OperatorHZBuildError(
                    "full-input property replay requires property rows and "
                    "a positive alpha candidate budget"
                )
            requested_device = (
                "cuda"
                if self.property_tail_suffix_alpha_device == "auto"
                and torch.cuda.is_available()
                else (
                    "cpu"
                    if self.property_tail_suffix_alpha_device == "auto"
                    else self.property_tail_suffix_alpha_device
                )
            )
            alpha_device = torch.device(requested_device)
            torch_bounds: Dict[int, Bounds] = {
                lid: Bounds(
                    lb=torch.as_tensor(
                        value["lb"],
                        device=alpha_device,
                        dtype=torch.float64,
                    ).reshape(1, -1),
                    ub=torch.as_tensor(
                        value["ub"],
                        device=alpha_device,
                        dtype=torch.float64,
                    ).reshape(1, -1),
                )
                for lid, value in certified_bounds.items()
            }
            local_deadline = time.monotonic() + float(
                self.property_tail_suffix_alpha_time_limit
            )
            if self.deadline is not None:
                local_deadline = min(local_deadline, float(self.deadline))
            with torch.device(alpha_device):
                optimized = generate_query_dual_candidates(
                    net=self.net,
                    bounds_dict=torch_bounds,
                    property_rows=self.property_upper_C,
                    property_upper_only=True,
                    steps=self.property_tail_suffix_alpha_steps,
                    block_size=max(
                        1, int(self.property_upper_C.shape[0])
                    ),
                    deadline=local_deadline,
                    descriptor_only=True,
                    selected_target_rows=(),
                )
            if (
                not validate_query_dual_candidates(optimized)
                or optimized.status != "descriptors_generated"
                or len(optimized.query_descriptors) != 1
                or len(optimized.alpha_trees) != 1
            ):
                raise OperatorHZBuildError(
                    "full-input alpha optimizer did not return one "
                    "complete descriptor"
                )
            optimized_alpha = _flat_alpha_tree(
                optimized.alpha_trees[0],
                net=self.net,
                start_lid=None,
            )
            replay = replay_query_lower_bounds(
                self.net,
                certified_bounds,
                query_rows=-self.property_upper_C,
                query_bias=self.property_upper_thresholds,
                alpha_by_relu=optimized_alpha,
                chunk_size=128,
                deadline=self.deadline,
            )
            if not validate_query_dual_replay_result(replay):
                raise OperatorHZBuildError(
                    "full-input optimized replay failed live validation"
                )
            lower = np.ascontiguousarray(
                replay.lower_bounds, dtype=np.float64
            ).reshape(-1)
            if lower.size != self.property_upper_C.shape[0]:
                raise OperatorHZBuildError(
                    "full-input replay returned the wrong rival count"
                )
            upper = np.ascontiguousarray(-lower, dtype=np.float64)
            candidate = _AffineExpr(
                c=upper,
                G=sp.csr_matrix(
                    (upper.size, self.n_cont), dtype=np.float64
                ),
                err=np.zeros(upper.size, dtype=np.float64),
                affine_depth=0,
            )
            replay_receipt = replay.receipt
            optimizer_seconds = float(
                optimized.timings[0]["optimize_seconds"]
            )
            empty_coefficients = np.zeros(
                (upper.size, 0), dtype=np.float64
            )
            input_specs = [
                int(layer.id)
                for layer in self.net.layers
                if _kind(layer.kind) == "INPUT_SPEC"
            ]
            if len(input_specs) != 1:
                raise OperatorHZBuildError(
                    "full-input replay requires exactly one INPUT_SPEC"
                )
            receipt.update(
                {
                    "status": "verified_affine_suffix",
                    "proof_authority": True,
                    "replay_strategy": "optimized_only_full_input",
                    "output_form": "full_input_property_constant",
                    "uniform_endpoint_replays_omitted": True,
                    "baseline_substitutes_for_endpoint_fallbacks": True,
                    "stop_layer_id": int(input_specs[0]),
                    "stop_layer_kind": "INPUT_SPEC",
                    "dominating_add_candidates_nearest_first": [
                        int(value) for value in candidates
                    ],
                    "nearest_add_skipped": True,
                    "crosses_all_dominating_adds": True,
                    "query_count": int(candidate.size),
                    "stop_width": 0,
                    "coefficient_bytes": 0,
                    "coefficient_sha256": _f64_array_sha256(
                        empty_coefficients
                    ),
                    "scalar_sha256": _f64_array_sha256(lower),
                    "alpha_extremes": [],
                    "alpha_one_selected_rows": 0,
                    "alpha_one_selected_mask_sha256": hashlib.sha256(
                        np.zeros(
                            candidate.size, dtype=np.uint8
                        ).tobytes()
                    ).hexdigest(),
                    "optimized_alpha_selected_rows": int(candidate.size),
                    "selected_option_sha256": hashlib.sha256(
                        np.full(
                            candidate.size, 2, dtype=np.int8
                        ).tobytes()
                    ).hexdigest(),
                    "optimized_alpha": {
                        "status": "replayed",
                        "candidate_only": True,
                        "proof_authority": False,
                        "steps": int(
                            self.property_tail_suffix_alpha_steps
                        ),
                        "time_limit_seconds": float(
                            self.property_tail_suffix_alpha_time_limit
                        ),
                        "device": str(
                            self.property_tail_suffix_alpha_device
                        ),
                        "effective_device": requested_device,
                        "candidate_receipt_sha256": optimized.receipt[
                            "receipt_sha256"
                        ],
                        "candidate_alpha_sha256": (
                            optimized.query_descriptors[0].alpha_sha256
                        ),
                        "replay_receipt_sha256": replay_receipt[
                            "receipt_sha256"
                        ],
                        "optimizer_seconds": optimizer_seconds,
                    },
                    "alpha_zero_replay_receipt_sha256": None,
                    "alpha_one_replay_receipt_sha256": None,
                    "replay_net_sha256": replay_receipt["hashes"][
                        "net_sha256"
                    ],
                    "replay_bounds_sha256": replay_receipt["hashes"][
                        "bounds_sha256"
                    ],
                    "replay_query_sha256": replay_receipt["hashes"][
                        "query_sha256"
                    ],
                    "bound_layer_count": int(len(certified_bounds)),
                    "composition_rule": (
                        "LB(-C@output+threshold) over INPUT_SPEC implies "
                        "C@output-threshold<=-LB"
                    ),
                    "full_input_upper_min": float(np.min(upper)),
                    "full_input_upper_max": float(np.max(upper)),
                    "full_input_negative_rows": int(
                        np.count_nonzero(upper < 0.0)
                    ),
                    "elapsed_seconds": float(
                        time.monotonic() - started
                    ),
                }
            )
            self.property_full_input_replay_result = replay
            return candidate, receipt
        except Exception as exc:
            receipt.update(
                {
                    "status": "error_fallback_baseline",
                    "proof_authority": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "elapsed_seconds": float(
                        time.monotonic() - started
                    ),
                }
            )
            return None, receipt

    @staticmethod
    def _slice_property_query_alpha(
        optimized_alpha: Mapping[int, Any],
        *,
        rival_ids: Sequence[int],
        full_query_count: int,
    ) -> Dict[int, np.ndarray]:
        """Project a full-query frozen alpha tree onto selected rival rows."""

        rival_index = np.asarray(rival_ids, dtype=np.int64)
        selected: Dict[int, np.ndarray] = {}
        for alpha_layer_id, raw_alpha in optimized_alpha.items():
            alpha = np.asarray(raw_alpha, dtype=np.float64)
            if alpha.ndim == 2 and alpha.shape[0] == full_query_count:
                alpha = alpha[rival_index, :]
            elif (
                alpha.ndim == 3
                and alpha.shape[0] == 1
                and alpha.shape[1] == full_query_count
            ):
                alpha = alpha[:, rival_index, :]
            selected[int(alpha_layer_id)] = np.ascontiguousarray(
                alpha, dtype=np.float64
            )
        return selected

    def _build_exact_phase_conditional_suffix_rows(
        self,
        *,
        output_layer_id: int,
        stop_lid: int,
        source_expr: _AffineExpr,
        certified_bounds: Mapping[int, Mapping[str, np.ndarray]],
        optimized_alpha: Mapping[int, Any],
    ) -> Dict[str, Any]:
        """Replay branch-conditional suffix planes for retained exact ReLUs."""

        receipt: Dict[str, Any] = {
            "schema": "operator_hz_exact_phase_conditional_suffix_v2",
            "status": "no_eligible_exact_suffix_relu",
            "proof_authority": False,
            "records": [],
        }
        if not self.property_exact_phase_records:
            return receipt
        from act.back_end.hybridz_tf.query_dual_replay import (
            replay_query_affine_lower_to_layer,
            validate_query_dual_affine_lower_plane,
        )

        topology_position = {
            int(layer.id): index for index, layer in enumerate(self.net.layers)
        }
        stop_position = topology_position[int(stop_lid)]
        output_position = topology_position[int(output_layer_id)]
        eligible = [
            record
            for record in self.property_exact_phase_records
            if (
                stop_position
                < topology_position.get(int(record["layer_id"]), -1)
                < output_position
                and int(record["layer_id"]) in certified_bounds
            )
        ]
        if not eligible:
            return receipt
        eligible.sort(key=lambda record: int(record["exact_order"]))
        if len(eligible) > 2:
            raise OperatorHZBuildError(
                "conditional suffix replay supports exact depth at most two"
            )

        targets = []
        rival_ids = tuple(
            dict.fromkeys(
                int(rival)
                for record in eligible
                for rival in record["rival_ids"]
            )
        )
        if (
            not rival_ids
            or min(rival_ids) < 0
            or max(rival_ids) >= self.property_upper_C.shape[0]
        ):
            raise OperatorHZBuildError(
                "conditional suffix rival schedule is invalid"
            )
        for record in eligible:
            layer_id = int(record["layer_id"])
            row = int(record["row"])
            original = certified_bounds[layer_id]
            original_lower = np.asarray(
                original["lb"], dtype=np.float64
            ).reshape(-1)
            original_upper = np.asarray(
                original["ub"], dtype=np.float64
            ).reshape(-1)
            if (
                not 0 <= row < original_lower.size
                or not (
                    original_lower[row] < 0.0 < original_upper[row]
                )
            ):
                raise OperatorHZBuildError(
                    "conditional suffix target is not cube-unstable"
                )
            targets.append(
                {
                    "binary_col_id": int(record["binary_col_id"]),
                    "layer_id": layer_id,
                    "row": row,
                    "original_lower": float(original_lower[row]),
                    "original_upper": float(original_upper[row]),
                }
            )

        full_query_count = int(self.property_upper_C.shape[0])
        conditional_alpha = self._slice_property_query_alpha(
            optimized_alpha,
            rival_ids=rival_ids,
            full_query_count=full_query_count,
        )
        rival_index = np.asarray(rival_ids, dtype=np.int64)
        assignments = []
        for phases in product((-1, 1), repeat=len(targets)):
            conditioned = {
                int(lid): {
                    "lb": np.asarray(value["lb"], dtype=np.float64),
                    "ub": np.asarray(value["ub"], dtype=np.float64),
                }
                for lid, value in certified_bounds.items()
            }
            guards = []
            conditioned_targets = []
            for target, phase in zip(targets, phases):
                layer_id = int(target["layer_id"])
                row = int(target["row"])
                lower = np.asarray(
                    conditioned[layer_id]["lb"], dtype=np.float64
                ).copy()
                upper = np.asarray(
                    conditioned[layer_id]["ub"], dtype=np.float64
                ).copy()
                if phase < 0:
                    upper[row] = 0.0
                else:
                    lower[row] = 0.0
                conditioned[layer_id] = {
                    "lb": np.ascontiguousarray(lower),
                    "ub": np.ascontiguousarray(upper),
                }
                guards.append(
                    {
                        "binary_col_id": int(
                            target["binary_col_id"]
                        ),
                        "phase": int(phase),
                        "layer_id": layer_id,
                        "row": row,
                    }
                )
                conditioned_targets.append(
                    {
                        "layer_id": layer_id,
                        "row": row,
                        "phase": int(phase),
                        "conditioned_lower": float(lower[row]),
                        "conditioned_upper": float(upper[row]),
                    }
                )
            plane = replay_query_affine_lower_to_layer(
                self.net,
                conditioned,
                stop_lid=int(stop_lid),
                query_rows=-self.property_upper_C[rival_index],
                query_bias=self.property_upper_thresholds[rival_index],
                alpha_by_relu=conditional_alpha,
                chunk_size=128,
                deadline=self.deadline,
            )
            if not validate_query_dual_affine_lower_plane(plane):
                raise OperatorHZBuildError(
                    "conditional suffix replay failed validation"
                )
            expression = self._affine(
                source_expr,
                sp.csr_matrix(
                    -plane.coefficients, dtype=np.float64
                ),
                -plane.scalar,
                layer_id=int(output_layer_id),
            )
            if expression.size != len(rival_ids):
                raise OperatorHZBuildError(
                    "conditional suffix row count mismatch"
                )
            audit_receipt = {
                "schema": "operator_hz_exact_phase_conditional_suffix_v2",
                "proof_rule": (
                    "joint_exact_phase_implies_joint_conditioned_"
                    "preactivation_boxes;independent_affine_suffix_"
                    "replay;outward_operator_composition"
                ),
                "binary_guards": guards,
                "conditioned_targets": conditioned_targets,
                "rival_ids": list(rival_ids),
                "replay_receipt_sha256": plane.receipt[
                    "receipt_sha256"
                ],
                "coefficients_sha256": plane.receipt[
                    "coefficients_sha256"
                ],
                "scalar_sha256": plane.receipt["scalar_sha256"],
                "proof_authority": True,
            }
            guarded_plane = {
                "binary_guards": tuple(guards),
                "layer_id": int(targets[0]["layer_id"]),
                "row": int(targets[0]["row"]),
                "center": expression.c.copy(),
                "generator": expression.G.copy(),
                "error": expression.err.copy(),
                "rival_ids": rival_ids,
                "receipt": audit_receipt,
            }
            self.property_conditional_suffix_rows.append(guarded_plane)
            assignments.append(
                {
                    **audit_receipt,
                    "cube_upper_min": float(
                        np.min(self._cube_bounds(expression)[1])
                    ),
                    "cube_upper_max": float(
                        np.max(self._cube_bounds(expression)[1])
                    ),
                }
            )
        receipt.update(
            {
                "status": "applied",
                "proof_authority": True,
                "joint_depth": int(len(targets)),
                "record_count": int(len(targets)),
                "conditional_plane_count": int(
                    len(assignments) * len(rival_ids)
                ),
                "targets": targets,
                "rival_ids": list(rival_ids),
                "assignments": assignments,
            }
        )
        return receipt

    def _build_deep_property_suffix_candidate(
        self,
        *,
        output_layer_id: int,
        stop_lid: int,
        candidates: Tuple[int, ...],
        source_expr: _AffineExpr,
        certified_bounds: Mapping[
            int, Mapping[str, np.ndarray]
        ],
        receipt: Dict[str, Any],
        started: float,
    ) -> Tuple[Optional[_AffineExpr], Dict[str, Any]]:
        """Replay only the optimized alpha for a long residual suffix.

        The ordinary Fraction tail remains the per-rival fallback, so uniform
        alpha endpoint planes are redundant here.  Avoiding those two full
        CPU replays is a performance transformation, not a proof shortcut.
        """

        try:
            from act.back_end.hybridz_tf.query_dual_candidates import (
                generate_query_dual_candidates,
                validate_query_dual_candidates,
            )
            from act.back_end.hybridz_tf.query_dual_pipeline import (
                _flat_alpha_tree,
            )
            from act.back_end.hybridz_tf.query_dual_replay import (
                replay_query_affine_lower_to_layer,
                validate_query_dual_affine_lower_plane,
            )

            if (
                self.property_upper_C is None
                or self.property_upper_thresholds is None
            ):
                raise OperatorHZBuildError(
                    "deep suffix replay has no property rows"
                )
            requested_device = (
                "cuda"
                if self.property_tail_suffix_alpha_device == "auto"
                and torch.cuda.is_available()
                else (
                    "cpu"
                    if self.property_tail_suffix_alpha_device == "auto"
                    else self.property_tail_suffix_alpha_device
                )
            )
            alpha_device = torch.device(requested_device)
            torch_bounds: Dict[int, Bounds] = {
                lid: Bounds(
                    lb=torch.as_tensor(
                        value["lb"],
                        device=alpha_device,
                        dtype=torch.float64,
                    ).reshape(1, -1),
                    ub=torch.as_tensor(
                        value["ub"],
                        device=alpha_device,
                        dtype=torch.float64,
                    ).reshape(1, -1),
                )
                for lid, value in certified_bounds.items()
            }
            local_deadline = time.monotonic() + float(
                self.property_tail_suffix_alpha_time_limit
            )
            if self.deadline is not None:
                local_deadline = min(local_deadline, float(self.deadline))
            with torch.device(alpha_device):
                optimized = generate_query_dual_candidates(
                    net=self.net,
                    bounds_dict=torch_bounds,
                    property_rows=self.property_upper_C,
                    property_upper_only=True,
                    steps=self.property_tail_suffix_alpha_steps,
                    block_size=max(
                        1, int(self.property_upper_C.shape[0])
                    ),
                    deadline=local_deadline,
                    descriptor_only=True,
                    selected_target_rows=(),
                )
            if (
                not validate_query_dual_candidates(optimized)
                or optimized.status != "descriptors_generated"
                or len(optimized.query_descriptors) != 1
                or len(optimized.alpha_trees) != 1
            ):
                raise OperatorHZBuildError(
                    "deep suffix alpha optimizer did not return one "
                    "complete descriptor"
                )
            optimized_alpha = _flat_alpha_tree(
                optimized.alpha_trees[0],
                net=self.net,
                start_lid=None,
            )
            plane = replay_query_affine_lower_to_layer(
                self.net,
                certified_bounds,
                stop_lid=int(stop_lid),
                query_rows=-self.property_upper_C,
                query_bias=self.property_upper_thresholds,
                alpha_by_relu=optimized_alpha,
                chunk_size=128,
                deadline=self.deadline,
            )
            if not validate_query_dual_affine_lower_plane(plane):
                raise OperatorHZBuildError(
                    "deep optimized suffix replay failed live validation"
                )
            candidate = self._affine(
                source_expr,
                sp.csr_matrix(
                    -plane.coefficients, dtype=np.float64
                ),
                -plane.scalar,
                layer_id=int(output_layer_id),
            )
            if candidate.size != self.property_upper_C.shape[0]:
                raise OperatorHZBuildError(
                    "deep suffix replay returned the wrong rival count"
                )
            replay_receipt = plane.receipt
            optimizer_seconds = float(
                optimized.timings[0]["optimize_seconds"]
            )
            conditional_phase_receipt = (
                self._build_exact_phase_conditional_suffix_rows(
                    output_layer_id=int(output_layer_id),
                    stop_lid=int(stop_lid),
                    source_expr=source_expr,
                    certified_bounds=certified_bounds,
                    optimized_alpha=optimized_alpha,
                )
            )
            receipt.update(
                {
                    "status": "verified_affine_suffix",
                    "proof_authority": True,
                    "replay_strategy": "optimized_only_deep_suffix",
                    "uniform_endpoint_replays_omitted": True,
                    "baseline_substitutes_for_endpoint_fallbacks": True,
                    "stop_layer_id": int(stop_lid),
                    "stop_layer_kind": "ADD",
                    "dominating_add_candidates_nearest_first": [
                        int(value) for value in candidates
                    ],
                    "nearest_add_skipped": True,
                    "query_count": int(candidate.size),
                    "stop_width": int(plane.coefficients.shape[1]),
                    "coefficient_bytes": int(
                        plane.coefficients.nbytes
                    ),
                    "coefficient_sha256": replay_receipt[
                        "coefficients_sha256"
                    ],
                    "scalar_sha256": replay_receipt["scalar_sha256"],
                    "alpha_extremes": [],
                    "alpha_one_selected_rows": 0,
                    "alpha_one_selected_mask_sha256": hashlib.sha256(
                        np.zeros(candidate.size, dtype=np.uint8).tobytes()
                    ).hexdigest(),
                    "optimized_alpha_selected_rows": int(candidate.size),
                    "selected_option_sha256": hashlib.sha256(
                        np.full(
                            candidate.size, 2, dtype=np.int8
                        ).tobytes()
                    ).hexdigest(),
                    "optimized_alpha": {
                        "status": "replayed",
                        "candidate_only": True,
                        "proof_authority": False,
                        "steps": int(
                            self.property_tail_suffix_alpha_steps
                        ),
                        "time_limit_seconds": float(
                            self.property_tail_suffix_alpha_time_limit
                        ),
                        "device": str(
                            self.property_tail_suffix_alpha_device
                        ),
                        "effective_device": requested_device,
                        "candidate_receipt_sha256": optimized.receipt[
                            "receipt_sha256"
                        ],
                        "candidate_alpha_sha256": (
                            optimized.query_descriptors[0].alpha_sha256
                        ),
                        "replay_receipt_sha256": replay_receipt[
                            "receipt_sha256"
                        ],
                        "optimizer_seconds": optimizer_seconds,
                    },
                    "exact_phase_conditional_suffix": (
                        conditional_phase_receipt
                    ),
                    "alpha_zero_replay_receipt_sha256": None,
                    "alpha_one_replay_receipt_sha256": None,
                    "replay_net_sha256": replay_receipt["hashes"][
                        "net_sha256"
                    ],
                    "replay_bounds_sha256": replay_receipt["hashes"][
                        "bounds_sha256"
                    ],
                    "replay_query_sha256": replay_receipt["hashes"][
                        "query_sha256"
                    ],
                    "bound_layer_count": int(len(certified_bounds)),
                    "composition_rule": (
                        "s+a@ADD<=(-C@output+threshold) implies "
                        "C@output-threshold<=-s-a@ADD"
                    ),
                    "elapsed_seconds": float(
                        time.monotonic() - started
                    ),
                }
            )
            return candidate, receipt
        except Exception as exc:
            receipt.update(
                {
                    "status": "error_fallback_baseline",
                    "proof_authority": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "elapsed_seconds": float(
                        time.monotonic() - started
                    ),
                }
            )
            return None, receipt

    def _build_property_tail_upper(
        self,
        *,
        output_layer_id: int,
    ) -> _AffineExpr:
        """Export Fraction-audited property upper rows over the prefix frame."""

        snapshot = self.property_tail_snapshot
        if (
            self.property_upper_C is None
            or self.property_upper_thresholds is None
            or snapshot is None
            or self.property_tail_output_layer_id != int(output_layer_id)
        ):
            raise OperatorHZBuildError(
                "property upper tail snapshot/topology is unavailable"
            )
        output_layer = self._layer_by_id[int(output_layer_id)]
        try:
            matrix, bias = sparse_dense_matrix_from_layer(output_layer)
        except Exception as exc:
            raise OperatorHZBuildError(
                f"failed to recover final DENSE for property tail: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if matrix.shape[1] != snapshot.preactivation.size:
            raise OperatorHZBuildError(
                "property tail DENSE/preactivation width mismatch"
            )
        planes, intercepts, receipt = _property_relu_upper_planes(
            self.property_upper_C,
            self.property_upper_thresholds,
            matrix,
            bias,
            snapshot.lower,
            snapshot.upper,
        )
        baseline_expr = self._affine(
            snapshot.preactivation,
            sp.csr_matrix(planes, dtype=np.float64),
            intercepts,
            layer_id=int(output_layer_id),
        )
        baseline_lb, baseline_ub = self._cube_bounds(baseline_expr)
        upper_expr = baseline_expr
        envelope_planes = planes
        envelope_intercepts = intercepts
        alpha_candidate_planes: Optional[np.ndarray] = None
        alpha_candidate_intercepts: Optional[np.ndarray] = None
        alternative_rival_ids: List[int] = []
        alternative_plane_kinds: List[str] = []
        alpha_receipt: Dict[str, Any] = {
            "schema": "property_tail_negative_alpha_candidates_v1",
            "status": "disabled",
            "proof_authority": False,
            "candidate_only": True,
            "requested_steps": int(self.property_tail_alpha_steps),
            "time_limit_seconds": float(
                self.property_tail_alpha_time_limit
            ),
            "selected_rivals": 0,
        }
        alpha_enabled = bool(
            self.property_tail_alpha_steps > 0
            and self.property_tail_alpha_time_limit > 0.0
        )
        if alpha_enabled:
            try:
                from act.back_end.hybridz_tf.property_tail_candidates import (
                    optimize_property_tail_negative_alpha,
                )

                approximate_q = np.asarray(
                    (
                        sp.csr_matrix(self.property_upper_C)
                        @ sp.csr_matrix(matrix)
                    ).toarray(),
                    dtype=np.float64,
                )
                candidates = optimize_property_tail_negative_alpha(
                    preactivation_center=snapshot.preactivation.c,
                    preactivation_generators=snapshot.preactivation.G,
                    preactivation_error=snapshot.preactivation.err,
                    baseline_planes=planes,
                    baseline_intercepts=intercepts,
                    property_coefficients=approximate_q,
                    lower=snapshot.lower,
                    upper=snapshot.upper,
                    steps=self.property_tail_alpha_steps,
                    time_limit=self.property_tail_alpha_time_limit,
                    learning_rate=self.property_tail_alpha_learning_rate,
                    max_cells=self.property_tail_alpha_max_cells,
                    deadline=self.deadline,
                    device=self.property_tail_alpha_device,
                )
                alpha_receipt = dict(candidates.receipt)
                if np.any(candidates.alpha != 0.0):
                    (
                        candidate_planes,
                        candidate_intercepts,
                        candidate_exact_receipt,
                    ) = _property_relu_upper_planes(
                        self.property_upper_C,
                        self.property_upper_thresholds,
                        matrix,
                        bias,
                        snapshot.lower,
                        snapshot.upper,
                        negative_alpha=candidates.alpha,
                    )
                    alpha_candidate_planes = candidate_planes
                    alpha_candidate_intercepts = candidate_intercepts
                    candidate_expr = self._affine(
                        snapshot.preactivation,
                        sp.csr_matrix(
                            candidate_planes, dtype=np.float64
                        ),
                        candidate_intercepts,
                        layer_id=int(output_layer_id),
                    )
                    _candidate_lb, candidate_ub = self._cube_bounds(
                        candidate_expr
                    )
                    # Keep every exact-audited row which actually differs
                    # from alpha=0.  A candidate with a looser free cube can
                    # still have a much tighter support over the prefix HZ
                    # constraints; the grouped solver retains the baseline
                    # alongside it and therefore cannot lose that fallback.
                    retained = (
                        np.any(candidate_planes != planes, axis=1)
                        | (candidate_intercepts != intercepts)
                    )
                    retained_rows = np.flatnonzero(retained).astype(
                        np.int64, copy=False
                    )
                    alternative_rival_ids = [
                        int(value) for value in retained_rows
                    ]
                    alternative_plane_kinds = [
                        "negative_alpha_materialized"
                        for _ in retained_rows
                    ]
                    if retained_rows.size:
                        candidate_tail_G = candidate_expr.G[
                            retained_rows, :
                        ].tocsr()
                        envelope_G = sp.vstack(
                            [baseline_expr.G, candidate_tail_G],
                            format="csr",
                        )
                        envelope_G.eliminate_zeros()
                        upper_expr = _AffineExpr(
                            c=np.concatenate(
                                [
                                    baseline_expr.c,
                                    candidate_expr.c[retained_rows],
                                ]
                            ),
                            G=envelope_G,
                            err=np.concatenate(
                                [
                                    baseline_expr.err,
                                    candidate_expr.err[retained_rows],
                                ]
                            ),
                            affine_depth=max(
                                candidate_expr.affine_depth,
                                baseline_expr.affine_depth,
                            ),
                        )
                        envelope_planes = np.vstack(
                            [
                                planes,
                                candidate_planes[retained_rows, :],
                            ]
                        )
                        envelope_intercepts = np.concatenate(
                            [
                                intercepts,
                                candidate_intercepts[retained_rows],
                            ]
                        )
                        groups: List[Tuple[int, ...]] = [
                            (int(row),)
                            for row in range(baseline_expr.size)
                        ]
                        for offset, rival in enumerate(retained_rows):
                            groups[int(rival)] = (
                                int(rival),
                                int(baseline_expr.size + offset),
                            )
                        self.property_tail_row_groups = tuple(groups)
                    envelope_lb, envelope_ub = self._cube_bounds(upper_expr)
                    cube_delta = (
                        baseline_ub[retained_rows]
                        - candidate_ub[retained_rows]
                    )
                    retained_mask_digest = hashlib.sha256(
                        np.ascontiguousarray(retained).tobytes()
                    ).hexdigest()
                    alpha_receipt.update(
                        {
                            "exact_candidate_audit": (
                                candidate_exact_receipt
                            ),
                            "selection_authority": (
                                "retain_all_distinct_fraction_audited_"
                                "candidate_planes_v1"
                            ),
                            "selection_proof_authority": False,
                            "baseline_fallback_retained_per_rival": True,
                            "selected_rivals": int(
                                retained_rows.size
                            ),
                            "selected_rival_preview": [
                                int(value)
                                for value in retained_rows[:16]
                            ],
                            "retained_mask_sha256": retained_mask_digest,
                            "cube_improved_rivals": int(
                                np.count_nonzero(cube_delta > 0.0)
                            ),
                            "cube_upper_improvement_sum": float(
                                np.sum(np.maximum(cube_delta, 0.0))
                            ),
                            "cube_upper_improvement_max": float(
                                np.max(np.maximum(cube_delta, 0.0))
                                if cube_delta.size else 0.0
                            ),
                            "candidate_minus_baseline_cube_min": float(
                                np.min(-cube_delta)
                                if cube_delta.size else 0.0
                            ),
                            "candidate_minus_baseline_cube_max": float(
                                np.max(-cube_delta)
                                if cube_delta.size else 0.0
                            ),
                            "envelope_cube_lb_min": float(
                                np.min(envelope_lb)
                            ),
                            "envelope_cube_ub_max": float(
                                np.max(envelope_ub)
                            ),
                            "baseline_cube_lb_min": float(
                                np.min(baseline_lb)
                            ),
                            "baseline_cube_ub_max": float(
                                np.max(baseline_ub)
                            ),
                        }
                    )
                else:
                    alpha_receipt.update(
                        {
                            "selected_rivals": 0,
                            "selection_authority": (
                                "retain_all_distinct_fraction_audited_"
                                "candidate_planes_v1"
                            ),
                            "selection_proof_authority": False,
                            "baseline_fallback_retained_per_rival": True,
                        }
                    )
            except Exception as exc:
                # Candidate generation and row selection never have proof
                # authority.  Any failure returns to the exact alpha=0 plane.
                upper_expr = baseline_expr
                envelope_planes = planes
                envelope_intercepts = intercepts
                alternative_rival_ids = []
                alternative_plane_kinds = []
                self.property_tail_row_groups = ()
                alpha_candidate_planes = None
                alpha_candidate_intercepts = None
                alpha_receipt = {
                    **alpha_receipt,
                    "status": "error_fallback_baseline",
                    "proof_authority": False,
                    "candidate_only": True,
                    "selected_rivals": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            self._check_deadline("property_tail_alpha_candidates")

        pairhull_enabled = bool(
            self.property_tail_pairhull_budget > 0
            and self.property_tail_pairhull_time_limit > 0.0
        )
        empty_pair_rows = np.empty(
            (0, snapshot.preactivation.size), dtype=np.float64
        )
        empty_pair_intercepts = np.empty(0, dtype=np.float64)
        empty_pair_rows_sha256 = hashlib.sha256(
            np.ascontiguousarray(empty_pair_rows).tobytes()
        ).hexdigest()
        empty_pair_intercepts_sha256 = hashlib.sha256(
            np.ascontiguousarray(empty_pair_intercepts).tobytes()
        ).hexdigest()
        pairhull_receipt: Dict[str, Any] = {
            "schema": "operator_hz_property_tail_pairhull_v1",
            "enabled": pairhull_enabled,
            "status": "pending" if pairhull_enabled else "disabled",
            "safe_only": True,
            "proof_authority": False,
            "selection_candidate_only": True,
            "selection_proof_authority": False,
            "exact_search_complete": bool(not pairhull_enabled),
            "error_included": True,
            "compact_sparse_projection": True,
            "baseline_fallback_retained_per_rival": True,
            "foundation_slopes_reused": True,
            "foundation_intercept_outward_slack_inherited": True,
            "full_row_outward_affine": False,
            "prunes_prefix_frame": False,
            "pair_budget": int(self.property_tail_pairhull_budget),
            "time_limit_seconds": float(
                self.property_tail_pairhull_time_limit
            ),
            "budget_semantics": "global_unique_disjoint_pairs_v1",
            "max_rows_per_rival": 1,
            "global_pair_count": 0,
            "selected_rivals": 0,
            "selected_rival_ids": [],
            "candidate_rows_sha256": empty_pair_rows_sha256,
            "candidate_intercepts_sha256": (
                empty_pair_intercepts_sha256
            ),
        }
        if pairhull_enabled:
            pairhull_started = time.monotonic()
            pairhull_stop_at = (
                pairhull_started
                + self.property_tail_pairhull_time_limit
            )
            if self.deadline is not None:
                pairhull_stop_at = min(
                    pairhull_stop_at, float(self.deadline)
                )
            pairhull_state = (
                upper_expr,
                envelope_planes,
                envelope_intercepts,
                tuple(self.property_tail_row_groups),
                list(alternative_rival_ids),
                list(alternative_plane_kinds),
            )
            try:
                from act.back_end.hybridz_tf.property_pairhull_candidates import (
                    build_property_pairhull_candidates,
                    finalize_property_pairhull_candidates_receipt,
                    verify_property_pairhull_candidates_receipt,
                )

                foundation_plane_list = [planes]
                foundation_intercept_list = [intercepts]
                foundation_names = ["baseline"]
                if (
                    alpha_candidate_planes is not None
                    and alpha_candidate_intercepts is not None
                ):
                    foundation_plane_list.append(alpha_candidate_planes)
                    foundation_intercept_list.append(
                        alpha_candidate_intercepts
                    )
                    foundation_names.append("negative_alpha")
                pair_candidates = build_property_pairhull_candidates(
                    property_matrix=self.property_upper_C,
                    output_weight=matrix,
                    preactivation_center=snapshot.preactivation.c,
                    preactivation_generators=snapshot.preactivation.G,
                    preactivation_error=snapshot.preactivation.err,
                    lower=snapshot.lower,
                    upper=snapshot.upper,
                    foundation_planes=np.stack(
                        foundation_plane_list, axis=0
                    ),
                    foundation_intercepts=np.stack(
                        foundation_intercept_list, axis=0
                    ),
                    foundation_names=foundation_names,
                    pair_budget=self.property_tail_pairhull_budget,
                    time_limit=self.property_tail_pairhull_time_limit,
                    deadline=pairhull_stop_at,
                )
                inner_receipt = dict(pair_candidates.receipt)
                candidate_count = int(pair_candidates.rival_ids.size)
                if (
                    not verify_property_pairhull_candidates_receipt(
                        inner_receipt
                    )
                    or pair_candidates.rival_ids.ndim != 1
                    or pair_candidates.foundation_indices.shape
                    != (candidate_count,)
                    or pair_candidates.pair_indices.shape
                    != (candidate_count, 2)
                    or pair_candidates.planes.shape
                    != (candidate_count, snapshot.preactivation.size)
                    or pair_candidates.intercepts.shape
                    != (candidate_count,)
                    or np.unique(pair_candidates.rival_ids).size
                    != candidate_count
                    or np.any(pair_candidates.rival_ids < 0)
                    or np.any(
                        pair_candidates.rival_ids
                        >= baseline_expr.size
                    )
                    or np.any(pair_candidates.foundation_indices < 0)
                    or np.any(
                        pair_candidates.foundation_indices
                        >= len(foundation_plane_list)
                    )
                    or not np.all(np.isfinite(pair_candidates.planes))
                    or not np.all(
                        np.isfinite(pair_candidates.intercepts)
                    )
                    or inner_receipt.get("selected_candidates")
                    != candidate_count
                ):
                    raise OperatorHZBuildError(
                        "PairHull candidate batch failed operator-side "
                        "shape/checksum validation"
                    )
                if candidate_count:
                    if (
                        np.any(np.diff(pair_candidates.rival_ids) <= 0)
                        or np.any(pair_candidates.pair_indices < 0)
                        or np.any(
                            pair_candidates.pair_indices
                            >= snapshot.preactivation.size
                        )
                        or np.any(
                            pair_candidates.pair_indices[:, 0]
                            == pair_candidates.pair_indices[:, 1]
                        )
                    ):
                        raise OperatorHZBuildError(
                            "PairHull candidate ids/pairs are not canonical"
                        )
                    expected_foundation_planes = np.stack(
                        [
                            foundation_plane_list[int(foundation)][
                                int(rival), :
                            ]
                            for rival, foundation in zip(
                                pair_candidates.rival_ids,
                                pair_candidates.foundation_indices,
                            )
                        ],
                        axis=0,
                    )
                    expected_foundation_intercepts = np.asarray(
                        [
                            foundation_intercept_list[int(foundation)][
                                int(rival)
                            ]
                            for rival, foundation in zip(
                                pair_candidates.rival_ids,
                                pair_candidates.foundation_indices,
                            )
                        ],
                        dtype=np.float64,
                    )
                    if (
                        not np.array_equal(
                            pair_candidates.planes,
                            expected_foundation_planes,
                        )
                        or not np.all(
                            pair_candidates.intercepts
                            < expected_foundation_intercepts
                        )
                    ):
                        raise OperatorHZBuildError(
                            "PairHull must reuse a foundation slope and "
                            "strictly lower its stored intercept"
                        )
                pairhull_receipt.update(
                    {
                        "candidate_receipt": inner_receipt,
                        "exact_search_complete": bool(
                            inner_receipt.get("whole_batch_complete", False)
                        ),
                        "global_pair_count": int(
                            inner_receipt.get("global_pair_count", 0)
                        ),
                    }
                )
                if pair_candidates.rival_ids.size:
                    pair_expr = self._affine(
                        snapshot.preactivation,
                        sp.csr_matrix(
                            pair_candidates.planes, dtype=np.float64
                        ),
                        pair_candidates.intercepts,
                        layer_id=int(output_layer_id),
                    )
                    _pair_lb, pair_ub = self._cube_bounds(pair_expr)
                    current_groups: List[Tuple[int, ...]] = (
                        list(self.property_tail_row_groups)
                        if self.property_tail_row_groups
                        else [
                            (int(row),)
                            for row in range(baseline_expr.size)
                        ]
                    )
                    if len(current_groups) != baseline_expr.size:
                        raise OperatorHZBuildError(
                            "PairHull property groups have the wrong base "
                            "count"
                        )
                    _current_lb, current_ub = self._cube_bounds(upper_expr)
                    prior_group_ub = np.asarray(
                        [
                            min(
                                float(current_ub[int(row)])
                                for row in current_groups[
                                    int(rival)
                                ]
                            )
                            for rival in pair_candidates.rival_ids
                        ],
                        dtype=np.float64,
                    )
                    guarded_improvement = prior_group_ub - pair_ub
                    retained = guarded_improvement > 0.0
                    retained_indices = np.flatnonzero(retained).astype(
                        np.int64, copy=False
                    )
                    if retained_indices.size:
                        retained_rivals = pair_candidates.rival_ids[
                            retained_indices
                        ].astype(np.int64, copy=False)
                        if np.unique(retained_rivals).size != retained_rivals.size:
                            raise OperatorHZBuildError(
                                "PairHull emitted more than one row per rival"
                            )
                        pair_start = int(upper_expr.size)
                        pair_G = pair_expr.G[
                            retained_indices, :
                        ].tocsr()
                        combined_G = sp.vstack(
                            [upper_expr.G, pair_G], format="csr"
                        )
                        combined_G.eliminate_zeros()
                        upper_expr = _AffineExpr(
                            c=np.concatenate(
                                [
                                    upper_expr.c,
                                    pair_expr.c[retained_indices],
                                ]
                            ),
                            G=combined_G,
                            err=np.concatenate(
                                [
                                    upper_expr.err,
                                    pair_expr.err[retained_indices],
                                ]
                            ),
                            affine_depth=max(
                                upper_expr.affine_depth,
                                pair_expr.affine_depth,
                            ),
                        )
                        retained_planes = pair_candidates.planes[
                            retained_indices, :
                        ]
                        retained_intercepts = pair_candidates.intercepts[
                            retained_indices
                        ]
                        envelope_planes = np.vstack(
                            [envelope_planes, retained_planes]
                        )
                        envelope_intercepts = np.concatenate(
                            [
                                envelope_intercepts,
                                retained_intercepts,
                            ]
                        )
                        for offset, rival in enumerate(retained_rivals):
                            rival_id = int(rival)
                            current_groups[rival_id] = (
                                *current_groups[rival_id],
                                int(pair_start + offset),
                            )
                        self.property_tail_row_groups = tuple(
                            current_groups
                        )
                        alternative_rival_ids.extend(
                            int(value) for value in retained_rivals
                        )
                        alternative_plane_kinds.extend(
                            "pairhull_joint_materialized"
                            for _ in retained_rivals
                        )
                        retained_improvement = guarded_improvement[
                            retained_indices
                        ]
                        pairhull_receipt.update(
                            {
                                "status": "applied",
                                "proof_authority": True,
                                "full_row_outward_affine": True,
                                "selected_rivals": int(
                                    retained_rivals.size
                                ),
                                "selected_rival_ids": [
                                    int(value)
                                    for value in retained_rivals
                                ],
                                "selected_foundation_indices": [
                                    int(value)
                                    for value in pair_candidates
                                    .foundation_indices[retained_indices]
                                ],
                                "selected_pair_indices": [
                                    [int(value) for value in pair]
                                    for pair in pair_candidates.pair_indices[
                                        retained_indices
                                    ]
                                ],
                                "candidate_rows_sha256": hashlib.sha256(
                                    np.ascontiguousarray(
                                        retained_planes
                                    ).tobytes()
                                ).hexdigest(),
                                "candidate_intercepts_sha256": (
                                    hashlib.sha256(
                                        np.ascontiguousarray(
                                            retained_intercepts
                                        ).tobytes()
                                    ).hexdigest()
                                ),
                                "guarded_cube_improved_rivals": int(
                                    retained_rivals.size
                                ),
                                "guarded_cube_improvement_sum": float(
                                    np.sum(retained_improvement)
                                ),
                                "guarded_cube_improvement_max": float(
                                    np.max(retained_improvement)
                                ),
                                "operator_discarded_nonimproving_rows": int(
                                    retained.size
                                    - retained_indices.size
                                ),
                            }
                        )
                    else:
                        pairhull_receipt.update(
                            {
                                "status": (
                                    "no_strict_guarded_cube_improvement"
                                ),
                                "proof_authority": False,
                                "selected_rivals": 0,
                                "selected_rival_ids": [],
                                "operator_discarded_nonimproving_rows": int(
                                    retained.size
                                ),
                            }
                        )
                else:
                    pairhull_receipt.update(
                        {
                            "status": str(
                                inner_receipt.get(
                                    "status", "no_candidate_rows"
                                )
                            ),
                            "proof_authority": False,
                        }
                    )
                if time.monotonic() >= pairhull_stop_at:
                    raise OperatorHZBuildError(
                        "property-tail PairHull total time limit expired "
                        "after candidate reconstruction"
                    )
                pairhull_receipt["operator_elapsed_seconds"] = float(
                    time.monotonic() - pairhull_started
                )
                pairhull_receipt = (
                    finalize_property_pairhull_candidates_receipt(
                        pairhull_receipt
                    )
                )
            except Exception as exc:
                (
                    upper_expr,
                    envelope_planes,
                    envelope_intercepts,
                    prior_groups,
                    alternative_rival_ids,
                    alternative_plane_kinds,
                ) = pairhull_state
                self.property_tail_row_groups = prior_groups
                pairhull_receipt.update(
                    {
                        "status": "error_fallback_foundations",
                        "proof_authority": False,
                        "exact_search_complete": False,
                        "selected_rivals": 0,
                        "selected_rival_ids": [],
                        "selected_foundation_indices": [],
                        "selected_pair_indices": [],
                        "full_row_outward_affine": False,
                        "candidate_rows_sha256": empty_pair_rows_sha256,
                        "candidate_intercepts_sha256": (
                            empty_pair_intercepts_sha256
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "operator_elapsed_seconds": float(
                            time.monotonic() - pairhull_started
                        ),
                    }
                )
                # The exception may occur after a candidate row was appended
                # and its applied-only audit fields were populated.  The HZ,
                # groups, and alternative-row maps above have already been
                # restored, so no guarded-applied statistic may survive in
                # the fail-closed receipt either.
                for applied_only_field in (
                    "guarded_cube_improved_rivals",
                    "guarded_cube_improvement_sum",
                    "guarded_cube_improvement_max",
                    "operator_discarded_nonimproving_rows",
                ):
                    pairhull_receipt.pop(applied_only_field, None)
                try:
                    from act.back_end.hybridz_tf.property_pairhull_candidates import (
                        finalize_property_pairhull_candidates_receipt,
                    )

                    pairhull_receipt = (
                        finalize_property_pairhull_candidates_receipt(
                            pairhull_receipt
                        )
                    )
                except Exception:
                    # The final verifier rejects a malformed receipt.  The
                    # underlying baseline/alpha rows remain intact here.
                    pass
            self._check_deadline("property_tail_pairhull_candidates")
        if "receipt_sha256" not in pairhull_receipt:
            from act.back_end.hybridz_tf.property_pairhull_candidates import (
                finalize_property_pairhull_candidates_receipt,
            )

            pairhull_receipt = (
                finalize_property_pairhull_candidates_receipt(
                    pairhull_receipt
                )
            )

        add_source_receipt: Dict[str, Any] = {
            "schema": "property_tail_add_source_planes_v1",
            "enabled": bool(self.property_tail_add_source_planes),
            "status": (
                "pending"
                if self.property_tail_add_source_planes
                else "disabled"
            ),
            "safe_only": True,
            "proof_authority": False,
            "materialized_baseline_retained_per_rival": True,
            "prunes_materialized_frame": False,
        }
        if self.property_tail_add_source_planes:
            try:
                add_snapshot = self.property_tail_add_source_snapshot
                if (
                    add_snapshot is None
                    or self.property_tail_add_source_layer_id is None
                    or add_snapshot.add_layer_id
                    != self.property_tail_add_source_layer_id
                ):
                    raise OperatorHZBuildError(
                        "final materialized ADD source snapshot is unavailable"
                    )
                retained_relation_blocks = tuple(
                    self.ub_blocks[
                        add_snapshot.ub_block_count_before:
                        add_snapshot.ub_block_count_after
                    ]
                )
                retained_relation_rows = tuple(
                    int(block.Ac.shape[0])
                    for block in retained_relation_blocks
                )
                retained_relation_tags = tuple(
                    str(block.tag) for block in retained_relation_blocks
                )
                retained_relation_digest = _constraint_blocks_sha256(
                    retained_relation_blocks
                )
                if (
                    add_snapshot.n_cont_after != snapshot.n_cont
                    or add_snapshot.n_bin != snapshot.n_bin
                    or add_snapshot.eq_block_count_after
                    > snapshot.eq_block_count
                    or add_snapshot.ub_block_count_after
                    > snapshot.ub_block_count
                    or add_snapshot.expression.G.shape[1]
                    > add_snapshot.n_cont_before
                    or add_snapshot.n_cont_after
                    != add_snapshot.n_cont_before + add_snapshot.new_cont
                    or add_snapshot.eq_block_count_after
                    != add_snapshot.eq_block_count_before
                    or add_snapshot.ub_block_count_after
                    != (
                        add_snapshot.ub_block_count_before
                        + len(add_snapshot.relation_block_rows)
                    )
                    or add_snapshot.new_ub
                    != sum(add_snapshot.relation_block_rows)
                    or len(add_snapshot.relation_block_rows)
                    != len(add_snapshot.relation_block_tags)
                    or any(
                        not tag.startswith(
                            f"add_materialize:{add_snapshot.add_layer_id}:"
                        )
                        for tag in add_snapshot.relation_block_tags
                    )
                    or retained_relation_rows
                    != add_snapshot.relation_block_rows
                    or retained_relation_tags
                    != add_snapshot.relation_block_tags
                    or retained_relation_digest
                    != add_snapshot.relation_blocks_sha256
                ):
                    raise OperatorHZBuildError(
                        "final materialized ADD source/frame invariant failed"
                    )
                source_bridge = add_snapshot.expression
                bridge_kinds: List[str] = []
                bridge_parameter_receipts: List[Dict[str, Any]] = []
                for bridge_layer_id in (
                    self.property_tail_add_source_bridge_layer_ids
                ):
                    bridge_layer = self._layer_by_id[int(bridge_layer_id)]
                    bridge_kind = _kind(bridge_layer.kind)
                    bridge_kinds.append(bridge_kind)
                    if bridge_kind == "FLATTEN":
                        if len(bridge_layer.out_vars) != source_bridge.size:
                            raise OperatorHZBuildError(
                                "ADD source FLATTEN bridge changed width"
                            )
                        bridge_parameter_receipts.append(
                            {
                                "layer_id": int(bridge_layer.id),
                                "kind": bridge_kind,
                                "input_size": int(source_bridge.size),
                                "output_size": int(
                                    len(bridge_layer.out_vars)
                                ),
                            }
                        )
                    elif bridge_kind == "DENSE":
                        bridge_matrix, bridge_bias = (
                            sparse_dense_matrix_from_layer(bridge_layer)
                        )
                        bridge_parameter_receipts.append(
                            {
                                "layer_id": int(bridge_layer.id),
                                "kind": bridge_kind,
                                "matrix_shape": [
                                    int(value)
                                    for value in bridge_matrix.shape
                                ],
                                "matrix_sha256": _csr_sha256(
                                    bridge_matrix
                                ),
                                "bias_sha256": hashlib.sha256(
                                    np.ascontiguousarray(
                                        bridge_bias,
                                        dtype=np.float64,
                                    ).tobytes()
                                ).hexdigest(),
                            }
                        )
                        source_bridge = self._affine(
                            source_bridge,
                            bridge_matrix,
                            bridge_bias,
                            layer_id=int(bridge_layer.id),
                        )
                    else:
                        raise OperatorHZBuildError(
                            "ADD source bridge contains a non-FLATTEN/DENSE "
                            f"operator {bridge_layer_id}:{bridge_kind}"
                        )
                if source_bridge.size != snapshot.preactivation.size:
                    raise OperatorHZBuildError(
                        "ADD source bridge/final preactivation width mismatch"
                    )
                source_preactivation = _AffineExpr(
                    c=source_bridge.c,
                    G=_pad_cols(
                        source_bridge.G, snapshot.n_cont
                    ),
                    err=source_bridge.err,
                    affine_depth=source_bridge.affine_depth,
                )
                source_baseline_expr = self._affine(
                    source_preactivation,
                    sp.csr_matrix(planes, dtype=np.float64),
                    intercepts,
                    layer_id=int(output_layer_id),
                )
                if source_baseline_expr.G.shape[1] != snapshot.n_cont:
                    raise OperatorHZBuildError(
                        "ADD source property rows do not use the prefix frame"
                    )
                _source_lb, source_ub = self._cube_bounds(
                    source_baseline_expr
                )
                _before_source_lb, before_source_ub = self._cube_bounds(
                    upper_expr
                )
                source_start = int(upper_expr.size)
                source_G = sp.vstack(
                    [upper_expr.G, source_baseline_expr.G],
                    format="csr",
                )
                source_G.eliminate_zeros()
                candidate_upper_expr = _AffineExpr(
                    c=np.concatenate(
                        [upper_expr.c, source_baseline_expr.c]
                    ),
                    G=source_G,
                    err=np.concatenate(
                        [upper_expr.err, source_baseline_expr.err]
                    ),
                    affine_depth=max(
                        upper_expr.affine_depth,
                        source_baseline_expr.affine_depth,
                    ),
                )
                candidate_groups: List[Tuple[int, ...]] = (
                    list(self.property_tail_row_groups)
                    if self.property_tail_row_groups
                    else [
                        (int(row),)
                        for row in range(baseline_expr.size)
                    ]
                )
                if len(candidate_groups) != baseline_expr.size:
                    raise OperatorHZBuildError(
                        "ADD source property groups have the wrong base count"
                    )
                for rival in range(baseline_expr.size):
                    candidate_groups[rival] = (
                        *candidate_groups[rival],
                        int(source_start + rival),
                    )
                _after_source_lb, after_source_ub = self._cube_bounds(
                    candidate_upper_expr
                )
                # The last element of each expanded group is its new source
                # row; exclude it when measuring the prior envelope.
                group_upper_prior = np.asarray(
                    [
                        min(
                            float(before_source_ub[int(row)])
                            for row in group[:-1]
                        )
                        for group in candidate_groups
                    ],
                    dtype=np.float64,
                )
                group_upper_after = np.asarray(
                    [
                        min(
                            float(after_source_ub[int(row)])
                            for row in group
                        )
                        for group in candidate_groups
                    ],
                    dtype=np.float64,
                )
                improvement = group_upper_prior - group_upper_after
                source_column_counts = np.bincount(
                    add_snapshot.expression.G.indices,
                    minlength=add_snapshot.n_cont_before,
                )

                # Commit only after every source row and grouping invariant
                # has passed.  The materialized baseline rows and equality
                # bands remain untouched.
                upper_expr = candidate_upper_expr
                self.property_tail_row_groups = tuple(candidate_groups)
                envelope_planes = np.vstack(
                    [envelope_planes, planes]
                )
                envelope_intercepts = np.concatenate(
                    [envelope_intercepts, intercepts]
                )
                alternative_rival_ids.extend(
                    int(rival)
                    for rival in range(baseline_expr.size)
                )
                alternative_plane_kinds.extend(
                    "add_source_alpha0"
                    for _ in range(baseline_expr.size)
                )
                add_source_receipt.update(
                    {
                        "status": "applied",
                        "proof_authority": True,
                        "proof_rule": (
                            "inductive_add_source_affine_enclosure+"
                            "fraction_audited_property_plane+"
                            "outward_affine_roundoff;"
                            "materialized_relation_retained"
                        ),
                        "add_layer_id": int(add_snapshot.add_layer_id),
                        "source_row_count": int(
                            source_baseline_expr.size
                        ),
                        "source_expression_nnz": int(
                            add_snapshot.expression.G.nnz
                        ),
                        "source_expression_size": int(
                            add_snapshot.expression.size
                        ),
                        "source_active_columns": int(
                            np.count_nonzero(source_column_counts)
                        ),
                        "source_shared_columns": int(
                            np.count_nonzero(source_column_counts > 1)
                        ),
                        "source_max_column_row_uses": int(
                            np.max(source_column_counts)
                            if source_column_counts.size else 0
                        ),
                        "source_cross_row_reuse_nnz": int(
                            np.sum(
                                np.maximum(source_column_counts - 1, 0)
                            )
                        ),
                        "source_expression_sha256": _csr_sha256(
                            add_snapshot.expression.G
                        ),
                        "source_expression_center_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                add_snapshot.expression.c
                            ).tobytes()
                        ).hexdigest(),
                        "source_expression_error_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                add_snapshot.expression.err
                            ).tobytes()
                        ).hexdigest(),
                        "bridge_layer_ids": [
                            int(value)
                            for value in (
                                self.property_tail_add_source_bridge_layer_ids
                            )
                        ],
                        "bridge_layer_kinds": bridge_kinds,
                        "bridge_parameter_receipts": (
                            bridge_parameter_receipts
                        ),
                        "bridge_topology": (
                            "ADD->final_RELU"
                            if not bridge_kinds
                            else "ADD->FLATTEN->DENSE->final_RELU"
                        ),
                        "source_preactivation_nnz": int(
                            source_bridge.G.nnz
                        ),
                        "source_preactivation_size": int(
                            source_bridge.size
                        ),
                        "source_preactivation_center_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                source_bridge.c
                            ).tobytes()
                        ).hexdigest(),
                        "source_preactivation_generator_sha256": (
                            _csr_sha256(source_bridge.G)
                        ),
                        "source_preactivation_error_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                source_bridge.err
                            ).tobytes()
                        ).hexdigest(),
                        "source_n_cont_before": int(
                            add_snapshot.n_cont_before
                        ),
                        "materialized_n_cont_after": int(
                            add_snapshot.n_cont_after
                        ),
                        "materialized_n_bin": int(add_snapshot.n_bin),
                        "materialized_new_cont": int(
                            add_snapshot.new_cont
                        ),
                        "materialized_new_ub": int(add_snapshot.new_ub),
                        "materialized_eq_block_count_before": int(
                            add_snapshot.eq_block_count_before
                        ),
                        "materialized_eq_block_count_after": int(
                            add_snapshot.eq_block_count_after
                        ),
                        "materialized_ub_block_count_before": int(
                            add_snapshot.ub_block_count_before
                        ),
                        "materialized_ub_block_count_after": int(
                            add_snapshot.ub_block_count_after
                        ),
                        "materialized_relation_block_rows": [
                            int(value)
                            for value in add_snapshot.relation_block_rows
                        ],
                        "materialized_relation_block_tags": list(
                            add_snapshot.relation_block_tags
                        ),
                        "materialized_relation_blocks_sha256": (
                            add_snapshot.relation_blocks_sha256
                        ),
                        "materialized_relation_revalidated_at_export": True,
                        "materialized_relation_retained": True,
                        "source_cube_certified_rows_at_zero": int(
                            np.count_nonzero(source_ub < 0.0)
                        ),
                        "effective_cube_improved_groups": int(
                            np.count_nonzero(improvement > 0.0)
                        ),
                        "effective_cube_improvement_sum": float(
                            np.sum(np.maximum(improvement, 0.0))
                        ),
                        "effective_cube_improvement_max": float(
                            np.max(np.maximum(improvement, 0.0))
                            if improvement.size else 0.0
                        ),
                        "effective_cube_upper_before_max": float(
                            np.max(group_upper_prior)
                        ),
                        "effective_cube_upper_after_max": float(
                            np.max(group_upper_after)
                        ),
                        "source_cube_upper_min": float(
                            np.min(source_ub)
                        ),
                        "source_cube_upper_max": float(
                            np.max(source_ub)
                        ),
                    }
                )
            except Exception as exc:
                # This is a safe-only tightening candidate.  A construction
                # failure leaves the already audited materialized rows intact.
                add_source_receipt.update(
                    {
                        "status": "error_fallback_materialized",
                        "proof_authority": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                )
            self._check_deadline("property_tail_add_source_planes")
        suffix_expr, suffix_receipt = (
            self._build_property_suffix_candidate(
                output_layer_id=int(output_layer_id)
            )
        )
        if suffix_expr is not None:
            candidate_groups: List[Tuple[int, ...]] = (
                list(self.property_tail_row_groups)
                if self.property_tail_row_groups
                else [
                    (int(row),)
                    for row in range(baseline_expr.size)
                ]
            )
            prior_flattened = [
                int(row)
                for group in candidate_groups
                for row in group
            ]
            if (
                len(candidate_groups) != baseline_expr.size
                or len(prior_flattened) != upper_expr.size
                or len(set(prior_flattened)) != upper_expr.size
                or set(prior_flattened) != set(range(upper_expr.size))
            ):
                raise OperatorHZBuildError(
                    "property suffix replay found malformed pre-existing "
                    "property groups"
                )
            prior_lb, prior_ub = self._cube_bounds(upper_expr)
            suffix_lb, suffix_ub = self._cube_bounds(suffix_expr)
            group_upper_before = np.asarray(
                [
                    min(float(prior_ub[int(row)]) for row in group)
                    for group in candidate_groups
                ],
                dtype=np.float64,
            )
            group_upper_after = np.minimum(
                group_upper_before, suffix_ub
            )
            suffix_start = int(upper_expr.size)
            width = max(upper_expr.G.shape[1], suffix_expr.G.shape[1])
            candidate_G = sp.vstack(
                [
                    _pad_cols(upper_expr.G, width),
                    _pad_cols(suffix_expr.G, width),
                ],
                format="csr",
            )
            candidate_G.eliminate_zeros()
            upper_expr = _AffineExpr(
                c=np.concatenate([upper_expr.c, suffix_expr.c]),
                G=candidate_G,
                err=np.concatenate([upper_expr.err, suffix_expr.err]),
                affine_depth=max(
                    upper_expr.affine_depth,
                    suffix_expr.affine_depth,
                ),
            )
            suffix_rows = np.arange(
                suffix_start,
                suffix_start + baseline_expr.size,
                dtype=np.int64,
            )
            for rival, row in enumerate(suffix_rows):
                candidate_groups[rival] = (
                    *candidate_groups[rival],
                    int(row),
                )
            self.property_tail_row_groups = tuple(candidate_groups)
            alternative_rival_ids.extend(
                int(value) for value in range(baseline_expr.size)
            )
            alternative_plane_kinds.extend(
                (
                    "query_dual_full_input_property_constant"
                    if suffix_receipt.get("output_form")
                    == "full_input_property_constant"
                    else "query_dual_shared_suffix_add_projection"
                )
                for _ in range(baseline_expr.size)
            )
            # Legacy tail-plane hashes use final-preactivation coordinates.
            # The suffix rows live at an earlier ADD, so zero placeholders
            # keep row accounting aligned while the authoritative expression
            # hashes and replay receipt bind their real coefficients.
            suffix_placeholders = np.zeros(
                (baseline_expr.size, envelope_planes.shape[1]),
                dtype=np.float64,
            )
            envelope_planes = np.vstack(
                [envelope_planes, suffix_placeholders]
            )
            envelope_intercepts = np.concatenate(
                [
                    envelope_intercepts,
                    np.zeros(baseline_expr.size, dtype=np.float64),
                ]
            )
            improvement = group_upper_before - group_upper_after
            suffix_receipt.update(
                {
                    "status": "applied",
                    "proof_authority": True,
                    "row_start": int(suffix_start),
                    "row_count": int(baseline_expr.size),
                    "row_indices_sha256": _f64_array_sha256(
                        suffix_rows.astype(np.float64)
                    ),
                    "legacy_final_preactivation_placeholders": True,
                    "placeholder_sha256": _f64_array_sha256(
                        suffix_placeholders
                    ),
                    "free_cube_improved_groups": int(
                        np.count_nonzero(improvement > 0.0)
                    ),
                    "free_cube_improvement_max": float(
                        np.max(improvement) if improvement.size else 0.0
                    ),
                    "free_cube_improvement_mean": float(
                        np.mean(np.maximum(improvement, 0.0))
                        if improvement.size else 0.0
                    ),
                    "suffix_cube_lower_min": float(np.min(suffix_lb)),
                    "suffix_cube_upper_max": float(np.max(suffix_ub)),
                }
            )
        self._check_deadline("property_tail_suffix_replay")
        verified_query_dual_property: Optional[Dict[str, Any]] = None
        if self.verified_query_dual_feedback is not None:
            if self.verified_query_dual_property_upper is None:
                raise OperatorHZBuildError(
                    "verified query-dual property snapshot is unavailable"
                )
            property_upper = np.ascontiguousarray(
                self.verified_query_dual_property_upper,
                dtype=np.float64,
            )
            if (
                property_upper.shape != (baseline_expr.size,)
                or not np.all(np.isfinite(property_upper))
            ):
                raise OperatorHZBuildError(
                    "verified query-dual property upper has the wrong shape "
                    "or contains a non-finite value"
                )
            candidate_groups: List[Tuple[int, ...]] = (
                list(self.property_tail_row_groups)
                if self.property_tail_row_groups
                else [
                    (int(row),)
                    for row in range(baseline_expr.size)
                ]
            )
            prior_flattened = [
                int(row)
                for group in candidate_groups
                for row in group
            ]
            if (
                len(candidate_groups) != baseline_expr.size
                or len(prior_flattened) != upper_expr.size
                or len(set(prior_flattened)) != upper_expr.size
                or set(prior_flattened) != set(range(upper_expr.size))
            ):
                raise OperatorHZBuildError(
                    "verified query-dual property constants found malformed "
                    "pre-existing property groups"
                )

            constant_start = int(upper_expr.size)
            constant_rows = np.arange(
                constant_start,
                constant_start + baseline_expr.size,
                dtype=np.int64,
            )
            zero_generators = sp.csr_matrix(
                (baseline_expr.size, upper_expr.G.shape[1]),
                dtype=np.float64,
            )
            candidate_G = sp.vstack(
                [upper_expr.G, zero_generators],
                format="csr",
            )
            candidate_upper_expr = _AffineExpr(
                c=np.concatenate([upper_expr.c, property_upper]),
                G=candidate_G,
                err=np.concatenate(
                    [
                        upper_expr.err,
                        np.zeros(
                            baseline_expr.size,
                            dtype=np.float64,
                        ),
                    ]
                ),
                affine_depth=upper_expr.affine_depth,
            )
            zero_planes = np.zeros(
                (baseline_expr.size, snapshot.preactivation.size),
                dtype=np.float64,
            )
            candidate_envelope_planes = np.vstack(
                [envelope_planes, zero_planes]
            )
            candidate_envelope_intercepts = np.concatenate(
                [envelope_intercepts, property_upper]
            )
            for rival, row in enumerate(constant_rows):
                candidate_groups[rival] = (
                    *candidate_groups[rival],
                    int(row),
                )

            # Commit the complete batch only after every value, row, and
            # pre-existing group has passed.  There is no per-rival fallback
            # because a partially consumed transaction would be unauditable.
            upper_expr = candidate_upper_expr
            envelope_planes = candidate_envelope_planes
            envelope_intercepts = candidate_envelope_intercepts
            self.property_tail_row_groups = tuple(candidate_groups)
            alternative_rival_ids.extend(
                int(rival)
                for rival in range(baseline_expr.size)
            )
            alternative_plane_kinds.extend(
                "verified_query_dual_property_constant"
                for _ in range(baseline_expr.size)
            )
            feedback_receipt = self.verified_query_dual_receipt
            if feedback_receipt is None:
                raise OperatorHZBuildError(
                    "verified query-dual receipt snapshot is unavailable"
                )
            verified_query_dual_property = {
                "schema": (
                    "operator_hz_verified_query_dual_property_constant_v1"
                ),
                "status": "applied",
                "proof_authority": True,
                "safe_only": True,
                "baseline_fallback_retained_per_rival": True,
                "constant_row_count": int(baseline_expr.size),
                "constant_row_indices": [
                    int(value) for value in constant_rows
                ],
                "constant_row_indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        constant_rows, dtype=np.int64
                    ).tobytes()
                ).hexdigest(),
                "constant_rival_ids": [
                    int(value)
                    for value in range(baseline_expr.size)
                ],
                "constant_values_hex": [
                    float(value).hex() for value in property_upper
                ],
                "constant_values_sha256": _f64_array_sha256(
                    property_upper
                ),
                "zero_envelope_planes_sha256": _f64_array_sha256(
                    zero_planes
                ),
                "no_output_error_generators": True,
                "root_boxes_sha256": feedback_receipt[
                    "root_boxes_sha256"
                ],
                "final_boxes_sha256": feedback_receipt[
                    "final_boxes_sha256"
                ],
                "property_spec_sha256": feedback_receipt[
                    "property_spec_sha256"
                ],
                "property_upper_sha256": feedback_receipt[
                    "property_upper_sha256"
                ],
                "transaction_receipt_sha256": feedback_receipt[
                    "receipt_sha256"
                ],
            }
            self._check_deadline(
                "property_tail_verified_query_dual_constants"
            )
        if not self.property_tail_row_groups:
            self.property_tail_row_groups = tuple(
                (int(row),) for row in range(baseline_expr.size)
            )
        flattened_group_rows = [
            int(row)
            for group in self.property_tail_row_groups
            for row in group
        ]
        if (
            len(flattened_group_rows) != upper_expr.size
            or len(set(flattened_group_rows)) != upper_expr.size
            or set(flattened_group_rows) != set(range(upper_expr.size))
        ):
            raise OperatorHZBuildError(
                "property-tail row groups do not partition exported planes"
            )
        alternative_count = int(upper_expr.size - baseline_expr.size)
        if (
            alternative_count < 0
            or len(alternative_rival_ids) != alternative_count
            or len(alternative_plane_kinds) != alternative_count
            or envelope_planes.shape[0] != upper_expr.size
            or envelope_intercepts.size != upper_expr.size
        ):
            raise OperatorHZBuildError(
                "property-tail alternative row receipt is inconsistent"
            )
        if (
            upper_expr.G.shape[1] < snapshot.n_cont
        ):
            upper_expr = _AffineExpr(
                c=upper_expr.c,
                G=_pad_cols(upper_expr.G, snapshot.n_cont),
                err=upper_expr.err,
                affine_depth=upper_expr.affine_depth,
            )
        if (
            upper_expr.G.shape[1] < snapshot.n_cont
            or upper_expr.G[:, snapshot.n_cont:].nnz != 0
        ):
            raise OperatorHZBuildError(
                "property upper expression depends on pruned tail columns: "
                f"shape={upper_expr.G.shape}, "
                f"snapshot_n_cont={snapshot.n_cont}, "
                "trailing_nnz="
                f"{upper_expr.G[:, snapshot.n_cont:].nnz}"
            )

        pruned_eq_blocks = self.eq_blocks[snapshot.eq_block_count:]
        pruned_ub_blocks = self.ub_blocks[snapshot.ub_block_count:]
        current_n_cont = int(self.n_cont)
        current_n_bin = int(self.n_bin)
        current_exact_used = int(self.exact_used)
        self.n_cont = int(snapshot.n_cont)
        self.n_bin = int(snapshot.n_bin)
        self.col_ids = self.col_ids[: self.n_cont]
        self.bcol_ids = self.bcol_ids[: self.n_bin]
        self.eq_blocks = self.eq_blocks[: snapshot.eq_block_count]
        self.ub_blocks = self.ub_blocks[: snapshot.ub_block_count]
        self.exact_used = int(snapshot.exact_used)
        upper_expr = _AffineExpr(
            c=upper_expr.c,
            G=upper_expr.G[:, : self.n_cont].tocsr(),
            err=upper_expr.err,
            affine_depth=upper_expr.affine_depth,
        )

        property_digest = hashlib.sha256()
        property_digest.update(
            np.asarray(self.property_upper_C.shape, dtype=np.int64).tobytes()
        )
        property_digest.update(self.property_upper_C.tobytes())
        property_digest.update(
            np.asarray(
                self.property_upper_thresholds.shape, dtype=np.int64
            ).tobytes()
        )
        property_digest.update(self.property_upper_thresholds.tobytes())
        receipt.update(
            {
                "enabled": True,
                "safe_only": True,
                "proof_rule": (
                    "exact_dyadic_CW+stored_plane_fraction_endpoint_envelope+"
                    "exact_pairhull_projection_when_enabled+"
                    "verified_shared_suffix_ADD_projection_when_enabled+"
                    "operator_prefix_support+grouped_alternative_plane_"
                    "coverage"
                ),
                "relu_layer_id": int(snapshot.relu_layer_id),
                "output_layer_id": int(output_layer_id),
                "property_sha256": property_digest.hexdigest(),
                "prefix_n_cont": int(snapshot.n_cont),
                "prefix_n_bin": int(snapshot.n_bin),
                "pruned_n_cont": int(current_n_cont - snapshot.n_cont),
                "pruned_n_bin": int(current_n_bin - snapshot.n_bin),
                "pruned_eq_rows": int(
                    sum(block.Ac.shape[0] for block in pruned_eq_blocks)
                ),
                "pruned_ub_rows": int(
                    sum(block.Ac.shape[0] for block in pruned_ub_blocks)
                ),
                "pruned_constraint_nnz": int(
                    sum(
                        block.Ac.nnz + block.Ab.nnz
                        for block in (*pruned_eq_blocks, *pruned_ub_blocks)
                    )
                ),
                "discarded_tail_exact_count": int(
                    current_exact_used - snapshot.exact_used
                ),
                "upper_expression_nnz": int(upper_expr.G.nnz),
                "baseline_plane_count": int(planes.shape[0]),
                "alternative_plane_count": int(
                    envelope_planes.shape[0] - planes.shape[0]
                ),
                "alternative_plane_rival_ids": alternative_rival_ids,
                "alternative_plane_kinds": alternative_plane_kinds,
                "exported_plane_count": int(envelope_planes.shape[0]),
                "exported_planes_sha256": hashlib.sha256(
                    np.ascontiguousarray(envelope_planes).tobytes()
                ).hexdigest(),
                "exported_intercepts_sha256": hashlib.sha256(
                    np.ascontiguousarray(envelope_intercepts).tobytes()
                ).hexdigest(),
                "upper_expression_error_max": (
                    float(np.max(upper_expr.err))
                    if upper_expr.err.size else 0.0
                ),
                "upper_expression_center_sha256": hashlib.sha256(
                    np.ascontiguousarray(upper_expr.c).tobytes()
                ).hexdigest(),
                "upper_expression_generator_sha256": _csr_sha256(
                    upper_expr.G
                ),
                "upper_expression_error_sha256": hashlib.sha256(
                    np.ascontiguousarray(upper_expr.err).tobytes()
                ).hexdigest(),
                "property_row_groups": [
                    [int(value) for value in group]
                    for group in self.property_tail_row_groups
                ],
                "property_row_groups_sha256": hashlib.sha256(
                    repr(self.property_tail_row_groups).encode("ascii")
                ).hexdigest(),
                "negative_alpha_candidates": alpha_receipt,
                "pairhull_candidates": pairhull_receipt,
                "add_source_planes": add_source_receipt,
                "shared_suffix_replay": suffix_receipt,
                **(
                    {
                        "verified_query_dual_property_constants": (
                            verified_query_dual_property
                        )
                    }
                    if verified_query_dual_property is not None
                    else {}
                ),
            }
        )
        self.property_tail_receipt = receipt
        return upper_expr

    def _process(self, layer: Any) -> None:
        lid = int(layer.id)
        kind = _kind(layer.kind)
        if kind not in _SUPPORTED_KINDS:
            raise OperatorHZBuildError(
                f"unsupported operator-HZ layer {lid}:{kind}; supported="
                f"{sorted(_SUPPORTED_KINDS)}"
            )
        c0, b0 = self.n_cont, self.n_bin
        e0 = self._eq_row_count()
        u0 = self._ub_row_count()

        previous_allocation_layer = self._allocation_layer_id
        self._allocation_layer_id = lid
        try:
            if kind == "INPUT":
                info = self._build_input(layer)
            elif kind == "INPUT_SPEC":
                info = self._build_identity(layer, require_box_spec=True)
            elif kind in _AFFINE_KINDS:
                info = self._build_affine(layer)
            elif kind == "ADD":
                info = self._build_add(layer)
            elif kind == "RELU":
                info = self._build_relu(layer)
            elif kind in {"FLATTEN", "ASSERT"}:
                info = self._build_identity(layer, require_box_spec=False)
            else:  # pragma: no cover - guarded above.
                raise OperatorHZBuildError(
                    f"unhandled supported layer {lid}:{kind}"
                )
        finally:
            self._allocation_layer_id = previous_allocation_layer

        expr = self.exprs[lid]
        if (
            not np.all(np.isfinite(expr.c))
            or not np.all(np.isfinite(expr.G.data))
            or not np.all(np.isfinite(expr.err))
            or np.any(expr.err < 0.0)
        ):
            raise OperatorHZBuildError(f"non-finite expression at layer {lid}:{kind}")
        meta: Dict[str, Any] = {
            "layer_id": lid,
            "kind": kind,
            "n_out": int(expr.size),
            "value_nnz": int(expr.G.nnz),
            "roundoff_error_nonzero": int(np.count_nonzero(expr.err)),
            "roundoff_error_max": (
                float(np.max(expr.err)) if expr.err.size else 0.0
            ),
            "affine_depth": int(expr.affine_depth),
            # ``after`` facts are not proof inputs.  Recomputing a complete
            # cube for every layer solely to report width diagnostics added a
            # second sparse scan to the latency-critical build path.  No
            # repository consumer reads the removed width fields; keep one
            # explicit marker instead of performing non-semantic work.
            "fact_audit": "omitted_nonsemantic_hot_path_v1",
            **info,
        }
        # Audit the declared counts against actual frame deltas.
        meta["frame_cont_delta"] = int(self.n_cont - c0)
        meta["frame_bin_delta"] = int(self.n_bin - b0)
        meta["frame_eq_delta"] = int(self._eq_row_count() - e0)
        meta["frame_ub_delta"] = int(self._ub_row_count() - u0)
        self.layer_metadata.append(meta)
        if lid in self.layer_frame_snapshots:
            raise OperatorHZBuildError(
                f"layer frame snapshot was constructed twice for layer {lid}"
            )
        self.layer_frame_snapshots[lid] = _LayerFrameSnapshot(
            n_cont=int(self.n_cont),
            n_bin=int(self.n_bin),
            eq_rows=self._eq_row_count(),
            ub_rows=self._ub_row_count(),
            eq_block_count=self._eq_block_count(),
            ub_block_count=self._ub_block_count(),
        )

    def _maybe_apply_property_micro_rlt(
        self,
        hz: SparseHZono,
    ) -> Tuple[SparseHZono, Dict[str, Any], Tuple[str, ...]]:
        """Transactionally append the two-bit parent-relaxation micro-RLT.

        The lift is deliberately unavailable to layer-local/prefix solves:
        all layer frame snapshots were frozen before this method, and every
        generated row is appended after the complete ordinary upper-row
        matrix.  Fixed-phase children do not need these rows because the
        signed RLT products reduce to redundant copies of their source rows.
        """

        base_counts = {
            "n_out": int(hz.n_out),
            "n_cont": int(hz.n_cont),
            "n_bin": int(hz.n_bin),
            "n_eq": int(hz.n_eq),
            "n_ub": int(hz.n_ub),
        }
        receipt: Dict[str, Any] = {
            "schema": "operator_hz_property_micro_rlt_v1",
            "requested_product_factor_cap": int(
                self.property_micro_rlt_product_cap
            ),
            "requested_packet_mode": self.property_micro_rlt_packet_mode,
            "selected_packet_record_indices": [],
            "selected_packet_count": 0,
            "enabled": bool(self.property_micro_rlt_product_cap > 0),
            "status": "no_op_disabled",
            "proof_authority": False,
            "selection_proof_authority": False,
            "base_counts": dict(base_counts),
            "result_counts": dict(base_counts),
            "exact_record_count": int(self.exact_phase_record_count),
            "exact_relu_records": [],
            "common_focused_rival_id": None,
            "source_rows_by_binary": [],
            "generated_upper_row_tags": [],
            "intended_consumer": (
                "parent_binary_relaxation_before_exact_phase_enumeration"
            ),
            "parent_pre_enumeration_tightness_potential_only": True,
            "fixed_phase_rows_are_redundant": True,
            "fixed_phase_projection_gain": False,
            "fixed_phase_rows_retained": True,
            "fixed_phase_solver_overhead_only": True,
            "scope": "parent_pre_phase_fix",
            "phase_solver_integration": False,
            "early_row_prefixes_include_generated_rows": False,
            "excluded_from_early_row_prefixes": True,
            "claimed_c38_prefix_tightening": False,
            "ordinary_upper_row_prefix_retained": True,
            "live_result_validation_required": True,
            "live_result_validation_passed": False,
            "property_micro_rlt_receipt_sha256": None,
            "selected_source_row_nnz_cap": (
                _PROPERTY_MICRO_RLT_SELECTED_ROW_NNZ_CAP
            ),
            "requirement_scan_nnz_cap": (
                _PROPERTY_MICRO_RLT_REQUIREMENT_SCAN_NNZ_CAP
            ),
            "required_selected_source_row_nnz": None,
            "required_product_factors": None,
            "requirement_count_complete": False,
            "selected_source_nnz_cap_exceeded": False,
            "product_factor_cap_exceeded": False,
            "primary_cap_failure": None,
            "supported_product_factor_cap_max": (
                _PROPERTY_MICRO_RLT_PRODUCT_FACTOR_CAP_MAX
            ),
            "auxiliary_continuous_provenance_layer_id": -1,
            "auxiliary_continuous_provenance_semantics": (
                "factor_product_auxiliary_not_network_layer"
            ),
            "binary_provenance_extended": False,
            "no_op_reason": "explicit_product_factor_cap_is_zero",
        }

        def finish() -> Dict[str, Any]:
            payload = dict(receipt)
            payload.pop("receipt_sha256", None)
            receipt["receipt_sha256"] = _canonical_json_sha256(payload)
            return receipt

        if self.property_micro_rlt_product_cap <= 0:
            return hz, finish(), ()

        records = list(self.property_exact_phase_records)
        ineligible_reason: Optional[str] = None
        if len(records) != 2:
            ineligible_reason = (
                "requires_exactly_two_property_selected_exact_relu_records"
            )
        elif not all(
            record.get("property_selected") is True for record in records
        ):
            ineligible_reason = "exact_relu_records_are_not_property_selected"
        elif not all(
            record.get("focused_rivals_explicit") is True
            for record in records
        ):
            ineligible_reason = "focused_rival_schedule_is_not_explicit"
        else:
            focused = [
                tuple(int(value) for value in record.get("rival_ids", ()))
                for record in records
            ]
            if (
                any(len(values) != 1 for values in focused)
                or focused[0] != focused[1]
            ):
                ineligible_reason = (
                    "exact_relu_records_do_not_share_one_focused_rival"
                )
            else:
                receipt["common_focused_rival_id"] = int(focused[0][0])
        if ineligible_reason is not None:
            receipt.update(
                {
                    "status": "no_op_ineligible",
                    "no_op_reason": ineligible_reason,
                }
            )
            return hz, finish(), ()

        binary_positions = [
            int(record["binary_position"]) for record in records
        ]
        if (
            len(set(binary_positions)) != 2
            or min(binary_positions) < 0
            or max(binary_positions) >= hz.n_bin
            or hz.bcol_ids is None
            or any(
                int(
                    np.asarray(hz.bcol_ids, dtype=np.int64).reshape(-1)[
                        int(record["binary_position"])
                    ]
                )
                != int(record["binary_col_id"])
                for record in records
            )
        ):
            raise OperatorHZBuildError(
                "property micro-RLT exact binary mapping is malformed"
            )
        ordinary_ub_tags = tuple(
            str(block.tag)
            for block in self.ub_blocks
            for _row in range(int(block.Ac.shape[0]))
        )
        if len(ordinary_ub_tags) != hz.n_ub:
            raise OperatorHZBuildError(
                "property micro-RLT ordinary upper-row tags are misaligned"
            )

        exact_rows: List[Dict[str, int]] = []
        for record in records:
            raw_rows = record.get("exact_upper_rows")
            if not isinstance(raw_rows, Mapping):
                raise OperatorHZBuildError(
                    "property micro-RLT exact row map is missing"
                )
            try:
                mapped = {
                    name: int(raw_rows[name])
                    for name in ("lower", "x_branch", "zero_branch")
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise OperatorHZBuildError(
                    "property micro-RLT exact row map is malformed"
                ) from exc
            layer_id = int(record["layer_id"])
            expected_tags = {
                "lower": f"relu_exact_lower:{layer_id}",
                "x_branch": f"relu_exact_x_branch:{layer_id}",
                "zero_branch": f"relu_exact_zero_branch:{layer_id}",
            }
            if (
                len(set(mapped.values())) != 3
                or min(mapped.values()) < 0
                or max(mapped.values()) >= hz.n_ub
                or any(
                    ordinary_ub_tags[mapped[name]] != expected_tags[name]
                    for name in expected_tags
                )
            ):
                raise OperatorHZBuildError(
                    "property micro-RLT exact global upper-row mapping "
                    "failed its live tag audit"
                )
            exact_rows.append(mapped)
        all_exact_rows = [
            int(mapped[name])
            for mapped in exact_rows
            for name in ("lower", "x_branch", "zero_branch")
        ]
        if len(set(all_exact_rows)) != len(all_exact_rows):
            raise OperatorHZBuildError(
                "property micro-RLT exact global upper rows are not unique"
            )
        assert hz.Aub is not None
        for record, mapped in zip(records, exact_rows):
            selected_binary = int(record["binary_position"])
            lower_binary = hz.Aub.getrow(mapped["lower"])
            x_binary = hz.Aub.getrow(mapped["x_branch"])
            zero_binary = hz.Aub.getrow(mapped["zero_branch"])
            if (
                lower_binary.nnz != 0
                or x_binary.nnz != 1
                or zero_binary.nnz != 1
                or int(x_binary.indices[0]) != selected_binary
                or int(zero_binary.indices[0]) != selected_binary
                or not np.isfinite(x_binary.data[0])
                or not np.isfinite(zero_binary.data[0])
                or x_binary.data[0] <= 0.0
                or zero_binary.data[0] >= 0.0
            ):
                raise OperatorHZBuildError(
                    "property micro-RLT exact selected-binary coefficient "
                    "structure failed its live audit"
                )
        receipt["exact_relu_records"] = [
            {
                "layer_id": int(record["layer_id"]),
                "row": int(record["row"]),
                "binary_position": int(record["binary_position"]),
                "binary_col_id": int(record["binary_col_id"]),
                "lower_upper_row": int(mapped["lower"]),
                "x_branch_upper_row": int(mapped["x_branch"]),
                "zero_branch_upper_row": int(mapped["zero_branch"]),
            }
            for record, mapped in zip(records, exact_rows)
        ]

        directed_packets = (
            (
                binary_positions[0],
                (
                    exact_rows[0]["lower"],
                    exact_rows[0]["x_branch"],
                    exact_rows[0]["zero_branch"],
                    exact_rows[1]["lower"],
                ),
            ),
            (
                binary_positions[1],
                (
                    exact_rows[1]["lower"],
                    exact_rows[1]["x_branch"],
                    exact_rows[1]["zero_branch"],
                    exact_rows[0]["lower"],
                ),
            ),
        )
        if self.property_micro_rlt_packet_mode == "both":
            selected_packet_indices = (0, 1)
        elif self.property_micro_rlt_packet_mode == "first":
            selected_packet_indices = (0,)
        else:
            selected_packet_indices = (1,)
        source_rows_by_binary = {
            int(directed_packets[index][0]): tuple(
                sorted(
                    int(row)
                    for row in directed_packets[index][1]
                )
            )
            for index in selected_packet_indices
        }
        receipt["selected_packet_record_indices"] = [
            int(index) for index in selected_packet_indices
        ]
        receipt["selected_packet_count"] = int(
            len(selected_packet_indices)
        )
        receipt["source_rows_by_binary"] = [
            {
                "binary_position": int(binary),
                "source_upper_rows": [
                    int(row) for row in rows
                ],
            }
            for binary, rows in sorted(source_rows_by_binary.items())
        ]

        self._check_deadline("property_micro_rlt_before")
        try:
            from act.back_end.hybridz_tf.property_micro_rlt import (
                PropertyMicroRLTError,
                apply_property_micro_rlt,
                verify_property_micro_rlt_result,
            )
        except Exception as exc:
            raise OperatorHZBuildError(
                "property micro-RLT implementation import failed: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            ) from exc
        try:
            result = apply_property_micro_rlt(
                hz,
                source_rows_by_binary=source_rows_by_binary,
                max_binary_factors=2,
                max_source_rows_per_binary=4,
                max_product_factors=int(
                    self.property_micro_rlt_product_cap
                ),
                max_selected_row_nnz=(
                    _PROPERTY_MICRO_RLT_SELECTED_ROW_NNZ_CAP
                ),
                max_requirement_scan_nnz=(
                    _PROPERTY_MICRO_RLT_REQUIREMENT_SCAN_NNZ_CAP
                ),
            )
        except PropertyMicroRLTError as exc:
            reason_code = getattr(exc, "reason_code", None)
            if reason_code not in {
                "requirement_scan_cap_exceeded",
                "selected_source_row_nnz_cap_exceeded",
                "product_factor_cap_exceeded",
            }:
                raise OperatorHZBuildError(
                    "property micro-RLT construction failed: "
                    f"{str(exc)[:500]}"
                ) from exc
            required_selected_nnz = getattr(
                exc,
                "selected_source_row_nnz_required",
                None,
            )
            required_product_factors = getattr(
                exc,
                "product_factors_required",
                None,
            )
            receipt.update(
                {
                    "status": "no_op_cap_exceeded",
                    "no_op_reason": str(exc),
                    "required_selected_source_row_nnz": (
                        required_selected_nnz
                    ),
                    "required_product_factors": (
                        required_product_factors
                    ),
                    "requirement_count_complete": bool(
                        getattr(
                            exc,
                            "requirement_count_complete",
                            False,
                        )
                    ),
                    "selected_source_nnz_cap_exceeded": (
                        required_selected_nnz is not None
                        and int(required_selected_nnz)
                        > _PROPERTY_MICRO_RLT_SELECTED_ROW_NNZ_CAP
                    ),
                    "product_factor_cap_exceeded": (
                        required_product_factors is not None
                        and int(required_product_factors)
                        > int(self.property_micro_rlt_product_cap)
                    ),
                    "primary_cap_failure": str(reason_code),
                }
            )
            return hz, finish(), ()
        except Exception as exc:
            if isinstance(exc, OperatorHZBuildTimeout):
                raise
            raise OperatorHZBuildError(
                "property micro-RLT construction raised "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            ) from exc

        if verify_property_micro_rlt_result(result) is not True:
            raise OperatorHZBuildError(
                "property micro-RLT failed live result validation"
            )
        self._check_deadline("property_micro_rlt_after")
        candidate = result.hz
        lift_receipt = result.receipt
        old_n_cont = int(hz.n_cont)
        if (
            candidate.n_out != hz.n_out
            or candidate.n_bin != hz.n_bin
            or candidate.n_eq != hz.n_eq
            or candidate.n_cont < old_n_cont
            or candidate.n_ub <= hz.n_ub
            or candidate.col_ids is None
            or hz.col_ids is None
            or not np.array_equal(
                np.asarray(candidate.col_ids, dtype=np.int64)[:old_n_cont],
                np.asarray(hz.col_ids, dtype=np.int64),
            )
        ):
            raise OperatorHZBuildError(
                "property micro-RLT changed the ordinary HZ prefix"
            )

        candidate_ids = np.asarray(
            candidate.col_ids, dtype=np.int64
        ).reshape(-1)
        new_ids = candidate_ids[old_n_cont:]
        if not isinstance(lift_receipt, Mapping):
            raise OperatorHZBuildError(
                "property micro-RLT live receipt is not a mapping"
            )
        raw_generated_names = lift_receipt.get("generated_row_names")
        if not isinstance(raw_generated_names, list):
            raise OperatorHZBuildError(
                "property micro-RLT generated-row names are missing"
            )
        generated_names = tuple(
            "property_micro_rlt:" + str(name)
            for name in raw_generated_names
        )
        if len(generated_names) != candidate.n_ub - hz.n_ub:
            raise OperatorHZBuildError(
                "property micro-RLT generated-row tags are incomplete"
            )
        try:
            staged_new_product_factors = int(
                lift_receipt["new_product_factors"]
            )
            staged_new_upper_rows = int(
                lift_receipt["new_upper_rows"]
            )
            staged_lift_sha256 = str(lift_receipt["receipt_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OperatorHZBuildError(
                "property micro-RLT live receipt counts are malformed"
            ) from exc
        if (
            staged_new_product_factors != candidate.n_cont - hz.n_cont
            or staged_new_upper_rows != candidate.n_ub - hz.n_ub
            or len(staged_lift_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in staged_lift_sha256
            )
        ):
            raise OperatorHZBuildError(
                "property micro-RLT live receipt counts/hashes do not bind "
                "the candidate"
            )
        staged_col_ids = [
            int(value) for value in candidate_ids.tolist()
        ]
        staged_provenance = dict(self.cont_column_layer_by_id)
        for stable_id in new_ids.tolist():
            staged_provenance[int(stable_id)] = -1
        receipt.update(
            {
                "status": "applied",
                "proof_authority": True,
                "no_op_reason": None,
                "result_counts": {
                    "n_out": int(candidate.n_out),
                    "n_cont": int(candidate.n_cont),
                    "n_bin": int(candidate.n_bin),
                    "n_eq": int(candidate.n_eq),
                    "n_ub": int(candidate.n_ub),
                },
                "new_product_factors": staged_new_product_factors,
                "required_selected_source_row_nnz": int(
                    lift_receipt["selected_source_row_nnz"]
                ),
                "required_product_factors": (
                    staged_new_product_factors
                ),
                "requirement_count_complete": True,
                "selected_source_nnz_cap_exceeded": False,
                "product_factor_cap_exceeded": False,
                "primary_cap_failure": None,
                "new_upper_rows": staged_new_upper_rows,
                "generated_upper_row_start": int(hz.n_ub),
                "generated_upper_row_tags": list(generated_names),
                "property_micro_rlt_receipt_sha256": staged_lift_sha256,
                "live_result_validation_passed": True,
            }
        )
        finished_receipt = finish()

        # Commit builder dimensions/IDs/provenance only after the complete
        # candidate, live receipt, counts, hash, and generated-row tags have
        # passed.  Every earlier failure therefore leaves ``self`` untouched.
        self.n_cont = int(candidate.n_cont)
        self.col_ids = staged_col_ids
        self.cont_column_layer_by_id = staged_provenance
        return candidate, finished_receipt, generated_names

    def build(self) -> OperatorHZBuild:
        started = time.monotonic()
        process_cpu_started = time.process_time()
        usage_started = resource.getrusage(resource.RUSAGE_SELF)
        performance_diagnostic: Dict[str, Any] = {
            "schema": "operator_hz_build_performance_diagnostic_v1",
            "candidate_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "layers": [],
            "stages": {},
        }
        self._check_deadline("topology")
        order = self._topological_layers()
        self._initialize_constraint_program_sink(order)
        if (
            self.property_tail_suffix_blocks > 0
            and self.property_tail_suffix_blocks != 8
        ):
            if self.property_tail_output_layer_id is None:
                raise OperatorHZBuildError(
                    "property suffix replay has no output layer"
                )
            stop_lid, _candidates = self._property_suffix_stop_layer(
                output_layer_id=int(self.property_tail_output_layer_id)
            )
            self.property_suffix_stop_layer_id = int(stop_lid)
        topology_started = time.monotonic()
        for layer in order:
            self._check_deadline(f"layer_{int(layer.id)}_before")
            layer_wall_started = time.monotonic()
            layer_cpu_started = time.process_time()
            layer_usage_started = resource.getrusage(resource.RUSAGE_SELF)
            self._process(layer)
            layer_usage_finished = resource.getrusage(resource.RUSAGE_SELF)
            layer_wall_seconds = float(
                time.monotonic() - layer_wall_started
            )
            layer_cpu_seconds = float(
                time.process_time() - layer_cpu_started
            )
            performance_diagnostic["layers"].append({
                "layer_id": int(layer.id),
                "kind": _kind(layer.kind),
                "wall_seconds": layer_wall_seconds,
                "process_cpu_seconds": layer_cpu_seconds,
                "minor_faults_delta": int(
                    layer_usage_finished.ru_minflt
                    - layer_usage_started.ru_minflt
                ),
                "major_faults_delta": int(
                    layer_usage_finished.ru_majflt
                    - layer_usage_started.ru_majflt
                ),
                "voluntary_context_switches_delta": int(
                    layer_usage_finished.ru_nvcsw
                    - layer_usage_started.ru_nvcsw
                ),
                "involuntary_context_switches_delta": int(
                    layer_usage_finished.ru_nivcsw
                    - layer_usage_started.ru_nivcsw
                ),
            })
            self._check_deadline(f"layer_{int(layer.id)}_after")
        performance_diagnostic["stages"]["topology_layers_wall_seconds"] = (
            float(time.monotonic() - topology_started)
        )

        self._check_deadline("final_assembly")
        inputs = [layer for layer in order if _kind(layer.kind) == "INPUT"]
        asserts = [layer for layer in order if _kind(layer.kind) == "ASSERT"]
        if len(inputs) != 1 or len(asserts) != 1:
            raise OperatorHZBuildError(
                f"strict operator-HZ requires one INPUT and one ASSERT; "
                f"got INPUT={len(inputs)}, ASSERT={len(asserts)}"
            )
        self._validate_input_spec_enclosure(order)
        assert_layer = asserts[0]
        assert_preds = self._preds(assert_layer, 1)
        output_layer_id = int(assert_preds[0])
        property_upper_output = self.property_upper_C is not None
        final_output_started = time.monotonic()
        if property_upper_output:
            output = self._build_property_tail_upper(
                output_layer_id=output_layer_id
            )
        else:
            output = self._align(self.exprs[int(assert_layer.id)])

        # Do not drop the final affine numerical allowance.  Materializing at
        # most one independent factor per logit is cheap (100 for CIFAR100,
        # 200 for TinyImageNet) and makes the exported HZ semantics explicit.
        output_error = np.asarray(output.err, dtype=np.float64).reshape(-1)
        output_error_rows = np.flatnonzero(output_error > 0.0).astype(
            np.int64, copy=False
        )
        output_error_cols = self._allocate_cont(
            int(output_error_rows.size),
            layer_id=int(output_layer_id),
        )
        output_G = _pad_cols(output.G, self.n_cont)
        if output_error_rows.size:
            output_G = (
                output_G
                + sp.csr_matrix(
                    (
                        output_error[output_error_rows],
                        (output_error_rows, output_error_cols),
                    ),
                    shape=(output.size, self.n_cont),
                    dtype=np.float64,
                )
            ).tocsr()
            output_G.eliminate_zeros()
        output = _AffineExpr(
            output.c,
            output_G,
            np.zeros(output.size, dtype=np.float64),
            affine_depth=output.affine_depth,
        )
        performance_diagnostic["stages"][
            "final_output_materialization_wall_seconds"
        ] = float(time.monotonic() - final_output_started)

        # Topology propagation is complete and ``output`` now owns every
        # value array needed below.  None of the traversal expressions or
        # residual/correlation shadow caches are consulted during final CSR
        # assembly, sealing, or metadata construction.  Snapshot their small
        # diagnostic counts, then release the large sparse graphs before the
        # final constraint matrix is materialized.  This is a lifetime-only
        # optimization: no coefficient, row, bound, or proof receipt changes.
        cache_release_started = time.monotonic()
        correlation_shadow_sources_captured = int(
            len(self.correlation_add_sources)
        )
        correlation_shadow_rows_prepared = int(
            sum(
                len(rows)
                for rows, _shadow in self.correlation_relu_shadows.values()
            )
        )
        traversal_cache_release = {
            "schema": "operator_hz_traversal_cache_release_v1",
            "status": "released_before_final_sparse_assembly",
            "proof_authority": False,
            "numeric_semantics_changed": False,
            "expr_count": int(len(self.exprs)),
            "expr_value_nnz_reference_sum": int(
                sum(int(expr.G.nnz) for expr in self.exprs.values())
            ),
            "residual_skip_shadow_count": int(
                len(self.residual_skip_shadows)
            ),
            "residual_skip_shadow_nnz_reference_sum": int(
                sum(
                    int(expr.G.nnz)
                    for expr in self.residual_skip_shadows.values()
                )
            ),
            "correlation_source_count": (
                correlation_shadow_sources_captured
            ),
            "correlation_source_nnz_reference_sum": int(
                sum(
                    int(expr.G.nnz)
                    for expr in self.correlation_add_sources.values()
                )
            ),
            "correlation_shadow_count": int(
                len(self.correlation_relu_shadows)
            ),
            "correlation_shadow_nnz_reference_sum": int(
                sum(
                    int(expr.G.nnz)
                    for _rows, expr in self.correlation_relu_shadows.values()
                )
            ),
        }
        self.exprs.clear()
        self.residual_skip_shadows.clear()
        self.correlation_add_sources.clear()
        self.correlation_relu_shadows.clear()
        performance_diagnostic["stages"][
            "traversal_cache_release_wall_seconds"
        ] = float(time.monotonic() - cache_release_started)

        final_assembly_started = time.monotonic()
        constraint_program = None
        constraint_sink = self._constraint_program_sink
        if constraint_sink is None:
            Ac = _stack_padded(
                (block.Ac for block in self.eq_blocks), width=self.n_cont
            )
            Ab = _stack_padded(
                (block.Ab for block in self.eq_blocks), width=self.n_bin
            )
            b = (
                np.concatenate([block.rhs for block in self.eq_blocks])
                if self.eq_blocks
                else np.zeros(0, dtype=np.float64)
            )
            Auc = _stack_padded(
                (block.Ac for block in self.ub_blocks), width=self.n_cont
            )
            Aub = _stack_padded(
                (block.Ab for block in self.ub_blocks), width=self.n_bin
            )
            ub = (
                np.concatenate([block.rhs for block in self.ub_blocks])
                if self.ub_blocks
                else np.zeros(0, dtype=np.float64)
            )
            base_eq_tag_rows = tuple(
                (str(block.tag), int(block.Ac.shape[0]))
                for block in self.eq_blocks
            )
            base_ub_tag_rows = tuple(
                (str(block.tag), int(block.Ac.shape[0]))
                for block in self.ub_blocks
            )
            released_equality_blocks = int(len(self.eq_blocks))
            released_upper_blocks = int(len(self.ub_blocks))
            released_equality_nnz = int(
                sum(int(block.Ac.nnz) for block in self.eq_blocks)
            )
            released_upper_nnz = int(
                sum(int(block.Ac.nnz) for block in self.ub_blocks)
            )
        else:
            if self.eq_blocks or self.ub_blocks:
                raise OperatorHZBuildError(
                    "constraint-program final assembly retained mutable "
                    "legacy blocks"
                )
            Ac = sp.csr_matrix((0, self.n_cont), dtype=np.float64)
            Ab = sp.csr_matrix((0, self.n_bin), dtype=np.float64)
            b = np.zeros(0, dtype=np.float64)
            (
                constraint_program,
                Auc,
                Aub,
                ub,
                replayed_upper_tags,
            ) = constraint_sink.seal_and_replay(
                expected_continuous_ids=self.col_ids,
                expected_binary_ids=self.bcol_ids,
            )
            base_eq_tag_rows = ()
            base_ub_tag_rows = tuple(constraint_sink.legacy_tag_rows)
            expected_upper_tags = tuple(
                tag
                for tag, rows in base_ub_tag_rows
                for _row in range(int(rows))
            )
            if (
                replayed_upper_tags != expected_upper_tags
                or int(ub.size) != constraint_sink.virtual_rows
                or int(Auc.shape[0]) != constraint_sink.virtual_rows
                or int(Aub.shape[0]) != constraint_sink.virtual_rows
                or constraint_program.virtual_facet_rows != int(ub.size)
                or constraint_program.source_rows
                != constraint_sink.source_rows
            ):
                raise OperatorHZBuildError(
                    "constraint-program legacy replay changed final row "
                    "count or order"
                )
            released_equality_blocks = 0
            released_upper_blocks = int(
                len(constraint_sink.legacy_tag_rows)
            )
            released_equality_nnz = 0
            # Preserve the established metadata meaning: continuous nnz in
            # the fully expanded legacy upper-facet matrix, not compressed
            # native source storage.
            released_upper_nnz = int(
                constraint_sink.legacy_cont_nnz
            )
        base_solver_constraint_row_tags = tuple(
            tag
            for tag, rows in (*base_eq_tag_rows, *base_ub_tag_rows)
            for _row in range(rows)
        )
        traversal_cache_release.update({
            "equality_block_count_released": released_equality_blocks,
            "upper_block_count_released": released_upper_blocks,
            "equality_block_nnz_released": released_equality_nnz,
            "upper_block_nnz_released": released_upper_nnz,
        })
        constraint_blocks_released_before_constructor = bool(
            self.property_micro_rlt_product_cap <= 0
        )
        if constraint_blocks_released_before_constructor:
            self.eq_blocks.clear()
            self.ub_blocks.clear()
        traversal_cache_release[
            "constraint_blocks_released_before_constructor"
        ] = constraint_blocks_released_before_constructor
        Gb = sp.csr_matrix((output.size, self.n_bin), dtype=np.float64)
        performance_diagnostic["stages"][
            "final_sparse_assembly_wall_seconds"
        ] = float(time.monotonic() - final_assembly_started)

        sparse_hz_constructor_started = time.monotonic()
        hz = _assemble_owned_operator_sparse_hz(
            c=output.c,
            Gc=output.G,
            Gb=Gb,
            Ac=Ac,
            Ab=Ab,
            b=b,
            Auc=Auc,
            Aub=Aub,
            ub=ub,
            col_ids=np.asarray(self.col_ids, dtype=np.int64),
            bcol_ids=np.asarray(self.bcol_ids, dtype=np.int64),
        )
        performance_diagnostic["stages"][
            "sparse_hz_constructor_wall_seconds"
        ] = float(time.monotonic() - sparse_hz_constructor_started)
        # This is the sole integration point for the optional factor lift.
        # It runs after the complete ordinary SparseHZono exists, but before
        # any process-local capability, provenance, prefix, input-replay, or
        # constructive-nonempty decoration is attached.  A no-op returns the
        # original object and leaves every builder dimension/ID unchanged.
        micro_rlt_started = time.monotonic()
        (
            hz,
            property_micro_rlt_receipt,
            property_micro_rlt_upper_tags,
        ) = self._maybe_apply_property_micro_rlt(hz)
        if not constraint_blocks_released_before_constructor:
            self.eq_blocks.clear()
            self.ub_blocks.clear()
        traversal_cache_release[
            "constraint_blocks_released_after_micro_rlt"
        ] = bool(not constraint_blocks_released_before_constructor)
        performance_diagnostic["stages"][
            "property_micro_rlt_wall_seconds"
        ] = float(time.monotonic() - micro_rlt_started)
        continuous_column_layers = np.asarray(
            [
                int(self.cont_column_layer_by_id.get(int(stable_id), -1))
                for stable_id in self.col_ids
            ],
            dtype=np.int64,
        )
        if continuous_column_layers.size != hz.n_cont:
            raise OperatorHZBuildError(
                "continuous column provenance width mismatch"
            )
        setattr(
            hz,
            "_solver_continuous_column_layer_ids",
            continuous_column_layers,
        )
        setattr(
            hz,
            "_solver_constraint_row_tags",
            (
                base_solver_constraint_row_tags
                + property_micro_rlt_upper_tags
            ),
        )
        if self.property_conditional_suffix_rows:
            _hz_attach_exact_phase_conditional_property_rows_from_operator(
                hz,
                self.property_conditional_suffix_rows,
            )
        # C16: bind each deep shared-suffix output row to the exact constraint
        # prefix present immediately after its stop ADD.  This is process-local
        # scheduling metadata only; the solver revalidates every count against
        # the actual final CSR matrices and independently checks every dual
        # certificate over the selected stored rows.  Retaining all generator
        # variables (including final numerical-error factors) avoids any
        # objective projection or hidden roundoff assumption.
        row_prefix_frames: Dict[int, Dict[str, Any]] = {}
        if self.property_tail_receipt is not None:
            suffix_receipt = self.property_tail_receipt.get(
                "shared_suffix_replay", {}
            )
            if (
                isinstance(suffix_receipt, dict)
                and suffix_receipt.get("status") == "applied"
                and suffix_receipt.get("proof_authority") is True
                and suffix_receipt.get("output_form")
                != "full_input_property_constant"
            ):
                stop_lid = int(suffix_receipt["stop_layer_id"])
                source_snapshot = (
                    self.property_suffix_add_source_snapshot
                )
                if (
                    source_snapshot is None
                    or source_snapshot.add_layer_id != stop_lid
                ):
                    raise OperatorHZBuildError(
                        "property suffix stop layer has no correlated "
                        "pre-materialization frame snapshot"
                    )
                frame = _LayerFrameSnapshot(
                    n_cont=int(source_snapshot.n_cont),
                    n_bin=int(source_snapshot.n_bin),
                    eq_rows=int(source_snapshot.eq_rows),
                    ub_rows=int(source_snapshot.ub_rows),
                    eq_block_count=int(
                        source_snapshot.eq_block_count
                    ),
                    ub_block_count=int(
                        source_snapshot.ub_block_count
                    ),
                )
                row_start = int(suffix_receipt["row_start"])
                row_count = int(suffix_receipt["row_count"])
                if (
                    row_start < 0
                    or row_count <= 0
                    or row_start + row_count > hz.n_out
                    or not 0 <= frame.n_cont <= hz.n_cont
                    or not 0 <= frame.n_bin <= hz.n_bin
                    or not 0 <= frame.eq_rows <= hz.n_eq
                    or not 0 <= frame.ub_rows <= hz.n_ub
                ):
                    raise OperatorHZBuildError(
                        "property suffix row-local prefix frame is malformed"
                    )
                prefix_eq = hz.Ac[: frame.eq_rows, :].tocsr()
                prefix_ub = (
                    sp.csr_matrix(
                        (0, hz.n_cont), dtype=np.float64
                    )
                    if hz.Auc is None
                    else hz.Auc[: frame.ub_rows, :].tocsr()
                )
                prefix_eq_sha256 = _csr_sha256(prefix_eq)
                prefix_ub_sha256 = _csr_sha256(prefix_ub)
                suffix_receipt.update(
                    {
                        "row_local_prefix_lp_schema": (
                            "operator_hz_row_constraint_prefix_v1"
                        ),
                        "row_local_prefix_lp_proof_rule": (
                            "drop_later_constraints_only_keeps_full_"
                            "feasible_set_as_subset"
                        ),
                        "row_local_prefix_lp_retains_all_variables": True,
                        "row_local_prefix_lp_candidate_only": True,
                        "prefix_frame_n_cont": int(frame.n_cont),
                        "prefix_frame_n_bin": int(frame.n_bin),
                        "prefix_frame_eq_rows": int(frame.eq_rows),
                        "prefix_frame_ub_rows": int(frame.ub_rows),
                        "prefix_frame_eq_block_count": int(
                            frame.eq_block_count
                        ),
                        "prefix_frame_ub_block_count": int(
                            frame.ub_block_count
                        ),
                        "prefix_frame_eq_csr_sha256": (
                            prefix_eq_sha256
                        ),
                        "prefix_frame_ub_csr_sha256": (
                            prefix_ub_sha256
                        ),
                    }
                )
                for row in range(row_start, row_start + row_count):
                    row_prefix_frames[int(row)] = {
                        "schema": (
                            "operator_hz_row_constraint_prefix_v1"
                        ),
                        "spec_row": int(row),
                        "output_row": int(row),
                        "stop_layer_id": int(stop_lid),
                        "n_cont": int(frame.n_cont),
                        "n_bin": int(frame.n_bin),
                        "eq_rows": int(frame.eq_rows),
                        "ub_rows": int(frame.ub_rows),
                        "eq_csr_sha256": prefix_eq_sha256,
                        "ub_csr_sha256": prefix_ub_sha256,
                    }
        setattr(hz, "_solver_row_constraint_prefix_frames", row_prefix_frames)
        setattr(
            hz,
            "_property_full_input_replay_result",
            self.property_full_input_replay_result,
        )
        if (
            self.input_col_ids is None
            or self.input_center is None
            or self.input_radius is None
            or self.input_layer_id is None
        ):
            raise OperatorHZBuildError("operator-HZ input provenance was not built")
        hz.full_col_ids = self.input_col_ids.copy()
        hz.operator_input_center = self.input_center.copy()
        hz.operator_input_radius = self.input_radius.copy()
        constructive_reason = (
            "property_micro_rlt_exact_integer_extension:"
            "operator_hz_outward_transfer_induction_v1"
            if property_micro_rlt_receipt.get("status") == "applied"
            else "operator_hz_outward_transfer_induction_v1"
        )
        hz_mark_constructively_nonempty(
            hz,
            constructive_reason,
        )

        metadata: Dict[str, Any] = {
            "schema": "operator_hz_local_graph_v1",
            "soundness": (
                "affine_roundoff_envelopes+local_equality_bands+"
                "relu_box_lower_or_guarded_exact_big_m;"
                "cube_bounds_independent_of_interval_facts;"
                "optional_add_affine_relu_rows_use_stored_center_outward_"
                "mass_prescreen_then_chunked_composition_and_collapsed_"
                "rowmass_cube_recheck;"
                "optional_original_constraint_bounds_require_independent_"
                "longdouble_lagrangian_certificate;"
                "optional_property_conditioned_add_affine_rows_intersect_"
                "the_materialized_cube_with_an_outward_pre_add_shadow;"
                "optional_residual_phase_screen_recomposes_only_unstable_"
                "rows_and_commits_only_outward_proven_phases;"
                "optional_residual_bound_screen_keeps_only_strict_outward_"
                "row_bound_intersections_and_releases_transient_generators;"
                "optional_relu_residual_normal_form_uses_fraction_endpoint_"
                "envelope_and_explicit_property_targets;"
                "optional_positive_exact_budget_uses_only_explicit_"
                "property_gap_adjoint_targets_and_exact_big_m;"
                "optional_same_layer_exact_reservoir_changes_only_"
                "post_screen_exact_row_selection_and_leaves_unused_"
                "backups_on_the_ordinary_triangle;"
                "optional_safe_only_property_tail_uses_exact_dyadic_CW_and_"
                "fraction_endpoint_upper_planes;"
                "optional_final_add_source_rows_retain_the_materialized_"
                "frame_and_use_the_inductive_pre_materialization_enclosure;"
                "optional_two_exact_relu_property_micro_rlt_appends_"
                "fraction_audited_parent_relaxation_rows_after_all_early_"
                "constraint_prefixes;"
                "final_roundoff_diagonal_materialized"
            ),
            "supported_kinds": sorted(_SUPPORTED_KINDS),
            "exact_budget_requested": int(self.exact_budget),
            "exact_budget_used": int(self.exact_used),
            "property_micro_rlt": property_micro_rlt_receipt,
            "exact_selection": (
                "property_gap_adjoint_same_layer_rbs_reservoir_v1"
                if self.exact_target_reservoir is not None
                else (
                    "property_gap_adjoint_facility_targets_v1"
                    if (
                        self.exact_budget > 0
                        and self.residual_targets is not None
                    )
                    else "topological_prefix_v1"
                )
            ),
            "exact_target_reservoir_requested": bool(
                self.exact_target_reservoir is not None
            ),
            "exact_target_reservoir_primary_count": int(
                sum(len(rows) for rows in self.residual_targets.values())
                if self.exact_target_reservoir is not None
                and self.residual_targets is not None
                else 0
            ),
            "exact_target_reservoir_backup_count": int(
                sum(
                    len(rows)
                    for rows in self.exact_target_reservoir.values()
                )
                if self.exact_target_reservoir is not None
                else 0
            ),
            "exact_target_reservoir_replacements_used": int(
                sum(
                    int(item.get("replacement_count", 0))
                    for item in self.exact_target_reservoir_receipts
                )
            ),
            "exact_target_reservoir_primary_rbs_tightened": int(
                sum(
                    len(item.get("primary_rows_rbs_tightened", ()))
                    for item in self.exact_target_reservoir_receipts
                )
            ),
            "exact_target_reservoir_selected_rbs_tightened": int(
                sum(
                    len(item.get("selected_rows_rbs_tightened", ()))
                    for item in self.exact_target_reservoir_receipts
                )
            ),
            "exact_target_reservoir_shortfall": int(
                sum(
                    int(item.get("shortfall", 0))
                    for item in self.exact_target_reservoir_receipts
                )
            ),
            "exact_target_reservoir_receipts": (
                self.exact_target_reservoir_receipts
            ),
            "preactivation_lp_budget_requested": int(
                self.preactivation_lp_budget
            ),
            "preactivation_lp_budget_used": int(
                self.preactivation_lp_used
            ),
            "preactivation_lp_time_limit": float(
                self.preactivation_lp_time_limit
            ),
            "preactivation_lp_elapsed_seconds": float(
                0.0
                if self.preactivation_lp_started_at is None
                else max(
                    0.0,
                    time.monotonic() - self.preactivation_lp_started_at,
                )
            ),
            "preactivation_lp_snapshot_seconds": float(
                self.preactivation_lp_snapshot_seconds
            ),
            "preactivation_lp_candidate_seconds": float(
                self.preactivation_lp_candidate_seconds
            ),
            "preactivation_lp_certificate_seconds": float(
                self.preactivation_lp_certificate_seconds
            ),
            "preactivation_lp_persistent_model_builds": int(
                self.preactivation_lp_model_builds
            ),
            "preactivation_lp_deadline_stage": (
                self.preactivation_lp_deadline_stage
            ),
            "preactivation_targets_explicit": bool(
                self.preactivation_targets is not None
            ),
            "preactivation_target_count": int(
                sum(
                    len(rows)
                    for rows in self.preactivation_targets.values()
                )
                if self.preactivation_targets is not None
                else 0
            ),
            "preactivation_lp_certificate_schema": (
                "operator_hz_preactivation_lagrangian_v1"
            ),
            "correlation_targets_explicit": bool(
                self.correlation_targets is not None
            ),
            "correlation_target_count": int(
                sum(
                    len(rows)
                    for rows in self.correlation_targets.values()
                )
                if self.correlation_targets is not None
                else 0
            ),
            "correlation_shadow_sources_captured": int(
                correlation_shadow_sources_captured
            ),
            "correlation_shadow_rows_prepared": int(
                correlation_shadow_rows_prepared
            ),
            "correlation_shadow_rows_tightened": int(
                sum(
                    int(item.get("rows_tightened", 0))
                    for item in self.correlation_shadow_receipts
                    if item.get("status") == "applied"
                )
            ),
            "correlation_shadow_receipts": (
                self.correlation_shadow_receipts
            ),
            "residual_phase_screen_requested": bool(
                self.residual_phase_screen
            ),
            "residual_bound_screen_requested": bool(
                self.residual_bound_screen
            ),
            "residual_bound_screen_rows_tightened": int(
                sum(
                    int(item.get("retained_count", 0))
                    for item in self.residual_phase_screen_receipts
                    if (
                        item.get("status") == "prepared"
                        and item.get("mode")
                        == "strict_bound_improvement"
                    )
                )
            ),
            "residual_phase_screen_layers_prepared": int(
                sum(
                    item.get("status") == "prepared"
                    for item in self.residual_phase_screen_receipts
                )
            ),
            "residual_phase_screen_rows_scanned": int(
                sum(
                    int(item.get("unstable_rows_scanned", 0))
                    for item in self.residual_phase_screen_receipts
                )
            ),
            "residual_phase_screen_stabilized_active": int(
                sum(
                    int(item.get("stabilized_active", 0))
                    for item in self.residual_phase_screen_receipts
                    if item.get("status") == "prepared"
                )
            ),
            "residual_phase_screen_stabilized_inactive": int(
                sum(
                    int(item.get("stabilized_inactive", 0))
                    for item in self.residual_phase_screen_receipts
                    if item.get("status") == "prepared"
                )
            ),
            "residual_phase_screen_elapsed_seconds": float(
                sum(
                    float(item.get("elapsed_seconds", 0.0))
                    for item in self.residual_phase_screen_receipts
                )
            ),
            "residual_phase_screen_receipts": (
                self.residual_phase_screen_receipts
            ),
            "residual_targets_explicit": bool(
                self.residual_targets is not None
            ),
            "residual_target_count": int(
                sum(len(rows) for rows in self.residual_targets.values())
                if self.residual_targets is not None
                else 0
            ),
            "residual_targets_applied": int(
                sum(
                    item.get("status") == "applied"
                    for item in self.residual_target_receipts
                )
            ),
            "residual_target_receipts": self.residual_target_receipts,
            "residual_normal_form": (
                "y=stored_secant*x+rho;"
                "rho_box_fraction_endpoint_envelope_v1"
            ),
            "property_upper_output": bool(property_upper_output),
            "property_upper_semantics": (
                "safe_only_affine_dominating_rows"
                if property_upper_output
                else "network_output_graph_enclosure"
            ),
            "property_tail_add_source_planes_requested": bool(
                self.property_tail_add_source_planes
            ),
            "property_tail_pairhull_budget_requested": int(
                self.property_tail_pairhull_budget
            ),
            "property_tail_pairhull_time_limit_requested": float(
                self.property_tail_pairhull_time_limit
            ),
            "property_tail_suffix_blocks_requested": int(
                self.property_tail_suffix_blocks
            ),
            "property_tail_suffix_alpha_steps_requested": int(
                self.property_tail_suffix_alpha_steps
            ),
            "property_tail_suffix_alpha_time_limit_requested": float(
                self.property_tail_suffix_alpha_time_limit
            ),
            "property_tail_suffix_alpha_device_requested": str(
                self.property_tail_suffix_alpha_device
            ),
            "property_tail_upper": (
                self.property_tail_receipt
                if self.property_tail_receipt is not None
                else {
                    "schema": "operator_hz_property_tail_fraction_v1",
                    "enabled": False,
                    "proof_authority": False,
                }
            ),
            "materialize_add": bool(self.materialize_add),
            "live_affine_relu_enabled": bool(not self.materialize_add),
            "live_affine_relu_inactive_authority": (
                "stored_center_outward_mass_prescreen_then_composed_"
                "cube_recheck_v2"
            ),
            "live_affine_relu_chunk_rows": int(_LIVE_AFFINE_CHUNK_ROWS),
            "live_affine_relu_total_time_limit": float(
                _LIVE_AFFINE_TOTAL_SECONDS
            ),
            "live_affine_relu_max_stored_nnz": int(
                _LIVE_AFFINE_MAX_STORED_NNZ
            ),
            "live_affine_relu_attempts": self.live_affine_fusion_attempts,
            "live_affine_relu_applied": int(
                sum(
                    item.get("status") == "applied"
                    for item in self.live_affine_fusion_attempts
                )
            ),
            "live_affine_relu_box_inactive_rows": int(
                sum(
                    int(item.get("box_inactive_rows", 0))
                    for item in self.live_affine_fusion_attempts
                    if item.get("status") == "applied"
                )
            ),
            "live_affine_relu_elapsed_seconds": float(
                sum(
                    float(item.get("elapsed_seconds", 0.0))
                    for item in self.live_affine_fusion_attempts
                )
            ),
            "projection_skip_chain_preservations": (
                self.projection_skip_chain_preservations
            ),
            "projection_skip_chain_applied": int(
                len(self.projection_skip_chain_preservations)
            ),
            "n_layers": int(len(order)),
            "n_cont": int(hz.n_cont),
            "n_bin": int(hz.n_bin),
            "n_eq": int(hz.n_eq),
            "n_ub": int(hz.n_ub),
            "value_nnz": int(hz.value_nnz),
            "constraint_nnz": int(hz.constraint_nnz),
            "input_dim": int(self.input_col_ids.size),
            "input_normalization_outward": True,
            "base_nonempty_certificate": (
                constructive_reason
            ),
            "input_radius_nonzero": int(np.count_nonzero(self.input_radius)),
            "output_dim": int(hz.n_out),
            "output_roundoff_generator_count": int(output_error_rows.size),
            "output_roundoff_error_max": (
                float(np.max(output_error)) if output_error.size else 0.0
            ),
            "input_layer_id": int(self.input_layer_id),
            "output_layer_id": int(output_layer_id),
            "assert_layer_id": int(assert_layer.id),
            "build_seconds": float(time.monotonic() - started),
            "sparse_hz_core_assembly": (
                "owned_canonical_no_recopy_v1"
            ),
            "traversal_cache_release": traversal_cache_release,
            "constraint_tags_eq": [
                {"tag": tag, "rows": rows}
                for tag, rows in base_eq_tag_rows
            ],
            "constraint_tags_ub": [
                {"tag": tag, "rows": rows}
                for tag, rows in base_ub_tag_rows
            ]
            + (
                [
                    {
                        "tag": "property_micro_rlt_generated",
                        "rows": int(len(property_micro_rlt_upper_tags)),
                    }
                ]
                if property_micro_rlt_upper_tags
                else []
            ),
            "materialization_events": self.materialization_events,
            "layers": self.layer_metadata,
        }
        if self.verified_query_dual_feedback is not None:
            feedback_receipt = self.verified_query_dual_receipt
            if feedback_receipt is None:
                raise OperatorHZBuildError(
                    "verified query-dual receipt snapshot is unavailable"
                )
            metadata["soundness"] = (
                f"{metadata['soundness']};"
                "verified_query_dual_bounds_and_property_constants_require_"
                "process_local_transaction_validation+independent_cpu_replay"
            )
            metadata["verified_query_dual_feedback"] = {
                "schema": "operator_hz_verified_query_dual_feedback_v1",
                "proof_authority": True,
                "target_relu_ids": [
                    int(value)
                    for value in self.verified_query_dual_target_ids
                ],
                "root_boxes_sha256": feedback_receipt[
                    "root_boxes_sha256"
                ],
                "final_boxes_sha256": feedback_receipt[
                    "final_boxes_sha256"
                ],
                "property_spec_sha256": feedback_receipt[
                    "property_spec_sha256"
                ],
                "property_upper_sha256": feedback_receipt[
                    "property_upper_sha256"
                ],
                "transaction_receipt_sha256": feedback_receipt[
                    "receipt_sha256"
                ],
                "process_local_validation": True,
                "live_full_validation_passes": 2,
                "validation_and_snapshot_seconds": float(
                    self.verified_query_dual_consume_seconds
                ),
                "receipt_rehydration_authority": False,
            }
        verified_preactivation_frame = None
        metadata["verified_preactivation_frame_export_requested"] = bool(
            self.export_verified_preactivation_frame
        )
        metadata["verified_preactivation_frame_exported"] = False
        if (
            self.residual_bound_screen
            and self.verified_query_dual_feedback is None
            and self.export_verified_preactivation_frame
        ):
            self._check_deadline("preactivation_frame_export")
            verified_preactivation_frame = (
                _make_operator_hz_preactivation_frame(
                    net=self.net,
                    bounds=self.verified_preactivation_bounds,
                    residual_rows_tightened=int(
                        metadata[
                            "residual_bound_screen_rows_tightened"
                        ]
                    ),
                )
            )
            metadata["verified_preactivation_frame"] = {
                "schema": _PREACTIVATION_FRAME_SCHEMA,
                "proof_authority": True,
                "network_sha256": (
                    verified_preactivation_frame.receipt[
                        "network_sha256"
                    ]
                ),
                "bounds_sha256": (
                    verified_preactivation_frame.receipt[
                        "bounds_sha256"
                    ]
                ),
                "receipt_sha256": (
                    verified_preactivation_frame.receipt[
                        "receipt_sha256"
                    ]
                ),
                "relu_layer_count": int(
                    len(verified_preactivation_frame.bounds)
                ),
                "process_local_validation_required": True,
            }
            metadata["verified_preactivation_frame_exported"] = True
        hz.operator_hz_metadata = metadata
        constructive_nonempty_seal = None
        seal_started = time.monotonic()
        if self.issue_constructive_nonempty_seal:
            if not hz_constructively_nonempty(hz):
                raise OperatorHZBuildError(
                    "constructive-nonempty theorem token is absent"
                )
            self._check_deadline(
                "constructive_nonempty_seal_digest_before"
            )
            from act.back_end.hybridz_tf.adaptive_phase_forest import (
                sparse_hz_semantic_digest,
            )

            parent_semantic_digest = sparse_hz_semantic_digest(hz)
            self._check_deadline(
                "constructive_nonempty_seal_digest_after"
            )
            constructive_nonempty_seal = (
                _make_operator_hz_constructive_nonempty_seal(
                    semantic_digest=parent_semantic_digest,
                    reason=constructive_reason,
                )
            )
        performance_diagnostic["stages"][
            "constructive_seal_wall_seconds"
        ] = float(time.monotonic() - seal_started)
        usage_finished = resource.getrusage(resource.RUSAGE_SELF)
        finished = time.monotonic()
        performance_diagnostic.update({
            "total_wall_seconds": float(finished - started),
            "total_process_cpu_seconds": float(
                time.process_time() - process_cpu_started
            ),
            "minor_faults_delta": int(
                usage_finished.ru_minflt - usage_started.ru_minflt
            ),
            "major_faults_delta": int(
                usage_finished.ru_majflt - usage_started.ru_majflt
            ),
            "voluntary_context_switches_delta": int(
                usage_finished.ru_nvcsw - usage_started.ru_nvcsw
            ),
            "involuntary_context_switches_delta": int(
                usage_finished.ru_nivcsw - usage_started.ru_nivcsw
            ),
        })
        metadata["build_seconds"] = float(finished - started)
        build = OperatorHZBuild(
            hz=hz,
            input_col_ids=self.input_col_ids.copy(),
            input_layer_id=int(self.input_layer_id),
            output_layer_id=int(output_layer_id),
            assert_layer_id=int(assert_layer.id),
            metadata=metadata,
            property_upper_output=bool(property_upper_output),
            property_upper_row_groups=(
                self.property_tail_row_groups
                if property_upper_output
                else ()
            ),
            verified_preactivation_frame=verified_preactivation_frame,
            constructive_nonempty_seal=(
                constructive_nonempty_seal
            ),
            performance_diagnostic=copy.deepcopy(
                performance_diagnostic
            ),
            constraint_program=constraint_program,
        )
        if constructive_nonempty_seal is not None:
            _register_operator_hz_constructive_nonempty_seal(
                constructive_nonempty_seal,
                build,
            )
        return build


def build_operator_hz(
    net: Net,
    before: Mapping[int, Fact],
    after: Mapping[int, Fact],
    *,
    exact_budget: int = 0,
    materialize_add: bool = True,
    preactivation_lp_budget: int = 0,
    preactivation_lp_time_limit: float = 0.0,
    preactivation_targets: Optional[Any] = None,
    correlation_targets: Optional[Any] = None,
    residual_phase_screen: bool = False,
    residual_bound_screen: bool = False,
    residual_targets: Optional[Any] = None,
    exact_target_reservoir: Optional[Any] = None,
    export_verified_preactivation_frame: bool = True,
    property_phase_focus_rivals: Optional[Any] = None,
    property_micro_rlt_product_cap: int = 0,
    property_micro_rlt_packet_mode: str = "both",
    property_upper_C: Optional[Any] = None,
    property_upper_thresholds: Optional[Any] = None,
    property_tail_add_source_planes: bool = False,
    property_tail_alpha_steps: int = 0,
    property_tail_alpha_time_limit: float = 0.0,
    property_tail_alpha_learning_rate: float = 0.08,
    property_tail_alpha_max_cells: int = 50_000_000,
    property_tail_alpha_device: str = "auto",
    property_tail_pairhull_budget: int = 0,
    property_tail_pairhull_time_limit: float = 0.0,
    property_tail_suffix_blocks: int = 0,
    property_tail_suffix_alpha_steps: int = 0,
    property_tail_suffix_alpha_time_limit: float = 0.0,
    property_tail_suffix_alpha_device: str = "auto",
    verified_query_dual_feedback: Optional[Any] = None,
    issue_constructive_nonempty_seal: bool = False,
    deadline: Optional[float] = None,
) -> OperatorHZBuild:
    """Build a strict, local-constraint HybridZ representation.

    Args:
        net: A single-lane ACT DAG with embedded ``INPUT_SPEC`` and ``ASSERT``.
        before: Bounds before each layer, retained for API symmetry and future
            audit extensions.
        after: Bounds after each layer.  They are checked and reported but are
            not trusted as ReLU big-M constants.
        exact_budget: ``0`` selects triangle relaxation for every unstable
            ReLU, ``-1`` selects the exact binary graph for every unstable
            ReLU, and a positive value selects the first K unstable neurons in
            deterministic topological/layer-row order.
        materialize_add: Introduce a local normalized frame at every ADD.
            Disabling this is sound but may create wider expressions downstream.
        preactivation_lp_budget: Maximum number of cube-unstable ReLU rows
            for which HiGHS may propose constrained lower/upper bounds over
            the already-built HZ frame.  ``0`` disables this optional Phase-1
            mechanism.
        preactivation_lp_time_limit: Shared wall-clock allowance, in seconds,
            for all preactivation candidate LPs in this build.  A bound is
            consumed only after independent long-double Lagrangian checking.
        preactivation_targets: Optional ordered ``(layer_id, row)`` schedule
            or mapping.  When supplied, only listed cube-unstable rows may be
            attempted; the global budget remains a hard upper bound.
        correlation_targets: Optional explicit ``(ReLU layer_id, row)``
            schedule.  For a narrow materialized
            ``ADD -> [FLATTEN] -> affine -> RELU`` route, only these affine
            rows are recomposed over the pre-materialization ADD expression.
            The resulting outward box is intersected with the ordinary cube;
            the materialized graph and equality bands remain authoritative.
        residual_phase_screen: Recompose every ordinary cube-unstable row on
            each supported residual ADD/affine route in bounded chunks, but
            retain only rows whose outward shadow proves a stable phase.
        residual_bound_screen: Use the same bounded transient scan but retain
            every strict outward lower/upper improvement, including still
            unstable rows, so downstream triangle slopes/intercepts tighten.
        residual_targets: Optional explicit property-derived
            ``(ReLU layer_id, row, guard)`` schedule.  ``guard`` is one of
            ``none``, ``zero``, ``identity``, or ``both``.  ``None`` disables
            residual normal form; there is deliberately no generic first-K
            policy.
        exact_target_reservoir: Optional ordered same-layer backup
            ``(ReLU layer_id, row)`` coordinates.  This requires bound-screen
            mode and exactly ``exact_budget`` primary residual targets.  Only
            a primary made stable by the screen opens a same-layer vacancy;
            unused backups retain the ordinary triangle relaxation.
        export_verified_preactivation_frame: Export the optional process-local
            RBS bound capability for a later query-dual pass.  Disabling this
            auxiliary export does not alter any HZ row or bound and is useful
            for consumers that accept only a closed, capability-free source.
        property_phase_focus_rivals: Optional mapping from an exact
            ``(ReLU layer_id, row)`` target to the rival rows whose
            branch-conditional suffix planes should be replayed.  Omitted
            rivals retain their ordinary sound property rows.  With a
            graph-output build and a positive micro-RLT cap, the same mapping
            is consumed only as non-authoritative focus metadata; it does not
            create or replace property rows.
        property_micro_rlt_product_cap: ``0`` disables the experimental
            factor-space lift.  A positive cap may append a degree-1 RLT only
            when exactly two explicitly property-selected exact ReLUs share
            one explicit focused rival.  Each bit uses its own three stored
            exact Big-M rows plus the other bit's lower row.  The rows are
            available only to the complete parent relaxation before phase
            enumeration; fixed-phase children make them redundant, and no
            early row-prefix tightening is claimed.
        property_micro_rlt_packet_mode: ``both`` selects both complete
            directed four-row packets.  ``first`` or ``second`` selects one
            complete packet, including every factor product and both signed
            RLT sides for each chosen row.  Selection has no proof authority.
        property_upper_C/property_upper_thresholds: Optional final property
            rows.  When both are supplied, a strict DENSE(RELU(.)) tail is
            replaced at the exported HZ output by Fraction-audited affine
            upper planes for ``C@y-threshold``.  The result may authorize SAFE
            only; it is not a graph image for falsification.
        property_tail_add_source_planes: Retain the ordinary materialized
            final-ADD rows and append one grouped alternative per rival using
            the ADD's pre-materialization affine source.  This can preserve
            cross-neuron correlation without deleting the materialized
            variables or their equality bands.
        property_tail_alpha_*: Candidate-only projected optimization of lower
            ReLU slopes for negative final-property coefficients.  It is
            disabled unless both steps and time are positive.  Every proposed
            plane is rebuilt by the Fraction endpoint oracle and selected only
            when its outward prefix-cube upper bound does not regress.
        property_tail_pairhull_*: Safe-only, exact-audited two-ReLU property
            planes over a bounded global pool of correlated preactivation
            pairs.  The float selector has no proof authority; every retained
            full row is reconstructed from Fraction supports and keeps its
            baseline/alpha row as a grouped fallback.
        property_tail_suffix_blocks: ``0`` disables truncated suffix replay.
            A positive value selects that many full residual blocks before
            the nearest dominating ADD, replays every property lower predicate
            only to that ADD, and composes the negated certified plane with
            the shared Operator-HZ prefix.  Ordinary tail rows remain grouped
            fallbacks for every rival.
        property_tail_suffix_alpha_*: Candidate-only per-property
            DualSolver optimization of every suffix ReLU lower slope.  The
            optimizer has no proof authority; its frozen alpha is accepted
            only after independent affine suffix replay, and the alpha-zero
            and alpha-one planes remain available to the row selector.
        verified_query_dual_feedback: Optional process-local transaction
            produced by the independent query-dual pipeline.  A full live
            capability validator must bind it to this exact network and
            property before any ReLU bound or property constant is consumed;
            serialized/self-hashed receipts have no authority.
        issue_constructive_nonempty_seal: Issue an owner-bound, process-local
            seal over the completed HZ semantic digest.  The default ``False``
            performs no digest or registry work.
        deadline: Optional absolute ``time.monotonic()`` deadline shared with
            analysis and solving.

    Returns:
        :class:`OperatorHZBuild`, containing the sparse HZ, stable input ids,
        and a detailed per-layer construction receipt.

    Raises:
        OperatorHZBuildError: On unsupported operators/topology, non-finite
            parameters or bounds, dimension mismatches, or malformed facts.
    """

    # Retain this explicit check so callers cannot accidentally pass ``None``
    # while believing interval facts were audited.
    if before is None or after is None:
        raise OperatorHZBuildError("before and after fact mappings are required")
    if type(issue_constructive_nonempty_seal) is not bool:
        raise OperatorHZBuildError(
            "issue_constructive_nonempty_seal must be a bool"
        )
    builder = _OperatorHZBuilder(
        net,
        before,
        after,
        exact_budget=int(exact_budget),
        materialize_add=bool(materialize_add),
        preactivation_lp_budget=int(preactivation_lp_budget),
        preactivation_lp_time_limit=float(preactivation_lp_time_limit),
        preactivation_targets=preactivation_targets,
        correlation_targets=correlation_targets,
        residual_phase_screen=bool(residual_phase_screen),
        residual_bound_screen=bool(residual_bound_screen),
        residual_targets=residual_targets,
        exact_target_reservoir=exact_target_reservoir,
        export_verified_preactivation_frame=(
            export_verified_preactivation_frame
        ),
        property_phase_focus_rivals=property_phase_focus_rivals,
        property_micro_rlt_product_cap=(
            property_micro_rlt_product_cap
        ),
        property_micro_rlt_packet_mode=(
            property_micro_rlt_packet_mode
        ),
        property_upper_C=property_upper_C,
        property_upper_thresholds=property_upper_thresholds,
        property_tail_add_source_planes=bool(
            property_tail_add_source_planes
        ),
        property_tail_alpha_steps=int(property_tail_alpha_steps),
        property_tail_alpha_time_limit=float(
            property_tail_alpha_time_limit
        ),
        property_tail_alpha_learning_rate=float(
            property_tail_alpha_learning_rate
        ),
        property_tail_alpha_max_cells=int(
            property_tail_alpha_max_cells
        ),
        property_tail_alpha_device=str(property_tail_alpha_device),
        property_tail_pairhull_budget=property_tail_pairhull_budget,
        property_tail_pairhull_time_limit=(
            property_tail_pairhull_time_limit
        ),
        property_tail_suffix_blocks=property_tail_suffix_blocks,
        property_tail_suffix_alpha_steps=(
            property_tail_suffix_alpha_steps
        ),
        property_tail_suffix_alpha_time_limit=(
            property_tail_suffix_alpha_time_limit
        ),
        property_tail_suffix_alpha_device=(
            property_tail_suffix_alpha_device
        ),
        verified_query_dual_feedback=verified_query_dual_feedback,
        issue_constructive_nonempty_seal=(
            issue_constructive_nonempty_seal
        ),
        deadline=deadline,
    )
    try:
        return builder.build()
    except BaseException as build_error:
        try:
            cleanup_error = builder._discard_open_constraint_program_sink()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            try:
                build_error.add_note(
                    "constraint-program pre-seal cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            except BaseException:
                pass
        raise


def operator_hz_self_test() -> Dict[str, Any]:
    """Run a controlled residual ``|x|`` audit without using propagation code.

    The toy graph has two sibling branches ``ReLU(x)`` and ``ReLU(-x)`` joined
    by ADD.  Thus it exercises stable generator identity, branch correlation,
    local ADD equality, relaxed ReLU triangles, and exact binary ReLU graphs.
    The independent MILP verdict checks that ``max |x| = 1`` on ``[-1, 1]``.
    """

    from fractions import Fraction

    from act.back_end.core import ConSet
    from act.back_end.solver.solver_hz import (
        hz_base_feasibility,
        hz_objbound_decide,
    )

    # Endpoint enclosure audit, including asymmetric decimal bounds for which
    # independently rounded midpoint/radius formulas are easy to get wrong.
    probe_lb = np.asarray(
        [-1.0, 0.1, -1.0e100, np.nextafter(2.0, -np.inf), 7.0],
        dtype=np.float64,
    )
    probe_ub = np.asarray(
        [1.0, 0.3, 3.0e100, np.nextafter(2.0, np.inf), 7.0],
        dtype=np.float64,
    )
    probe_c, probe_r = _enclosing_center_radius(
        probe_lb, probe_ub, name="self-test probe"
    )
    ld = np.longdouble
    if np.any(probe_c.astype(ld) - probe_r.astype(ld) > probe_lb.astype(ld)):
        raise AssertionError("normalized box shrank a lower endpoint")
    if np.any(probe_c.astype(ld) + probe_r.astype(ld) < probe_ub.astype(ld)):
        raise AssertionError("normalized box shrank an upper endpoint")
    if probe_c[-1] != 7.0 or probe_r[-1] != 0.0:
        raise AssertionError("point normalization did not stay exact")

    # Independent phase enumeration for the stored xi_b in {-1,+1} formula.
    # This does not call the builder's constraint helpers.
    phase_l, phase_u = -2.0, 3.0

    def phase_rows_hold(x: float, y: float, xi_b: float) -> bool:
        return bool(
            x - y <= 1e-12
            and y - x - 0.5 * phase_l * xi_b <= -0.5 * phase_l + 1e-12
            and y - 0.5 * phase_u * xi_b <= 0.5 * phase_u + 1e-12
            and -1e-12 <= y <= phase_u + 1e-12
        )

    for x_value in np.linspace(phase_l, phase_u, 21):
        y_value = max(0.0, float(x_value))
        good_phase = -1.0 if x_value < 0.0 else 1.0
        if not phase_rows_hold(float(x_value), y_value, good_phase):
            raise AssertionError(
                f"exact Big-M rejected graph point x={x_value}, y={y_value}"
            )
    if phase_rows_hold(1.0, 1.0, -1.0):
        raise AssertionError("zero phase accepted a positive ReLU input")
    if phase_rows_hold(-1.0, 0.0, 1.0):
        raise AssertionError("active phase accepted a negative ReLU input")
    if phase_rows_hold(1.0, 0.5, 1.0):
        raise AssertionError("active phase accepted y != x")

    triangle_l = np.asarray([-0.1], dtype=np.float64)
    triangle_u = np.asarray([0.3], dtype=np.float64)
    triangle_s, triangle_b, _ = _relu_triangle_parameters(
        triangle_l, triangle_u
    )
    sf = Fraction.from_float(float(triangle_s[0]))
    bf = Fraction.from_float(float(triangle_b[0]))
    for endpoint in (
        Fraction.from_float(float(triangle_l[0])),
        Fraction(0),
        Fraction.from_float(float(triangle_u[0])),
    ):
        relu_endpoint = max(Fraction(0), endpoint)
        if relu_endpoint > sf * endpoint + bf:
            raise AssertionError(
                "Fraction-audited triangle excluded an endpoint"
            )

    def layer(lid: int, kind: str, params: Dict[str, Any], out_vars: List[int]):
        return SimpleNamespace(
            id=lid,
            kind=kind,
            params=params,
            out_vars=out_vars,
            in_vars=[],
        )

    conv_contract = layer(
        90,
        "CONV2D",
        {
            "weight": torch.ones((2, 1, 3, 3), dtype=torch.float64),
            "bias": torch.zeros(2, dtype=torch.float64),
            "input_shape": (1, 1, 5, 5),
            "output_shape": (1, 2, 3, 3),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "data_format": "NCHW",
            "padding_mode": "zeros",
        },
        list(range(18)),
    )
    _validate_strict_conv2d_layer(conv_contract)
    conv_contract.params["output_shape"] = (1, 2, 4, 4)
    try:
        _validate_strict_conv2d_layer(conv_contract)
    except OperatorHZBuildError:
        pass
    else:
        raise AssertionError("invalid CONV2D output geometry was accepted")

    one = torch.tensor([[-1.0]], dtype=torch.float64)
    pos_one = torch.tensor([[1.0]], dtype=torch.float64)
    zero = torch.tensor([[0.0]], dtype=torch.float64)
    two = torch.tensor([[2.0]], dtype=torch.float64)
    Wp = torch.tensor([[1.0]], dtype=torch.float64)
    Wn = torch.tensor([[-1.0]], dtype=torch.float64)
    bias = torch.zeros(1, dtype=torch.float64)

    layers = [
        layer(0, "INPUT", {"shape": (1, 1)}, [0]),
        layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": one, "ub": pos_one}, [0]),
        layer(
            2,
            "DENSE",
            {"weight": Wp, "bias": bias, "in_features": 1, "out_features": 1},
            [1],
        ),
        layer(3, "RELU", {}, [2]),
        layer(
            4,
            "DENSE",
            {"weight": Wn, "bias": bias, "in_features": 1, "out_features": 1},
            [3],
        ),
        layer(5, "RELU", {}, [4]),
        layer(6, "ADD", {}, [5]),
        layer(7, "FLATTEN", {}, [5]),
        layer(8, "ASSERT", {"kind": "UNSAFE_LINEAR"}, [5]),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [1],
        5: [4],
        6: [3, 5],
        7: [6],
        8: [7],
    }
    succs: Dict[int, List[int]] = {lid: [] for lid in preds}
    for lid, ps in preds.items():
        for pid in ps:
            succs[pid].append(lid)
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={item.id: item for item in layers},
    )

    ranges = {
        0: (one, pos_one),
        1: (one, pos_one),
        2: (one, pos_one),
        3: (zero, pos_one),
        4: (one, pos_one),
        5: (zero, pos_one),
        6: (zero, two),
        7: (zero, two),
        8: (zero, two),
    }
    after = {
        lid: Fact(Bounds(lb.clone(), ub.clone()), ConSet())
        for lid, (lb, ub) in ranges.items()
    }
    before = dict(after)

    try:
        build_operator_hz(
            net,
            before,
            after,
            exact_budget=0,
            deadline=time.monotonic() - 1.0,
        )
    except OperatorHZBuildTimeout:
        pass
    else:
        raise AssertionError("expired shared operator deadline was ignored")

    relaxed = build_operator_hz(net, before, after, exact_budget=0)
    exact = build_operator_hz(net, before, after, exact_budget=-1)
    if relaxed.hz.n_bin != 0:
        raise AssertionError(f"relaxed toy allocated {relaxed.hz.n_bin} binaries")
    if exact.hz.n_bin != 2:
        raise AssertionError(f"exact toy expected 2 binaries, got {exact.hz.n_bin}")
    if exact.input_col_ids.size != 1 or exact.hz.n_out != 1:
        raise AssertionError("toy provenance/output shape mismatch")
    base_status, base_reason = hz_base_feasibility(exact.hz, time_limit=0.01)
    if (
        base_status != "FEASIBLE"
        or not str(base_reason).startswith("constructive:")
    ):
        raise AssertionError(
            f"operator construction theorem was not recognized: "
            f"{base_status}/{base_reason}"
        )

    # Degenerate-box point consistency: ReLU(0) belongs to exactly one stable
    # phase, creates neither a continuous nor a binary factor, and remains 0.
    layers[1].params = {"kind": "BOX", "lb": zero, "ub": zero}
    point_after = {
        lid: Fact(Bounds(zero.clone(), zero.clone()), ConSet())
        for lid in ranges
    }
    point = build_operator_hz(
        net, dict(point_after), point_after, exact_budget=-1
    )
    if (
        point.hz.n_cont != 0
        or point.hz.n_bin != 0
        or point.hz.n_eq != 0
        or point.hz.n_ub != 0
        or not np.array_equal(point.hz.c, np.zeros(1, dtype=np.float64))
    ):
        raise AssertionError(
            "point-consistency audit failed: "
            f"shape=({point.hz.n_cont},{point.hz.n_bin},"
            f"{point.hz.n_eq},{point.hz.n_ub}), c={point.hz.c}"
        )
    layers[1].params = {"kind": "BOX", "lb": one, "ub": pos_one}

    def _exact_output_cube_contains(
        built: OperatorHZBuild,
        exact_value: Fraction,
        *,
        label: str,
    ) -> None:
        if built.hz.n_out != 1:
            raise AssertionError(f"{label} expected one output")
        center_exact = Fraction.from_float(float(built.hz.c[0]))
        radius_exact = Fraction(0)
        row = built.hz.Gc.getrow(0)
        for value in row.data:
            radius_exact += abs(Fraction.from_float(float(value)))
        brow = built.hz.Gb.getrow(0)
        for value in brow.data:
            radius_exact += abs(Fraction.from_float(float(value)))
        if not (
            center_exact - radius_exact
            <= exact_value
            <= center_exact + radius_exact
        ):
            raise AssertionError(
                f"{label} exact-real value {exact_value} escaped output cube "
                f"[{center_exact-radius_exact}, {center_exact+radius_exact}]"
            )

    def _point_graph(
        graph_layers: List[Any],
        graph_preds: Dict[int, List[int]],
        sizes: Dict[int, int],
        *,
        input_value: torch.Tensor,
    ) -> OperatorHZBuild:
        graph_succs: Dict[int, List[int]] = {
            int(item.id): [] for item in graph_layers
        }
        for child, parents in graph_preds.items():
            for parent in parents:
                graph_succs[parent].append(child)
        graph_net = SimpleNamespace(
            layers=graph_layers,
            preds=graph_preds,
            succs=graph_succs,
            by_id={item.id: item for item in graph_layers},
        )
        graph_after: Dict[int, Fact] = {}
        for item in graph_layers:
            size = int(sizes[int(item.id)])
            if _kind(item.kind) in {"INPUT", "INPUT_SPEC"}:
                lo = input_value.reshape(1, -1).clone()
                hi = lo.clone()
            else:
                # Facts are audit-only in this builder.  A wide finite box
                # avoids accidentally supplying the exact oracle as a proof.
                lo = torch.full((1, size), -1.0e100, dtype=torch.float64)
                hi = torch.full((1, size), 1.0e100, dtype=torch.float64)
            graph_after[int(item.id)] = Fact(Bounds(lo, hi), ConSet())
        return build_operator_hz(
            graph_net,
            dict(graph_after),
            graph_after,
            exact_budget=0,
            materialize_add=True,
        )

    # Dense cancellation used to collapse exact-real 1e16+1-1e16=1 to the
    # rounded point 0 and directly produce a false SAFE.  The final numerical
    # generator must contain the Fraction oracle.
    cancel_input = torch.ones((1, 3), dtype=torch.float64)
    cancel_layers = [
        layer(20, "INPUT", {"shape": (1, 3)}, [0, 1, 2]),
        layer(
            21,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": cancel_input, "ub": cancel_input},
            [0, 1, 2],
        ),
        layer(
            22,
            "DENSE",
            {
                "weight": torch.tensor(
                    [[1.0e16, 1.0, -1.0e16]], dtype=torch.float64
                ),
                "bias": torch.zeros(1, dtype=torch.float64),
                "in_features": 3,
                "out_features": 1,
            },
            [3],
        ),
        layer(23, "ASSERT", {"kind": "UNSAFE_LINEAR"}, [3]),
    ]
    cancel = _point_graph(
        cancel_layers,
        {20: [], 21: [20], 22: [21], 23: [22]},
        {20: 3, 21: 3, 22: 1, 23: 1},
        input_value=cancel_input,
    )
    cancel_exact = (
        Fraction.from_float(1.0e16)
        + Fraction.from_float(1.0)
        - Fraction.from_float(1.0e16)
    )
    _exact_output_cube_contains(
        cancel, cancel_exact, label="dense cancellation"
    )

    # ADD cancellation is independent of sparse dot-product accumulation.
    zero_input = torch.zeros((1, 1), dtype=torch.float64)
    add_layers = [
        layer(30, "INPUT", {"shape": (1, 1)}, [0]),
        layer(
            31,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": zero_input, "ub": zero_input},
            [0],
        ),
        layer(
            32,
            "DENSE",
            {
                "weight": torch.zeros((1, 1), dtype=torch.float64),
                "bias": torch.tensor([1.0e16], dtype=torch.float64),
                "in_features": 1,
                "out_features": 1,
            },
            [1],
        ),
        layer(
            33,
            "DENSE",
            {
                "weight": torch.zeros((1, 1), dtype=torch.float64),
                "bias": torch.tensor([1.0], dtype=torch.float64),
                "in_features": 1,
                "out_features": 1,
            },
            [2],
        ),
        layer(34, "ADD", {}, [3]),
        layer(
            35,
            "DENSE",
            {
                "weight": torch.ones((1, 1), dtype=torch.float64),
                "bias": torch.tensor([-1.0e16], dtype=torch.float64),
                "in_features": 1,
                "out_features": 1,
            },
            [4],
        ),
        layer(36, "ASSERT", {"kind": "UNSAFE_LINEAR"}, [4]),
    ]
    add_cancel = _point_graph(
        add_layers,
        {
            30: [],
            31: [30],
            32: [31],
            33: [31],
            34: [32, 33],
            35: [34],
            36: [35],
        },
        {lid: 1 for lid in range(30, 37)},
        input_value=zero_input,
    )
    add_exact = (
        Fraction.from_float(1.0e16)
        + Fraction.from_float(1.0)
        - Fraction.from_float(1.0e16)
    )
    _exact_output_cube_contains(
        add_cancel, add_exact, label="ADD cancellation"
    )

    C = torch.tensor([[1.0]], dtype=torch.float64)
    safe, _ = hz_objbound_decide(
        exact.hz,
        C,
        torch.tensor([1.000001], dtype=torch.float64),
        is_unsafe_linear=False,
        time_limit=5.0,
    )
    unsafe, witness = hz_objbound_decide(
        exact.hz,
        C,
        torch.tensor([0.75], dtype=torch.float64),
        is_unsafe_linear=False,
        time_limit=5.0,
    )
    # The strict solver currently quarantines unvalidated infeasibility
    # reports as UNKNOWN.  At the builder gate we require only that it never
    # reverses either exact oracle result; witness completeness is measured by
    # the later certified-solver gates.
    if safe == "UNSAFE" or unsafe == "SAFE":
        raise AssertionError(
            f"toy MILP audit failed: above-one={safe}, above-0.75={unsafe}"
        )
    return {
        "ok": True,
        "relaxed": {
            "n_cont": relaxed.hz.n_cont,
            "n_bin": relaxed.hz.n_bin,
            "n_eq": relaxed.hz.n_eq,
            "n_ub": relaxed.hz.n_ub,
        },
        "exact": {
            "n_cont": exact.hz.n_cont,
            "n_bin": exact.hz.n_bin,
            "n_eq": exact.hz.n_eq,
            "n_ub": exact.hz.n_ub,
        },
        "milp": {"above_one": safe, "above_0_75": unsafe},
        "audits": {
            "normalized_box_encloses_endpoints": True,
            "point_normalization_exact": True,
            "exact_big_m_phase_enumeration": True,
            "relu_zero_point_consistency": True,
            "dense_cancellation_fraction_enclosed": True,
            "add_cancellation_fraction_enclosed": True,
            "triangle_fraction_endpoints": True,
            "constructive_base_nonempty_certificate": True,
            "strict_conv2d_contract": True,
            "shared_deadline_enforced": True,
        },
    }


if __name__ == "__main__":
    print(operator_hz_self_test())
