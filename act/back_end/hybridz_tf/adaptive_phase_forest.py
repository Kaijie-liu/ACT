"""Candidate-only adaptive exact-binary phase forest for Operator-HZ.

This module deliberately has no verdict authority and is not connected to
``verify_once`` or BaB dispatch.  It tests a narrower scheduling idea:

* consume an already-UNKNOWN root bound;
* split exactly one remaining HybridZ binary factor at each UNKNOWN node;
* validate the two fixed-phase children against the live parent;
* bound all children in one node-by-rival wave;
* prune SAFE children and recurse only on UNKNOWN children; and
* require explicit node conservation before reporting a successful candidate.

The existing verifier enumerates every leaf of a preselected depth-one or
depth-two phase cover.  This forest is different: a SAFE sibling is terminal,
so an asymmetric tree can use far fewer leaves than a blind ``2**depth``
cover.  The result remains diagnostic until a separate production authority
validates solver capabilities, deadlines, and terminal conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_enumerate_sparse_binary_phase_cover,
    hz_verify_sparse_binary_phase_child,
)


Lineage = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class RivalSpec:
    """One stable original ASSERT rival.

    ``assert_digest`` is supplied by the independent raw-ASSERT encoder.
    The adapter additionally hashes the stable ID, exact binary64 objective
    row, and threshold, so an objective/threshold swap changes the binding.
    """

    rival_id: int
    objective: Tuple[float, ...]
    threshold: float
    assert_digest: str

    @property
    def binding_digest(self) -> str:
        return rival_spec_binding_digest(self)


@dataclass(frozen=True)
class RivalUpperBound:
    """One numeric upper bound explicitly bound to its stable rival.

    The value and identity travel as one object.  In particular, a callback
    cannot reverse a naked vector of numbers while separately copying a
    correct batch-level binding receipt.
    """

    rival_id: int
    binding_digest: str
    upper: float


@dataclass(frozen=True)
class PhaseForestNode:
    """One live candidate node, addressed by stable binary-column IDs."""

    node_id: int
    depth: int
    lineage: Lineage
    hz: SparseHZono


@dataclass(frozen=True)
class PhaseNodeBound:
    """Safe-only result returned by a node-wave bound callback.

    ``rival_bounds`` contains one explicitly identified sound upper bound per
    original property rival.  ``binary_scores`` is optional candidate
    telemetry keyed by stable ``bcol_id``; it never carries proof authority.
    """

    node_id: int
    lineage: Lineage
    remaining_bcol_ids: Tuple[int, ...]
    rival_bounds: Tuple[RivalUpperBound, ...]
    property_digest: str
    node_semantic_digest: str
    verdict: str
    binary_scores: Tuple[Tuple[int, float], ...] = ()
    deadline_respected: bool = True
    error: Optional[str] = None
    proof_authority: bool = False


@dataclass(frozen=True)
class PhaseBoundWaveRequest:
    """One breadth-first wave sharing the complete original rival batch."""

    wave_index: int
    nodes: Tuple[PhaseForestNode, ...]
    rivals: Tuple[RivalSpec, ...]
    property_digest: str
    deadline: float
    proof_authority: bool = False

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def rival_count(self) -> int:
        return len(self.rivals)

    @property
    def thresholds(self) -> Tuple[float, ...]:
        return tuple(float(rival.threshold) for rival in self.rivals)

    @property
    def rival_bindings(self) -> Tuple[Tuple[int, str], ...]:
        return _ordered_rival_bindings(self.rivals)


@dataclass(frozen=True)
class AdaptivePhaseForestResult:
    """Non-authoritative candidate result and omission-firewall receipt."""

    status: str
    reason: str
    all_leaves_safe: bool
    proof_authority: bool
    receipt: Mapping[str, Any]


BoundWave = Callable[
    [PhaseBoundWaveRequest],
    Sequence[PhaseNodeBound],
]
SelectBinary = Callable[[PhaseForestNode, PhaseNodeBound], int]


class _CandidateFailure(RuntimeError):
    pass


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise _CandidateFailure(f"{name}_not_integer")
    return int(value)


def _normalize_rivals(
    rivals: Sequence[RivalSpec],
) -> Tuple[RivalSpec, ...]:
    if isinstance(rivals, (str, bytes)) or not isinstance(
        rivals, Sequence
    ):
        raise _CandidateFailure("rivals_not_sequence")
    normalized = []
    seen_ids = set()
    objective_width = None
    for raw in rivals:
        if not isinstance(raw, RivalSpec):
            raise _CandidateFailure("rival_wrong_type")
        rival_id = _strict_int(raw.rival_id, name="rival_id")
        if rival_id < 0 or rival_id in seen_ids:
            raise _CandidateFailure("rival_id_invalid_or_duplicate")
        objective = np.asarray(raw.objective, dtype=np.float64)
        if (
            objective.ndim != 1
            or objective.size < 1
            or not np.all(np.isfinite(objective))
        ):
            raise _CandidateFailure("rival_objective_malformed")
        if objective_width is None:
            objective_width = int(objective.size)
        elif int(objective.size) != objective_width:
            raise _CandidateFailure("rival_objective_width_mismatch")
        try:
            threshold = float(raw.threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _CandidateFailure("rival_threshold_not_numeric") from exc
        if not math.isfinite(threshold):
            raise _CandidateFailure("rival_threshold_nonfinite")
        assert_digest = raw.assert_digest
        if (
            not isinstance(assert_digest, str)
            or len(assert_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in assert_digest
            )
        ):
            raise _CandidateFailure("rival_assert_digest_malformed")
        normalized.append(
            RivalSpec(
                rival_id=rival_id,
                objective=tuple(
                    float(value) for value in objective.tolist()
                ),
                threshold=threshold,
                assert_digest=assert_digest,
            )
        )
        seen_ids.add(rival_id)
    if not normalized:
        raise _CandidateFailure("rivals_empty")
    return tuple(normalized)


def _hash_framed_bytes(
    digest: "hashlib._Hash",
    label: bytes,
    payload: bytes,
) -> None:
    digest.update(len(label).to_bytes(8, "little"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


_SEMANTIC_HASH_CHUNK_ITEMS = 1 << 18


def _hash_framed_numeric_array(
    digest: "hashlib._Hash",
    *,
    label: bytes,
    value: Any,
    canonical_dtype: np.dtype[Any] | type[Any],
) -> None:
    """Stream the legacy canonical array bytes without a full-size copy.

    The framing and native-endian canonical dtypes are deliberately identical
    to ``_hash_framed_bytes(..., np.asarray(..., dtype=...).tobytes())``.
    Integer CSR arrays therefore remain bound as int64 even when SciPy stores
    them as int32, but at most one small conversion chunk is live at a time.
    """

    array = np.asarray(value)
    dtype = np.dtype(canonical_dtype)
    if array.ndim != 1:
        raise _CandidateFailure("semantic_stream_array_malformed")
    payload_size = int(array.size) * int(dtype.itemsize)
    digest.update(len(label).to_bytes(8, "little"))
    digest.update(label)
    digest.update(payload_size.to_bytes(8, "little"))
    for start in range(0, int(array.size), _SEMANTIC_HASH_CHUNK_ITEMS):
        stop = min(
            int(array.size), start + _SEMANTIC_HASH_CHUNK_ITEMS
        )
        chunk = np.ascontiguousarray(array[start:stop], dtype=dtype)
        digest.update(memoryview(chunk).cast("B"))


def rival_spec_binding_digest(rival: RivalSpec) -> str:
    """Hash one stable rival ID, objective, threshold, and raw ASSERT."""

    normalized = _normalize_rivals((rival,))[0]
    digest = hashlib.sha256()
    digest.update(b"hybridz_adaptive_phase_rival_spec_v1")
    _hash_framed_bytes(
        digest,
        b"rival_id",
        str(normalized.rival_id).encode("ascii"),
    )
    objective = np.ascontiguousarray(
        normalized.objective, dtype=np.float64
    )
    _hash_framed_bytes(
        digest,
        b"objective_shape",
        np.asarray(objective.shape, dtype=np.int64).tobytes(),
    )
    _hash_framed_bytes(
        digest,
        b"objective_f64",
        objective.tobytes(order="C"),
    )
    _hash_framed_bytes(
        digest,
        b"threshold_f64",
        np.asarray(
            [normalized.threshold], dtype=np.float64
        ).tobytes(),
    )
    _hash_framed_bytes(
        digest,
        b"assert_sha256",
        normalized.assert_digest.encode("ascii"),
    )
    return digest.hexdigest()


def _ordered_rival_bindings(
    rivals: Sequence[RivalSpec],
) -> Tuple[Tuple[int, str], ...]:
    normalized = _normalize_rivals(rivals)
    return tuple(
        (rival.rival_id, rival_spec_binding_digest(rival))
        for rival in normalized
    )


def ordered_property_digest(
    rivals: Sequence[RivalSpec],
) -> str:
    """Bind the exact ordered rival batch consumed by every forest wave."""

    bindings = _ordered_rival_bindings(rivals)
    digest = hashlib.sha256()
    digest.update(b"hybridz_adaptive_phase_ordered_property_v1")
    digest.update(len(bindings).to_bytes(8, "little"))
    for rival_id, binding in bindings:
        _hash_framed_bytes(
            digest,
            b"rival_id",
            str(rival_id).encode("ascii"),
        )
        _hash_framed_bytes(
            digest,
            b"rival_binding",
            binding.encode("ascii"),
        )
    return digest.hexdigest()


def _semantic_hash_value(
    digest: "hashlib._Hash",
    value: Any,
) -> None:
    """Hash proof-relevant metadata with strict type framing."""

    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, (bool, np.bool_)):
        digest.update(b"B1" if bool(value) else b"B0")
        return
    if isinstance(value, (int, np.integer)):
        payload = str(int(value)).encode("ascii")
        _hash_framed_bytes(digest, b"I", payload)
        return
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise _CandidateFailure(
                "semantic_metadata_nonfinite_float"
            )
        _hash_framed_bytes(
            digest, b"F", scalar.hex().encode("ascii")
        )
        return
    if isinstance(value, str):
        _hash_framed_bytes(digest, b"S", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _hash_framed_bytes(digest, b"Y", value)
        return
    if isinstance(value, np.generic):
        _semantic_hash_value(digest, value.item())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise _CandidateFailure(
                "semantic_metadata_array_dtype_unsupported"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise _CandidateFailure(
                "semantic_metadata_array_nonfinite"
            )
        digest.update(b"A")
        _hash_framed_bytes(
            digest, b"dtype", array.dtype.str.encode("ascii")
        )
        _hash_framed_bytes(
            digest,
            b"shape",
            np.asarray(array.shape, dtype=np.int64).tobytes(),
        )
        _hash_framed_bytes(
            digest, b"data", array.tobytes(order="C")
        )
        return
    if sp.isspmatrix_csr(value):
        matrix = value
        if not _canonical_csr_structure_is_valid(matrix):
            raise _CandidateFailure("semantic_metadata_csr_malformed")
        digest.update(b"R")
        _hash_framed_bytes(
            digest,
            b"shape",
            np.asarray(matrix.shape, dtype=np.int64).tobytes(),
        )
        _hash_framed_bytes(
            digest,
            b"indptr",
            np.asarray(matrix.indptr, dtype=np.int64).tobytes(),
        )
        _hash_framed_bytes(
            digest,
            b"indices",
            np.asarray(matrix.indices, dtype=np.int64).tobytes(),
        )
        _hash_framed_bytes(
            digest,
            b"data",
            np.asarray(matrix.data, dtype=np.float64).tobytes(),
        )
        return
    if isinstance(value, Mapping):
        keys = list(value)
        if any(
            isinstance(key, bool) or not isinstance(key, (str, int))
            for key in keys
        ):
            raise _CandidateFailure(
                "semantic_metadata_mapping_key_unsupported"
            )
        keys.sort(
            key=lambda key: (
                0 if isinstance(key, str) else 1,
                key,
            )
        )
        digest.update(b"M" + len(keys).to_bytes(8, "little"))
        for key in keys:
            _semantic_hash_value(digest, key)
            _semantic_hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"L" + len(value).to_bytes(8, "little"))
        for item in value:
            _semantic_hash_value(digest, item)
        return
    # The private conditional seal is intentionally opaque, but exposes the
    # exact live hash it seals.  Bind both its concrete class and that hash.
    live_content = getattr(value, "live_content_sha256", None)
    if (
        isinstance(live_content, str)
        and len(live_content) == 64
    ):
        _hash_framed_bytes(
            digest,
            b"sealed_type",
            (
                f"{type(value).__module__}."
                f"{type(value).__qualname__}"
            ).encode("utf-8"),
        )
        _hash_framed_bytes(
            digest,
            b"sealed_live_content",
            live_content.encode("ascii"),
        )
        return
    raise _CandidateFailure("semantic_metadata_value_unsupported")


def _hash_dense_semantic_array(
    digest: "hashlib._Hash",
    *,
    name: str,
    value: Any,
    optional: bool = False,
) -> None:
    if value is None:
        if not optional:
            raise _CandidateFailure(f"semantic_{name}_missing")
        _hash_framed_bytes(digest, name.encode("ascii"), b"NONE")
        return
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or not np.all(np.isfinite(array))
    ):
        raise _CandidateFailure(f"semantic_{name}_malformed")
    _hash_framed_bytes(
        digest,
        f"{name}_shape".encode("ascii"),
        np.asarray(array.shape, dtype=np.int64).tobytes(),
    )
    _hash_framed_numeric_array(
        digest,
        label=name.encode("ascii"),
        value=array,
        canonical_dtype=np.float64,
    )


def _hash_csr_semantic_matrix(
    digest: "hashlib._Hash",
    *,
    name: str,
    value: Any,
    optional: bool = False,
) -> None:
    if value is None:
        if not optional:
            raise _CandidateFailure(f"semantic_{name}_missing")
        _hash_framed_bytes(digest, name.encode("ascii"), b"NONE")
        return
    if (
        not _canonical_csr_structure_is_valid(value)
        or value.dtype != np.dtype(np.float64)
    ):
        raise _CandidateFailure(f"semantic_{name}_malformed")
    matrix = value
    _hash_framed_bytes(
        digest,
        f"{name}_shape".encode("ascii"),
        np.asarray(matrix.shape, dtype=np.int64).tobytes(),
    )
    _hash_framed_numeric_array(
        digest,
        label=f"{name}_indptr".encode("ascii"),
        value=matrix.indptr,
        canonical_dtype=np.int64,
    )
    _hash_framed_numeric_array(
        digest,
        label=f"{name}_indices".encode("ascii"),
        value=matrix.indices,
        canonical_dtype=np.int64,
    )
    _hash_framed_numeric_array(
        digest,
        label=f"{name}_data".encode("ascii"),
        value=matrix.data,
        canonical_dtype=np.float64,
    )


def _canonical_csr_structure_is_valid(value: Any) -> bool:
    """Recompute CSR invariants instead of trusting SciPy's cached flags."""

    if not sp.isspmatrix_csr(value):
        return False
    try:
        rows, columns = value.shape
        indptr = np.asarray(value.indptr)
        indices = np.asarray(value.indices)
        data = np.asarray(value.data)
        if (
            type(rows) is not int
            or type(columns) is not int
            or rows < 0
            or columns < 0
            or indptr.ndim != 1
            or indices.ndim != 1
            or data.ndim != 1
            or not np.issubdtype(indptr.dtype, np.integer)
            or not np.issubdtype(indices.dtype, np.integer)
            or int(indptr.size) != rows + 1
            or int(indices.size) != int(data.size)
            or int(indptr[0]) != 0
            or int(indptr[-1]) != int(indices.size)
            or np.any(indptr < 0)
            or np.any(indptr > int(indices.size))
            or np.any(indptr[1:] < indptr[:-1])
            or (
                indices.size
                and (
                    np.any(indices < 0)
                    or np.any(indices >= columns)
                )
            )
            or (
                data.size
                and not np.all(np.isfinite(data))
            )
        ):
            return False
        # ``has_canonical_format`` and ``has_sorted_indices`` are cached by
        # SciPy and can remain true after direct mutation of ``indices``.
        # Recheck strict within-row ordering, which also excludes duplicates.
        # A single vectorized adjacent comparison is substantially cheaper
        # than one Python/NumPy dispatch per row.  Comparisons that cross a
        # CSR row boundary are masked using the independently validated
        # ``indptr``; decreasing column ids across different rows are valid.
        if indices.size > 1:
            nonincreasing = np.less_equal(indices[1:], indices[:-1])
            row_starts = indptr[1:-1]
            boundary_positions = row_starts[
                (row_starts > 0) & (row_starts < indices.size)
            ] - 1
            if boundary_positions.size:
                nonincreasing[boundary_positions] = False
            if np.any(nonincreasing):
                return False
        return True
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _hash_semantic_ids(
    digest: "hashlib._Hash",
    *,
    name: str,
    value: Any,
    expected: int,
) -> None:
    if value is None:
        raise _CandidateFailure(f"semantic_{name}_missing")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _CandidateFailure(
            f"semantic_{name}_malformed"
        ) from exc
    if (
        array.dtype != np.dtype(np.int64)
        or array.ndim != 1
        or int(array.size) != int(expected)
    ):
        raise _CandidateFailure(f"semantic_{name}_malformed")

    # Stable IDs are normally emitted in increasing order.  Audit that case
    # a chunk at a time so validation never materializes one Python integer
    # (and a set entry) per ID.  The scalar carried between chunks is needed
    # to catch a duplicate or reversal exactly at a chunk boundary.
    strictly_increasing = True
    previous_last: Optional[int] = None
    for start in range(0, int(array.size), _SEMANTIC_HASH_CHUNK_ITEMS):
        stop = min(
            int(array.size), start + _SEMANTIC_HASH_CHUNK_ITEMS
        )
        chunk = array[start:stop]
        if np.any(chunk < 0):
            raise _CandidateFailure(f"semantic_{name}_malformed")
        if (
            previous_last is not None
            and int(chunk[0]) <= previous_last
        ) or (
            chunk.size > 1
            and np.any(chunk[1:] <= chunk[:-1])
        ):
            strictly_increasing = False
            break
        if chunk.size:
            previous_last = int(chunk[-1])

    if not strictly_increasing:
        # Preserve acceptance of valid, unordered stable IDs.  Quicksort's
        # sole N-sized allocation is the int64 result (8*N bytes); duplicate
        # comparisons remain chunk bounded.  Only this validation copy is
        # sorted -- the original order below remains the hashed semantics.
        ordered = np.sort(array, kind="quicksort")
        if ordered.size and int(ordered[0]) < 0:
            raise _CandidateFailure(f"semantic_{name}_malformed")
        for start in range(
            1, int(ordered.size), _SEMANTIC_HASH_CHUNK_ITEMS
        ):
            stop = min(
                int(ordered.size),
                start + _SEMANTIC_HASH_CHUNK_ITEMS,
            )
            if np.any(
                ordered[start:stop] == ordered[start - 1 : stop - 1]
            ):
                raise _CandidateFailure(
                    f"semantic_{name}_malformed"
                )
        # Do not retain the N-sized audit copy while streaming a potentially
        # non-contiguous original array into the semantic digest.
        del ordered
    _hash_framed_bytes(
        digest,
        f"{name}_shape".encode("ascii"),
        np.asarray(array.shape, dtype=np.int64).tobytes(),
    )
    _hash_framed_numeric_array(
        digest,
        label=name.encode("ascii"),
        value=array,
        canonical_dtype=np.int64,
    )


def sparse_hz_semantic_digest(hz: SparseHZono) -> str:
    """Hash the full live SparseHZ set and all conditional proof metadata."""

    if not isinstance(hz, SparseHZono):
        raise _CandidateFailure("semantic_node_not_sparse_hz")
    digest = hashlib.sha256()
    digest.update(b"hybridz_adaptive_phase_sparse_hz_semantic_v1")
    _hash_dense_semantic_array(digest, name="c", value=hz.c)
    _hash_dense_semantic_array(digest, name="b", value=hz.b)
    _hash_dense_semantic_array(
        digest, name="ub", value=hz.ub, optional=True
    )
    for name in ("Gc", "Gb", "Ac", "Ab"):
        _hash_csr_semantic_matrix(
            digest, name=name, value=getattr(hz, name)
        )
    for name in ("Auc", "Aub"):
        _hash_csr_semantic_matrix(
            digest,
            name=name,
            value=getattr(hz, name),
            optional=True,
        )
    _hash_semantic_ids(
        digest,
        name="col_ids",
        value=hz.col_ids,
        expected=hz.n_cont,
    )
    _hash_semantic_ids(
        digest,
        name="bcol_ids",
        value=hz.bcol_ids,
        expected=hz.n_bin,
    )

    # Conditional metadata changes the semantics of future exact children.
    # Enumerate dynamically so a newly introduced conditional field fails
    # into the digest rather than being silently outside the binding.
    conditional_names = tuple(
        sorted(
            name
            for name in vars(hz)
            if "conditional" in name.lower()
        )
    )
    _semantic_hash_value(digest, conditional_names)
    for name in conditional_names:
        _semantic_hash_value(digest, name)
        _semantic_hash_value(digest, getattr(hz, name))
    return digest.hexdigest()


def _remaining_bcol_ids(hz: SparseHZono) -> Tuple[int, ...]:
    if not isinstance(hz, SparseHZono):
        raise _CandidateFailure("node_not_sparse_hz")
    if hz.bcol_ids is None:
        raise _CandidateFailure("missing_stable_bcol_ids")
    raw = np.asarray(hz.bcol_ids)
    if (
        raw.dtype != np.dtype(np.int64)
        or raw.ndim != 1
        or int(raw.size) != int(hz.n_bin)
        or (raw.size and np.any(raw < 0))
    ):
        raise _CandidateFailure("malformed_stable_bcol_ids")
    ids = tuple(int(value) for value in raw.tolist())
    if len(set(ids)) != len(ids):
        raise _CandidateFailure("duplicate_stable_bcol_ids")
    return ids


def _node_binding(node: PhaseForestNode) -> Tuple[Any, ...]:
    """Complete live binding checked around selectors and callbacks."""

    hz = node.hz
    return (
        int(node.node_id),
        int(node.depth),
        tuple(node.lineage),
        _remaining_bcol_ids(hz),
        int(hz.n_out),
        int(hz.n_cont),
        int(hz.n_bin),
        int(hz.n_eq),
        int(hz.n_ub),
        int(hz.constraint_nnz),
        sparse_hz_semantic_digest(hz),
    )


def _deadline_expired(deadline: float) -> bool:
    return time.monotonic() >= deadline


def _strict_safe_mask(
    upper: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    scale = np.maximum(
        1.0,
        np.maximum(np.abs(upper), np.abs(thresholds)),
    )
    tolerance = max(100.0 * np.finfo(np.float64).eps, 1.0e-11) * scale
    return upper < (thresholds - tolerance)


def _validate_node_bound(
    raw: object,
    node: PhaseForestNode,
    rivals: Tuple[RivalSpec, ...],
    property_digest: str,
) -> Tuple[PhaseNodeBound, bool]:
    if not isinstance(raw, PhaseNodeBound):
        raise _CandidateFailure("bound_result_wrong_type")
    if raw.proof_authority is not False:
        raise _CandidateFailure("bound_result_claimed_proof_authority")
    if raw.deadline_respected is not True:
        raise _CandidateFailure("bound_result_deadline_not_respected")
    if raw.error is not None:
        raise _CandidateFailure("bound_result_has_error")
    if _strict_int(raw.node_id, name="bound_node_id") != int(node.node_id):
        raise _CandidateFailure("bound_node_id_mismatch")
    if tuple(raw.lineage) != tuple(node.lineage):
        raise _CandidateFailure("bound_lineage_mismatch")
    remaining = _remaining_bcol_ids(node.hz)
    if tuple(raw.remaining_bcol_ids) != remaining:
        raise _CandidateFailure("bound_remaining_bcol_ids_mismatch")
    if raw.property_digest != property_digest:
        raise _CandidateFailure("bound_property_digest_mismatch")
    live_node_digest = sparse_hz_semantic_digest(node.hz)
    if raw.node_semantic_digest != live_node_digest:
        raise _CandidateFailure("bound_node_semantic_digest_mismatch")

    if not isinstance(raw.rival_bounds, tuple):
        raise _CandidateFailure("bound_rival_bounds_not_tuple")
    if len(raw.rival_bounds) != len(rivals):
        raise _CandidateFailure("bound_rival_shape_mismatch")
    upper_values = []
    for raw_rival_bound, rival in zip(raw.rival_bounds, rivals):
        if not isinstance(raw_rival_bound, RivalUpperBound):
            raise _CandidateFailure("bound_rival_bound_wrong_type")
        if (
            _strict_int(
                raw_rival_bound.rival_id,
                name="bound_rival_id",
            )
            != rival.rival_id
        ):
            raise _CandidateFailure("bound_rival_id_mismatch")
        if raw_rival_bound.binding_digest != rival.binding_digest:
            raise _CandidateFailure("bound_rival_binding_mismatch")
        if isinstance(raw_rival_bound.upper, (bool, np.bool_)):
            raise _CandidateFailure("bound_rival_upper_not_numeric")
        try:
            upper_value = float(raw_rival_bound.upper)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _CandidateFailure(
                "bound_rival_upper_not_numeric"
            ) from exc
        if not math.isfinite(upper_value):
            raise _CandidateFailure("bound_rival_nonfinite")
        upper_values.append(upper_value)

    upper = np.asarray(upper_values, dtype=np.float64)
    thresholds = np.asarray(
        [rival.threshold for rival in rivals], dtype=np.float64
    )
    if raw.verdict not in {"SAFE", "UNKNOWN"}:
        raise _CandidateFailure("bound_verdict_invalid")

    score_ids = set()
    for raw_id, raw_score in raw.binary_scores:
        score_id = _strict_int(raw_id, name="binary_score_id")
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _CandidateFailure("binary_score_not_numeric") from exc
        if (
            score_id not in remaining
            or score_id in score_ids
            or not math.isfinite(score)
        ):
            raise _CandidateFailure("binary_score_invalid")
        score_ids.add(score_id)

    independently_safe = bool(
        np.all(_strict_safe_mask(upper, thresholds))
    )
    if raw.verdict == "SAFE" and not independently_safe:
        raise _CandidateFailure("safe_verdict_without_strict_rival_bounds")
    return raw, bool(raw.verdict == "SAFE" and independently_safe)


def _default_select_binary(
    node: PhaseForestNode,
    bound: PhaseNodeBound,
) -> int:
    remaining = _remaining_bcol_ids(node.hz)
    if not remaining:
        raise _CandidateFailure("unknown_leaf_has_no_binary_factor")
    if not bound.binary_scores:
        return int(remaining[0])
    # Highest absolute property-conditioned contribution wins.  Stable IDs,
    # rather than mutable local positions, make ties and descendants
    # deterministic after earlier factors are removed.
    return int(
        min(
            bound.binary_scores,
            key=lambda item: (-abs(float(item[1])), int(item[0])),
        )[0]
    )


def _split_one_binary(
    node: PhaseForestNode,
    selected_bcol_id: int,
    *,
    next_node_id: int,
    deadline: float,
) -> Tuple[Tuple[PhaseForestNode, PhaseForestNode], int]:
    remaining = _remaining_bcol_ids(node.hz)
    selected_bcol_id = _strict_int(
        selected_bcol_id, name="selected_bcol_id"
    )
    positions = [
        position
        for position, bcol_id in enumerate(remaining)
        if bcol_id == selected_bcol_id
    ]
    if len(positions) != 1:
        raise _CandidateFailure("selected_bcol_id_not_unique_and_live")
    position = int(positions[0])
    try:
        raw_cover = hz_enumerate_sparse_binary_phase_cover(
            node.hz,
            positions=(position,),
            max_children=2,
            deadline=deadline,
        )
    except TimeoutError as exc:
        raise _CandidateFailure("split_deadline") from exc
    except Exception as exc:
        raise _CandidateFailure(
            f"split_error:{type(exc).__name__}"
        ) from exc
    if not isinstance(raw_cover, tuple) or len(raw_cover) != 2:
        raise _CandidateFailure(
            "split_cover_incomplete:"
            f"actual={len(raw_cover) if isinstance(raw_cover, tuple) else -1}"
        )

    by_sign: dict[int, SparseHZono] = {}
    assignments: dict[int, Tuple[Tuple[int, int], ...]] = {}
    for raw_item in raw_cover:
        if not isinstance(raw_item, tuple) or len(raw_item) != 2:
            raise _CandidateFailure("split_cover_item_malformed")
        assignment, child = raw_item
        if (
            not isinstance(assignment, tuple)
            or len(assignment) != 1
            or not isinstance(assignment[0], tuple)
            or len(assignment[0]) != 2
        ):
            raise _CandidateFailure("split_assignment_malformed")
        raw_position, raw_sign = assignment[0]
        if _strict_int(
            raw_position, name="split_position"
        ) != position:
            raise _CandidateFailure("split_wrong_binary_position")
        sign = _strict_int(raw_sign, name="split_sign")
        if sign not in {-1, 1}:
            raise _CandidateFailure("split_sign_not_binary")
        if sign in by_sign:
            raise _CandidateFailure(
                "split_assignments_overlap_and_complement_omitted"
            )
        if not isinstance(child, SparseHZono):
            raise _CandidateFailure("split_child_not_sparse_hz")
        if hz_verify_sparse_binary_phase_child(
            node.hz,
            assignment,
            child,
            deadline=deadline,
        ) is not True:
            raise _CandidateFailure("split_child_live_audit_failed")
        expected_remaining = tuple(
            value for value in remaining if value != selected_bcol_id
        )
        if _remaining_bcol_ids(child) != expected_remaining:
            raise _CandidateFailure("split_child_removed_wrong_stable_id")
        by_sign[sign] = child
        assignments[sign] = assignment
    if set(by_sign) != {-1, 1}:
        raise _CandidateFailure("split_complement_omitted")

    children = []
    cursor = int(next_node_id)
    for sign in (-1, 1):
        children.append(
            PhaseForestNode(
                node_id=cursor,
                depth=int(node.depth) + 1,
                lineage=(
                    *tuple(node.lineage),
                    (int(selected_bcol_id), int(sign)),
                ),
                hz=by_sign[sign],
            )
        )
        cursor += 1
    return (children[0], children[1]), cursor


def run_adaptive_phase_forest_candidate(
    root_hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    root_bound: PhaseNodeBound,
    bound_wave: BoundWave,
    *,
    deadline: float,
    max_depth: int = 8,
    max_nodes: int = 64,
    select_binary: Optional[SelectBinary] = None,
) -> AdaptivePhaseForestResult:
    """Explore an adaptive SAFE-only exact-binary forest.

    The root result is supplied because production would invoke this adapter
    only after the ordinary root HybridZ attempt returned UNKNOWN.  Every
    subsequent breadth-first wave is passed to ``bound_wave`` exactly once,
    with all original rivals retained.  Any malformed split, result, timeout,
    cap, non-finite bound, or conservation mismatch returns a proofless
    fallback with no partial certification.
    """

    counters = {
        "roots": 1,
        "children_expected": 0,
        "children_minted": 0,
        "processed": 0,
        "certified": 0,
        "branched": 0,
        "unresolved": 0,
    }
    wave_sizes: list[int] = []
    selected_ids: list[int] = []
    safe_lineages: list[Lineage] = []
    unresolved_lineages: list[Lineage] = []
    integrity_errors: list[str] = []
    root_binary_count = 0
    backend_batch_calls = 0
    threshold_array = np.zeros(0, dtype=np.float64)
    normalized_rivals: Tuple[RivalSpec, ...] = ()
    property_sha256 = ""
    root_was_unknown = False

    def finalize(
        *,
        status: str,
        reason: str,
        all_leaves_safe: bool,
    ) -> AdaptivePhaseForestResult:
        created = counters["roots"] + counters["children_minted"]
        active = max(0, created - counters["processed"])
        creation_ok = (
            created == counters["processed"] + active
        )
        outcome_ok = (
            counters["processed"]
            == counters["certified"]
            + counters["branched"]
            + counters["unresolved"]
        )
        partition_ok = (
            counters["children_expected"]
            == counters["children_minted"]
        )
        conservation_complete = bool(
            creation_ok
            and outcome_ok
            and partition_ok
            and active == 0
            and counters["unresolved"] == 0
            and not integrity_errors
            and all_leaves_safe
        )
        if status == "all_leaves_safe_candidate" and not conservation_complete:
            status = "fallback"
            reason = "node_conservation_incomplete"
            all_leaves_safe = False
        full_cover_leaf_count = (
            1 << root_binary_count
            if 0 <= root_binary_count <= 62
            else None
        )
        observed_lineages = [
            *safe_lineages,
            *unresolved_lineages,
        ]
        max_depth_reached = max(
            (len(lineage) for lineage in observed_lineages),
            default=len(selected_ids) if selected_ids else 0,
        )
        same_depth_full_cover_leaf_count = (
            1 << max_depth_reached
            if 0 <= max_depth_reached <= 62
            else None
        )
        receipt = {
            "schema": (
                "hybridz_adaptive_exact_binary_phase_forest_candidate_v1"
            ),
            "proof_authority": False,
            "status": status,
            "reason": reason,
            "root_was_unknown": bool(root_was_unknown),
            "root_binary_count": int(root_binary_count),
            "rival_count": int(len(threshold_array)),
            "rival_ids": [
                int(rival.rival_id) for rival in normalized_rivals
            ],
            "ordered_property_sha256": (
                property_sha256 if property_sha256 else None
            ),
            "max_depth": int(max_depth)
            if isinstance(max_depth, (int, np.integer))
            and not isinstance(max_depth, (bool, np.bool_))
            else None,
            "max_nodes": int(max_nodes)
            if isinstance(max_nodes, (int, np.integer))
            and not isinstance(max_nodes, (bool, np.bool_))
            else None,
            "backend_batch_calls": int(backend_batch_calls),
            "wave_sizes": [int(value) for value in wave_sizes],
            "shared_node_rival_batching": True,
            "selected_bcol_ids": [int(value) for value in selected_ids],
            "safe_leaf_lineages": [
                [[int(col_id), int(sign)] for col_id, sign in lineage]
                for lineage in safe_lineages
            ],
            "unresolved_lineages": [
                [[int(col_id), int(sign)] for col_id, sign in lineage]
                for lineage in unresolved_lineages
            ],
            "adaptive_safe_leaf_count": int(len(safe_lineages)),
            "full_cover_leaf_count": full_cover_leaf_count,
            "full_cover_log2_leaf_count": int(root_binary_count),
            "max_depth_reached": int(max_depth_reached),
            "same_depth_full_cover_leaf_count": (
                same_depth_full_cover_leaf_count
            ),
            "adaptive_child_bound_count": int(sum(wave_sizes)),
            "counters": {
                **{key: int(value) for key, value in counters.items()},
                "active": int(active),
            },
            "node_conservation": {
                "creation": bool(creation_ok),
                "outcome": bool(outcome_ok),
                "child_partition": bool(partition_ok),
                "complete": bool(conservation_complete),
            },
            "integrity_errors": sorted(set(integrity_errors)),
        }
        return AdaptivePhaseForestResult(
            status=status,
            reason=reason,
            all_leaves_safe=bool(
                all_leaves_safe and conservation_complete
            ),
            proof_authority=False,
            receipt=receipt,
        )

    try:
        if not isinstance(root_hz, SparseHZono):
            raise _CandidateFailure("root_not_sparse_hz")
        root_ids = _remaining_bcol_ids(root_hz)
        root_binary_count = len(root_ids)
        if root_binary_count < 1:
            raise _CandidateFailure("root_has_no_binary_factor")
        normalized_rivals = _normalize_rivals(rivals)
        if len(normalized_rivals[0].objective) != int(root_hz.n_out):
            raise _CandidateFailure(
                "rival_objective_root_output_width_mismatch"
            )
        threshold_array = np.asarray(
            [rival.threshold for rival in normalized_rivals],
            dtype=np.float64,
        )
        property_sha256 = ordered_property_digest(
            normalized_rivals
        )
        try:
            deadline = float(deadline)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _CandidateFailure("deadline_not_numeric") from exc
        if not math.isfinite(deadline):
            raise _CandidateFailure("deadline_not_finite")
        max_depth = _strict_int(max_depth, name="max_depth")
        max_nodes = _strict_int(max_nodes, name="max_nodes")
        if max_depth < 1:
            raise _CandidateFailure("max_depth_not_positive")
        if max_nodes < 3:
            raise _CandidateFailure("max_nodes_too_small")
        if not callable(bound_wave):
            raise _CandidateFailure("bound_wave_not_callable")
        if select_binary is not None and not callable(select_binary):
            raise _CandidateFailure("select_binary_not_callable")
        if _deadline_expired(deadline):
            raise _CandidateFailure("deadline_before_root")

        root = PhaseForestNode(
            node_id=0,
            depth=0,
            lineage=(),
            hz=root_hz,
        )
        validated_root, root_safe = _validate_node_bound(
            root_bound,
            root,
            normalized_rivals,
            property_sha256,
        )
        counters["processed"] = 1
        if root_safe:
            counters["certified"] = 1
            safe_lineages.append(())
            return finalize(
                status="fallback",
                reason="root_already_safe",
                all_leaves_safe=False,
            )
        if validated_root.verdict != "UNKNOWN":
            raise _CandidateFailure("root_not_unknown")
        root_was_unknown = True

        current: list[Tuple[PhaseForestNode, PhaseNodeBound]] = [
            (root, validated_root)
        ]
        next_node_id = 1
        wave_index = 1
        selector = select_binary or _default_select_binary

        while current:
            if _deadline_expired(deadline):
                counters["unresolved"] += len(current)
                unresolved_lineages.extend(
                    node.lineage for node, _bound in current
                )
                return finalize(
                    status="fallback",
                    reason="deadline_before_split_wave",
                    all_leaves_safe=False,
                )
            if any(node.depth >= max_depth for node, _bound in current):
                counters["unresolved"] += len(current)
                unresolved_lineages.extend(
                    node.lineage for node, _bound in current
                )
                return finalize(
                    status="fallback",
                    reason="max_depth",
                    all_leaves_safe=False,
                )
            projected_nodes = (
                counters["roots"]
                + counters["children_minted"]
                + 2 * len(current)
            )
            if projected_nodes > max_nodes:
                counters["unresolved"] += len(current)
                unresolved_lineages.extend(
                    node.lineage for node, _bound in current
                )
                return finalize(
                    status="fallback",
                    reason="max_nodes",
                    all_leaves_safe=False,
                )

            local_children: list[PhaseForestNode] = []
            local_selected: list[int] = []
            proposed_next_id = next_node_id
            selector_bindings_before = tuple(
                _node_binding(node) for node, _bound in current
            )
            try:
                for node, node_bound in current:
                    selected = _strict_int(
                        selector(node, node_bound),
                        name="selector_result",
                    )
                    if tuple(
                        _node_binding(
                            live_node
                        )
                        for live_node, _live_bound in current
                    ) != selector_bindings_before:
                        raise _CandidateFailure(
                            "selector_mutated_node_semantics"
                        )
                    children, proposed_next_id = _split_one_binary(
                        node,
                        selected,
                        next_node_id=proposed_next_id,
                        deadline=deadline,
                    )
                    local_children.extend(children)
                    local_selected.append(selected)
                if tuple(
                    _node_binding(node) for node, _bound in current
                ) != selector_bindings_before:
                    raise _CandidateFailure(
                        "selector_mutated_node_semantics"
                    )
            except _CandidateFailure as exc:
                counters["children_expected"] += 2 * len(current)
                counters["unresolved"] += len(current)
                unresolved_lineages.extend(
                    node.lineage for node, _bound in current
                )
                integrity_errors.append(str(exc))
                return finalize(
                    status="fallback",
                    reason=str(exc),
                    all_leaves_safe=False,
                )

            counters["children_expected"] += len(local_children)
            counters["children_minted"] += len(local_children)
            counters["branched"] += len(current)
            selected_ids.extend(local_selected)
            next_node_id = proposed_next_id

            if _deadline_expired(deadline):
                return finalize(
                    status="fallback",
                    reason="deadline_before_bound_wave",
                    all_leaves_safe=False,
                )
            bindings_before = tuple(
                _node_binding(node) for node in local_children
            )
            request = PhaseBoundWaveRequest(
                wave_index=wave_index,
                nodes=tuple(local_children),
                rivals=normalized_rivals,
                property_digest=property_sha256,
                deadline=deadline,
            )
            wave_sizes.append(len(local_children))
            backend_batch_calls += 1
            try:
                raw_results = bound_wave(request)
            except Exception as exc:
                integrity_errors.append(
                    f"bound_wave_error:{type(exc).__name__}"
                )
                return finalize(
                    status="fallback",
                    reason=f"bound_wave_error:{type(exc).__name__}",
                    all_leaves_safe=False,
                )
            if _deadline_expired(deadline):
                return finalize(
                    status="fallback",
                    reason="deadline_after_bound_wave",
                    all_leaves_safe=False,
                )
            if (
                ordered_property_digest(request.rivals)
                != property_sha256
                or request.property_digest != property_sha256
            ):
                integrity_errors.append(
                    "bound_wave_mutated_property_binding"
                )
                return finalize(
                    status="fallback",
                    reason="bound_wave_mutated_property_binding",
                    all_leaves_safe=False,
                )
            if tuple(_node_binding(node) for node in local_children) != (
                bindings_before
            ):
                integrity_errors.append("bound_wave_mutated_node_binding")
                return finalize(
                    status="fallback",
                    reason="bound_wave_mutated_node_binding",
                    all_leaves_safe=False,
                )
            if not isinstance(raw_results, (tuple, list)):
                integrity_errors.append("bound_wave_result_not_sequence")
                return finalize(
                    status="fallback",
                    reason="bound_wave_result_not_sequence",
                    all_leaves_safe=False,
                )
            if len(raw_results) != len(local_children):
                integrity_errors.append("bound_wave_result_count_mismatch")
                return finalize(
                    status="fallback",
                    reason="bound_wave_result_count_mismatch",
                    all_leaves_safe=False,
                )

            validated_wave: list[
                Tuple[PhaseForestNode, PhaseNodeBound, bool]
            ] = []
            try:
                for node, raw_bound in zip(
                    local_children, raw_results
                ):
                    validated, is_safe = _validate_node_bound(
                        raw_bound,
                        node,
                        normalized_rivals,
                        property_sha256,
                    )
                    validated_wave.append((node, validated, is_safe))
            except _CandidateFailure as exc:
                integrity_errors.append(str(exc))
                return finalize(
                    status="fallback",
                    reason=str(exc),
                    all_leaves_safe=False,
                )

            counters["processed"] += len(validated_wave)
            current = []
            for node, node_bound, is_safe in validated_wave:
                if is_safe:
                    counters["certified"] += 1
                    safe_lineages.append(node.lineage)
                else:
                    current.append((node, node_bound))
            wave_index += 1

        return finalize(
            status="all_leaves_safe_candidate",
            reason="all_adaptive_leaves_strictly_safe",
            all_leaves_safe=True,
        )
    except _CandidateFailure as exc:
        integrity_errors.append(str(exc))
        return finalize(
            status="fallback",
            reason=str(exc),
            all_leaves_safe=False,
        )
    except Exception as exc:
        reason = f"unexpected_error:{type(exc).__name__}"
        integrity_errors.append(reason)
        return finalize(
            status="fallback",
            reason=reason,
            all_leaves_safe=False,
        )


__all__ = [
    "AdaptivePhaseForestResult",
    "PhaseBoundWaveRequest",
    "PhaseForestNode",
    "PhaseNodeBound",
    "RivalSpec",
    "RivalUpperBound",
    "ordered_property_digest",
    "rival_spec_binding_digest",
    "run_adaptive_phase_forest_candidate",
    "sparse_hz_semantic_digest",
]
