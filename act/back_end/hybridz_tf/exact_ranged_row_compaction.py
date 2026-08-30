#!/usr/bin/env python3
"""Candidate-only exact folding of signed upper-row pairs.

The Operator-HZ builder represents a numerically guarded equality band as two
upper rows::

    A z <= u_forward
   -A z <= u_reverse

HiGHS and several other LP/MIP interfaces accept one ranged row.  When the two
stored coefficient rows are *bitwise* sign-negations, the conjunction above is
identically::

   -u_reverse <= A z <= u_forward.

This module demonstrates that structural compaction without changing the
authoritative source frame.  It is deliberately disconnected from the solver,
verifier, configuration, and production Operator-HZ path.  Digests are only
tamper-evident diagnostics; they are never provenance or proof authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp


_SOURCE_SCHEMA = "act.forward_exact.signed_upper_source.v1"
_CANDIDATE_SCHEMA = "act.forward_exact.native_ranged_candidate.v1"
_RECEIPT_SCHEMA = "act.forward_exact.native_ranged_receipt.v1"
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "source_sha256",
        "candidate_sha256",
        "source_rows",
        "candidate_rows",
        "source_constraint_nnz",
        "candidate_constraint_nnz",
        "folded_pair_count",
        "exact_bitwise_sign_negation_required",
        "hash_is_identity_authority",
        "source_frame_retained_by_candidate",
        "source_frame_required_for_replay",
        "candidate_only",
        "provenance_authority",
        "authenticity_authority",
        "proof_authority",
        "verdict_authority",
        "production_integration",
        "triangle_relaxation_called",
        "branch_and_bound_called",
        "backward_called",
        "dual_called",
        "solver_called",
    }
)
_RECEIPT_INT_KEYS = frozenset(
    {
        "source_rows",
        "candidate_rows",
        "source_constraint_nnz",
        "candidate_constraint_nnz",
        "folded_pair_count",
    }
)
_RECEIPT_BOOL_KEYS = _RECEIPT_KEYS - _RECEIPT_INT_KEYS - frozenset(
    {"schema", "source_sha256", "candidate_sha256"}
)


def _is_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _receipt_state_is_exact(value: Any) -> bool:
    return bool(
        type(value) is dict
        and all(type(key) is str for key in value)
        and frozenset(value) == _RECEIPT_KEYS
        and type(value.get("schema")) is str
        and value["schema"] == _RECEIPT_SCHEMA
        and _is_sha256(value.get("source_sha256"))
        and _is_sha256(value.get("candidate_sha256"))
        and all(type(value[key]) is int and value[key] >= 0 for key in _RECEIPT_INT_KEYS)
        and all(type(value[key]) is bool for key in _RECEIPT_BOOL_KEYS)
        and value["candidate_only"] is True
        and value["exact_bitwise_sign_negation_required"] is True
        and value["source_frame_required_for_replay"] is True
        and all(
            value[key] is False
            for key in (
                "hash_is_identity_authority",
                "source_frame_retained_by_candidate",
                "provenance_authority",
                "authenticity_authority",
                "proof_authority",
                "verdict_authority",
                "production_integration",
                "triangle_relaxation_called",
                "branch_and_bound_called",
                "backward_called",
                "dual_called",
                "solver_called",
            )
        )
    )


def _frozen_array(value: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype:
        raise ValueError(f"{name} must be an exact {np.dtype(dtype).name} ndarray")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    copied = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)
    copied.setflags(write=False)
    return copied


def _frozen_csr(value: Any, *, rows: int | None, cols: int | None, name: str) -> sp.csr_matrix:
    if type(value) is not sp.csr_matrix or value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be an exact float64 csr_matrix")
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank two")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{name} row count mismatch")
    if cols is not None and value.shape[1] != cols:
        raise ValueError(f"{name} column count mismatch")
    if (
        type(value.data) is not np.ndarray
        or type(value.indices) is not np.ndarray
        or type(value.indptr) is not np.ndarray
        or value.data.dtype != np.dtype(np.float64)
        or value.indices.dtype != np.dtype(np.int32)
        or value.indptr.dtype != np.dtype(np.int32)
        or value.data.ndim != 1
        or value.indices.ndim != 1
        or value.indptr.ndim != 1
        or not value.data.flags.c_contiguous
        or not value.indices.flags.c_contiguous
        or not value.indptr.flags.c_contiguous
    ):
        raise ValueError(f"{name} must use int32 CSR indices")
    if value.indptr.size != value.shape[0] + 1:
        raise ValueError(f"{name} has a malformed indptr length")
    if (
        int(value.indptr[0]) != 0
        or np.any(value.indptr[1:] < value.indptr[:-1])
        or int(value.indptr[-1]) != value.data.size
        or value.indices.size != value.data.size
    ):
        raise ValueError(f"{name} has malformed CSR pointer bounds")
    if value.indices.size and (
        np.any(value.indices < 0) or np.any(value.indices >= value.shape[1])
    ):
        raise ValueError(f"{name} has an out-of-domain column index")
    for row in range(value.shape[0]):
        start = int(value.indptr[row])
        stop = int(value.indptr[row + 1])
        if stop - start > 1 and np.any(
            value.indices[start + 1 : stop] <= value.indices[start : stop - 1]
        ):
            raise ValueError(f"{name} row indices are not strictly increasing")
    if not value.has_canonical_format or not value.has_sorted_indices:
        raise ValueError(f"{name} must be canonical and sorted")
    if value.nnz and (
        not np.all(np.isfinite(value.data)) or np.any(value.data == 0.0)
    ):
        raise ValueError(f"{name} must be finite and contain no explicit zeros")
    data = np.frombuffer(value.data.tobytes(order="C"), dtype=np.float64)
    indices = np.frombuffer(value.indices.tobytes(order="C"), dtype=np.int32)
    indptr = np.frombuffer(value.indptr.tobytes(order="C"), dtype=np.int32)
    out = sp.csr_matrix((data, indices, indptr), shape=value.shape, copy=False)
    if (
        not out.has_canonical_format
        or not out.has_sorted_indices
        or out.indptr.size != out.shape[0] + 1
        or int(out.indptr[0]) != 0
        or int(out.indptr[-1]) != out.data.size
        or out.indices.size != out.data.size
        or (
            out.indices.size
            and (
                np.any(out.indices < 0)
                or np.any(out.indices >= out.shape[1])
            )
        )
    ):
        raise ValueError(f"{name} immutable snapshot is not canonical")
    out.data.setflags(write=False)
    out.indices.setflags(write=False)
    out.indptr.setflags(write=False)
    return out


def _update_array_digest(hasher: Any, name: str, value: np.ndarray) -> None:
    hasher.update(name.encode("ascii"))
    hasher.update(value.dtype.str.encode("ascii"))
    hasher.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    hasher.update(value.tobytes(order="C"))


def _update_csr_digest(hasher: Any, name: str, value: sp.csr_matrix) -> None:
    hasher.update(name.encode("ascii"))
    hasher.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    _update_array_digest(hasher, f"{name}.indptr", value.indptr)
    _update_array_digest(hasher, f"{name}.indices", value.indices)
    _update_array_digest(hasher, f"{name}.data", value.data)


def _frame_digest(
    A_cont: sp.csr_matrix,
    A_bin: sp.csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    row_tags: Tuple[str, ...],
) -> str:
    hasher = hashlib.sha256()
    _update_csr_digest(hasher, "A_cont", A_cont)
    _update_csr_digest(hasher, "A_bin", A_bin)
    _update_array_digest(hasher, "lower", lower)
    _update_array_digest(hasher, "upper", upper)
    hasher.update(
        json.dumps(row_tags, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    )
    return hasher.hexdigest()


@dataclass(frozen=True)
class SignedUpperSource:
    """Private immutable snapshot of an upper-row source frame."""

    A_cont: sp.csr_matrix
    A_bin: sp.csr_matrix
    upper: np.ndarray
    row_tags: Tuple[str, ...]
    source_sha256: str = ""
    schema: str = _SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != _SOURCE_SCHEMA:
            raise ValueError("source schema is invalid")
        if type(self.row_tags) is not tuple or not all(
            type(tag) is str and tag for tag in self.row_tags
        ):
            raise ValueError("row_tags must be a tuple of nonempty builtin strings")
        if type(self.upper) is not np.ndarray or self.upper.ndim != 1:
            raise ValueError("upper must be a rank-one ndarray")
        upper = _frozen_array(self.upper, dtype=np.dtype(np.float64), name="upper")
        if not np.all(np.isfinite(upper)):
            raise ValueError("upper must be finite")
        A_cont = _frozen_csr(
            self.A_cont, rows=upper.size, cols=None, name="A_cont"
        )
        A_bin = _frozen_csr(
            self.A_bin, rows=upper.size, cols=None, name="A_bin"
        )
        if len(self.row_tags) != upper.size:
            raise ValueError("row tag count mismatch")
        if type(self.source_sha256) is not str or (
            self.source_sha256 != "" and not _is_sha256(self.source_sha256)
        ):
            raise ValueError("source digest is malformed")
        lower = np.full(upper.size, -np.inf, dtype=np.float64)
        lower.setflags(write=False)
        observed = _frame_digest(A_cont, A_bin, lower, upper, self.row_tags)
        if self.source_sha256 and self.source_sha256 != observed:
            raise ValueError("source digest is stale")
        object.__setattr__(self, "A_cont", A_cont)
        object.__setattr__(self, "A_bin", A_bin)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "source_sha256", observed)


def _snapshot_source(value: Any) -> SignedUpperSource:
    if type(value) is not SignedUpperSource:
        raise ValueError("source has the wrong exact type")
    if type(value.row_tags) is not tuple:
        raise ValueError("live source row tags are malformed")
    return SignedUpperSource(
        value.A_cont,
        value.A_bin,
        value.upper,
        tuple(value.row_tags),
        source_sha256=value.source_sha256,
        schema=value.schema,
    )


def _row_payload(matrix: sp.csr_matrix, row: int) -> Tuple[np.ndarray, np.ndarray]:
    start = int(matrix.indptr[row])
    stop = int(matrix.indptr[row + 1])
    return matrix.indices[start:stop], matrix.data[start:stop]


def _exact_negative_row(
    A_cont: sp.csr_matrix,
    A_bin: sp.csr_matrix,
    forward: int,
    reverse: int,
) -> bool:
    for matrix in (A_cont, A_bin):
        fi, fd = _row_payload(matrix, forward)
        ri, rd = _row_payload(matrix, reverse)
        if not np.array_equal(fi, ri) or fd.size != rd.size:
            return False
        negated = np.negative(fd)
        if not np.array_equal(negated.view(np.uint64), rd.view(np.uint64)):
            return False
    return True


def _base_tag(tag: str, suffix: str) -> str | None:
    marker = f":{suffix}"
    return tag[: -len(marker)] if tag.endswith(marker) and len(tag) > len(marker) else None


def _pair_schedule(tags: Tuple[str, ...]) -> Tuple[Tuple[int, int, str], ...]:
    forward: Dict[str, List[int]] = {}
    reverse: Dict[str, List[int]] = {}
    for row, tag in enumerate(tags):
        base = _base_tag(tag, "forward")
        if base is not None:
            forward.setdefault(base, []).append(row)
            continue
        base = _base_tag(tag, "reverse")
        if base is not None:
            reverse.setdefault(base, []).append(row)
    pairs: List[Tuple[int, int, str]] = []
    for base in sorted(set(forward) & set(reverse)):
        left = forward[base]
        right = reverse[base]
        if len(left) != len(right):
            continue
        pairs.extend((frow, rrow, base) for frow, rrow in zip(left, right))
    pairs.sort()
    return tuple(pairs)


def _stack_rows(matrix: sp.csr_matrix, rows: Iterable[int]) -> sp.csr_matrix:
    indices = np.fromiter(rows, dtype=np.int64)
    if indices.size == 0:
        return sp.csr_matrix((0, matrix.shape[1]), dtype=np.float64)
    out = matrix[indices, :].tocsr()
    out.eliminate_zeros()
    out.sort_indices()
    return out


@dataclass(frozen=True)
class ExactRangedRowCandidate:
    """Non-authoritative exact native-row compaction candidate."""

    A_cont: sp.csr_matrix
    A_bin: sp.csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    row_tags: Tuple[str, ...]
    source_to_candidate: np.ndarray
    folded_pairs: Tuple[Tuple[int, int], ...]
    receipt: Any
    candidate_sha256: str
    proof_authority: bool = False
    verdict_authority: bool = False
    production_integration: bool = False
    schema: str = _CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema) is not str
            or self.schema != _CANDIDATE_SCHEMA
            or self.proof_authority is not False
            or self.verdict_authority is not False
            or self.production_integration is not False
        ):
            raise ValueError("candidate authority firewall is invalid")
        if type(self.lower) is not np.ndarray or self.lower.ndim != 1:
            raise ValueError("candidate lower is malformed")
        lower = _frozen_array(self.lower, dtype=np.dtype(np.float64), name="lower")
        upper = _frozen_array(self.upper, dtype=np.dtype(np.float64), name="upper")
        if (
            lower.shape != upper.shape
            or np.any(np.isnan(lower))
            or np.any(np.isposinf(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise ValueError("candidate row bounds are malformed")
        A_cont = _frozen_csr(self.A_cont, rows=lower.size, cols=None, name="candidate A_cont")
        A_bin = _frozen_csr(self.A_bin, rows=lower.size, cols=None, name="candidate A_bin")
        if type(self.row_tags) is not tuple or len(self.row_tags) != lower.size or not all(
            type(tag) is str and tag for tag in self.row_tags
        ):
            raise ValueError("candidate row tags are malformed")
        mapping = _frozen_array(
            self.source_to_candidate,
            dtype=np.dtype(np.int64),
            name="source_to_candidate",
        )
        if mapping.ndim != 1 or np.any(mapping < 0) or (
            mapping.size and int(mapping.max()) >= lower.size
        ):
            raise ValueError("source row mapping is malformed")
        if type(self.folded_pairs) is not tuple or not all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is int
            and type(pair[1]) is int
            and 0 <= pair[0] < mapping.size
            and 0 <= pair[1] < mapping.size
            and pair[0] != pair[1]
            for pair in self.folded_pairs
        ):
            raise ValueError("folded pair mapping is malformed")
        if type(self.receipt) is not dict or type(self.candidate_sha256) is not str:
            raise ValueError("candidate receipt is malformed")
        observed = _frame_digest(A_cont, A_bin, lower, upper, self.row_tags)
        if not _is_sha256(self.candidate_sha256) or self.candidate_sha256 != observed:
            raise ValueError("candidate frame digest is stale")
        frozen_receipt = json.loads(
            json.dumps(self.receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        if not _receipt_state_is_exact(frozen_receipt):
            raise ValueError("candidate receipt schema is invalid")
        if (
            frozen_receipt["candidate_sha256"] != observed
            or frozen_receipt["candidate_rows"] != int(lower.size)
            or frozen_receipt["candidate_constraint_nnz"]
            != int(A_cont.nnz + A_bin.nnz)
            or frozen_receipt["folded_pair_count"] != len(self.folded_pairs)
        ):
            raise ValueError("candidate receipt is not bound to the frame")
        object.__setattr__(self, "A_cont", A_cont)
        object.__setattr__(self, "A_bin", A_bin)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "source_to_candidate", mapping)
        object.__setattr__(self, "receipt", MappingProxyType(frozen_receipt))


def fold_exact_signed_upper_pairs(source: SignedUpperSource) -> ExactRangedRowCandidate:
    """Fold structurally paired, bitwise opposite upper rows into ranges."""

    frame = _snapshot_source(source)
    schedule = _pair_schedule(frame.row_tags)
    reverse_to_forward: Dict[int, Tuple[int, str]] = {}
    for forward, reverse, base in schedule:
        if forward == reverse or forward in reverse_to_forward or reverse in reverse_to_forward:
            raise ValueError("row pair schedule reuses a source row")
        if _exact_negative_row(frame.A_cont, frame.A_bin, forward, reverse):
            reverse_to_forward[reverse] = (forward, base)

    kept_rows: List[int] = []
    new_tags: List[str] = []
    source_to_candidate = np.full(frame.upper.size, -1, dtype=np.int64)
    pair_by_forward = {
        forward: (reverse, base)
        for reverse, (forward, base) in reverse_to_forward.items()
    }
    lowers: List[float] = []
    uppers: List[float] = []
    folded_pairs: List[Tuple[int, int]] = []
    for source_row in range(frame.upper.size):
        if source_row in reverse_to_forward:
            continue
        candidate_row = len(kept_rows)
        kept_rows.append(source_row)
        source_to_candidate[source_row] = candidate_row
        paired = pair_by_forward.get(source_row)
        if paired is None:
            lowers.append(-math.inf)
            uppers.append(float(frame.upper[source_row]))
            new_tags.append(frame.row_tags[source_row])
        else:
            reverse, base = paired
            lower = -float(frame.upper[reverse])
            upper = float(frame.upper[source_row])
            lowers.append(lower)
            uppers.append(upper)
            new_tags.append(f"range:{base}")
            source_to_candidate[reverse] = candidate_row
            folded_pairs.append((source_row, reverse))
    if np.any(source_to_candidate < 0):
        raise ValueError("source-to-candidate mapping is incomplete")

    A_cont = _stack_rows(frame.A_cont, kept_rows)
    A_bin = _stack_rows(frame.A_bin, kept_rows)
    lower = np.asarray(lowers, dtype=np.float64)
    upper = np.asarray(uppers, dtype=np.float64)
    row_tags = tuple(new_tags)
    candidate_sha = _frame_digest(A_cont, A_bin, lower, upper, row_tags)
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "source_sha256": frame.source_sha256,
        "candidate_sha256": candidate_sha,
        "source_rows": int(frame.upper.size),
        "candidate_rows": int(lower.size),
        "source_constraint_nnz": int(frame.A_cont.nnz + frame.A_bin.nnz),
        "candidate_constraint_nnz": int(A_cont.nnz + A_bin.nnz),
        "folded_pair_count": int(len(folded_pairs)),
        "exact_bitwise_sign_negation_required": True,
        "hash_is_identity_authority": False,
        "source_frame_retained_by_candidate": False,
        "source_frame_required_for_replay": True,
        "candidate_only": True,
        "provenance_authority": False,
        "authenticity_authority": False,
        "proof_authority": False,
        "verdict_authority": False,
        "production_integration": False,
        "triangle_relaxation_called": False,
        "branch_and_bound_called": False,
        "backward_called": False,
        "dual_called": False,
        "solver_called": False,
    }
    return ExactRangedRowCandidate(
        A_cont=A_cont,
        A_bin=A_bin,
        lower=lower,
        upper=upper,
        row_tags=row_tags,
        source_to_candidate=source_to_candidate,
        folded_pairs=tuple(folded_pairs),
        receipt=receipt,
        candidate_sha256=candidate_sha,
    )


def source_and_candidate_membership(
    source: SignedUpperSource,
    candidate: ExactRangedRowCandidate,
    continuous: np.ndarray,
    binary: np.ndarray,
) -> Tuple[bool, bool]:
    """Diagnostic exact-binary64 membership comparison for tests only."""

    frame = _snapshot_source(source)
    checked = candidate
    xc = np.asarray(continuous, dtype=np.float64).reshape(-1)
    xb = np.asarray(binary, dtype=np.float64).reshape(-1)
    if xc.size != frame.A_cont.shape[1] or xb.size != frame.A_bin.shape[1]:
        raise ValueError("membership point width mismatch")
    source_value = np.asarray(frame.A_cont @ xc + frame.A_bin @ xb).reshape(-1)
    candidate_value = np.asarray(checked.A_cont @ xc + checked.A_bin @ xb).reshape(-1)
    return (
        bool(np.all(source_value <= frame.upper)),
        bool(np.all(candidate_value >= checked.lower) and np.all(candidate_value <= checked.upper)),
    )


def _csr_bitwise_equal(left: Any, right: Any) -> bool:
    return bool(
        type(left) is sp.csr_matrix
        and type(right) is sp.csr_matrix
        and left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data.view(np.uint64), right.data.view(np.uint64))
    )


def validate_exact_ranged_candidate(
    source: SignedUpperSource,
    candidate: ExactRangedRowCandidate,
) -> bool:
    """Strict structural replay against the original source snapshot."""

    try:
        frame = _snapshot_source(source)
        if (
            type(candidate) is not ExactRangedRowCandidate
            or type(candidate.schema) is not str
            or candidate.schema != _CANDIDATE_SCHEMA
            or candidate.proof_authority is not False
            or candidate.verdict_authority is not False
            or candidate.production_integration is not False
            or type(candidate.receipt) is not MappingProxyType
            or not _receipt_state_is_exact(dict(candidate.receipt))
            or type(candidate.candidate_sha256) is not str
            or not _is_sha256(candidate.candidate_sha256)
        ):
            return False
        live = ExactRangedRowCandidate(
            A_cont=candidate.A_cont,
            A_bin=candidate.A_bin,
            lower=candidate.lower,
            upper=candidate.upper,
            row_tags=candidate.row_tags,
            source_to_candidate=candidate.source_to_candidate,
            folded_pairs=candidate.folded_pairs,
            receipt=dict(candidate.receipt),
            candidate_sha256=candidate.candidate_sha256,
            proof_authority=candidate.proof_authority,
            verdict_authority=candidate.verdict_authority,
            production_integration=candidate.production_integration,
            schema=candidate.schema,
        )
        expected = fold_exact_signed_upper_pairs(frame)
        return bool(
            live.candidate_sha256 == expected.candidate_sha256
            and live.row_tags == expected.row_tags
            and live.folded_pairs == expected.folded_pairs
            and _csr_bitwise_equal(live.A_cont, expected.A_cont)
            and _csr_bitwise_equal(live.A_bin, expected.A_bin)
            and np.array_equal(live.lower.view(np.uint64), expected.lower.view(np.uint64))
            and np.array_equal(live.upper.view(np.uint64), expected.upper.view(np.uint64))
            and np.array_equal(live.source_to_candidate, expected.source_to_candidate)
            and json.dumps(dict(live.receipt), sort_keys=True, separators=(",", ":"))
            == json.dumps(dict(expected.receipt), sort_keys=True, separators=(",", ":"))
        )
    except (AttributeError, TypeError, ValueError, OverflowError, MemoryError):
        return False


__all__ = [
    "SignedUpperSource",
    "ExactRangedRowCandidate",
    "fold_exact_signed_upper_pairs",
    "source_and_candidate_membership",
    "validate_exact_ranged_candidate",
]
