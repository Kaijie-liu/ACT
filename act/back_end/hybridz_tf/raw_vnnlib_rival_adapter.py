#!/usr/bin/env python3
"""Exact, candidate-only raw VNNLIB TOP1 to ``RivalSpec`` adapter.

The adapter recognizes one deliberately narrow property:

* one strict VNNLIB 1.0-flat or single-network VNNLIB 2.0 source;
* zero or more scalar affine input bounds;
* exactly one output ASSERT rooted at ``or``;
* one affine output-only non-strict comparison per branch; and
* the complete unsafe TOP1 family ``Y_true - Y_j <= 0``.

Raw numeric syntax is interpreted with exact :class:`fractions.Fraction`
arithmetic.  In particular, binary floating-point rounding is never used to
decide whether a raw coefficient is ``-1``, ``0`` or ``1``.  The live ASSERT
boundary is independently narrow: an exact five-key B=1 TOP1 mapping backed by
float32/float64 tensors or arrays and an int64 label.

Issuance exposes metadata only.  It does not expose usable ``RivalSpec``
objects.  A bounded, process-local, expiring capability must be consumed
against the same source file identity and exact live ASSERT bytes.  Successful
consumption returns a :class:`ConsumedRivalBatch` with a process-local,
owner-bound identity.  Everything remains candidate-only and
``proof_authority=False``; this module is not connected to a verdict path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple
import weakref

import numpy as np
import torch

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    rival_spec_binding_digest,
)


_SCHEMA = "act.raw_vnnlib_top1_rival_candidate.v2"
_ASSERT_DIGEST_SCHEMA = "act.raw_vnnlib_top1_assert_binding.v2"
_LIVE_ASSERT_SCHEMA = "act.raw_vnnlib_top1_live_assert.v2"
_CONSUMED_BATCH_SCHEMA = "act.raw_vnnlib_top1_consumed_batch.v1"
_TRANSFORM = (
    "exact_raw_unsafe_le_to_top1_violation_upper_v2:"
    "objective=-canonical_le_y_coefficients;"
    "threshold=-canonical_le_threshold"
)
_TOP1_KIND = "TOP1_ROBUST"

_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_MAX_INPUT_DIM = 10_000_000
_MAX_OUTPUT_DIM = 10_000
_MAX_TENSOR_RANK = 16
_MAX_LIVE_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_SEXPR_TOKENS = 1_000_000
_MAX_SEXPR_DEPTH = 256
_MAX_AFFINE_NODES = 100_000
_MAX_NUMERIC_TOKEN_BYTES = 4096
_MAX_ABS_DECIMAL_EXPONENT = 10_000
_MAX_FRACTION_BITS = 65_536
_DEFAULT_OPERATION_TIMEOUT_SECONDS = 30.0
_DEFAULT_CAPABILITY_TTL_SECONDS = 300.0
_MAX_CAPABILITY_TTL_SECONDS = 3600.0
_MAX_LIVE_RECORDS = 4096

_CAPABILITY_SENTINEL = object()
_BATCH_IDENTITY_SENTINEL = object()
_LIVE_LOCK = threading.RLock()

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LEGACY_SYMBOL_RE = re.compile(r"([XY])_([0-9]+)\Z")
_INDEXED_SYMBOL_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]\Z"
)
_CANONICAL_UNSIGNED_INTEGER_RE = re.compile(
    r"(?:0|[1-9][0-9]*)\Z"
)
_NUMERIC_RE = re.compile(
    r"[+-]?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?(?:0|[1-9][0-9]*))?\Z"
)
_BOOLEAN_OR_ARITHMETIC_OPERATORS = {
    "and",
    "or",
    "<",
    ">",
    "<=",
    ">=",
    "=",
    "==",
    "+",
    "-",
    "*",
    "/",
}


class RawVNNLibRivalAdapterError(ValueError):
    """Fail-closed adapter rejection."""


class _LiveCapability:
    """Opaque, non-serializable process-local issuance capability."""

    __slots__ = ("nonce", "__weakref__")

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise TypeError("raw VNNLIB rival capabilities are issuer-only")
        self.nonce = secrets.token_hex(32)

    def __reduce__(self):
        raise TypeError("raw VNNLIB rival capabilities are process-local")


class _ConsumedBatchIdentity:
    """Issuer-only identity bound to one exact consumed batch object."""

    __slots__ = (
        "nonce",
        "process_id",
        "_owner_ref",
        "_issued_batch_sha256",
        "_issued_receipt_sha256",
        "_issued_receipt",
        "_issued_rivals",
        "_issued_rival_bindings",
    )

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _BATCH_IDENTITY_SENTINEL:
            raise TypeError("consumed rival identities are issuer-only")
        self.nonce = secrets.token_hex(32)
        self.process_id = os.getpid()
        self._owner_ref: weakref.ReferenceType[Any] | None = None
        self._issued_batch_sha256: str | None = None
        self._issued_receipt_sha256: str | None = None
        self._issued_receipt: Mapping[str, Any] | None = None
        self._issued_rivals: Tuple[RivalSpec, ...] | None = None
        self._issued_rival_bindings: Tuple[Tuple[int, str], ...] | None = None

    def bind_owner(
        self,
        owner: object,
        *,
        batch_sha256: str,
        receipt_sha256: str,
        receipt: Mapping[str, Any],
        rivals: Tuple[RivalSpec, ...],
        rival_bindings: Tuple[Tuple[int, str], ...],
    ) -> None:
        if self._owner_ref is not None:
            raise RuntimeError("consumed rival identity already bound")
        if (
            type(owner) is not ConsumedRivalBatch
            or not _valid_sha256(batch_sha256)
            or not _valid_sha256(receipt_sha256)
            or type(receipt) is not MappingProxyType
            or type(rivals) is not tuple
            or type(rival_bindings) is not tuple
            or any(type(rival) is not RivalSpec for rival in rivals)
        ):
            raise RuntimeError("consumed rival identity snapshot invalid")
        self._owner_ref = weakref.ref(owner)
        self._issued_batch_sha256 = batch_sha256
        self._issued_receipt_sha256 = receipt_sha256
        self._issued_receipt = receipt
        self._issued_rivals = rivals
        self._issued_rival_bindings = rival_bindings

    def owns(self, owner: object) -> bool:
        return (
            self.process_id == os.getpid()
            and self._owner_ref is not None
            and self._owner_ref() is owner
        )

    def matches_issued_snapshot(
        self,
        *,
        batch_sha256: str,
        receipt_sha256: str,
        receipt: Mapping[str, Any],
        rivals: Tuple[RivalSpec, ...],
        rival_bindings: Tuple[Tuple[int, str], ...],
    ) -> bool:
        return (
            self._issued_batch_sha256 == batch_sha256
            and self._issued_receipt_sha256 == receipt_sha256
            and self._issued_receipt is receipt
            and self._issued_rivals is rivals
            and self._issued_rival_bindings == rival_bindings
        )

    def __reduce__(self):
        raise TypeError("consumed rival identities are process-local")


@dataclass(frozen=True)
class RawVNNLibTop1Row:
    """Candidate metadata for one raw branch; contains no usable objective."""

    encoded_row: int
    competitor_class: int
    assert_ordinal: int
    boolean_path: Tuple[int, ...]
    canonical_atom: str
    transform: str
    assert_digest: str
    rival_binding_sha256: str


@dataclass(frozen=True, eq=False)
class RawVNNLibTop1Candidate:
    """Non-authoritative metadata plus one expiring, single-use capability."""

    source_path: str
    source_device: int
    source_inode: int
    source_size: int
    source_mtime_ns: int
    vnnlib_sha256: str
    dialect: str
    true_class: int
    output_assert_ordinal: int
    rows: Tuple[RawVNNLibTop1Row, ...]
    live_assert_sha256: str
    candidate_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _live_capability: object = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, eq=False)
class ConsumedRivalBatch:
    """Post-consumption candidate batch with an owner-bound live identity."""

    source_path: str
    vnnlib_sha256: str
    dialect: str
    true_class: int
    output_assert_ordinal: int
    _rivals: Tuple[RivalSpec, ...] = field(repr=False)
    live_assert_sha256: str
    candidate_sha256: str
    batch_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _live_identity: object = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def rivals(self) -> Tuple[RivalSpec, ...]:
        """Expose rivals only through the exact owner-bound batch."""

        if not validate_consumed_raw_vnnlib_rival_batch(self):
            raise RawVNNLibRivalAdapterError(
                "consumed_rival_batch_live_identity_invalid"
            )
        return self._rivals


@dataclass(frozen=True)
class _SourceIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _ArraySnapshot:
    array: np.ndarray
    source_kind: str
    source_dtype: str
    source_shape: Tuple[int, ...]
    source_bytes: bytes


@dataclass(frozen=True)
class _LiveAssertSnapshot:
    kind: str
    M: int
    y_true: int
    C: np.ndarray
    thresholds: np.ndarray
    C_source_kind: str
    C_source_dtype: str
    thresholds_source_kind: str
    thresholds_source_dtype: str
    y_true_source_kind: str
    y_true_source_dtype: str
    digest: str


@dataclass(frozen=True)
class _RawAtom:
    assert_ordinal: int
    boolean_path: Tuple[int, ...]
    canonical_atom: str
    true_class: int
    competitor_class: int


@dataclass(frozen=True)
class _ExactAffine:
    x: Mapping[int, Fraction]
    y: Mapping[int, Fraction]
    constant: Fraction


@dataclass(frozen=True)
class _SymbolLayout:
    dialect: str
    num_inputs: int
    num_outputs: int
    legacy_symbols: Mapping[str, Tuple[str, int]]
    input_name: str | None = None
    input_shape: Tuple[int, ...] = ()
    output_name: str | None = None
    output_shape: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _LiveRecord:
    capability: _LiveCapability
    candidate_ref: weakref.ReferenceType[RawVNNLibTop1Candidate]
    rivals: Tuple[RivalSpec, ...]
    issued_rivals: Tuple[RivalSpec, ...]
    candidate_receipt: Mapping[str, Any]
    source_path: Path
    source_identity: _SourceIdentity
    expected_sha256: str
    live_assert_sha256: str
    candidate_sha256: str
    candidate_receipt_sha256: str
    rival_bindings: Tuple[Tuple[int, str], ...]
    process_id: int
    issued_monotonic: float
    expires_monotonic: float


_LIVE_RECORDS: dict[int, _LiveRecord] = {}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(child) for child in value)
    return value


def _exact_frozen_equal(actual: Any, expected: Any) -> bool:
    """Compare recursively frozen receipt values without type coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, MappingProxyType):
        if len(actual) != len(expected):
            return False
        actual_keys = tuple(actual.keys())
        for expected_key, expected_value in expected.items():
            matches = [
                key
                for key in actual_keys
                if type(key) is type(expected_key) and key == expected_key
            ]
            if len(matches) != 1 or not _exact_frozen_equal(
                actual[matches[0]], expected_value
            ):
                return False
        return True
    if type(expected) is tuple:
        return len(actual) == len(expected) and all(
            _exact_frozen_equal(left, right)
            for left, right in zip(actual, expected)
        )
    if type(expected) is frozenset:
        return actual == expected
    return actual == expected


def _receipt_matches_exact(
    receipt: Any,
    *,
    expected_payload: Mapping[str, Any],
    expected_sha256: str,
) -> bool:
    if not _valid_sha256(expected_sha256):
        return False
    try:
        if _canonical_sha256(expected_payload) != expected_sha256:
            return False
        return _exact_frozen_equal(
            receipt, _deep_freeze(expected_payload)
        )
    except Exception:
        return False


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_deadline(value: Any) -> float:
    now = time.monotonic()
    if value is None:
        return now + _DEFAULT_OPERATION_TIMEOUT_SECONDS
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Real
    ):
        raise RawVNNLibRivalAdapterError("deadline_not_real")
    deadline = float(value)
    if not math.isfinite(deadline):
        raise RawVNNLibRivalAdapterError("deadline_nonfinite")
    if deadline <= now:
        raise RawVNNLibRivalAdapterError("operation_deadline_expired")
    return deadline


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise RawVNNLibRivalAdapterError(
            f"operation_deadline_expired:{stage}"
        )


def _normalize_ttl(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Real
    ):
        raise RawVNNLibRivalAdapterError("capability_ttl_not_real")
    ttl = float(value)
    if (
        not math.isfinite(ttl)
        or ttl <= 0.0
        or ttl > _MAX_CAPABILITY_TTL_SECONDS
    ):
        raise RawVNNLibRivalAdapterError(
            "capability_ttl_out_of_range"
        )
    return ttl


def _read_bound_source(
    raw_path: str | Path,
    expected_sha256: str,
    *,
    deadline: float,
) -> Tuple[Path, bytes, str, _SourceIdentity]:
    if not _valid_sha256(expected_sha256):
        raise RawVNNLibRivalAdapterError(
            "expected_vnnlib_sha256_malformed"
        )
    _check_deadline(deadline, stage="source_resolve")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_source_path_invalid"
        ) from exc

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        _check_deadline(deadline, stage="source_open")
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RawVNNLibRivalAdapterError(
                "vnnlib_source_not_regular_file"
            )
        if before.st_size < 1 or before.st_size > _MAX_SOURCE_BYTES:
            raise RawVNNLibRivalAdapterError(
                "vnnlib_source_size_out_of_range"
            )

        payload = bytearray()
        while True:
            _check_deadline(deadline, stage="source_read")
            remaining = _MAX_SOURCE_BYTES + 1 - len(payload)
            if remaining <= 0:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib_source_size_out_of_range"
                )
            chunk = os.read(
                descriptor, min(_READ_CHUNK_BYTES, remaining)
            )
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > _MAX_SOURCE_BYTES:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib_source_size_out_of_range"
                )

        after = os.fstat(descriptor)
        before_identity = _SourceIdentity(
            device=int(before.st_dev),
            inode=int(before.st_ino),
            size=int(before.st_size),
            mtime_ns=int(before.st_mtime_ns),
        )
        after_identity = _SourceIdentity(
            device=int(after.st_dev),
            inode=int(after.st_ino),
            size=int(after.st_size),
            mtime_ns=int(after.st_mtime_ns),
        )
        if before_identity != after_identity:
            raise RawVNNLibRivalAdapterError(
                "vnnlib_source_changed_during_read"
            )
    except RawVNNLibRivalAdapterError:
        raise
    except OSError as exc:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_source_read_failed"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    source_payload = bytes(payload)
    if not source_payload:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_source_size_out_of_range"
        )
    observed = hashlib.sha256(source_payload).hexdigest()
    if observed != expected_sha256:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_source_sha256_mismatch"
        )
    _check_deadline(deadline, stage="source_hash")
    return path, source_payload, observed, before_identity


def _array_snapshot(
    value: Any,
    *,
    name: str,
    expected_ndim: int,
    integer: bool = False,
) -> _ArraySnapshot:
    allowed_numpy = (
        {np.dtype(np.int64)}
        if integer
        else {np.dtype(np.float32), np.dtype(np.float64)}
    )
    allowed_torch = (
        {torch.int64}
        if integer
        else {torch.float32, torch.float64}
    )

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise RawVNNLibRivalAdapterError(
                f"{name}_tensor_layout_unsupported"
            )
        if value.dtype not in allowed_torch:
            raise RawVNNLibRivalAdapterError(
                f"{name}_dtype_unsupported"
            )
        shape = tuple(int(item) for item in value.shape)
        if len(shape) != expected_ndim:
            raise RawVNNLibRivalAdapterError(
                f"{name}_rank_mismatch"
            )
        byte_count = int(value.numel()) * int(value.element_size())
        if byte_count > _MAX_LIVE_ARRAY_BYTES:
            raise RawVNNLibRivalAdapterError(
                f"{name}_byte_cap_exceeded"
            )
        try:
            array = (
                value.detach()
                .cpu()
                .contiguous()
                .numpy()
                .copy(order="C")
            )
        except Exception as exc:
            raise RawVNNLibRivalAdapterError(
                f"{name}_snapshot_failed"
            ) from exc
        source_kind = "torch"
        source_dtype = (
            "int64"
            if integer
            else ("float32" if value.dtype == torch.float32 else "float64")
        )
    elif isinstance(value, np.ndarray):
        if value.dtype not in allowed_numpy:
            raise RawVNNLibRivalAdapterError(
                f"{name}_dtype_unsupported"
            )
        shape = tuple(int(item) for item in value.shape)
        if len(shape) != expected_ndim:
            raise RawVNNLibRivalAdapterError(
                f"{name}_rank_mismatch"
            )
        if int(value.nbytes) > _MAX_LIVE_ARRAY_BYTES:
            raise RawVNNLibRivalAdapterError(
                f"{name}_byte_cap_exceeded"
            )
        array = np.ascontiguousarray(value).copy(order="C")
        source_kind = "numpy"
        source_dtype = str(value.dtype)
    else:
        raise RawVNNLibRivalAdapterError(
            f"{name}_must_be_tensor_or_ndarray"
        )

    if integer:
        if array.dtype != np.dtype(np.int64):
            raise RawVNNLibRivalAdapterError(
                f"{name}_dtype_unsupported"
            )
    else:
        if array.dtype not in {
            np.dtype(np.float32),
            np.dtype(np.float64),
        }:
            raise RawVNNLibRivalAdapterError(
                f"{name}_dtype_unsupported"
            )
        if not np.all(np.isfinite(array)):
            raise RawVNNLibRivalAdapterError(f"{name}_nonfinite")

    return _ArraySnapshot(
        array=array,
        source_kind=source_kind,
        source_dtype=source_dtype,
        source_shape=shape,
        source_bytes=array.tobytes(order="C"),
    )


def _hash_raw_array(
    digest: "hashlib._Hash",
    *,
    name: bytes,
    snapshot: _ArraySnapshot,
) -> None:
    for label, payload in (
        (b"name", name),
        (b"source_kind", snapshot.source_kind.encode("ascii")),
        (b"source_dtype", snapshot.source_dtype.encode("ascii")),
        (
            b"shape",
            np.asarray(snapshot.source_shape, dtype=np.int64).tobytes(),
        ),
        (b"raw_bytes", snapshot.source_bytes),
    ):
        digest.update(len(label).to_bytes(8, "little"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)


def _snapshot_live_assert(
    assert_params: Mapping[str, Any],
    *,
    deadline: float,
) -> _LiveAssertSnapshot:
    _check_deadline(deadline, stage="live_assert_begin")
    if not isinstance(assert_params, Mapping):
        raise RawVNNLibRivalAdapterError(
            "live_assert_params_not_mapping"
        )
    required = {"kind", "C", "thresholds", "M", "y_true"}
    keys = set(assert_params.keys())
    if keys != required or any(type(key) is not str for key in keys):
        missing = sorted(required - keys)
        extra = sorted(str(key) for key in keys - required)
        raise RawVNNLibRivalAdapterError(
            "live_assert_params_keyset_mismatch:"
            f"missing={','.join(missing)};extra={','.join(extra)}"
        )

    kind = assert_params["kind"]
    if type(kind) is not str or kind != _TOP1_KIND:
        raise RawVNNLibRivalAdapterError(
            "live_assert_kind_not_top1"
        )
    M_value = assert_params["M"]
    if type(M_value) is not int:
        raise RawVNNLibRivalAdapterError("live_M_not_exact_int")
    M = int(M_value)
    if M < 1 or M + 1 > _MAX_OUTPUT_DIM:
        raise RawVNNLibRivalAdapterError(
            "live_M_out_of_range"
        )

    C_raw = _array_snapshot(
        assert_params["C"], name="live_C", expected_ndim=2
    )
    thresholds_raw = _array_snapshot(
        assert_params["thresholds"],
        name="live_thresholds",
        expected_ndim=2,
    )
    y_true_raw = _array_snapshot(
        assert_params["y_true"],
        name="live_y_true",
        expected_ndim=1,
        integer=True,
    )
    if C_raw.source_kind != thresholds_raw.source_kind or (
        C_raw.source_dtype != thresholds_raw.source_dtype
    ):
        raise RawVNNLibRivalAdapterError(
            "live_float_storage_mismatch"
        )
    if C_raw.source_shape != (M, M + 1):
        raise RawVNNLibRivalAdapterError(
            "live_C_not_complete_top1_shape"
        )
    if thresholds_raw.source_shape != (1, M):
        raise RawVNNLibRivalAdapterError(
            "live_thresholds_not_B1_by_M"
        )
    if y_true_raw.source_shape != (1,):
        raise RawVNNLibRivalAdapterError(
            "live_y_true_must_be_one_integer"
        )
    y_true = int(y_true_raw.array[0])
    if y_true < 0 or y_true >= M + 1:
        raise RawVNNLibRivalAdapterError(
            "live_y_true_out_of_range"
        )

    digest = hashlib.sha256()
    digest.update(_LIVE_ASSERT_SCHEMA.encode("ascii"))
    digest.update(kind.encode("ascii"))
    digest.update(M.to_bytes(8, "little"))
    _hash_raw_array(digest, name=b"C", snapshot=C_raw)
    _hash_raw_array(
        digest, name=b"thresholds", snapshot=thresholds_raw
    )
    _hash_raw_array(
        digest, name=b"y_true", snapshot=y_true_raw
    )
    _check_deadline(deadline, stage="live_assert_hash")
    return _LiveAssertSnapshot(
        kind=kind,
        M=M,
        y_true=y_true,
        C=np.asarray(C_raw.array, dtype=np.float64, order="C"),
        thresholds=np.asarray(
            thresholds_raw.array, dtype=np.float64, order="C"
        ),
        C_source_kind=C_raw.source_kind,
        C_source_dtype=C_raw.source_dtype,
        thresholds_source_kind=thresholds_raw.source_kind,
        thresholds_source_dtype=thresholds_raw.source_dtype,
        y_true_source_kind=y_true_raw.source_kind,
        y_true_source_dtype=y_true_raw.source_dtype,
        digest=digest.hexdigest(),
    )


def _tokenize_exact_sexpr(
    content: str,
    *,
    deadline: float,
) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(content)
    while index < length:
        if (index & 0x3FFF) == 0:
            _check_deadline(deadline, stage="vnnlib_tokenize")
        character = content[index]
        if character.isspace():
            index += 1
            continue
        if character == ";":
            newline = content.find("\n", index)
            index = length if newline < 0 else newline + 1
            continue
        if character in "()":
            tokens.append(character)
            index += 1
        else:
            start = index
            in_brackets = False
            while index < length:
                character = content[index]
                if in_brackets:
                    if character == "]":
                        in_brackets = False
                    index += 1
                    continue
                if character == "[":
                    in_brackets = True
                    index += 1
                    continue
                if (
                    character.isspace()
                    or character in "()"
                    or character == ";"
                ):
                    break
                index += 1
            if in_brackets:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib_bracket_token_unterminated"
                )
            token = content[start:index]
            if not token:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib_empty_token"
                )
            tokens.append(token)
        if len(tokens) > _MAX_SEXPR_TOKENS:
            raise RawVNNLibRivalAdapterError(
                "vnnlib_token_cap_exceeded"
            )
    return tokens


def _parse_all_forms_exact(
    content: str,
    *,
    deadline: float,
) -> list[Any]:
    tokens = _tokenize_exact_sexpr(content, deadline=deadline)
    forms: list[Any] = []
    stack: list[list[Any]] = []
    for offset, token in enumerate(tokens):
        if (offset & 0x3FFF) == 0:
            _check_deadline(deadline, stage="vnnlib_sexpr_parse")
        if token == "(":
            if len(stack) >= _MAX_SEXPR_DEPTH:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib_sexpr_depth_exceeded"
                )
            stack.append([])
        elif token == ")":
            if not stack:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib_unexpected_close_paren"
                )
            completed = stack.pop()
            if stack:
                stack[-1].append(completed)
            else:
                forms.append(completed)
        elif stack:
            stack[-1].append(token)
        else:
            forms.append(token)
    if stack:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_unclosed_paren"
        )
    return forms


def _parse_shape_token(
    token: Any,
    *,
    name: str,
    element_cap: int,
) -> Tuple[int, ...]:
    if not isinstance(token, str):
        raise RawVNNLibRivalAdapterError(
            f"{name}_shape_not_token"
        )
    stripped = token.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise RawVNNLibRivalAdapterError(
            f"{name}_shape_malformed"
        )
    body = stripped[1:-1]
    parts = [part.strip() for part in body.split(",")]
    if (
        not parts
        or len(parts) > _MAX_TENSOR_RANK
        or any(
            not part
            or _CANONICAL_UNSIGNED_INTEGER_RE.fullmatch(part) is None
            for part in parts
        )
    ):
        raise RawVNNLibRivalAdapterError(
            f"{name}_shape_malformed"
        )
    dimensions = tuple(int(part) for part in parts)
    if any(dimension <= 0 for dimension in dimensions):
        raise RawVNNLibRivalAdapterError(
            f"{name}_shape_nonpositive"
        )
    total = 1
    for dimension in dimensions:
        if dimension > element_cap or total > element_cap // dimension:
            raise RawVNNLibRivalAdapterError(
                f"{name}_dimension_exceeds_cap"
            )
        total *= dimension
    return dimensions


def _shape_numel(shape: Sequence[int]) -> int:
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


def _validate_set_logic(form: Sequence[Any]) -> None:
    if len(form) != 2 or form[1] != "QF_LRA":
        raise RawVNNLibRivalAdapterError(
            "set_logic_must_be_QF_LRA"
        )


def _build_symbol_layout(
    forms: Sequence[Any],
    *,
    expected_output_width: int,
    deadline: float,
) -> _SymbolLayout:
    commands: list[str] = []
    for form in forms:
        if not isinstance(form, list) or not form:
            raise RawVNNLibRivalAdapterError(
                "top_level_form_malformed"
            )
        if not isinstance(form[0], str):
            raise RawVNNLibRivalAdapterError(
                "top_level_command_malformed"
            )
        commands.append(form[0])

    has_v2 = any(
        command
        in {
            "vnnlib-version",
            "declare-network",
            "declare-input",
            "declare-output",
        }
        for command in commands
    )
    has_v1 = "declare-const" in commands
    if has_v1 and has_v2:
        raise RawVNNLibRivalAdapterError(
            "mixed_vnnlib_declaration_dialects"
        )
    if not has_v1 and not has_v2:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_declaration_dialect_unrecognized"
        )

    set_logic_count = commands.count("set-logic")
    if set_logic_count > 1:
        raise RawVNNLibRivalAdapterError(
            "set_logic_duplicated"
        )
    if set_logic_count == 1 and commands[0] != "set-logic":
        raise RawVNNLibRivalAdapterError(
            "set_logic_must_precede_all_commands"
        )
    for form in forms:
        if form[0] == "set-logic":
            _validate_set_logic(form)

    if has_v1:
        allowed = {"set-logic", "declare-const", "assert"}
        if any(command not in allowed for command in commands):
            raise RawVNNLibRivalAdapterError(
                "legacy_top_level_command_unsupported"
            )
        symbols: dict[str, Tuple[str, int]] = {}
        declared: dict[str, set[int]] = {"X": set(), "Y": set()}
        assertion_seen = False
        for form in forms:
            _check_deadline(deadline, stage="legacy_declarations")
            if form[0] == "assert":
                if len(form) != 2:
                    raise RawVNNLibRivalAdapterError(
                        "assert_arity_invalid"
                    )
                assertion_seen = True
                continue
            if form[0] != "declare-const":
                continue
            if assertion_seen:
                raise RawVNNLibRivalAdapterError(
                    "legacy_declaration_after_assert"
                )
            if len(form) != 3 or form[2] != "Real":
                raise RawVNNLibRivalAdapterError(
                    "legacy_declaration_must_be_Real"
                )
            name = form[1]
            match = (
                _LEGACY_SYMBOL_RE.fullmatch(name)
                if isinstance(name, str)
                else None
            )
            if match is None:
                raise RawVNNLibRivalAdapterError(
                    "legacy_declaration_name_invalid"
                )
            prefix = match.group(1)
            index = int(match.group(2))
            if name != f"{prefix}_{index}":
                raise RawVNNLibRivalAdapterError(
                    "legacy_declaration_name_not_canonical"
                )
            if name in symbols:
                raise RawVNNLibRivalAdapterError(
                    "legacy_declaration_duplicated"
                )
            symbols[name] = (prefix, index)
            declared[prefix].add(index)

        for prefix, cap in (
            ("X", _MAX_INPUT_DIM),
            ("Y", _MAX_OUTPUT_DIM),
        ):
            indices = declared[prefix]
            if not indices:
                raise RawVNNLibRivalAdapterError(
                    f"legacy_{prefix}_declarations_missing"
                )
            maximum = max(indices)
            if maximum + 1 > cap:
                raise RawVNNLibRivalAdapterError(
                    f"legacy_{prefix}_dimension_exceeds_cap"
                )
            _check_deadline(
                deadline,
                stage=f"legacy_{prefix}_contiguity",
            )
            if len(indices) != maximum + 1:
                raise RawVNNLibRivalAdapterError(
                    f"legacy_{prefix}_declarations_not_contiguous"
                )
        num_inputs = len(declared["X"])
        num_outputs = len(declared["Y"])
        if num_outputs < 2:
            raise RawVNNLibRivalAdapterError(
                "top1_requires_at_least_two_outputs"
            )
        if num_outputs != expected_output_width:
            raise RawVNNLibRivalAdapterError(
                "raw_live_output_width_mismatch"
            )
        return _SymbolLayout(
            dialect="vnnlib-1.0-flat",
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            legacy_symbols=MappingProxyType(symbols),
        )

    allowed = {
        "set-logic",
        "vnnlib-version",
        "declare-network",
        "declare-input",
        "declare-output",
        "assert",
    }
    if any(command not in allowed for command in commands):
        raise RawVNNLibRivalAdapterError(
            "vnnlib2_top_level_command_unsupported"
        )
    for required in (
        "vnnlib-version",
        "declare-network",
        "declare-input",
        "declare-output",
    ):
        if commands.count(required) != 1:
            raise RawVNNLibRivalAdapterError(
                f"vnnlib2_{required.replace('-', '_')}_count_invalid"
            )

    network_name: str | None = None
    input_name: str | None = None
    output_name: str | None = None
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()
    assertion_seen = False
    for form in forms:
        _check_deadline(deadline, stage="vnnlib2_declarations")
        command = form[0]
        if command == "assert":
            if len(form) != 2:
                raise RawVNNLibRivalAdapterError(
                    "assert_arity_invalid"
                )
            assertion_seen = True
        elif command == "vnnlib-version":
            if assertion_seen:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib2_declaration_after_assert"
                )
            if len(form) != 2 or form[1] != "2.0":
                raise RawVNNLibRivalAdapterError(
                    "vnnlib2_version_not_exact"
                )
        elif command == "declare-network":
            if assertion_seen:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib2_declaration_after_assert"
                )
            if (
                len(form) != 2
                or not isinstance(form[1], str)
                or _IDENTIFIER_RE.fullmatch(form[1]) is None
            ):
                raise RawVNNLibRivalAdapterError(
                    "vnnlib2_network_declaration_malformed"
                )
            network_name = form[1]
        elif command in {"declare-input", "declare-output"}:
            if assertion_seen:
                raise RawVNNLibRivalAdapterError(
                    "vnnlib2_declaration_after_assert"
                )
            if (
                len(form) != 4
                or not isinstance(form[1], str)
                or _IDENTIFIER_RE.fullmatch(form[1]) is None
                or form[2] != "Real"
            ):
                raise RawVNNLibRivalAdapterError(
                    f"vnnlib2_{command.replace('-', '_')}_must_be_Real"
                )
            if command == "declare-input":
                input_name = form[1]
                input_shape = _parse_shape_token(
                    form[3],
                    name="vnnlib2_input",
                    element_cap=_MAX_INPUT_DIM,
                )
            else:
                output_name = form[1]
                output_shape = _parse_shape_token(
                    form[3],
                    name="vnnlib2_output",
                    element_cap=_MAX_OUTPUT_DIM,
                )

    if network_name is None or input_name is None or output_name is None:
        raise RawVNNLibRivalAdapterError(
            "vnnlib2_required_declaration_missing"
        )
    if input_name == output_name:
        raise RawVNNLibRivalAdapterError(
            "vnnlib2_input_output_name_collision"
        )
    if network_name in {input_name, output_name}:
        raise RawVNNLibRivalAdapterError(
            "vnnlib2_network_tensor_name_collision"
        )
    num_inputs = _shape_numel(input_shape)
    num_outputs = _shape_numel(output_shape)
    if num_outputs < 2:
        raise RawVNNLibRivalAdapterError(
            "top1_requires_at_least_two_outputs"
        )
    if num_outputs != expected_output_width:
        raise RawVNNLibRivalAdapterError(
            "raw_live_output_width_mismatch"
        )
    return _SymbolLayout(
        dialect="vnnlib-2.0",
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        legacy_symbols=MappingProxyType({}),
        input_name=input_name,
        input_shape=input_shape,
        output_name=output_name,
        output_shape=output_shape,
    )


def _ravel_index(
    indices: Sequence[int],
    shape: Sequence[int],
) -> int:
    if len(indices) != len(shape):
        raise RawVNNLibRivalAdapterError(
            "vnnlib2_variable_index_rank_mismatch"
        )
    flat = 0
    for index, dimension in zip(indices, shape):
        if index < 0 or index >= dimension:
            raise RawVNNLibRivalAdapterError(
                "vnnlib2_variable_index_out_of_bounds"
            )
        flat = flat * int(dimension) + int(index)
    return flat


def _resolve_symbol(
    token: str,
    layout: _SymbolLayout,
) -> Tuple[str, int] | None:
    if layout.dialect == "vnnlib-1.0-flat":
        resolved = layout.legacy_symbols.get(token)
        if resolved is not None:
            return resolved
        if _LEGACY_SYMBOL_RE.fullmatch(token) is not None:
            raise RawVNNLibRivalAdapterError(
                "legacy_assert_references_undeclared_symbol"
            )
        if "[" in token or "]" in token:
            raise RawVNNLibRivalAdapterError(
                "legacy_assert_uses_bracket_symbol"
            )
        return None

    match = _INDEXED_SYMBOL_RE.fullmatch(token)
    if match is None:
        if _LEGACY_SYMBOL_RE.fullmatch(token) is not None:
            raise RawVNNLibRivalAdapterError(
                "vnnlib2_flat_alias_forbidden"
            )
        if "[" in token or "]" in token:
            raise RawVNNLibRivalAdapterError(
                "vnnlib2_indexed_symbol_malformed"
            )
        return None
    name = match.group(1)
    raw_indices = [part.strip() for part in match.group(2).split(",")]
    if not raw_indices or any(
        not part
        or _CANONICAL_UNSIGNED_INTEGER_RE.fullmatch(part) is None
        for part in raw_indices
    ):
        raise RawVNNLibRivalAdapterError(
            "vnnlib2_variable_index_malformed"
        )
    indices = tuple(int(part) for part in raw_indices)
    if name == layout.input_name:
        return "X", _ravel_index(indices, layout.input_shape)
    if name == layout.output_name:
        return "Y", _ravel_index(indices, layout.output_shape)
    raise RawVNNLibRivalAdapterError(
        "vnnlib2_assert_references_undeclared_symbol"
    )


def _parse_fraction_token(token: str) -> Fraction | None:
    if _NUMERIC_RE.fullmatch(token) is None:
        return None
    if len(token.encode("utf-8")) > _MAX_NUMERIC_TOKEN_BYTES:
        raise RawVNNLibRivalAdapterError(
            "numeric_token_too_long"
        )
    exponent_match = re.search(r"[eE]([+-]?)([0-9]+)\Z", token)
    if exponent_match is not None:
        exponent_digits = exponent_match.group(2)
        if len(exponent_digits) > 6:
            raise RawVNNLibRivalAdapterError(
                "numeric_exponent_out_of_range"
            )
        exponent = int(exponent_digits)
        if exponent > _MAX_ABS_DECIMAL_EXPONENT:
            raise RawVNNLibRivalAdapterError(
                "numeric_exponent_out_of_range"
            )
    try:
        return _checked_fraction(Fraction(token))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RawVNNLibRivalAdapterError(
            "numeric_token_invalid"
        ) from exc


def _checked_fraction(value: Fraction) -> Fraction:
    if (
        value.numerator.bit_length() > _MAX_FRACTION_BITS
        or value.denominator.bit_length() > _MAX_FRACTION_BITS
    ):
        raise RawVNNLibRivalAdapterError(
            "exact_fraction_bit_cap_exceeded"
        )
    return value


def _clean_coefficients(
    coefficients: Mapping[int, Fraction],
) -> dict[int, Fraction]:
    return {
        int(index): value
        for index, value in coefficients.items()
        if value != 0
    }


def _affine_add(
    left: _ExactAffine,
    right: _ExactAffine,
) -> _ExactAffine:
    x = dict(left.x)
    y = dict(left.y)
    for index, value in right.x.items():
        x[index] = _checked_fraction(
            x.get(index, Fraction(0)) + value
        )
    for index, value in right.y.items():
        y[index] = _checked_fraction(
            y.get(index, Fraction(0)) + value
        )
    return _ExactAffine(
        x=_clean_coefficients(x),
        y=_clean_coefficients(y),
        constant=_checked_fraction(left.constant + right.constant),
    )


def _affine_scale(
    value: _ExactAffine,
    scalar: Fraction,
) -> _ExactAffine:
    return _ExactAffine(
        x=_clean_coefficients(
            {
                index: _checked_fraction(scalar * item)
                for index, item in value.x.items()
            }
        ),
        y=_clean_coefficients(
            {
                index: _checked_fraction(scalar * item)
                for index, item in value.y.items()
            }
        ),
        constant=_checked_fraction(scalar * value.constant),
    )


def _affine_has_variables(value: _ExactAffine) -> bool:
    return bool(value.x or value.y)


def _bump_affine_budget(
    budget: list[int],
    *,
    deadline: float,
) -> None:
    budget[0] += 1
    if budget[0] > _MAX_AFFINE_NODES:
        raise RawVNNLibRivalAdapterError(
            "affine_expression_node_cap_exceeded"
        )
    if (budget[0] & 0x3FF) == 0:
        _check_deadline(deadline, stage="exact_affine_parse")


def _parse_exact_affine(
    expression: Any,
    *,
    layout: _SymbolLayout,
    deadline: float,
    budget: list[int],
) -> _ExactAffine:
    _bump_affine_budget(budget, deadline=deadline)
    if isinstance(expression, str):
        resolved = _resolve_symbol(expression, layout)
        if resolved is not None:
            kind, index = resolved
            return _ExactAffine(
                x={index: Fraction(1)} if kind == "X" else {},
                y={index: Fraction(1)} if kind == "Y" else {},
                constant=Fraction(0),
            )
        numeric = _parse_fraction_token(expression)
        if numeric is None:
            raise RawVNNLibRivalAdapterError(
                "affine_expression_unknown_symbol"
            )
        return _ExactAffine(x={}, y={}, constant=numeric)
    if not isinstance(expression, list) or not expression:
        raise RawVNNLibRivalAdapterError(
            "affine_expression_malformed"
        )
    operator = expression[0]
    if not isinstance(operator, str):
        raise RawVNNLibRivalAdapterError(
            "affine_operator_malformed"
        )
    operands = expression[1:]
    if operator == "+":
        if len(operands) < 2:
            raise RawVNNLibRivalAdapterError(
                "affine_plus_arity_invalid"
            )
        result = _ExactAffine(x={}, y={}, constant=Fraction(0))
        for operand in operands:
            result = _affine_add(
                result,
                _parse_exact_affine(
                    operand,
                    layout=layout,
                    deadline=deadline,
                    budget=budget,
                ),
            )
        return result
    if operator == "-":
        if len(operands) not in {1, 2}:
            raise RawVNNLibRivalAdapterError(
                "affine_minus_arity_invalid"
            )
        left = _parse_exact_affine(
            operands[0],
            layout=layout,
            deadline=deadline,
            budget=budget,
        )
        if len(operands) == 1:
            return _affine_scale(left, Fraction(-1))
        right = _parse_exact_affine(
            operands[1],
            layout=layout,
            deadline=deadline,
            budget=budget,
        )
        return _affine_add(left, _affine_scale(right, Fraction(-1)))
    if operator == "*":
        if len(operands) != 2:
            raise RawVNNLibRivalAdapterError(
                "affine_multiply_arity_invalid"
            )
        left = _parse_exact_affine(
            operands[0],
            layout=layout,
            deadline=deadline,
            budget=budget,
        )
        right = _parse_exact_affine(
            operands[1],
            layout=layout,
            deadline=deadline,
            budget=budget,
        )
        if _affine_has_variables(left) and _affine_has_variables(right):
            raise RawVNNLibRivalAdapterError(
                "affine_nonlinear_product"
            )
        if _affine_has_variables(left):
            return _affine_scale(left, right.constant)
        return _affine_scale(right, left.constant)
    if operator == "/":
        if len(operands) != 2:
            raise RawVNNLibRivalAdapterError(
                "affine_divide_arity_invalid"
            )
        numerator = _parse_exact_affine(
            operands[0],
            layout=layout,
            deadline=deadline,
            budget=budget,
        )
        denominator = _parse_exact_affine(
            operands[1],
            layout=layout,
            deadline=deadline,
            budget=budget,
        )
        if _affine_has_variables(denominator):
            raise RawVNNLibRivalAdapterError(
                "affine_nonlinear_divisor"
            )
        if denominator.constant == 0:
            raise RawVNNLibRivalAdapterError(
                "affine_division_by_zero"
            )
        return _affine_scale(
            numerator, Fraction(1, 1) / denominator.constant
        )
    raise RawVNNLibRivalAdapterError(
        "affine_operator_unsupported"
    )


def _canonical_inequality_exact(
    atom: Any,
    *,
    layout: _SymbolLayout,
    deadline: float,
) -> Tuple[Mapping[int, Fraction], Mapping[int, Fraction], Fraction]:
    if (
        not isinstance(atom, list)
        or len(atom) != 3
        or atom[0] not in {"<=", ">="}
    ):
        raise RawVNNLibRivalAdapterError(
            "output_branch_not_nonstrict_affine_atom"
        )
    budget = [0]
    left = _parse_exact_affine(
        atom[1], layout=layout, deadline=deadline, budget=budget
    )
    right = _parse_exact_affine(
        atom[2], layout=layout, deadline=deadline, budget=budget
    )
    difference = _affine_add(
        left, _affine_scale(right, Fraction(-1))
    )
    threshold = -difference.constant
    x = dict(difference.x)
    y = dict(difference.y)
    if atom[0] == ">=":
        x = {
            index: -value for index, value in x.items()
        }
        y = {
            index: -value for index, value in y.items()
        }
        threshold = -threshold
    return (
        MappingProxyType(_clean_coefficients(x)),
        MappingProxyType(_clean_coefficients(y)),
        threshold,
    )


def _collect_variable_kinds(
    expression: Any,
    *,
    layout: _SymbolLayout,
    deadline: float,
    budget: list[int],
) -> set[str]:
    _bump_affine_budget(budget, deadline=deadline)
    if isinstance(expression, str):
        resolved = _resolve_symbol(expression, layout)
        if resolved is not None:
            return {resolved[0]}
        if _parse_fraction_token(expression) is not None:
            return set()
        raise RawVNNLibRivalAdapterError(
            "assert_expression_unknown_symbol"
        )
    if not isinstance(expression, list) or not expression:
        raise RawVNNLibRivalAdapterError(
            "assert_expression_malformed"
        )
    operator = expression[0]
    if (
        not isinstance(operator, str)
        or operator not in _BOOLEAN_OR_ARITHMETIC_OPERATORS
    ):
        raise RawVNNLibRivalAdapterError(
            "assert_operator_unsupported"
        )
    kinds: set[str] = set()
    for child in expression[1:]:
        kinds.update(
            _collect_variable_kinds(
                child,
                layout=layout,
                deadline=deadline,
                budget=budget,
            )
        )
    return kinds


def _is_simple_input_bound(
    body: Any,
    *,
    layout: _SymbolLayout,
    deadline: float,
) -> bool:
    try:
        x, y, _threshold = _canonical_inequality_exact(
            body, layout=layout, deadline=deadline
        )
    except RawVNNLibRivalAdapterError:
        return False
    return len(x) == 1 and not y


def _canonical_atom(atom: Any) -> str:
    return _canonical_bytes(atom).decode("ascii")


def _unwrap_output_branch(
    branch: Any,
    *,
    branch_index: int,
) -> Tuple[Any, Tuple[int, ...]]:
    path = [int(branch_index)]
    node = branch
    while isinstance(node, list) and node and node[0] == "and":
        if len(node) != 2:
            raise RawVNNLibRivalAdapterError(
                "output_branch_has_extra_conjunct"
            )
        node = node[1]
        path.append(0)
    if isinstance(node, list) and node and node[0] in {"and", "or"}:
        raise RawVNNLibRivalAdapterError(
            "output_branch_nested_boolean_unsupported"
        )
    return node, tuple(path)


def _recognize_raw_top1(
    source_payload: bytes,
    *,
    source_sha256: str,
    expected_output_width: int,
    deadline: float,
) -> Tuple[str, int, int, Tuple[_RawAtom, ...]]:
    try:
        content = source_payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RawVNNLibRivalAdapterError(
            "vnnlib_source_not_utf8"
        ) from exc
    forms = _parse_all_forms_exact(content, deadline=deadline)
    layout = _build_symbol_layout(
        forms,
        expected_output_width=expected_output_width,
        deadline=deadline,
    )

    output_assertions: list[Tuple[int, Any]] = []
    assert_ordinal = 0
    for form in forms:
        if form[0] != "assert":
            continue
        body = form[1]
        kinds = _collect_variable_kinds(
            body, layout=layout, deadline=deadline, budget=[0]
        )
        if "Y" in kinds:
            output_assertions.append((assert_ordinal, body))
        elif "X" in kinds and _is_simple_input_bound(
            body, layout=layout, deadline=deadline
        ):
            pass
        elif "X" in kinds:
            raise RawVNNLibRivalAdapterError(
                "input_assert_not_simple_scalar_bound"
            )
        else:
            raise RawVNNLibRivalAdapterError(
                "assert_has_no_declared_variable"
            )
        assert_ordinal += 1

    if not output_assertions:
        raise RawVNNLibRivalAdapterError(
            "input_only_property_has_no_raw_rivals"
        )
    if len(output_assertions) != 1:
        raise RawVNNLibRivalAdapterError(
            "requires_exactly_one_output_assert"
        )
    output_assert_ordinal, output_body = output_assertions[0]
    if (
        not isinstance(output_body, list)
        or len(output_body) < 2
        or output_body[0] != "or"
    ):
        raise RawVNNLibRivalAdapterError(
            "output_assert_root_not_or"
        )

    atoms: list[_RawAtom] = []
    inferred_true: int | None = None
    seen_competitors: set[int] = set()
    for branch_index, raw_branch in enumerate(output_body[1:]):
        _check_deadline(deadline, stage="top1_branch_recognition")
        atom, boolean_path = _unwrap_output_branch(
            raw_branch, branch_index=branch_index
        )
        x_coefficients, y_coefficients, threshold = (
            _canonical_inequality_exact(
                atom, layout=layout, deadline=deadline
            )
        )
        if x_coefficients:
            raise RawVNNLibRivalAdapterError(
                "output_atom_references_x"
            )
        if threshold != 0:
            raise RawVNNLibRivalAdapterError(
                "classification_atom_threshold_not_zero"
            )
        if len(y_coefficients) != 2:
            raise RawVNNLibRivalAdapterError(
                "classification_atom_not_two_class"
            )
        positive = [
            int(index)
            for index, value in y_coefficients.items()
            if value == 1
        ]
        negative = [
            int(index)
            for index, value in y_coefficients.items()
            if value == -1
        ]
        if (
            len(positive) != 1
            or len(negative) != 1
            or len(positive) + len(negative) != len(y_coefficients)
        ):
            raise RawVNNLibRivalAdapterError(
                "classification_atom_not_unit_margin"
            )
        true_class = positive[0]
        competitor = negative[0]
        if inferred_true is None:
            inferred_true = true_class
        elif true_class != inferred_true:
            raise RawVNNLibRivalAdapterError(
                "classification_atoms_disagree_on_true_class"
            )
        if competitor in seen_competitors:
            raise RawVNNLibRivalAdapterError(
                "classification_competitor_duplicated"
            )
        seen_competitors.add(competitor)
        atoms.append(
            _RawAtom(
                assert_ordinal=int(output_assert_ordinal),
                boolean_path=boolean_path,
                canonical_atom=_canonical_atom(atom),
                true_class=true_class,
                competitor_class=competitor,
            )
        )

    if inferred_true is None:
        raise RawVNNLibRivalAdapterError(
            "classification_true_class_missing"
        )
    expected = set(range(layout.num_outputs)) - {int(inferred_true)}
    if seen_competitors != expected:
        raise RawVNNLibRivalAdapterError(
            "classification_competitor_coverage_incomplete"
        )
    if len(atoms) != layout.num_outputs - 1:
        raise RawVNNLibRivalAdapterError(
            "classification_rival_count_mismatch"
        )
    if not _valid_sha256(source_sha256):
        raise RawVNNLibRivalAdapterError(
            "internal_source_sha256_malformed"
        )
    return (
        layout.dialect,
        layout.num_outputs,
        int(inferred_true),
        tuple(atoms),
    )


def _assert_digest(
    *,
    vnnlib_sha256: str,
    dialect: str,
    atom: _RawAtom,
) -> str:
    return _canonical_sha256(
        {
            "schema": _ASSERT_DIGEST_SCHEMA,
            "vnnlib_sha256": vnnlib_sha256,
            "dialect": dialect,
            "assert_ordinal": int(atom.assert_ordinal),
            "boolean_path": list(atom.boolean_path),
            "canonical_atom": atom.canonical_atom,
            "transform": _TRANSFORM,
            "true_class": int(atom.true_class),
            "competitor_class": int(atom.competitor_class),
        }
    )


def _expected_rows(
    *,
    atoms: Sequence[_RawAtom],
    num_outputs: int,
    true_class: int,
    vnnlib_sha256: str,
    dialect: str,
) -> Tuple[
    Tuple[RawVNNLibTop1Row, ...],
    Tuple[RivalSpec, ...],
    np.ndarray,
    np.ndarray,
]:
    by_competitor = {
        int(atom.competitor_class): atom for atom in atoms
    }
    competitors = sorted(by_competitor)
    C = np.zeros((len(competitors), num_outputs), dtype=np.float64)
    thresholds = np.zeros((1, len(competitors)), dtype=np.float64)
    rows: list[RawVNNLibTop1Row] = []
    rivals: list[RivalSpec] = []
    for encoded_row, competitor in enumerate(competitors):
        C[encoded_row, competitor] = 1.0
        C[encoded_row, true_class] = -1.0
        atom = by_competitor[competitor]
        assert_digest = _assert_digest(
            vnnlib_sha256=vnnlib_sha256,
            dialect=dialect,
            atom=atom,
        )
        rival = RivalSpec(
            rival_id=int(competitor),
            objective=tuple(float(value) for value in C[encoded_row]),
            threshold=0.0,
            assert_digest=assert_digest,
        )
        binding = rival_spec_binding_digest(rival)
        rivals.append(rival)
        rows.append(
            RawVNNLibTop1Row(
                encoded_row=int(encoded_row),
                competitor_class=int(competitor),
                assert_ordinal=int(atom.assert_ordinal),
                boolean_path=tuple(atom.boolean_path),
                canonical_atom=str(atom.canonical_atom),
                transform=_TRANSFORM,
                assert_digest=assert_digest,
                rival_binding_sha256=binding,
            )
        )
    return tuple(rows), tuple(rivals), C, thresholds


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _row_payload(row: RawVNNLibTop1Row) -> Mapping[str, Any]:
    return {
        "encoded_row": int(row.encoded_row),
        "competitor_class": int(row.competitor_class),
        "assert_ordinal": int(row.assert_ordinal),
        "boolean_path": list(row.boolean_path),
        "canonical_atom": row.canonical_atom,
        "transform": row.transform,
        "assert_digest": row.assert_digest,
        "rival_binding_sha256": row.rival_binding_sha256,
    }


def _candidate_content_payload(
    *,
    source_path: str,
    source_identity: _SourceIdentity,
    vnnlib_sha256: str,
    dialect: str,
    true_class: int,
    output_assert_ordinal: int,
    rows: Sequence[RawVNNLibTop1Row],
    live_assert_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema": _SCHEMA,
        "source_path": source_path,
        "source_identity": {
            "device": int(source_identity.device),
            "inode": int(source_identity.inode),
            "size": int(source_identity.size),
            "mtime_ns": int(source_identity.mtime_ns),
        },
        "vnnlib_sha256": vnnlib_sha256,
        "dialect": dialect,
        "true_class": int(true_class),
        "output_assert_ordinal": int(output_assert_ordinal),
        "live_assert_sha256": live_assert_sha256,
        "rows": [_row_payload(row) for row in rows],
    }


def _candidate_source_identity(
    candidate: RawVNNLibTop1Candidate,
) -> _SourceIdentity:
    return _SourceIdentity(
        device=int(candidate.source_device),
        inode=int(candidate.source_inode),
        size=int(candidate.source_size),
        mtime_ns=int(candidate.source_mtime_ns),
    )


def _candidate_content_digest(
    candidate: RawVNNLibTop1Candidate,
) -> str:
    return _canonical_sha256(
        _candidate_content_payload(
            source_path=candidate.source_path,
            source_identity=_candidate_source_identity(candidate),
            vnnlib_sha256=candidate.vnnlib_sha256,
            dialect=candidate.dialect,
            true_class=candidate.true_class,
            output_assert_ordinal=candidate.output_assert_ordinal,
            rows=candidate.rows,
            live_assert_sha256=candidate.live_assert_sha256,
        )
    )


def _candidate_runtime_types_exact(
    candidate: RawVNNLibTop1Candidate,
) -> bool:
    if (
        type(candidate) is not RawVNNLibTop1Candidate
        or type(candidate.source_path) is not str
        or type(candidate.source_device) is not int
        or type(candidate.source_inode) is not int
        or type(candidate.source_size) is not int
        or type(candidate.source_mtime_ns) is not int
        or type(candidate.vnnlib_sha256) is not str
        or type(candidate.dialect) is not str
        or type(candidate.true_class) is not int
        or type(candidate.output_assert_ordinal) is not int
        or type(candidate.rows) is not tuple
        or type(candidate.live_assert_sha256) is not str
        or type(candidate.candidate_sha256) is not str
        or type(candidate.proof_authority) is not bool
    ):
        return False
    for row in candidate.rows:
        if (
            type(row) is not RawVNNLibTop1Row
            or type(row.encoded_row) is not int
            or type(row.competitor_class) is not int
            or type(row.assert_ordinal) is not int
            or type(row.boolean_path) is not tuple
            or any(type(item) is not int for item in row.boolean_path)
            or type(row.canonical_atom) is not str
            or type(row.transform) is not str
            or type(row.assert_digest) is not str
            or type(row.rival_binding_sha256) is not str
        ):
            return False
    return True


def _candidate_receipt_payload(
    *,
    content_payload: Mapping[str, Any],
    candidate_sha256: str,
) -> Mapping[str, Any]:
    return {
        **content_payload,
        "status": "raw_live_top1_match_candidate",
        "candidate_only": True,
        "proof_authority": False,
        "stable_rival_id_authority": "competitor_class_index",
        "live_boundary": (
            "weak_owner_exact_identity_single_use_ttl_cap_"
            "source_fd_identity_rehash_live_raw_bytes"
        ),
        "usable_rivals_exposed": False,
        "candidate_sha256": candidate_sha256,
    }


def _batch_content_payload(
    *,
    candidate: RawVNNLibTop1Candidate,
    rivals: Sequence[RivalSpec],
) -> Mapping[str, Any]:
    return {
        "schema": _CONSUMED_BATCH_SCHEMA,
        "source_path": candidate.source_path,
        "vnnlib_sha256": candidate.vnnlib_sha256,
        "dialect": candidate.dialect,
        "true_class": int(candidate.true_class),
        "output_assert_ordinal": int(candidate.output_assert_ordinal),
        "live_assert_sha256": candidate.live_assert_sha256,
        "candidate_sha256": candidate.candidate_sha256,
        "rivals": [
            {
                "rival_id": int(rival.rival_id),
                "binding_sha256": rival_spec_binding_digest(rival),
            }
            for rival in rivals
        ],
    }


def _batch_content_payload_from_batch(
    batch: ConsumedRivalBatch,
) -> Mapping[str, Any]:
    return {
        "schema": _CONSUMED_BATCH_SCHEMA,
        "source_path": batch.source_path,
        "vnnlib_sha256": batch.vnnlib_sha256,
        "dialect": batch.dialect,
        "true_class": int(batch.true_class),
        "output_assert_ordinal": int(batch.output_assert_ordinal),
        "live_assert_sha256": batch.live_assert_sha256,
        "candidate_sha256": batch.candidate_sha256,
        "rivals": [
            {
                "rival_id": int(rival.rival_id),
                "binding_sha256": rival_spec_binding_digest(rival),
            }
            for rival in batch._rivals
        ],
    }


def _batch_runtime_types_exact(batch: ConsumedRivalBatch) -> bool:
    return (
        type(batch) is ConsumedRivalBatch
        and type(batch.source_path) is str
        and type(batch.vnnlib_sha256) is str
        and type(batch.dialect) is str
        and type(batch.true_class) is int
        and type(batch.output_assert_ordinal) is int
        and type(batch._rivals) is tuple
        and all(type(rival) is RivalSpec for rival in batch._rivals)
        and type(batch.live_assert_sha256) is str
        and type(batch.candidate_sha256) is str
        and type(batch.batch_sha256) is str
        and type(batch.proof_authority) is bool
    )


def _batch_receipt_payload(
    *,
    content_payload: Mapping[str, Any],
    batch_sha256: str,
) -> Mapping[str, Any]:
    return {
        **content_payload,
        "status": "raw_live_top1_candidate_consumed",
        "candidate_only": True,
        "proof_authority": False,
        "owner_bound_live_identity": True,
        "batch_sha256": batch_sha256,
    }


def _batch_content_digest(batch: ConsumedRivalBatch) -> str:
    return _canonical_sha256(_batch_content_payload_from_batch(batch))


def _exact_rival_bindings(
    rivals: Any,
) -> Tuple[Tuple[int, str], ...] | None:
    if type(rivals) is not tuple or any(
        type(rival) is not RivalSpec for rival in rivals
    ):
        return None
    try:
        return tuple(
            (int(rival.rival_id), rival_spec_binding_digest(rival))
            for rival in rivals
        )
    except Exception:
        return None


def _purge_live_records_locked(now: float) -> None:
    stale = [
        key
        for key, record in _LIVE_RECORDS.items()
        if type(record) is not _LiveRecord
        or record.process_id != os.getpid()
        or record.expires_monotonic <= now
        or record.candidate_ref() is None
    ]
    for key in stale:
        _LIVE_RECORDS.pop(key, None)


def _discard_live_record(capability_id: int) -> None:
    with _LIVE_LOCK:
        _LIVE_RECORDS.pop(capability_id, None)


def _register_live_record(
    *,
    candidate: RawVNNLibTop1Candidate,
    capability: _LiveCapability,
    rivals: Tuple[RivalSpec, ...],
    source_path: Path,
    source_identity: _SourceIdentity,
    expected_sha256: str,
    live_assert_sha256: str,
    candidate_sha256: str,
    candidate_receipt_sha256: str,
    ttl_seconds: float,
) -> None:
    rival_bindings = _exact_rival_bindings(rivals)
    if (
        type(candidate) is not RawVNNLibTop1Candidate
        or type(capability) is not _LiveCapability
        or rival_bindings is None
        or not _valid_sha256(candidate_sha256)
        or not _valid_sha256(candidate_receipt_sha256)
    ):
        raise RawVNNLibRivalAdapterError(
            "live_candidate_registration_snapshot_invalid"
        )
    issued = time.monotonic()
    capability_id = id(capability)
    candidate_ref = weakref.ref(
        candidate,
        lambda _reference, key=capability_id: _discard_live_record(key),
    )
    record = _LiveRecord(
        capability=capability,
        candidate_ref=candidate_ref,
        rivals=rivals,
        issued_rivals=rivals,
        candidate_receipt=candidate.receipt,
        source_path=source_path,
        source_identity=source_identity,
        expected_sha256=expected_sha256,
        live_assert_sha256=live_assert_sha256,
        candidate_sha256=candidate_sha256,
        candidate_receipt_sha256=candidate_receipt_sha256,
        rival_bindings=rival_bindings,
        process_id=os.getpid(),
        issued_monotonic=issued,
        expires_monotonic=issued + ttl_seconds,
    )
    with _LIVE_LOCK:
        _purge_live_records_locked(issued)
        if len(_LIVE_RECORDS) >= _MAX_LIVE_RECORDS:
            raise RawVNNLibRivalAdapterError(
                "live_candidate_registry_capacity_exceeded"
            )
        _LIVE_RECORDS[capability_id] = record


def issue_raw_vnnlib_top1_candidate(
    vnnlib_path: str | Path,
    *,
    expected_vnnlib_sha256: str,
    live_assert_params: Mapping[str, Any],
    deadline: float | None = None,
    capability_ttl_seconds: float = _DEFAULT_CAPABILITY_TTL_SECONDS,
) -> RawVNNLibTop1Candidate:
    """Issue metadata plus an expiring process-local capability.

    ``deadline`` is an absolute :func:`time.monotonic` deadline.  A bounded
    default is installed when omitted.
    """

    operation_deadline = _normalize_deadline(deadline)
    ttl_seconds = _normalize_ttl(capability_ttl_seconds)
    path, source_payload, source_sha256, source_identity = (
        _read_bound_source(
            vnnlib_path,
            expected_vnnlib_sha256,
            deadline=operation_deadline,
        )
    )
    live = _snapshot_live_assert(
        live_assert_params, deadline=operation_deadline
    )
    dialect, num_outputs, true_class, atoms = _recognize_raw_top1(
        source_payload,
        source_sha256=source_sha256,
        expected_output_width=int(live.C.shape[1]),
        deadline=operation_deadline,
    )
    if live.M != num_outputs - 1:
        raise RawVNNLibRivalAdapterError(
            "raw_live_rival_count_mismatch"
        )
    if live.y_true != true_class:
        raise RawVNNLibRivalAdapterError(
            "raw_live_true_class_mismatch"
        )
    rows, rivals, expected_C, expected_thresholds = _expected_rows(
        atoms=atoms,
        num_outputs=num_outputs,
        true_class=true_class,
        vnnlib_sha256=source_sha256,
        dialect=dialect,
    )
    if not _bitwise_equal(live.C, expected_C):
        raise RawVNNLibRivalAdapterError(
            "raw_live_C_bit_mismatch"
        )
    if not _bitwise_equal(live.thresholds, expected_thresholds):
        raise RawVNNLibRivalAdapterError(
            "raw_live_threshold_bit_mismatch"
        )
    output_assert_ordinal = rows[0].assert_ordinal
    content_payload = _candidate_content_payload(
        source_path=str(path),
        source_identity=source_identity,
        vnnlib_sha256=source_sha256,
        dialect=dialect,
        true_class=true_class,
        output_assert_ordinal=output_assert_ordinal,
        rows=rows,
        live_assert_sha256=live.digest,
    )
    candidate_sha256 = _canonical_sha256(content_payload)
    receipt_payload = _candidate_receipt_payload(
        content_payload=content_payload,
        candidate_sha256=candidate_sha256,
    )
    candidate_receipt_sha256 = _canonical_sha256(receipt_payload)
    receipt = _deep_freeze(receipt_payload)
    capability = _LiveCapability(_CAPABILITY_SENTINEL)
    candidate = RawVNNLibTop1Candidate(
        source_path=str(path),
        source_device=source_identity.device,
        source_inode=source_identity.inode,
        source_size=source_identity.size,
        source_mtime_ns=source_identity.mtime_ns,
        vnnlib_sha256=source_sha256,
        dialect=dialect,
        true_class=true_class,
        output_assert_ordinal=output_assert_ordinal,
        rows=rows,
        live_assert_sha256=live.digest,
        candidate_sha256=candidate_sha256,
        receipt=receipt,
        _live_capability=capability,
    )
    _register_live_record(
        candidate=candidate,
        capability=capability,
        rivals=rivals,
        source_path=path,
        source_identity=source_identity,
        expected_sha256=source_sha256,
        live_assert_sha256=live.digest,
        candidate_sha256=candidate_sha256,
        candidate_receipt_sha256=candidate_receipt_sha256,
        ttl_seconds=ttl_seconds,
    )
    _check_deadline(operation_deadline, stage="candidate_registration")
    return candidate


def revoke_raw_vnnlib_top1_candidate(
    candidate: RawVNNLibTop1Candidate,
) -> bool:
    """Explicitly revoke an unconsumed exact candidate."""

    if type(candidate) is not RawVNNLibTop1Candidate:
        raise RawVNNLibRivalAdapterError("candidate_wrong_type")
    capability = candidate._live_capability
    with _LIVE_LOCK:
        _purge_live_records_locked(time.monotonic())
        record = _LIVE_RECORDS.get(id(capability))
        if record is None:
            return False
        if (
            type(record) is not _LiveRecord
            or type(capability) is not _LiveCapability
            or record.capability is not capability
        ):
            raise RawVNNLibRivalAdapterError(
                "candidate_live_capability_mismatch"
            )
        if record.candidate_ref() is not candidate:
            raise RawVNNLibRivalAdapterError(
                "candidate_live_identity_mismatch"
            )
        _LIVE_RECORDS.pop(id(capability), None)
        return True


def _build_consumed_batch(
    *,
    candidate: RawVNNLibTop1Candidate,
    rivals: Tuple[RivalSpec, ...],
) -> ConsumedRivalBatch:
    payload = _batch_content_payload(candidate=candidate, rivals=rivals)
    batch_sha256 = _canonical_sha256(payload)
    receipt_payload = _batch_receipt_payload(
        content_payload=payload,
        batch_sha256=batch_sha256,
    )
    receipt_sha256 = _canonical_sha256(receipt_payload)
    rival_bindings = _exact_rival_bindings(rivals)
    if rival_bindings is None:
        raise RawVNNLibRivalAdapterError(
            "consumed_rival_snapshot_invalid"
        )
    identity = _ConsumedBatchIdentity(_BATCH_IDENTITY_SENTINEL)
    batch = ConsumedRivalBatch(
        source_path=candidate.source_path,
        vnnlib_sha256=candidate.vnnlib_sha256,
        dialect=candidate.dialect,
        true_class=candidate.true_class,
        output_assert_ordinal=candidate.output_assert_ordinal,
        _rivals=rivals,
        live_assert_sha256=candidate.live_assert_sha256,
        candidate_sha256=candidate.candidate_sha256,
        batch_sha256=batch_sha256,
        receipt=_deep_freeze(receipt_payload),
        _live_identity=identity,
    )
    identity.bind_owner(
        batch,
        batch_sha256=batch_sha256,
        receipt_sha256=receipt_sha256,
        receipt=batch.receipt,
        rivals=rivals,
        rival_bindings=rival_bindings,
    )
    return batch


def validate_consumed_raw_vnnlib_rival_batch(
    batch: ConsumedRivalBatch,
) -> bool:
    """Check that ``batch`` is the exact live owner for this process."""

    if not _batch_runtime_types_exact(batch):
        return False
    identity = batch._live_identity
    if (
        type(identity) is not _ConsumedBatchIdentity
        or not identity.owns(batch)
        or batch.proof_authority is not False
        or not _valid_sha256(batch.batch_sha256)
    ):
        return False
    try:
        rival_bindings = _exact_rival_bindings(batch._rivals)
        if rival_bindings is None:
            return False
        payload = _batch_content_payload_from_batch(batch)
        observed_batch_sha256 = _canonical_sha256(payload)
        receipt_payload = _batch_receipt_payload(
            content_payload=payload,
            batch_sha256=observed_batch_sha256,
        )
        receipt_sha256 = _canonical_sha256(receipt_payload)
        return (
            observed_batch_sha256 == batch.batch_sha256
            and identity.matches_issued_snapshot(
                batch_sha256=observed_batch_sha256,
                receipt_sha256=receipt_sha256,
                receipt=batch.receipt,
                rivals=batch._rivals,
                rival_bindings=rival_bindings,
            )
            and _receipt_matches_exact(
                batch.receipt,
                expected_payload=receipt_payload,
                expected_sha256=receipt_sha256,
            )
        )
    except Exception:
        return False


def consume_raw_vnnlib_top1_candidate(
    candidate: RawVNNLibTop1Candidate,
    *,
    live_assert_params: Mapping[str, Any],
    deadline: float | None = None,
) -> ConsumedRivalBatch:
    """Consume one exact live candidate and return an owner-bound batch.

    A copied candidate is rejected without revoking the original.  Once the
    exact candidate starts source/live validation, its capability is consumed
    even when subsequent validation fails.
    """

    operation_deadline = _normalize_deadline(deadline)
    if type(candidate) is not RawVNNLibTop1Candidate:
        raise RawVNNLibRivalAdapterError("candidate_wrong_type")
    capability = candidate._live_capability
    with _LIVE_LOCK:
        _purge_live_records_locked(time.monotonic())
        record = _LIVE_RECORDS.get(id(capability))
        if (
            record is None
            or type(record) is not _LiveRecord
            or type(capability) is not _LiveCapability
            or record.capability is not capability
        ):
            raise RawVNNLibRivalAdapterError(
                "candidate_live_capability_missing_consumed_or_expired"
            )
        if record.candidate_ref() is not candidate:
            raise RawVNNLibRivalAdapterError(
                "candidate_live_identity_mismatch"
            )
        _LIVE_RECORDS.pop(id(capability), None)

    if record.process_id != os.getpid():
        raise RawVNNLibRivalAdapterError(
            "candidate_capability_crossed_process"
        )
    if candidate.proof_authority is not False:
        raise RawVNNLibRivalAdapterError(
            "candidate_claims_proof_authority"
        )
    rival_bindings = _exact_rival_bindings(record.rivals)
    if (
        rival_bindings is None
        or rival_bindings != record.rival_bindings
        or record.rivals is not record.issued_rivals
        or candidate.receipt is not record.candidate_receipt
        or not _candidate_runtime_types_exact(candidate)
    ):
        raise RawVNNLibRivalAdapterError(
            "candidate_runtime_snapshot_mismatch"
        )
    content_payload = _candidate_content_payload(
        source_path=candidate.source_path,
        source_identity=_candidate_source_identity(candidate),
        vnnlib_sha256=candidate.vnnlib_sha256,
        dialect=candidate.dialect,
        true_class=candidate.true_class,
        output_assert_ordinal=candidate.output_assert_ordinal,
        rows=candidate.rows,
        live_assert_sha256=candidate.live_assert_sha256,
    )
    receipt_payload = _candidate_receipt_payload(
        content_payload=content_payload,
        candidate_sha256=candidate.candidate_sha256,
    )
    if (
        candidate.candidate_sha256 != record.candidate_sha256
        or _canonical_sha256(content_payload) != record.candidate_sha256
        or not _receipt_matches_exact(
            candidate.receipt,
            expected_payload=receipt_payload,
            expected_sha256=record.candidate_receipt_sha256,
        )
    ):
        raise RawVNNLibRivalAdapterError(
            "candidate_content_binding_mismatch"
        )
    _path, _payload, observed_sha256, observed_identity = (
        _read_bound_source(
            record.source_path,
            record.expected_sha256,
            deadline=operation_deadline,
        )
    )
    if (
        observed_sha256 != candidate.vnnlib_sha256
        or observed_identity != record.source_identity
        or observed_identity != _candidate_source_identity(candidate)
    ):
        raise RawVNNLibRivalAdapterError(
            "candidate_source_identity_binding_mismatch"
        )
    live = _snapshot_live_assert(
        live_assert_params, deadline=operation_deadline
    )
    if (
        live.digest != record.live_assert_sha256
        or live.digest != candidate.live_assert_sha256
    ):
        raise RawVNNLibRivalAdapterError(
            "candidate_live_assert_binding_mismatch"
        )
    _check_deadline(operation_deadline, stage="candidate_consume_finish")
    return _build_consumed_batch(
        candidate=candidate, rivals=record.rivals
    )


__all__ = [
    "ConsumedRivalBatch",
    "RawVNNLibRivalAdapterError",
    "RawVNNLibTop1Candidate",
    "RawVNNLibTop1Row",
    "consume_raw_vnnlib_top1_candidate",
    "issue_raw_vnnlib_top1_candidate",
    "revoke_raw_vnnlib_top1_candidate",
    "validate_consumed_raw_vnnlib_rival_batch",
]
