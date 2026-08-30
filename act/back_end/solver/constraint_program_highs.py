#!/usr/bin/env python3
# ===- constraint_program_highs.py - streamed native HiGHS candidate ----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Disconnected Phase-C candidate for streaming a constraint program to HiGHS.

This module is deliberately *not* verifier integration.  A factory-created
handoff binds one exact :class:`SparseHZono` object to one authentic sealed
``ConstraintProgram`` object in a process-local, weak, ABA-guarded registry.
That binding prevents a handoff object from self-signing or changing owners,
but it is not a producer-authenticated provenance capability.  Receipts from
this module therefore have no proof, verdict, solver-status, or receipt
authority.

Only native RANGE/LE batches are consumed.  For binary factors the program
uses ``xi_b in {-1,+1}``, whereas HiGHS receives ``z in {0,1}``.  Each source
row is streamed as::

    lower + sum(Ab) <= Ac*x + (2*Ab)*z <= upper + sum(Ab)

The sum is accumulated over the exact dyadic value of every stored binary64
coefficient.  Finite lower bounds are rounded toward minus infinity and
finite upper bounds toward plus infinity when the exact shifted dyadic is not
representable as binary64.  Coefficients are never thresholded or eliminated;
non-finite or inexact/overflowing ``2*Ab`` transformations fail closed.

No whole-program sparse concatenate is built.  Both loading and incumbent
validation use closeable native cursors with at most 256 source rows.  HiGHS
native state and cursors are cleared in ``finally`` paths, and a cleanup error
never replaces the primary exception object.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import os
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import constraint_program as _program_core
from act.back_end.solver.constraint_program import (
    ConstraintProgram,
    ExternalFactorID,
    FactorKind,
    NativeConstraintBatch,
)
from act.back_end.solver.solver_hz import SparseHZono

try:  # HiGHS is optional for importability; use fails closed when absent.
    import highspy as _highspy
except Exception:  # pragma: no cover - exercised only without optional dep
    _highspy = None


_SCHEMA = "act.constraint_program.highs_stream_candidate.v1"
_MAX_BATCH_ROWS = 256
_MAX_COLUMNS = 10_000_000
_MAX_SOURCE_ROWS = 2_000_000_000
_MAX_SOURCE_NNZ = 2_000_000_000
_MAX_BATCH_NNZ = 100_000_000
_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_FINITE_FRACTION = Fraction.from_float(float(np.finfo(np.float64).max))
_REQUESTED_SMALL_MATRIX_VALUE = 1.0e-12


class ConstraintProgramHighsError(RuntimeError):
    """The disconnected native loader could not establish its full contract."""


def _note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


def _exact_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ConstraintProgramHighsError(f"{name} must be an exact builtin bool")
    return value


def _max_rows(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_BATCH_ROWS:
        raise ConstraintProgramHighsError(
            "max_rows must be an exact builtin int in [1, 256]"
        )
    return value


def _tolerance(value: Any) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ConstraintProgramHighsError(
            "validation_tolerance must be a finite nonnegative builtin float"
        )
    return value


def _objective(value: Any, columns: int) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.ndim != 1
        or value.size != columns
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise ConstraintProgramHighsError(
            "objective must be one finite C-contiguous float64 vector in "
            "original (continuous, xi-binary) factor order"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _exact_csr(value: Any, *, rows: int, columns: int, name: str) -> sp.csr_matrix:
    if (
        type(value) is not sp.csr_matrix
        or value.dtype != np.dtype(np.float64)
        or value.shape != (rows, columns)
        or type(value.data) is not np.ndarray
        or type(value.indices) is not np.ndarray
        or type(value.indptr) is not np.ndarray
        or value.data.ndim != 1
        or value.indices.ndim != 1
        or value.indptr.ndim != 1
        or value.data.size != value.indices.size
        or value.indptr.size != rows + 1
        or value.indices.dtype.kind not in "iu"
        or value.indptr.dtype.kind not in "iu"
        or int(value.indptr[0]) != 0
        or int(value.indptr[-1]) != int(value.data.size)
        or np.any(value.indptr[1:] < value.indptr[:-1])
        or (
            value.indices.size
            and (
                np.any(value.indices < 0)
                or np.any(value.indices >= columns)
            )
        )
        or not np.all(np.isfinite(value.data))
        or np.any(value.data == 0.0)
    ):
        raise ConstraintProgramHighsError(f"{name} is not exact canonical float64 CSR")
    for row in range(rows):
        start, stop = int(value.indptr[row]), int(value.indptr[row + 1])
        if stop - start > 1 and np.any(
            value.indices[start + 1 : stop]
            <= value.indices[start : stop - 1]
        ):
            raise ConstraintProgramHighsError(
                f"{name} row indices are not strictly increasing"
            )
    return value


def _raw_program_ids(
    value: Any,
    *,
    kind: FactorKind,
    name: str,
) -> Tuple[Tuple[int, ...], Tuple[Any, ...]]:
    if type(value) is not tuple:
        raise ConstraintProgramHighsError(f"{name} IDs are not an exact tuple")
    raw = []
    namespaces = []
    for factor in value:
        if (
            type(factor) is not ExternalFactorID
            or factor.kind is not kind
            or type(factor.raw_id) is not int
            or factor.raw_id < 0
        ):
            raise ConstraintProgramHighsError(f"{name} factor identity is malformed")
        raw.append(factor.raw_id)
        namespaces.append(factor.namespace_identity)
    if len(set(raw)) != len(raw):
        raise ConstraintProgramHighsError(f"{name} factor IDs are not unique")
    return tuple(raw), tuple(namespaces)


def _array_structure(value: np.ndarray) -> Tuple[Any, ...]:
    return (
        value.shape,
        value.dtype.str,
        value.strides,
        bool(value.flags.c_contiguous),
        bool(value.flags.f_contiguous),
    )


def _csr_structure(value: sp.csr_matrix) -> Tuple[Any, ...]:
    return (
        value.shape,
        value.dtype.str,
        bool(value.has_canonical_format),
        bool(value.has_sorted_indices),
        _array_structure(value.data),
        _array_structure(value.indices),
        _array_structure(value.indptr),
    )


def _capture_hz_graph(hz: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Capture the exact live object graph and structural/ID truth we consume."""

    if type(hz) is not SparseHZono or type(vars(hz)) is not dict:
        raise ConstraintProgramHighsError("handoff requires one exact SparseHZono")
    live = vars(hz)
    arrays = []
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        value = live.get(name)
        if type(value) is not np.ndarray:
            raise ConstraintProgramHighsError(f"SparseHZono.{name} was rebound")
        arrays.append(value)
    c, b, ub, col_ids, bcol_ids = arrays
    if (
        c.dtype != np.dtype(np.float64)
        or b.dtype != np.dtype(np.float64)
        or ub.dtype != np.dtype(np.float64)
        or col_ids.dtype != np.dtype(np.int64)
        or bcol_ids.dtype != np.dtype(np.int64)
        or any(value.ndim != 1 or not value.flags.c_contiguous for value in arrays)
    ):
        raise ConstraintProgramHighsError("SparseHZono vector graph is malformed")
    matrices = []
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        value = live.get(name)
        if type(value) is not sp.csr_matrix:
            raise ConstraintProgramHighsError(f"SparseHZono.{name} was rebound")
        if (
            value.dtype != np.dtype(np.float64)
            or not value.has_canonical_format
            or not value.has_sorted_indices
            or type(value.data) is not np.ndarray
            or type(value.indices) is not np.ndarray
            or type(value.indptr) is not np.ndarray
        ):
            raise ConstraintProgramHighsError(
                f"SparseHZono.{name} is not owned canonical float64 CSR"
            )
        matrices.append(value)
    Gc, Gb, Ac, Ab, Auc, Aub = matrices
    n_out = int(c.size)
    n_cont = int(col_ids.size)
    n_bin = int(bcol_ids.size)
    n_eq = int(b.size)
    n_ub = int(ub.size)
    if (
        Gc.shape != (n_out, n_cont)
        or Gb.shape != (n_out, n_bin)
        or Ac.shape != (n_eq, n_cont)
        or Ab.shape != (n_eq, n_bin)
        or Auc.shape != (n_ub, n_cont)
        or Aub.shape != (n_ub, n_bin)
    ):
        raise ConstraintProgramHighsError("SparseHZono dimensions changed")
    continuous = tuple(int(item) for item in col_ids.tolist())
    binary = tuple(int(item) for item in bcol_ids.tolist())
    if (
        any(item < 0 for item in continuous + binary)
        or len(set(continuous)) != len(continuous)
        or len(set(binary)) != len(binary)
        or set(continuous).intersection(binary)
    ):
        raise ConstraintProgramHighsError(
            "SparseHZono stable IDs must be nonnegative, unique, and kind-disjoint"
        )
    objects = [hz, *arrays, *matrices]
    for matrix in matrices:
        objects.extend((matrix.data, matrix.indices, matrix.indptr))
    structure = (
        tuple(_array_structure(value) for value in arrays),
        tuple(_csr_structure(value) for value in matrices),
        n_out,
        n_cont,
        n_bin,
        n_eq,
        n_ub,
        continuous,
        binary,
    )
    return tuple(objects), structure


@dataclass(frozen=True)
class _HandoffTerminalState:
    """One-way retirement bit retained even if registry cleanup is interrupted."""

    # The dataclass is frozen only at the public Python layer.  Retirement is
    # one internal ``object.__setattr__`` store and is never reversed.
    retired: bool = False


@dataclass(frozen=True)
class _HandoffRecord:
    handoff_reference: Any
    hz_reference: Any
    program_reference: Any
    hz_graph_references: Tuple[Any, ...]
    hz_graph_identity: Tuple[int, ...]
    hz_structure: Tuple[Any, ...]
    continuous_ids: Tuple[int, ...]
    binary_ids: Tuple[int, ...]
    namespace_identity: Any
    source_rows: int
    source_nnz: int
    virtual_rows: int
    program_digest: str
    process_id: int
    terminal_state: _HandoffTerminalState


_HANDOFF_LOCK = threading.Lock()
_HANDOFF_REGISTRY: dict[int, _HandoffRecord] = {}
_HANDOFF_FACTORY_TOKEN = object()


class ConstraintProgramHighsHandoff:
    """Factory-issued, registry-owned disconnected source/model binding."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *_args: Any, **kwargs: Any) -> "ConstraintProgramHighsHandoff":
        if (
            cls is not ConstraintProgramHighsHandoff
            or kwargs.pop("_token", None) is not _HANDOFF_FACTORY_TOKEN
            or _args
            or kwargs
        ):
            raise TypeError("ConstraintProgramHighsHandoff cannot be self-signed")
        return object.__new__(cls)

    @property
    def continuous_ids(self) -> Tuple[int, ...]:
        return _handoff_record(self).continuous_ids

    @property
    def binary_ids(self) -> Tuple[int, ...]:
        return _handoff_record(self).binary_ids

    @property
    def source_rows(self) -> int:
        return _handoff_record(self).source_rows

    @property
    def source_nnz(self) -> int:
        return _handoff_record(self).source_nnz

    @property
    def receipt(self) -> Mapping[str, Any]:
        record = _handoff_record(self)
        return _receipt(
            record,
            max_rows=None,
            native_model_loaded=False,
            native_solver_run=False,
            rows_loaded=0,
            nnz_loaded=0,
            incumbent_validated=False,
        )

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False

    @property
    def solver_status_authority(self) -> bool:
        return False

    def load(
        self,
        objective: np.ndarray,
        *,
        maximize: bool = False,
        max_rows: int = 256,
    ) -> "ConstraintProgramHighsResult":
        """Load then clear one native model without running the solver."""

        return _execute(
            self,
            objective,
            maximize=maximize,
            max_rows=max_rows,
            validation_tolerance=1.0e-7,
            solve=False,
        )

    def solve(
        self,
        objective: np.ndarray,
        *,
        maximize: bool = False,
        max_rows: int = 256,
        validation_tolerance: float = 1.0e-7,
    ) -> "ConstraintProgramHighsResult":
        """Load, solve, stream-validate any optimal incumbent, then clear."""

        return _execute(
            self,
            objective,
            maximize=maximize,
            max_rows=max_rows,
            validation_tolerance=validation_tolerance,
            solve=True,
        )


class ConstraintProgramHighsFactory:
    """Factory that validates and publishes disconnected exact-object bindings."""

    __slots__ = ()

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConstraintProgramHighsFactory":
        if cls is not ConstraintProgramHighsFactory or _args or _kwargs:
            raise TypeError("ConstraintProgramHighsFactory takes no arguments/subclasses")
        return object.__new__(cls)

    def bind(
        self,
        hz: SparseHZono,
        program: ConstraintProgram,
    ) -> ConstraintProgramHighsHandoff:
        return _bind(hz, program)


def bind_constraint_program_highs(
    hz: SparseHZono,
    program: ConstraintProgram,
) -> ConstraintProgramHighsHandoff:
    """Convenience factory entry; the returned handoff remains non-authoritative."""

    return ConstraintProgramHighsFactory().bind(hz, program)


def _bind(
    hz: SparseHZono,
    program: ConstraintProgram,
) -> ConstraintProgramHighsHandoff:
    graph, structure = _capture_hz_graph(hz)
    if type(program) is not ConstraintProgram:
        raise ConstraintProgramHighsError(
            "handoff requires one exact authentic ConstraintProgram"
        )
    try:
        continuous_ids, continuous_namespaces = _raw_program_ids(
            program.continuous_ids,
            kind=FactorKind.CONTINUOUS,
            name="program continuous",
        )
        binary_ids, binary_namespaces = _raw_program_ids(
            program.binary_ids,
            kind=FactorKind.BINARY,
            name="program binary",
        )
        source_rows = program.source_rows
        source_nnz = program.source_nnz
        virtual_rows = program.virtual_facet_rows
        fallback_pairs = program.fallback_pairs
        digest = program.digest
        representation = program.representation_authority
        replay = program.replay_authority
    except BaseException as error:
        if type(error) is ConstraintProgramHighsError:
            raise
        raise ConstraintProgramHighsError(
            "constraint program authenticity validation failed"
        ) from error
    namespaces = continuous_namespaces + binary_namespaces
    if namespaces and any(value is not namespaces[0] for value in namespaces):
        raise ConstraintProgramHighsError("program factor namespaces differ")
    namespace = namespaces[0] if namespaces else None
    hz_continuous = structure[-2]
    hz_binary = structure[-1]
    n_eq = structure[5]
    n_ub = structure[6]
    if (
        continuous_ids != hz_continuous
        or binary_ids != hz_binary
        or set(continuous_ids).intersection(binary_ids)
    ):
        raise ConstraintProgramHighsError(
            "program and SparseHZono stable factor IDs/order differ"
        )
    if (
        type(source_rows) is not int
        or type(source_nnz) is not int
        or type(virtual_rows) is not int
        or source_rows < 0
        or source_nnz < 0
        or virtual_rows < 0
        or source_rows > virtual_rows
        or n_eq != 0
        or n_ub != virtual_rows
        or fallback_pairs != 0
        or representation is not True
        or replay is not True
    ):
        raise ConstraintProgramHighsError(
            "program/HZ row accounting is not the Phase-B all-native contract"
        )
    columns = len(continuous_ids) + len(binary_ids)
    if (
        columns > min(_MAX_COLUMNS, _INT32_MAX)
        or source_rows > _MAX_SOURCE_ROWS
        or source_nnz > _MAX_SOURCE_NNZ
    ):
        raise ConstraintProgramHighsError("handoff exceeds bounded native resources")

    handoff = ConstraintProgramHighsHandoff(_token=_HANDOFF_FACTORY_TOKEN)
    object_id = id(handoff)

    def cleanup(reference: Any) -> None:
        for _attempt in range(4):
            try:
                with _HANDOFF_LOCK:
                    current = _HANDOFF_REGISTRY.get(object_id)
                    if (
                        current is not None
                        and (
                            current.handoff_reference is reference
                            or current.hz_reference is reference
                            or current.program_reference is reference
                        )
                    ):
                        _HANDOFF_REGISTRY.pop(object_id, None)
                return
            except BaseException:
                continue

    handoff_reference = weakref.ref(handoff, cleanup)
    hz_reference = weakref.ref(hz, cleanup)
    program_reference = weakref.ref(program, cleanup)
    graph_references = tuple(weakref.ref(value) for value in graph)
    record = _HandoffRecord(
        handoff_reference,
        hz_reference,
        program_reference,
        graph_references,
        tuple(id(value) for value in graph),
        structure,
        continuous_ids,
        binary_ids,
        namespace,
        source_rows,
        source_nnz,
        virtual_rows,
        digest,
        os.getpid(),
        _HandoffTerminalState(),
    )
    # Re-capture immediately before publication so a changed mutable HZ graph
    # cannot cross validation and registry insertion unnoticed.
    current_graph, current_structure = _capture_hz_graph(hz)
    if (
        current_structure != structure
        or len(current_graph) != len(graph)
        or any(current is not captured for current, captured in zip(current_graph, graph))
    ):
        raise ConstraintProgramHighsError("SparseHZono changed during handoff binding")
    with _HANDOFF_LOCK:
        existing = _HANDOFF_REGISTRY.get(object_id)
        if existing is not None and existing.handoff_reference() is not None:
            raise ConstraintProgramHighsError("handoff registry ID was reused")
        _HANDOFF_REGISTRY[object_id] = record
    _validated_handoff(handoff, expected_record=record)
    return handoff


def _validate_record(
    record: _HandoffRecord,
    handoff: ConstraintProgramHighsHandoff,
) -> Tuple[SparseHZono, ConstraintProgram]:
    if (
        type(record) is not _HandoffRecord
        or type(record.terminal_state) is not _HandoffTerminalState
        or record.terminal_state.retired is not False
        or record.process_id != os.getpid()
        or record.handoff_reference() is not handoff
    ):
        raise ConstraintProgramHighsError("handoff registry provenance is stale")
    hz = record.hz_reference()
    program = record.program_reference()
    if type(hz) is not SparseHZono or type(program) is not ConstraintProgram:
        raise ConstraintProgramHighsError("handoff owner was collected or substituted")
    graph, structure = _capture_hz_graph(hz)
    captured = tuple(reference() for reference in record.hz_graph_references)
    if (
        any(value is None for value in captured)
        or tuple(id(value) for value in graph) != record.hz_graph_identity
        or len(graph) != len(captured)
        or any(current is not old for current, old in zip(graph, captured))
        or structure != record.hz_structure
        or structure[-2] != record.continuous_ids
        or structure[-1] != record.binary_ids
    ):
        raise ConstraintProgramHighsError("SparseHZono live object graph changed")
    continuous_ids, continuous_namespaces = _raw_program_ids(
        program.continuous_ids,
        kind=FactorKind.CONTINUOUS,
        name="program continuous",
    )
    binary_ids, binary_namespaces = _raw_program_ids(
        program.binary_ids,
        kind=FactorKind.BINARY,
        name="program binary",
    )
    if (
        continuous_ids != record.continuous_ids
        or binary_ids != record.binary_ids
        or any(
            namespace is not record.namespace_identity
            for namespace in continuous_namespaces + binary_namespaces
        )
        or program.source_rows != record.source_rows
        or program.source_nnz != record.source_nnz
        or program.virtual_facet_rows != record.virtual_rows
        or program.fallback_pairs != 0
        or program.digest != record.program_digest
    ):
        raise ConstraintProgramHighsError("constraint program live graph changed")
    if record.terminal_state.retired is not False:
        raise ConstraintProgramHighsError("handoff is terminally retired")
    return hz, program


def _retirement_cleanup_attempt(
    record: _HandoffRecord,
    handoff: ConstraintProgramHighsHandoff,
) -> None:
    """One identity-guarded removal attempt, split out for fault injection."""

    with _HANDOFF_LOCK:
        current = _HANDOFF_REGISTRY.get(id(handoff))
        if current is record:
            _HANDOFF_REGISTRY.pop(id(handoff), None)


def _retire_handoff(
    record: _HandoffRecord,
    handoff: ConstraintProgramHighsHandoff,
    primary: BaseException,
) -> None:
    """Durably retire before bounded fallible cleanup, preserving ``primary``."""

    # This exact internal state has no user callback.  It is the durable
    # linearization point: registry removal is only leak cleanup afterward.
    object.__setattr__(record.terminal_state, "retired", True)
    cleanup_raised = False
    removed = False
    for _attempt in range(4):
        try:
            _retirement_cleanup_attempt(record, handoff)
            removed = True
            break
        except BaseException:
            cleanup_raised = True
    if cleanup_raised or not removed:
        _note(primary, "handoff retirement registry cleanup also failed")


def _validated_handoff(
    value: Any,
    *,
    expected_record: Optional[_HandoffRecord] = None,
) -> Tuple[_HandoffRecord, SparseHZono, ConstraintProgram]:
    """The sole consuming lookup/validation primitive for live handoffs."""

    if type(value) is not ConstraintProgramHighsHandoff:
        raise ConstraintProgramHighsError("handoff has the wrong exact type")
    try:
        with _HANDOFF_LOCK:
            record = _HANDOFF_REGISTRY.get(id(value))
    except BaseException as primary:
        if expected_record is not None:
            _retire_handoff(expected_record, value, primary)
        raise
    if record is None:
        raise ConstraintProgramHighsError("handoff is forged, stale, or collected")
    if expected_record is not None and record is not expected_record:
        primary = ConstraintProgramHighsError(
            "handoff registry record changed during validation"
        )
        _retire_handoff(expected_record, value, primary)
        raise primary
    try:
        hz, program = _validate_record(record, value)
    except BaseException as primary:
        _retire_handoff(record, value, primary)
        raise
    return record, hz, program


def _handoff_record(value: Any) -> _HandoffRecord:
    record, _hz, _program = _validated_handoff(value)
    return record


def _new_highs() -> Any:
    if _highspy is None:
        raise ConstraintProgramHighsError("highspy backend is unavailable")
    return _highspy.Highs()


def _require_highs_ok(status: Any, operation: str) -> None:
    if _highspy is None or status != _highspy.HighsStatus.kOk:
        raise ConstraintProgramHighsError(f"HiGHS {operation} failed: {status}")


def _clear_highs(highs: Any) -> None:
    _require_highs_ok(highs.clear(), "clear")


def _read_highs_float_option(highs: Any, name: str) -> float:
    result = highs.getOptionValue(name)
    if (
        type(result) is not tuple
        or len(result) != 2
        or _highspy is None
        or result[0] != _highspy.HighsStatus.kOk
        or not isinstance(result[1], (float, np.floating))
        or not math.isfinite(float(result[1]))
    ):
        raise ConstraintProgramHighsError(f"HiGHS option {name} readback failed")
    return float(result[1])


def _configure_matrix_thresholds(
    highs: Any,
) -> Tuple[float, float, float, float]:
    """Pin/read back thresholds whose silent filtering would change rows."""

    _require_highs_ok(
        highs.setOptionValue(
            "small_matrix_value", _REQUESTED_SMALL_MATRIX_VALUE
        ),
        "set small_matrix_value",
    )
    small = _read_highs_float_option(highs, "small_matrix_value")
    large = _read_highs_float_option(highs, "large_matrix_value")
    infinite = _read_highs_float_option(highs, "infinite_bound")
    infinite_cost = _read_highs_float_option(highs, "infinite_cost")
    if (
        small != _REQUESTED_SMALL_MATRIX_VALUE
        or not 0.0 < small < large < infinite
        or infinite_cost <= large
    ):
        raise ConstraintProgramHighsError(
            "HiGHS matrix threshold readback changed or is malformed"
        )
    return small, large, infinite, infinite_cost


def _outward_float(value: Fraction, *, lower: bool) -> float:
    if type(value) is not Fraction or abs(value) > _MAX_FINITE_FRACTION:
        raise ConstraintProgramHighsError("shifted native bound overflows binary64")
    nearest = float(value)
    if not math.isfinite(nearest):
        raise ConstraintProgramHighsError("shifted native bound is non-finite")
    represented = Fraction.from_float(nearest)
    if lower and represented > value:
        nearest = float(np.nextafter(nearest, -np.inf))
    elif not lower and represented < value:
        nearest = float(np.nextafter(nearest, np.inf))
    if not math.isfinite(nearest):
        raise ConstraintProgramHighsError("outward shifted bound is non-finite")
    final = Fraction.from_float(nearest)
    if (lower and final > value) or (not lower and final < value):
        raise ConstraintProgramHighsError("directed bound rounding failed")
    return nearest


def _nearest_float(value: Fraction, *, name: str) -> float:
    if type(value) is not Fraction or abs(value) > _MAX_FINITE_FRACTION:
        raise ConstraintProgramHighsError(f"{name} overflows binary64")
    result = float(value)
    if not math.isfinite(result):
        raise ConstraintProgramHighsError(f"{name} is non-finite")
    return result


def _double_exact(value: float) -> float:
    if not math.isfinite(value):
        raise ConstraintProgramHighsError("native binary coefficient is non-finite")
    doubled = value * 2.0
    if (
        not math.isfinite(doubled)
        or Fraction.from_float(doubled) != 2 * Fraction.from_float(value)
        or (value != 0.0 and doubled == 0.0)
    ):
        raise ConstraintProgramHighsError(
            "2*Ab is non-finite, underflowed, or not exact"
        )
    return doubled


def _batch_ids(batch: NativeConstraintBatch) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Any, ...]]:
    continuous, continuous_namespaces = _raw_program_ids(
        batch.continuous_ids,
        kind=FactorKind.CONTINUOUS,
        name="batch continuous",
    )
    binary, binary_namespaces = _raw_program_ids(
        batch.binary_ids,
        kind=FactorKind.BINARY,
        name="batch binary",
    )
    return continuous, binary, continuous_namespaces + binary_namespaces


def _transform_batch(
    batch: NativeConstraintBatch,
    record: _HandoffRecord,
    *,
    small_matrix_value: float,
    large_matrix_value: float,
    infinite_bound: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if type(batch) is not NativeConstraintBatch:
        raise ConstraintProgramHighsError("native cursor yielded the wrong batch type")
    rows = batch.row_count
    n_cont = len(record.continuous_ids)
    n_bin = len(record.binary_ids)
    Ac = _exact_csr(batch.A_cont, rows=rows, columns=n_cont, name="native A_cont")
    Ab = _exact_csr(batch.A_bin, rows=rows, columns=n_bin, name="native A_bin")
    lower = batch.lower
    upper = batch.upper
    if (
        type(lower) is not np.ndarray
        or type(upper) is not np.ndarray
        or lower.dtype != np.dtype(np.float64)
        or upper.dtype != np.dtype(np.float64)
        or lower.shape != (rows,)
        or upper.shape != (rows,)
        or not lower.flags.c_contiguous
        or not upper.flags.c_contiguous
        or np.any(np.isnan(lower))
        or np.any(np.isposinf(lower))
        or np.any(~np.isfinite(upper))
    ):
        raise ConstraintProgramHighsError("native row bounds are malformed")
    continuous, binary, namespaces = _batch_ids(batch)
    if (
        continuous != record.continuous_ids
        or binary != record.binary_ids
        or any(value is not record.namespace_identity for value in namespaces)
    ):
        raise ConstraintProgramHighsError("native batch factor frame changed")
    nnz = int(Ac.nnz + Ab.nnz)
    if nnz > min(_MAX_BATCH_NNZ, _INT32_MAX):
        raise ConstraintProgramHighsError("native batch exceeds bounded nnz resources")
    starts = np.empty(rows + 1, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    shifted_lower = np.empty(rows, dtype=np.float64)
    shifted_upper = np.empty(rows, dtype=np.float64)
    starts[0] = 0
    cursor = 0
    for row in range(rows):
        c_start, c_stop = int(Ac.indptr[row]), int(Ac.indptr[row + 1])
        b_start, b_stop = int(Ab.indptr[row]), int(Ab.indptr[row + 1])
        c_count = c_stop - c_start
        b_count = b_stop - b_start
        stop = cursor + c_count + b_count
        if stop > _INT32_MAX:
            raise ConstraintProgramHighsError("native batch indptr overflows int32")
        if c_count:
            c_indices = np.asarray(Ac.indices[c_start:c_stop], dtype=np.int64)
            if int(c_indices[0]) < 0 or int(c_indices[-1]) >= n_cont:
                raise ConstraintProgramHighsError("A_cont column is outside its frame")
            indices[cursor : cursor + c_count] = c_indices.astype(np.int32)
            continuous_data = Ac.data[c_start:c_stop]
            continuous_abs = np.abs(continuous_data)
            if np.any(continuous_abs <= small_matrix_value) or np.any(
                continuous_abs >= large_matrix_value
            ):
                raise ConstraintProgramHighsError(
                    "A_cont coefficient crosses a HiGHS filtering threshold"
                )
            data[cursor : cursor + c_count] = continuous_data
        exact_shift = Fraction(0)
        if b_count:
            b_indices = np.asarray(Ab.indices[b_start:b_stop], dtype=np.int64)
            if int(b_indices[0]) < 0 or int(b_indices[-1]) >= n_bin:
                raise ConstraintProgramHighsError("A_bin column is outside its frame")
            binary_cursor = cursor + c_count
            shifted_indices = b_indices + n_cont
            if int(shifted_indices[-1]) > _INT32_MAX:
                raise ConstraintProgramHighsError("shifted A_bin column overflows int32")
            indices[binary_cursor:stop] = shifted_indices.astype(np.int32)
            for position, stored in enumerate(Ab.data[b_start:b_stop]):
                coefficient = float(stored)
                doubled = _double_exact(coefficient)
                if (
                    abs(doubled) <= small_matrix_value
                    or abs(doubled) >= large_matrix_value
                ):
                    raise ConstraintProgramHighsError(
                        "2*Ab coefficient crosses a HiGHS filtering threshold"
                    )
                data[binary_cursor + position] = doubled
                exact_shift += Fraction.from_float(coefficient)
        lower_value = float(lower[row])
        upper_value = float(upper[row])
        shifted_lower[row] = (
            -float(_highspy.kHighsInf)
            if math.isinf(lower_value) and lower_value < 0.0
            else _outward_float(
                Fraction.from_float(lower_value) + exact_shift,
                lower=True,
            )
        )
        shifted_upper[row] = _outward_float(
            Fraction.from_float(upper_value) + exact_shift,
            lower=False,
        )
        if (
            (
                math.isfinite(shifted_lower[row])
                and abs(float(shifted_lower[row])) >= infinite_bound
            )
            or abs(float(shifted_upper[row])) >= infinite_bound
        ):
            raise ConstraintProgramHighsError(
                "finite shifted row bound crosses HiGHS infinite_bound"
            )
        if shifted_lower[row] > shifted_upper[row]:
            raise ConstraintProgramHighsError("shifted native row became contradictory")
        cursor = stop
        starts[row + 1] = cursor
    if cursor != nnz:
        raise ConstraintProgramHighsError("native batch nnz accounting changed")
    return starts, indices, data, shifted_lower, shifted_upper, nnz


def _consume_cursor(
    program: ConstraintProgram,
    *,
    max_rows: int,
    consumer: Callable[[NativeConstraintBatch], None],
) -> None:
    cursor = None
    primary: Optional[BaseException] = None
    try:
        cursor = program.iter_native_batches(max_rows=max_rows)
        for batch in cursor:
            consumer(batch)
    except BaseException as error:
        primary = error
    finally:
        if cursor is not None:
            first_cleanup_error: Optional[BaseException] = None
            closed = False
            for _attempt in range(4):
                try:
                    cursor.close()
                    closed = True
                    break
                except BaseException as cleanup_error:
                    if first_cleanup_error is None:
                        first_cleanup_error = cleanup_error
            if first_cleanup_error is not None or not closed:
                cleanup_error = (
                    first_cleanup_error
                    if first_cleanup_error is not None
                    else ConstraintProgramHighsError(
                        "native cursor cleanup did not converge"
                    )
                )
                if primary is None:
                    primary = cleanup_error
                else:
                    _note(
                        primary,
                        "native cursor cleanup raised during bounded close",
                    )
    if primary is not None:
        raise primary


def _load_rows(
    highs: Any,
    program: ConstraintProgram,
    record: _HandoffRecord,
    *,
    max_rows: int,
    small_matrix_value: float,
    large_matrix_value: float,
    infinite_bound: float,
) -> Tuple[int, int]:
    offset = 0
    loaded_nnz = 0

    def consume(batch: NativeConstraintBatch) -> None:
        nonlocal offset, loaded_nnz
        if (
            batch.row_offset != offset
            or not 0 < batch.row_count <= max_rows
            or batch.total_rows != record.source_rows
        ):
            raise ConstraintProgramHighsError("native batch offsets changed")
        starts, indices, data, lower, upper, nnz = _transform_batch(
            batch,
            record,
            small_matrix_value=small_matrix_value,
            large_matrix_value=large_matrix_value,
            infinite_bound=infinite_bound,
        )
        _require_highs_ok(
            highs.addRows(
                batch.row_count,
                lower,
                upper,
                nnz,
                starts,
                indices,
                data,
            ),
            "addRows",
        )
        expected_rows = offset + batch.row_count
        expected_nnz = loaded_nnz + nnz
        if (
            type(highs.getNumCol()) is not int
            or highs.getNumCol()
            != len(record.continuous_ids) + len(record.binary_ids)
            or type(highs.getNumRow()) is not int
            or highs.getNumRow() != expected_rows
            or type(highs.getNumNz()) is not int
            or highs.getNumNz() != expected_nnz
        ):
            raise ConstraintProgramHighsError(
                "HiGHS changed columns, rows, or nnz during streamed addRows"
            )
        offset += batch.row_count
        loaded_nnz += nnz
        if loaded_nnz > record.source_nnz:
            raise ConstraintProgramHighsError("loaded nnz exceeds program accounting")

    _consume_cursor(program, max_rows=max_rows, consumer=consume)
    if offset != record.source_rows or loaded_nnz != record.source_nnz:
        raise ConstraintProgramHighsError("native cursor ended before exact accounting")
    return offset, loaded_nnz


def _fraction_row_value(matrix: sp.csr_matrix, row: int, values: np.ndarray) -> Fraction:
    result = Fraction(0)
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    for position in range(start, stop):
        result += Fraction.from_float(float(matrix.data[position])) * Fraction.from_float(
            float(values[int(matrix.indices[position])])
        )
    return result


def _validate_incumbent(
    program: ConstraintProgram,
    record: _HandoffRecord,
    continuous: np.ndarray,
    binary: np.ndarray,
    *,
    max_rows: int,
    tolerance: float,
) -> None:
    if (
        continuous.shape != (len(record.continuous_ids),)
        or binary.shape != (len(record.binary_ids),)
        or not np.all(np.isfinite(continuous))
        or not np.all(np.isfinite(binary))
        or np.any(continuous < -1.0)
        or np.any(continuous > 1.0)
        or np.any((binary != -1.0) & (binary != 1.0))
    ):
        raise ConstraintProgramHighsError("HiGHS incumbent factor values are malformed")
    offset = 0

    def consume(batch: NativeConstraintBatch) -> None:
        nonlocal offset
        if (
            batch.row_offset != offset
            or batch.total_rows != record.source_rows
            or not 0 < batch.row_count <= max_rows
        ):
            raise ConstraintProgramHighsError(
                "incumbent validator observed changed batch offsets"
            )
        rows = batch.row_count
        Ac = _exact_csr(
            batch.A_cont,
            rows=rows,
            columns=len(record.continuous_ids),
            name="validator A_cont",
        )
        Ab = _exact_csr(
            batch.A_bin,
            rows=rows,
            columns=len(record.binary_ids),
            name="validator A_bin",
        )
        lower = batch.lower
        upper = batch.upper
        continuous_ids, binary_ids, namespaces = _batch_ids(batch)
        if (
            continuous_ids != record.continuous_ids
            or binary_ids != record.binary_ids
            or any(value is not record.namespace_identity for value in namespaces)
        ):
            raise ConstraintProgramHighsError(
                "incumbent validator observed changed factor IDs"
            )
        for row in range(rows):
            value = _fraction_row_value(Ac, row, continuous)
            value += _fraction_row_value(Ab, row, binary)
            lower_value = float(lower[row])
            if (
                (
                    not math.isinf(lower_value)
                    and value < Fraction.from_float(lower_value)
                )
                or value > Fraction.from_float(float(upper[row]))
            ):
                raise ConstraintProgramHighsError(
                    f"HiGHS incumbent violates original program row {offset + row}"
                )
        offset += rows

    _consume_cursor(program, max_rows=max_rows, consumer=consume)
    if offset != record.source_rows:
        raise ConstraintProgramHighsError("incumbent validation ended early")


def _mapped_objective(
    objective: np.ndarray,
    n_cont: int,
) -> Tuple[np.ndarray, float]:
    binary = objective[n_cont:]
    mapped = np.empty(objective.size, dtype=np.float64)
    mapped[:n_cont] = objective[:n_cont]
    exact_sum = Fraction(0)
    for index, value in enumerate(binary):
        coefficient = float(value)
        mapped[n_cont + index] = _double_exact(coefficient)
        exact_sum += Fraction.from_float(coefficient)
    offset = _nearest_float(-exact_sum, name="mapped objective offset")
    return mapped, offset


def _load_columns(
    highs: Any,
    record: _HandoffRecord,
    objective: np.ndarray,
    *,
    maximize: bool,
    infinite_cost: float,
) -> None:
    n_cont = len(record.continuous_ids)
    n_bin = len(record.binary_ids)
    columns = n_cont + n_bin
    costs, offset = _mapped_objective(objective, n_cont)
    if (
        np.any(np.abs(costs) >= infinite_cost)
        or abs(offset) >= infinite_cost
    ):
        raise ConstraintProgramHighsError(
            "mapped objective crosses HiGHS infinite_cost"
        )
    lower = np.concatenate(
        (
            np.full(n_cont, -1.0, dtype=np.float64),
            np.zeros(n_bin, dtype=np.float64),
        )
    )
    upper = np.ones(columns, dtype=np.float64)
    starts = np.zeros(columns + 1, dtype=np.int32)
    empty_indices = np.empty(0, dtype=np.int32)
    empty_data = np.empty(0, dtype=np.float64)
    _require_highs_ok(
        highs.addCols(
            columns,
            costs,
            lower,
            upper,
            0,
            starts,
            empty_indices,
            empty_data,
        ),
        "addCols",
    )
    if (
        type(highs.getNumCol()) is not int
        or highs.getNumCol() != columns
        or type(highs.getNumRow()) is not int
        or highs.getNumRow() != 0
        or type(highs.getNumNz()) is not int
        or highs.getNumNz() != 0
    ):
        raise ConstraintProgramHighsError(
            "HiGHS changed the empty column-frame postcondition"
        )
    if n_bin:
        binary_indices = np.arange(n_cont, columns, dtype=np.int32)
        integrality = np.full(
            n_bin,
            int(_highspy.HighsVarType.kInteger),
            dtype=np.uint8,
        )
        _require_highs_ok(
            highs.changeColsIntegrality(n_bin, binary_indices, integrality),
            "changeColsIntegrality",
        )
    _require_highs_ok(highs.changeObjectiveOffset(offset), "changeObjectiveOffset")
    if maximize:
        _require_highs_ok(
            highs.changeObjectiveSense(_highspy.ObjSense.kMaximize),
            "changeObjectiveSense",
        )


@dataclass(frozen=True)
class ConstraintProgramHighsResult:
    """Immutable diagnostic result; no field is solver/verdict authority."""

    model_status: str
    run_status: str
    objective_value: Optional[float]
    continuous: np.ndarray
    binary: np.ndarray
    rows_loaded: int
    nnz_loaded: int
    incumbent_validated: bool
    receipt: Mapping[str, Any]


def _readonly(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    result.setflags(write=False)
    return result


def _receipt(
    record: _HandoffRecord,
    *,
    max_rows: Optional[int],
    native_model_loaded: bool,
    native_solver_run: bool,
    rows_loaded: int,
    nnz_loaded: int,
    incumbent_validated: bool,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": _SCHEMA,
            "candidate_only": True,
            "production_integration": False,
            "consumer_integration": False,
            "producer_authenticated": False,
            "disconnected_exact_object_binding": True,
            "hz_binding_scope": "exact_object_graph_shape_and_factor_ids_only",
            "hz_semantic_snapshot": False,
            "hz_value_snapshot_authenticated": False,
            "hz_coefficient_content_consumed": False,
            "full_hz_live_graph_authenticated": False,
            "future_producer_solver_integration_blocked": True,
            "registry_owned_handoff": True,
            "program_digest": record.program_digest,
            "digest_is_diagnostic_only": True,
            "authenticity_from_digest": False,
            "native_batches_only": True,
            "whole_program_sparse_stack_built": False,
            "binary_mapping": "xi_b=2*z-1",
            "native_model_loaded": native_model_loaded,
            "model_loaded": native_model_loaded,
            "model_cleared_before_return": native_model_loaded,
            "binary_integrality_applicable": bool(record.binary_ids),
            "binary_integrality_loaded": bool(
                native_model_loaded and record.binary_ids
            ),
            "integrality_loaded": bool(
                native_model_loaded and record.binary_ids
            ),
            "exact_dyadic_binary_row_sum": True,
            "directed_outward_shifted_bounds": True,
            "tiny_coefficients_deleted": False,
            "max_rows": max_rows,
            "source_rows": record.source_rows,
            "source_nnz": record.source_nnz,
            "rows_loaded": rows_loaded,
            "nnz_loaded": nnz_loaded,
            "native_solver_run": native_solver_run,
            "incumbent_validated_in_original_xi_coordinates": incumbent_validated,
            "incumbent_row_validation_tolerance": 0.0,
            "receipt_authority": False,
            "representation_authority": False,
            "replay_authority": False,
            "proof_authority": False,
            "verdict_authority": False,
            "solver_status_authority": False,
            "triangle_relaxation_called": False,
            "act_network_branch_and_bound_called": False,
            "branch_and_bound_claim_scope": "ACT_network_BaB_only",
            "no_claim_about_highs_internal_mip_branching": True,
            "highs_internal_mip_search_not_audited": bool(
                native_model_loaded and record.binary_ids
            ),
            "highs_internal_mip_algorithm_is_not_act_network_bab": True,
            "backward_called": False,
            "dual_called": False,
            "scip_called": False,
            "real_model_called": False,
            "large_model_called": False,
        }
    )


def _execute(
    handoff: ConstraintProgramHighsHandoff,
    objective_value: np.ndarray,
    *,
    maximize: bool,
    max_rows: int,
    validation_tolerance: float,
    solve: bool,
) -> ConstraintProgramHighsResult:
    maximize = _exact_bool(maximize, name="maximize")
    solve = _exact_bool(solve, name="solve")
    max_rows = _max_rows(max_rows)
    tolerance = _tolerance(validation_tolerance)
    record = _handoff_record(handoff)
    record, _hz, program = _validated_handoff(
        handoff, expected_record=record
    )
    objective = _objective(
        objective_value,
        len(record.continuous_ids) + len(record.binary_ids),
    )
    highs = None
    result: Optional[ConstraintProgramHighsResult] = None
    primary: Optional[BaseException] = None
    try:
        highs = _new_highs()
        _require_highs_ok(highs.setOptionValue("output_flag", False), "set output_flag")
        _require_highs_ok(highs.setOptionValue("threads", 1), "set threads")
        (
            small_matrix_value,
            large_matrix_value,
            infinite_bound,
            infinite_cost,
        ) = _configure_matrix_thresholds(highs)
        _load_columns(
            highs,
            record,
            objective,
            maximize=maximize,
            infinite_cost=infinite_cost,
        )
        rows_loaded, nnz_loaded = _load_rows(
            highs,
            program,
            record,
            max_rows=max_rows,
            small_matrix_value=small_matrix_value,
            large_matrix_value=large_matrix_value,
            infinite_bound=infinite_bound,
        )
        model_status = "not_run"
        run_status = "not_run"
        objective_result: Optional[float] = None
        continuous = _readonly(())
        binary = _readonly(())
        incumbent_validated = False
        if solve:
            status = highs.run()
            _require_highs_ok(status, "run")
            run_status = str(status)
            native_model_status = highs.getModelStatus()
            model_status = str(native_model_status)
            if native_model_status == _highspy.HighsModelStatus.kOptimal:
                solution = highs.getSolution()
                values = np.asarray(solution.col_value, dtype=np.float64).reshape(-1)
                columns = len(record.continuous_ids) + len(record.binary_ids)
                if values.size != columns or not np.all(np.isfinite(values)):
                    raise ConstraintProgramHighsError("HiGHS returned a malformed incumbent")
                n_cont = len(record.continuous_ids)
                x = np.array(values[:n_cont], dtype=np.float64, copy=True)
                z = np.array(values[n_cont:], dtype=np.float64, copy=True)
                rounded = np.rint(z)
                if np.any(np.abs(z - rounded) > tolerance) or np.any(
                    (rounded != 0.0) & (rounded != 1.0)
                ):
                    raise ConstraintProgramHighsError(
                        "HiGHS optimal incumbent is not binary within tolerance"
                    )
                xi = 2.0 * rounded - 1.0
                _validate_incumbent(
                    program,
                    record,
                    x,
                    xi,
                    max_rows=max_rows,
                    tolerance=tolerance,
                )
                continuous = _readonly(x)
                binary = _readonly(xi)
                incumbent_validated = True
                native_objective = float(highs.getObjectiveValue())
                exact_objective = Fraction(0)
                for coefficient, value in zip(objective[:n_cont], x):
                    exact_objective += Fraction.from_float(
                        float(coefficient)
                    ) * Fraction.from_float(float(value))
                for coefficient, value in zip(objective[n_cont:], xi):
                    exact_objective += Fraction.from_float(
                        float(coefficient)
                    ) * Fraction.from_float(float(value))
                objective_result = _nearest_float(
                    exact_objective,
                    name="validated original-coordinate objective",
                )
                if not math.isfinite(native_objective):
                    raise ConstraintProgramHighsError(
                        "HiGHS optimal objective is non-finite"
                    )
        result = ConstraintProgramHighsResult(
            model_status=model_status,
            run_status=run_status,
            objective_value=objective_result,
            continuous=continuous,
            binary=binary,
            rows_loaded=rows_loaded,
            nnz_loaded=nnz_loaded,
            incumbent_validated=incumbent_validated,
            receipt=_receipt(
                record,
                max_rows=max_rows,
                native_model_loaded=True,
                native_solver_run=solve,
                rows_loaded=rows_loaded,
                nnz_loaded=nnz_loaded,
                incumbent_validated=incumbent_validated,
            ),
        )
    except BaseException as error:
        primary = error
    finally:
        if highs is not None:
            first_cleanup_error: Optional[BaseException] = None
            cleared = False
            for _attempt in range(4):
                try:
                    _clear_highs(highs)
                    cleared = True
                    break
                except BaseException as cleanup_error:
                    if first_cleanup_error is None:
                        first_cleanup_error = cleanup_error
            if first_cleanup_error is not None or not cleared:
                cleanup_error = (
                    first_cleanup_error
                    if first_cleanup_error is not None
                    else ConstraintProgramHighsError(
                        "HiGHS cleanup did not converge"
                    )
                )
                if primary is None:
                    primary = cleanup_error
                else:
                    _note(
                        primary,
                        "HiGHS cleanup also failed",
                    )
        try:
            _validated_handoff(handoff, expected_record=record)
        except BaseException as graph_error:
            if primary is None:
                primary = graph_error
            else:
                _note(
                    primary,
                    "handoff live-graph final validation also failed",
                )
    if primary is not None:
        raise primary
    if result is None:
        raise ConstraintProgramHighsError("native loader produced no result")
    return result


def source_sha256() -> str:
    """Diagnostic-only source hash; never accepted as runtime authenticity."""

    with open(__file__, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


__all__ = (
    "ConstraintProgramHighsError",
    "ConstraintProgramHighsFactory",
    "ConstraintProgramHighsHandoff",
    "ConstraintProgramHighsResult",
    "bind_constraint_program_highs",
    "source_sha256",
)
