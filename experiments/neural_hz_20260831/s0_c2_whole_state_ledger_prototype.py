"""Isolated whole-state numeric-storage ledger for S0-C2-v1.

This module is a pure, experiment-only prototype.  It does not inspect the
Python heap and it never mutates, publishes, or releases a production root.
Instead it snapshots the explicitly registered Neural-HZ roots, simulates one
request-local path-specific compare-and-swap rewrite plus the same consumer-GC
boundary on the baseline and candidate snapshots, and then applies two strict
gates::

    candidate.resident_bytes   < baseline.resident_bytes
    candidate.resident_entries < baseline.resident_entries

The frozen accounting convention is deliberately narrow:

* a dense NumPy/Torch backing storage contributes its physical bytes and its
  storage element count once;
* a SciPy CSR contributes bytes from ``data``, ``indices`` and ``indptr``, but
  entries from ``data.size`` only;
* a phase ``Bounds`` contributes both dense backing storages and their entries;
* Python allocator/object overhead, logical expanded nnz, and measured RSS are
  separate diagnostics and are not part of either strict gate.

NumPy aliases are keyed by the final ndarray owner allocation.  A dense view
keeps that entire allocation alive, so even a small slice is charged the final
owner's complete physical bytes and dense element count.  CSR bytes use the
same owner rule, but the entry convention remains ``data.size``; a CSR data
view that does not cover its complete owner fails closed.  View byte spans are
still audited: exact or disjoint views deduplicate the same owner, while
partly overlapping spans, gapped views, ndarray owners backed by an unknown
external buffer, and incompatible dtype aliases fail closed.  Torch aliases
use the untyped-storage device/data pointer/byte length internally, but pointer
values are never retained in public provenance.

Only the explicit production-shaped roots below are recognized.  This is not
an arbitrary Python object-graph walker, an allocator/RSS proof, an atomic
publication transaction, or evidence of a TinyImageNet/formal gain.
Graph consumer-managed sparse-HZ/expression/bounds cache roots cannot be
directly substituted; only the identical baseline/candidate GC simulation may
release them.  A future production adapter must expose its candidate through
request-local roots or add a separately audited graph-publication interface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
import operator
from typing import Final
import weakref

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.exact_linear_op import (
    CSRLinearOp,
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr,
    SparseHZAffineTerm,
)
from act.back_end.solver.solver_hz import SparseHZono


WHOLE_STATE_RULE_ID: Final = "s0_c2_whole_state_numeric_ledger_v1"

WHOLE_STATE_NO_CLAIMS: Final = (
    "python_allocator_and_object_payload_bytes_are_not_in_numeric_gate",
    "measured_peak_rss_is_not_in_numeric_gate",
    "logical_expanded_nnz_and_work_are_separate_diagnostics",
    "pure_snapshot_simulation_is_not_atomic_production_publication",
    "no_production_lineage_emission_tiny_hit_or_formal_gain_is_claimed",
)

ROOT_NAMESPACES: Final = frozenset(
    {
        "sparse_hz",
        "affine_expr",
        "precomputed_relu",
        "phase_bounds",
        "descriptor",
        "artifact",
        "active",
        "pending",
    }
)

# These roots are owned by graph consumer accounting.  A hypothetical
# candidate may add/replace request-local active/descriptor/artifact state,
# but it may not manufacture a reduction by directly deleting or replacing a
# graph cache entry.  Only _simulate_consumer_gc may release these namespaces.
GRAPH_GC_MANAGED_NAMESPACES: Final = frozenset(
    {"sparse_hz", "affine_expr", "phase_bounds"}
)


class WholeStateReject(ValueError):
    """Stable fail-closed rejection from the isolated ledger."""


class _MissingRoot:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic convenience
        return "ROOT_MISSING"


ROOT_MISSING: Final = _MissingRoot()


@dataclass(frozen=True)
class WholeStateRoots:
    """The complete explicit strong-root boundary used by the prototype.

    Mappings are borrowed read-only.  ``snapshot_whole_state`` copies their
    key/value bindings before walking them; contained objects are never copied
    or mutated.  Extra-root namespaces make descriptor/artifact ownership and
    active/pending transaction retention explicit instead of guessing it from
    the Python heap.
    """

    sparse_hz: Mapping[object, object] = field(default_factory=dict)
    affine_expr: Mapping[object, object] = field(default_factory=dict)
    precomputed_relu: Mapping[object, object] = field(default_factory=dict)
    phase_bounds: Mapping[object, object] = field(default_factory=dict)
    descriptor: Mapping[object, object] = field(default_factory=dict)
    artifact: Mapping[object, object] = field(default_factory=dict)
    active: Mapping[object, object] = field(default_factory=dict)
    pending: Mapping[object, object] = field(default_factory=dict)
    remaining_consumers: Mapping[object, object] = field(default_factory=dict)
    pinned_layers: frozenset[object] = frozenset()
    consumer_gc_enabled: bool = True


@dataclass(frozen=True)
class RootPath:
    """One exact root binding, optionally one precomputed tuple slot."""

    namespace: str
    key: object
    slot: int | None = None


@dataclass(frozen=True)
class RootSubstitution:
    """Identity-CAS mutation applied only to a private root snapshot.

    ``expected is ROOT_MISSING`` is the only way to insert a new root.  A
    whole binding can be removed with ``remove=True``.  Individual
    precomputed slots can be replaced but not removed because removal would
    create a malformed production tuple.
    """

    path: RootPath
    expected: object
    replacement: object = None
    remove: bool = False


@dataclass(frozen=True)
class ConsumerGCStep:
    """Graph predecessor occurrences consumed at one boundary."""

    predecessors: tuple[object, ...]


@dataclass(frozen=True)
class StrongRetentionPlan:
    """Explicit supported plan object whose strong roots must be charged."""

    roots: tuple[object, ...]
    label: str = "pending_plan"


@dataclass(frozen=True)
class StorageProvenance:
    """Address-free public provenance for one deduplicated backing span."""

    token: str
    storage_kind: str
    resident_bytes: int
    resident_entries: int
    entry_semantics: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class WholeStateLedger:
    """Immutable numeric ledger; object/RSS claims remain explicitly separate."""

    rule_id: str
    resident_bytes: int
    resident_entries: int
    numeric_storage_count: int
    storage_provenance: tuple[StorageProvenance, ...]
    object_counts: tuple[tuple[str, int], ...]
    live_weak_references_ignored: int
    dead_weak_references_ignored: int
    csr_entry_convention: str
    dense_entry_convention: str
    phase_bounds_entries_included: bool
    python_object_bytes_included: bool
    measured_rss_bytes: None
    no_claims: tuple[str, ...] = WHOLE_STATE_NO_CLAIMS

    def object_count(self, kind: str) -> int:
        return dict(self.object_counts).get(kind, 0)


@dataclass(frozen=True)
class WholeStateDecision:
    """Root-free result of one hypothetical comparison.

    The result intentionally retains no input root, expected-CAS object,
    replacement object, or plan.  A future production transaction must repeat
    the comparison under its publication lock.
    """

    accepted: bool
    reason: str
    before: WholeStateLedger | None
    after: WholeStateLedger | None
    baseline_released_paths: tuple[str, ...]
    candidate_released_paths: tuple[str, ...]
    remaining_consumers_after: tuple[tuple[str, int], ...]
    no_claims: tuple[str, ...] = WHOLE_STATE_NO_CLAIMS


@dataclass
class _MutableRootSnapshot:
    maps: dict[str, dict[object, object]]
    remaining_consumers: dict[object, int]
    pinned_layers: frozenset[object]
    consumer_gc_enabled: bool

    def clone(self) -> "_MutableRootSnapshot":
        return _MutableRootSnapshot(
            maps={name: dict(values) for name, values in self.maps.items()},
            remaining_consumers=dict(self.remaining_consumers),
            pinned_layers=self.pinned_layers,
            consumer_gc_enabled=self.consumer_gc_enabled,
        )


@dataclass
class _StorageRecord:
    storage_kind: str
    resident_bytes: int
    resident_entries: int
    entry_semantics: str
    roles: set[str]


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise WholeStateReject(f"{name}_must_be_nonnegative_integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise WholeStateReject(f"{name}_must_be_nonnegative_integer") from exc
    parsed = int(parsed)
    if parsed < 0:
        raise WholeStateReject(f"{name}_must_be_nonnegative_integer")
    return parsed


def _safe_key_label(key: object) -> str:
    """Return a stable address-free label for production-shaped root keys."""

    if isinstance(key, str):
        return repr(key)
    if isinstance(key, (int, np.integer)) and not isinstance(
        key, (bool, np.bool_)
    ):
        return str(int(key))
    if key is None:
        return "None"
    if isinstance(key, tuple):
        return "(" + ",".join(_safe_key_label(value) for value in key) + ")"
    raise WholeStateReject(
        f"unsupported_root_key_type:{type(key).__name__}"
    )


def _snapshot_mapping(value: object, *, name: str) -> dict[object, object]:
    if not isinstance(value, Mapping):
        raise WholeStateReject(f"{name}_must_be_mapping")
    try:
        items = tuple(value.items())
    except Exception as exc:
        raise WholeStateReject(f"{name}_snapshot_failed") from exc
    result: dict[object, object] = {}
    for key, item in items:
        _safe_key_label(key)
        try:
            if key in result:
                raise WholeStateReject(f"{name}_duplicate_key")
            result[key] = item
        except (TypeError, ValueError) as exc:
            if isinstance(exc, WholeStateReject):
                raise
            raise WholeStateReject(f"{name}_invalid_key") from exc
    return result


def _snapshot_roots(roots: WholeStateRoots) -> _MutableRootSnapshot:
    if type(roots) is not WholeStateRoots:
        raise WholeStateReject("roots_must_be_WholeStateRoots")
    if type(roots.consumer_gc_enabled) is not bool:
        raise WholeStateReject("consumer_gc_enabled_must_be_bool")

    maps = {
        name: _snapshot_mapping(getattr(roots, name), name=name)
        for name in (
            "sparse_hz",
            "affine_expr",
            "precomputed_relu",
            "phase_bounds",
            "descriptor",
            "artifact",
            "active",
            "pending",
        )
    }
    raw_remaining = _snapshot_mapping(
        roots.remaining_consumers, name="remaining_consumers"
    )
    remaining = {
        key: _strict_nonnegative_int(
            value, name=f"remaining_consumers[{_safe_key_label(key)}]"
        )
        for key, value in raw_remaining.items()
    }
    if not isinstance(roots.pinned_layers, frozenset):
        raise WholeStateReject("pinned_layers_must_be_frozenset")
    for key in roots.pinned_layers:
        _safe_key_label(key)
    return _MutableRootSnapshot(
        maps=maps,
        remaining_consumers=remaining,
        pinned_layers=roots.pinned_layers,
        consumer_gc_enabled=roots.consumer_gc_enabled,
    )


def _byte_bounds(array: np.ndarray) -> tuple[int, int]:
    # NumPy 2 moved byte_bounds; support both names without changing the
    # frozen accounting semantics.
    function = getattr(np, "byte_bounds", None)
    if function is None:
        function = np.lib.array_utils.byte_bounds
    lower, upper = function(array)
    return int(lower), int(upper)


class _NumericVisitor:
    def __init__(self) -> None:
        self._seen_objects: set[int] = set()
        self._active_compounds: set[int] = set()
        self._object_counts: Counter[str] = Counter()
        self._storage_records: dict[tuple[object, ...], _StorageRecord] = {}
        self._numpy_owner_intervals: dict[
            int, list[tuple[int, int, tuple[object, ...]]]
        ] = {}
        self._numpy_owner_tokens: dict[int, str] = {}
        self._live_weak = 0
        self._dead_weak = 0

    def _mark_object(self, value: object, kind: str) -> bool:
        identity = id(value)
        if identity in self._seen_objects:
            return False
        self._seen_objects.add(identity)
        self._object_counts[kind] += 1
        return True

    def _enter_compound(self, value: object, kind: str, role: str) -> None:
        """Count objects once but traverse every strong-root alias role."""

        identity = id(value)
        if identity in self._active_compounds:
            raise WholeStateReject(f"strong_root_cycle:{kind}:{role}")
        self._mark_object(value, kind)
        self._active_compounds.add(identity)

    def _leave_compound(self, value: object) -> None:
        self._active_compounds.remove(id(value))

    def _register_storage(
        self,
        *,
        key: tuple[object, ...],
        storage_kind: str,
        resident_bytes: int,
        resident_entries: int,
        entry_semantics: str,
        role: str,
    ) -> None:
        if resident_bytes < 0 or resident_entries < 0:
            raise WholeStateReject("negative_numeric_storage_metric")
        existing = self._storage_records.get(key)
        if existing is None:
            self._storage_records[key] = _StorageRecord(
                storage_kind=storage_kind,
                resident_bytes=resident_bytes,
                resident_entries=resident_entries,
                entry_semantics=entry_semantics,
                roles={role},
            )
            return
        if (
            existing.storage_kind != storage_kind
            or existing.resident_bytes != resident_bytes
            or existing.resident_entries != resident_entries
            or existing.entry_semantics != entry_semantics
        ):
            raise WholeStateReject(f"incompatible_storage_alias:{role}")
        existing.roles.add(role)

    def _visit_numpy(
        self,
        array: np.ndarray,
        role: str,
        *,
        entry_semantics: str,
    ) -> None:
        if type(array) is not np.ndarray:
            raise WholeStateReject(
                f"unknown_numpy_subclass:{type(array).__name__}:{role}"
            )
        self._mark_object(array, "numpy.ndarray")
        if array.dtype.hasobject or array.dtype.kind not in "biuf":
            raise WholeStateReject(f"unknown_numpy_numeric_dtype:{role}")

        owner = array
        owner_chain: set[int] = set()
        while isinstance(owner.base, np.ndarray):
            if id(owner) in owner_chain:
                raise WholeStateReject(f"numpy_base_cycle:{role}")
            owner_chain.add(id(owner))
            owner = owner.base
            if type(owner) is not np.ndarray:
                raise WholeStateReject(
                    f"unknown_numpy_owner_subclass:{role}"
                )
        if owner.base is not None:
            raise WholeStateReject(f"unknown_numpy_external_buffer:{role}")

        lower, upper = _byte_bounds(array)
        span_bytes = upper - lower
        if span_bytes < 0:
            raise WholeStateReject(f"invalid_numpy_byte_span:{role}")
        if span_bytes != int(array.nbytes):
            # A byte span containing unreachable gaps cannot be assigned a
            # unique dense-entry count under the frozen convention.
            raise WholeStateReject(f"gapped_numpy_view:{role}")

        owner_id = id(owner)
        owner_token = self._numpy_owner_tokens.setdefault(
            owner_id, f"numpy-owner-{len(self._numpy_owner_tokens) + 1:04d}"
        )
        owner_lower, owner_upper = _byte_bounds(owner)
        owner_span_bytes = owner_upper - owner_lower
        if owner_span_bytes < 0 or owner_span_bytes != int(owner.nbytes):
            raise WholeStateReject(f"gapped_numpy_owner:{role}")
        if lower < owner_lower or upper > owner_upper:
            raise WholeStateReject(f"numpy_view_outside_owner:{role}")
        if array.dtype != owner.dtype:
            raise WholeStateReject(f"incompatible_numpy_dtype_alias:{role}")

        # A live view strongly retains the complete final owner allocation.
        # The view span is therefore lineage/audit evidence, not the physical
        # storage key or the resident-byte quantity.
        key = ("numpy", owner_id)
        for old_lower, old_upper, old_key in self._numpy_owner_intervals.get(
            owner_id, ()
        ):
            exact = lower == old_lower and upper == old_upper
            overlap = max(lower, old_lower) < min(upper, old_upper)
            if overlap and not exact:
                raise WholeStateReject(
                    f"partial_numpy_overlap:{owner_token}:{role}"
                )
            if exact and old_key != key:  # pragma: no cover - defensive
                raise WholeStateReject(f"numpy_span_key_mismatch:{role}")
        if not any(
            lower == old_lower and upper == old_upper
            for old_lower, old_upper, _ in self._numpy_owner_intervals.get(
                owner_id, ()
            )
        ):
            self._numpy_owner_intervals.setdefault(owner_id, []).append(
                (lower, upper, key)
            )

        if entry_semantics == "csr_index":
            entries = 0
            normalized_semantics = "csr_index_bytes_only"
        elif entry_semantics == "csr_data":
            # The frozen entry gate is logical CSR data.size, while the byte
            # gate charges the complete owner allocation.  A short CSR data
            # view would require a separate interval-union entry ledger; this
            # prototype rejects it instead of substituting owner.size and
            # creating a false coefficient-entry decrease.
            if (
                lower != owner_lower
                or upper != owner_upper
                or int(array.size) != int(owner.size)
            ):
                raise WholeStateReject(f"csr_data_not_full_owner:{role}")
            entries = int(array.size)
            normalized_semantics = "stored_csr_data_elements"
        elif entry_semantics == "dense":
            entries = int(owner.size)
            normalized_semantics = "stored_numeric_elements"
        else:  # pragma: no cover - only internal callers select semantics
            raise WholeStateReject("unknown_numpy_entry_semantics")
        self._register_storage(
            key=key,
            storage_kind="numpy",
            resident_bytes=owner_span_bytes,
            resident_entries=entries,
            entry_semantics=normalized_semantics,
            role=role,
        )

    def _visit_torch(self, tensor: torch.Tensor, role: str) -> None:
        self._mark_object(tensor, "torch.Tensor")
        if tensor.layout != torch.strided or tensor.is_sparse:
            raise WholeStateReject(f"unknown_torch_layout:{role}")
        if tensor.is_quantized or tensor.is_complex():
            raise WholeStateReject(f"unknown_torch_numeric_dtype:{role}")
        if tensor.device.type == "meta":
            raise WholeStateReject(f"unknown_torch_meta_storage:{role}")
        try:
            storage = tensor.untyped_storage()
            storage_bytes = int(storage.nbytes())
            storage_pointer = int(storage.data_ptr())
            element_bytes = int(tensor.element_size())
        except Exception as exc:
            raise WholeStateReject(f"torch_storage_unavailable:{role}") from exc
        if storage_bytes < 0 or element_bytes <= 0:
            raise WholeStateReject(f"invalid_torch_storage:{role}")
        if storage_bytes % element_bytes:
            raise WholeStateReject(f"incompatible_torch_dtype_alias:{role}")
        entries = storage_bytes // element_bytes
        # The pointer is used only in this ephemeral private key.  Public
        # provenance below assigns address-free sequential storage tokens.
        empty_disambiguator = id(storage) if storage_bytes == 0 else None
        key = (
            "torch",
            str(storage.device),
            storage_pointer,
            storage_bytes,
            empty_disambiguator,
        )
        self._register_storage(
            key=key,
            storage_kind="torch",
            resident_bytes=storage_bytes,
            resident_entries=entries,
            entry_semantics="stored_numeric_elements",
            role=role,
        )

    @staticmethod
    def _is_csr(value: object) -> bool:
        csr_types = (sp.csr_matrix,)
        csr_array = getattr(sp, "csr_array", None)
        if csr_array is not None:
            csr_types = csr_types + (csr_array,)
        return isinstance(value, csr_types)

    def _visit_csr(self, matrix: object, role: str) -> None:
        if not self._is_csr(matrix):
            raise WholeStateReject(f"unknown_sparse_format:{role}")
        self._mark_object(matrix, "scipy.csr")
        if matrix.ndim != 2:
            raise WholeStateReject(f"csr_must_be_two_dimensional:{role}")
        if matrix.data.dtype.kind not in "biuf":
            raise WholeStateReject(f"unknown_csr_data_dtype:{role}")
        if matrix.indices.dtype.kind not in "iu" or matrix.indptr.dtype.kind not in "iu":
            raise WholeStateReject(f"unknown_csr_index_dtype:{role}")
        if int(matrix.data.size) != int(matrix.indices.size):
            raise WholeStateReject(f"csr_data_index_size_mismatch:{role}")
        self._visit_numpy(
            matrix.data, f"{role}.data", entry_semantics="csr_data"
        )
        self._visit_numpy(
            matrix.indices, f"{role}.indices", entry_semantics="csr_index"
        )
        self._visit_numpy(
            matrix.indptr, f"{role}.indptr", entry_semantics="csr_index"
        )

    def _visit_operator(self, value: object, role: str) -> None:
        if self._is_csr(value):
            self._visit_csr(value, role)
            return
        if type(value) is CSRLinearOp:
            self._mark_object(value, "operator.CSRLinearOp")
            matrix = getattr(value, "_matrix", None)
            if not self._is_csr(matrix):
                raise WholeStateReject(f"malformed_CSRLinearOp:{role}")
            self._visit_csr(matrix, f"{role}.matrix")
            return
        if type(value) is DiagonalLinearOp:
            self._mark_object(value, "operator.DiagonalLinearOp")
            diagonal = getattr(value, "_diagonal", None)
            if type(diagonal) is not np.ndarray:
                raise WholeStateReject(f"malformed_DiagonalLinearOp:{role}")
            self._visit_numpy(
                diagonal, f"{role}.diagonal", entry_semantics="dense"
            )
            return
        if type(value) is ImplicitConv2DOp:
            self._mark_object(value, "operator.ImplicitConv2DOp")
            kernel = getattr(value, "_kernel", None)
            row_mask = getattr(value, "_row_mask", ROOT_MISSING)
            if type(kernel) is not np.ndarray or row_mask is ROOT_MISSING:
                raise WholeStateReject(f"malformed_ImplicitConv2DOp:{role}")
            self._visit_numpy(
                kernel, f"{role}.kernel", entry_semantics="dense"
            )
            if row_mask is not None:
                if type(row_mask) is not np.ndarray:
                    raise WholeStateReject(
                        f"malformed_ImplicitConv2DOp_mask:{role}"
                    )
                self._visit_numpy(
                    row_mask, f"{role}.row_mask", entry_semantics="dense"
                )
            return
        raise WholeStateReject(f"unknown_operator:{type(value).__name__}:{role}")

    def _visit_hz(self, hz: SparseHZono, role: str) -> None:
        self._enter_compound(hz, "SparseHZono", role)
        try:
            for field_name in ("c",):
                self.visit(
                    getattr(hz, field_name), f"{role}.value.{field_name}"
                )
            for field_name in ("Gc", "Gb"):
                matrix = getattr(hz, field_name)
                if not self._is_csr(matrix):
                    raise WholeStateReject(
                        f"malformed_hz_value:{field_name}"
                    )
                self._visit_csr(matrix, f"{role}.value.{field_name}")
            for field_name in ("Ac", "Ab", "Auc", "Aub"):
                matrix = getattr(hz, field_name)
                if not self._is_csr(matrix):
                    raise WholeStateReject(
                        f"malformed_hz_predicate:{field_name}"
                    )
                self._visit_csr(
                    matrix, f"{role}.predicate.{field_name}"
                )
            for field_name in ("b", "ub"):
                self.visit(
                    getattr(hz, field_name),
                    f"{role}.predicate.{field_name}",
                )
        finally:
            self._leave_compound(hz)

    def _visit_term(self, term: SparseHZAffineTerm, role: str) -> None:
        self._enter_compound(term, "SparseHZAffineTerm", role)
        try:
            if type(term.source) is not SparseHZono:
                raise WholeStateReject(f"unknown_affine_source:{role}")
            self._visit_hz(term.source, f"{role}.source")
            if type(term.operators) is not tuple:
                raise WholeStateReject(
                    f"affine_operators_must_be_tuple:{role}"
                )
            for index, affine_operator in enumerate(term.operators):
                self._visit_operator(
                    affine_operator, f"{role}.operators[{index}]"
                )
        finally:
            self._leave_compound(term)

    def _visit_expr(self, expr: SparseHZAffineExpr, role: str) -> None:
        self._enter_compound(expr, "SparseHZAffineExpr", role)
        try:
            if type(expr.bias) is not np.ndarray:
                raise WholeStateReject(f"unknown_affine_bias:{role}")
            self._visit_numpy(
                expr.bias, f"{role}.bias", entry_semantics="dense"
            )
            if type(expr.terms) is not tuple:
                raise WholeStateReject(f"affine_terms_must_be_tuple:{role}")
            for index, term in enumerate(expr.terms):
                if type(term) is not SparseHZAffineTerm:
                    raise WholeStateReject(
                        f"unknown_affine_term:{type(term).__name__}:{role}"
                    )
                self._visit_term(term, f"{role}.terms[{index}]")
        finally:
            self._leave_compound(expr)

    def _visit_bounds(self, bounds: Bounds, role: str) -> None:
        self._enter_compound(bounds, "Bounds", role)
        try:
            if not isinstance(bounds.lb, torch.Tensor) or not isinstance(
                bounds.ub, torch.Tensor
            ):
                raise WholeStateReject(
                    f"unknown_phase_bounds_payload:{role}"
                )
            self._visit_torch(bounds.lb, f"{role}.lb")
            self._visit_torch(bounds.ub, f"{role}.ub")
        finally:
            self._leave_compound(bounds)

    def _visit_plan(self, plan: StrongRetentionPlan, role: str) -> None:
        self._enter_compound(plan, "StrongRetentionPlan", role)
        try:
            if type(plan.roots) is not tuple or type(plan.label) is not str:
                raise WholeStateReject(
                    f"malformed_strong_retention_plan:{role}"
                )
            for index, retained in enumerate(plan.roots):
                self.visit(retained, f"{role}.strong_roots[{index}]")
        finally:
            self._leave_compound(plan)

    def visit(self, value: object, role: str) -> None:
        if value is None:
            return
        if isinstance(value, weakref.ReferenceType):
            self._mark_object(value, "weakref")
            if value() is None:
                self._dead_weak += 1
            else:
                self._live_weak += 1
            # A weak reference is never promoted into a strong ledger root.
            return
        if type(value) is np.ndarray:
            self._visit_numpy(value, role, entry_semantics="dense")
            return
        if isinstance(value, torch.Tensor):
            self._visit_torch(value, role)
            return
        if self._is_csr(value):
            self._visit_csr(value, role)
            return
        if type(value) is SparseHZono:
            self._visit_hz(value, role)
            return
        if type(value) is SparseHZAffineExpr:
            self._visit_expr(value, role)
            return
        if type(value) is SparseHZAffineTerm:
            self._visit_term(value, role)
            return
        if type(value) is Bounds:
            self._visit_bounds(value, role)
            return
        if type(value) is StrongRetentionPlan:
            self._visit_plan(value, role)
            return
        if type(value) in (CSRLinearOp, DiagonalLinearOp, ImplicitConv2DOp):
            self._visit_operator(value, role)
            return
        if sp.issparse(value):
            raise WholeStateReject(f"unknown_sparse_format:{role}")
        if isinstance(value, (memoryview, bytearray, bytes)) or hasattr(
            value, "__array_interface__"
        ):
            raise WholeStateReject(
                f"unknown_numeric_root:{type(value).__name__}:{role}"
            )
        raise WholeStateReject(
            f"unsupported_strong_root:{type(value).__name__}:{role}"
        )

    def ledger(self) -> WholeStateLedger:
        provenance: list[StorageProvenance] = []
        for index, record in enumerate(self._storage_records.values(), start=1):
            provenance.append(
                StorageProvenance(
                    token=f"storage-{index:04d}",
                    storage_kind=record.storage_kind,
                    resident_bytes=record.resident_bytes,
                    resident_entries=record.resident_entries,
                    entry_semantics=record.entry_semantics,
                    roles=tuple(sorted(record.roles)),
                )
            )
        return WholeStateLedger(
            rule_id=WHOLE_STATE_RULE_ID,
            resident_bytes=sum(item.resident_bytes for item in provenance),
            resident_entries=sum(
                item.resident_entries for item in provenance
            ),
            numeric_storage_count=len(provenance),
            storage_provenance=tuple(provenance),
            object_counts=tuple(sorted(self._object_counts.items())),
            live_weak_references_ignored=self._live_weak,
            dead_weak_references_ignored=self._dead_weak,
            csr_entry_convention="data.size_only;indices_and_indptr_bytes_only",
            dense_entry_convention="deduplicated_backing_storage_elements",
            phase_bounds_entries_included=True,
            python_object_bytes_included=False,
            measured_rss_bytes=None,
        )


def _visit_precomputed(
    visitor: _NumericVisitor, value: object, role: str
) -> None:
    if type(value) is not tuple or len(value) not in (3, 4, 5):
        raise WholeStateReject(f"malformed_precomputed_tuple:{role}")
    visitor._enter_compound(value, "precomputed_tuple", role)
    try:
        if type(value[0]) is not SparseHZono:
            raise WholeStateReject(f"precomputed_slot0_must_be_hz:{role}")
        visitor.visit(value[0], f"{role}.slot0")
        for slot in (1, 2):
            if type(value[slot]) is not np.ndarray and not isinstance(
                value[slot], torch.Tensor
            ):
                raise WholeStateReject(
                    f"precomputed_slot{slot}_must_be_dense:{role}"
                )
            visitor.visit(value[slot], f"{role}.slot{slot}")
        if len(value) >= 4:
            if (
                value[3] is not None
                and type(value[3]) is not SparseHZAffineExpr
            ):
                raise WholeStateReject(
                    f"precomputed_slot3_must_be_expr_or_none:{role}"
                )
            visitor.visit(value[3], f"{role}.slot3")
        if len(value) == 5:
            if value[4] is not None and type(value[4]) is not Bounds:
                raise WholeStateReject(
                    f"precomputed_slot4_must_be_bounds_or_none:{role}"
                )
            visitor.visit(value[4], f"{role}.slot4")
    finally:
        visitor._leave_compound(value)


def _measure_snapshot(snapshot: _MutableRootSnapshot) -> WholeStateLedger:
    visitor = _NumericVisitor()
    for namespace in (
        "sparse_hz",
        "affine_expr",
        "precomputed_relu",
        "phase_bounds",
        "descriptor",
        "artifact",
        "active",
        "pending",
    ):
        labeled_items = [
            (_safe_key_label(key), key, value)
            for key, value in snapshot.maps[namespace].items()
        ]
        for key_label, _, value in sorted(labeled_items, key=lambda item: item[0]):
            role = f"{namespace}[{key_label}]"
            if namespace == "sparse_hz":
                if type(value) is not SparseHZono:
                    raise WholeStateReject(f"sparse_hz_root_malformed:{role}")
                visitor.visit(value, role)
            elif namespace == "affine_expr":
                if type(value) is not SparseHZAffineExpr:
                    raise WholeStateReject(f"affine_expr_root_malformed:{role}")
                visitor.visit(value, role)
            elif namespace == "precomputed_relu":
                _visit_precomputed(visitor, value, role)
            elif namespace == "phase_bounds":
                if type(value) is not Bounds:
                    raise WholeStateReject(f"phase_bounds_root_malformed:{role}")
                visitor.visit(value, role)
            else:
                visitor.visit(value, role)
    return visitor.ledger()


def snapshot_whole_state(roots: WholeStateRoots) -> WholeStateLedger:
    """Measure supported strong roots without mutating input containers."""

    return _measure_snapshot(_snapshot_roots(roots))


def _path_binding(
    snapshot: _MutableRootSnapshot, path: RootPath
) -> tuple[bool, object]:
    if type(path) is not RootPath or path.namespace not in ROOT_NAMESPACES:
        raise WholeStateReject("unknown_root_path_namespace")
    _safe_key_label(path.key)
    if path.slot is not None:
        if path.namespace != "precomputed_relu":
            raise WholeStateReject("root_path_slot_only_for_precomputed")
        slot = _strict_nonnegative_int(path.slot, name="root_path_slot")
        if slot > 4:
            raise WholeStateReject("root_path_slot_out_of_range")
    mapping = snapshot.maps[path.namespace]
    if path.key not in mapping:
        return False, ROOT_MISSING
    value = mapping[path.key]
    if path.slot is None:
        return True, value
    if type(value) is not tuple or path.slot >= len(value):
        raise WholeStateReject("root_path_precomputed_slot_missing")
    return True, value[path.slot]


def _apply_substitutions(
    snapshot: _MutableRootSnapshot,
    substitutions: tuple[RootSubstitution, ...],
) -> None:
    if type(substitutions) is not tuple:
        raise WholeStateReject("substitutions_must_be_tuple")
    seen_paths: set[tuple[str, object, int | None]] = set()
    for substitution in substitutions:
        if type(substitution) is not RootSubstitution:
            raise WholeStateReject("unknown_substitution_type")
        path = substitution.path
        if type(path) is not RootPath:
            raise WholeStateReject("unknown_root_path_type")
        if path.namespace in GRAPH_GC_MANAGED_NAMESPACES:
            raise WholeStateReject(
                "graph_gc_managed_root_substitution_forbidden"
            )
        try:
            path_key = (path.namespace, path.key, path.slot)
            if path_key in seen_paths:
                raise WholeStateReject("duplicate_root_substitution_path")
            seen_paths.add(path_key)
        except TypeError as exc:
            raise WholeStateReject("unhashable_root_substitution_path") from exc

        present, current = _path_binding(snapshot, path)
        if substitution.expected is ROOT_MISSING:
            if present:
                raise WholeStateReject("path_cas_expected_missing_but_present")
        elif not present or current is not substitution.expected:
            raise WholeStateReject("path_cas_expected_identity_mismatch")

        mapping = snapshot.maps[path.namespace]
        if substitution.remove:
            if path.slot is not None:
                raise WholeStateReject("cannot_remove_precomputed_slot")
            if substitution.expected is ROOT_MISSING:
                raise WholeStateReject("cannot_remove_missing_root")
            del mapping[path.key]
            continue
        if substitution.replacement is ROOT_MISSING:
            raise WholeStateReject("replacement_cannot_be_ROOT_MISSING")
        if path.slot is None:
            mapping[path.key] = substitution.replacement
        else:
            original_tuple = mapping[path.key]
            staged_tuple = list(original_tuple)
            staged_tuple[path.slot] = substitution.replacement
            mapping[path.key] = tuple(staged_tuple)


def _validate_gc_steps(steps: tuple[ConsumerGCStep, ...]) -> None:
    if type(steps) is not tuple:
        raise WholeStateReject("consumer_gc_steps_must_be_tuple")
    for step in steps:
        if type(step) is not ConsumerGCStep or type(step.predecessors) is not tuple:
            raise WholeStateReject("malformed_consumer_gc_step")
        for predecessor in step.predecessors:
            _safe_key_label(predecessor)


def _simulate_consumer_gc(
    snapshot: _MutableRootSnapshot,
    steps: tuple[ConsumerGCStep, ...],
) -> tuple[str, ...]:
    _validate_gc_steps(steps)
    if not snapshot.consumer_gc_enabled:
        return ()
    released: list[str] = []
    for step in steps:
        for predecessor in step.predecessors:
            remaining = snapshot.remaining_consumers.get(predecessor)
            if remaining is None or remaining <= 0:
                raise WholeStateReject("invalid_sparse_consumer_accounting")
            remaining -= 1
            snapshot.remaining_consumers[predecessor] = remaining
            if remaining or predecessor in snapshot.pinned_layers:
                continue
            key_label = _safe_key_label(predecessor)
            for namespace in ("sparse_hz", "affine_expr", "phase_bounds"):
                if predecessor in snapshot.maps[namespace]:
                    del snapshot.maps[namespace][predecessor]
                    released.append(f"{namespace}[{key_label}]")
    return tuple(released)


def _remaining_summary(
    remaining: Mapping[object, int]
) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (_safe_key_label(key), int(value))
            for key, value in remaining.items()
        )
    )


def assess_hypothetical_substitution(
    roots: WholeStateRoots,
    substitutions: tuple[RootSubstitution, ...],
    *,
    consumer_gc_steps: tuple[ConsumerGCStep, ...] = (),
) -> WholeStateDecision:
    """Apply identity-CAS substitutions to a private snapshot and gate it.

    Substitutions are staged before consumer GC.  The same GC steps are then
    applied independently to the untouched baseline snapshot and the staged
    candidate snapshot, so ordinary last-consumer release cannot masquerade as
    a candidate reduction.  Ordinary exceptions fail closed; asynchronous
    ``BaseException`` subclasses are deliberately not intercepted.
    """

    before_ledger: WholeStateLedger | None = None
    after_ledger: WholeStateLedger | None = None
    baseline_released: tuple[str, ...] = ()
    candidate_released: tuple[str, ...] = ()
    remaining_summary: tuple[tuple[str, int], ...] = ()
    try:
        initial = _snapshot_roots(roots)
        baseline = initial.clone()
        candidate = initial.clone()
        _apply_substitutions(candidate, substitutions)
        baseline_released = _simulate_consumer_gc(
            baseline, consumer_gc_steps
        )
        candidate_released = _simulate_consumer_gc(
            candidate, consumer_gc_steps
        )
        before_ledger = _measure_snapshot(baseline)
        after_ledger = _measure_snapshot(candidate)
        remaining_summary = _remaining_summary(candidate.remaining_consumers)
        if after_ledger.resident_bytes >= before_ledger.resident_bytes:
            return WholeStateDecision(
                accepted=False,
                reason="resident_bytes_not_strictly_reduced",
                before=before_ledger,
                after=after_ledger,
                baseline_released_paths=baseline_released,
                candidate_released_paths=candidate_released,
                remaining_consumers_after=remaining_summary,
            )
        if after_ledger.resident_entries >= before_ledger.resident_entries:
            return WholeStateDecision(
                accepted=False,
                reason="resident_entries_not_strictly_reduced",
                before=before_ledger,
                after=after_ledger,
                baseline_released_paths=baseline_released,
                candidate_released_paths=candidate_released,
                remaining_consumers_after=remaining_summary,
            )
        return WholeStateDecision(
            accepted=True,
            reason="accepted_strict_whole_state_reduction",
            before=before_ledger,
            after=after_ledger,
            baseline_released_paths=baseline_released,
            candidate_released_paths=candidate_released,
            remaining_consumers_after=remaining_summary,
        )
    except WholeStateReject as exc:
        return WholeStateDecision(
            accepted=False,
            reason=str(exc),
            before=before_ledger,
            after=after_ledger,
            baseline_released_paths=baseline_released,
            candidate_released_paths=candidate_released,
            remaining_consumers_after=remaining_summary,
        )
    except Exception as exc:
        return WholeStateDecision(
            accepted=False,
            reason=f"unexpected_ledger_error:{type(exc).__name__}",
            before=before_ledger,
            after=after_ledger,
            baseline_released_paths=baseline_released,
            candidate_released_paths=candidate_released,
            remaining_consumers_after=remaining_summary,
        )


__all__ = [
    "ConsumerGCStep",
    "ROOT_MISSING",
    "RootPath",
    "RootSubstitution",
    "StorageProvenance",
    "StrongRetentionPlan",
    "WHOLE_STATE_NO_CLAIMS",
    "WHOLE_STATE_RULE_ID",
    "WholeStateDecision",
    "WholeStateLedger",
    "WholeStateReject",
    "WholeStateRoots",
    "assess_hypothetical_substitution",
    "snapshot_whole_state",
]
