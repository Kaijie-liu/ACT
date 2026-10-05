"""Exact known-schema roots and protocol-5 bytearray ownership, not heap GC."""

import json

import numpy as np
import torch

from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InputSpec, OutputSpec
from experiments.neural_hz_20260831 import s0_c2_whole_state_ledger_prototype as core

SCHEMAS = {
    LabeledInputTensor: frozenset(("tensor", "label")),
    InputSpec: frozenset(("kind", "lb", "ub", "center", "eps", "A", "b", "p_norm",
                          "perturbed_positions", "budget", "synonym_table")),
    OutputSpec: frozenset(("kind", "c", "d", "y_true", "margin", "lb", "ub")),
}


def parameter_roots(net):
    roots, active = {}, set()

    def visit(value, path):
        if type(value) is np.ndarray or isinstance(value, torch.Tensor):
            label = json.dumps(path, separators=(",", ":"))
            if label in roots:
                raise ValueError("duplicate parameter root path")
            roots[label] = value
            return
        if value is None or type(value) in (int, float, str, bool) or isinstance(value, (np.generic, torch.dtype, torch.device)):
            return
        if id(value) in active:
            raise ValueError("cyclic parameter container")
        active.add(id(value))
        try:
            if type(value) in SCHEMAS:
                payload = vars(value)
                if frozenset(payload) != SCHEMAS[type(value)]:
                    raise ValueError(f"unregistered schema fields: {type(value).__name__}")
                for name in sorted(payload):
                    visit(payload[name], (*path, name))
            elif type(value) is dict:
                for key, item in value.items():
                    if type(key) not in (str, int):
                        raise ValueError("unregistered parameter key")
                    visit(item, (*path, key))
            elif type(value) in (tuple, list):
                for index, item in enumerate(value):
                    visit(item, (*path, index))
            else:
                raise ValueError(f"unknown parameter payload: {type(value).__name__}")
        finally:
            active.remove(id(value))

    for layer in net.layers:
        visit(layer.params, ("network", layer.id))
    return roots


class KnownBufferVisitor(core._NumericVisitor):
    def __init__(self):
        super().__init__()
        self.buffer_spans = {}
        self.buffer_dtypes = {}

    def _visit_numpy(self, array, role, *, entry_semantics):
        if type(array) is not np.ndarray:
            return super()._visit_numpy(array, role, entry_semantics=entry_semantics)
        owner = array
        seen = set()
        while isinstance(owner.base, np.ndarray):
            if id(owner) in seen or type(owner.base) is not np.ndarray:
                raise core.WholeStateReject(f"invalid_numpy_owner_chain:{role}")
            seen.add(id(owner))
            owner = owner.base
        view = owner.base
        if type(view) is not memoryview or type(view.obj) is not bytearray:
            return super()._visit_numpy(array, role, entry_semantics=entry_semantics)
        backing = view.obj
        if (view.format != "B" or view.ndim != 1 or not view.c_contiguous
                or view.readonly or view.nbytes != len(backing)):
            raise core.WholeStateReject(f"unsupported_bytearray_view:{role}")
        if array.dtype.hasobject or array.dtype.kind not in "biuf" or owner.dtype != array.dtype:
            raise core.WholeStateReject(f"incompatible_bytearray_dtype:{role}")
        lower, upper = core._byte_bounds(array)
        owner_lower, owner_upper = core._byte_bounds(owner)
        # This uint8 view is non-owning; it does not allocate/copy the backing data.
        byte_view = np.frombuffer(backing, dtype=np.uint8)
        base_lower, base_upper = core._byte_bounds(byte_view)
        if (upper - lower != array.nbytes or owner_upper - owner_lower != owner.nbytes
                or base_upper - base_lower != len(backing) or len(backing) % array.dtype.itemsize):
            raise core.WholeStateReject(f"gapped_or_misaligned_bytearray_view:{role}")
        if not (base_lower <= owner_lower <= lower <= upper <= owner_upper <= base_upper):
            raise core.WholeStateReject(f"bytearray_view_outside_owner:{role}")
        identity = id(backing)
        if identity in self.buffer_dtypes and self.buffer_dtypes[identity] != array.dtype:
            raise core.WholeStateReject(f"incompatible_bytearray_dtype_alias:{role}")
        self.buffer_dtypes[identity] = array.dtype
        intervals = self.buffer_spans.setdefault(identity, [])
        for start, stop in intervals:
            if max(start, lower) < min(stop, upper) and (start, stop) != (lower, upper):
                raise core.WholeStateReject(f"partial_bytearray_overlap:{role}")
        if (lower, upper) not in intervals:
            intervals.append((lower, upper))
        if entry_semantics == "csr_data":
            if (lower, upper) != (base_lower, base_upper) or array.nbytes != len(backing):
                raise core.WholeStateReject(f"csr_data_not_full_bytearray_owner:{role}")
            entries, semantics = array.size, "stored_csr_data_elements"
        elif entry_semantics == "csr_index":
            entries, semantics = 0, "csr_index_bytes_only"
        elif entry_semantics == "dense":
            entries, semantics = len(backing) // array.dtype.itemsize, "stored_numeric_elements"
        else:
            raise core.WholeStateReject("unknown_numpy_entry_semantics")
        self._mark_object(array, "numpy.ndarray")
        self._mark_object(backing, "known_pickle_bytearray")
        self._register_storage(key=("pickle_bytearray", identity), storage_kind="pickle_bytearray",
            resident_bytes=len(backing), resident_entries=int(entries), entry_semantics=semantics, role=role)


def snapshot_known_buffers(roots):
    snapshot = core._snapshot_roots(roots)
    visitor = KnownBufferVisitor()
    for namespace in ("sparse_hz", "affine_expr", "precomputed_relu", "phase_bounds",
                      "descriptor", "artifact", "active", "pending"):
        items = [(core._safe_key_label(key), key, value) for key, value in snapshot.maps[namespace].items()]
        for label, key, value in sorted(items, key=lambda item: item[0]):
            role = f"{namespace}[{label}]"
            expected = {"sparse_hz": core.SparseHZono, "affine_expr": core.SparseHZAffineExpr, "phase_bounds": core.Bounds}.get(namespace)
            if expected is not None and type(value) is not expected:
                raise core.WholeStateReject(f"malformed_registered_root:{role}")
            if namespace == "precomputed_relu":
                core._visit_precomputed(visitor, value, role)
            else:
                visitor.visit(value, role)
    return visitor.ledger()
