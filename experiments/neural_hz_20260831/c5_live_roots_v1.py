"""Known-schema live TF roots; immutable identity/payload fingerprint."""

from dataclasses import dataclass
import hashlib
import json
import sys
import weakref

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, Con, ConSet, Fact, Layer, Net
from act.back_end.hybridz_tf.exact_linear_op import CSRLinearOp, DiagonalLinearOp, ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr, SparseHZAffineTerm
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c5_explicit_schema_ledger_v2 import SCHEMAS
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots

KNOWN_FIELDS = {
    **SCHEMAS,
    Net: frozenset(("layers", "preds", "succs", "by_id", "_topo_cache")),
    Layer: frozenset(("id", "kind", "params", "in_vars", "out_vars", "cache")),
    Fact: frozenset(("bounds", "cons")),
    ConSet: frozenset(("S",)),
    Con: frozenset(("kind", "var_ids", "meta", "A", "b", "C", "d")),
}


@dataclass
class LiveRoots:
    numeric: dict
    fingerprint: str
    schema_counts: dict
    python_shallow_bytes: int
    unique_objects: int

    def measure(self, extra=None):
        return snapshot_partial_csr_owners(WholeStateRoots(active={**self.numeric, **(extra or {})}, consumer_gc_enabled=False))


def collect(tf, extra=None):
    numeric, seen, counts = {}, set(), {}
    digest = hashlib.sha256()
    shallow = 0
    source_hashes = {}

    def token(value):
        digest.update(str(value).encode())
        digest.update(b"\0")

    def payload(value):
        if sp.issparse(value):
            token(value.shape)
            for name in ("data", "indices", "indptr"):
                payload(getattr(value, name))
        elif isinstance(value, torch.Tensor):
            token((tuple(value.shape), str(value.dtype), str(value.device)))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            token((value.shape, str(value.dtype)))
            digest.update(value.tobytes())

    def source_hash(value):
        if id(value) not in source_hashes:
            source_hashes[id(value)] = source_digest(value)
        return source_hashes[id(value)]

    def operator_hash(value):
        token((id(value), type(value).__name__, value.shape))
        if sp.issparse(value):
            payload(value)
        elif type(value) is ImplicitConv2DOp:
            token((value.input_shape, value.output_shape, value._stride, value._padding, value._dilation, value._groups))
            payload(value._kernel)
            if value._row_mask is not None:
                payload(value._row_mask)
        elif type(value) is CSRLinearOp:
            payload(value._matrix)
        elif type(value) is DiagonalLinearOp:
            payload(value._diagonal)
        else:
            raise ValueError("unknown expression operator")

    def visit(value, path, active):
        nonlocal shallow
        token(path)
        if value is None or type(value) in (str, int, float, bool, bytes) or isinstance(value, (np.generic, torch.dtype, torch.device)):
            token((type(value).__name__, value))
            if id(value) not in seen:
                seen.add(id(value))
                shallow += sys.getsizeof(value)
            return
        identity = id(value)
        token((type(value).__name__, identity))
        if identity in active:
            raise ValueError(f"live root cycle: {path}")
        if identity in seen:
            return
        seen.add(identity)
        counts[type(value).__name__] = counts.get(type(value).__name__, 0) + 1
        shallow += sys.getsizeof(value)
        role = json.dumps(path, separators=(",", ":"))
        if type(value) is np.ndarray or isinstance(value, torch.Tensor) or sp.isspmatrix_csr(value):
            numeric[role] = value
            payload(value)
        elif type(value) is SparseHZono:
            numeric[role] = value
            token(source_hash(value))
        elif type(value) is Bounds:
            numeric[role] = value
            payload(value.lb)
            payload(value.ub)
        elif type(value) is SparseHZAffineExpr:
            numeric[role] = value
            token((value.n_out, value.frame_id))
            payload(value.bias)
            for term in value.terms:
                token((id(term.source), source_hash(term.source)))
                for op in term.operators:
                    operator_hash(op)
        elif type(value) is SparseHZAffineTerm:
            numeric[role] = value
            token((id(value.source), source_hash(value.source)))
            for op in value.operators:
                operator_hash(op)
        elif type(value) in (CSRLinearOp, DiagonalLinearOp, ImplicitConv2DOp):
            numeric[role] = value
            operator_hash(value)
        elif type(value) is weakref.WeakValueDictionary:
            # Values are weak; normal weak-entry expiry is not input mutation.
            token("weak_arena_values_not_strong_roots")
        else:
            active.add(identity)
            try:
                if type(value) in KNOWN_FIELDS:
                    fields = vars(value)
                    if frozenset(fields) != KNOWN_FIELDS[type(value)]:
                        raise ValueError(f"unknown live fields: {type(value).__name__}: {sorted(fields)}")
                    for name in sorted(fields):
                        visit(fields[name], (*path, name), active)
                elif type(value) is dict:
                    for key in sorted(value, key=repr):
                        visit(key, (*path, "key"), active)
                        visit(value[key], (*path, repr(key)), active)
                elif type(value) in (list, tuple, set, frozenset):
                    items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
                    for index, item in enumerate(items):
                        visit(item, (*path, index), active)
                else:
                    raise ValueError(f"unknown live root: {path}: {type(value).__name__}")
            finally:
                active.remove(identity)

    # Instance field names are all traversed; no unregistered field is skipped.
    for name, value in sorted(vars(tf).items()):
        visit(value, ("tf", name), set())
    for name, value in sorted((extra or {}).items()):
        visit(value, ("extra", name), set())
    return LiveRoots(numeric, digest.hexdigest(), counts, shallow, len(seen))
