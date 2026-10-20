"""Default-off two-row gate/margin relation for the existing native F0 HZ.

No network, support, optimizer or gate-range query is made here. Sources are
supplied HZs, whose upstream conversion remains an explicit premise. The exact
checker lives outside the ACT import tree and has no numerical dependencies.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from scoped_source.hz_request_store import POLICY, decode, digest, require, sha
from scoped_source.check_gate_margin_rows import (
    BINDING, FACTORS, GATE_SEMANTICS, MATRICES, OUTPUTS, ROWS, SCHEMA, VECTORS,
    check, context, freeze, inspect, validate_binding,
)


@dataclass(frozen=True)
class GateMarginBinding:
    """Owned canonical bytes, not an alias to a mutable native HZ."""
    payload: bytes
    sha256: str


def snapshot(hz, work, *, f0=False):
    """Capture canonical native CSR entries without floating coalescing."""
    work.tick()
    require(hz.frame_id is not None and type(hz.frame_id) is int and type(hz.exact) is bool,
            "explicit native frame/exact flag")
    require(hz.n_cont + hz.n_bin <= FACTORS + (2 if f0 else 0)
            and hz.n_out <= OUTPUTS
            and hz.n_eq + hz.n_ineq <= ROWS + (6 if f0 else 0), "finite native source capacity")
    result = {"frame_id": hz.frame_id, "exact": bool(hz.exact)}
    for key in VECTORS:
        arr = getattr(hz, key)
        require(isinstance(arr, np.ndarray) and arr.dtype == np.float64 and arr.ndim == 1,
                "native float64 vector")
        result[key] = []
        for x in arr:
            work.visit("coefficient_visits")
            require(np.isfinite(x), "finite native vector")
            result[key].append(float(x))
    for key in MATRICES:
        mat = getattr(hz, key)
        require(sp.isspmatrix_csr(mat) and mat.dtype == np.float64, "native float64 CSR required")
        require(mat.data.ndim == mat.indices.ndim == mat.indptr.ndim == 1
                and np.issubdtype(mat.indices.dtype, np.integer)
                and np.issubdtype(mat.indptr.dtype, np.integer),
                "native CSR integral index buffers")
        nr = len(result["c" if key.startswith("G") else "b" if key in ("Ac", "Ab") else "ub"])
        width = hz.n_cont if key.endswith("c") else hz.n_bin
        require(mat.shape == (nr, width) and len(mat.data) == len(mat.indices)
                and len(mat.data) <= nr * width and len(mat.indptr) == nr + 1
                and int(mat.indptr[0]) == 0 and int(mat.indptr[-1]) == len(mat.data),
                "native CSR dimensions")
        values, indices, ptr = [], [], [0]
        for i in range(nr):
            work.visit("row_visits")
            lo, hi = map(int, mat.indptr[i:i + 2])
            require(0 <= lo <= hi <= len(mat.data), "native CSR pointer order")
            row = {}
            for p in range(lo, hi):
                work.visit("coefficient_visits")
                j, value = int(mat.indices[p]), float(mat.data[p])
                require(0 <= j < width and j not in row and math.isfinite(value), "native CSR duplicate/value")
                row[j] = value
            # Reordering unique stored entries is exact. Unlike sum_duplicates,
            # this performs no floating addition or coefficient normalization.
            work.visit("coefficient_visits", len(row))
            for j, value in sorted(row.items()):
                if value:
                    indices.append(j); values.append(value)
            ptr.append(len(indices))
        result[key] = {"shape": [nr, width], "data": values, "indices": indices, "indptr": ptr}
    work.tick()
    return result


def metadata(pair_hz):
    keys = ("shared_continuous", "shared_binary", "a_private_continuous", "b_private_continuous",
            "a_private_binary", "b_private_binary")
    require(all(type(getattr(pair_hz, k)) is int for k in keys), "native factor count type")
    return {**{k: getattr(pair_hz, k) for k in keys}, "a_rows": list(pair_hz.a_rows),
            "b_rows": list(pair_hz.b_rows), "relation_mode": pair_hz.relation_mode}


def bind_sources(*, entry, expert_a, expert_b, router, pair_hz, caller_context,
                 gate_semantics, router_eq_rows, router_le_rows, work):
    """Bind an unsliced shared pair and explicit retained router-row maps.

The caller declares which supplied expert belongs to each canonical pair slot;
no model/checkpoint identity is inferred from output block positions.
"""
    context(caller_context)
    require(gate_semantics == GATE_SEMANTICS, "selected-softmax top-2 semantics required")
    require(pair_hz.relation_mode == "shared_input", "gate relation requires shared input")
    live = {"entry": entry, "a": expert_a, "b": expert_b, "router": router,
            "joint": pair_hz.output_hz, "input": pair_hz.input_hz}
    value = {"schema": BINDING, "gate_semantics": gate_semantics, "context": caller_context,
             "sources": {k: snapshot(hz, work) for k, hz in live.items()},
             "metadata": metadata(pair_hz), "router_eq_rows": list(router_eq_rows),
             "router_le_rows": list(router_le_rows)}
    raw, anchor = freeze(value, work)
    require(len(raw) <= POLICY["header_bytes"], "bounded binding header")
    owned = decode(raw)
    validate_binding(owned, expected_hash=anchor, work=work)
    for key, hz in live.items():
        require(snapshot(hz, work) == owned["sources"][key], "native source changed while binding")
    require(metadata(pair_hz) == owned["metadata"] and caller_context == owned["context"],
            "native metadata/context changed")
    work.tick()
    return GateMarginBinding(raw, anchor)


def _rounded_row(coefficients, rhs, work):
    """Untrusted producer; the receiver recomputes the ideal rows separately."""
    rounded, error = {}, F(0)
    for j, value in coefficients.items():
        work.visit("coefficient_visits")
        try:
            v = float(value)
        except OverflowError:
            raise ValueError("nonfinite relation coefficient") from None
        require(math.isfinite(v), "nonfinite relation coefficient")
        if v:
            rounded[j] = v
        error = work.exact(error + abs(work.exact(work.exact(F(v)) - value)))
    needed = work.exact(rhs + error)
    try:
        result = float(needed)
    except OverflowError:
        raise ValueError("nonfinite outward relation RHS") from None
    if work.exact(F(result)) < needed:
        result = math.nextafter(result, math.inf)
    require(math.isfinite(result), "nonfinite outward relation RHS")
    work.tick()
    return rounded, result


def _extend(hz, row_pairs):
    def csr(which):
        values, indices, ptr = [], [], [0]
        for row, _ in row_pairs:
            for j, value in sorted(row.items()):
                if (j < hz.n_cont) == which:
                    indices.append(j if which else j - hz.n_cont); values.append(value)
            ptr.append(len(indices))
        return sp.csr_matrix((values, indices, ptr),
                             shape=(2, hz.n_cont if which else hz.n_bin), dtype=np.float64)
    return SparseHZono(
        c=hz.c.copy(), Gc=hz.Gc.copy(), Gb=hz.Gb.copy(), Ac=hz.Ac.copy(),
        Ab=hz.Ab.copy(), b=hz.b.copy(),
        Auc=sp.vstack([hz.Auc, csr(True)], format="csr"),
        Aub=sp.vstack([hz.Aub, csr(False)], format="csr"),
        ub=np.concatenate([hz.ub, np.asarray([x[1] for x in row_pairs], dtype=np.float64)]),
        frame_id=hz.frame_id, exact=False,
    )


def apply_rows(*, binding, pair_hz, router, pair, q, offset, old_f0,
               lambda_slot, product_slot, lambda_center, lambda_radius, gate, work):
    """Return unchanged F0 on unsupported sign/embedding, never on bad identity."""
    require(type(binding) is GateMarginBinding and type(binding.payload) is bytes, "owned gate binding")
    require(pair_hz.relation_mode == "shared_input", "independent input gate relation refused")
    work.tick()
    require(len(binding.payload) <= POLICY["header_bytes"], "bounded binding header")
    work.visit("coefficient_visits", (len(binding.payload) + 4095) // 4096)
    require(digest(binding.payload) == sha(binding.sha256), "binding bytes/hash")
    value = decode(binding.payload)
    work.tick()
    require(freeze(value, work)[0] == binding.payload, "canonical binding bytes")
    require(value["context"]["pair"] == list(pair), "canonical expert pair binding")
    require(metadata(pair_hz) == value["metadata"], "active factor metadata changed")
    active_live = {"router": router, "joint": pair_hz.output_hz, "input": pair_hz.input_hz}
    active = {k: snapshot(hz, work) for k, hz in active_live.items()}
    before = snapshot(old_f0, work, f0=True)
    parameters = {"lambda_slot": lambda_slot, "product_slot": product_slot,
                  "center": float(lambda_center), "radius": float(lambda_radius),
                  "gate": list(map(float, gate)), "q": list(map(float, q)), "offset": float(offset)}
    current, _, m0, margin, C, R, premise, embedding = inspect(
        value, active, before, parameters, expected_binding=binding.sha256, work=work)
    status = "APPLIED" if embedding and premise else "NO_SIGN_PREMISE" if embedding else "GATE_EMBEDDING_REFUSED"
    output = old_f0
    if status == "APPLIED":
        sign = premise["side"]
        # Original margin's binary block is after base.nc; F0 added two
        # continuous factors, so binary coordinates move by two, not by sc.
        linear = {j if j < current["joint"]["nc"] else j + 2:
                  work.exact(-F(sign, 4) * v) for j, v in margin.items()}
        linear[lambda_slot] = work.exact(sign * R)
        half = {lambda_slot: work.exact(-sign * R)}
        row_pairs = [
            _rounded_row(linear, work.exact(sign * work.exact(F(1, 2) + m0 / 4 - C)), work),
            _rounded_row(half, work.exact(sign * work.exact(C - F(1, 2))), work),
        ]
        work.tick()
        output = _extend(old_f0, row_pairs)
        work.tick()
    record = {"schema": SCHEMA, "binding": value, "active": active, "old_f0": before,
              "parameters": parameters, "premise": premise, "status": status,
              "new_f0": snapshot(output, work, f0=True)}
    receipt = check(record, expected_binding=binding.sha256, expected_context=value["context"],
                    expected_property=(parameters["q"], parameters["offset"]), work=work)
    for key, hz in active_live.items():
        require(snapshot(hz, work) == active[key], "active source changed during relation check")
    require(snapshot(old_f0, work, f0=True) == before, "original F0 changed")
    require(snapshot(output, work, f0=True) == record["new_f0"], "received F0 changed")
    work.tick()
    return output, record, receipt
