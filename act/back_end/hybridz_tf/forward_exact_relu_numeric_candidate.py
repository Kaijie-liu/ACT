"""Numerically exact, forward-only compact ReLU graph candidate.

This module is deliberately disconnected from verifier dispatch.  It builds a
``SparseHZono`` whose *stored binary64 coefficients, interpreted as exact real
numbers*, describe the complete ReLU graph of the captured input HZ.  It does
not call the production ReLU builder or a solver and carries no proof or
production authority.

The usual two-factor compact encoding is

    y = (beta/2) * (1 - xi_2)
    x = y + (alpha/2) * (xi_1 + z)
    xi_1 >= -z,  xi_2 >= z,  z in {-1, 1}.

Writing its link as one binary64 row is subtle: ``c - beta/2`` need not itself
be a binary64 number.  Rounding that right-hand side changes the graph.  Here
the exact difference is split into ``hi + lo``.  If ``lo`` is nonzero, one
continuous factor ``kappa`` is shared by the whole layer, fixed by the exact
row ``kappa = 1``, and each affected link is stored as

    original_lhs - lo * kappa = hi.

No equality band is used.  For an exceptional bound whose half is not exactly
representable (notably the minimum subnormal), the candidate uses the
division-free graph

    x = alpha*q + beta*s,  y = beta*s,
    q >= 0, s >= 0, 2*q - z <= 1, 2*s + z <= 1.

It has the same two continuous factors, one binary, and one equality per
unstable neuron as the compact graph, but four rather than two upper rows for
that exceptional neuron.  Thus no finite subnormal is silently rounded away.

Bounds are recomputed from a read-only raw snapshot.  Integer dyadic
accumulation sums the exact stored coefficients and rounds each endpoint
outward without constructing Python ``Fraction`` objects in the hot path.
Caller-provided tighter bounds have no authority in this disconnected module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_reserve_fresh_col_ids_above,
)


SCHEMA = "act.forward_exact_relu_numeric_candidate.v2"
COMPACT = "compact_expansion"
HALF_FREE = "half_free"
ACTIVE = "active_copy"
INACTIVE = "inactive_zero"


@dataclass(frozen=True)
class _CSRFrame:
    shape: Tuple[int, int]
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray

    def csr(self, *, extra_cols: int = 0) -> sp.csr_matrix:
        out = sp.csr_matrix(
            (self.data.copy(), self.indices.copy(), self.indptr.copy()),
            shape=(self.shape[0], self.shape[1] + int(extra_cols)),
            dtype=np.float64,
        )
        out.sort_indices()
        return out


@dataclass(frozen=True)
class _SparseFrame:
    c: np.ndarray
    Gc: _CSRFrame
    Gb: _CSRFrame
    Ac: _CSRFrame
    Ab: _CSRFrame
    b: np.ndarray
    Auc: _CSRFrame
    Aub: _CSRFrame
    ub: np.ndarray
    col_ids: np.ndarray
    bcol_ids: np.ndarray

    @property
    def n_out(self) -> int:
        return int(self.c.size)

    @property
    def n_cont(self) -> int:
        return int(self.Gc.shape[1])

    @property
    def n_bin(self) -> int:
        return int(self.Gb.shape[1])

    @property
    def n_eq(self) -> int:
        return int(self.Ac.shape[0])

    @property
    def n_ub(self) -> int:
        return int(self.Auc.shape[0])


@dataclass(frozen=True)
class ExactReLUNumericCandidate:
    """One frozen wrapper with sealed numeric buffers and no authority."""

    hz: SparseHZono
    lower: np.ndarray
    upper: np.ndarray
    encoding_by_output: Tuple[str, ...]
    phase_counts: Tuple[int, int, int]
    receipt: Mapping[str, Any]


def _raw_dict(value: Any, *, name: str) -> dict:
    try:
        raw = object.__getattribute__(value, "__dict__")
    except Exception as exc:  # pragma: no cover - exact builtins have __dict__
        raise TypeError(f"{name} has no raw instance dictionary") from exc
    if type(raw) is not dict:
        raise TypeError(f"{name} raw instance dictionary is not exact")
    return raw


def _readonly_vector(
    value: Any,
    *,
    name: str,
    dtype: np.dtype,
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype or value.ndim != 1:
        raise TypeError(f"{name} must be an exact one-dimensional {dtype} ndarray")
    if not value.flags.c_contiguous or any(int(s) < 0 for s in value.strides):
        raise ValueError(f"{name} must have a contiguous positive layout")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only for an atomic numeric snapshot")
    # A read-only view of a writable base is not an immutable numeric frame:
    # another alias could still change it while the snapshot is copied.
    owner: Any = value
    seen = set()
    while isinstance(getattr(owner, "base", None), np.ndarray):
        if id(owner) in seen:
            raise ValueError(f"{name} has a cyclic ndarray ownership chain")
        seen.add(id(owner))
        owner = owner.base
        if owner.flags.writeable:
            raise ValueError(f"{name} has a writable ndarray base alias")
    base = getattr(owner, "base", None)
    if type(base) is not bytes:
        if base is None:
            raise ValueError(f"{name} owns reopenable read-only storage")
        raise ValueError(f"{name} has an unaudited external buffer owner")
    if dtype == np.dtype(np.float64) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _validate_csr_buffers(
    shape: Tuple[int, int],
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    *,
    name: str,
) -> None:
    if (
        type(data) is not np.ndarray
        or data.dtype != np.dtype(np.float64)
        or data.ndim != 1
        or type(indices) is not np.ndarray
        or type(indptr) is not np.ndarray
        or indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
        or indptr.dtype != indices.dtype
        or indices.ndim != 1
        or indptr.ndim != 1
    ):
        raise TypeError(f"{name} has non-exact CSR buffers")
    if data.size and (not np.all(np.isfinite(data)) or np.any(data == 0.0)):
        raise ValueError(f"{name} must have finite, explicitly nonzero data")
    if len(shape) != 2 or min(shape) < 0:
        raise ValueError(f"{name} has an invalid shape")
    if indptr.size != shape[0] + 1 or data.size != indices.size:
        raise ValueError(f"{name} CSR buffer lengths disagree with its shape")
    if indptr.size == 0 or int(indptr[0]) != 0 or int(indptr[-1]) != data.size:
        raise ValueError(f"{name} has invalid CSR row pointers")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError(f"{name} CSR row pointers are not monotone")
    if indices.size and (np.any(indices < 0) or np.any(indices >= shape[1])):
        raise ValueError(f"{name} has a CSR column outside its shape")
    for row in range(shape[0]):
        start = int(indptr[row])
        stop = int(indptr[row + 1])
        if stop - start > 1 and np.any(
            indices[start + 1 : stop] <= indices[start : stop - 1]
        ):
            raise ValueError(f"{name} CSR rows must have strictly increasing columns")


def _readonly_csr(value: Any, *, name: str) -> _CSRFrame:
    if type(value) is not sp.csr_matrix:
        raise TypeError(f"{name} must be an exact float64 csr_matrix")
    raw = _raw_dict(value, name=name).copy()
    if any(field not in raw for field in ("data", "indices", "indptr", "_shape")):
        raise ValueError(f"{name} is missing a raw CSR field")
    data = raw["data"]
    indices = raw["indices"]
    indptr = raw["indptr"]
    _readonly_vector(data, name=f"{name}.data", dtype=np.dtype(np.float64))
    if indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise TypeError(f"{name}.indices must use an exact signed index dtype")
    if indptr.dtype != indices.dtype:
        raise TypeError(f"{name} index buffers must use one exact dtype")
    _readonly_vector(indices, name=f"{name}.indices", dtype=indices.dtype)
    _readonly_vector(indptr, name=f"{name}.indptr", dtype=indptr.dtype)
    if data.size and (not np.all(np.isfinite(data)) or np.any(data == 0.0)):
        raise ValueError(f"{name} must have finite, explicitly nonzero data")
    shape_raw = raw["_shape"]
    if type(shape_raw) is not tuple or any(type(v) is not int for v in shape_raw):
        raise TypeError(f"{name} raw shape must be an exact integer tuple")
    shape = tuple(int(v) for v in shape_raw)
    _validate_csr_buffers(shape, data, indices, indptr, name=name)
    # Copy captured component references, never a property-routed matrix view.
    snapshot = _CSRFrame(
        shape=(shape[0], shape[1]),
        data=data.copy(),
        indices=indices.copy(),
        indptr=indptr.copy(),
    )
    _validate_csr_buffers(
        snapshot.shape,
        snapshot.data,
        snapshot.indices,
        snapshot.indptr,
        name=f"{name}.snapshot",
    )
    return snapshot


def _empty_frame(rows: int, cols: int) -> _CSRFrame:
    return _CSRFrame(
        shape=(int(rows), int(cols)),
        data=np.zeros(0, dtype=np.float64),
        indices=np.zeros(0, dtype=np.int32),
        indptr=np.zeros(int(rows) + 1, dtype=np.int32),
    )


def _capture_sparse_frame(hz: Any) -> _SparseFrame:
    """Capture one raw, read-only frame without consulting spoofable properties."""

    if type(hz) is not SparseHZono:
        raise TypeError("hz must be an exact SparseHZono")
    # One shallow dictionary copy captures all field identities while the GIL
    # is held.  Subsequent attribute swaps on the live object cannot create a
    # mixed validate/use frame; every numeric buffer below comes from this map.
    raw = _raw_dict(hz, name="hz").copy()
    required = ("c", "Gc", "Gb", "Ac", "Ab", "b", "col_ids", "bcol_ids")
    if any(name not in raw for name in required):
        raise ValueError("hz is missing a raw SparseHZono field")

    c_ref = _readonly_vector(raw["c"], name="hz.c", dtype=np.dtype(np.float64))
    b_ref = _readonly_vector(raw["b"], name="hz.b", dtype=np.dtype(np.float64))
    col_ref = _readonly_vector(
        raw["col_ids"], name="hz.col_ids", dtype=np.dtype(np.int64)
    )
    bcol_ref = _readonly_vector(
        raw["bcol_ids"], name="hz.bcol_ids", dtype=np.dtype(np.int64)
    )
    Gc = _readonly_csr(raw["Gc"], name="hz.Gc")
    Gb = _readonly_csr(raw["Gb"], name="hz.Gb")
    Ac = _readonly_csr(raw["Ac"], name="hz.Ac")
    Ab = _readonly_csr(raw["Ab"], name="hz.Ab")

    upper_values = (raw.get("Auc"), raw.get("Aub"), raw.get("ub"))
    if all(value is None for value in upper_values):
        Auc = _empty_frame(0, Gc.shape[1])
        Aub = _empty_frame(0, Gb.shape[1])
        ub = np.zeros(0, dtype=np.float64)
    elif any(value is None for value in upper_values):
        raise ValueError("upper constraints require Auc, Aub, and ub together")
    else:
        Auc = _readonly_csr(upper_values[0], name="hz.Auc")
        Aub = _readonly_csr(upper_values[1], name="hz.Aub")
        ub_ref = _readonly_vector(
            upper_values[2], name="hz.ub", dtype=np.dtype(np.float64)
        )
        ub = ub_ref.copy()

    c = c_ref.copy()
    b = b_ref.copy()
    col_ids = col_ref.copy()
    bcol_ids = bcol_ref.copy()
    n_out = int(c.size)
    n_cont = int(Gc.shape[1])
    n_bin = int(Gb.shape[1])
    if Gc.shape[0] != n_out or Gb.shape[0] != n_out:
        raise ValueError("value matrix shape disagrees with raw center")
    if Ac.shape[1] != n_cont or Ab.shape[1] != n_bin:
        raise ValueError("equality matrix width disagrees with factor frame")
    if Ac.shape[0] != Ab.shape[0] or Ac.shape[0] != b.size:
        raise ValueError("equality row count disagrees with raw rhs")
    if Auc.shape[1] != n_cont or Aub.shape[1] != n_bin:
        raise ValueError("upper matrix width disagrees with factor frame")
    if Auc.shape[0] != Aub.shape[0] or Auc.shape[0] != ub.size:
        raise ValueError("upper row count disagrees with raw rhs")
    if col_ids.size != n_cont or bcol_ids.size != n_bin:
        raise ValueError("stable id count disagrees with factor frame")
    if np.any(col_ids < 0) or np.any(bcol_ids < 0):
        raise ValueError("stable ids must be nonnegative")
    if np.unique(col_ids).size != col_ids.size:
        raise ValueError("continuous stable ids must be unique")
    if np.unique(bcol_ids).size != bcol_ids.size:
        raise ValueError("binary stable ids must be unique")
    if np.intersect1d(col_ids, bcol_ids, assume_unique=True).size:
        raise ValueError("continuous and binary stable id namespaces overlap")

    return _SparseFrame(
        c=c,
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        col_ids=col_ids,
        bcol_ids=bcol_ids,
    )


def _normalize_dyadic(value: Tuple[int, int]) -> Tuple[int, int]:
    numerator, exponent = value
    if numerator == 0:
        return 0, 0
    while exponent > 0 and numerator % 2 == 0:
        numerator //= 2
        exponent -= 1
    return numerator, exponent


def _dyadic(value: float) -> Tuple[int, int]:
    numerator, denominator = float(value).as_integer_ratio()
    if denominator <= 0 or denominator & (denominator - 1):
        raise AssertionError("binary64 value is not dyadic")
    return _normalize_dyadic((numerator, denominator.bit_length() - 1))


def _add_dyadic(
    left: Tuple[int, int], right: Tuple[int, int]
) -> Tuple[int, int]:
    exponent = max(left[1], right[1])
    return _normalize_dyadic((
        (left[0] << (exponent - left[1]))
        + (right[0] << (exponent - right[1])),
        exponent,
    ))


def _negate_dyadic(value: Tuple[int, int]) -> Tuple[int, int]:
    return -value[0], value[1]


def _compare_float_to_dyadic(value: float, exact: Tuple[int, int]) -> int:
    numerator, denominator = float(value).as_integer_ratio()
    left = numerator << exact[1]
    right = exact[0] * denominator
    return (left > right) - (left < right)


def _directed_float(value: Tuple[int, int], *, upward: bool) -> float:
    numerator, exponent = value
    try:
        rounded = numerator / (1 << exponent)
    except OverflowError as exc:
        raise ValueError("exact HZ box endpoint exceeds finite binary64") from exc
    if not np.isfinite(rounded):
        raise ValueError("exact HZ box endpoint exceeds finite binary64")
    comparison = _compare_float_to_dyadic(rounded, value)
    if upward and comparison < 0:
        rounded = float(np.nextafter(np.float64(rounded), np.float64(np.inf)))
    elif not upward and comparison > 0:
        rounded = float(np.nextafter(np.float64(rounded), np.float64(-np.inf)))
    if not np.isfinite(rounded):
        raise ValueError("outward HZ box endpoint exceeds finite binary64")
    check = _compare_float_to_dyadic(rounded, value)
    if (upward and check < 0) or ((not upward) and check > 0):
        raise AssertionError("directed binary64 conversion moved inward")
    return rounded


def _rigorous_box_from_frame(frame: _SparseFrame) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.empty(frame.n_out, dtype=np.float64)
    upper = np.empty(frame.n_out, dtype=np.float64)
    for row in range(frame.n_out):
        radius = (0, 0)
        for matrix in (frame.Gc, frame.Gb):
            start = int(matrix.indptr[row])
            stop = int(matrix.indptr[row + 1])
            for value in matrix.data[start:stop]:
                radius = _add_dyadic(radius, _dyadic(abs(float(value))))
        center = _dyadic(float(frame.c[row]))
        lower[row] = _directed_float(
            _add_dyadic(center, _negate_dyadic(radius)), upward=False
        )
        upper[row] = _directed_float(_add_dyadic(center, radius), upward=True)
    return _sealed_array(lower), _sealed_array(upper)


def rigorous_sparse_hz_box_arrays(hz: SparseHZono) -> Tuple[np.ndarray, np.ndarray]:
    """Return independently recomputed, outward binary64 box endpoints."""

    return _rigorous_box_from_frame(_capture_sparse_frame(hz))


def _validate_supplied_bounds(
    value: Any,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    if value is None:
        return
    if type(value) is not Bounds:
        raise TypeError("pre_bounds must be an exact Bounds")
    raw = _raw_dict(value, name="pre_bounds").copy()
    if set(("lb", "ub")) - set(raw):
        raise ValueError("pre_bounds is missing a raw tensor")
    arrays = []
    for name in ("lb", "ub"):
        tensor = raw[name]
        if type(tensor) is not torch.Tensor:
            raise TypeError(f"pre_bounds.{name} must be an exact torch.Tensor")
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.requires_grad
            or not tensor.is_contiguous()
        ):
            raise ValueError("pre_bounds tensors must be contiguous CPU float64 constants")
        arrays.append(tensor.detach().numpy().reshape(-1).copy())
    if not (np.array_equal(arrays[0], lower) and np.array_equal(arrays[1], upper)):
        raise ValueError("uncertified pre_bounds differ from the independent outward box")


def _exact_half(value: float) -> Optional[float]:
    half = float(np.float64(value) * np.float64(0.5))
    if not np.isfinite(half):
        return None
    if _dyadic(half * 2.0) != _dyadic(float(value)):
        return None
    return half


def _difference_expansion(left: float, right: float) -> Optional[Tuple[float, float]]:
    """Return binary64 ``hi, lo`` with exact ``left-right == hi+lo``."""

    negated_right = -float(right)
    hi = float(left) + negated_right
    if not np.isfinite(hi):
        return None
    # Knuth TwoSum.  The integer equality remains the final exactness check.
    virtual = hi - float(left)
    lo = (float(left) - (hi - virtual)) + (negated_right - virtual)
    if not np.isfinite(lo):
        return None
    exact = _add_dyadic(
        _dyadic(float(left)), _negate_dyadic(_dyadic(float(right)))
    )
    if _add_dyadic(_dyadic(hi), _dyadic(lo)) != exact:
        return None
    return hi, lo


def _csr_from_triplets(
    rows: Sequence[int],
    cols: Sequence[int],
    data: Sequence[float],
    *,
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    if not rows:
        return sp.csr_matrix(shape, dtype=np.float64)
    out = sp.coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=shape,
        dtype=np.float64,
    ).tocsr()
    out.eliminate_zeros()
    out.sort_indices()
    return out


def _append_csr_row(
    frame: _CSRFrame,
    source_row: int,
    target_row: int,
    rows: list[int],
    cols: list[int],
    data: list[float],
    *,
    scale: float,
) -> None:
    start = int(frame.indptr[source_row])
    stop = int(frame.indptr[source_row + 1])
    for col, value in zip(frame.indices[start:stop], frame.data[start:stop]):
        rows.append(int(target_row))
        cols.append(int(col))
        # Negation is exact for every finite, explicitly nonzero binary64.
        data.append(float(scale * float(value)))


def _sealed_array(value: np.ndarray) -> np.ndarray:
    """Return a C-layout ndarray backed by immutable ``bytes`` storage."""

    array = np.asarray(value)
    sealed = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    if sealed.flags.writeable or not sealed.flags.c_contiguous:
        raise AssertionError("failed to seal a candidate numeric buffer")
    return sealed


def _freeze_sparse_hz(hz: SparseHZono) -> None:
    raw = _raw_dict(hz, name="candidate.hz")
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        value = raw.get(name)
        if type(value) is np.ndarray:
            raw[name] = _sealed_array(value)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = raw.get(name)
        if type(matrix) is sp.csr_matrix:
            matrix_raw = _raw_dict(matrix, name=f"candidate.hz.{name}")
            shape = matrix_raw.get("_shape")
            if type(shape) is not tuple:
                raise TypeError(f"candidate.hz.{name} has no exact raw shape")
            _validate_csr_buffers(
                shape,
                matrix_raw.get("data"),
                matrix_raw.get("indices"),
                matrix_raw.get("indptr"),
                name=f"candidate.hz.{name}",
            )
            matrix.data = _sealed_array(matrix_raw["data"])
            matrix.indices = _sealed_array(matrix_raw["indices"])
            matrix.indptr = _sealed_array(matrix_raw["indptr"])
            _validate_csr_buffers(
                shape,
                matrix.data,
                matrix.indices,
                matrix.indptr,
                name=f"candidate.hz.{name}.sealed",
            )


def _reserve_ids(frame: _SparseFrame, n_cont: int, n_bin: int) -> Tuple[np.ndarray, np.ndarray]:
    floor = -1
    if frame.col_ids.size:
        floor = max(floor, int(frame.col_ids.max()))
    if frame.bcol_ids.size:
        floor = max(floor, int(frame.bcol_ids.max()))
    new_cont = (
        hz_reserve_fresh_col_ids_above(
            int(n_cont), lower_bound_exclusive=floor, device="cpu"
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    if new_cont.size:
        floor = int(new_cont[-1])
    new_bin = (
        hz_reserve_fresh_col_ids_above(
            int(n_bin), lower_bound_exclusive=floor, device="cpu"
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return new_cont.copy(), new_bin.copy()


def build_forward_exact_relu_numeric_candidate(
    hz: SparseHZono,
    *,
    pre_bounds: Optional[Bounds] = None,
) -> ExactReLUNumericCandidate:
    """Build the exact ReLU graph of a strict read-only sparse-HZ snapshot."""

    frame = _capture_sparse_frame(hz)
    lower, upper = _rigorous_box_from_frame(frame)
    _validate_supplied_bounds(pre_bounds, lower=lower, upper=upper)

    active_mask = lower >= 0.0
    inactive_mask = (~active_mask) & (upper <= 0.0)
    unstable_idx = np.flatnonzero(~(active_mask | inactive_mask)).astype(np.int64)
    active_idx = np.flatnonzero(active_mask).astype(np.int64)
    k = int(unstable_idx.size)

    encodings = [INACTIVE] * frame.n_out
    for row in active_idx:
        encodings[int(row)] = ACTIVE

    compact = np.zeros(k, dtype=np.bool_)
    alpha_half = np.zeros(k, dtype=np.float64)
    beta_half = np.zeros(k, dtype=np.float64)
    rhs_hi = np.zeros(k, dtype=np.float64)
    rhs_lo = np.zeros(k, dtype=np.float64)
    for local, source_row in enumerate(unstable_idx):
        alpha = float(lower[source_row])
        beta = float(upper[source_row])
        ah = _exact_half(alpha)
        bh = _exact_half(beta)
        expansion = None if bh is None else _difference_expansion(frame.c[source_row], bh)
        if ah is not None and bh is not None and expansion is not None:
            compact[local] = True
            alpha_half[local] = ah
            beta_half[local] = bh
            rhs_hi[local], rhs_lo[local] = expansion
            encodings[int(source_row)] = COMPACT
        else:
            encodings[int(source_row)] = HALF_FREE

    needs_kappa = bool(np.any(rhs_lo[compact] != 0.0))
    added_cont = 2 * k + int(needs_kappa)
    added_bin = k
    n_cont = frame.n_cont + added_cont
    n_bin = frame.n_bin + added_bin
    q_cols = frame.n_cont + np.arange(k, dtype=np.int64)
    s_cols = frame.n_cont + k + np.arange(k, dtype=np.int64)
    z_cols = frame.n_bin + np.arange(k, dtype=np.int64)
    kappa_col = frame.n_cont + 2 * k if needs_kappa else None

    out_c = np.zeros(frame.n_out, dtype=np.float64)
    value_c_rows: list[int] = []
    value_c_cols: list[int] = []
    value_c_data: list[float] = []
    value_b_rows: list[int] = []
    value_b_cols: list[int] = []
    value_b_data: list[float] = []
    for source_row in active_idx:
        row = int(source_row)
        out_c[row] = frame.c[row]
        _append_csr_row(
            frame.Gc, row, row,
            value_c_rows, value_c_cols, value_c_data,
            scale=1.0,
        )
        _append_csr_row(
            frame.Gb, row, row,
            value_b_rows, value_b_cols, value_b_data,
            scale=1.0,
        )
    for local, source_row in enumerate(unstable_idx):
        row = int(source_row)
        value_c_rows.append(row)
        value_c_cols.append(int(s_cols[local]))
        if compact[local]:
            out_c[row] = beta_half[local]
            value_c_data.append(float(-beta_half[local]))
        else:
            value_c_data.append(float(upper[row]))
    out_Gc = _csr_from_triplets(
        value_c_rows, value_c_cols, value_c_data,
        shape=(frame.n_out, n_cont),
    )
    out_Gb = _csr_from_triplets(
        value_b_rows, value_b_cols, value_b_data,
        shape=(frame.n_out, n_bin),
    )

    eq_c_rows: list[int] = []
    eq_c_cols: list[int] = []
    eq_c_data: list[float] = []
    eq_b_rows: list[int] = []
    eq_b_cols: list[int] = []
    eq_b_data: list[float] = []
    eq_rhs = np.zeros(k + int(needs_kappa), dtype=np.float64)
    for local, source_row in enumerate(unstable_idx):
        source = int(source_row)
        row = int(local)
        if compact[local]:
            eq_c_rows.extend((row, row))
            eq_c_cols.extend((int(q_cols[local]), int(s_cols[local])))
            eq_c_data.extend((float(alpha_half[local]), float(-beta_half[local])))
            eq_b_rows.append(row)
            eq_b_cols.append(int(z_cols[local]))
            eq_b_data.append(float(alpha_half[local]))
            eq_rhs[row] = rhs_hi[local]
            if rhs_lo[local] != 0.0:
                assert kappa_col is not None
                eq_c_rows.append(row)
                eq_c_cols.append(int(kappa_col))
                eq_c_data.append(float(-rhs_lo[local]))
        else:
            eq_c_rows.extend((row, row))
            eq_c_cols.extend((int(q_cols[local]), int(s_cols[local])))
            eq_c_data.extend((float(lower[source]), float(upper[source])))
            eq_rhs[row] = frame.c[source]
        _append_csr_row(
            frame.Gc, source, row,
            eq_c_rows, eq_c_cols, eq_c_data,
            scale=-1.0,
        )
        _append_csr_row(
            frame.Gb, source, row,
            eq_b_rows, eq_b_cols, eq_b_data,
            scale=-1.0,
        )
    if needs_kappa:
        assert kappa_col is not None
        eq_c_rows.append(k)
        eq_c_cols.append(int(kappa_col))
        eq_c_data.append(1.0)
        eq_rhs[k] = 1.0
    relu_Ac = _csr_from_triplets(
        eq_c_rows, eq_c_cols, eq_c_data,
        shape=(eq_rhs.size, n_cont),
    )
    relu_Ab = _csr_from_triplets(
        eq_b_rows, eq_b_cols, eq_b_data,
        shape=(eq_rhs.size, n_bin),
    )

    upper_c_rows: list[int] = []
    upper_c_cols: list[int] = []
    upper_c_data: list[float] = []
    upper_b_rows: list[int] = []
    upper_b_cols: list[int] = []
    upper_b_data: list[float] = []
    upper_rhs: list[float] = []
    next_upper = 0
    for local in range(k):
        q = int(q_cols[local])
        s = int(s_cols[local])
        z = int(z_cols[local])
        if compact[local]:
            # -xi_1-z <= 0; -xi_2+z <= 0.
            upper_c_rows.extend((next_upper, next_upper + 1))
            upper_c_cols.extend((q, s))
            upper_c_data.extend((-1.0, -1.0))
            upper_b_rows.extend((next_upper, next_upper + 1))
            upper_b_cols.extend((z, z))
            upper_b_data.extend((-1.0, 1.0))
            upper_rhs.extend((0.0, 0.0))
            next_upper += 2
        else:
            # -q<=0; -s<=0; 2q-z<=1; 2s+z<=1.
            upper_c_rows.extend(
                (next_upper, next_upper + 1, next_upper + 2, next_upper + 3)
            )
            upper_c_cols.extend((q, s, q, s))
            upper_c_data.extend((-1.0, -1.0, 2.0, 2.0))
            upper_b_rows.extend((next_upper + 2, next_upper + 3))
            upper_b_cols.extend((z, z))
            upper_b_data.extend((-1.0, 1.0))
            upper_rhs.extend((0.0, 0.0, 1.0, 1.0))
            next_upper += 4
    relu_Auc = _csr_from_triplets(
        upper_c_rows, upper_c_cols, upper_c_data,
        shape=(next_upper, n_cont),
    )
    relu_Aub = _csr_from_triplets(
        upper_b_rows, upper_b_cols, upper_b_data,
        shape=(next_upper, n_bin),
    )

    old_Ac = frame.Ac.csr(extra_cols=added_cont)
    old_Ab = frame.Ab.csr(extra_cols=added_bin)
    old_Auc = frame.Auc.csr(extra_cols=added_cont)
    old_Aub = frame.Aub.csr(extra_cols=added_bin)
    Ac = sp.vstack((old_Ac, relu_Ac), format="csr")
    Ab = sp.vstack((old_Ab, relu_Ab), format="csr")
    Auc = sp.vstack((old_Auc, relu_Auc), format="csr")
    Aub = sp.vstack((old_Aub, relu_Aub), format="csr")
    b = np.concatenate((frame.b.copy(), eq_rhs))
    ub = np.concatenate((frame.ub.copy(), np.asarray(upper_rhs, dtype=np.float64)))
    for matrix in (out_Gc, out_Gb, Ac, Ab, Auc, Aub):
        matrix.eliminate_zeros()
        matrix.sort_indices()

    new_cont_ids, new_bin_ids = _reserve_ids(frame, added_cont, added_bin)
    col_ids = np.concatenate((frame.col_ids.copy(), new_cont_ids))
    bcol_ids = np.concatenate((frame.bcol_ids.copy(), new_bin_ids))
    out = SparseHZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        col_ids=col_ids,
        bcol_ids=bcol_ids,
    )
    _freeze_sparse_hz(out)

    compact_count = int(compact.sum())
    half_free_count = int(k - compact_count)
    residual_count = int(np.count_nonzero(rhs_lo[compact]))
    source_constraint_nnz = int(
        frame.Ac.data.size
        + frame.Ab.data.size
        + frame.Auc.data.size
        + frame.Aub.data.size
    )
    output_constraint_nnz = int(
        out.Ac.nnz + out.Ab.nnz + out.Auc.nnz + out.Aub.nnz
    )
    receipt = MappingProxyType(
        {
            "schema": SCHEMA,
            "proof_authority": False,
            "production_authority": False,
            "authenticity_verified": False,
            "forward_only": True,
            "complete_graph_exact_over_stored_reals": True,
            "equality_band": False,
            "solver_called": False,
            "bounds_internally_recomputed": True,
            "integer_dyadic_bounds": True,
            "python_fraction_used": False,
            "caller_tightening_authority": False,
            "raw_readonly_snapshot": True,
            "numeric_buffers_bytes_sealed": True,
            "public_hz_rebind_protected": False,
            "active": int(active_mask.sum()),
            "inactive": int(inactive_mask.sum()),
            "unstable": k,
            "compact_rows": compact_count,
            "compact_residual_rows": residual_count,
            "half_free_rows": half_free_count,
            "shared_fixed_one_factor": bool(needs_kappa),
            "added_cont": int(added_cont),
            "added_bin": int(added_bin),
            "added_eq": int(k + int(needs_kappa)),
            "added_upper": int(2 * compact_count + 4 * half_free_count),
            "added_constraint_nnz": int(
                output_constraint_nnz - source_constraint_nnz
            ),
        }
    )
    return ExactReLUNumericCandidate(
        hz=out,
        lower=lower,
        upper=upper,
        encoding_by_output=tuple(encodings),
        phase_counts=(int(active_mask.sum()), int(inactive_mask.sum()), k),
        receipt=receipt,
    )


__all__ = [
    "ACTIVE",
    "COMPACT",
    "ExactReLUNumericCandidate",
    "HALF_FREE",
    "INACTIVE",
    "SCHEMA",
    "build_forward_exact_relu_numeric_candidate",
    "rigorous_sparse_hz_box_arrays",
]
