#!/usr/bin/env python3
# ===- property_micro_rlt.py - bounded factor-space RLT -----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# ===----------------------------------------------------------------===#
"""A bounded, exact-stored-float degree-1 RLT for sparse HybridZ factors.

For a selected signed binary factor ``s in {-1, 1}`` and a source upper
constraint

``r(q) = a_c xi_c + a_b xi_b - rhs <= 0``,

both products ``(1+s) r <= 0`` and ``(1-s) r <= 0`` are valid.  Products
``s*q`` with every other factor in the selected rows are represented by
shared continuous auxiliaries in ``[-1, 1]`` and their four exact McCormick
hull rows.  Continuous products remain directional in the selected binary;
the commutative binary product ``s_i*s_j`` uses one unordered-pair auxiliary
for both multiplication orientations.  Reintroducing integrality therefore
gives exactly the original set, while relaxing the binaries can be strictly
tighter.

Selection has no proof authority.  This module only constructs and audits the
rows requested by the caller; all original rows and factors are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_fresh_col_ids,
)


class PropertyMicroRLTError(ValueError):
    """The requested bounded RLT lift is malformed or exceeds its cap."""

    def __init__(
        self,
        message: str,
        *,
        selected_source_row_nnz_required: int | None = None,
        product_factors_required: int | None = None,
        reason_code: str | None = None,
        requirement_count_complete: bool = False,
    ) -> None:
        super().__init__(message)
        self.selected_source_row_nnz_required = (
            selected_source_row_nnz_required
        )
        self.product_factors_required = product_factors_required
        self.reason_code = reason_code
        self.requirement_count_complete = bool(
            requirement_count_complete
        )


@dataclass(frozen=True)
class PropertyMicroRLTResult:
    hz: SparseHZono
    receipt: Dict[str, Any]


def _csr_sha256(value: sp.csr_matrix) -> str:
    matrix = sp.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _array_sha256(value: Any, *, dtype: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _optional_ids_sha256(value: Any) -> str | None:
    if value is None:
        return None
    return _array_sha256(value, dtype=np.int64)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def verify_property_micro_rlt_receipt(
    receipt: Mapping[str, Any],
) -> bool:
    try:
        expected = receipt["receipt_sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        payload = dict(receipt)
        del payload["receipt_sha256"]
        return _canonical_sha256(payload) == expected
    except (KeyError, TypeError, ValueError):
        return False


def _strict_receipt_int(
    receipt: Mapping[str, Any],
    key: str,
) -> int:
    value = receipt[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, Integral
    ):
        raise ValueError(f"{key} is not an integer")
    return int(value)


def verify_property_micro_rlt_result(
    result: PropertyMicroRLTResult,
) -> bool:
    """Live-check a micro-RLT receipt against its actual sparse HZ.

    A self-hash proves only that a receipt is internally unchanged.  This
    validator additionally re-hashes the live matrices, right-hand sides, and
    stable IDs, checks every recorded dimension/count, and verifies that the
    original prefix is retained with zero coefficients on all new columns.
    """

    try:
        if not isinstance(result, PropertyMicroRLTResult):
            return False
        hz = result.hz
        receipt = result.receipt
        if (
            not isinstance(hz, SparseHZono)
            or not isinstance(receipt, Mapping)
            or not verify_property_micro_rlt_receipt(receipt)
            or receipt.get("schema") != "act.property_micro_rlt.v1"
            or receipt.get("intended_consumer")
            != "parent_binary_relaxation_before_exact_phase_enumeration"
            or receipt.get("fixed_phase_rows_are_redundant") is not True
            or receipt.get("binary_factor_encoding") != "signed_pm1"
            or receipt.get("solver_relaxation_encoding")
            != "z_in_0_1_with_signed_factor_s_equals_2z_minus_1"
        ):
            return False
        attached = getattr(hz, "_property_micro_rlt_receipt", None)
        if not isinstance(attached, Mapping) or dict(attached) != dict(
            receipt
        ):
            return False

        base_n_out = _strict_receipt_int(receipt, "base_n_out")
        base_n_cont = _strict_receipt_int(receipt, "base_n_cont")
        base_n_bin = _strict_receipt_int(receipt, "base_n_bin")
        base_n_eq = _strict_receipt_int(receipt, "base_n_eq")
        base_n_ub = _strict_receipt_int(receipt, "base_n_ub")
        result_n_out = _strict_receipt_int(receipt, "result_n_out")
        result_n_cont = _strict_receipt_int(
            receipt, "result_n_cont"
        )
        result_n_bin = _strict_receipt_int(receipt, "result_n_bin")
        result_n_eq = _strict_receipt_int(receipt, "result_n_eq")
        result_n_ub = _strict_receipt_int(receipt, "result_n_ub")
        n_aux = _strict_receipt_int(
            receipt, "new_product_factors"
        )
        product_hull_rows = _strict_receipt_int(
            receipt, "new_product_hull_rows"
        )
        rlt_rows = _strict_receipt_int(receipt, "new_rlt_rows")
        new_upper_rows = _strict_receipt_int(
            receipt, "new_upper_rows"
        )
        selected_binary_count = _strict_receipt_int(
            receipt, "selected_binary_count"
        )
        selected_source_count = _strict_receipt_int(
            receipt, "selected_source_row_count"
        )

        if min(
            base_n_out,
            base_n_cont,
            base_n_bin,
            base_n_eq,
            base_n_ub,
            result_n_out,
            result_n_cont,
            result_n_bin,
            result_n_eq,
            result_n_ub,
            n_aux,
            product_hull_rows,
            rlt_rows,
            new_upper_rows,
            selected_binary_count,
            selected_source_count,
        ) < 0:
            return False
        if (
            result_n_out != base_n_out
            or result_n_bin != base_n_bin
            or result_n_eq != base_n_eq
            or result_n_cont != base_n_cont + n_aux
            or result_n_ub != base_n_ub + new_upper_rows
            or product_hull_rows != 4 * n_aux
            or rlt_rows != 2 * selected_source_count
            or new_upper_rows != product_hull_rows + rlt_rows
            or hz.n_out != result_n_out
            or hz.n_cont != result_n_cont
            or hz.n_bin != result_n_bin
            or hz.n_eq != result_n_eq
            or hz.n_ub != result_n_ub
        ):
            return False

        if (
            hz.Gc.shape != (result_n_out, result_n_cont)
            or hz.Gb.shape != (result_n_out, result_n_bin)
            or hz.Ac.shape != (result_n_eq, result_n_cont)
            or hz.Ab.shape != (result_n_eq, result_n_bin)
            or hz.Auc is None
            or hz.Aub is None
            or hz.ub is None
            or hz.Auc.shape != (result_n_ub, result_n_cont)
            or hz.Aub.shape != (result_n_ub, result_n_bin)
            or np.asarray(hz.c).reshape(-1).size != result_n_out
            or np.asarray(hz.b).reshape(-1).size != result_n_eq
            or np.asarray(hz.ub).reshape(-1).size != result_n_ub
        ):
            return False
        matrices = (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)
        if any(
            not matrix.has_canonical_format
            or (
                matrix.nnz
                and not np.all(np.isfinite(matrix.data))
            )
            for matrix in matrices
        ):
            return False
        if (
            not np.all(np.isfinite(hz.c))
            or not np.all(np.isfinite(hz.b))
            or not np.all(np.isfinite(hz.ub))
        ):
            return False
        if hz.col_ids is not None:
            live_cont_ids = np.asarray(
                hz.col_ids, dtype=np.int64
            ).reshape(-1)
            if (
                live_cont_ids.size != result_n_cont
                or np.unique(live_cont_ids).size
                != live_cont_ids.size
            ):
                return False
        else:
            live_cont_ids = np.zeros(0, dtype=np.int64)
        if hz.bcol_ids is not None:
            live_bin_ids = np.asarray(
                hz.bcol_ids, dtype=np.int64
            ).reshape(-1)
            if (
                live_bin_ids.size != result_n_bin
                or np.unique(live_bin_ids).size != live_bin_ids.size
            ):
                return False
        else:
            live_bin_ids = np.zeros(0, dtype=np.int64)
        if (
            live_cont_ids.size
            and live_bin_ids.size
            and np.intersect1d(
                live_cont_ids,
                live_bin_ids,
                assume_unique=True,
            ).size
        ):
            return False

        base_Gc = hz.Gc[:, :base_n_cont].tocsr()
        base_Ac = hz.Ac[:, :base_n_cont].tocsr()
        base_Auc = hz.Auc[:base_n_ub, :base_n_cont].tocsr()
        base_Aub = hz.Aub[:base_n_ub, :].tocsr()
        if (
            hz.Gc[:, base_n_cont:].nnz != 0
            or hz.Ac[:, base_n_cont:].nnz != 0
            or hz.Auc[:base_n_ub, base_n_cont:].nnz != 0
        ):
            return False

        live_hashes = {
            "base_center_sha256": _array_sha256(
                hz.c, dtype=np.float64
            ),
            "result_center_sha256": _array_sha256(
                hz.c, dtype=np.float64
            ),
            "base_output_continuous_csr_sha256": _csr_sha256(
                base_Gc
            ),
            "result_output_continuous_csr_sha256": _csr_sha256(
                hz.Gc
            ),
            "base_output_binary_csr_sha256": _csr_sha256(hz.Gb),
            "result_output_binary_csr_sha256": _csr_sha256(hz.Gb),
            "base_equality_continuous_csr_sha256": _csr_sha256(
                base_Ac
            ),
            "result_equality_continuous_csr_sha256": _csr_sha256(
                hz.Ac
            ),
            "base_equality_binary_csr_sha256": _csr_sha256(hz.Ab),
            "result_equality_binary_csr_sha256": _csr_sha256(hz.Ab),
            "base_equality_rhs_sha256": _array_sha256(
                hz.b, dtype=np.float64
            ),
            "result_equality_rhs_sha256": _array_sha256(
                hz.b, dtype=np.float64
            ),
            "base_upper_csr_sha256": _csr_sha256(base_Auc),
            "result_upper_csr_sha256": _csr_sha256(hz.Auc),
            "base_upper_binary_csr_sha256": _csr_sha256(base_Aub),
            "result_upper_binary_csr_sha256": _csr_sha256(hz.Aub),
            "base_upper_rhs_sha256": _array_sha256(
                hz.ub[:base_n_ub], dtype=np.float64
            ),
            "result_upper_rhs_sha256": _array_sha256(
                hz.ub, dtype=np.float64
            ),
            "base_continuous_col_ids_sha256": (
                _optional_ids_sha256(
                    None
                    if hz.col_ids is None
                    else hz.col_ids[:base_n_cont]
                )
            ),
            "result_continuous_col_ids_sha256": (
                _optional_ids_sha256(hz.col_ids)
            ),
            "base_binary_col_ids_sha256": _optional_ids_sha256(
                hz.bcol_ids
            ),
            "result_binary_col_ids_sha256": _optional_ids_sha256(
                hz.bcol_ids
            ),
        }
        if any(receipt.get(key) != value for key, value in live_hashes.items()):
            return False

        selection = receipt.get("selection")
        product_records = receipt.get("product_records")
        rlt_records = receipt.get("rlt_records")
        row_names = receipt.get("generated_row_names")
        if (
            not isinstance(selection, list)
            or len(selection) != selected_binary_count
            or not isinstance(product_records, list)
            or len(product_records) != n_aux
            or not isinstance(rlt_records, list)
            or len(rlt_records) != selected_source_count
            or not isinstance(row_names, list)
            or len(row_names) != new_upper_rows
        ):
            return False

        normalized_selection = []
        previous_binary = -1
        for entry in selection:
            if not isinstance(entry, Mapping):
                return False
            binary = _strict_receipt_int(entry, "binary_position")
            raw_rows = entry.get("source_upper_rows")
            if (
                not previous_binary < binary < base_n_bin
                or not isinstance(raw_rows, list)
                or not raw_rows
                or any(
                    isinstance(row, (bool, np.bool_))
                    or not isinstance(row, Integral)
                    for row in raw_rows
                )
            ):
                return False
            rows = tuple(int(row) for row in raw_rows)
            if (
                rows != tuple(sorted(rows))
                or len(set(rows)) != len(rows)
                or min(rows) < 0
                or max(rows) >= base_n_ub
            ):
                return False
            normalized_selection.append((binary, rows))
            previous_binary = binary
        if (
            sum(len(rows) for _, rows in normalized_selection)
            != selected_source_count
        ):
            return False
        selected_nnz = sum(
            int(
                base_Auc.indptr[row + 1]
                - base_Auc.indptr[row]
            )
            + int(
                base_Aub.indptr[row + 1]
                - base_Aub.indptr[row]
            )
            for _, rows in normalized_selection
            for row in rows
        )
        if _strict_receipt_int(
            receipt, "selected_source_row_nnz"
        ) != selected_nnz:
            return False

        expected_product_keys: set[_ProductKey] = set()
        expected_orientations: Dict[
            _ProductKey, set[Tuple[int, int]]
        ] = {}
        for selected_binary, rows in normalized_selection:
            for row in rows:
                c_start = int(base_Auc.indptr[row])
                c_end = int(base_Auc.indptr[row + 1])
                for column in base_Auc.indices[c_start:c_end]:
                    expected_product_keys.add(
                        _continuous_product_key(
                            selected_binary, int(column)
                        )
                    )
                b_start = int(base_Aub.indptr[row])
                b_end = int(base_Aub.indptr[row + 1])
                for column in base_Aub.indices[b_start:b_end]:
                    other = int(column)
                    if other == selected_binary:
                        continue
                    key = _binary_product_key(
                        selected_binary, other
                    )
                    expected_product_keys.add(key)
                    expected_orientations.setdefault(
                        key, set()
                    ).add((selected_binary, other))
        ordered_product_keys = sorted(expected_product_keys)
        if len(ordered_product_keys) != n_aux:
            return False

        aux_positions = set()
        for offset, (record, expected_key) in enumerate(
            zip(product_records, ordered_product_keys)
        ):
            if not isinstance(record, Mapping):
                return False
            aux = _strict_receipt_int(
                record, "aux_continuous_position"
            )
            if aux != base_n_cont + offset:
                return False
            aux_positions.add(aux)
            binary = _strict_receipt_int(
                record, "binary_position"
            )
            factor = _strict_receipt_int(
                record, "factor_position"
            )
            kind = record.get("factor_kind")
            key = (str(kind), binary, factor)
            if key != expected_key:
                return False
            if kind == "continuous":
                if (
                    not 0 <= binary < base_n_bin
                    or not 0 <= factor < base_n_cont
                    or _strict_receipt_int(
                        record, "orientation_sign"
                    )
                    != 1
                ):
                    return False
            elif kind == "binary":
                pair = record.get("unordered_binary_pair")
                orientations = record.get("orientation_uses")
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or any(
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, Integral)
                        for value in pair
                    )
                ):
                    return False
                pair_tuple = tuple(int(value) for value in pair)
                if (
                    pair_tuple != tuple(sorted(pair_tuple))
                    or pair_tuple[0] == pair_tuple[1]
                    or binary != pair_tuple[0]
                    or factor != pair_tuple[1]
                    or pair_tuple[0] < 0
                    or pair_tuple[1] >= base_n_bin
                    or not isinstance(orientations, list)
                    or not orientations
                    or record.get("commutative") is not True
                    or _strict_receipt_int(
                        record, "orientation_sign"
                    )
                    != 1
                ):
                    return False
                normalized_orientations = []
                for orientation in orientations:
                    if not isinstance(orientation, Mapping):
                        return False
                    selected = _strict_receipt_int(
                        orientation, "selected_binary_position"
                    )
                    other = _strict_receipt_int(
                        orientation, "other_binary_position"
                    )
                    if (
                        tuple(sorted((selected, other))) != pair_tuple
                        or selected == other
                        or _strict_receipt_int(
                            orientation, "orientation_sign"
                        )
                        != 1
                    ):
                        return False
                    normalized_orientations.append((selected, other))
                if (
                    normalized_orientations
                    != sorted(normalized_orientations)
                    or set(normalized_orientations)
                    != expected_orientations.get(
                        expected_key, set()
                    )
                ):
                    return False
            else:
                return False
        if aux_positions != set(range(base_n_cont, result_n_cont)):
            return False

        expected_names = []
        for kind, left, right in ordered_product_keys:
            expected_names.extend(
                f"product[{kind},{left},{right}].{suffix}"
                for suffix in (
                    "lower_pp",
                    "lower_nn",
                    "upper_np",
                    "upper_pn",
                )
            )
        expected_rlt_records = []
        for selected_binary, rows in normalized_selection:
            for source_row in rows:
                generated = [
                    len(expected_names),
                    len(expected_names) + 1,
                ]
                expected_names.extend(
                    [
                        f"rlt[{selected_binary},{source_row}].plus",
                        f"rlt[{selected_binary},{source_row}].minus",
                    ]
                )
                expected_rlt_records.append(
                    {
                        "binary_position": selected_binary,
                        "source_upper_row": source_row,
                        "generated_local_rows": generated,
                    }
                )
        if (
            row_names != expected_names
            or rlt_records != expected_rlt_records
        ):
            return False
        return True
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _stored_fraction(value: float, *, name: str) -> Fraction:
    stored = float(value)
    if not math.isfinite(stored):
        raise PropertyMicroRLTError(f"{name} must be finite")
    return Fraction.from_float(stored)


def _outward_float(value: Fraction, *, name: str) -> float:
    try:
        stored = float(value)
    except OverflowError as exc:
        raise PropertyMicroRLTError(
            f"{name} overflowed binary64"
        ) from exc
    if not math.isfinite(stored):
        raise PropertyMicroRLTError(f"{name} overflowed binary64")
    if Fraction.from_float(stored) < value:
        stored = math.nextafter(stored, math.inf)
    if not math.isfinite(stored) or Fraction.from_float(stored) < value:
        raise PropertyMicroRLTError(
            f"{name} could not be rounded outward"
        )
    return stored


def _store_upper_row(
    *,
    cont: Mapping[int, Fraction],
    binary: Mapping[int, Fraction],
    rhs: Fraction,
    n_cont: int,
    n_bin: int,
    row_name: str,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Store one exact inequality and widen its RHS for coefficient rounding."""

    stored_cont: Dict[int, float] = {}
    stored_binary: Dict[int, float] = {}
    coefficient_error = Fraction(0)
    for destination, exact_values, width in (
        (stored_cont, cont, n_cont),
        (stored_binary, binary, n_bin),
    ):
        for raw_index, exact_value in exact_values.items():
            index = int(raw_index)
            if index < 0 or index >= width:
                raise PropertyMicroRLTError(
                    f"{row_name}: coefficient index {index} outside [0,{width})"
                )
            if exact_value == 0:
                continue
            try:
                stored = float(exact_value)
            except OverflowError as exc:
                raise PropertyMicroRLTError(
                    f"{row_name}: coefficient overflow"
                ) from exc
            if not math.isfinite(stored):
                raise PropertyMicroRLTError(
                    f"{row_name}: coefficient overflow"
                )
            destination[index] = stored
            coefficient_error += abs(
                Fraction.from_float(stored) - exact_value
            )

    # Every factor, including each new product factor, lies in [-1,1].
    # Hence this single exact L1 allowance covers every coefficient rounding
    # direction without relying on the candidate optimizer.
    stored_rhs = _outward_float(
        rhs + coefficient_error,
        name=f"{row_name}.rhs",
    )
    return stored_cont, stored_binary, stored_rhs


def _normalized_selection(
    selection: Mapping[int, Sequence[int]],
    *,
    n_bin: int,
    n_ub: int,
    max_binary_factors: int,
    max_source_rows_per_binary: int,
) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    if not isinstance(selection, Mapping) or not selection:
        raise PropertyMicroRLTError(
            "RLT selection must be a nonempty mapping"
        )
    if len(selection) > int(max_binary_factors):
        raise PropertyMicroRLTError(
            "RLT binary-factor cap exceeded"
        )
    normalized = []
    for raw_binary, raw_rows in selection.items():
        if isinstance(raw_binary, (bool, np.bool_)) or not isinstance(
            raw_binary, Integral
        ):
            raise PropertyMicroRLTError(
                "binary positions must be integers"
            )
        binary = int(raw_binary)
        if binary < 0 or binary >= n_bin:
            raise PropertyMicroRLTError(
                f"binary position {binary} outside [0,{n_bin})"
            )
        try:
            raw_row_values = tuple(raw_rows)
        except TypeError as exc:
            raise PropertyMicroRLTError(
                "source rows must be an iterable of integers"
            ) from exc
        if any(
            isinstance(row, (bool, np.bool_))
            or not isinstance(row, Integral)
            for row in raw_row_values
        ):
            raise PropertyMicroRLTError(
                "source row positions must be integers"
            )
        rows = tuple(int(row) for row in raw_row_values)
        if (
            not rows
            or len(rows) > int(max_source_rows_per_binary)
            or len(set(rows)) != len(rows)
        ):
            raise PropertyMicroRLTError(
                "source rows must be nonempty, unique, and within cap"
            )
        if any(row < 0 or row >= n_ub for row in rows):
            raise PropertyMicroRLTError(
                f"source row outside [0,{n_ub})"
            )
        normalized.append((binary, tuple(sorted(rows))))
    return tuple(sorted(normalized))


_ProductKey = Tuple[str, int, int]


def _continuous_product_key(
    selected_binary: int,
    continuous_column: int,
) -> _ProductKey:
    return (
        "continuous",
        int(selected_binary),
        int(continuous_column),
    )


def _binary_product_key(
    selected_binary: int,
    other_binary: int,
) -> _ProductKey:
    selected = int(selected_binary)
    other = int(other_binary)
    if selected == other:
        raise PropertyMicroRLTError(
            "a binary square must not allocate a product factor"
        )
    return ("binary", min(selected, other), max(selected, other))


def apply_property_micro_rlt(
    hz: SparseHZono,
    *,
    source_rows_by_binary: Mapping[int, Sequence[int]],
    max_binary_factors: int = 4,
    max_source_rows_per_binary: int = 8,
    max_product_factors: int = 4096,
    max_selected_row_nnz: int = 65536,
    max_requirement_scan_nnz: int = 65536,
) -> PropertyMicroRLTResult:
    """Append a bounded degree-1 RLT lift to a sparse HybridZ representation."""

    if not isinstance(hz, SparseHZono):
        raise TypeError("property micro-RLT requires SparseHZono")
    if hz.Auc is None or hz.Aub is None or hz.ub is None:
        raise PropertyMicroRLTError(
            "property micro-RLT requires upper constraints"
        )
    if min(
        int(max_binary_factors),
        int(max_source_rows_per_binary),
        int(max_product_factors),
        int(max_selected_row_nnz),
        int(max_requirement_scan_nnz),
    ) < 1:
        raise PropertyMicroRLTError("all RLT resource caps must be positive")

    live_Auc = sp.csr_matrix(hz.Auc, dtype=np.float64, copy=False)
    live_Aub = sp.csr_matrix(hz.Aub, dtype=np.float64, copy=False)
    live_ub = np.asarray(hz.ub, dtype=np.float64).reshape(-1)
    if (
        not np.all(np.isfinite(live_Auc.data))
        or not np.all(np.isfinite(live_Aub.data))
        or not np.all(np.isfinite(live_ub))
    ):
        raise PropertyMicroRLTError(
            "base upper constraints contain non-finite stored values"
        )

    selection = _normalized_selection(
        source_rows_by_binary,
        n_bin=hz.n_bin,
        n_ub=hz.n_ub,
        max_binary_factors=max_binary_factors,
        max_source_rows_per_binary=max_source_rows_per_binary,
    )
    selected_rows = [
        int(row)
        for _selected_binary, rows in selection
        for row in rows
    ]
    raw_selected_nnz = sum(
        int(live_Auc.indptr[row + 1] - live_Auc.indptr[row])
        + int(live_Aub.indptr[row + 1] - live_Aub.indptr[row])
        for row in selected_rows
    )
    if raw_selected_nnz > int(max_requirement_scan_nnz):
        raise PropertyMicroRLTError(
            "RLT requirement-scan nnz cap exceeded before "
            "canonicalization: "
            f"raw_required={raw_selected_nnz}, "
            f"cap={int(max_requirement_scan_nnz)}",
            reason_code="requirement_scan_cap_exceeded",
            requirement_count_complete=False,
        )

    # SparseHZono validates dimensions but does not promise canonical CSR.
    # Canonicalize only the at-most-small selected row multiset before
    # allocating Python product keys.  A cap miss can then remain a cheap,
    # complete no-op without copying the full ordinary constraint matrix.
    requirement_Auc = sp.csr_matrix(
        live_Auc[selected_rows], dtype=np.float64, copy=True
    )
    requirement_Aub = sp.csr_matrix(
        live_Aub[selected_rows], dtype=np.float64, copy=True
    )
    requirement_Auc.sum_duplicates()
    requirement_Auc.eliminate_zeros()
    requirement_Auc.sort_indices()
    requirement_Aub.sum_duplicates()
    requirement_Aub.eliminate_zeros()
    requirement_Aub.sort_indices()
    selected_nnz = sum(
        int(
            requirement_Auc.indptr[row + 1]
            - requirement_Auc.indptr[row]
        )
        + int(
            requirement_Aub.indptr[row + 1]
            - requirement_Aub.indptr[row]
        )
        for row in range(len(selected_rows))
    )
    # One shared product factor per (selected binary, other factor) pair.
    product_keys: list[_ProductKey] = []
    product_key_set: set[_ProductKey] = set()
    binary_product_orientations: Dict[
        _ProductKey, set[Tuple[int, int]]
    ] = {}
    selected_row_offset = 0
    for selected_binary, rows in selection:
        for _row in rows:
            c_start = int(requirement_Auc.indptr[selected_row_offset])
            c_end = int(requirement_Auc.indptr[selected_row_offset + 1])
            for column in requirement_Auc.indices[c_start:c_end]:
                key = _continuous_product_key(
                    selected_binary, int(column)
                )
                if key not in product_key_set:
                    product_key_set.add(key)
                    product_keys.append(key)
            b_start = int(requirement_Aub.indptr[selected_row_offset])
            b_end = int(requirement_Aub.indptr[selected_row_offset + 1])
            for column in requirement_Aub.indices[b_start:b_end]:
                if int(column) == selected_binary:
                    continue
                key = _binary_product_key(
                    selected_binary, int(column)
                )
                binary_product_orientations.setdefault(
                    key, set()
                ).add((selected_binary, int(column)))
                if key not in product_key_set:
                    product_key_set.add(key)
                    product_keys.append(key)
            selected_row_offset += 1
    product_keys.sort()
    if selected_nnz > int(max_selected_row_nnz):
        raise PropertyMicroRLTError(
            "selected source-row nnz cap exceeded: "
            f"required={selected_nnz}, "
            f"cap={int(max_selected_row_nnz)}; "
            f"product_factors_required={len(product_keys)}",
            selected_source_row_nnz_required=int(selected_nnz),
            product_factors_required=int(len(product_keys)),
            reason_code="selected_source_row_nnz_cap_exceeded",
            requirement_count_complete=True,
        )
    if len(product_keys) > int(max_product_factors):
        raise PropertyMicroRLTError(
            "RLT product-factor cap exceeded: "
            f"required={len(product_keys)}, "
            f"cap={int(max_product_factors)}",
            selected_source_row_nnz_required=int(selected_nnz),
            product_factors_required=int(len(product_keys)),
            reason_code="product_factor_cap_exceeded",
            requirement_count_complete=True,
        )

    # The request passed every cheap requirement cap.  Only now allocate and
    # canonicalize the complete matrices used by the proof-bearing lift.
    Auc = sp.csr_matrix(live_Auc, dtype=np.float64, copy=True)
    Aub = sp.csr_matrix(live_Aub, dtype=np.float64, copy=True)
    ub = live_ub.copy()
    Auc.sum_duplicates()
    Auc.eliminate_zeros()
    Auc.sort_indices()
    Aub.sum_duplicates()
    Aub.eliminate_zeros()
    Aub.sort_indices()

    old_n_cont = hz.n_cont
    n_aux = len(product_keys)
    new_n_cont = old_n_cont + n_aux
    product_column = {
        key: old_n_cont + offset
        for offset, key in enumerate(product_keys)
    }

    exact_rows: list[
        Tuple[Dict[int, Fraction], Dict[int, Fraction], Fraction, str]
    ] = []

    one = Fraction(1)
    minus_one = Fraction(-1)
    # Convex hull of v = s*q for s in {-1,1}, q in [-1,1].
    for factor_kind, left_position, right_position in product_keys:
        aux = product_column[
            (factor_kind, left_position, right_position)
        ]
        q_cont: Dict[int, Fraction] = {}
        q_binary: Dict[int, Fraction] = {}
        if factor_kind == "continuous":
            selected_binary = int(left_position)
            factor_column = int(right_position)
            q_cont[factor_column] = one
        else:
            # Binary products are commutative.  The canonical lower endpoint
            # is used only to emit one symmetric McCormick hull; RLT rows for
            # either multiplication orientation look up this same auxiliary
            # without a sign change.
            selected_binary = int(left_position)
            factor_column = int(right_position)
            q_binary[factor_column] = one

        def product_row(
            *,
            q_scale: Fraction,
            s_scale: Fraction,
            v_scale: Fraction,
            suffix: str,
        ) -> None:
            cont = {
                index: q_scale * value
                for index, value in q_cont.items()
            }
            cont[aux] = v_scale
            binary = {
                index: q_scale * value
                for index, value in q_binary.items()
            }
            binary[selected_binary] = (
                binary.get(selected_binary, Fraction(0))
                + s_scale
            )
            exact_rows.append(
                (
                    cont,
                    binary,
                    one,
                    f"product[{factor_kind},{left_position},"
                    f"{right_position}].{suffix}",
                )
            )

        product_row(
            q_scale=one,
            s_scale=one,
            v_scale=minus_one,
            suffix="lower_pp",
        )
        product_row(
            q_scale=minus_one,
            s_scale=minus_one,
            v_scale=minus_one,
            suffix="lower_nn",
        )
        product_row(
            q_scale=one,
            s_scale=minus_one,
            v_scale=one,
            suffix="upper_np",
        )
        product_row(
            q_scale=minus_one,
            s_scale=one,
            v_scale=one,
            suffix="upper_pn",
        )

    rlt_records = []
    for selected_binary, rows in selection:
        for source_row in rows:
            source_cont: Dict[int, Fraction] = {}
            source_binary: Dict[int, Fraction] = {}
            c_start = int(Auc.indptr[source_row])
            c_end = int(Auc.indptr[source_row + 1])
            for column, value in zip(
                Auc.indices[c_start:c_end],
                Auc.data[c_start:c_end],
            ):
                source_cont[int(column)] = _stored_fraction(
                    float(value),
                    name=f"Auc[{source_row},{int(column)}]",
                )
            b_start = int(Aub.indptr[source_row])
            b_end = int(Aub.indptr[source_row + 1])
            for column, value in zip(
                Aub.indices[b_start:b_end],
                Aub.data[b_start:b_end],
            ):
                source_binary[int(column)] = _stored_fraction(
                    float(value),
                    name=f"Aub[{source_row},{int(column)}]",
                )
            source_rhs = _stored_fraction(
                float(ub[source_row]),
                name=f"ub[{source_row}]",
            )
            selected_coefficient = source_binary.get(
                selected_binary, Fraction(0)
            )
            generated_indices = []
            for sign, suffix in ((1, "plus"), (-1, "minus")):
                sign_fraction = Fraction(sign)
                cont = dict(source_cont)
                binary = dict(source_binary)
                binary[selected_binary] = (
                    binary.get(selected_binary, Fraction(0))
                    - sign_fraction * source_rhs
                )
                for column, value in source_cont.items():
                    aux = product_column[
                        _continuous_product_key(
                            selected_binary, column
                        )
                    ]
                    cont[aux] = (
                        cont.get(aux, Fraction(0))
                        + sign_fraction * value
                    )
                for column, value in source_binary.items():
                    if column == selected_binary:
                        continue
                    aux = product_column[
                        _binary_product_key(
                            selected_binary, column
                        )
                    ]
                    cont[aux] = (
                        cont.get(aux, Fraction(0))
                        + sign_fraction * value
                    )
                rhs = (
                    source_rhs
                    - sign_fraction * selected_coefficient
                )
                generated_indices.append(len(exact_rows))
                exact_rows.append(
                    (
                        cont,
                        binary,
                        rhs,
                        f"rlt[{selected_binary},{source_row}].{suffix}",
                    )
                )
            rlt_records.append(
                {
                    "binary_position": selected_binary,
                    "source_upper_row": source_row,
                    "generated_local_rows": generated_indices,
                }
            )

    new_cont_rows: list[int] = []
    new_cont_cols: list[int] = []
    new_cont_data: list[float] = []
    new_bin_rows: list[int] = []
    new_bin_cols: list[int] = []
    new_bin_data: list[float] = []
    new_rhs = np.zeros(len(exact_rows), dtype=np.float64)
    row_names = []
    for row_index, (cont, binary, rhs, row_name) in enumerate(
        exact_rows
    ):
        stored_cont, stored_binary, stored_rhs = _store_upper_row(
            cont=cont,
            binary=binary,
            rhs=rhs,
            n_cont=new_n_cont,
            n_bin=hz.n_bin,
            row_name=row_name,
        )
        for column, value in stored_cont.items():
            new_cont_rows.append(row_index)
            new_cont_cols.append(column)
            new_cont_data.append(value)
        for column, value in stored_binary.items():
            new_bin_rows.append(row_index)
            new_bin_cols.append(column)
            new_bin_data.append(value)
        new_rhs[row_index] = stored_rhs
        row_names.append(row_name)

    Auc_padded = sp.hstack(
        [
            Auc,
            sp.csr_matrix((hz.n_ub, n_aux), dtype=np.float64),
        ],
        format="csr",
    )
    new_Auc = sp.csr_matrix(
        (
            np.asarray(new_cont_data, dtype=np.float64),
            (
                np.asarray(new_cont_rows, dtype=np.int64),
                np.asarray(new_cont_cols, dtype=np.int64),
            ),
        ),
        shape=(len(exact_rows), new_n_cont),
        dtype=np.float64,
    )
    new_Aub = sp.csr_matrix(
        (
            np.asarray(new_bin_data, dtype=np.float64),
            (
                np.asarray(new_bin_rows, dtype=np.int64),
                np.asarray(new_bin_cols, dtype=np.int64),
            ),
        ),
        shape=(len(exact_rows), hz.n_bin),
        dtype=np.float64,
    )
    result_Auc = sp.vstack([Auc_padded, new_Auc], format="csr")
    result_Aub = sp.vstack([Aub, new_Aub], format="csr")
    result_ub = np.concatenate([ub, new_rhs])

    zero_output = sp.csr_matrix(
        (hz.n_out, n_aux), dtype=np.float64
    )
    result_Gc = sp.hstack([hz.Gc, zero_output], format="csr")
    result_Ac = sp.hstack(
        [
            hz.Ac,
            sp.csr_matrix((hz.n_eq, n_aux), dtype=np.float64),
        ],
        format="csr",
    )
    result_col_ids = None
    if hz.col_ids is not None:
        old_ids = np.asarray(
            hz.col_ids, dtype=np.int64
        ).reshape(-1)
        old_binary_ids = (
            np.zeros(0, dtype=np.int64)
            if hz.bcol_ids is None
            else np.asarray(
                hz.bcol_ids, dtype=np.int64
            ).reshape(-1)
        )
        existing_ids = np.concatenate(
            [old_ids, old_binary_ids]
        )
        if np.unique(existing_ids).size != existing_ids.size:
            raise PropertyMicroRLTError(
                "base continuous and binary generator IDs overlap"
            )
        fresh_ids = (
            hz_fresh_col_ids(n_aux, device="cpu")
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
            .reshape(-1)
        )
        if (
            fresh_ids.size != n_aux
            or np.unique(fresh_ids).size != fresh_ids.size
            or np.any(fresh_ids < 0)
        ):
            raise PropertyMicroRLTError(
                "global continuous generator allocator returned malformed IDs"
            )
        if (
            fresh_ids.size
            and np.intersect1d(
                existing_ids, fresh_ids, assume_unique=True
            ).size
        ):
            raise PropertyMicroRLTError(
                "fresh continuous generator IDs collide with the base frame"
            )
        result_col_ids = np.concatenate([old_ids.copy(), fresh_ids])

    result_hz = SparseHZono(
        c=hz.c.copy(),
        Gc=result_Gc,
        Gb=hz.Gb.copy(),
        Ac=result_Ac,
        Ab=hz.Ab.copy(),
        b=hz.b.copy(),
        Auc=result_Auc,
        Aub=result_Aub,
        ub=result_ub,
        col_ids=result_col_ids,
        bcol_ids=(
            None
            if hz.bcol_ids is None
            else np.asarray(hz.bcol_ids, dtype=np.int64).copy()
        ),
    )
    if bool(getattr(hz, "_solver_known_nonempty", False)):
        setattr(result_hz, "_solver_known_nonempty", True)
        setattr(
            result_hz,
            "_solver_known_nonempty_reason",
            "property_micro_rlt_exact_integer_extension:"
            + str(
                getattr(
                    hz,
                    "_solver_known_nonempty_reason",
                    "base_known_nonempty",
                )
            ),
        )

    product_records = []
    for kind, left, right in product_keys:
        record: Dict[str, Any] = {
            # For a binary pair these are canonical endpoints, not a claim
            # about which endpoint was the selected RLT multiplier.
            "binary_position": int(left),
            "factor_kind": str(kind),
            "factor_position": int(right),
            "aux_continuous_position": int(
                product_column[(kind, left, right)]
            ),
            "orientation_sign": 1,
        }
        if kind == "binary":
            orientations = sorted(
                binary_product_orientations.get(
                    (kind, left, right), set()
                )
            )
            record.update(
                {
                    "unordered_binary_pair": [
                        int(left),
                        int(right),
                    ],
                    "commutative": True,
                    "orientation_uses": [
                        {
                            "selected_binary_position": int(selected),
                            "other_binary_position": int(other),
                            "orientation_sign": 1,
                        }
                        for selected, other in orientations
                    ],
                }
            )
        product_records.append(record)
    payload: Dict[str, Any] = {
        "schema": "act.property_micro_rlt.v1",
        "proof_authority": True,
        "selection_proof_authority": False,
        "construction": (
            "signed_degree1_rlt_both_(1+s)r_and_(1-s)r+"
            "four_row_product_hull+exact_fraction_storage_guard"
        ),
        "intended_consumer": (
            "parent_binary_relaxation_before_exact_phase_enumeration"
        ),
        "binary_factor_encoding": "signed_pm1",
        "solver_relaxation_encoding": (
            "z_in_0_1_with_signed_factor_s_equals_2z_minus_1"
        ),
        "fixed_phase_rows_are_redundant": True,
        "integer_set_equivalent": True,
        "original_rows_retained": True,
        "selected_binary_count": len(selection),
        "selected_source_row_count": sum(
            len(rows) for _, rows in selection
        ),
        "selected_source_row_nnz": selected_nnz,
        "new_product_factors": n_aux,
        "new_product_hull_rows": 4 * n_aux,
        "new_rlt_rows": 2
        * sum(len(rows) for _, rows in selection),
        "new_upper_rows": len(exact_rows),
        "base_n_out": hz.n_out,
        "base_n_cont": old_n_cont,
        "base_n_bin": hz.n_bin,
        "base_n_eq": hz.n_eq,
        "result_n_cont": result_hz.n_cont,
        "base_n_ub": hz.n_ub,
        "result_n_out": result_hz.n_out,
        "result_n_bin": result_hz.n_bin,
        "result_n_eq": result_hz.n_eq,
        "result_n_ub": result_hz.n_ub,
        "selection": [
            {
                "binary_position": binary,
                "source_upper_rows": list(rows),
            }
            for binary, rows in selection
        ],
        "product_records": product_records,
        "rlt_records": rlt_records,
        "generated_row_names": row_names,
        "base_center_sha256": _array_sha256(
            hz.c, dtype=np.float64
        ),
        "result_center_sha256": _array_sha256(
            result_hz.c, dtype=np.float64
        ),
        "base_output_continuous_csr_sha256": _csr_sha256(hz.Gc),
        "result_output_continuous_csr_sha256": _csr_sha256(
            result_hz.Gc
        ),
        "base_output_binary_csr_sha256": _csr_sha256(hz.Gb),
        "result_output_binary_csr_sha256": _csr_sha256(
            result_hz.Gb
        ),
        "base_equality_continuous_csr_sha256": _csr_sha256(hz.Ac),
        "result_equality_continuous_csr_sha256": _csr_sha256(
            result_hz.Ac
        ),
        "base_equality_binary_csr_sha256": _csr_sha256(hz.Ab),
        "result_equality_binary_csr_sha256": _csr_sha256(
            result_hz.Ab
        ),
        "base_equality_rhs_sha256": _array_sha256(
            hz.b, dtype=np.float64
        ),
        "result_equality_rhs_sha256": _array_sha256(
            result_hz.b, dtype=np.float64
        ),
        "base_upper_csr_sha256": _csr_sha256(Auc),
        "base_upper_binary_csr_sha256": _csr_sha256(Aub),
        "result_upper_csr_sha256": _csr_sha256(result_Auc),
        "result_upper_binary_csr_sha256": _csr_sha256(
            result_Aub
        ),
        "base_upper_rhs_sha256": _array_sha256(
            ub, dtype=np.float64
        ),
        "result_upper_rhs_sha256": _array_sha256(
            result_ub, dtype=np.float64
        ),
        "base_continuous_col_ids_sha256": _optional_ids_sha256(
            hz.col_ids
        ),
        "result_continuous_col_ids_sha256": _optional_ids_sha256(
            result_hz.col_ids
        ),
        "base_binary_col_ids_sha256": _optional_ids_sha256(
            hz.bcol_ids
        ),
        "result_binary_col_ids_sha256": _optional_ids_sha256(
            result_hz.bcol_ids
        ),
        "live_result_validation_required": True,
        "receipt_self_hash_alone_has_proof_authority": False,
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    setattr(result_hz, "_property_micro_rlt_receipt", dict(payload))
    return PropertyMicroRLTResult(result_hz, payload)
