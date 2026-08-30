# ===- constraint_program_highs_schedule_candidate.py -------------------===#
"""Disconnected solver-ready primary representation experiment.

This module deliberately does *not* import the frozen constraint-program core,
Operator-HZ, or the legacy solver.  It measures one primary representation:

* binary-free RANGE/LE blocks are stored directly in HiGHS coordinates;
* sparse mixed rows are transformed once at construction with exact dyadic
  integer arithmetic; and
* unsupported numeric/mixed cases fail immediately instead of selecting a
  fallback implementation.

It has no producer, proof, verdict, or production authority.  It is an offline
common-path performance and algebra candidate only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import highspy
except Exception:  # pragma: no cover - optional backend
    highspy = None


_SMALL = 1.0e-12
_LARGE = 1.0e15
_INFINITE = 1.0e20
_MAX_I32 = int(np.iinfo(np.int32).max)


class SolverReadyScheduleError(RuntimeError):
    """Malformed candidate input or backend failure."""


class SolverReadyUnsupported(SolverReadyScheduleError):
    """The bounded common path is inapplicable; caller must return UNKNOWN."""


@dataclass(frozen=True)
class SourceBlock:
    """Transient exact source rows supplied by an offline producer fixture."""

    family: str  # ``range`` or ``le``
    A_cont: sp.csr_matrix
    A_bin: sp.csr_matrix
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _Segment:
    rows: int
    columns: int
    data: bytes
    indices: bytes
    indptr: bytes
    lower: bytes
    upper: bytes
    direct_binary_free: bool
    source_binary_nnz: int

    @property
    def nnz(self) -> int:
        return len(self.data) // 8

    def arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return views of bytes validated once by the schedule builder.

        This candidate deliberately has one validation boundary: callers must
        pass the exact schedule returned by ``build_solver_ready_schedule``.
        Re-scanning every immutable CSR buffer immediately before ``addRows``
        made the common Q=1 path pay for the same graph twice.
        """
        if (
            type(self) is not _Segment
            or type(self.rows) is not int
            or type(self.columns) is not int
            or self.rows <= 0
            or self.columns < 0
            or any(type(value) is not bytes for value in (
                self.data,
                self.indices,
                self.indptr,
                self.lower,
                self.upper,
            ))
            or len(self.indices) != self.nnz * 4
            or len(self.indptr) != (self.rows + 1) * 4
            or len(self.lower) != self.rows * 8
            or len(self.upper) != self.rows * 8
            or type(self.direct_binary_free) is not bool
            or type(self.source_binary_nnz) is not int
            or self.source_binary_nnz < 0
        ):
            raise SolverReadyScheduleError("solver-ready segment is malformed")
        data = np.frombuffer(self.data, dtype=np.float64)
        indices = np.frombuffer(self.indices, dtype=np.int32)
        indptr = np.frombuffer(self.indptr, dtype=np.int32)
        lower = np.frombuffer(self.lower, dtype=np.float64)
        upper = np.frombuffer(self.upper, dtype=np.float64)
        return data, indices, indptr, lower, upper


@dataclass(frozen=True)
class SolverReadySchedule:
    """One bytes-backed solver-coordinate representation; never authority."""

    n_cont: int
    n_bin: int
    continuous_ids: Tuple[int, ...]
    binary_ids: Tuple[int, ...]
    segments: Tuple[_Segment, ...]
    source_rows: int
    source_nnz: int
    direct_rows: int
    mixed_rows: int

    @property
    def receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "act.solver_ready_primary_candidate.v1",
                "candidate_only": True,
                "single_primary_representation": True,
                "legacy_representation_retained": False,
                "runtime_fallback_count": 0,
                "python_fraction_used": False,
                "postsolve_full_constraint_replay": False,
                "input_graph_scans": 1,
                "load_time_full_graph_rescan": False,
                "builder_returned_schedule_required": True,
                "public_schedule_forgery_protected": False,
                "producer_authenticated": False,
                "production_integration": False,
                "proof_authority": False,
                "verdict_authority": False,
                "solver_status_authority": False,
                "source_rows": self.source_rows,
                "source_nnz": self.source_nnz,
                "direct_rows": self.direct_rows,
                "mixed_rows": self.mixed_rows,
                "segment_count": len(self.segments),
                "triangle_relaxation_called": False,
                "act_network_branch_and_bound_called": False,
                "backward_called": False,
                "dual_called": False,
                "real_model_called": False,
                "large_model_called": False,
            }
        )


@dataclass(frozen=True)
class SolverReadyResult:
    model_status: str
    rows_loaded: int
    nnz_loaded: int
    receipt: Mapping[str, Any]


def _exact_ids(value: Any, *, name: str) -> Tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int or item < 0 for item in value):
        raise SolverReadyScheduleError(f"{name} must be exact nonnegative int IDs")
    if len(set(value)) != len(value):
        raise SolverReadyScheduleError(f"{name} IDs are not unique")
    return value


def _snapshot_csr(
    value: Any, *, rows: int, columns: int, name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        type(value) is not sp.csr_matrix
        or value.dtype != np.dtype(np.float64)
        or value.indices.dtype != np.dtype(np.int32)
        or value.indptr.dtype != np.dtype(np.int32)
    ):
        raise SolverReadyScheduleError(
            f"{name} must be exact float64 CSR with int32 structure"
        )
    # Capture each live input buffer directly into its one immutable owner.
    # The views below are then both the validation input and emitted payload.
    data = np.frombuffer(value.data.tobytes(order="C"), dtype=np.float64)
    indices = np.frombuffer(value.indices.tobytes(order="C"), dtype=np.int32)
    indptr = np.frombuffer(value.indptr.tobytes(order="C"), dtype=np.int32)
    if (
        value.shape != (rows, columns)
        or data.ndim != 1
        or indices.ndim != 1
        or indptr.shape != (rows + 1,)
        or data.size != indices.size
        or int(indptr[0]) != 0
        or int(indptr[-1]) != data.size
        or np.any(indptr[1:] < indptr[:-1])
        or (indices.size and (np.any(indices < 0) or np.any(indices >= columns)))
        or not np.all(np.isfinite(data))
        or np.any(data == 0.0)
    ):
        raise SolverReadyScheduleError(f"{name} is malformed")
    if indices.size > 1:
        nonincreasing = indices[1:] <= indices[:-1]
        boundaries = indptr[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < indices.size)]
        nonincreasing[boundaries - 1] = False
        if np.any(nonincreasing):
            raise SolverReadyScheduleError(f"{name} is not canonical")
    return data, indices, indptr


def _snapshot_bounds(
    lower: Any, upper: Any, *, rows: int
) -> Tuple[np.ndarray, np.ndarray]:
    if (
        type(lower) is not np.ndarray
        or type(upper) is not np.ndarray
        or lower.dtype != np.dtype(np.float64)
        or upper.dtype != np.dtype(np.float64)
        or lower.shape != (rows,)
        or upper.shape != (rows,)
    ):
        raise SolverReadyScheduleError("row bounds must be exact float64 vectors")
    lo = np.frombuffer(lower.tobytes(order="C"), dtype=np.float64)
    hi = np.frombuffer(upper.tobytes(order="C"), dtype=np.float64)
    if (
        np.any(np.isnan(lo))
        or np.any(np.isposinf(lo))
        or not np.all(np.isfinite(hi))
        or np.any(lo > hi)
    ):
        raise SolverReadyScheduleError("row bounds are malformed")
    return lo, hi


def _dyadic(value: float) -> Tuple[int, int]:
    numerator, denominator = float(value).as_integer_ratio()
    if denominator <= 0 or denominator & (denominator - 1):
        raise SolverReadyScheduleError("binary64 ratio is not dyadic")
    return numerator, denominator.bit_length() - 1


def _sum_dyadic(values: np.ndarray) -> Tuple[int, int]:
    if values.size == 0:
        return 0, 0
    pieces = [_dyadic(float(item)) for item in values]
    exponent = max(item[1] for item in pieces)
    numerator = sum(item[0] << (exponent - item[1]) for item in pieces)
    return _normalize_dyadic(numerator, exponent)


def _normalize_dyadic(numerator: int, exponent: int) -> Tuple[int, int]:
    if numerator == 0:
        return 0, 0
    while exponent > 0 and numerator % 2 == 0:
        numerator //= 2
        exponent -= 1
    return numerator, exponent


def _add_dyadic(left: Tuple[int, int], right: Tuple[int, int]) -> Tuple[int, int]:
    exponent = max(left[1], right[1])
    return _normalize_dyadic(
        (left[0] << (exponent - left[1]))
        + (right[0] << (exponent - right[1])),
        exponent,
    )


def _compare_float_to_dyadic(value: float, exact: Tuple[int, int]) -> int:
    numerator, denominator = value.as_integer_ratio()
    left = numerator << exact[1]
    right = exact[0] * denominator
    return (left > right) - (left < right)


def _dyadic_float(exact: Tuple[int, int], *, direction: str) -> float:
    numerator, exponent = exact
    if numerator == 0:
        return 0.0
    try:
        nearest = numerator / (1 << exponent)
    except OverflowError as error:
        raise SolverReadyUnsupported("dyadic value overflows binary64") from error
    if not math.isfinite(nearest):
        raise SolverReadyUnsupported("dyadic value is non-finite")
    comparison = _compare_float_to_dyadic(nearest, exact)
    if direction == "lower" and comparison > 0:
        nearest = float(np.nextafter(nearest, -np.inf))
    elif direction == "upper" and comparison < 0:
        nearest = float(np.nextafter(nearest, np.inf))
    elif direction not in {"lower", "upper", "nearest"}:
        raise SolverReadyScheduleError("unknown dyadic rounding direction")
    if not math.isfinite(nearest):
        raise SolverReadyUnsupported("directed dyadic value overflows")
    final = _compare_float_to_dyadic(nearest, exact)
    if (direction == "lower" and final > 0) or (direction == "upper" and final < 0):
        raise SolverReadyScheduleError("directed dyadic rounding failed")
    return nearest


def _double_exact(value: float) -> float:
    doubled = float(value) * 2.0
    if (
        not math.isfinite(doubled)
        or _dyadic(doubled) != _add_dyadic(_dyadic(value), _dyadic(value))
        or (value != 0.0 and doubled == 0.0)
    ):
        raise SolverReadyUnsupported("2*A_bin is not exact finite binary64")
    return doubled


def _threshold_coefficients(values: np.ndarray) -> None:
    if values.size and (
        np.any(np.abs(values) <= _SMALL) or np.any(np.abs(values) >= _LARGE)
    ):
        raise SolverReadyUnsupported("coefficient is outside the direct HiGHS contract")


def _threshold_bounds(lower: np.ndarray, upper: np.ndarray) -> None:
    finite_lower = lower[np.isfinite(lower)]
    if (
        (finite_lower.size and np.any(np.abs(finite_lower) >= _INFINITE))
        or np.any(np.abs(upper) >= _INFINITE)
    ):
        raise SolverReadyUnsupported("row bound is outside the direct HiGHS contract")


def _owned_bytes(value: np.ndarray) -> bytes:
    """Reuse a whole bytes-backed capture, otherwise freeze a generated array."""

    if type(value.base) is bytes and value.nbytes == len(value.base):
        return value.base
    return value.tobytes(order="C")


def _segment_direct(
    Ac: Tuple[np.ndarray, np.ndarray, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    rows: int,
    columns: int,
) -> _Segment:
    data, indices, indptr = Ac
    _threshold_coefficients(data)
    _threshold_bounds(lower, upper)
    if data.size > _MAX_I32 or indptr[-1] > _MAX_I32 or columns > _MAX_I32:
        raise SolverReadyUnsupported("direct block exceeds int32 resources")
    return _Segment(
        rows,
        columns,
        _owned_bytes(data),
        _owned_bytes(indices),
        _owned_bytes(indptr),
        _owned_bytes(lower),
        _owned_bytes(upper),
        True,
        0,
    )


def _segment_mixed(
    Ac: Tuple[np.ndarray, np.ndarray, np.ndarray],
    Ab: Tuple[np.ndarray, np.ndarray, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    rows: int,
    n_cont: int,
    n_bin: int,
    max_binary_nnz_per_row: int,
) -> _Segment:
    c_data, c_indices, c_indptr = Ac
    b_data, b_indices, b_indptr = Ab
    counts = np.diff(b_indptr)
    if counts.size and int(counts.max()) > max_binary_nnz_per_row:
        raise SolverReadyUnsupported("mixed row exceeds the bounded common binary fanin")
    total = int(c_data.size + b_data.size)
    if total > _MAX_I32 or n_cont + n_bin > _MAX_I32:
        raise SolverReadyUnsupported("mixed block exceeds int32 resources")
    data = np.empty(total, dtype=np.float64)
    indices = np.empty(total, dtype=np.int32)
    indptr = np.empty(rows + 1, dtype=np.int32)
    shifted_lower = np.empty(rows, dtype=np.float64)
    shifted_upper = np.empty(rows, dtype=np.float64)
    indptr[0] = 0
    cursor = 0
    for row in range(rows):
        cs, ce = int(c_indptr[row]), int(c_indptr[row + 1])
        bs, be = int(b_indptr[row]), int(b_indptr[row + 1])
        c_count, b_count = ce - cs, be - bs
        stop = cursor + c_count + b_count
        data[cursor:cursor + c_count] = c_data[cs:ce]
        indices[cursor:cursor + c_count] = c_indices[cs:ce]
        shift = _sum_dyadic(b_data[bs:be])
        for local, value in enumerate(b_data[bs:be]):
            data[cursor + c_count + local] = _double_exact(float(value))
        if b_count:
            indices[cursor + c_count:stop] = n_cont + b_indices[bs:be]
        lo = float(lower[row])
        shifted_lower[row] = (
            -np.inf
            if math.isinf(lo) and lo < 0.0
            else _dyadic_float(_add_dyadic(_dyadic(lo), shift), direction="lower")
        )
        shifted_upper[row] = _dyadic_float(
            _add_dyadic(_dyadic(float(upper[row])), shift), direction="upper"
        )
        cursor = stop
        indptr[row + 1] = cursor
    _threshold_coefficients(data)
    _threshold_bounds(shifted_lower, shifted_upper)
    return _Segment(
        rows,
        n_cont + n_bin,
        _owned_bytes(data),
        _owned_bytes(indices),
        _owned_bytes(indptr),
        _owned_bytes(shifted_lower),
        _owned_bytes(shifted_upper),
        False,
        int(b_data.size),
    )


def build_solver_ready_schedule(
    blocks: Sequence[SourceBlock],
    *,
    continuous_ids: Tuple[int, ...],
    binary_ids: Tuple[int, ...],
    max_binary_nnz_per_row: int = 2,
) -> SolverReadySchedule:
    """Build one primary solver representation or reject the common path."""

    cont_ids = _exact_ids(continuous_ids, name="continuous")
    bin_ids = _exact_ids(binary_ids, name="binary")
    if set(cont_ids).intersection(bin_ids):
        raise SolverReadyScheduleError("factor IDs collide across kinds")
    if type(blocks) not in {tuple, list} or not blocks:
        raise SolverReadyScheduleError("blocks must be a nonempty builtin sequence")
    if type(max_binary_nnz_per_row) is not int or not 1 <= max_binary_nnz_per_row <= 2:
        raise SolverReadyScheduleError("max_binary_nnz_per_row must be 1 or 2")
    segments = []
    total_rows = total_nnz = direct_rows = mixed_rows = 0
    for block in blocks:
        if type(block) is not SourceBlock or block.family not in {"range", "le"}:
            raise SolverReadyScheduleError("source block is malformed")
        if type(block.A_cont) is not sp.csr_matrix:
            raise SolverReadyScheduleError("source block lost A_cont")
        rows = int(block.A_cont.shape[0])
        if rows <= 0:
            raise SolverReadyScheduleError("empty source blocks are not stored")
        Ac = _snapshot_csr(block.A_cont, rows=rows, columns=len(cont_ids), name="A_cont")
        Ab = _snapshot_csr(block.A_bin, rows=rows, columns=len(bin_ids), name="A_bin")
        lower, upper = _snapshot_bounds(block.lower, block.upper, rows=rows)
        if block.family == "le" and not np.all(np.isneginf(lower)):
            raise SolverReadyScheduleError("LE source block must have -inf lower bounds")
        if Ab[0].size == 0:
            segment = _segment_direct(
                Ac, lower, upper, rows=rows, columns=len(cont_ids) + len(bin_ids)
            )
            direct_rows += rows
        else:
            segment = _segment_mixed(
                Ac,
                Ab,
                lower,
                upper,
                rows=rows,
                n_cont=len(cont_ids),
                n_bin=len(bin_ids),
                max_binary_nnz_per_row=max_binary_nnz_per_row,
            )
            mixed_rows += rows
        segments.append(segment)
        total_rows += rows
        total_nnz += segment.nnz
    return SolverReadySchedule(
        len(cont_ids),
        len(bin_ids),
        cont_ids,
        bin_ids,
        tuple(segments),
        total_rows,
        total_nnz,
        direct_rows,
        mixed_rows,
    )


def _mapped_objective(objective: np.ndarray, n_cont: int) -> Tuple[np.ndarray, float]:
    if (
        type(objective) is not np.ndarray
        or objective.dtype != np.dtype(np.float64)
        or objective.ndim != 1
        or not objective.flags.c_contiguous
        or not np.all(np.isfinite(objective))
    ):
        raise SolverReadyScheduleError("objective must be finite C-contiguous float64")
    mapped = np.array(objective, dtype=np.float64, copy=True)
    original_binary = mapped[n_cont:]
    # Network/property objectives normally do not price ReLU phase bits.  Keep
    # that overwhelmingly common case vector-only and avoid exact scalar work.
    if not np.any(original_binary):
        return mapped, 0.0
    original_binary = np.array(original_binary, dtype=np.float64, copy=True)
    for index, value in enumerate(original_binary):
        mapped[n_cont + index] = _double_exact(float(value))
    offset_exact = _sum_dyadic(-original_binary)
    offset = _dyadic_float(offset_exact, direction="nearest")
    if np.any(np.abs(mapped) >= _INFINITE) or abs(offset) >= _INFINITE:
        raise SolverReadyUnsupported("objective is outside the direct HiGHS contract")
    return mapped, offset


def solve_solver_ready_schedule(
    schedule: SolverReadySchedule,
    objective: np.ndarray,
    *,
    maximize: bool = False,
) -> SolverReadyResult:
    """Load one primary schedule, solve once, clear, and return no authority."""

    if highspy is None:
        raise SolverReadyUnsupported("highspy is unavailable")
    if type(schedule) is not SolverReadySchedule or type(maximize) is not bool:
        raise SolverReadyScheduleError("schedule or maximize has the wrong exact type")
    if objective.shape != (schedule.n_cont + schedule.n_bin,):
        raise SolverReadyScheduleError("objective width differs from schedule")
    costs, offset = _mapped_objective(objective, schedule.n_cont)
    highs = highspy.Highs()
    primary: Optional[BaseException] = None
    result: Optional[SolverReadyResult] = None
    try:
        for name, value in (
            ("output_flag", False),
            ("threads", 1),
            ("small_matrix_value", _SMALL),
            ("large_matrix_value", _LARGE),
            ("infinite_bound", _INFINITE),
        ):
            if highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise SolverReadyUnsupported("HiGHS option setup failed")
        columns = schedule.n_cont + schedule.n_bin
        starts = np.zeros(columns + 1, dtype=np.int32)
        lower = np.concatenate((
            np.full(schedule.n_cont, -1.0), np.zeros(schedule.n_bin)
        ))
        upper = np.ones(columns, dtype=np.float64)
        if highs.addCols(
            columns,
            costs,
            lower,
            upper,
            0,
            starts,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        ) != highspy.HighsStatus.kOk:
            raise SolverReadyUnsupported("HiGHS addCols failed")
        if schedule.n_bin:
            if highs.changeColsIntegrality(
                schedule.n_bin,
                np.arange(schedule.n_cont, columns, dtype=np.int32),
                np.full(
                    schedule.n_bin,
                    int(highspy.HighsVarType.kInteger),
                    dtype=np.uint8,
                ),
            ) != highspy.HighsStatus.kOk:
                raise SolverReadyUnsupported("HiGHS integrality setup failed")
        if highs.changeObjectiveOffset(offset) != highspy.HighsStatus.kOk:
            raise SolverReadyUnsupported("HiGHS objective offset failed")
        if maximize and highs.changeObjectiveSense(
            highspy.ObjSense.kMaximize
        ) != highspy.HighsStatus.kOk:
            raise SolverReadyUnsupported("HiGHS objective sense failed")
        rows_loaded = nnz_loaded = 0
        for segment in schedule.segments:
            data, indices, indptr, row_lower, row_upper = segment.arrays()
            if highs.addRows(
                segment.rows,
                row_lower,
                row_upper,
                segment.nnz,
                indptr,
                indices,
                data,
            ) != highspy.HighsStatus.kOk:
                raise SolverReadyUnsupported("HiGHS addRows rejected the direct contract")
            rows_loaded += segment.rows
            nnz_loaded += segment.nnz
        if (
            highs.getNumCol() != columns
            or highs.getNumRow() != schedule.source_rows
            or highs.getNumNz() != schedule.source_nnz
        ):
            raise SolverReadyScheduleError("HiGHS model accounting changed")
        if highs.run() != highspy.HighsStatus.kOk:
            raise SolverReadyUnsupported("HiGHS run failed")
        result = SolverReadyResult(
            str(highs.getModelStatus()),
            rows_loaded,
            nnz_loaded,
            MappingProxyType(
                {
                    **dict(schedule.receipt),
                    "native_model_loaded": True,
                    "native_solver_run": True,
                    "witness_validated": False,
                    "full_promotion_gate_complete": False,
                    "promotion": False,
                }
            ),
        )
    except BaseException as error:
        primary = error
    finally:
        try:
            status = highs.clear()
            if status != highspy.HighsStatus.kOk:
                raise SolverReadyScheduleError("HiGHS clear failed")
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                try:
                    primary.add_note("solver-ready HiGHS cleanup also failed")
                except BaseException:
                    pass
    if primary is not None:
        raise primary
    if result is None:
        raise SolverReadyScheduleError("solver-ready run produced no result")
    return result


__all__ = (
    "SolverReadyResult",
    "SolverReadySchedule",
    "SolverReadyScheduleError",
    "SolverReadyUnsupported",
    "SourceBlock",
    "build_solver_ready_schedule",
    "solve_solver_ready_schedule",
)
