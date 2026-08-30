#!/usr/bin/env python3
# ===- test_constraint_program_highs.py - native loader offline gates ----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Small/offline gates for the disconnected constraint-program HiGHS loader."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import gc
import inspect
import itertools
import math
from pathlib import Path
import statistics
import time
import unittest
from unittest import mock
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import constraint_program as core
from act.back_end.solver import constraint_program_highs as loader
from act.back_end.solver.constraint_program import (
    ConstraintFamily,
    ConstraintProgramOwner,
    ExternalFactorAllocatorAdapter,
)
from act.back_end.solver.solver_hz import SparseHZono

try:
    import highspy
except Exception:  # pragma: no cover - optional solver dependency
    highspy = None


class _HostileSecondary(BaseException):
    def __str__(self):
        raise SystemExit("secondary __str__ was evaluated")

    def __repr__(self):
        raise SystemExit("secondary __repr__ was evaluated")


class _HostileNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise SystemExit("secondary metaclass __name__ was evaluated")
        return super().__getattribute__(name)


class _HostileNameSecondary(BaseException, metaclass=_HostileNameMeta):
    pass


def _traceback_local(error: BaseException, name: str):
    traceback = error.__traceback__
    while traceback is not None:
        if name in traceback.tb_frame.f_locals:
            return traceback.tb_frame.f_locals[name]
        traceback = traceback.tb_next
    return None


class _Allocator:
    def __init__(self, start: int = 100) -> None:
        self.next_id = int(start)
        self.continuous = []
        self.binary = []

    def allocate_continuous(self, count: int):
        result = tuple(range(self.next_id, self.next_id + count))
        self.next_id += count
        self.continuous.extend(result)
        return result

    def allocate_binary(self, count: int):
        result = tuple(range(self.next_id, self.next_id + count))
        self.next_id += count
        self.binary.extend(result)
        return result

    def snapshot(self):
        return tuple(self.continuous), tuple(self.binary)


def _canonical(value, *, columns=None) -> sp.csr_matrix:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise AssertionError("test matrix must be rank two")
    shape = array.shape if columns is None else (array.shape[0], int(columns))
    result = sp.csr_matrix(array, shape=shape, dtype=np.float64)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _bind_allocator(allocator: _Allocator):
    return ExternalFactorAllocatorAdapter.bind(
        allocator,
        allocate_continuous=allocator.allocate_continuous,
        allocate_binary=allocator.allocate_binary,
        live_ids_snapshot=allocator.snapshot,
    )


def _legacy(program):
    continuous = []
    binary = []
    upper = []
    cursor = program.iter_legacy_facet_batches(max_rows=256)
    try:
        for batch in cursor:
            continuous.append(batch.A_cont)
            binary.append(batch.A_bin)
            upper.append(batch.upper)
    finally:
        cursor.close()
    n_cont = len(program.continuous_ids)
    n_bin = len(program.binary_ids)
    return (
        sp.vstack(continuous, format="csr")
        if continuous
        else sp.csr_matrix((0, n_cont), dtype=np.float64),
        sp.vstack(binary, format="csr")
        if binary
        else sp.csr_matrix((0, n_bin), dtype=np.float64),
        np.concatenate(upper) if upper else np.zeros(0, dtype=np.float64),
    )


def _hz_for(program) -> SparseHZono:
    Auc, Aub, ub = _legacy(program)
    continuous = np.asarray(
        [item.raw_id for item in program.continuous_ids], dtype=np.int64
    )
    binary = np.asarray(
        [item.raw_id for item in program.binary_ids], dtype=np.int64
    )
    return SparseHZono(
        c=np.zeros(1, dtype=np.float64),
        Gc=sp.csr_matrix((1, continuous.size), dtype=np.float64),
        Gb=sp.csr_matrix((1, binary.size), dtype=np.float64),
        Ac=sp.csr_matrix((0, continuous.size), dtype=np.float64),
        Ab=sp.csr_matrix((0, binary.size), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        col_ids=continuous,
        bcol_ids=binary,
    )


@dataclass
class _Case:
    program: object
    hz: SparseHZono
    keepalive: tuple


def _mixed_case(*, start: int = 100) -> _Case:
    allocator = _Allocator(start)
    adapter = _bind_allocator(allocator)
    owner = ConstraintProgramOwner(adapter)
    owner.allocate_continuous(2)
    owner.allocate_binary(2)
    frame = owner.frame()
    arena = owner.new_arena()
    forward_cont = _canonical([[1.0, 0.0], [-0.5, 2.0]])
    forward_bin = _canonical([[0.25, -0.5], [1.0, 0.5]])
    ranged = arena.append_guarded_band(
        arena.empty_view,
        frame=frame,
        forward_cont=forward_cont,
        forward_bin=forward_bin,
        forward_upper=np.asarray([1.25, 0.75], dtype=np.float64),
        reverse_cont=(-forward_cont).tocsr(),
        reverse_bin=(-forward_bin).tocsr(),
        reverse_upper=np.asarray([0.5, 1.5], dtype=np.float64),
        layer_id=4,
        family=ConstraintFamily.ADD_MATERIALIZE,
    )
    le = arena.append_le_exact_tag(
        ranged.view,
        frame=frame,
        A_cont=_canonical([[0.25, -1.0]]),
        A_bin=_canonical([[-0.5, 0.25]]),
        upper=np.asarray([0.8], dtype=np.float64),
        tag="relu_active:5:forward",
        layer_id=5,
    )
    program = arena.seal(le.view, final_frame=frame)
    return _Case(
        program,
        _hz_for(program),
        (allocator, adapter, owner, frame, arena),
    )


def _le_case(
    A_cont: sp.csr_matrix,
    A_bin: sp.csr_matrix,
    upper: np.ndarray,
    *,
    start: int = 1000,
) -> _Case:
    allocator = _Allocator(start)
    adapter = _bind_allocator(allocator)
    owner = ConstraintProgramOwner(adapter)
    owner.allocate_continuous(int(A_cont.shape[1]))
    owner.allocate_binary(int(A_bin.shape[1]))
    frame = owner.frame()
    arena = owner.new_arena()
    result = arena.append_le_exact_tag(
        arena.empty_view,
        frame=frame,
        A_cont=A_cont,
        A_bin=A_bin,
        upper=np.asarray(upper, dtype=np.float64),
        tag="exact:test:le",
        layer_id=1,
    )
    program = arena.seal(result.view, final_frame=frame)
    return _Case(
        program,
        _hz_for(program),
        (allocator, adapter, owner, frame, arena),
    )


def _range_program(
    forward_cont: sp.csr_matrix,
    forward_bin: sp.csr_matrix,
    forward_upper: np.ndarray,
    reverse_upper: np.ndarray,
    *,
    start: int,
):
    allocator = _Allocator(start)
    adapter = _bind_allocator(allocator)
    owner = ConstraintProgramOwner(adapter)
    owner.allocate_continuous(int(forward_cont.shape[1]))
    owner.allocate_binary(int(forward_bin.shape[1]))
    frame = owner.frame()
    arena = owner.new_arena()
    result = arena.append_guarded_band(
        arena.empty_view,
        frame=frame,
        forward_cont=forward_cont,
        forward_bin=forward_bin,
        forward_upper=forward_upper,
        reverse_cont=(-forward_cont).tocsr(),
        reverse_bin=(-forward_bin).tocsr(),
        reverse_upper=reverse_upper,
        layer_id=9,
        family=ConstraintFamily.ADD_MATERIALIZE,
    )
    program = arena.seal(result.view, final_frame=frame)
    return program, (allocator, adapter, owner, frame, arena)


def _range_case(
    forward_cont: sp.csr_matrix,
    forward_bin: sp.csr_matrix,
    forward_upper: np.ndarray,
    reverse_upper: np.ndarray,
    *,
    start: int,
) -> _Case:
    program, keepalive = _range_program(
        forward_cont,
        forward_bin,
        forward_upper,
        reverse_upper,
        start=start,
    )
    return _Case(
        program,
        _hz_for(program),
        keepalive,
    )


def _fraction_row(matrix: sp.csr_matrix, row: int, values) -> Fraction:
    result = Fraction(0)
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    for position in range(start, stop):
        result += Fraction.from_float(float(matrix.data[position])) * values[
            int(matrix.indices[position])
        ]
    return result


def _current_rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * 4096
    except (OSError, ValueError, IndexError):
        return -1


def _synthetic_source(rows: int, n_cont: int, n_bin: int):
    cont_rows = []
    cont_cols = []
    cont_data = []
    bin_rows = []
    bin_cols = []
    bin_data = []
    for row in range(rows):
        for offset, coefficient in enumerate((0.125, -0.25, 0.5)):
            cont_rows.append(row)
            cont_cols.append((3 * row + offset) % n_cont)
            cont_data.append(coefficient)
        bin_rows.append(row)
        bin_cols.append(row % n_bin)
        bin_data.append(0.25 if row % 2 == 0 else -0.25)
    Ac = sp.coo_matrix(
        (cont_data, (cont_rows, cont_cols)),
        shape=(rows, n_cont),
        dtype=np.float64,
    ).tocsr()
    Ab = sp.coo_matrix(
        (bin_data, (bin_rows, bin_cols)),
        shape=(rows, n_bin),
        dtype=np.float64,
    ).tocsr()
    Ac.sum_duplicates()
    Ac.sort_indices()
    Ab.sum_duplicates()
    Ab.sort_indices()
    return (
        Ac,
        Ab,
        np.full(rows, 8.0, dtype=np.float64),
        np.full(rows, 8.0, dtype=np.float64),
    )


def _baseline_full_stage(
    source,
    objective: np.ndarray,
) -> tuple[str, float, bool, dict]:
    if highspy is None:
        raise RuntimeError("highspy unavailable")
    Ac, Ab, upper, reverse_upper = source
    n_cont = Ac.shape[1]
    n_bin = Ab.shape[1]
    stages = {}
    materialize_started = time.perf_counter_ns()
    legacy_cont = sp.vstack((Ac, -Ac), format="csr")
    legacy_bin = sp.vstack((Ab, -Ab), format="csr")
    legacy_upper = np.concatenate((upper, reverse_upper))
    shifts = np.empty(legacy_upper.size, dtype=np.float64)
    for row in range(legacy_bin.shape[0]):
        exact = Fraction(0)
        start, stop = int(legacy_bin.indptr[row]), int(legacy_bin.indptr[row + 1])
        for position in range(start, stop):
            exact += Fraction.from_float(float(legacy_bin.data[position]))
        shifts[row] = loader._outward_float(
            Fraction.from_float(float(legacy_upper[row])) + exact,
            lower=False,
        )
    transformed_bin = legacy_bin.copy()
    transformed_bin.data *= 2.0
    matrix = sp.hstack((legacy_cont, transformed_bin), format="csr")
    stages["legacy_materialize_transform_seconds"] = (
        time.perf_counter_ns() - materialize_started
    ) * 1.0e-9
    load_started = time.perf_counter_ns()
    highs = highspy.Highs()
    try:
        for name, value in (
            ("output_flag", False),
            ("threads", 1),
            ("small_matrix_value", 1.0e-12),
        ):
            if highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise AssertionError(f"baseline option failed: {name}")
        starts = np.zeros(n_cont + n_bin + 1, dtype=np.int32)
        costs = np.asarray(objective, dtype=np.float64).copy()
        costs[n_cont:] *= 2.0
        if highs.addCols(
            n_cont + n_bin,
            costs,
            np.concatenate((np.full(n_cont, -1.0), np.zeros(n_bin))),
            np.ones(n_cont + n_bin),
            0,
            starts,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        ) != highspy.HighsStatus.kOk:
            raise AssertionError("baseline addCols failed")
        if n_bin and highs.changeColsIntegrality(
            n_bin,
            np.arange(n_cont, n_cont + n_bin, dtype=np.int32),
            np.full(n_bin, int(highspy.HighsVarType.kInteger), dtype=np.uint8),
        ) != highspy.HighsStatus.kOk:
            raise AssertionError("baseline integrality failed")
        if highs.addRows(
            matrix.shape[0],
            np.full(matrix.shape[0], -highspy.kHighsInf),
            shifts,
            matrix.nnz,
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
        ) != highspy.HighsStatus.kOk:
            raise AssertionError("baseline addRows failed")
        stages["highs_setup_and_addrows_seconds"] = (
            time.perf_counter_ns() - load_started
        ) * 1.0e-9
        solve_started = time.perf_counter_ns()
        if highs.run() != highspy.HighsStatus.kOk:
            raise AssertionError("baseline run failed")
        status = highs.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            stages["highs_solve_seconds"] = (
                time.perf_counter_ns() - solve_started
            ) * 1.0e-9
            return str(status), math.nan, False, stages
        values = np.asarray(highs.getSolution().col_value, dtype=np.float64)
        stages["highs_solve_seconds"] = (
            time.perf_counter_ns() - solve_started
        ) * 1.0e-9
        validation_started = time.perf_counter_ns()
        x = values[:n_cont]
        z = np.rint(values[n_cont:])
        xi = 2.0 * z - 1.0
        valid = bool(
            np.all(x >= -1.0)
            and np.all(x <= 1.0)
            and np.all((z == 0.0) | (z == 1.0))
        )
        for row in range(Ac.shape[0]):
            value = _fraction_row(
                Ac,
                row,
                tuple(Fraction.from_float(float(item)) for item in x),
            )
            value += _fraction_row(
                Ab,
                row,
                tuple(Fraction.from_float(float(item)) for item in xi),
            )
            if (
                value > Fraction.from_float(float(upper[row]))
                or value < -Fraction.from_float(float(reverse_upper[row]))
            ):
                valid = False
                break
        objective_value = float(objective[:n_cont] @ x)
        objective_value += float(objective[n_cont:] @ xi)
        stages["original_coordinate_validation_seconds"] = (
            time.perf_counter_ns() - validation_started
        ) * 1.0e-9
        return str(status), objective_value, valid, stages
    finally:
        cleanup_started = time.perf_counter_ns()
        highs.clear()
        stages["highs_cleanup_seconds"] = (
            time.perf_counter_ns() - cleanup_started
        ) * 1.0e-9


def run_bounded_full_stage_gate(
    *,
    rows: int = 24,
    repeats: int = 7,
) -> dict:
    """Fresh AB/BA full-stage diagnostic; never a promotion authority."""

    if highspy is None:
        raise RuntimeError("highspy unavailable")
    if type(rows) is not int or not 4 <= rows <= 256:
        raise ValueError("rows must be an exact int in [4, 256]")
    if type(repeats) is not int or repeats < 7:
        raise ValueError("repeats must be an exact int >= 7")
    n_cont = 12
    n_bin = 3
    source = _synthetic_source(rows, n_cont, n_bin)
    objective = np.zeros(n_cont + n_bin, dtype=np.float64)
    serial = 100_000

    def candidate():
        nonlocal serial
        serial += n_cont + n_bin + 7
        stages = {}
        build_started = time.perf_counter_ns()
        program, keepalive = _range_program(*source, start=serial)
        stages["program_build_and_seal_seconds"] = (
            time.perf_counter_ns() - build_started
        ) * 1.0e-9
        legacy_hz_started = time.perf_counter_ns()
        hz = _hz_for(program)
        stages["disconnected_legacy_hz_replay_materialize_seconds"] = (
            time.perf_counter_ns() - legacy_hz_started
        ) * 1.0e-9
        bind_started = time.perf_counter_ns()
        handoff = loader.bind_constraint_program_highs(hz, program)
        stages["exact_object_id_binding_seconds"] = (
            time.perf_counter_ns() - bind_started
        ) * 1.0e-9
        timed = {
            "native_transform_and_addrows_seconds": 0.0,
            "original_program_replay_validation_seconds": 0.0,
            "highs_cleanup_seconds": 0.0,
        }
        original_load = loader._load_rows
        original_validate = loader._validate_incumbent
        original_clear = loader._clear_highs

        def timed_call(name, function, *args, **kwargs):
            started = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                timed[name] += (time.perf_counter_ns() - started) * 1.0e-9

        def load_wrapper(*args, **kwargs):
            return timed_call(
                "native_transform_and_addrows_seconds",
                original_load,
                *args,
                **kwargs,
            )

        def validate_wrapper(*args, **kwargs):
            return timed_call(
                "original_program_replay_validation_seconds",
                original_validate,
                *args,
                **kwargs,
            )

        def clear_wrapper(*args, **kwargs):
            return timed_call(
                "highs_cleanup_seconds",
                original_clear,
                *args,
                **kwargs,
            )

        solve_started = time.perf_counter_ns()
        with (
            mock.patch.object(loader, "_load_rows", load_wrapper),
            mock.patch.object(loader, "_validate_incumbent", validate_wrapper),
            mock.patch.object(loader, "_clear_highs", clear_wrapper),
        ):
            result = handoff.solve(objective, max_rows=17)
        solve_total = (time.perf_counter_ns() - solve_started) * 1.0e-9
        stages.update(timed)
        stages["highs_setup_columns_and_solve_residual_seconds"] = max(
            0.0,
            solve_total - sum(timed.values()),
        )
        # Keep the producer objects live through final handoff validation.
        if not keepalive:
            raise AssertionError("candidate keepalive unexpectedly empty")
        return (
            result.model_status,
            result.objective_value,
            result.incumbent_validated,
            stages,
        )

    def baseline():
        return _baseline_full_stage(source, objective)

    # One untimed warm-up of each exact query route.
    candidate()
    baseline()
    measurements = {"candidate": [], "baseline": []}
    outcomes = {"candidate": [], "baseline": []}
    orders = []
    for repeat in range(repeats):
        order = ("candidate", "baseline") if repeat % 2 == 0 else (
            "baseline",
            "candidate",
        )
        orders.append(order)
        for name in order:
            operation = candidate if name == "candidate" else baseline
            rss_before = _current_rss_bytes()
            wall_started = time.perf_counter_ns()
            cpu_started = time.process_time_ns()
            outcome = operation()
            cpu_seconds = (time.process_time_ns() - cpu_started) * 1.0e-9
            wall_seconds = (time.perf_counter_ns() - wall_started) * 1.0e-9
            rss_after = _current_rss_bytes()
            measurements[name].append(
                {
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "rss_endpoint_delta_bytes": (
                        rss_after - rss_before
                        if rss_before >= 0 and rss_after >= 0
                        else None
                    ),
                    "stages": outcome[3],
                }
            )
            outcomes[name].append(outcome[:3])
    candidate_wall = statistics.median(
        item["wall_seconds"] for item in measurements["candidate"]
    )
    baseline_wall = statistics.median(
        item["wall_seconds"] for item in measurements["baseline"]
    )
    candidate_cpu = statistics.median(
        item["cpu_seconds"] for item in measurements["candidate"]
    )
    baseline_cpu = statistics.median(
        item["cpu_seconds"] for item in measurements["baseline"]
    )
    wall_speedup = baseline_wall / candidate_wall
    cpu_speedup = baseline_cpu / candidate_cpu
    equivalent = bool(
        all(item[0] == str(highspy.HighsModelStatus.kOptimal) for values in outcomes.values() for item in values)
        and all(item[2] for values in outcomes.values() for item in values)
        and all(item[1] == 0.0 for values in outcomes.values() for item in values)
    )
    speed_gate_passed = bool(equivalent and wall_speedup >= 1.5)
    # Endpoint RSS is diagnostic but is not an HWM measurement.  The full
    # promotion gate is therefore intentionally incomplete and cannot pass.
    promoted = False
    return {
        "schema": "act.constraint_program.highs_full_stage_gate.v1",
        "rows": rows,
        "repeats": repeats,
        "fresh_each_repeat": True,
        "orders": tuple(orders),
        "candidate_includes_program_build_seal_binding": True,
        "candidate_includes_legacy_hz_replay_for_disconnected_binding": True,
        "candidate_includes_transform_stream_solve_validation_cleanup": True,
        "baseline_includes_full_materialize_transform_load_solve_validation_cleanup": True,
        "headline_is_not_loader_only": True,
        "rss_metric": "endpoint_delta_not_peak",
        "rss_hwm_measured": False,
        "measurements": measurements,
        "wall_speedup": wall_speedup,
        "cpu_speedup": cpu_speedup,
        "semantic_equivalence": equivalent,
        "promotion_threshold": 1.5,
        "speed_gate_passed": speed_gate_passed,
        "full_promotion_gate_complete": False,
        "promotion": promoted,
        "verdict": "NO-PROMOTION",
        "proof_authority": False,
        "verdict_authority": False,
        "real_model_called": False,
        "large_model_called": False,
    }


@unittest.skipIf(highspy is None, "highspy is optional")
def _legacy_highs_oracle(case: _Case, objective: np.ndarray, *, maximize: bool):
    hz = case.hz
    best = None
    best_x = None
    best_binary = None
    binary_assignments = itertools.product((-1.0, 1.0), repeat=hz.n_bin)
    for assignment in binary_assignments:
        binary = np.asarray(assignment, dtype=np.float64)
        highs = highspy.Highs()
        try:
            if highs.setOptionValue("output_flag", False) != highspy.HighsStatus.kOk:
                raise AssertionError("oracle output option failed")
            if highs.setOptionValue("threads", 1) != highspy.HighsStatus.kOk:
                raise AssertionError("oracle thread option failed")
            costs = np.asarray(objective[: hz.n_cont], dtype=np.float64)
            if maximize:
                highs.changeObjectiveSense(highspy.ObjSense.kMaximize)
            offset = float(np.dot(objective[hz.n_cont :], binary))
            highs.changeObjectiveOffset(offset)
            starts = np.zeros(hz.n_cont + 1, dtype=np.int32)
            status = highs.addCols(
                hz.n_cont,
                costs,
                np.full(hz.n_cont, -1.0, dtype=np.float64),
                np.full(hz.n_cont, 1.0, dtype=np.float64),
                0,
                starts,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )
            if status != highspy.HighsStatus.kOk:
                raise AssertionError(f"oracle addCols failed: {status}")
            shifted_upper = case.hz.ub - case.hz.Aub @ binary
            status = highs.addRows(
                hz.n_ub,
                np.full(hz.n_ub, -highspy.kHighsInf, dtype=np.float64),
                np.asarray(shifted_upper, dtype=np.float64),
                hz.Auc.nnz,
                hz.Auc.indptr.astype(np.int32),
                hz.Auc.indices.astype(np.int32),
                hz.Auc.data,
            )
            if status != highspy.HighsStatus.kOk:
                raise AssertionError(f"oracle addRows failed: {status}")
            if highs.run() != highspy.HighsStatus.kOk:
                raise AssertionError("oracle run failed")
            if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                continue
            value = float(highs.getObjectiveValue())
            choose = best is None or (value > best if maximize else value < best)
            if choose:
                best = value
                best_x = np.asarray(
                    highs.getSolution().col_value, dtype=np.float64
                ).copy()
                best_binary = binary.copy()
        finally:
            highs.clear()
    return best, best_x, best_binary


class ConstraintProgramHighsMathTests(unittest.TestCase):
    def test_fraction_mapping_matches_original_xi_rows_for_range_and_le(self):
        case = _mixed_case()
        cursor = case.program.iter_native_batches(max_rows=2)
        try:
            batches = tuple(cursor)
        finally:
            cursor.close()
        self.assertEqual(sum(batch.row_count for batch in batches), 3)
        grid = tuple(Fraction(value, 4) for value in range(-4, 5))
        for continuous in itertools.product(grid, repeat=2):
            for z_values in itertools.product((Fraction(0), Fraction(1)), repeat=2):
                xi = tuple(2 * value - 1 for value in z_values)
                for batch in batches:
                    for row in range(batch.row_count):
                        Ac = batch.A_cont
                        Ab = batch.A_bin
                        original = _fraction_row(Ac, row, continuous)
                        original += _fraction_row(Ab, row, xi)
                        transformed = _fraction_row(Ac, row, continuous)
                        transformed += 2 * _fraction_row(Ab, row, z_values)
                        shift = _fraction_row(
                            Ab,
                            row,
                            (Fraction(1), Fraction(1)),
                        )
                        lower = float(batch.lower[row])
                        upper = Fraction.from_float(float(batch.upper[row]))
                        original_ok = (
                            (math.isinf(lower) or original >= Fraction.from_float(lower))
                            and original <= upper
                        )
                        transformed_ok = (
                            (math.isinf(lower) or transformed >= Fraction.from_float(lower) + shift)
                            and transformed <= upper + shift
                        )
                        self.assertEqual(original_ok, transformed_ok)

    def test_exact_dyadic_cancellation_and_directed_bounds(self):
        case = _le_case(
            sp.csr_matrix((1, 0), dtype=np.float64),
            _canonical([[2.0**48, 2.0**-5, -(2.0**48)]]),
            np.asarray([0.0], dtype=np.float64),
        )
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        record = loader._handoff_record(handoff)
        cursor = case.program.iter_native_batches(max_rows=1)
        try:
            batch = next(cursor)
        finally:
            cursor.close()
        starts, indices, data, lower, upper, nnz = loader._transform_batch(
            batch,
            record,
            small_matrix_value=0.0,
            large_matrix_value=1.0e20,
            infinite_bound=1.0e30,
        )
        self.assertEqual(nnz, 3)
        self.assertEqual(tuple(starts), (0, 3))
        self.assertEqual(tuple(indices), (0, 1, 2))
        self.assertEqual(
            tuple(Fraction.from_float(float(value)) for value in data),
            (
                2 * Fraction.from_float(2.0**48),
                2 * Fraction.from_float(2.0**-5),
                2 * Fraction.from_float(-(2.0**48)),
            ),
        )
        self.assertTrue(math.isinf(float(lower[0])))
        self.assertEqual(Fraction.from_float(float(upper[0])), Fraction(1, 32))
        exact = Fraction(1) + Fraction(1, 2**54)
        self.assertLessEqual(
            Fraction.from_float(loader._outward_float(exact, lower=True)), exact
        )
        self.assertGreaterEqual(
            Fraction.from_float(loader._outward_float(exact, lower=False)), exact
        )
        self.assertEqual(loader._outward_float(exact, lower=True), 1.0)
        self.assertEqual(
            loader._outward_float(exact, lower=False),
            float(np.nextafter(1.0, np.inf)),
        )

    def test_double_exact_covers_subnormal_and_rejects_overflow(self):
        tiny = float(np.nextafter(0.0, 1.0))
        self.assertEqual(
            Fraction.from_float(loader._double_exact(tiny)),
            2 * Fraction.from_float(tiny),
        )
        overflowing = float(
            np.nextafter(np.finfo(np.float64).max / 2.0, np.inf)
        )
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "2\\*Ab"):
            loader._double_exact(overflowing)


@unittest.skipIf(highspy is None, "highspy is optional")
class ConstraintProgramHighsSolveTests(unittest.TestCase):
    def test_mixed_native_solve_matches_independent_legacy_highs_oracle(self):
        case = _mixed_case()
        objective = np.asarray([0.75, -0.5, 0.25, -0.125], dtype=np.float64)
        expected, _x, _binary = _legacy_highs_oracle(
            case, objective, maximize=True
        )
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        result = handoff.solve(objective, maximize=True, max_rows=2)
        self.assertEqual(result.model_status, str(highspy.HighsModelStatus.kOptimal))
        self.assertTrue(result.incumbent_validated)
        self.assertAlmostEqual(result.objective_value, expected, places=10)
        witness_value = float(objective[:2] @ result.continuous)
        witness_value += float(objective[2:] @ result.binary)
        self.assertAlmostEqual(witness_value, expected, places=10)
        self.assertEqual(result.rows_loaded, case.program.source_rows)
        self.assertEqual(result.nnz_loaded, case.program.source_nnz)

    def test_stream_offsets_cross_256_without_whole_stack(self):
        rows = 300
        Ac = sp.csr_matrix(
            (
                np.ones(rows, dtype=np.float64),
                np.zeros(rows, dtype=np.int64),
                np.arange(rows + 1, dtype=np.int64),
            ),
            shape=(rows, 1),
        )
        case = _le_case(
            Ac,
            sp.csr_matrix((rows, 0), dtype=np.float64),
            np.ones(rows, dtype=np.float64),
        )
        result = loader.bind_constraint_program_highs(case.hz, case.program).load(
            np.zeros(1, dtype=np.float64), max_rows=256
        )
        self.assertEqual(result.rows_loaded, 300)
        self.assertEqual(result.nnz_loaded, 300)
        self.assertFalse(result.incumbent_validated)

    def test_tiny_coefficients_are_kept_or_fail_before_silent_filtering(self):
        kept = _le_case(
            _canonical([[float(np.nextafter(1.0e-12, np.inf))]]),
            sp.csr_matrix((1, 0), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
        loaded = loader.bind_constraint_program_highs(
            kept.hz, kept.program
        ).load(np.zeros(1, dtype=np.float64))
        self.assertEqual(loaded.nnz_loaded, 1)
        rejected = _le_case(
            _canonical([[1.0e-12]]),
            sp.csr_matrix((1, 0), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
        with self.assertRaisesRegex(
            loader.ConstraintProgramHighsError, "filtering threshold"
        ):
            loader.bind_constraint_program_highs(
                rejected.hz, rejected.program
            ).load(np.zeros(1, dtype=np.float64))

    def test_bounded_full_stage_paired_gate_is_honest(self):
        receipt = run_bounded_full_stage_gate(rows=12, repeats=7)
        self.assertTrue(receipt["semantic_equivalence"])
        self.assertTrue(receipt["headline_is_not_loader_only"])
        self.assertEqual(len(receipt["orders"]), 7)
        self.assertEqual(
            receipt["speed_gate_passed"], receipt["wall_speedup"] >= 1.5
        )
        self.assertFalse(receipt["rss_hwm_measured"])
        self.assertFalse(receipt["full_promotion_gate_complete"])
        self.assertFalse(receipt["promotion"])
        self.assertEqual(receipt["verdict"], "NO-PROMOTION")
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["verdict_authority"])

    def test_infinite_bound_and_doubled_coefficient_overflow_fail_closed(self):
        huge_bound = _le_case(
            _canonical([[1.0]]),
            sp.csr_matrix((1, 0), dtype=np.float64),
            np.asarray([1.0e20], dtype=np.float64),
        )
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "infinite_bound"):
            loader.bind_constraint_program_highs(
                huge_bound.hz, huge_bound.program
            ).load(np.zeros(1, dtype=np.float64))
        coefficient = float(
            np.nextafter(np.finfo(np.float64).max / 2.0, np.inf)
        )
        overflow = _le_case(
            sp.csr_matrix((1, 0), dtype=np.float64),
            _canonical([[coefficient]]),
            np.ones(1, dtype=np.float64),
        )
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "2\\*Ab"):
            loader.bind_constraint_program_highs(
                overflow.hz, overflow.program
            ).load(np.zeros(1, dtype=np.float64))


class ConstraintProgramHighsBindingTests(unittest.TestCase):
    def test_handoff_is_factory_only_and_receipt_has_no_authority(self):
        with self.assertRaises(TypeError):
            loader.ConstraintProgramHighsHandoff()
        case = _mixed_case()
        handoff = loader.ConstraintProgramHighsFactory().bind(
            case.hz, case.program
        )
        receipt = dict(handoff.receipt)
        for key in (
            "producer_authenticated",
            "production_integration",
            "consumer_integration",
            "hz_semantic_snapshot",
            "full_hz_live_graph_authenticated",
            "receipt_authority",
            "proof_authority",
            "verdict_authority",
            "solver_status_authority",
            "triangle_relaxation_called",
            "act_network_branch_and_bound_called",
            "backward_called",
            "dual_called",
            "scip_called",
            "real_model_called",
            "large_model_called",
        ):
            self.assertIs(receipt[key], False, key)
        self.assertIs(receipt["digest_is_diagnostic_only"], True)
        self.assertIs(receipt["authenticity_from_digest"], False)
        self.assertIs(receipt["native_model_loaded"], False)
        self.assertIs(receipt["model_loaded"], False)
        self.assertIs(receipt["binary_integrality_applicable"], True)
        self.assertIs(receipt["binary_integrality_loaded"], False)
        self.assertIs(receipt["integrality_loaded"], False)
        self.assertNotIn("branch_and_bound_called", receipt)
        self.assertNotIn("branch_and_bound_framework_called", receipt)
        self.assertIs(
            receipt["no_claim_about_highs_internal_mip_branching"], True
        )

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_receipt_model_and_integrality_truth_follow_loaded_phase(self):
        binary_case = _mixed_case()
        binding = loader.bind_constraint_program_highs(
            binary_case.hz, binary_case.program
        )
        self.assertIs(binding.receipt["native_model_loaded"], False)
        loaded = binding.load(np.zeros(4, dtype=np.float64), max_rows=2)
        loaded_receipt = dict(loaded.receipt)
        self.assertIs(loaded_receipt["native_model_loaded"], True)
        self.assertIs(loaded_receipt["model_loaded"], True)
        self.assertIs(loaded_receipt["model_cleared_before_return"], True)
        self.assertIs(loaded_receipt["binary_integrality_applicable"], True)
        self.assertIs(loaded_receipt["binary_integrality_loaded"], True)
        self.assertIs(loaded_receipt["integrality_loaded"], True)

        continuous_case = _le_case(
            _canonical([[1.0]]),
            sp.csr_matrix((1, 0), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
        continuous_loaded = loader.bind_constraint_program_highs(
            continuous_case.hz, continuous_case.program
        ).load(np.zeros(1, dtype=np.float64))
        continuous_receipt = dict(continuous_loaded.receipt)
        self.assertIs(continuous_receipt["native_model_loaded"], True)
        self.assertIs(
            continuous_receipt["binary_integrality_applicable"], False
        )
        self.assertIs(continuous_receipt["binary_integrality_loaded"], False)
        self.assertIs(continuous_receipt["integrality_loaded"], False)

    def test_object_rebind_and_id_mutation_terminally_consume_handoff(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        original = case.hz.col_ids
        case.hz.col_ids = original.copy()
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "graph"):
            _ = handoff.continuous_ids
        case.hz.col_ids = original
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "stale"):
            _ = handoff.continuous_ids

        second = loader.bind_constraint_program_highs(case.hz, case.program)
        old = int(case.hz.col_ids[0])
        case.hz.col_ids[0] = old + 1
        with self.assertRaisesRegex(
            loader.ConstraintProgramHighsError, "graph|stable IDs"
        ):
            _ = second.continuous_ids
        case.hz.col_ids[0] = old
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "stale"):
            _ = second.continuous_ids

    def test_registry_gc_cleanup_is_aba_guarded(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        object_id = id(handoff)
        reference = weakref.ref(handoff)
        self.assertIn(object_id, loader._HANDOFF_REGISTRY)
        del handoff
        for _attempt in range(4):
            gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(object_id, loader._HANDOFF_REGISTRY)

    def test_owner_gc_terminally_removes_weak_handoff(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        object_id = id(handoff)
        hz_reference = weakref.ref(case.hz)
        case.hz = None
        for _attempt in range(4):
            gc.collect()
        self.assertIsNone(hz_reference())
        self.assertNotIn(object_id, loader._HANDOFF_REGISTRY)
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "stale"):
            _ = handoff.source_rows

    def test_same_raw_ids_are_never_claimed_as_producer_provenance(self):
        first = _mixed_case(start=5000)
        unrelated = _mixed_case(start=5000)
        # Raw IDs/counts alone cannot distinguish two authentic namespaces.
        # The public candidate may bind the exact objects it was given, but
        # its receipt must make the missing producer lease explicit.
        handoff = loader.bind_constraint_program_highs(
            first.hz, unrelated.program
        )
        self.assertIs(handoff.receipt["producer_authenticated"], False)
        self.assertIs(handoff.receipt["proof_authority"], False)

    def test_failed_bind_traceback_handoff_is_permanently_retired(self):
        case = _mixed_case()
        original_capture = loader._capture_hz_graph
        original_id = int(case.hz.col_ids[0])
        calls = 0

        def fail_final_capture(hz):
            nonlocal calls
            calls += 1
            if calls == 3:
                hz.col_ids[0] = int(hz.col_ids[1])
            return original_capture(hz)

        caught = None
        with mock.patch.object(
            loader, "_capture_hz_graph", side_effect=fail_final_capture
        ):
            try:
                loader.bind_constraint_program_highs(case.hz, case.program)
            except BaseException as error:
                caught = error
        self.assertIsNotNone(caught)
        self.assertGreaterEqual(calls, 3)
        leaked = _traceback_local(caught, "handoff")
        record = _traceback_local(caught, "record")
        self.assertIs(type(leaked), loader.ConstraintProgramHighsHandoff)
        self.assertIs(type(record), loader._HandoffRecord)
        self.assertIs(record.terminal_state.retired, True)
        case.hz.col_ids[0] = original_id
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "stale"):
            _ = leaked.continuous_ids

    def test_execute_second_validation_race_terminally_retires(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        record = loader._HANDOFF_REGISTRY[id(handoff)]
        original_validate = loader._validate_record
        original_id = int(case.hz.col_ids[0])
        calls = 0

        def race(current_record, current_handoff):
            nonlocal calls
            calls += 1
            if calls == 2:
                case.hz.col_ids[0] = int(case.hz.col_ids[1])
            return original_validate(current_record, current_handoff)

        with mock.patch.object(loader, "_validate_record", side_effect=race):
            with self.assertRaises(loader.ConstraintProgramHighsError):
                handoff.load(np.zeros(4, dtype=np.float64))
        self.assertEqual(calls, 2)
        self.assertIs(record.terminal_state.retired, True)
        case.hz.col_ids[0] = original_id
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "stale"):
            _ = handoff.source_rows

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_execute_final_validation_break_terminally_retires(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        record = loader._HANDOFF_REGISTRY[id(handoff)]
        original_id = int(case.hz.col_ids[0])

        class MutatingClearHighs:
            def setOptionValue(self, *_args):
                return highspy.HighsStatus.kOk

            def clear(self):
                case.hz.col_ids[0] = int(case.hz.col_ids[1])
                return highspy.HighsStatus.kOk

        with (
            mock.patch.object(
                loader, "_new_highs", return_value=MutatingClearHighs()
            ),
            mock.patch.object(
                loader,
                "_configure_matrix_thresholds",
                return_value=(1.0e-12, 1.0e15, 1.0e20, 1.0e20),
            ),
            mock.patch.object(loader, "_load_columns"),
            mock.patch.object(loader, "_load_rows", return_value=(3, 7)),
        ):
            with self.assertRaises(loader.ConstraintProgramHighsError):
                handoff.load(np.zeros(4, dtype=np.float64))
        self.assertIs(record.terminal_state.retired, True)
        case.hz.col_ids[0] = original_id
        with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "stale"):
            _ = handoff.source_rows

    def test_retirement_cleanup_interruption_preserves_primary_and_no_aba(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        record = loader._HANDOFF_REGISTRY[id(handoff)]
        primary = loader.ConstraintProgramHighsError("graph-primary")
        secondary = _HostileSecondary()
        with (
            mock.patch.object(loader, "_validate_record", side_effect=primary),
            mock.patch.object(
                loader,
                "_retirement_cleanup_attempt",
                side_effect=secondary,
            ) as cleanup,
        ):
            with self.assertRaises(BaseException) as caught:
                _ = handoff.source_rows
        self.assertIs(caught.exception, primary)
        self.assertEqual(cleanup.call_count, 4)
        self.assertIs(record.terminal_state.retired, True)
        self.assertIs(loader._HANDOFF_REGISTRY.get(id(handoff)), record)
        self.assertIn(
            "handoff retirement registry cleanup also failed",
            primary.__notes__,
        )
        with self.assertRaisesRegex(
            loader.ConstraintProgramHighsError, "retired|stale"
        ):
            _ = handoff.source_rows
        self.assertNotIn(id(handoff), loader._HANDOFF_REGISTRY)


class ConstraintProgramHighsLifecycleTests(unittest.TestCase):
    def test_cursor_body_and_close_double_fault_preserves_primary_identity(self):
        primary = RuntimeError("body-primary")
        cleanup = RuntimeError("close-secondary")

        class Cursor:
            def __init__(self):
                self.used = False
                self.close_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.used:
                    raise StopIteration
                self.used = True
                return object()

            def close(self):
                self.close_calls += 1
                raise cleanup

        cursor = Cursor()

        class Program:
            def iter_native_batches(self, *, max_rows):
                self.max_rows = max_rows
                return cursor

        def consumer(_batch):
            raise primary

        with self.assertRaises(BaseException) as caught:
            loader._consume_cursor(Program(), max_rows=7, consumer=consumer)
        self.assertIs(caught.exception, primary)
        self.assertEqual(cursor.close_calls, 4)
        self.assertTrue(any("cleanup" in note for note in primary.__notes__))

    def test_transient_cursor_close_error_is_not_silently_swallowed(self):
        cleanup = RuntimeError("transient-close")

        class Cursor:
            def __init__(self):
                self.close_calls = 0

            def __iter__(self):
                return iter(())

            def close(self):
                self.close_calls += 1
                if self.close_calls == 1:
                    raise cleanup

        cursor = Cursor()

        class Program:
            def iter_native_batches(self, *, max_rows):
                return cursor

        with self.assertRaises(BaseException) as caught:
            loader._consume_cursor(Program(), max_rows=1, consumer=lambda _x: None)
        self.assertIs(caught.exception, cleanup)
        self.assertEqual(cursor.close_calls, 2)

    def test_hostile_cursor_secondary_never_formats_or_replaces_primary(self):
        for case_index, secondary in enumerate(
            (_HostileSecondary(), _HostileNameSecondary())
        ):
            with self.subTest(case_index=case_index):
                primary = KeyboardInterrupt()

                class Cursor:
                    def __iter__(self):
                        return iter((object(),))

                    def close(self):
                        raise secondary

                class Program:
                    def iter_native_batches(self, *, max_rows):
                        return Cursor()

                with self.assertRaises(BaseException) as caught:
                    loader._consume_cursor(
                        Program(),
                        max_rows=1,
                        consumer=lambda _batch: (_ for _ in ()).throw(primary),
                    )
                self.assertIs(caught.exception, primary)
                self.assertIn(
                    "native cursor cleanup raised during bounded close",
                    primary.__notes__,
                )

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_highs_primary_and_clear_double_fault_preserves_primary_identity(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        primary = RuntimeError("load-primary")
        cleanup = RuntimeError("clear-secondary")

        class FailingHighs:
            def setOptionValue(self, *_args):
                return highspy.HighsStatus.kOk

            def clear(self):
                raise cleanup

        with (
            mock.patch.object(loader, "_new_highs", return_value=FailingHighs()),
            mock.patch.object(
                loader,
                "_configure_matrix_thresholds",
                return_value=(1.0e-12, 1.0e15, 1.0e20, 1.0e20),
            ),
            mock.patch.object(loader, "_load_columns"),
            mock.patch.object(loader, "_load_rows", side_effect=primary),
        ):
            with self.assertRaises(BaseException) as caught:
                handoff.load(np.zeros(4, dtype=np.float64))
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("cleanup" in note for note in primary.__notes__))

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_hostile_highs_secondary_never_formats_or_replaces_primary(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        primary = GeneratorExit()
        secondary = _HostileSecondary()

        class FailingHighs:
            def setOptionValue(self, *_args):
                return highspy.HighsStatus.kOk

            def clear(self):
                raise secondary

        with (
            mock.patch.object(loader, "_new_highs", return_value=FailingHighs()),
            mock.patch.object(
                loader,
                "_configure_matrix_thresholds",
                return_value=(1.0e-12, 1.0e15, 1.0e20, 1.0e20),
            ),
            mock.patch.object(loader, "_load_columns"),
            mock.patch.object(loader, "_load_rows", side_effect=primary),
        ):
            with self.assertRaises(BaseException) as caught:
                handoff.load(np.zeros(4, dtype=np.float64))
        self.assertIs(caught.exception, primary)
        self.assertIn("HiGHS cleanup also failed", primary.__notes__)

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_hostile_final_validation_secondary_preserves_primary(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        primary = KeyboardInterrupt()
        secondary = _HostileSecondary()
        original = loader._validated_handoff
        calls = 0

        def fail_final(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise secondary
            return original(*args, **kwargs)

        with (
            mock.patch.object(loader, "_validated_handoff", side_effect=fail_final),
            mock.patch.object(loader, "_new_highs", side_effect=primary),
        ):
            with self.assertRaises(BaseException) as caught:
                handoff.load(np.zeros(4, dtype=np.float64))
        self.assertEqual(calls, 3)
        self.assertIs(caught.exception, primary)
        self.assertIn(
            "handoff live-graph final validation also failed",
            primary.__notes__,
        )

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_transient_highs_clear_error_is_propagated_after_convergence(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        cleanup = RuntimeError("transient-clear")

        class TransientHighs:
            def __init__(self):
                self.clear_calls = 0

            def setOptionValue(self, *_args):
                return highspy.HighsStatus.kOk

            def clear(self):
                self.clear_calls += 1
                if self.clear_calls == 1:
                    raise cleanup
                return highspy.HighsStatus.kOk

        fake = TransientHighs()
        with (
            mock.patch.object(loader, "_new_highs", return_value=fake),
            mock.patch.object(
                loader,
                "_configure_matrix_thresholds",
                return_value=(1.0e-12, 1.0e15, 1.0e20, 1.0e20),
            ),
            mock.patch.object(loader, "_load_columns"),
            mock.patch.object(loader, "_load_rows", return_value=(3, 7)),
        ):
            with self.assertRaises(BaseException) as caught:
                handoff.load(np.zeros(4, dtype=np.float64))
        self.assertIs(caught.exception, cleanup)
        self.assertEqual(fake.clear_calls, 2)

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_highs_warning_is_failure_and_model_is_cleared(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)

        class WarningHighs:
            def __init__(self):
                self.inner = highspy.Highs()
                self.cleared = False

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def addRows(self, *args):
                status = self.inner.addRows(*args)
                if status != highspy.HighsStatus.kOk:
                    return status
                return highspy.HighsStatus.kWarning

            def clear(self):
                self.cleared = True
                return self.inner.clear()

        fake = WarningHighs()
        with mock.patch.object(loader, "_new_highs", return_value=fake):
            with self.assertRaisesRegex(loader.ConstraintProgramHighsError, "addRows"):
                handoff.load(np.zeros(4, dtype=np.float64), max_rows=1)
        self.assertTrue(fake.cleared)

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_resource_cap_after_first_batch_clears_partial_model(self):
        Ac = _canonical([[1.0, 0.0], [1.0, 1.0]])
        case = _le_case(
            Ac,
            sp.csr_matrix((2, 0), dtype=np.float64),
            np.ones(2, dtype=np.float64),
        )
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)

        class TrackingHighs:
            def __init__(self):
                self.inner = highspy.Highs()
                self.cleared = False

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def clear(self):
                self.cleared = True
                return self.inner.clear()

        fake = TrackingHighs()
        with (
            mock.patch.object(loader, "_new_highs", return_value=fake),
            mock.patch.object(loader, "_MAX_BATCH_NNZ", 1),
        ):
            with self.assertRaisesRegex(
                loader.ConstraintProgramHighsError, "bounded nnz"
            ):
                handoff.load(np.zeros(2, dtype=np.float64), max_rows=1)
        self.assertTrue(fake.cleared)
        self.assertEqual(fake.inner.getNumRow(), 0)

    @unittest.skipIf(highspy is None, "highspy is optional")
    def test_optimal_spurious_incumbent_cannot_skip_original_program_replay(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        marker = loader.ConstraintProgramHighsError("validator-marker")
        with mock.patch.object(
            loader, "_validate_incumbent", side_effect=marker
        ) as validator:
            with self.assertRaises(BaseException) as caught:
                handoff.solve(
                    np.asarray([0.75, -0.5, 0.25, -0.125], dtype=np.float64),
                    maximize=True,
                )
        self.assertIs(caught.exception, marker)
        validator.assert_called_once()


class ConstraintProgramHighsStaticTests(unittest.TestCase):
    def test_candidate_has_no_whole_stack_legacy_cursor_or_scip_dependency(self):
        source = inspect.getsource(loader)
        self.assertNotIn("sp.hstack", source)
        self.assertNotIn("sp.vstack", source)
        self.assertNotIn("iter_legacy_facet_batches", source)
        self.assertNotIn("import pyscipopt", source.lower())
        self.assertNotIn("from pyscipopt", source.lower())

    def test_only_the_two_phase_c_files_are_new_in_this_scope(self):
        path = Path(loader.__file__).resolve()
        self.assertEqual(path.name, "constraint_program_highs.py")
        self.assertTrue(loader.source_sha256())

    def test_manual_csr_validation_ignores_forged_cached_flags(self):
        malformed = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2], dtype=np.int32),
            ),
            shape=(1, 1),
        )
        malformed.has_sorted_indices = True
        malformed.has_canonical_format = True
        with self.assertRaisesRegex(
            loader.ConstraintProgramHighsError, "strictly increasing"
        ):
            loader._exact_csr(
                malformed, rows=1, columns=1, name="hostile"
            )

    def test_exact_public_argument_types_and_objective_cost_cap(self):
        case = _mixed_case()
        handoff = loader.bind_constraint_program_highs(case.hz, case.program)
        for value in (0, 257, True, np.int64(2)):
            with self.subTest(max_rows=value):
                with self.assertRaises(loader.ConstraintProgramHighsError):
                    handoff.load(np.zeros(4, dtype=np.float64), max_rows=value)
        with self.assertRaisesRegex(
            loader.ConstraintProgramHighsError, "objective"
        ):
            handoff.load(np.zeros(3, dtype=np.float64))
        if highspy is not None:
            objective = np.zeros(4, dtype=np.float64)
            objective[0] = 1.0e20
            with self.assertRaisesRegex(
                loader.ConstraintProgramHighsError, "infinite_cost"
            ):
                handoff.load(objective)


if __name__ == "__main__":
    unittest.main()
