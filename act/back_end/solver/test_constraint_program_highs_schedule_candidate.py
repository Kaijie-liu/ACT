# ===- test_constraint_program_highs_schedule_candidate.py -------------===#

from __future__ import annotations

from fractions import Fraction
import gc
import inspect
import itertools
import json
import math
import os
import subprocess
import statistics
import sys
import time
import unittest

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import constraint_program_highs_schedule_candidate as candidate

try:
    import highspy
except Exception:  # pragma: no cover
    highspy = None


def _csr(value) -> sp.csr_matrix:
    result = sp.csr_matrix(np.asarray(value, dtype=np.float64))
    result.sum_duplicates()
    result.sort_indices()
    return result


def _fraction_row(matrix: sp.csr_matrix, row: int, values) -> Fraction:
    result = Fraction(0)
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    for position in range(start, stop):
        result += Fraction.from_float(float(matrix.data[position])) * values[
            int(matrix.indices[position])
        ]
    return result


def _small_blocks():
    ranged = candidate.SourceBlock(
        "range",
        _csr([[1.0, -0.5], [0.25, 0.0]]),
        _csr([[0.25, 0.0], [0.0, -0.5]]),
        np.asarray([-0.5, -0.75], dtype=np.float64),
        np.asarray([1.25, 0.5], dtype=np.float64),
    )
    le = candidate.SourceBlock(
        "le",
        _csr([[0.5, 0.25]]),
        _csr([[0.0, 0.125]]),
        np.asarray([-np.inf], dtype=np.float64),
        np.asarray([0.75], dtype=np.float64),
    )
    return ranged, le


def _pattern_csr(rows: int, columns: int, total_nnz: int, *, seed: int) -> sp.csr_matrix:
    if not 0 <= total_nnz <= rows * columns:
        raise ValueError("invalid pattern size")
    base, extra = divmod(total_nnz, rows)
    indptr = np.empty(rows + 1, dtype=np.int32)
    indptr[0] = 0
    indices = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float64)
    cursor = 0
    coefficients = (0.125, -0.25, 0.5, -1.0)
    for row in range(rows):
        count = base + (1 if row < extra else 0)
        start = (seed + row * 17) % columns
        row_indices = np.asarray(
            sorted((start + offset * 5) % columns for offset in range(count)),
            dtype=np.int32,
        )
        # The fixture dimensions keep count < columns and step 5 coprime to 512.
        if np.unique(row_indices).size != count:
            raise AssertionError("fixture generated duplicate columns")
        stop = cursor + count
        indices[cursor:stop] = row_indices
        for local in range(count):
            data[cursor + local] = coefficients[(row + local) % len(coefficients)]
        cursor = stop
        indptr[row + 1] = cursor
    return sp.csr_matrix(
        (data, indices, indptr), shape=(rows, columns), dtype=np.float64
    )


def _c89_ratio_source(*, divisor: int = 512, mixed_rows: int = 4):
    add_rows = 40_960 // divisor
    source_rows = 57_418 // divisor
    other_rows = source_rows - add_rows
    add_nnz = 3_024_384 // divisor
    source_nnz = 6_243_172 // divisor
    other_nnz = source_nnz - add_nnz
    n_cont, n_bin = 512, 4
    if not 0 <= mixed_rows <= other_rows:
        raise ValueError("mixed_rows is outside fixture")
    add = candidate.SourceBlock(
        "range",
        _pattern_csr(add_rows, n_cont, add_nnz, seed=3),
        sp.csr_matrix((add_rows, n_bin), dtype=np.float64),
        np.full(add_rows, -1000.0, dtype=np.float64),
        np.full(add_rows, 1000.0, dtype=np.float64),
    )
    continuous_other = other_nnz - mixed_rows
    direct_other_rows = other_rows - mixed_rows
    mixed_cont_nnz = mixed_rows
    direct_cont_nnz = continuous_other - mixed_cont_nnz
    blocks = [add]
    if direct_other_rows:
        blocks.append(candidate.SourceBlock(
            "le",
            _pattern_csr(direct_other_rows, n_cont, direct_cont_nnz, seed=11),
            sp.csr_matrix((direct_other_rows, n_bin), dtype=np.float64),
            np.full(direct_other_rows, -np.inf, dtype=np.float64),
            np.full(direct_other_rows, 1000.0, dtype=np.float64),
        ))
    bin_indptr = np.arange(mixed_rows + 1, dtype=np.int32)
    bin_indices = np.empty(mixed_rows, dtype=np.int32)
    bin_data = np.empty(mixed_rows, dtype=np.float64)
    for row in range(mixed_rows):
        bin_indices[row] = row % n_bin
        bin_data[row] = 0.25 if row % 2 == 0 else -0.25
    if mixed_rows:
        blocks.append(candidate.SourceBlock(
            "le",
            _pattern_csr(mixed_rows, n_cont, mixed_cont_nnz, seed=29),
            sp.csr_matrix(
                (bin_data, bin_indices, bin_indptr),
                shape=(mixed_rows, n_bin),
                dtype=np.float64,
            ),
            np.full(mixed_rows, -np.inf, dtype=np.float64),
            np.full(mixed_rows, 1000.0, dtype=np.float64),
        ))
    metadata = {
        "divisor": divisor,
        "source_rows": source_rows,
        "virtual_rows": add_rows * 2 + other_rows,
        "source_nnz": source_nnz,
        "virtual_nnz": add_nnz * 2 + other_nnz,
        "add_rows": add_rows,
        "mixed_rows": mixed_rows,
        "n_cont": n_cont,
        "n_bin": n_bin,
        "homogeneous_block_partition": True,
    }
    return tuple(blocks), metadata


def _exact_shift(values: np.ndarray) -> Fraction:
    return sum((Fraction.from_float(float(item)) for item in values), Fraction(0))


def _outward(value: Fraction, *, lower: bool) -> float:
    nearest = float(value)
    represented = Fraction.from_float(nearest)
    if lower and represented > value:
        nearest = float(np.nextafter(nearest, -np.inf))
    elif not lower and represented < value:
        nearest = float(np.nextafter(nearest, np.inf))
    return nearest


def _legacy_solver(blocks, objective: np.ndarray, n_cont: int, n_bin: int):
    cont = []
    binary = []
    upper = []
    for block in blocks:
        cont.append(block.A_cont)
        binary.append(block.A_bin)
        upper.append(block.upper)
        if block.family == "range":
            cont.append((-block.A_cont).tocsr())
            binary.append((-block.A_bin).tocsr())
            upper.append(-block.lower)
    Ac = sp.vstack(cont, format="csr")
    Ab = sp.vstack(binary, format="csr")
    bounds = np.concatenate(upper)
    shifted = np.empty(bounds.size, dtype=np.float64)
    for row in range(Ab.shape[0]):
        start, stop = int(Ab.indptr[row]), int(Ab.indptr[row + 1])
        shifted[row] = (
            bounds[row]
            if start == stop
            else _outward(
                Fraction.from_float(float(bounds[row]))
                + _exact_shift(Ab.data[start:stop]),
                lower=False,
            )
        )
    transformed = Ab.copy()
    transformed.data *= 2.0
    matrix = sp.hstack((Ac, transformed), format="csr")
    highs = highspy.Highs()
    try:
        for name, value in (
            ("output_flag", False),
            ("threads", 1),
            ("small_matrix_value", 1.0e-12),
        ):
            if highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise AssertionError("legacy option failed")
        columns = n_cont + n_bin
        costs = np.array(objective, dtype=np.float64, copy=True)
        costs[n_cont:] *= 2.0
        offset = -float(_exact_shift(objective[n_cont:]))
        if highs.addCols(
            columns,
            costs,
            np.concatenate((np.full(n_cont, -1.0), np.zeros(n_bin))),
            np.ones(columns),
            0,
            np.zeros(columns + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        ) != highspy.HighsStatus.kOk:
            raise AssertionError("legacy addCols failed")
        if n_bin and highs.changeColsIntegrality(
            n_bin,
            np.arange(n_cont, columns, dtype=np.int32),
            np.full(n_bin, int(highspy.HighsVarType.kInteger), dtype=np.uint8),
        ) != highspy.HighsStatus.kOk:
            raise AssertionError("legacy integrality failed")
        if highs.changeObjectiveOffset(offset) != highspy.HighsStatus.kOk:
            raise AssertionError("legacy offset failed")
        if highs.addRows(
            matrix.shape[0],
            np.full(matrix.shape[0], -highspy.kHighsInf),
            shifted,
            matrix.nnz,
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
        ) != highspy.HighsStatus.kOk:
            raise AssertionError("legacy addRows failed")
        if highs.run() != highspy.HighsStatus.kOk:
            raise AssertionError("legacy solve failed")
        return str(highs.getModelStatus()), matrix.shape[0], matrix.nnz
    finally:
        highs.clear()


def run_c89_ratio_q1_stoploss(*, divisor: int = 64, mixed_rows: int = 4, repeats: int = 7):
    if highspy is None:
        raise RuntimeError("highspy unavailable")
    if type(repeats) is not int or repeats < 7:
        raise ValueError("repeats must be >=7")
    blocks, metadata = _c89_ratio_source(divisor=divisor, mixed_rows=mixed_rows)
    cont_ids = tuple(range(metadata["n_cont"]))
    bin_ids = tuple(range(10_000, 10_000 + metadata["n_bin"]))
    objective = np.zeros(metadata["n_cont"] + metadata["n_bin"], dtype=np.float64)

    def primary():
        schedule = candidate.build_solver_ready_schedule(
            blocks, continuous_ids=cont_ids, binary_ids=bin_ids
        )
        result = candidate.solve_solver_ready_schedule(schedule, objective)
        return result.model_status, result.rows_loaded, result.nnz_loaded, schedule.receipt

    def legacy():
        status, rows, nnz = _legacy_solver(
            blocks, objective, metadata["n_cont"], metadata["n_bin"]
        )
        return status, rows, nnz, None

    primary()
    legacy()
    measurements = {"primary": [], "legacy": []}
    outcomes = {"primary": [], "legacy": []}
    orders = []
    for repeat in range(repeats):
        order = ("primary", "legacy") if repeat % 2 == 0 else ("legacy", "primary")
        orders.append(order)
        for name in order:
            operation = primary if name == "primary" else legacy
            wall = time.perf_counter_ns()
            cpu = time.process_time_ns()
            outcome = operation()
            measurements[name].append(
                {
                    "wall_seconds": (time.perf_counter_ns() - wall) * 1.0e-9,
                    "cpu_seconds": (time.process_time_ns() - cpu) * 1.0e-9,
                }
            )
            outcomes[name].append(outcome)
    primary_wall = statistics.median(item["wall_seconds"] for item in measurements["primary"])
    legacy_wall = statistics.median(item["wall_seconds"] for item in measurements["legacy"])
    primary_cpu = statistics.median(item["cpu_seconds"] for item in measurements["primary"])
    legacy_cpu = statistics.median(item["cpu_seconds"] for item in measurements["legacy"])
    primary_p95 = float(np.percentile(
        [item["wall_seconds"] for item in measurements["primary"]], 95
    ))
    legacy_p95 = float(np.percentile(
        [item["wall_seconds"] for item in measurements["legacy"]], 95
    ))
    semantic = all("kOptimal" in item[0] for values in outcomes.values() for item in values)
    wall_speedup = legacy_wall / primary_wall
    return {
        "schema": "act.solver_ready_primary.c89_ratio_q1_stoploss.v1",
        **metadata,
        "orders": tuple(orders),
        "measurements": measurements,
        "semantic_equivalence": semantic,
        "primary_wall_seconds": primary_wall,
        "legacy_wall_seconds": legacy_wall,
        "primary_wall_p95_seconds": primary_p95,
        "legacy_wall_p95_seconds": legacy_p95,
        "wall_speedup": wall_speedup,
        "cpu_speedup": legacy_cpu / primary_cpu,
        "single_topology_gate_passed": semantic and wall_speedup >= 1.50,
        "single_primary_representation": True,
        "legacy_materialization_in_primary": False,
        "postsolve_full_replay": False,
        "warmup_pair_count": 1,
        "first_response_included": False,
        "rss_hwm_measured": False,
        "witness_validated": False,
        "full_promotion_gate_complete": False,
        "promotion": False,
        "real_model_called": False,
        "large_model_called": False,
    }


def _proc_hwm_bytes() -> int:
    with open("/proc/self/status", "r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    raise RuntimeError("VmHWM is unavailable")


def _fresh_process_payload(mode: str, divisor: int) -> dict:
    if mode not in {"primary", "legacy"}:
        raise ValueError("unknown fresh-process mode")
    blocks, metadata = _c89_ratio_source(divisor=divisor, mixed_rows=4)
    cont_ids = tuple(range(metadata["n_cont"]))
    bin_ids = tuple(range(10_000, 10_000 + metadata["n_bin"]))
    objective = np.zeros(metadata["n_cont"] + metadata["n_bin"], dtype=np.float64)
    gc.collect()
    entry_hwm = _proc_hwm_bytes()
    wall = time.perf_counter_ns()
    cpu = time.process_time_ns()
    if mode == "primary":
        schedule = candidate.build_solver_ready_schedule(
            blocks, continuous_ids=cont_ids, binary_ids=bin_ids
        )
        result = candidate.solve_solver_ready_schedule(schedule, objective)
        outcome = (result.model_status, result.rows_loaded, result.nnz_loaded)
    else:
        outcome = _legacy_solver(
            blocks, objective, metadata["n_cont"], metadata["n_bin"]
        )
    elapsed_wall = (time.perf_counter_ns() - wall) * 1.0e-9
    elapsed_cpu = (time.process_time_ns() - cpu) * 1.0e-9
    terminal_hwm = _proc_hwm_bytes()
    return {
        "mode": mode,
        "wall_seconds": elapsed_wall,
        "cpu_seconds": elapsed_cpu,
        "entry_hwm_bytes": entry_hwm,
        "terminal_hwm_bytes": terminal_hwm,
        "hwm_delta_bytes": max(0, terminal_hwm - entry_hwm),
        "outcome": outcome,
        **metadata,
    }


def run_c89_ratio_fresh_process_gate(*, divisor: int = 64, repeats: int = 7):
    if type(divisor) is not int or divisor < 40:
        raise ValueError("divisor must be an exact int >=40")
    if type(repeats) is not int or not 1 <= repeats <= 15:
        raise ValueError("repeats must be in [1,15]")
    samples = {"primary": [], "legacy": []}
    orders = []
    code = (
        "import json; "
        "from act.back_end.solver.test_constraint_program_highs_schedule_candidate "
        "import _fresh_process_payload; "
        "print('ACT_FRESH_JSON='+json.dumps(_fresh_process_payload(%r,%d),sort_keys=True))"
    )
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    for repeat in range(repeats):
        order = ("primary", "legacy") if repeat % 2 == 0 else ("legacy", "primary")
        orders.append(order)
        for mode in order:
            completed = subprocess.run(
                [sys.executable, "-c", code % (mode, divisor)],
                cwd=os.getcwd(),
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=True,
            )
            records = [
                line.removeprefix("ACT_FRESH_JSON=")
                for line in completed.stdout.splitlines()
                if line.startswith("ACT_FRESH_JSON=")
            ]
            if len(records) != 1:
                raise AssertionError("fresh child returned an invalid protocol")
            samples[mode].append(json.loads(records[0]))
    primary_wall = [item["wall_seconds"] for item in samples["primary"]]
    legacy_wall = [item["wall_seconds"] for item in samples["legacy"]]
    primary_median = statistics.median(primary_wall)
    legacy_median = statistics.median(legacy_wall)
    primary_p95 = float(np.percentile(primary_wall, 95))
    legacy_p95 = float(np.percentile(legacy_wall, 95))
    primary_hwm = max(item["hwm_delta_bytes"] for item in samples["primary"])
    legacy_hwm = max(item["hwm_delta_bytes"] for item in samples["legacy"])
    speedup = legacy_median / primary_median
    return {
        "schema": "act.solver_ready_primary.fresh_process_gate.v1",
        "divisor": divisor,
        "repeats": repeats,
        "orders": tuple(orders),
        "samples": samples,
        "primary_cold_wall_median_seconds": primary_median,
        "legacy_cold_wall_median_seconds": legacy_median,
        "primary_cold_wall_p95_seconds": primary_p95,
        "legacy_cold_wall_p95_seconds": legacy_p95,
        "cold_wall_speedup": speedup,
        "p95_wall_speedup": legacy_p95 / primary_p95,
        "primary_worst_hwm_delta_bytes": primary_hwm,
        "legacy_worst_hwm_delta_bytes": legacy_hwm,
        "hwm_ratio": primary_hwm / legacy_hwm if legacy_hwm else None,
        "single_topology_gate_passed": speedup >= 1.50,
        "cold_p95_reported": True,
        "rss_hwm_measured": True,
        "witness_validated": False,
        "full_promotion_gate_complete": False,
        "promotion": False,
        "real_model_called": False,
        "large_model_called": False,
    }


class SolverReadyAlgebraTests(unittest.TestCase):
    def test_fraction_point_and_jacobian_equivalence(self):
        blocks = _small_blocks()
        schedule = candidate.build_solver_ready_schedule(
            blocks, continuous_ids=(10, 11), binary_ids=(20, 21)
        )
        self.assertEqual(schedule.source_rows, 3)
        self.assertEqual(schedule.continuous_ids, (10, 11))
        self.assertEqual(schedule.binary_ids, (20, 21))
        rows = []
        for segment in schedule.segments:
            data, indices, indptr, lower, upper = segment.arrays()
            matrix = sp.csr_matrix(
                (data, indices, indptr),
                shape=(segment.rows, segment.columns),
            )
            rows.extend((matrix, lower, upper, row) for row in range(segment.rows))
        grid = tuple(Fraction(value, 2) for value in range(-2, 3))
        source_rows = []
        for block in blocks:
            source_rows.extend((block, row) for row in range(block.A_cont.shape[0]))
        self.assertEqual(len(rows), len(source_rows))
        for x in itertools.product(grid, repeat=2):
            for z in itertools.product((Fraction(0), Fraction(1)), repeat=2):
                xi = tuple(2 * value - 1 for value in z)
                for (matrix, lower, upper, row), (block, source_row) in zip(rows, source_rows):
                    original = _fraction_row(block.A_cont, source_row, x)
                    original += _fraction_row(block.A_bin, source_row, xi)
                    transformed = _fraction_row(matrix, row, x + z)
                    original_ok = (
                        (math.isinf(float(block.lower[source_row])) or original >= Fraction.from_float(float(block.lower[source_row])))
                        and original <= Fraction.from_float(float(block.upper[source_row]))
                    )
                    transformed_ok = (
                        (math.isinf(float(lower[row])) or transformed >= Fraction.from_float(float(lower[row])))
                        and transformed <= Fraction.from_float(float(upper[row]))
                    )
                    self.assertEqual(original_ok, transformed_ok)
                # Jacobian coefficients are exact under xi=2z-1.
                self.assertEqual(rows[0][0][0, 0], blocks[0].A_cont[0, 0])
                self.assertEqual(rows[0][0][0, 2], 2.0 * blocks[0].A_bin[0, 0])

    def test_direct_block_and_input_snapshot(self):
        Ac = _csr([[1.0, -0.5], [0.25, 0.125]])
        Ab = sp.csr_matrix((2, 1), dtype=np.float64)
        lower = np.asarray([-1.0, -2.0], dtype=np.float64)
        upper = np.asarray([1.0, 2.0], dtype=np.float64)
        block = candidate.SourceBlock("range", Ac, Ab, lower, upper)
        schedule = candidate.build_solver_ready_schedule(
            (block,), continuous_ids=(1, 2), binary_ids=(3,)
        )
        Ac.data[:] = 99.0
        lower[:] = 99.0
        segment = schedule.segments[0]
        data, _, _, saved_lower, _ = segment.arrays()
        self.assertTrue(segment.direct_binary_free)
        np.testing.assert_array_equal(data, np.asarray([1.0, -0.5, 0.25, 0.125]))
        np.testing.assert_array_equal(saved_lower, np.asarray([-1.0, -2.0]))
        self.assertEqual(schedule.direct_rows, 2)
        self.assertFalse(schedule.receipt["legacy_representation_retained"])

    def test_rare_mixed_fanin_and_threshold_reject_without_fallback(self):
        block = candidate.SourceBlock(
            "range",
            _csr([[1.0]]),
            _csr([[0.25, -0.5, 0.125]]),
            np.asarray([-1.0]),
            np.asarray([1.0]),
        )
        with self.assertRaisesRegex(candidate.SolverReadyUnsupported, "fanin"):
            candidate.build_solver_ready_schedule(
                (block,), continuous_ids=(1,), binary_ids=(2, 3, 4)
            )
        tiny = candidate.SourceBlock(
            "le",
            _csr([[1.0e-13]]),
            sp.csr_matrix((1, 0), dtype=np.float64),
            np.asarray([-np.inf]),
            np.asarray([1.0]),
        )
        with self.assertRaisesRegex(candidate.SolverReadyUnsupported, "coefficient"):
            candidate.build_solver_ready_schedule(
                (tiny,), continuous_ids=(1,), binary_ids=()
            )


@unittest.skipIf(highspy is None, "highspy optional")
class SolverReadyBackendTests(unittest.TestCase):
    def test_mixed_schedule_loads_exact_row_and_keeps_no_authority(self):
        schedule = candidate.build_solver_ready_schedule(
            _small_blocks(), continuous_ids=(10, 11), binary_ids=(20, 21)
        )
        result = candidate.solve_solver_ready_schedule(
            schedule, np.zeros(4, dtype=np.float64)
        )
        self.assertIn("kOptimal", result.model_status)
        self.assertEqual(result.rows_loaded, 3)
        self.assertFalse(result.receipt["proof_authority"])
        self.assertFalse(result.receipt["witness_validated"])
        self.assertFalse(result.receipt["promotion"])

    def test_c89_ratio_structure_and_q1_stoploss(self):
        result = run_c89_ratio_q1_stoploss(divisor=64, mixed_rows=4, repeats=15)
        self.assertTrue(result["semantic_equivalence"])
        self.assertEqual(result["source_rows"], 897)
        self.assertEqual(result["virtual_rows"], 1_537)
        self.assertEqual(result["source_nnz"], 97_549)
        self.assertEqual(result["virtual_nnz"], 144_805)
        self.assertTrue(result["single_primary_representation"])
        self.assertTrue(result["homogeneous_block_partition"])
        self.assertTrue(result["single_topology_gate_passed"])
        self.assertFalse(result["legacy_materialization_in_primary"])
        self.assertFalse(result["postsolve_full_replay"])
        self.assertFalse(result["first_response_included"])
        self.assertFalse(result["promotion"])

    def test_fresh_process_protocol_reports_cold_p95_and_hwm(self):
        result = run_c89_ratio_fresh_process_gate(divisor=512, repeats=1)
        self.assertEqual(len(result["samples"]["primary"]), 1)
        self.assertEqual(len(result["samples"]["legacy"]), 1)
        self.assertTrue(result["cold_p95_reported"])
        self.assertTrue(result["rss_hwm_measured"])
        self.assertFalse(result["witness_validated"])
        self.assertFalse(result["full_promotion_gate_complete"])
        self.assertFalse(result["promotion"])


class SolverReadyStaticTests(unittest.TestCase):
    def test_hot_module_has_no_core_fraction_legacy_or_fallback_stack(self):
        source = inspect.getsource(candidate)
        self.assertNotIn("fractions", source)
        self.assertNotIn("constraint_program import", source)
        self.assertNotIn("operator_hz", source)
        self.assertNotIn("solver_hz", source)
        self.assertNotIn("sp.hstack", source)
        self.assertNotIn("sp.vstack", source)
        self.assertNotIn("iter_legacy", source)
        self.assertNotIn("triangle", source.lower().replace("triangle_relaxation_called", ""))
        self.assertEqual(candidate.SolverReadySchedule.receipt.fget.__name__, "receipt")


if __name__ == "__main__":
    unittest.main()
