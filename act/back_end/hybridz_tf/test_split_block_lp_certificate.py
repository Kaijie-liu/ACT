from __future__ import annotations

import itertools
import time
import tracemalloc
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import solver_hz as solver_hz_module
from act.back_end.solver.solver_hz import (
    _hz_independent_lp_lagrangian_upper,
    _hz_independent_split_block_lp_lagrangian_upper,
    _hz_longdouble_to_outward_float64_upper,
)


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _fraction_solve(matrix, rhs):
    n = len(rhs)
    augmented = [
        [Fraction(value) for value in matrix[row]]
        + [Fraction(rhs[row])]
        for row in range(n)
    ]
    for column in range(n):
        pivot = next(
            (
                row
                for row in range(column, n)
                if augmented[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        scale = augmented[column][column]
        augmented[column] = [
            value / scale for value in augmented[column]
        ]
        for row in range(n):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [
                    left - scale * right
                    for left, right in zip(
                        augmented[row], augmented[column]
                    )
                ]
    return tuple(augmented[row][-1] for row in range(n))


def _exact_vertex_maximum(case) -> Fraction:
    """Exact LP maximum over all stored-float polytope vertices."""

    Auc = case["Auc_dense"]
    Aub = case["Aub_dense"]
    Ac = case["Ac_dense"]
    Ab = case["Ab_dense"]
    lower = np.concatenate(
        [case["continuous_lb"], case["binary_lb"]]
    )
    upper = np.concatenate(
        [case["continuous_ub"], case["binary_ub"]]
    )
    n_variables = int(lower.size)
    inequalities = []
    for row in range(Auc.shape[0]):
        coefficients = tuple(
            _fraction(value)
            for value in np.concatenate([Auc[row], Aub[row]])
        )
        inequalities.append((coefficients, _fraction(case["ub"][row])))
    for column in range(n_variables):
        upper_row = [Fraction(0) for _ in range(n_variables)]
        upper_row[column] = Fraction(1)
        inequalities.append((tuple(upper_row), _fraction(upper[column])))
        lower_row = [Fraction(0) for _ in range(n_variables)]
        lower_row[column] = Fraction(-1)
        inequalities.append((tuple(lower_row), -_fraction(lower[column])))
    equalities = []
    for row in range(Ac.shape[0]):
        equalities.append(
            (
                tuple(
                    _fraction(value)
                    for value in np.concatenate([Ac[row], Ab[row]])
                ),
                _fraction(case["b"][row]),
            )
        )

    row = case["C_row"]
    G = np.concatenate([case["Gc_dense"], case["Gb_dense"]], axis=1)
    q = tuple(
        sum(
            _fraction(row[output]) * _fraction(G[output, column])
            for output in range(row.size)
        )
        for column in range(n_variables)
    )
    kappa = (
        sum(
            _fraction(row[output]) * _fraction(case["c"][output])
            for output in range(row.size)
        )
        - _fraction(case["threshold"])
    )

    active_count = n_variables - len(equalities)
    best = None
    for active in itertools.combinations(inequalities, active_count):
        system = equalities + list(active)
        point = _fraction_solve(
            [coefficients for coefficients, _ in system],
            [rhs for _, rhs in system],
        )
        if point is None:
            continue
        if any(
            sum(a * x for a, x in zip(coefficients, point)) != rhs
            for coefficients, rhs in equalities
        ):
            continue
        if any(
            sum(a * x for a, x in zip(coefficients, point)) > rhs
            for coefficients, rhs in inequalities
        ):
            continue
        value = kappa + sum(a * x for a, x in zip(q, point))
        best = value if best is None else max(best, value)
    if best is None:
        raise AssertionError("test polytope unexpectedly has no vertex")
    return best


def _random_case(seed: int):
    rng = np.random.default_rng(seed)

    def quantized(shape):
        return rng.integers(-4, 5, size=shape).astype(np.float64) / 4.0

    case = {
        "c": quantized(2),
        "Gc_dense": quantized((2, 2)),
        "Gb_dense": quantized((2, 1)),
        "C_row": quantized(2),
        "threshold": float(quantized(1)[0]),
        "Auc_dense": quantized((2, 2)),
        "Aub_dense": quantized((2, 1)),
        # This equality retains a two-dimensional bounded feasible slice.
        "Ac_dense": np.asarray([[1.0, -1.0]], dtype=np.float64),
        "Ab_dense": np.asarray([[0.0]], dtype=np.float64),
        "ub": np.asarray([1.0, 1.5], dtype=np.float64),
        "b": np.asarray([0.0], dtype=np.float64),
        "continuous_lb": -np.ones(2, dtype=np.float64),
        "continuous_ub": np.ones(2, dtype=np.float64),
        "binary_lb": -np.ones(1, dtype=np.float64),
        "binary_ub": np.ones(1, dtype=np.float64),
        "upper_row_dual": -rng.uniform(0.0, 2.0, size=2),
        "equality_row_dual": rng.uniform(-2.0, 2.0, size=1),
    }
    for name in ("Gc", "Gb", "Auc", "Aub", "Ac", "Ab"):
        case[name] = sp.csr_matrix(case[f"{name}_dense"])
    return case


def _split_call(case, **updates):
    arguments = {
        key: case[key]
        for key in (
            "c",
            "Gc",
            "Gb",
            "C_row",
            "threshold",
            "Auc",
            "Aub",
            "Ac",
            "Ab",
            "ub",
            "b",
            "continuous_lb",
            "continuous_ub",
            "binary_lb",
            "binary_ub",
            "upper_row_dual",
            "equality_row_dual",
        )
    }
    arguments.update(updates)
    return _hz_independent_split_block_lp_lagrangian_upper(**arguments)


def _legacy_call(case):
    generator = sp.hstack([case["Gc"], case["Gb"]], format="csr")
    upper = sp.hstack([case["Auc"], case["Aub"]], format="csr")
    equality = sp.hstack([case["Ac"], case["Ab"]], format="csr")
    A = sp.vstack([upper, equality], format="csr")
    return _hz_independent_lp_lagrangian_upper(
        c=case["c"],
        Gc=generator,
        C_row=case["C_row"],
        threshold=case["threshold"],
        A=A,
        rl=np.concatenate(
            [np.full(case["ub"].size, -np.inf), case["b"]]
        ),
        ru=np.concatenate([case["ub"], case["b"]]),
        lb=np.concatenate(
            [case["continuous_lb"], case["binary_lb"]]
        ),
        ub=np.concatenate(
            [case["continuous_ub"], case["binary_ub"]]
        ),
        row_dual=np.concatenate(
            [case["upper_row_dual"], case["equality_row_dual"]]
        ),
    )


class TestSplitBlockLPCertificate(unittest.TestCase):
    def test_random_toys_are_outward_of_legacy_and_exact_fraction_lp(self):
        for seed in range(24):
            with self.subTest(seed=seed):
                case = _random_case(seed)
                legacy, legacy_receipt = _legacy_call(case)
                split, split_receipt = _split_call(case)
                self.assertEqual(legacy_receipt["status"], "verified_upper")
                self.assertEqual(split_receipt["status"], "verified_upper")
                self.assertGreaterEqual(split, legacy)
                split_exact = Fraction(*split.as_integer_ratio())
                self.assertGreaterEqual(split_exact, _exact_vertex_maximum(case))
                self.assertEqual(split_receipt["assembled_sparse_nnz"], 0)
                self.assertFalse(split_receipt["uses_sparse_hstack"])
                self.assertFalse(split_receipt["uses_sparse_vstack"])

    def test_longdouble_upper_materializes_outward_binary64_regression(self):
        center = 1.6511817471748444
        generator = -0.6348737903692667
        objective = 0.660172821018199
        threshold = -0.929355665068524
        case = {
            "c": np.array([center], dtype=np.float64),
            "Gc": sp.csr_matrix(
                np.array([[generator]], dtype=np.float64)
            ),
            "Gb": sp.csr_matrix((1, 0), dtype=np.float64),
            "C_row": np.array([objective], dtype=np.float64),
            "threshold": threshold,
            "Auc": sp.csr_matrix((0, 1), dtype=np.float64),
            "Aub": sp.csr_matrix((0, 0), dtype=np.float64),
            "Ac": sp.csr_matrix((0, 1), dtype=np.float64),
            "Ab": sp.csr_matrix((0, 0), dtype=np.float64),
            "ub": np.empty(0, dtype=np.float64),
            "b": np.empty(0, dtype=np.float64),
            "continuous_lb": np.array([-1.0], dtype=np.float64),
            "continuous_ub": np.array([1.0], dtype=np.float64),
            "binary_lb": np.empty(0, dtype=np.float64),
            "binary_ub": np.empty(0, dtype=np.float64),
            "upper_row_dual": np.empty(0, dtype=np.float64),
            "equality_row_dual": np.empty(0, dtype=np.float64),
        }
        exact = (
            _fraction(objective) * _fraction(center)
            - _fraction(threshold)
            + abs(_fraction(objective) * _fraction(generator))
        )
        value, receipt = _split_call(case)
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertLess(Fraction.from_float(float(value)), exact)
        outward = _hz_longdouble_to_outward_float64_upper(value)
        self.assertEqual(receipt["upper"], outward)
        self.assertGreaterEqual(Fraction.from_float(outward), exact)
        self.assertEqual(
            receipt["upper_float64_rounding"],
            "toward_positive_infinity_from_longdouble_v1",
        )

    def test_zero_equality_and_zero_binary_blocks_match_outward(self):
        case = _random_case(91)
        case.update({
            "Gb_dense": np.zeros((2, 0), dtype=np.float64),
            "Aub_dense": np.zeros((2, 0), dtype=np.float64),
            "Ac_dense": np.zeros((0, 2), dtype=np.float64),
            "Ab_dense": np.zeros((0, 0), dtype=np.float64),
            "b": np.zeros(0, dtype=np.float64),
            "binary_lb": np.zeros(0, dtype=np.float64),
            "binary_ub": np.zeros(0, dtype=np.float64),
            "equality_row_dual": np.zeros(0, dtype=np.float64),
        })
        for name in ("Gb", "Aub", "Ac", "Ab"):
            case[name] = sp.csr_matrix(case[f"{name}_dense"])
        legacy, legacy_receipt = _legacy_call(case)
        split, split_receipt = _split_call(case)
        self.assertEqual(legacy_receipt["status"], "verified_upper")
        self.assertEqual(split_receipt["status"], "verified_upper")
        self.assertGreaterEqual(split, legacy)
        self.assertGreaterEqual(
            Fraction(*split.as_integer_ratio()),
            _exact_vertex_maximum(case),
        )

    def test_dual_dust_illegal_sign_and_nonfinite_are_weakened_to_zero(self):
        case = _random_case(101)
        case["upper_row_dual"] = np.asarray(
            [np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)]
        )
        case["equality_row_dual"] = np.asarray([np.nan])
        legacy, legacy_receipt = _legacy_call(case)
        split, split_receipt = _split_call(case)
        self.assertEqual(legacy_receipt["status"], "verified_upper")
        self.assertEqual(split_receipt["status"], "verified_upper")
        self.assertEqual(split_receipt["illegal_sign_projected"], 1)
        self.assertEqual(split_receipt["nonfinite_dual_zeroed"], 1)
        self.assertEqual(split_receipt["dual_nnz"], 1)
        self.assertGreaterEqual(split, legacy)

    def test_malformed_noncanonical_nonfinite_and_deadline_fail_closed(self):
        case = _random_case(7)
        value, receipt = _split_call(
            case,
            upper_row_dual=np.zeros(3, dtype=np.float64),
        )
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("invalid:ValueError"))

        value, receipt = _split_call(
            case,
            Gc=case["Gc"].astype(np.float32),
        )
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("invalid:ValueError"))

        noncanonical = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2, 2], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        self.assertFalse(noncanonical.has_canonical_format)
        value, receipt = _split_call(case, Gc=noncanonical)
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("invalid:ValueError"))

        bad = case["Auc"].copy()
        bad.data[0] = np.inf
        value, receipt = _split_call(case, Auc=bad)
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("invalid:ValueError"))

        value, receipt = _split_call(case, deadline=time.monotonic() - 1.0)
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("deadline_exhausted:"))
        self.assertIsNone(receipt["upper"])

    def test_no_sparse_stack_is_reachable(self):
        case = _random_case(33)
        with (
            mock.patch.object(
                solver_hz_module._sp,
                "hstack",
                side_effect=AssertionError("hstack forbidden"),
            ),
            mock.patch.object(
                solver_hz_module._sp,
                "vstack",
                side_effect=AssertionError("vstack forbidden"),
            ),
        ):
            value, receipt = _split_call(case)
        self.assertIsNotNone(value)
        self.assertEqual(receipt["status"], "verified_upper")

    def test_250k_500k_1m_nnz_fixed_topology_has_bounded_peak(self):
        rows = 1000
        columns = 1000
        measured_peaks = []
        analytical_ceilings = []
        for row_width in (250, 500, 1000):
            indices = np.tile(
                np.arange(row_width, dtype=np.int32), rows
            )
            indptr = np.arange(
                0, rows * row_width + 1, row_width, dtype=np.int32
            )
            Auc = sp.csr_matrix(
                (
                    np.ones(rows * row_width, dtype=np.float64),
                    indices,
                    indptr,
                ),
                shape=(rows, columns),
            )
            case = {
                "c": np.zeros(1, dtype=np.float64),
                "Gc": sp.csr_matrix(
                    ([1.0], ([0], [0])), shape=(1, columns)
                ),
                "Gb": sp.csr_matrix((1, 0), dtype=np.float64),
                "C_row": np.ones(1, dtype=np.float64),
                "threshold": 0.0,
                "Auc": Auc,
                "Aub": sp.csr_matrix((rows, 0), dtype=np.float64),
                "Ac": sp.csr_matrix((0, columns), dtype=np.float64),
                "Ab": sp.csr_matrix((0, 0), dtype=np.float64),
                "ub": np.full(rows, 2000.0, dtype=np.float64),
                "b": np.zeros(0, dtype=np.float64),
                "continuous_lb": -np.ones(columns, dtype=np.float64),
                "continuous_ub": np.ones(columns, dtype=np.float64),
                "binary_lb": np.zeros(0, dtype=np.float64),
                "binary_ub": np.zeros(0, dtype=np.float64),
                "upper_row_dual": -np.ones(rows, dtype=np.float64),
                "equality_row_dual": np.zeros(0, dtype=np.float64),
            }
            tracemalloc.start()
            value, receipt = _split_call(case)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertEqual(receipt["status"], "verified_upper")
            self.assertIsNotNone(value)
            self.assertEqual(
                receipt["input_sparse_nnz"], rows * row_width + 1
            )
            self.assertEqual(receipt["assembled_sparse_nnz"], 0)
            self.assertLessEqual(
                peak,
                receipt["analytical_dense_workspace_bytes_ceiling"],
            )
            measured_peaks.append(peak)
            analytical_ceilings.append(
                receipt["analytical_dense_workspace_bytes_ceiling"]
            )
        # The declared dense workspace is topology-only and hence identical;
        # the observed peak grows far slower than the 4x source nnz increase.
        self.assertEqual(len(set(analytical_ceilings)), 1)
        self.assertLess(measured_peaks[-1], 2 * measured_peaks[0])


if __name__ == "__main__":
    unittest.main()
