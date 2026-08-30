#!/usr/bin/env python3
"""Proof-firewall tests for grouped alternative property upper planes."""

from __future__ import annotations

from fractions import Fraction
import random
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import (
    SparseHZono,
    _augment_safe_groups_with_dyadic_mixtures,
    _dyadic_pair_cube_candidate,
    hz_mark_constructively_nonempty,
    hz_objbound_decide,
)


def _state(
    center: list[float],
    generators: list[list[float]],
    *,
    equality: tuple[list[float], float] | None = None,
) -> SparseHZono:
    n_out = len(center)
    n_cont = len(generators[0]) if generators else 0
    if equality is None:
        Ac = sp.csr_matrix((0, n_cont), dtype=np.float64)
        b = np.zeros(0, dtype=np.float64)
    else:
        Ac = sp.csr_matrix([equality[0]], dtype=np.float64)
        b = np.asarray([equality[1]], dtype=np.float64)
    hz = SparseHZono(
        c=np.asarray(center, dtype=np.float64),
        Gc=sp.csr_matrix(generators, dtype=np.float64).reshape(
            (n_out, n_cont)
        ),
        Gb=sp.csr_matrix((n_out, 0), dtype=np.float64),
        Ac=Ac,
        Ab=sp.csr_matrix((Ac.shape[0], 0), dtype=np.float64),
        b=b,
    )
    hz_mark_constructively_nonempty(hz, reason="grouped_upper_test")
    return hz


class PropertyUpperGroupTests(unittest.TestCase):
    def test_any_cube_certified_plane_resolves_group(self) -> None:
        hz = _state([-1.0, 1.0], [[], []])
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            safe_row_groups=((0, 1),),
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        self.assertEqual(
            hz._solver_objbound_stats["safe_row_groups_resolved"], 1
        )

    def test_every_property_group_requires_a_certified_plane(self) -> None:
        hz = _state(
            [-1.0, 1.0, 0.5],
            [[], [], []],
        )
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            safe_row_groups=((0, 1), (2,)),
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

    def test_lp_can_use_baseline_when_cube_better_candidate_regresses(self) -> None:
        # xi is fixed at -1.  Plane 0 (baseline) is .5*xi+.5 and therefore
        # has constrained value 0, while plane 1 (candidate) is constant .5.
        # Candidate has the better free cube upper (.5 vs 1), yet baseline is
        # the only plane proving the threshold .25.  Grouped support must keep
        # both and certify via plane 0.
        hz = _state(
            [0.5, 0.5],
            [[0.5], [0.0]],
            equality=([1.0], -1.0),
        )
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.full(2, 0.25, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
            safe_row_groups=((0, 1),),
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        self.assertIn(
            0, hz._solver_objbound_stats["lp_certified_row_ids"]
        )

    def test_invalid_or_unsafe_group_semantics_fail_closed(self) -> None:
        hz = _state([-1.0, -1.0], [[], []])
        for groups in (((0,),), ((0, 0), (1,)), ((0,), (2,))):
            verdict, witness = hz_objbound_decide(
                hz,
                np.eye(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=1.0,
                safe_row_groups=groups,
            )
            self.assertEqual(verdict, "UNKNOWN")
            self.assertIsNone(witness)
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=True,
            time_limit=1.0,
            safe_row_groups=((0,), (1,)),
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            safe_row_groups=((0,), (1,)),
            expected_safe_group_count=1,
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            require_base_feasible=False,
            safe_row_groups=((0,), (1,)),
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

    def test_malformed_or_nonfinite_objectives_fail_closed(self) -> None:
        hz = _state([-1.0], [[]])
        for C, thresholds in (
            (np.eye(1), np.asarray([0.0, 1.0])),
            (np.asarray([[np.nan]]), np.asarray([0.0])),
            (np.eye(1), np.asarray([np.inf])),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                C,
                thresholds,
                is_unsafe_linear=False,
                time_limit=1.0,
                safe_row_groups=((0,),),
            )
            self.assertEqual(verdict, "UNKNOWN")
            self.assertIsNone(witness)

    def test_lp_skips_sibling_after_group_is_certified(self) -> None:
        hz = _state(
            [0.5, 0.5],
            [[0.5], [0.4]],
            equality=([1.0], -1.0),
        )
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.full(2, 0.25, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
            safe_row_groups=((0, 1),),
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        self.assertEqual(
            hz._solver_objbound_stats["lp_group_redundant_rows"], 1
        )

    def test_nonverified_negative_checker_value_has_no_authority(self) -> None:
        hz = _state(
            [0.5],
            [[0.5]],
            equality=([1.0], -1.0),
        )
        with mock.patch(
            "act.back_end.solver.solver_hz."
            "_hz_independent_lp_lagrangian_upper",
            return_value=(
                np.longdouble(-1.0),
                {"status": "candidate_only", "dual_nnz": 1},
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.eye(1, dtype=np.float64),
                np.asarray([0.25], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                lp_prefilter_fraction=1.0,
                lp_prefilter_max_seconds=1.0,
                safe_row_groups=((0,),),
            )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

    def test_group_cutoff_is_strict(self) -> None:
        tol = 1.0e-7
        at_cutoff = _state([-tol], [[]])
        verdict, witness = hz_objbound_decide(
            at_cutoff,
            np.eye(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            tol=tol,
            safe_row_groups=((0,),),
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

        strictly_below = _state(
            [float(np.nextafter(-tol, -np.inf))], [[]]
        )
        verdict, witness = hz_objbound_decide(
            strictly_below,
            np.eye(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            tol=tol,
            safe_row_groups=((0,),),
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)

    def test_validated_plane_witness_is_suppressed(self) -> None:
        hz = _state([0.0], [[1.0]])
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
            safe_row_groups=((0,),),
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        stats = hz._solver_objbound_stats
        self.assertEqual(stats["lp_validated_witness_rows"], 1)
        self.assertTrue(stats["safe_row_group_witness_suppressed"])

    def test_dyadic_group_mixture_certifies_complementary_planes(self) -> None:
        # p0=x-.2 and p1=-x-.2 each have free-cube supremum .8, but their
        # exact half/half convex combination is the constant -.2.
        hz = _state([-0.2, -0.2], [[1.0], [-1.0]])
        off_verdict, off_witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=0.0,
            lp_prefilter_max_seconds=0.0,
            safe_row_groups=((0, 1),),
            safe_group_mixture_grid_bits=0,
        )
        self.assertEqual(off_verdict, "UNKNOWN")
        self.assertIsNone(off_witness)

        hz = _state([-0.2, -0.2], [[1.0], [-1.0]])
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=0.0,
            lp_prefilter_max_seconds=0.0,
            safe_row_groups=((0, 1),),
            expected_safe_group_count=1,
            safe_group_mixture_grid_bits=1,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        receipt = hz._solver_objbound_stats[
            "safe_row_dyadic_mixture"
        ]
        self.assertEqual(receipt["status"], "generated")
        self.assertEqual(receipt["selected_groups"], 1)
        self.assertEqual(receipt["denominator"], 2)
        self.assertEqual(receipt["selected"][0]["numerator"], 1)
        self.assertEqual(receipt["selected"][0]["appended_row"], 2)
        selected = receipt["selected"][0]
        self.assertEqual(
            Fraction.from_float(selected["left_weight"]),
            Fraction(1, 2),
        )
        self.assertEqual(
            Fraction.from_float(selected["right_weight"]),
            Fraction(1, 2),
        )
        self.assertTrue(selected["stored_dyadic_weights_validated"])
        self.assertTrue(receipt["stored_dyadic_weights_validated"])
        self.assertTrue(selected["discrete_bracket_validated"])
        self.assertTrue(
            selected["exact_stored_float_grid_argmin_validated"]
        )
        self.assertEqual(
            len(selected["exact_objective_sha256"]), 64
        )
        self.assertNotIn("_proxy_exact_numerator", selected)
        self.assertGreater(selected["guarded_cube_gain"], 0.0)
        self.assertLess(selected["guarded_mixed_cube_upper"], 0.0)
        self.assertEqual(
            receipt["guarded_all_group_best_cube_upper_count"], 1
        )
        self.assertEqual(receipt["guarded_cube_gain_count"], 1)
        self.assertAlmostEqual(
            receipt["guarded_cube_gain_sum"],
            selected["guarded_cube_gain"],
        )
        for digest_name in (
            "weights_sha256",
            "guarded_all_group_best_cube_upper_sha256",
            "guarded_cube_gain_sha256",
            "guarded_selected_records_sha256",
        ):
            self.assertEqual(len(receipt[digest_name]), 64)
        self.assertEqual(
            receipt["guarded_cube_authority"],
            "outward_hz_cube_checker",
        )
        self.assertEqual(receipt["guarded_cube_certified_groups"], 1)
        self.assertTrue(receipt["dyadic_convexity_validated"])
        self.assertFalse(receipt["proof_authority"])
        self.assertEqual(
            hz._solver_objbound_stats["safe_row_group_winners"][0][
                "row"
            ],
            2,
        )

    def test_dyadic_group_mixture_contract_fails_closed(self) -> None:
        hz = _state([-0.2, -0.2], [[1.0], [-1.0]])
        for kwargs in (
            {"safe_group_mixture_grid_bits": True},
            {"safe_group_mixture_grid_bits": 25},
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.eye(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=1.0,
                safe_row_groups=((0, 1),),
                **kwargs,
            )
            self.assertEqual(verdict, "UNKNOWN")
            self.assertIsNone(witness)

        verdict, witness = hz_objbound_decide(
            hz,
            np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            safe_row_groups=((0, 1),),
            safe_group_mixture_grid_bits=1,
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)

    def test_dyadic_pair_candidate_matches_exhaustive_grid(self) -> None:
        rng = random.Random(103991)
        denominator = 16
        for _case in range(64):
            center0 = rng.randint(-8, 8) / 8.0
            center1 = rng.randint(-8, 8) / 8.0
            row0 = np.asarray(
                [[rng.randint(-8, 8) / 8.0 for _ in range(5)]],
                dtype=np.float64,
            )
            row1 = np.asarray(
                [[rng.randint(-8, 8) / 8.0 for _ in range(5)]],
                dtype=np.float64,
            )
            candidate = _dyadic_pair_cube_candidate(
                center0,
                sp.csr_matrix(row0),
                center1,
                sp.csr_matrix(row1),
                denominator=denominator,
            )
            exhaustive = []
            for numerator in range(denominator + 1):
                weight0 = Fraction(numerator, denominator)
                weight1 = 1 - weight0
                exact = (
                    weight0 * Fraction.from_float(center0)
                    + weight1 * Fraction.from_float(center1)
                    + sum(
                        abs(
                            weight0 * Fraction.from_float(row0[0, col])
                            + weight1 * Fraction.from_float(row1[0, col])
                        )
                        for col in range(row0.shape[1])
                    )
                )
                exhaustive.append(float(exact))
            endpoint = min(exhaustive[0], exhaustive[-1])
            best_interior = min(exhaustive[1:-1])
            materially_better = best_interior < endpoint - (
                64.0
                * np.finfo(np.float64).eps
                * (1.0 + abs(endpoint))
            )
            self.assertEqual(candidate is not None, materially_better)
            if candidate is not None:
                numerator = int(candidate["numerator"])
                self.assertEqual(
                    exhaustive[numerator],
                    best_interior,
                )
                self.assertEqual(
                    Fraction(numerator, denominator)
                    + Fraction(denominator - numerator, denominator),
                    1,
                )
                self.assertTrue(candidate["discrete_bracket_validated"])

    def test_dyadic_pair_adversarial_grids_match_fraction_oracle(
        self,
    ) -> None:
        tiny = float(np.nextafter(0.0, 1.0))
        # Flat minimizers, quantized roots immediately around a grid point,
        # large cancellation/dynamic range, repeated roots, and subnormal
        # coefficients are all checked against exact stored-float Fractions.
        cases = (
            (
                0.0,
                0.0,
                [0.75, 0.25],
                [-0.25, -0.75],
                16,
            ),
            (-0.25, -0.25, [16.0], [-1.0], 16),
            (-0.25, -0.25, [14.0], [-1.0], 16),
            (
                -0.5,
                0.25,
                [2.0**40, -(2.0**-20), 0.0, 3.0],
                [-(2.0**36), 2.0**-20, 4.0, -1.0],
                32,
            ),
            (
                0.0,
                0.0,
                [8.0, -8.0, 1.0, -1.0],
                [-8.0, 8.0, -1.0, 1.0],
                64,
            ),
            (
                -0.25,
                -0.25,
                [1.0, tiny, -tiny],
                [-1.0, -tiny, tiny],
                16,
            ),
        )
        for center0, center1, raw0, raw1, denominator in cases:
            row0 = np.asarray([raw0], dtype=np.float64)
            row1 = np.asarray([raw1], dtype=np.float64)
            candidate = _dyadic_pair_cube_candidate(
                center0,
                sp.csr_matrix(row0),
                center1,
                sp.csr_matrix(row1),
                denominator=denominator,
            )
            self.assertIsNotNone(candidate)
            exact_grid = []
            for numerator in range(denominator + 1):
                weight0 = Fraction(numerator, denominator)
                weight1 = 1 - weight0
                exact_grid.append(
                    weight0 * Fraction.from_float(center0)
                    + weight1 * Fraction.from_float(center1)
                    + sum(
                        abs(
                            weight0 * Fraction.from_float(row0[0, col])
                            + weight1
                            * Fraction.from_float(row1[0, col])
                        )
                        for col in range(row0.shape[1])
                    )
                )
            exact_best = min(exact_grid)
            exact_argmins = {
                index
                for index, value in enumerate(exact_grid)
                if value == exact_best
            }
            self.assertIn(
                int(candidate["numerator"]), exact_argmins
            )
            self.assertTrue(candidate["discrete_bracket_validated"])
            self.assertEqual(
                candidate["candidate_arithmetic"],
                "python_int_exact_binary64_dyadic",
            )
            self.assertTrue(
                candidate[
                    "exact_stored_float_grid_argmin_validated"
                ]
            )

    def test_dyadic_pair_search_scales_with_sparse_support(self) -> None:
        # A ten-million-column shape with two stored entries catches accidental
        # densification while staying essentially free in memory and runtime.
        width = 10_000_003
        columns = np.asarray([7, width - 1], dtype=np.int64)
        rows = np.zeros(columns.size, dtype=np.int64)
        row0 = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                (rows, columns),
            ),
            shape=(1, width),
        )
        row1 = sp.csr_matrix(
            (
                np.asarray([-1.0, -2.0], dtype=np.float64),
                (rows, columns),
            ),
            shape=(1, width),
        )
        candidate = _dyadic_pair_cube_candidate(
            -0.25,
            row0,
            -0.25,
            row1,
            denominator=16,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["numerator"], 8)
        self.assertEqual(candidate["aligned_generator_nnz"], 2)
        self.assertIsNone(
            _dyadic_pair_cube_candidate(
                -0.25,
                row0,
                -0.25,
                row1,
                denominator=16,
                max_union_terms=1,
            )
        )
        with self.assertRaises(ValueError):
            _dyadic_pair_cube_candidate(
                0.0,
                row0,
                0.0,
                row1,
                denominator=3,
            )

    def test_dyadic_group_exact_search_cap_appends_nothing(self) -> None:
        center = np.asarray([-0.2, -0.2], dtype=np.float64)
        generators = sp.csr_matrix(
            [[1.0], [-1.0]], dtype=np.float64
        )
        objectives = np.eye(2, dtype=np.float64)
        thresholds = np.zeros(2, dtype=np.float64)
        (
            returned_objectives,
            returned_thresholds,
            returned_groups,
            receipt,
        ) = _augment_safe_groups_with_dyadic_mixtures(
            center,
            generators,
            sp.csr_matrix((2, 0), dtype=np.float64),
            objectives,
            thresholds,
            ((0, 1),),
            grid_bits=1,
            exact_total_term_cap=1,
            exact_pair_term_cap=8,
        )
        self.assertEqual(
            receipt["status"],
            "exact_search_budget_exceeded_no_append",
        )
        self.assertEqual(
            receipt["exact_search_budget_reason"],
            "total_term_cap",
        )
        self.assertEqual(receipt["selected_groups"], 0)
        self.assertEqual(receipt["appended_rows"], 0)
        self.assertFalse(receipt["exact_search_complete"])
        np.testing.assert_array_equal(
            returned_objectives, objectives
        )
        np.testing.assert_array_equal(
            returned_thresholds, thresholds
        )
        self.assertEqual(returned_groups, ((0, 1),))

        *_, deadline_receipt = (
            _augment_safe_groups_with_dyadic_mixtures(
                center,
                generators,
                sp.csr_matrix((2, 0), dtype=np.float64),
                objectives,
                thresholds,
                ((0, 1),),
                grid_bits=1,
                candidate_deadline=0.0,
            )
        )
        self.assertEqual(
            deadline_receipt["status"],
            "exact_search_budget_exceeded_no_append",
        )
        self.assertEqual(
            deadline_receipt["exact_search_budget_reason"],
            "candidate_deadline",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
