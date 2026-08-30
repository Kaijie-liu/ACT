#!/usr/bin/env python3
# ===- test_property_pairhull_candidates.py - sparse batch toys --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Soundness, tightness, sparsity, and stop-loss tests for PairHull batches."""

from __future__ import annotations

import copy
from fractions import Fraction
import math
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.property_pairhull_candidates import (
    build_property_pairhull_candidates,
    finalize_property_pairhull_candidates_receipt,
    verify_pairhull_candidate_receipt,
    verify_property_pairhull_candidates_receipt,
)


F = Fraction


def _parse_fraction(value):
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return F(int(numerator), int(denominator))
    return F(int(value))


def _outward(value):
    stored = float(value)
    if F.from_float(stored) < value:
        stored = math.nextafter(stored, math.inf)
    return stored


def _dyadic_decisive(*, intercept=1.0, error=(0.0, 0.0)):
    # z0=t, z1=t+1/4+e/16.  The diagonal support proves z0-z1<=-3/16.
    return build_property_pairhull_candidates(
        property_matrix=np.asarray([[1.0, -1.0]]),
        output_weight=np.eye(2, dtype=np.float64),
        preactivation_center=np.asarray([0.0, 0.25]),
        preactivation_generators=sp.csr_matrix(
            [[1.0, 0.0], [1.0, 1.0 / 16.0]]
        ),
        preactivation_error=np.asarray(error, dtype=np.float64),
        lower=np.asarray([-1.0 - error[0], -13.0 / 16.0 - error[1]]),
        upper=np.asarray([1.0 + error[0], 21.0 / 16.0 + error[1]]),
        foundation_planes=np.asarray([[0.0, 0.0]]),
        foundation_intercepts=np.asarray([intercept]),
        foundation_names=("baseline",),
        pair_budget=1,
        time_limit=5.0,
    )


class PropertyPairHullCandidateSoundnessTests(unittest.TestCase):
    def test_dyadic_decisive_candidate_is_exact_zero(self):
        result = _dyadic_decisive()
        self.assertEqual(result.status, "generated")
        np.testing.assert_array_equal(result.rival_ids, [0])
        np.testing.assert_array_equal(result.foundation_indices, [0])
        np.testing.assert_array_equal(result.pair_indices, [[0, 1]])
        np.testing.assert_array_equal(result.planes, [[0.0, 0.0]])
        np.testing.assert_array_equal(result.intercepts, [0.0])

        receipt = result.receipt
        self.assertTrue(receipt["whole_batch_complete"])
        self.assertTrue(receipt["foundation_rows_must_remain_retained"])
        self.assertTrue(receipt["foundation_rows_retained_by_caller"])
        self.assertFalse(receipt["pair_selector_proof_authority"])
        self.assertEqual(receipt["global_pair_count"], 1)
        self.assertTrue(receipt["global_pairs_disjoint"])
        self.assertEqual(receipt["float_proposals"], 1)
        self.assertEqual(receipt["exact_beta_evaluations"], 1)
        record = receipt["candidate_records"][0]
        self.assertEqual(record["q_exact"], ["1", "-1"])
        self.assertEqual(record["singleton_requirements_exact"], ["1", "0"])
        self.assertEqual(record["pair_beta_exact"], "0")
        self.assertTrue(record["outward_intercept_validated"])
        self.assertTrue(
            verify_property_pairhull_candidates_receipt(receipt)
        )

    def test_foundation_outward_slack_is_retained_not_reconstructed_away(self):
        result = _dyadic_decisive(intercept=1.125)
        self.assertEqual(result.status, "generated")
        self.assertEqual(result.intercepts[0], 0.125)
        record = result.receipt["candidate_records"][0]
        self.assertEqual(
            _parse_fraction(record["reconstructed_intercept_exact"]),
            F(1, 8),
        )
        self.assertEqual(
            _parse_fraction(record["stored_intercept_reduction_exact"]),
            F(1),
        )

    def test_rowwise_error_is_independent_and_prevents_false_zero(self):
        result = build_property_pairhull_candidates(
            property_matrix=[[1.0, -1.0]],
            output_weight=np.eye(2),
            preactivation_center=[0.0, 0.0],
            preactivation_generators=sp.csr_matrix([[1.0], [1.0]]),
            preactivation_error=[0.25, 0.125],
            lower=[-1.25, -1.125],
            upper=[1.25, 1.125],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[1.25],
            pair_budget=1,
            time_limit=5.0,
        )
        self.assertEqual(result.status, "generated")
        self.assertEqual(result.intercepts[0], 0.375)
        record = result.receipt["candidate_records"][0]
        self.assertEqual(record["pair_beta_exact"], "3/8")
        projection = result.receipt["selected_projections"][0]
        self.assertEqual(
            projection["rowwise_error_hex"],
            [float(0.25).hex(), float(0.125).hex()],
        )

    def test_exact_stored_binary64_q_matches_independent_fraction_oracle(self):
        C = np.asarray([[0.1, -0.2, 0.3]], dtype=np.float64)
        weight = np.asarray(
            [
                [0.4, -0.7],
                [0.5, 0.8],
                [-0.6, 0.9],
            ],
            dtype=np.float64,
        )
        exact_q = []
        for neuron in range(2):
            exact_q.append(
                sum(
                    (
                        F.from_float(float(C[0, output]))
                        * F.from_float(float(weight[output, neuron]))
                        for output in range(3)
                    ),
                    F(0),
                )
            )
        self.assertLess(exact_q[0], 0)
        self.assertGreater(exact_q[1], 0)
        singleton = exact_q[1]
        result = build_property_pairhull_candidates(
            property_matrix=C,
            output_weight=weight,
            preactivation_center=[0.0, 0.0],
            preactivation_generators=sp.csr_matrix([[1.0], [1.0]]),
            preactivation_error=[0.0, 0.0],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[_outward(singleton)],
            pair_budget=1,
            time_limit=5.0,
        )
        self.assertEqual(result.status, "generated")
        record = result.receipt["candidate_records"][0]
        self.assertEqual(
            [_parse_fraction(value) for value in record["q_exact"]],
            exact_q,
        )
        self.assertEqual(record["pair_beta_exact"], "0")
        self.assertGreaterEqual(
            F.from_float(float(result.intercepts[0])),
            _parse_fraction(record["reconstructed_intercept_exact"]),
        )


class PropertyPairHullCandidateBoundedWorkTests(unittest.TestCase):
    def test_global_pairs_are_unique_disjoint_and_exact_work_is_per_rival(self):
        generators = sp.csr_matrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0 / 16.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0 / 16.0],
            ]
        )
        result = build_property_pairhull_candidates(
            property_matrix=[
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            output_weight=np.eye(4),
            preactivation_center=[0.0, 0.25, 0.0, 0.25],
            preactivation_generators=generators,
            preactivation_error=np.zeros(4),
            lower=[-1.0, -13.0 / 16.0, -1.0, -13.0 / 16.0],
            upper=[1.0, 21.0 / 16.0, 1.0, 21.0 / 16.0],
            foundation_planes=np.zeros((2, 4)),
            foundation_intercepts=np.ones(2),
            pair_budget=2,
            time_limit=5.0,
        )
        self.assertEqual(result.status, "generated")
        self.assertEqual(result.receipt["global_pair_count"], 2)
        selected = [
            tuple(record["pair"])
            for record in result.receipt["selector"][
                "selected_pair_records"
            ]
        ]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(
            len({neuron for pair in selected for neuron in pair}),
            2 * len(selected),
        )
        self.assertEqual(result.receipt["float_proposals"], 2)
        self.assertLessEqual(
            result.receipt["exact_beta_evaluations"],
            result.receipt["rivals"],
        )
        self.assertLessEqual(
            result.receipt["exact_beta_evaluations"],
            result.receipt["float_proposals"],
        )
        np.testing.assert_array_equal(result.rival_ids, [0, 1])

    def test_ten_million_column_two_nnz_matrix_is_never_densified(self):
        generators = sp.csr_matrix(
            (
                np.asarray([1.0, 1.0]),
                np.asarray([9_999_999, 9_999_999], dtype=np.int64),
                np.asarray([0, 1, 2], dtype=np.int64),
            ),
            shape=(2, 10_000_000),
        )
        started = time.monotonic()
        result = build_property_pairhull_candidates(
            property_matrix=[[1.0, -1.0]],
            output_weight=np.eye(2),
            preactivation_center=[0.0, 0.25],
            preactivation_generators=generators,
            preactivation_error=[0.0, 0.0],
            lower=[-1.0, -0.75],
            upper=[1.0, 1.25],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[1.0],
            pair_budget=1,
            time_limit=5.0,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(result.status, "generated")
        self.assertEqual(result.receipt["continuous_columns"], 10_000_000)
        self.assertEqual(result.receipt["generator_nnz"], 2)
        self.assertEqual(
            result.receipt["selected_projections"][0][
                "compact_generator_columns"
            ],
            1,
        )
        self.assertFalse(
            result.receipt["selector"]["generator_matrix_densified"]
        )
        self.assertLess(elapsed, 3.0)

    def test_expired_deadline_discards_the_whole_batch(self):
        result = build_property_pairhull_candidates(
            property_matrix=[[1.0, -1.0]],
            output_weight=np.eye(2),
            preactivation_center=[0.0, 0.0],
            preactivation_generators=sp.csr_matrix([[1.0], [1.0]]),
            preactivation_error=[0.0, 0.0],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[1.0],
            pair_budget=1,
            time_limit=5.0,
            deadline=time.monotonic() - 1.0,
        )
        self.assertEqual(result.status, "deadline_fallback_foundations")
        self.assertFalse(result.receipt["whole_batch_complete"])
        self.assertEqual(result.rival_ids.size, 0)
        self.assertEqual(result.planes.shape, (0, 2))
        self.assertEqual(result.receipt["selected_candidates"], 0)

    def test_exact_beta_crossing_deadline_discards_the_whole_batch(self):
        from act.back_end.hybridz_tf import (
            property_pairhull_candidates as candidate_module,
        )

        original = candidate_module.exact_pairhull_beta

        def delayed_exact(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(0.03)
            return result

        with mock.patch.object(
            candidate_module,
            "exact_pairhull_beta",
            side_effect=delayed_exact,
        ):
            result = build_property_pairhull_candidates(
                property_matrix=[[1.0, -1.0]],
                output_weight=np.eye(2),
                preactivation_center=[0.0, 0.25],
                preactivation_generators=sp.csr_matrix(
                    [[1.0, 0.0], [1.0, 1.0 / 16.0]]
                ),
                preactivation_error=[0.0, 0.0],
                lower=[-1.0, -13.0 / 16.0],
                upper=[1.0, 21.0 / 16.0],
                foundation_planes=[[0.0, 0.0]],
                foundation_intercepts=[1.0],
                pair_budget=1,
                time_limit=0.02,
            )
        self.assertEqual(result.status, "deadline_fallback_foundations")
        self.assertFalse(result.receipt["whole_batch_complete"])
        self.assertEqual(result.rival_ids.size, 0)
        self.assertEqual(result.receipt["partial_candidates_discarded"], 0)
        self.assertIn("exact PairHull beta", result.receipt["error"])

    def test_compact_union_cap_discards_the_whole_batch(self):
        result = build_property_pairhull_candidates(
            property_matrix=[[1.0, -1.0]],
            output_weight=np.eye(2),
            preactivation_center=[0.0, 0.25],
            preactivation_generators=sp.csr_matrix(
                [[1.0, 1.0], [1.0, 0.5]]
            ),
            preactivation_error=[0.0, 0.0],
            lower=[-2.0, -1.75],
            upper=[2.0, 2.25],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[2.0],
            pair_budget=1,
            time_limit=5.0,
            max_pair_union_nnz=1,
        )
        self.assertEqual(result.status, "resource_cap_fallback_foundations")
        self.assertFalse(result.receipt["whole_batch_complete"])
        self.assertEqual(result.rival_ids.size, 0)
        self.assertIn("compact", result.receipt["error"])

    def test_selector_posting_cap_is_explicit_not_partial_complete(self):
        result = build_property_pairhull_candidates(
            property_matrix=[[1.0, -1.0]],
            output_weight=np.eye(2),
            preactivation_center=[0.0, 0.0],
            preactivation_generators=sp.csr_matrix([[1.0], [1.0]]),
            preactivation_error=[0.0, 0.0],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[1.0],
            pair_budget=1,
            time_limit=5.0,
            row_topk=8,
            max_selector_postings=8,
        )
        # Two rows times row_topk exceeds the explicitly supplied cap even
        # though these toy rows happen to have only one nonzero each.
        self.assertEqual(result.status, "resource_cap_fallback_foundations")
        self.assertFalse(result.receipt["whole_batch_complete"])
        self.assertEqual(result.rival_ids.size, 0)

    def test_invalid_budget_is_an_explicit_safe_fallback(self):
        result = build_property_pairhull_candidates(
            property_matrix=[[1.0, -1.0]],
            output_weight=np.eye(2),
            preactivation_center=[0.0, 0.0],
            preactivation_generators=sp.csr_matrix([[1.0], [1.0]]),
            preactivation_error=[0.0, 0.0],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            foundation_planes=[[0.0, 0.0]],
            foundation_intercepts=[1.0],
            pair_budget=9,
            time_limit=5.0,
        )
        self.assertEqual(result.status, "error_fallback_foundations")
        self.assertFalse(result.receipt["whole_batch_complete"])
        self.assertEqual(result.receipt["error_type"], "ValueError")


class PropertyPairHullCandidateReceiptTests(unittest.TestCase):
    def test_nested_tamper_is_detected_and_generic_helper_is_public(self):
        result = _dyadic_decisive()
        self.assertTrue(verify_pairhull_candidate_receipt(result.receipt))
        tampered = copy.deepcopy(result.receipt)
        tampered["candidate_records"][0]["exact_pairhull_receipt"][
            "beta_exact"
        ] = "1"
        self.assertFalse(verify_pairhull_candidate_receipt(tampered))

        wrapper = finalize_property_pairhull_candidates_receipt(
            {
                "schema": "operator_hz_property_tail_pairhull_v1",
                "candidate_receipt": result.receipt,
            }
        )
        self.assertTrue(verify_pairhull_candidate_receipt(wrapper))
        wrapper["candidate_receipt"]["status"] = "tampered"
        self.assertFalse(verify_pairhull_candidate_receipt(wrapper))


if __name__ == "__main__":
    unittest.main()
