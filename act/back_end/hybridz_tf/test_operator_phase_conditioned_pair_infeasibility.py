#!/usr/bin/env python3
"""Controlled soundness gates for pair-local PCOH pattern infeasibility."""

from __future__ import annotations

from dataclasses import replace
import itertools
import math
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import (
    operator_phase_conditioned_pair_infeasibility as pair_core,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_hz import build_operator_hz
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _DTYPE,
    _dense,
    _k4_corner_build,
    _layer,
    _rivals,
)


def _exact_relu_build(weights, biases):
    weight_array = np.asarray(weights, dtype=np.float64)
    bias_array = np.asarray(biases, dtype=np.float64)
    count, width = weight_array.shape
    lower = torch.full((1, width), -1.0, dtype=_DTYPE)
    upper = torch.full((1, width), 1.0, dtype=_DTYPE)
    layers = [
        _layer(0, "INPUT", {"shape": (1, width)}, width=width),
        _layer(
            1,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": lower, "ub": upper},
            width=width,
        ),
        _dense(2, weight_array, bias_array),
        _layer(3, "RELU", width=count),
        _dense(
            4,
            (
                tuple(0.0 for _ in range(count)),
                tuple(1.0 for _ in range(count)),
                tuple(0.5 for _ in range(count)),
            ),
            (0.75, 0.0, 0.0),
        ),
        _layer(5, "ASSERT", width=3),
    ]
    predecessors = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    successors = {layer.id: [] for layer in layers}
    for child, parents in predecessors.items():
        for parent in parents:
            successors[parent].append(child)
    network = SimpleNamespace(
        layers=layers,
        preds=predecessors,
        succs=successors,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {}
    for layer in layers:
        layer_width = len(layer.out_vars)
        if layer.kind in {"INPUT", "INPUT_SPEC"}:
            fact_lower = lower.clone()
            fact_upper = upper.clone()
        else:
            fact_lower = torch.full(
                (1, layer_width), -1.0e30, dtype=_DTYPE
            )
            fact_upper = torch.full(
                (1, layer_width), 1.0e30, dtype=_DTYPE
            )
        facts[layer.id] = Fact(Bounds(fact_lower, fact_upper), ConSet())
    return build_operator_hz(
        network,
        facts,
        facts,
        exact_budget=count,
        materialize_add=True,
    )


def _mixed_sign_build():
    # f0=x+1/2 and f1=x-1/2.  (inactive, active) would require
    # x<=-1/2 and x>=+1/2 simultaneously.
    return _exact_relu_build(((1.0,), (1.0,)), (0.5, -0.5))


def _triple_only_build():
    # Active phases require x>=0, y>=0, and -x-y-1/2>=0.  The triple is
    # impossible, but every signed pair has a witness in [-1,1]^2.
    return _exact_relu_build(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)),
        (0.0, 0.0, -0.5),
    )


def _selection(build):
    return derive_operator_exact_relu_property_phase_literals(
        build, _rivals()
    )


def _run(build, selection=None, *, caps=pair_core.PairLocalCaps(), seconds=30.0):
    if selection is None:
        selection = _selection(build)
    stable_ids = tuple(mapping.stable_bcol_id for mapping in selection.mappings)
    result = pair_core.run_phase_conditioned_pair_infeasibility_candidate(
        build,
        _rivals(),
        selection,
        stable_bit_ids=stable_ids,
        deadline=time.monotonic() + seconds,
        caps=caps,
    )
    return selection, result


class PairLocalK4CornerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _k4_corner_build()
        cls.selection, cls.bundle = _run(cls.build, seconds=40.0)

    def test_six_positive_edges_cover_exactly_eleven_patterns(self):
        bundle = self.bundle
        certified = tuple(
            record for record in bundle.records
            if record.status == "certified_conflict"
        )
        self.assertEqual(len(bundle.records), 24)
        self.assertEqual(len(certified), 6)
        self.assertEqual(len(bundle.certificates), 6)
        self.assertTrue(
            all(record.pair[0][1] == 1 and record.pair[1][1] == 1 for record in certified)
        )
        empty = tuple(
            item for item in bundle.coverage
            if item.status == "certified_empty_by_pair"
        )
        unknown = tuple(
            item for item in bundle.coverage
            if item.status == "not_certified_empty"
        )
        self.assertEqual(len(bundle.coverage), 16)
        self.assertEqual(len(empty), 11)
        self.assertEqual(len(unknown), 5)
        self.assertTrue(all(item.eta_fixed_value == -1 for item in empty))
        self.assertTrue(all(item.eta_fixed_value is None for item in unknown))
        self.assertEqual(
            {item.pattern for item in unknown},
            {
                (-1, -1, -1, -1),
                (1, -1, -1, -1),
                (-1, 1, -1, -1),
                (-1, -1, 1, -1),
                (-1, -1, -1, 1),
            },
        )

    def test_every_candidate_model_is_six_rows_full_width_and_closed(self):
        expected_columns = self.build.hz.n_cont + self.build.hz.n_bin
        for record in self.bundle.records:
            self.assertEqual(len(record.source_upper_rows), 6)
            self.assertEqual(len(set(record.source_upper_rows)), 6)
            self.assertEqual(record.local_model_rows, 6)
            self.assertEqual(record.local_model_columns, expected_columns)
            self.assertTrue(record.model_closed)
            self.assertEqual(record.solver_threads, 1)
        receipt = self.bundle.receipt
        self.assertFalse(receipt["full_parent_milp_used"])
        self.assertFalse(receipt["full_parent_csr_loaded_into_candidate"])
        self.assertFalse(receipt["full_parent_snapshot_created"])
        self.assertFalse(receipt["sparse_hstack_used"])
        self.assertFalse(receipt["sparse_vstack_used"])
        self.assertTrue(receipt["all_models_closed"])

    def test_live_bundle_verifier_accepts_exact_k4_result(self):
        self.assertTrue(
            pair_core.verify_phase_conditioned_pair_infeasibility_bundle(
                self.build,
                _rivals(),
                self.selection,
                self.bundle,
                deadline=time.monotonic() + 40.0,
            )
        )

    def test_bundle_verifier_scans_source_frame_exactly_once(self):
        original = pair_core._ordered_source_frame_digest
        with mock.patch.object(
            pair_core,
            "_ordered_source_frame_digest",
            wraps=original,
        ) as source_frame_digest:
            self.assertTrue(
                pair_core.verify_phase_conditioned_pair_infeasibility_bundle(
                    self.build,
                    _rivals(),
                    self.selection,
                    self.bundle,
                    deadline=time.monotonic() + 40.0,
                )
            )
        self.assertEqual(source_frame_digest.call_count, 1)
        self.assertTrue(
            self.bundle.receipt[
                "certificate_replays_reuse_precomputed_source_frame"
            ]
        )
        self.assertEqual(
            self.bundle.receipt["source_frame_digest_computations"],
            1,
        )

    def test_bundle_record_coverage_receipt_and_digest_tamper_fail(self):
        first = self.bundle.records[0]
        record_tamper = replace(first, status="certified_conflict")
        cases = (
            replace(
                self.bundle,
                records=(record_tamper, *self.bundle.records[1:]),
            ),
            replace(
                self.bundle,
                coverage=(
                    replace(self.bundle.coverage[0], eta_fixed_value=0),
                    *self.bundle.coverage[1:],
                ),
            ),
            replace(
                self.bundle,
                receipt={**self.bundle.receipt, "solver_threads": 2},
            ),
            replace(self.bundle, bundle_sha256="0" * 64),
            replace(self.bundle, proof_authority=True),
        )
        for index, candidate in enumerate(cases):
            with self.subTest(index=index):
                self.assertFalse(
                    pair_core.verify_phase_conditioned_pair_infeasibility_bundle(
                        self.build,
                        _rivals(),
                        self.selection,
                        candidate,
                        deadline=time.monotonic() + 40.0,
                    )
                )


class PairLocalControlsTests(unittest.TestCase):
    def test_mixed_sign_conflict_is_preserved_exactly(self):
        build = _mixed_sign_build()
        selection, bundle = _run(build)
        certified = tuple(
            record for record in bundle.records
            if record.status == "certified_conflict"
        )
        self.assertEqual(len(bundle.records), 4)
        self.assertEqual(len(certified), 1)
        stable_ids = bundle.stable_bit_ids
        self.assertEqual(
            certified[0].pair,
            ((stable_ids[0], -1), (stable_ids[1], 1)),
        )
        self.assertEqual(
            [item.pattern for item in bundle.coverage if item.status == "certified_empty_by_pair"],
            [(-1, 1)],
        )
        self.assertTrue(
            pair_core.verify_phase_conditioned_pair_infeasibility_bundle(
                build,
                _rivals(),
                selection,
                bundle,
                deadline=time.monotonic() + 30.0,
            )
        )

    def test_no_sparse_stack_is_called(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        with mock.patch.object(
            sp, "hstack", side_effect=AssertionError("hstack forbidden")
        ), mock.patch.object(
            sp, "vstack", side_effect=AssertionError("vstack forbidden")
        ):
            _, bundle = _run(build, selection)
        self.assertEqual(
            sum(record.status == "certified_conflict" for record in bundle.records),
            1,
        )

    def test_wrong_ray_sign_is_rejected_by_live_fraction_replay(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        original = pair_core._solve_local_pair

        def wrong_sign(*args, **kwargs):
            outcome = original(*args, **kwargs)
            if outcome.raw_ray is not None:
                return replace(outcome, raw_ray=-outcome.raw_ray)
            return outcome

        with mock.patch.object(pair_core, "_solve_local_pair", side_effect=wrong_sign):
            _, bundle = _run(build, selection)
        self.assertFalse(bundle.certificates)
        self.assertTrue(
            any(record.status == "exact_replay_rejected" for record in bundle.records)
        )
        self.assertTrue(
            all(item.status == "not_certified_empty" for item in bundle.coverage)
        )

    def test_wrong_zero_padding_row_map_is_rejected(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        original = pair_core._zero_pad_local_ray

        def wrong_map(raw_ray, source_rows, *, full_rows):
            value = original(raw_ray, source_rows, full_rows=full_rows)
            return np.roll(value, 1)

        with mock.patch.object(pair_core, "_zero_pad_local_ray", side_effect=wrong_map):
            _, bundle = _run(build, selection)
        self.assertFalse(bundle.certificates)
        self.assertTrue(
            any(record.status == "exact_replay_rejected" for record in bundle.records)
        )

    def test_wrong_local_rhs_cannot_survive_live_parent_replay(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        original = pair_core._solve_local_pair

        def wrong_rhs(hz, **kwargs):
            rows = kwargs["source_rows"]
            saved = hz.ub[np.asarray(rows, dtype=np.int64)].copy()
            try:
                # This makes the numerical six-row model artificially tight.
                # Restore the live source before the exact checker sees it.
                hz.ub[np.asarray(rows, dtype=np.int64)] = saved - 100.0
                return original(hz, **kwargs)
            finally:
                hz.ub[np.asarray(rows, dtype=np.int64)] = saved

        with mock.patch.object(pair_core, "_solve_local_pair", side_effect=wrong_rhs):
            _, bundle = _run(build, selection)
        certified = tuple(
            record.pair
            for record in bundle.records
            if record.status == "certified_conflict"
        )
        # The one genuinely impossible mixed phase may still replay.  The
        # extra infeasibilities caused only by the false RHS must not.
        self.assertLessEqual(len(certified), 1)
        self.assertTrue(
            any(record.status == "exact_replay_rejected" for record in bundle.records)
        )

    def test_stale_rhs_and_selection_fail_before_local_solver(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        row = selection.mappings[0].lower_upper_row
        build.hz.ub[row] = np.nextafter(build.hz.ub[row], math.inf)
        with mock.patch.object(
            pair_core,
            "_solve_local_pair",
            side_effect=AssertionError("stale source reached solver"),
        ), self.assertRaisesRegex(
            pair_core.PhaseConditionedPairInfeasibilityError,
            "selection_live_verification_failed",
        ):
            _run(build, selection)

    def test_triple_only_empty_pattern_is_not_claimed_by_pair_coverage(self):
        build = _triple_only_build()
        selection, bundle = _run(build)
        self.assertFalse(bundle.certificates)
        self.assertTrue(
            all(item.status == "not_certified_empty" for item in bundle.coverage)
        )
        triple = next(item for item in bundle.coverage if item.pattern == (1, 1, 1))
        self.assertEqual(triple.status, "not_certified_empty")
        # Independent exact algebraic counterexample: the three active rows
        # require x>=0, y>=0, and x+y<=-1/2.
        self.assertFalse(
            any(
                x >= 0 and y >= 0 and x + y <= -0.5
                for x, y in itertools.product(
                    (-1.0, -0.5, 0.0, 0.5, 1.0), repeat=2
                )
            )
        )

    def test_nonoptimal_no_ray_and_close_failure_never_create_edges(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        n_variables = build.hz.n_cont + build.hz.n_bin
        outcomes = (
            pair_core._LocalSolveOutcome(
                "feasible_or_unknown", None, 6, n_variables, 1, True
            ),
            pair_core._LocalSolveOutcome(
                "infeasible_without_ray", None, 6, n_variables, 1, True
            ),
            pair_core._LocalSolveOutcome(
                "infeasible_with_ray", np.ones(6), 6, n_variables, 1, False
            ),
        )
        for outcome in outcomes:
            with self.subTest(status=outcome.status, closed=outcome.model_closed):
                with mock.patch.object(
                    pair_core, "_solve_local_pair", return_value=outcome
                ):
                    _, bundle = _run(build, selection)
                self.assertFalse(bundle.certificates)
                self.assertTrue(
                    all(item.status == "not_certified_empty" for item in bundle.coverage)
                )
                if not outcome.model_closed:
                    self.assertTrue(
                        all(record.status == "model_close_failed" for record in bundle.records)
                    )

    def test_deadline_and_caps_fail_closed(self):
        build = _mixed_sign_build()
        selection = _selection(build)
        stable_ids = tuple(mapping.stable_bcol_id for mapping in selection.mappings)
        with self.assertRaisesRegex(
            pair_core.PhaseConditionedPairInfeasibilityError,
            "deadline_expired",
        ):
            pair_core.run_phase_conditioned_pair_infeasibility_candidate(
                build,
                _rivals(),
                selection,
                stable_bit_ids=stable_ids,
                deadline=time.monotonic() - 1.0,
            )
        with self.assertRaisesRegex(
            pair_core.PhaseConditionedPairInfeasibilityError,
            "signed_pair_query_cap_exceeded",
        ):
            pair_core.run_phase_conditioned_pair_infeasibility_candidate(
                build,
                _rivals(),
                selection,
                stable_bit_ids=stable_ids,
                deadline=time.monotonic() + 30.0,
                caps=replace(
                    pair_core.PairLocalCaps(), max_signed_pair_queries=3
                ),
            )
        with self.assertRaises(
            pair_core.PhaseConditionedPairInfeasibilityError
        ):
            pair_core.run_phase_conditioned_pair_infeasibility_candidate(
                build,
                _rivals(),
                selection,
                stable_bit_ids=(stable_ids[0], stable_ids[0]),
                deadline=time.monotonic() + 30.0,
            )

    def test_k1_has_two_unknown_patterns_and_no_pair_queries(self):
        build = _exact_relu_build(((1.0,),), (0.0,))
        selection, bundle = _run(build)
        self.assertFalse(bundle.records)
        self.assertFalse(bundle.certificates)
        self.assertEqual(
            [(item.pattern, item.status) for item in bundle.coverage],
            [((-1,), "not_certified_empty"), ((1,), "not_certified_empty")],
        )
        self.assertTrue(
            pair_core.verify_phase_conditioned_pair_infeasibility_bundle(
                build,
                _rivals(),
                selection,
                bundle,
                deadline=time.monotonic() + 30.0,
            )
        )

    def test_low_local_nnz_cap_returns_partial_without_false_empty(self):
        build = _mixed_sign_build()
        selection, bundle = _run(
            build,
            caps=replace(pair_core.PairLocalCaps(), max_local_nonzeros=1),
        )
        self.assertEqual(bundle.status, "partial")
        self.assertFalse(bundle.certificates)
        self.assertTrue(
            all(record.status == "candidate_error" for record in bundle.records)
        )
        self.assertTrue(
            pair_core.verify_phase_conditioned_pair_infeasibility_bundle(
                build,
                _rivals(),
                selection,
                bundle,
                deadline=time.monotonic() + 30.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
