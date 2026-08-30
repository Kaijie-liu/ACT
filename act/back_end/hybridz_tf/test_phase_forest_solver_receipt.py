from __future__ import annotations

from dataclasses import replace
import hashlib
import itertools
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    PhaseBoundWaveRequest,
    PhaseForestNode,
    PhaseNodeBound,
    RivalSpec,
    RivalUpperBound,
    ordered_property_digest,
    run_adaptive_phase_forest_candidate,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.phase_forest_solver_receipt import (
    ToyPhaseSolverConfig,
    ToyPhaseSolverReceiptError,
    ToyReceiptedPhaseBoundWave,
    consume_toy_phase_rival_evidence,
    consume_toy_phase_solver_batch,
    new_toy_phase_solver_invocation_id,
    solve_toy_phase_rival_evidence,
    solve_toy_phase_solver_batch,
)
from act.back_end.solver.solver_hz import SparseHZono


def _clique_hz(n_binary: int = 3) -> SparseHZono:
    """Stable-set relaxation with integer maximum one and LP max n/2."""

    edges = tuple(itertools.combinations(range(n_binary), 2))
    rows = np.zeros((len(edges), n_binary), dtype=np.float64)
    for row, (left, right) in enumerate(edges):
        rows[row, left] = 1.0
        rows[row, right] = 1.0
    return SparseHZono(
        c=np.asarray([n_binary / 2.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, 0), dtype=np.float64),
        Gb=sp.csr_matrix(
            np.full((1, n_binary), 0.5, dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, n_binary), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((len(edges), 0), dtype=np.float64),
        Aub=sp.csr_matrix(rows, dtype=np.float64),
        ub=np.zeros(len(edges), dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=np.arange(
            700, 700 + n_binary, dtype=np.int64
        ),
    )


def _point_hz(value: float = 1.0) -> SparseHZono:
    return SparseHZono(
        c=np.asarray([value], dtype=np.float64),
        Gc=sp.csr_matrix((1, 0), dtype=np.float64),
        Gb=sp.csr_matrix((1, 0), dtype=np.float64),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, 0), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((0, 0), dtype=np.float64),
        Aub=sp.csr_matrix((0, 0), dtype=np.float64),
        ub=np.zeros(0, dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=np.zeros(0, dtype=np.int64),
    )


def _rivals(
    *,
    first_threshold: float = 1.25,
) -> tuple[RivalSpec, RivalSpec]:
    return (
        RivalSpec(
            rival_id=17,
            objective=(1.0,),
            threshold=float(first_threshold),
            assert_digest=hashlib.sha256(
                b"raw-assert-17"
            ).hexdigest(),
        ),
        RivalSpec(
            rival_id=91,
            objective=(0.5,),
            threshold=0.625,
            assert_digest=hashlib.sha256(
                b"raw-assert-91"
            ).hexdigest(),
        ),
    )


def _request(
    hz: SparseHZono,
    rivals: tuple[RivalSpec, ...],
    *,
    deadline: float,
    node_id: int = 4,
    wave_index: int = 3,
) -> PhaseBoundWaveRequest:
    return PhaseBoundWaveRequest(
        wave_index=wave_index,
        nodes=(
            PhaseForestNode(
                node_id=node_id,
                depth=0,
                lineage=(),
                hz=hz,
            ),
        ),
        rivals=rivals,
        property_digest=ordered_property_digest(rivals),
        deadline=deadline,
        proof_authority=False,
    )


def _single_evidence(
    hz: SparseHZono,
    rival: RivalSpec,
    *,
    deadline: float,
    config: ToyPhaseSolverConfig,
):
    node = PhaseForestNode(
        node_id=4, depth=0, lineage=(), hz=hz
    )
    invocation = new_toy_phase_solver_invocation_id()
    prop = ordered_property_digest((rival,))
    evidence = solve_toy_phase_rival_evidence(
        node,
        rival,
        config=config,
        deadline=deadline,
        invocation_id=invocation,
        wave_index=3,
        node_position=0,
        rival_position=0,
        property_digest=prop,
    )
    expected = {
        "node": node,
        "rival": rival,
        "config": config,
        "deadline": deadline,
        "invocation_id": invocation,
        "wave_index": 3,
        "node_position": 0,
        "rival_position": 0,
        "property_digest": prop,
    }
    return evidence, expected


class ToyPhaseSolverReceiptTests(unittest.TestCase):
    def test_nonzero_threshold_two_rival_honest_batch_and_relabel_reject(
        self,
    ) -> None:
        hz = _clique_hz()
        rivals = _rivals()
        deadline = time.monotonic() + 10.0
        request = _request(hz, rivals, deadline=deadline)
        config = ToyPhaseSolverConfig()
        invocation = new_toy_phase_solver_invocation_id()
        evidence = solve_toy_phase_solver_batch(
            request, config=config, invocation_id=invocation
        )
        self.assertEqual(len(evidence), 2)
        bounds, errors = consume_toy_phase_solver_batch(
            evidence,
            request,
            config=config,
            invocation_id=invocation,
        )
        self.assertEqual(errors, ())
        self.assertIsNotNone(bounds)
        self.assertEqual(
            [item.rival_id for item in bounds[0]], [17, 91]
        )
        self.assertGreater(bounds[0][0].upper, 1.5)
        self.assertGreater(bounds[0][1].upper, 0.75)
        for item in evidence:
            self.assertFalse(item.receipt["proof_authority"])
            self.assertTrue(item.receipt["toy_only"])
            self.assertFalse(item.receipt["verifier_connected"])
            self.assertFalse(item.receipt["bab_connected"])
            self.assertEqual(
                item.receipt["rival"]["threshold_hex"],
                float(
                    rivals[
                        item.receipt["invocation"][
                            "rival_position"
                        ]
                    ].threshold
                ).hex(),
            )

        # Fresh capabilities with the two live numeric result objects swapped
        # cannot be relabelled under the otherwise-correct rival IDs.
        invocation = new_toy_phase_solver_invocation_id()
        evidence = solve_toy_phase_solver_batch(
            request, config=config, invocation_id=invocation
        )
        relabelled = (
            replace(evidence[0], result=evidence[1].result),
            evidence[1],
        )
        bounds, errors = consume_toy_phase_solver_batch(
            relabelled,
            request,
            config=config,
            invocation_id=invocation,
        )
        self.assertIsNone(bounds)
        self.assertIn("batch_order_or_binding_mismatch", errors)

    def test_single_use_result_receipt_and_numeric_identity(
        self,
    ) -> None:
        config = ToyPhaseSolverConfig()

        with self.subTest("copied_receipt"):
            evidence, expected = _single_evidence(
                _clique_hz(),
                _rivals()[0],
                deadline=time.monotonic() + 10.0,
                config=config,
            )
            copied = replace(
                evidence, receipt=dict(evidence.receipt)
            )
            outcome = consume_toy_phase_rival_evidence(
                copied, **expected
            )
            self.assertFalse(outcome.valid)
            self.assertIn(
                "live_receipt_identity_mismatch", outcome.errors
            )

        with self.subTest("copied_result_wrong_numeric"):
            evidence, expected = _single_evidence(
                _clique_hz(),
                _rivals()[0],
                deadline=time.monotonic() + 10.0,
                config=config,
            )
            wrong_result = replace(
                evidence.result,
                upper=float(evidence.result.upper) + 100.0,
            )
            outcome = consume_toy_phase_rival_evidence(
                replace(evidence, result=wrong_result),
                **expected,
            )
            self.assertFalse(outcome.valid)
            self.assertIn(
                "live_result_identity_mismatch", outcome.errors
            )

        with self.subTest("live_receipt_tamper"):
            evidence, expected = _single_evidence(
                _clique_hz(),
                _rivals()[0],
                deadline=time.monotonic() + 10.0,
                config=config,
            )
            evidence.receipt["result"]["status"] = "UNKNOWN"
            outcome = consume_toy_phase_rival_evidence(
                evidence, **expected
            )
            self.assertFalse(outcome.valid)
            self.assertIn("receipt_mac_mismatch", outcome.errors)
            self.assertIn(
                "sealed_result_binding_mismatch", outcome.errors
            )

        with self.subTest("stale_second_consume"):
            evidence, expected = _single_evidence(
                _clique_hz(),
                _rivals()[0],
                deadline=time.monotonic() + 10.0,
                config=config,
            )
            first = consume_toy_phase_rival_evidence(
                evidence, **expected
            )
            second = consume_toy_phase_rival_evidence(
                evidence, **expected
            )
            self.assertTrue(first.valid)
            self.assertFalse(second.valid)
            self.assertIn(
                "missing_stale_or_forged_capability",
                second.errors,
            )
            with self.assertRaisesRegex(
                ToyPhaseSolverReceiptError,
                "duplicate_or_stale_invocation_slot",
            ):
                solve_toy_phase_rival_evidence(
                    expected["node"],
                    expected["rival"],
                    config=expected["config"],
                    deadline=expected["deadline"],
                    invocation_id=expected["invocation_id"],
                    wave_index=expected["wave_index"],
                    node_position=expected["node_position"],
                    rival_position=expected["rival_position"],
                    property_digest=expected["property_digest"],
                )

    def test_node_rival_config_status_and_live_hz_tamper_fail_closed(
        self,
    ) -> None:
        config = ToyPhaseSolverConfig()
        cases = (
            "node",
            "depth",
            "objective",
            "threshold",
            "assert",
            "config",
            "deadline",
            "status",
            "live_hz_c",
            "live_hz_csr",
            "live_hz_conditional",
        )
        for case in cases:
            with self.subTest(case=case):
                hz = _clique_hz()
                if case == "live_hz_conditional":
                    setattr(
                        hz,
                        "_toy_conditional_binding",
                        {"phase": 1, "row": 7},
                    )
                rival = _rivals()[0]
                evidence, expected = _single_evidence(
                    hz,
                    rival,
                    deadline=time.monotonic() + 10.0,
                    config=config,
                )
                if case == "node":
                    expected["node"] = replace(
                        expected["node"], node_id=999
                    )
                elif case == "depth":
                    expected["node"] = replace(
                        expected["node"], depth=7
                    )
                elif case == "objective":
                    expected["rival"] = replace(
                        rival, objective=(2.0,)
                    )
                elif case == "threshold":
                    expected["rival"] = replace(
                        rival,
                        threshold=np.nextafter(
                            rival.threshold, np.inf
                        ),
                    )
                elif case == "assert":
                    expected["rival"] = replace(
                        rival,
                        assert_digest=hashlib.sha256(
                            b"different-raw-assert"
                        ).hexdigest(),
                    )
                elif case == "config":
                    expected["config"] = replace(
                        config, max_variables=255
                    )
                elif case == "deadline":
                    expected["deadline"] = np.nextafter(
                        expected["deadline"], np.inf
                    )
                elif case == "status":
                    evidence = replace(
                        evidence,
                        result=replace(
                            evidence.result, status="UNKNOWN"
                        ),
                    )
                elif case == "live_hz_c":
                    hz.c[0] += 0.25
                elif case == "live_hz_csr":
                    hz.Aub.data[0] += 0.25
                else:
                    hz._toy_conditional_binding["phase"] = -1
                outcome = consume_toy_phase_rival_evidence(
                    evidence, **expected
                )
                self.assertFalse(outcome.valid)
                self.assertTrue(outcome.errors)
                if case.startswith("live_hz"):
                    self.assertIn(
                        "sealed_node_binding_mismatch",
                        outcome.errors,
                    )
                if case == "live_hz_c":
                    self.assertIn(
                        "independent_numeric_replay_mismatch",
                        outcome.errors,
                    )

    def test_batch_reorder_duplicate_and_omit_are_atomic_failures(
        self,
    ) -> None:
        config = ToyPhaseSolverConfig()
        for case in ("reorder", "duplicate", "omit"):
            with self.subTest(case=case):
                request = _request(
                    _clique_hz(),
                    _rivals(),
                    deadline=time.monotonic() + 10.0,
                )
                invocation = new_toy_phase_solver_invocation_id()
                evidence = solve_toy_phase_solver_batch(
                    request,
                    config=config,
                    invocation_id=invocation,
                )
                if case == "reorder":
                    altered = (evidence[1], evidence[0])
                    expected_error = (
                        "batch_order_or_binding_mismatch"
                    )
                elif case == "duplicate":
                    altered = (evidence[0], evidence[0])
                    expected_error = "batch_duplicate_capability"
                else:
                    altered = evidence[:-1]
                    expected_error = "batch_evidence_count_mismatch"
                bounds, errors = consume_toy_phase_solver_batch(
                    altered,
                    request,
                    config=config,
                    invocation_id=invocation,
                )
                self.assertIsNone(bounds)
                self.assertIn(expected_error, errors)
                if case == "duplicate":
                    stale = consume_toy_phase_rival_evidence(
                        evidence[1],
                        node=request.nodes[0],
                        rival=request.rivals[1],
                        config=config,
                        deadline=request.deadline,
                        invocation_id=invocation,
                        wave_index=request.wave_index,
                        node_position=0,
                        rival_position=1,
                        property_digest=request.property_digest,
                    )
                    self.assertFalse(stale.valid)
                    self.assertIn(
                        "missing_stale_or_forged_capability",
                        stale.errors,
                    )

        # Cross-splicing two otherwise valid invocation groups must revoke
        # both complete groups, including siblings omitted from the tuple.
        request = _request(
            _clique_hz(),
            _rivals(),
            deadline=time.monotonic() + 10.0,
        )
        first_invocation = new_toy_phase_solver_invocation_id()
        second_invocation = new_toy_phase_solver_invocation_id()
        first = solve_toy_phase_solver_batch(
            request,
            config=config,
            invocation_id=first_invocation,
        )
        second = solve_toy_phase_solver_batch(
            request,
            config=config,
            invocation_id=second_invocation,
        )
        bounds, errors = consume_toy_phase_solver_batch(
            (first[0], second[1]),
            request,
            config=config,
            invocation_id=first_invocation,
        )
        self.assertIsNone(bounds)
        self.assertIn("sealed_invocation_binding_mismatch", errors)
        for item, invocation, rival_position in (
            (first[1], first_invocation, 1),
            (second[0], second_invocation, 0),
        ):
            stale = consume_toy_phase_rival_evidence(
                item,
                node=request.nodes[0],
                rival=request.rivals[rival_position],
                config=config,
                deadline=request.deadline,
                invocation_id=invocation,
                wave_index=request.wave_index,
                node_position=0,
                rival_position=rival_position,
                property_digest=request.property_digest,
            )
            self.assertFalse(stale.valid)
            self.assertIn(
                "missing_stale_or_forged_capability",
                stale.errors,
            )

    def test_deadline_and_outward_threshold_boundary_fail_safe(
        self,
    ) -> None:
        hz = _clique_hz()
        rival = _rivals()[0]
        node = PhaseForestNode(4, 0, (), hz)
        with self.assertRaisesRegex(
            ToyPhaseSolverReceiptError, "deadline_before_lp"
        ):
            solve_toy_phase_rival_evidence(
                node,
                rival,
                config=ToyPhaseSolverConfig(),
                deadline=time.monotonic() - 1.0,
                invocation_id=new_toy_phase_solver_invocation_id(),
                wave_index=0,
                node_position=0,
                rival_position=0,
                property_digest=ordered_property_digest((rival,)),
            )

        evidence, expected = _single_evidence(
            _clique_hz(),
            rival,
            deadline=time.monotonic() + 10.0,
            config=ToyPhaseSolverConfig(),
        )
        with patch(
            "act.back_end.hybridz_tf."
            "phase_forest_solver_receipt.time.monotonic",
            return_value=expected["deadline"] + 1.0,
        ):
            outcome = consume_toy_phase_rival_evidence(
                evidence, **expected
            )
        self.assertFalse(outcome.valid)
        self.assertIn("deadline_expired_at_consume", outcome.errors)

        # LP max is exactly 1.5.  A threshold just one float above 1.5
        # must remain UNKNOWN because long-double->float64 is rounded outward.
        near = np.nextafter(np.float64(1.5), np.inf)
        rivals = _rivals(first_threshold=float(near))
        request = _request(
            _clique_hz(),
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        result = ToyReceiptedPhaseBoundWave()(request)
        self.assertEqual(result[0].verdict, "UNKNOWN")
        self.assertGreater(result[0].rival_bounds[0].upper, near)

        point_rival = replace(
            _rivals()[0], threshold=1.1
        )
        point_request = _request(
            _point_hz(),
            (point_rival,),
            deadline=time.monotonic() + 10.0,
        )
        point_adapter = ToyReceiptedPhaseBoundWave()
        point_result = point_adapter(point_request)
        self.assertEqual(point_result[0].verdict, "SAFE")
        self.assertFalse(point_result[0].proof_authority)
        self.assertEqual(
            point_adapter.audit_receipts[0]["invocation"][
                "backend"
            ],
            "exact_stored_float_point",
        )

    def test_receipted_adapter_completes_asymmetric_toy_prooflessly(
        self,
    ) -> None:
        hz = _clique_hz()
        rivals = _rivals()
        ids = tuple(int(value) for value in hz.bcol_ids.tolist())
        root_bound = PhaseNodeBound(
            node_id=0,
            lineage=(),
            remaining_bcol_ids=ids,
            rival_bounds=tuple(
                RivalUpperBound(
                    rival_id=rival.rival_id,
                    binding_digest=rival.binding_digest,
                    upper=scale,
                )
                for rival, scale in zip(rivals, (1.5, 0.75))
            ),
            property_digest=ordered_property_digest(rivals),
            node_semantic_digest=sparse_hz_semantic_digest(hz),
            verdict="UNKNOWN",
            proof_authority=False,
        )
        adapter = ToyReceiptedPhaseBoundWave()
        result = run_adaptive_phase_forest_candidate(
            hz,
            rivals,
            root_bound,
            adapter,
            deadline=time.monotonic() + 10.0,
            max_depth=2,
            max_nodes=7,
        )
        self.assertEqual(
            result.status, "all_leaves_safe_candidate"
        )
        self.assertTrue(result.all_leaves_safe)
        self.assertFalse(result.proof_authority)
        self.assertEqual(result.receipt["wave_sizes"], [2])
        self.assertEqual(len(adapter.audit_receipts), 4)
        self.assertTrue(
            all(
                receipt["proof_authority"] is False
                and receipt["portable_authority"] is False
                for receipt in adapter.audit_receipts
            )
        )


if __name__ == "__main__":
    unittest.main()
