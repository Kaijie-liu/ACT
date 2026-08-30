from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.hybridz_tf.test_binary_phase_split import (
    _verified_duplicate_relu_residual_net,
)
from act.back_end.hybridz_tf.test_operator_micro_rlt import _build
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_enumerate_sparse_binary_phase_cover,
    hz_mark_constructively_nonempty,
    hz_verify_sparse_binary_phase_child,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import (
    _audit_live_operator_property_micro_rlt,
    _audit_sparse_binary_phase_cover,
    _canonical_receipt_sha256,
    verify_once,
)
from act.util.stats import VerifyStatus


def _parent() -> SparseHZono:
    hz = SparseHZono(
        c=np.asarray([0.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, 0), dtype=np.float64),
        Gb=sp.csr_matrix((1, 2), dtype=np.float64),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((0, 0), dtype=np.float64),
        Aub=sp.csr_matrix((0, 2), dtype=np.float64),
        ub=np.zeros(0, dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=np.asarray([31, 47], dtype=np.int64),
    )
    return hz_mark_constructively_nonempty(
        hz, "phase_cover_integrity_toy"
    )


class PhaseCoverIntegrityTests(unittest.TestCase):
    def test_expired_absolute_deadline_stops_before_cover_or_audit(self):
        parent = _parent()
        expired = time.monotonic() - 1.0
        with self.assertRaisesRegex(
            TimeoutError, "phase_cover_entry"
        ):
            hz_enumerate_sparse_binary_phase_cover(
                parent,
                max_children=4,
                deadline=expired,
            )

        cover = hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=4
        )
        with self.assertRaisesRegex(
            TimeoutError, "phase_child_audit_entry"
        ):
            hz_verify_sparse_binary_phase_child(
                parent,
                cover[0][0],
                cover[0][1],
                deadline=expired,
            )

    def test_complete_cover_passes_independent_audit(self) -> None:
        parent = _parent()
        cover = hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=4
        )
        receipt = _audit_sparse_binary_phase_cover(
            parent, cover, phase_depth=2
        )
        self.assertTrue(receipt["proof_authority"])
        self.assertEqual(receipt["expected_child_count"], 4)
        self.assertEqual(receipt["unique_assignment_count"], 4)
        self.assertTrue(receipt["all_children_assignment_bound"])
        self.assertTrue(receipt["all_child_capabilities_valid"])

    def test_incomplete_and_duplicate_covers_fail_closed(self) -> None:
        for mutation, pattern in (
            (lambda cover: cover[:-1], "incomplete"),
            (
                lambda cover: cover[:-1] + (cover[0],),
                "duplicate or incomplete",
            ),
        ):
            with self.subTest(pattern=pattern):
                parent = _parent()
                cover = hz_enumerate_sparse_binary_phase_cover(
                    parent, max_children=4
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    _audit_sparse_binary_phase_cover(
                        parent,
                        mutation(cover),
                        phase_depth=2,
                    )

    def test_assignment_child_swap_and_receipt_tamper_fail_closed(
        self,
    ) -> None:
        parent = _parent()
        cover = hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=4
        )
        swapped = (
            (cover[0][0], cover[1][1]),
            (cover[1][0], cover[0][1]),
            *cover[2:],
        )
        with self.assertRaisesRegex(
            ValueError, "live projection audit|another assignment"
        ):
            _audit_sparse_binary_phase_cover(
                parent, swapped, phase_depth=2
            )

        parent = _parent()
        cover = hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=4
        )
        cover[0][1]._solver_binary_phase_fix = {
            **cover[0][1]._solver_binary_phase_fix,
            "parent_n_bin": 1,
        }
        with self.assertRaisesRegex(
            ValueError, "live projection audit|invalid phase-fix receipt"
        ):
            _audit_sparse_binary_phase_cover(
                parent, cover, phase_depth=2
            )

    def test_missing_private_child_capability_fails_closed(self) -> None:
        parent = _parent()
        cover = hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=4
        )
        delattr(
            cover[2][1], "_solver_exact_phase_cover_member_token"
        )
        with self.assertRaisesRegex(
            ValueError, "live projection audit|capability disagrees"
        ):
            _audit_sparse_binary_phase_cover(
                parent, cover, phase_depth=2
            )

    def test_verifier_returns_unknown_before_child_solve_for_short_cover(
        self,
    ) -> None:
        from act.back_end.solver import solver_hz

        original = solver_hz.hz_enumerate_sparse_binary_phase_cover

        def short_cover(
            parent,
            positions=None,
            *,
            max_children=16,
            deadline=None,
        ):
            return original(
                parent,
                positions=positions,
                max_children=max_children,
                deadline=deadline,
            )[:-1]

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with patch(
                "act.back_end.solver.solver_hz."
                "hz_enumerate_sparse_binary_phase_cover",
                side_effect=short_cover,
            ):
                result = verify_once(
                    _verified_duplicate_relu_residual_net(),
                    backend_cfg=BackendConfig(
                        solver="hybridz",
                        device="cpu",
                        dtype="float64",
                        hybridz=HybridZConfig(
                            timeout=5.0,
                            engine="operator_hz_objbound",
                            operator_exact_budget=1,
                            property_residual_budget=1,
                            property_residual_time_limit=1.0,
                            property_residual_max_adjoint_cells=32,
                            property_residual_pool_per_rival=2,
                            property_tail_upper=True,
                            property_tail_suffix_blocks=1,
                            lp_prefilter_fraction=1.0,
                            lp_prefilter_max_seconds=2.0,
                        ),
                    ),
                )[0]
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata["hz_verdict"], "UNKNOWN")
        receipt = result.metadata["property_phase_split"]
        self.assertEqual(receipt["status"], "invalid_exact_phase_cover")
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["children_run_in_parallel"])
        self.assertEqual(receipt["children"], [])
        self.assertFalse(
            receipt["phase_cover_audit"]["proof_authority"]
        )

    def test_verifier_reports_timeout_for_internal_cover_deadline(
        self,
    ) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with patch(
                "act.back_end.solver.solver_hz."
                "hz_enumerate_sparse_binary_phase_cover",
                side_effect=TimeoutError("controlled cover deadline"),
            ):
                result = verify_once(
                    _verified_duplicate_relu_residual_net(),
                    backend_cfg=BackendConfig(
                        solver="hybridz",
                        device="cpu",
                        dtype="float64",
                        hybridz=HybridZConfig(
                            timeout=5.0,
                            engine="operator_hz_objbound",
                            operator_exact_budget=1,
                            property_residual_budget=1,
                            property_residual_time_limit=1.0,
                            property_residual_max_adjoint_cells=32,
                            property_residual_pool_per_rival=2,
                            property_tail_upper=True,
                            property_tail_suffix_blocks=1,
                            lp_prefilter_fraction=1.0,
                            lp_prefilter_max_seconds=2.0,
                        ),
                    ),
                )[0]
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

        self.assertEqual(result.status, VerifyStatus.TIMEOUT)
        self.assertEqual(
            result.metadata["reason"],
            "shared_deadline_during_phase_cover",
        )
        phase = result.metadata["property_phase_split"]
        self.assertEqual(
            phase["status"], "shared_deadline_during_phase_cover"
        )
        self.assertTrue(phase["phase_cover_audit"]["timed_out"])
        self.assertEqual(
            phase["phase_cover_audit"]["timeout_stage"],
            "enumeration",
        )
        self.assertEqual(phase["children"], [])


class LiveMicroRLTBindingTests(unittest.TestCase):
    def test_live_lift_and_both_receipts_are_cryptographically_bound(
        self,
    ) -> None:
        build = _build(64)
        receipt = build.metadata["property_micro_rlt"]
        self.assertTrue(
            _audit_live_operator_property_micro_rlt(
                build.hz, receipt
            )
        )
        for mode in ("first", "second"):
            with self.subTest(packet_mode=mode):
                directed = _build(
                    64, property_micro_rlt_packet_mode=mode
                )
                self.assertTrue(
                    _audit_live_operator_property_micro_rlt(
                        directed.hz,
                        directed.metadata["property_micro_rlt"],
                    )
                )

        tampered_outer = dict(receipt)
        tampered_outer["new_product_factors"] = (
            int(tampered_outer["new_product_factors"]) + 1
        )
        self.assertFalse(
            _audit_live_operator_property_micro_rlt(
                build.hz, tampered_outer
            )
        )
        for count_key in (
            "required_selected_source_row_nnz",
            "required_product_factors",
        ):
            with self.subTest(rehashed_outer_count=count_key):
                rehashed_outer = dict(receipt)
                rehashed_outer[count_key] = (
                    int(rehashed_outer[count_key]) + 1
                )
                payload = dict(rehashed_outer)
                payload.pop("receipt_sha256")
                rehashed_outer["receipt_sha256"] = (
                    _canonical_receipt_sha256(payload)
                )
                self.assertFalse(
                    _audit_live_operator_property_micro_rlt(
                        build.hz, rehashed_outer
                    )
                )

        rehashed_mode = dict(receipt)
        rehashed_mode["requested_packet_mode"] = "first"
        payload = dict(rehashed_mode)
        payload.pop("receipt_sha256")
        rehashed_mode["receipt_sha256"] = (
            _canonical_receipt_sha256(payload)
        )
        self.assertFalse(
            _audit_live_operator_property_micro_rlt(
                build.hz, rehashed_mode
            )
        )

        build = _build(64)
        receipt = build.metadata["property_micro_rlt"]
        build.hz.ub[-1] = np.nextafter(build.hz.ub[-1], np.inf)
        self.assertFalse(
            _audit_live_operator_property_micro_rlt(
                build.hz, receipt
            )
        )

        build = _build(64)
        receipt = build.metadata["property_micro_rlt"]
        attached = dict(build.hz._property_micro_rlt_receipt)
        attached["new_upper_rows"] = int(
            attached["new_upper_rows"]
        ) + 1
        build.hz._property_micro_rlt_receipt = attached
        self.assertFalse(
            _audit_live_operator_property_micro_rlt(
                build.hz, receipt
            )
        )


if __name__ == "__main__":
    unittest.main()
