#!/usr/bin/env python3
"""Controlled unit tests for the isolated V5 authority sidecar."""

from __future__ import annotations

import copy
from dataclasses import replace
import time
import unittest
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_v5_authority as authority


def _sha(character: str) -> str:
    return character * 64


def _binding(
    *,
    frame: str = "2",
    stage_index: int = 0,
    deadline: float | None = None,
) -> authority.V5FrameBinding:
    end = time.monotonic() + 30.0 if deadline is None else float(deadline)
    return authority.V5FrameBinding(
        session_nonce_sha256=_sha("1"),
        frame_nonce_sha256=_sha(frame),
        frame_content_sha256=_sha("3"),
        root_receipt_sha256=_sha("4"),
        parent_chain_sha256=_sha("5"),
        deadline_monotonic_hex=end.hex(),
        stage_kind="TARGET",
        stage_index=stage_index,
        start_lid=7,
    )


def _support(
    owner,
    *,
    layer_id: int = 11,
    operator_kind: str = "DENSE",
    branch: str | None = None,
):
    selected_branch = (
        authority.BRANCH_DENSE if branch is None else branch
    )
    return authority._mint_frame_local_affine_support(
        owner,
        layer_id=layer_id,
        predecessor_id=7,
        operator_kind=operator_kind,
        branch=selected_branch,
        box_semantics="output",
        weight_sha256=_sha("6"),
        geometry_sha256=_sha("7"),
        source_lb_sha256=_sha("8"),
        source_ub_sha256=_sha("9"),
        numeric_platform_sha256=_sha("a"),
        implementation_sha256=_sha("b"),
        maxabs=np.asarray([1.0, 2.0], dtype=np.float64),
        support_s=np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
        box_mass=3.0,
    )


def _dense_key(index: int = 0, *, layer_id: int = 11):
    return authority.AffineExecutionKey(
        execution_index=index,
        layer_id=layer_id,
        predecessor_id=7,
        operator_kind="DENSE",
        branch=authority.BRANCH_DENSE,
    )


class QueryDualV5AuthorityTests(unittest.TestCase):
    def test_protocol_frame_and_branch_are_canonical(self):
        self.assertEqual(
            authority.PROTOCOL_MANIFEST_SHA256,
            authority._json_sha256(authority.PROTOCOL_MANIFEST),
        )
        self.assertEqual(
            authority.PROTOCOL_MANIFEST["numeric_protocol"],
            authority.NUMERIC_PROTOCOL,
        )
        binding = _binding()
        self.assertTrue(authority.validate_frame_binding(binding))
        copied = replace(
            binding, frame_content_sha256=binding.frame_content_sha256
        )
        self.assertTrue(authority.validate_frame_binding(copied))
        object.__setattr__(copied, "binding_sha256", _sha("f"))
        self.assertFalse(authority.validate_frame_binding(copied))

        below = authority.ConvBranchEvidence(
            nonzero_count=1, dense_count=9
        )
        boundary = authority.ConvBranchEvidence(
            nonzero_count=1, dense_count=8
        )
        above = authority.ConvBranchEvidence(
            nonzero_count=2, dense_count=15
        )
        self.assertEqual(
            below.selected_branch, authority.BRANCH_CONV_SPARSE
        )
        self.assertEqual(
            boundary.selected_branch, authority.BRANCH_CONV_SPARSE
        )
        self.assertEqual(
            above.selected_branch, authority.BRANCH_CONV_DENSE
        )
        for evidence in (below, boundary, above):
            self.assertTrue(
                authority.validate_conv_branch_evidence(evidence)
            )
        with self.assertRaises(authority.QueryDualV5AuthorityError):
            authority.ConvBranchEvidence(nonzero_count=True, dense_count=8)
        with self.assertRaises(authority.QueryDualV5AuthorityError):
            authority.ConvBranchEvidence(nonzero_count=9, dense_count=8)
        forged = authority.ConvBranchEvidence(
            nonzero_count=1, dense_count=8
        )
        object.__setattr__(
            forged, "selected_branch", authority.BRANCH_CONV_DENSE
        )
        self.assertFalse(
            authority.validate_conv_branch_evidence(forged)
        )

    def test_frame_local_support_is_immutable_and_copy_loses_authority(self):
        owner = authority._mint_frame_local_support_owner(_binding())
        support = _support(owner)
        self.assertTrue(
            authority.validate_frame_local_affine_support(support)
        )
        self.assertFalse(support.maxabs.flags.writeable)
        self.assertFalse(support.support_s.flags.writeable)
        with self.assertRaises(ValueError):
            support.maxabs.setflags(write=True)
        with self.assertRaises(ValueError):
            support.support_s[0] = 0.0
        with self.assertRaises(TypeError):
            support.receipt["weight_sha256"] = _sha("c")

        copied = copy.copy(support)
        self.assertIsNot(copied, support)
        self.assertFalse(
            authority.validate_frame_local_affine_support(copied)
        )
        rehashed_shape_copy = replace(
            support,
            weight_sha256=_sha("c"),
            content_sha256=_sha("d"),
        )
        self.assertFalse(
            authority.validate_frame_local_affine_support(
                rehashed_shape_copy
            )
        )

        other_owner = authority._mint_frame_local_support_owner(
            _binding(frame="c", stage_index=1)
        )
        with self.assertRaises(authority.QueryDualV5AuthorityError) as caught:
            authority._mint_scalar_guarded_result(
                other_owner,
                _dense_key(),
                support,
                nominal=np.zeros((1, 2), dtype=np.float64),
                scalar_guard=np.zeros(1, dtype=np.float64),
            )
        self.assertEqual(caught.exception.code, "INVALID_RESULT")

        dense_support = _support(owner)
        with self.assertRaises(
            authority.QueryDualV5AuthorityError
        ) as caught:
            authority._mint_guard_ledger(
                owner,
                (
                    authority.GuardExecutionExpectation(
                        key=_dense_key(0),
                        expected_policy=authority.POLICY_SCALAR,
                        expected_support_sha256=(
                            dense_support.content_sha256
                        ),
                    ),
                    authority.GuardExecutionExpectation(
                        key=_dense_key(2),
                        expected_policy=authority.POLICY_SCALAR,
                        expected_support_sha256=(
                            dense_support.content_sha256
                        ),
                    ),
                ),
            )
        self.assertEqual(caught.exception.code, "INVALID_EXPECTATION")

    def test_mixed_policy_ledger_has_complete_exclusive_coverage(self):
        owner = authority._mint_frame_local_support_owner(_binding())
        support = _support(owner)
        dense_key = _dense_key(0)
        branch = authority.ConvBranchEvidence(
            nonzero_count=1, dense_count=8
        )
        conv_key = authority.AffineExecutionKey(
            execution_index=1,
            layer_id=12,
            predecessor_id=7,
            operator_kind="CONV2D",
            branch=branch.selected_branch,
            branch_evidence=branch,
        )
        expectations = (
            authority.GuardExecutionExpectation(
                key=dense_key,
                expected_policy=authority.POLICY_SCALAR,
                expected_support_sha256=support.content_sha256,
            ),
            authority.GuardExecutionExpectation(
                key=conv_key,
                expected_policy=authority.POLICY_COMPONENTWISE,
                expected_support_sha256=None,
            ),
        )
        ledger = authority._mint_guard_ledger(owner, expectations)
        scalar = authority._mint_scalar_guarded_result(
            owner,
            dense_key,
            support,
            nominal=np.asarray([[1.0, -2.0]], dtype=np.float64),
            scalar_guard=np.asarray([0.0], dtype=np.float64),
        )
        componentwise = authority._mint_componentwise_result(
            owner,
            conv_key,
            nominal=np.asarray([[3.0, -4.0]], dtype=np.float64),
            radius=np.asarray([[0.25, 0.5]], dtype=np.float64),
        )
        self.assertTrue(authority.validate_affine_guard_result(scalar))
        self.assertTrue(
            authority.validate_affine_guard_result(componentwise)
        )
        ledger.record(scalar)
        ledger.record(componentwise)
        certificate = ledger.commit()
        self.assertFalse(certificate.proof_authority)
        self.assertTrue(
            authority.validate_guard_ledger_certificate(certificate)
        )
        self.assertEqual(certificate.receipt["execution_count"], 2)
        self.assertEqual(certificate.receipt["scalar_guard_count"], 1)
        self.assertEqual(
            certificate.receipt["componentwise_radius_count"], 1
        )
        copied = copy.copy(certificate)
        self.assertFalse(
            authority.validate_guard_ledger_certificate(copied)
        )

    def test_branch_policy_and_support_binding_fail_closed(self):
        owner = authority._mint_frame_local_support_owner(_binding())
        dense_key = _dense_key()
        conv_dense_evidence = authority.ConvBranchEvidence(
            nonzero_count=2, dense_count=15
        )
        conv_sparse_evidence = authority.ConvBranchEvidence(
            nonzero_count=1, dense_count=8
        )
        conv_dense_key = authority.AffineExecutionKey(
            execution_index=0,
            layer_id=12,
            predecessor_id=7,
            operator_kind="CONV2D",
            branch=conv_dense_evidence.selected_branch,
            branch_evidence=conv_dense_evidence,
        )
        conv_sparse_key = authority.AffineExecutionKey(
            execution_index=0,
            layer_id=12,
            predecessor_id=7,
            operator_kind="CONV2D",
            branch=conv_sparse_evidence.selected_branch,
            branch_evidence=conv_sparse_evidence,
        )

        wrong_expectations = (
            (dense_key, authority.POLICY_COMPONENTWISE, None),
            (conv_dense_key, authority.POLICY_COMPONENTWISE, None),
            (conv_sparse_key, authority.POLICY_SCALAR, _sha("c")),
        )
        for key, policy, support_sha in wrong_expectations:
            with self.subTest(branch=key.branch, policy=policy):
                with self.assertRaises(
                    authority.QueryDualV5AuthorityError
                ) as caught:
                    authority.GuardExecutionExpectation(
                        key=key,
                        expected_policy=policy,
                        expected_support_sha256=support_sha,
                    )
                self.assertEqual(caught.exception.code, "POLICY_MISMATCH")

        with self.assertRaises(authority.QueryDualV5AuthorityError):
            _support(
                owner,
                operator_kind="DENSE",
                branch=authority.BRANCH_CONV_DENSE,
            )
        with self.assertRaises(authority.QueryDualV5AuthorityError):
            _support(
                owner,
                layer_id=12,
                operator_kind="CONV2D",
                branch=authority.BRANCH_CONV_SPARSE,
            )

        conv_support = _support(
            owner,
            layer_id=12,
            operator_kind="CONV2D",
            branch=authority.BRANCH_CONV_DENSE,
        )
        self.assertEqual(
            conv_support.receipt["branch"],
            authority.BRANCH_CONV_DENSE,
        )
        self.assertTrue(
            authority.validate_frame_local_affine_support(conv_support)
        )

        with self.assertRaises(
            authority.QueryDualV5AuthorityError
        ) as caught:
            authority._mint_componentwise_result(
                owner,
                dense_key,
                nominal=np.ones((1, 2), dtype=np.float64),
                radius=np.ones((1, 2), dtype=np.float64),
            )
        self.assertEqual(caught.exception.code, "INVALID_RESULT")
        with self.assertRaises(
            authority.QueryDualV5AuthorityError
        ) as caught:
            authority._mint_componentwise_result(
                owner,
                conv_dense_key,
                nominal=np.ones((1, 2), dtype=np.float64),
                radius=np.ones((1, 2), dtype=np.float64),
            )
        self.assertEqual(caught.exception.code, "INVALID_RESULT")
        with self.assertRaises(
            authority.QueryDualV5AuthorityError
        ) as caught:
            authority._mint_scalar_guarded_result(
                owner,
                conv_sparse_key,
                conv_support,
                nominal=np.ones((1, 2), dtype=np.float64),
                scalar_guard=np.ones(1, dtype=np.float64),
            )
        self.assertEqual(caught.exception.code, "INVALID_RESULT")

    def test_double_charge_missing_and_policy_mismatch_fail_closed(self):
        owner = authority._mint_frame_local_support_owner(_binding())
        support = _support(owner)
        key = _dense_key()
        scalar_expectation = authority.GuardExecutionExpectation(
            key=key,
            expected_policy=authority.POLICY_SCALAR,
            expected_support_sha256=support.content_sha256,
        )
        scalar = authority._mint_scalar_guarded_result(
            owner,
            key,
            support,
            nominal=np.ones((1, 2), dtype=np.float64),
            scalar_guard=np.zeros(1, dtype=np.float64),
        )
        double = authority._mint_guard_ledger(
            owner, (scalar_expectation,)
        )
        double.record(scalar)
        with self.assertRaises(authority.QueryDualV5AuthorityError) as caught:
            double.record(scalar)
        self.assertEqual(caught.exception.code, "DOUBLE_CHARGE")
        with self.assertRaises(authority.QueryDualV5AuthorityError):
            double.commit()

        missing = authority._mint_guard_ledger(
            owner, (scalar_expectation,)
        )
        with self.assertRaises(authority.QueryDualV5AuthorityError) as caught:
            missing.commit()
        self.assertEqual(caught.exception.code, "MISSING_GUARD")

        support_mismatch = authority.GuardExecutionExpectation(
            key=key,
            expected_policy=authority.POLICY_SCALAR,
            expected_support_sha256=_sha("c"),
        )
        mismatch = authority._mint_guard_ledger(
            owner, (support_mismatch,)
        )
        with self.assertRaises(authority.QueryDualV5AuthorityError) as caught:
            mismatch.record(scalar)
        self.assertEqual(caught.exception.code, "POLICY_MISMATCH")

    def test_unknown_execution_rehash_and_result_copy_fail_closed(self):
        owner = authority._mint_frame_local_support_owner(_binding())
        support = _support(owner)
        expected_key = _dense_key(0)
        other_key = _dense_key(1)
        expectation = authority.GuardExecutionExpectation(
            key=expected_key,
            expected_policy=authority.POLICY_SCALAR,
            expected_support_sha256=support.content_sha256,
        )
        other = authority._mint_scalar_guarded_result(
            owner,
            other_key,
            support,
            nominal=np.ones((1, 2), dtype=np.float64),
            scalar_guard=np.ones(1, dtype=np.float64),
        )
        ledger = authority._mint_guard_ledger(owner, (expectation,))
        with self.assertRaises(authority.QueryDualV5AuthorityError) as caught:
            ledger.record(other)
        self.assertEqual(caught.exception.code, "UNEXPECTED_EXECUTION")

        valid = authority._mint_scalar_guarded_result(
            owner,
            expected_key,
            support,
            nominal=np.ones((1, 2), dtype=np.float64),
            scalar_guard=np.ones(1, dtype=np.float64),
        )
        copied = copy.copy(valid)
        self.assertFalse(authority.validate_affine_guard_result(copied))
        forged = replace(valid, trace_sha256=_sha("e"))
        self.assertFalse(authority.validate_affine_guard_result(forged))

    def test_deadline_expiry_never_commits_partial_coverage(self):
        now = time.monotonic()
        with self.assertRaises(authority.QueryDualV5AuthorityError) as caught:
            authority._mint_frame_local_support_owner(
                _binding(deadline=now - 1.0)
            )
        self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")

        binding = _binding(deadline=now + 30.0)
        owner = authority._mint_frame_local_support_owner(binding)
        support = _support(owner)
        key = _dense_key()
        expectation = authority.GuardExecutionExpectation(
            key=key,
            expected_policy=authority.POLICY_SCALAR,
            expected_support_sha256=support.content_sha256,
        )
        result = authority._mint_scalar_guarded_result(
            owner,
            key,
            support,
            nominal=np.ones((1, 2), dtype=np.float64),
            scalar_guard=np.ones(1, dtype=np.float64),
        )
        ledger = authority._mint_guard_ledger(owner, (expectation,))
        ledger.record(result)
        with mock.patch.object(
            authority.time,
            "monotonic",
            return_value=float.fromhex(
                binding.deadline_monotonic_hex
            )
            + 1.0,
        ):
            with self.assertRaises(
                authority.QueryDualV5AuthorityError
            ) as caught:
                ledger.commit()
        self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")
        with self.assertRaises(authority.QueryDualV5AuthorityError):
            ledger.commit()


if __name__ == "__main__":
    unittest.main()
