#!/usr/bin/env python3
"""Exact signed-support and omission-firewall toys for rival grouping."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import (
    property_phase_literal_groups as grouping,
)
from act.back_end.hybridz_tf.adaptive_phase_forest import RivalSpec
from act.back_end.hybridz_tf.property_phase_literal_groups import (
    PropertyLiteralGroup,
    PropertyLiteralGroupingError,
    derive_property_literal_groups_candidate,
    verify_property_literal_grouping_result,
)
from act.back_end.solver.solver_hz import SparseHZono


def _assert_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _rivals():
    return (
        RivalSpec(10, (1.0, 0.0), 0.0, _assert_digest("r10")),
        RivalSpec(20, (2.0, 0.0), 0.0, _assert_digest("r20")),
        RivalSpec(30, (0.0, 1.0), 0.0, _assert_digest("r30")),
        RivalSpec(40, (0.0, -1.0), 0.0, _assert_digest("r40")),
    )


def _parent() -> SparseHZono:
    # Stable IDs deliberately differ from physical column positions.
    # Columns have effects [+/+, +/- , 0/+].
    return SparseHZono(
        c=np.zeros(2, dtype=np.float64),
        Gc=sp.csr_matrix((2, 0), dtype=np.float64),
        Gb=sp.csr_matrix(
            np.asarray(
                [[1.0, 1.0, 0.0], [1.0, -1.0, 1.0]],
                dtype=np.float64,
            )
        ),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, 3), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((0, 0), dtype=np.float64),
        Aub=sp.csr_matrix((0, 3), dtype=np.float64),
        ub=np.zeros(0, dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=np.asarray([707, 101, 303], dtype=np.int64),
    )


def _scalar_parent(generator: float) -> SparseHZono:
    return SparseHZono(
        c=np.zeros(1, dtype=np.float64),
        Gc=sp.csr_matrix((1, 0), dtype=np.float64),
        Gb=sp.csr_matrix(
            np.asarray([[generator]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, 1), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((0, 0), dtype=np.float64),
        Aub=sp.csr_matrix((0, 1), dtype=np.float64),
        ub=np.zeros(0, dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=np.asarray([42], dtype=np.int64),
    )


def _clone(parent: SparseHZono) -> SparseHZono:
    return SparseHZono(
        c=np.array(parent.c, copy=True),
        Gc=parent.Gc.copy(),
        Gb=parent.Gb.copy(),
        Ac=parent.Ac.copy(),
        Ab=parent.Ab.copy(),
        b=np.array(parent.b, copy=True),
        Auc=parent.Auc.copy(),
        Aub=parent.Aub.copy(),
        ub=np.array(parent.ub, copy=True),
        col_ids=np.array(parent.col_ids, copy=True),
        bcol_ids=np.array(parent.bcol_ids, copy=True),
    )


def _semantic_summary(result):
    return tuple(
        (
            group.group_id,
            tuple(rival.rival_id for rival in group.rivals),
            tuple(
                (literal.stable_bcol_id, literal.phase)
                for literal in group.literals
            ),
            group.omitted_zero_bcol_ids,
            group.eligible_pair_count,
        )
        for group in result.groups
    )


class PropertyPhaseLiteralGroupTests(unittest.TestCase):
    def test_mixed_polarity_and_zero_effect_are_explicit_groups(
        self,
    ) -> None:
        parent = _parent()
        result = derive_property_literal_groups_candidate(
            parent, _rivals()
        )
        self.assertTrue(
            verify_property_literal_grouping_result(
                parent, _rivals(), result
            )
        )
        self.assertFalse(result.proof_authority)
        self.assertEqual(result.receipt["role"], "candidate_selection_only")
        self.assertEqual(len(result.groups), 3)

        shared = next(
            group
            for group in result.groups
            if tuple(rival.rival_id for rival in group.rivals)
            == (10, 20)
        )
        self.assertEqual(
            tuple(
                (literal.stable_bcol_id, literal.phase)
                for literal in shared.literals
            ),
            ((101, 1), (707, 1)),
        )
        self.assertEqual(shared.omitted_zero_bcol_ids, (303,))
        self.assertTrue(shared.cut_eligible)
        self.assertEqual(shared.eligible_pair_count, 1)

        rival_30 = next(
            group
            for group in result.groups
            if tuple(rival.rival_id for rival in group.rivals)
            == (30,)
        )
        self.assertEqual(
            tuple(
                (literal.stable_bcol_id, literal.phase)
                for literal in rival_30.literals
            ),
            ((101, -1), (303, 1), (707, 1)),
        )
        self.assertEqual(rival_30.omitted_zero_bcol_ids, ())

    def test_rival_reorder_preserves_canonical_groups(self) -> None:
        parent = _parent()
        rivals = _rivals()
        first = derive_property_literal_groups_candidate(
            parent, rivals
        )
        second = derive_property_literal_groups_candidate(
            parent, tuple(reversed(rivals))
        )
        self.assertEqual(
            _semantic_summary(first), _semantic_summary(second)
        )
        self.assertNotEqual(
            first.ordered_property_digest,
            second.ordered_property_digest,
        )
        self.assertTrue(
            verify_property_literal_grouping_result(
                parent, tuple(reversed(rivals)), second
            )
        )

    def test_column_permutation_uses_stable_ids(self) -> None:
        parent = _parent()
        first = derive_property_literal_groups_candidate(
            parent, _rivals()
        )
        permutation = np.asarray([2, 0, 1], dtype=np.int64)
        permuted = _clone(parent)
        permuted.Gb = parent.Gb[:, permutation].tocsr()
        permuted.Ab = parent.Ab[:, permutation].tocsr()
        permuted.Aub = parent.Aub[:, permutation].tocsr()
        for matrix in (permuted.Gb, permuted.Ab, permuted.Aub):
            matrix.sum_duplicates()
            matrix.sort_indices()
        permuted.bcol_ids = parent.bcol_ids[permutation]
        second = derive_property_literal_groups_candidate(
            permuted, _rivals()
        )
        self.assertEqual(
            _semantic_summary(first), _semantic_summary(second)
        )
        self.assertNotEqual(
            first.parent_semantic_digest,
            second.parent_semantic_digest,
        )
        # Literal binding includes the live parent, so stale bindings differ.
        self.assertNotEqual(
            first.groups[0].literals[0].binding_digest,
            second.groups[0].literals[0].binding_digest,
        )

    def test_same_count_group_and_receipt_tampering_fails(self) -> None:
        parent = _parent()
        rivals = _rivals()
        result = derive_property_literal_groups_candidate(
            parent, rivals
        )
        first = result.groups[0]
        malformed_group = replace(
            first,
            omitted_zero_bcol_ids=(
                *first.omitted_zero_bcol_ids,
                999,
            ),
        )
        malformed = replace(
            result,
            groups=(malformed_group, *result.groups[1:]),
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, malformed
            )
        )

        receipt = dict(result.receipt)
        receipt["cut_eligible_group_count"] += 1
        receipt_tampered = replace(result, receipt=receipt)
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, receipt_tampered
            )
        )

    def test_parent_effect_and_stable_id_mutations_fail(self) -> None:
        parent = _parent()
        rivals = _rivals()
        result = derive_property_literal_groups_candidate(
            parent, rivals
        )
        changed_effect = _clone(parent)
        changed_effect.Gb.data[0] = np.nextafter(
            changed_effect.Gb.data[0], np.inf
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                changed_effect, rivals, result
            )
        )

        duplicate_ids = _clone(parent)
        duplicate_ids.bcol_ids[1] = duplicate_ids.bcol_ids[0]
        with self.assertRaises(Exception):
            derive_property_literal_groups_candidate(
                duplicate_ids, rivals
            )

    def test_verifier_rejects_recursive_equality_gadgets(
        self,
    ) -> None:
        parent = _parent()
        rivals = _rivals()
        result = derive_property_literal_groups_candidate(
            parent, rivals
        )

        class EvilGroup:
            literals = ()
            eligible_pair_count = 0
            proof_authority = True

            def __eq__(self, other):
                return True

        shrunk = replace(
            result,
            groups=tuple(EvilGroup() for _ in result.groups),
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, shrunk
            )
        )

        class EvilLiteral:
            stable_bcol_id = 999999
            phase = -1
            binding_digest = "0" * 64

            def __eq__(self, other):
                return True

        nested_groups = tuple(
            replace(
                group,
                literals=tuple(
                    EvilLiteral() for _ in group.literals
                ),
            )
            for group in result.groups
        )
        nested = replace(result, groups=nested_groups)
        self.assertTrue(
            all(
                type(group) is PropertyLiteralGroup
                for group in nested.groups
            )
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, nested
            )
        )

        first_group = result.groups[0]
        illegal_literal = replace(
            first_group.literals[0],
            phase=0,
        )
        illegal = replace(
            result,
            groups=(
                replace(
                    first_group,
                    literals=(
                        illegal_literal,
                        *first_group.literals[1:],
                    ),
                ),
                *result.groups[1:],
            ),
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, illegal
            )
        )

        class TruthyEqualityGadget:
            def __bool__(self):
                return True

            def __eq__(self, other):
                return True

        proof_claim = replace(
            result,
            groups=(
                replace(
                    result.groups[0],
                    proof_authority=TruthyEqualityGadget(),
                ),
                *result.groups[1:],
            ),
        )
        self.assertTrue(
            bool(proof_claim.groups[0].proof_authority)
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, proof_claim
            )
        )

        class EvilString(str):
            def __eq__(self, other):
                return True

        receipt = dict(result.receipt)
        receipt["role"] = EvilString(receipt["role"])
        receipt_gadget = replace(result, receipt=receipt)
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, receipt_gadget
            )
        )

    def test_subnormal_opposite_exact_signs_do_not_merge(
        self,
    ) -> None:
        parent = _scalar_parent(0.5)
        tiny = float(
            np.nextafter(np.float64(0.0), np.float64(1.0))
        )
        rivals = (
            RivalSpec(
                1,
                (tiny,),
                0.0,
                _assert_digest("tiny-positive"),
            ),
            RivalSpec(
                2,
                (-tiny,),
                0.0,
                _assert_digest("tiny-negative"),
            ),
        )
        # Ordinary binary64 multiplication underflows both effects to zero.
        self.assertEqual(
            float(np.dot(np.asarray([tiny]), np.asarray([0.5]))),
            0.0,
        )
        self.assertEqual(
            float(np.dot(np.asarray([-tiny]), np.asarray([0.5]))),
            0.0,
        )

        result = derive_property_literal_groups_candidate(
            parent, rivals
        )
        by_rival = {
            group.rivals[0].rival_id: group
            for group in result.groups
        }
        self.assertEqual(set(by_rival), {1, 2})
        self.assertEqual(
            tuple(
                (
                    literal.stable_bcol_id,
                    literal.phase,
                )
                for literal in by_rival[1].literals
            ),
            ((42, 1),),
        )
        self.assertEqual(
            tuple(
                (
                    literal.stable_bcol_id,
                    literal.phase,
                )
                for literal in by_rival[2].literals
            ),
            ((42, -1),),
        )
        self.assertEqual(
            by_rival[1].omitted_zero_bcol_ids, ()
        )
        self.assertEqual(
            by_rival[2].omitted_zero_bcol_ids, ()
        )
        self.assertTrue(
            verify_property_literal_grouping_result(
                parent, rivals, result
            )
        )

    def test_signed_zero_is_omitted_but_bits_remain_bound(
        self,
    ) -> None:
        parent = _scalar_parent(1.0)
        rivals = (
            RivalSpec(
                1,
                (0.0,),
                0.0,
                _assert_digest("zero-positive"),
            ),
            RivalSpec(
                2,
                (-0.0,),
                -0.0,
                _assert_digest("zero-negative"),
            ),
        )
        result = derive_property_literal_groups_candidate(
            parent, rivals
        )
        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group.literals, ())
        self.assertEqual(group.omitted_zero_bcol_ids, (42,))
        self.assertNotEqual(
            group.rival_binding_digests[0],
            group.rival_binding_digests[1],
        )

    def test_rivals_are_canonical_immutable_copies(self) -> None:
        parent = _parent()
        objectives = [
            list(rival.objective) for rival in _rivals()
        ]
        mutable = tuple(
            RivalSpec(
                rival.rival_id,
                objectives[index],
                rival.threshold,
                rival.assert_digest,
            )
            for index, rival in enumerate(_rivals())
        )
        canonical = tuple(
            RivalSpec(
                rival.rival_id,
                tuple(float(value) for value in objective),
                float(rival.threshold),
                rival.assert_digest,
            )
            for rival, objective in zip(_rivals(), objectives)
        )
        result = derive_property_literal_groups_candidate(
            parent, mutable
        )
        self.assertTrue(
            verify_property_literal_grouping_result(
                parent, canonical, result
            )
        )
        for group in result.groups:
            for rival in group.rivals:
                self.assertIs(type(rival.objective), tuple)
                self.assertTrue(
                    all(
                        type(value) is float
                        for value in rival.objective
                    )
                )
        objectives[0][0] = -123.0
        self.assertTrue(
            verify_property_literal_grouping_result(
                parent, canonical, result
            )
        )
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, mutable, result
            )
        )

    def test_receipt_caps_require_independent_expected_caps(
        self,
    ) -> None:
        parent = _parent()
        rivals = _rivals()
        result = derive_property_literal_groups_candidate(
            parent, rivals
        )
        self.assertEqual(
            result.receipt["receipt_integrity"],
            "unkeyed_sha256_diagnostic_not_authentication",
        )
        self.assertEqual(
            result.receipt["caps_binding"],
            "caller_expected_caps_required_by_verifier",
        )
        receipt = dict(result.receipt)
        receipt["caps"] = dict(receipt["caps"])
        receipt["caps"]["max_binaries"] += 1
        payload = dict(receipt)
        del payload["receipt_sha256"]
        receipt["receipt_sha256"] = grouping._canonical_sha256(
            payload
        )
        resealed = replace(result, receipt=receipt)
        self.assertFalse(
            verify_property_literal_grouping_result(
                parent, rivals, resealed
            )
        )

    def test_deadline_expires_fail_closed(self) -> None:
        with patch.object(
            grouping.time,
            "monotonic",
            side_effect=(0.0, 2.0),
        ):
            with self.assertRaisesRegex(
                PropertyLiteralGroupingError,
                "grouping_deadline_expired",
            ):
                derive_property_literal_groups_candidate(
                    _parent(),
                    _rivals(),
                    timeout_seconds=1.0,
                )

    def test_caps_and_duplicate_rivals_fail_closed(self) -> None:
        parent = _parent()
        with self.assertRaises(PropertyLiteralGroupingError):
            derive_property_literal_groups_candidate(
                parent, _rivals(), max_binaries=2
            )
        with self.assertRaises(PropertyLiteralGroupingError):
            derive_property_literal_groups_candidate(
                parent, _rivals(), max_rivals=True
            )
        duplicate = (*_rivals(), replace(_rivals()[0]))
        with self.assertRaises(PropertyLiteralGroupingError):
            derive_property_literal_groups_candidate(
                parent, duplicate
            )


if __name__ == "__main__":
    unittest.main()
