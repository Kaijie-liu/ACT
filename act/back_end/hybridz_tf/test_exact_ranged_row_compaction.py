#!/usr/bin/env python3
"""Toy gates for exact signed-row to ranged-row compaction."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import random
from types import MappingProxyType
import unittest

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_ranged_row_compaction import (
    ExactRangedRowCandidate,
    SignedUpperSource,
    fold_exact_signed_upper_pairs,
    source_and_candidate_membership,
    validate_exact_ranged_candidate,
)


def _csr(rows, shape):
    matrix = sp.csr_matrix(np.asarray(rows, dtype=np.float64), shape=shape)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _source() -> SignedUpperSource:
    return SignedUpperSource(
        A_cont=_csr([[1.0, -2.0], [-1.0, 2.0], [0.5, 0.0]], (3, 2)),
        A_bin=_csr([[3.0], [-3.0], [0.0]], (3, 1)),
        upper=np.asarray([1.25, 0.75, 2.0], dtype=np.float64),
        row_tags=(
            "add_materialize:7:forward",
            "add_materialize:7:reverse",
            "relu_exact_lower:9",
        ),
    )


class ExactRangedRowCompactionTests(unittest.TestCase):
    def test_operator_exact_add_band_folds_without_touching_relu_graph(self) -> None:
        from act.back_end.hybridz_tf.operator_hz import build_operator_hz
        from act.back_end.hybridz_tf.test_operator_add_fusion import (
            OperatorAddFusionAuditTests,
        )

        toy = OperatorAddFusionAuditTests._residual_add_relu_toy()
        built = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=-1,
            materialize_add=True,
            export_verified_preactivation_frame=False,
        )
        hz = built.hz
        self.assertEqual((hz.n_bin, hz.n_ub, hz.constraint_nnz), (1, 5, 11))
        tags = tuple(hz._solver_constraint_row_tags)
        self.assertEqual(len(tags), hz.n_ub)
        self.assertEqual(tags[:2], (
            "add_materialize:3:forward",
            "add_materialize:3:reverse",
        ))
        source = SignedUpperSource(hz.Auc, hz.Aub, hz.ub, tags)
        candidate = fold_exact_signed_upper_pairs(source)
        self.assertTrue(validate_exact_ranged_candidate(source, candidate))
        self.assertEqual(candidate.folded_pairs, ((0, 1),))
        self.assertEqual(candidate.A_cont.shape[0], 4)
        self.assertEqual(candidate.receipt["candidate_constraint_nnz"], 9)
        self.assertEqual(
            candidate.row_tags[1:],
            tags[2:],
            msg="the three exact ReLU facets were altered or folded",
        )

    def test_pair_folds_to_one_exact_range(self) -> None:
        source = _source()
        candidate = fold_exact_signed_upper_pairs(source)
        self.assertTrue(validate_exact_ranged_candidate(source, candidate))
        self.assertEqual(candidate.A_cont.shape, (2, 2))
        self.assertEqual(candidate.A_bin.shape, (2, 1))
        self.assertEqual(candidate.folded_pairs, ((0, 1),))
        self.assertEqual(candidate.source_to_candidate.tolist(), [0, 0, 1])
        self.assertEqual(candidate.lower[0], -0.75)
        self.assertEqual(candidate.upper[0], 1.25)
        self.assertTrue(np.isneginf(candidate.lower[1]))
        self.assertEqual(candidate.upper[1], 2.0)
        self.assertEqual(candidate.row_tags[0], "range:add_materialize:7")
        self.assertEqual(candidate.receipt["source_rows"], 3)
        self.assertEqual(candidate.receipt["candidate_rows"], 2)
        self.assertEqual(candidate.receipt["folded_pair_count"], 1)
        self.assertLess(
            candidate.receipt["candidate_constraint_nnz"],
            candidate.receipt["source_constraint_nnz"],
        )
        self.assertIs(candidate.proof_authority, False)
        self.assertIs(candidate.verdict_authority, False)
        self.assertIs(candidate.production_integration, False)
        json.dumps(dict(candidate.receipt), allow_nan=False, sort_keys=True)

    def test_fraction_membership_is_identical(self) -> None:
        source = _source()
        candidate = fold_exact_signed_upper_pairs(source)
        grid = tuple(Fraction(value, 4) for value in range(-8, 9))
        for x0 in grid:
            for x1 in grid:
                for binary in (Fraction(-1), Fraction(1)):
                    forward = x0 - 2 * x1 + 3 * binary
                    extra = x0 / 2
                    source_ok = (
                        forward <= Fraction(5, 4)
                        and -forward <= Fraction(3, 4)
                        and extra <= Fraction(2)
                    )
                    ranged_ok = (
                        -Fraction(3, 4) <= forward <= Fraction(5, 4)
                        and extra <= Fraction(2)
                    )
                    self.assertEqual(source_ok, ranged_ok)
                    observed = source_and_candidate_membership(
                        source,
                        candidate,
                        np.asarray([float(x0), float(x1)], dtype=np.float64),
                        np.asarray([float(binary)], dtype=np.float64),
                    )
                    self.assertEqual(observed[0], observed[1])
                    self.assertEqual(observed[0], source_ok)

    def test_seeded_dyadic_random_membership(self) -> None:
        rng = random.Random(0xC0A7)
        for case in range(64):
            n_pairs = rng.randint(1, 5)
            n_cont = rng.randint(1, 5)
            n_bin = rng.randint(0, 3)
            continuous_rows = []
            binary_rows = []
            upper = []
            tags = []
            for pair in range(n_pairs):
                continuous = [rng.randint(-4, 4) for _ in range(n_cont)]
                binary = [rng.randint(-3, 3) for _ in range(n_bin)]
                if not any(continuous) and not any(binary):
                    continuous[0] = 1
                continuous_rows.extend((continuous, [-value for value in continuous]))
                binary_rows.extend((binary, [-value for value in binary]))
                upper.extend((rng.randint(-4, 8) / 4.0, rng.randint(-4, 8) / 4.0))
                tags.extend((f"random:{pair}:forward", f"random:{pair}:reverse"))
            continuous = [rng.randint(-4, 4) for _ in range(n_cont)]
            binary = [rng.randint(-3, 3) for _ in range(n_bin)]
            continuous_rows.append(continuous)
            binary_rows.append(binary)
            upper.append(rng.randint(-4, 8) / 4.0)
            tags.append("ordinary")
            source = SignedUpperSource(
                _csr(continuous_rows, (2 * n_pairs + 1, n_cont)),
                _csr(binary_rows, (2 * n_pairs + 1, n_bin)),
                np.asarray(upper, dtype=np.float64),
                tuple(tags),
            )
            candidate = fold_exact_signed_upper_pairs(source)
            self.assertTrue(validate_exact_ranged_candidate(source, candidate))
            self.assertEqual(len(candidate.folded_pairs), n_pairs)
            for _point in range(16):
                continuous_point = np.asarray(
                    [rng.randint(-4, 4) / 4.0 for _ in range(n_cont)],
                    dtype=np.float64,
                )
                binary_point = np.asarray(
                    [rng.choice((-1.0, 1.0)) for _ in range(n_bin)],
                    dtype=np.float64,
                )
                observed = source_and_candidate_membership(
                    source, candidate, continuous_point, binary_point
                )
                self.assertEqual(observed[0], observed[1], msg=f"case={case}")

    def test_ulp_or_nonmatching_tag_never_folds(self) -> None:
        base = _source()
        poisoned = base.A_cont.copy()
        poisoned.data[2] = np.nextafter(poisoned.data[2], -np.inf)
        poisoned_source = SignedUpperSource(
            poisoned,
            base.A_bin,
            base.upper,
            base.row_tags,
        )
        candidate = fold_exact_signed_upper_pairs(poisoned_source)
        self.assertEqual(candidate.folded_pairs, ())
        self.assertEqual(candidate.A_cont.shape[0], 3)

        wrong_tags = SignedUpperSource(
            base.A_cont,
            base.A_bin,
            base.upper,
            (
                "add_materialize:7:forward",
                "add_materialize:8:reverse",
                "relu_exact_lower:9",
            ),
        )
        candidate = fold_exact_signed_upper_pairs(wrong_tags)
        self.assertEqual(candidate.folded_pairs, ())

    def test_multirow_blocks_pair_by_local_order(self) -> None:
        source = SignedUpperSource(
            A_cont=_csr(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [0.0, -2.0],
                ],
                (4, 2),
            ),
            A_bin=_csr([[1.0], [2.0], [-1.0], [-2.0]], (4, 1)),
            upper=np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
            row_tags=(
                "relu_active:4:forward",
                "relu_active:4:forward",
                "relu_active:4:reverse",
                "relu_active:4:reverse",
            ),
        )
        candidate = fold_exact_signed_upper_pairs(source)
        self.assertEqual(candidate.folded_pairs, ((0, 2), (1, 3)))
        self.assertEqual(candidate.A_cont.shape[0], 2)
        self.assertEqual(candidate.source_to_candidate.tolist(), [0, 1, 0, 1])
        self.assertEqual(candidate.lower.tolist(), [-0.75, -1.0])
        self.assertEqual(candidate.upper.tolist(), [0.25, 0.5])
        self.assertTrue(validate_exact_ranged_candidate(source, candidate))

    def test_source_snapshot_and_candidate_verifier_reject_mutation(self) -> None:
        A_cont = _csr([[1.0], [-1.0]], (2, 1))
        A_bin = _csr([[1.0], [-1.0]], (2, 1))
        upper = np.asarray([1.0, 1.0], dtype=np.float64)
        source = SignedUpperSource(
            A_cont,
            A_bin,
            upper,
            ("band:forward", "band:reverse"),
        )
        A_cont.data[:] = 9.0
        A_bin.data[:] = 9.0
        upper[:] = -9.0
        candidate = fold_exact_signed_upper_pairs(source)
        self.assertTrue(validate_exact_ranged_candidate(source, candidate))
        self.assertIsInstance(candidate.receipt, MappingProxyType)
        with self.assertRaises(TypeError):
            candidate.receipt["proof_authority"] = True

        object.__setattr__(candidate, "proof_authority", True)
        self.assertFalse(validate_exact_ranged_candidate(source, candidate))

    def test_receipt_bool_int_type_collisions_are_rejected(self) -> None:
        source = _source()
        candidate = fold_exact_signed_upper_pairs(source)
        poisoned = dict(candidate.receipt)
        poisoned["proof_authority"] = 0
        with self.assertRaises(ValueError):
            ExactRangedRowCandidate(
                candidate.A_cont,
                candidate.A_bin,
                candidate.lower,
                candidate.upper,
                candidate.row_tags,
                candidate.source_to_candidate,
                candidate.folded_pairs,
                poisoned,
                candidate.candidate_sha256,
            )
        clean = fold_exact_signed_upper_pairs(source)
        object.__setattr__(
            clean,
            "source_to_candidate",
            clean.source_to_candidate.astype(np.float64),
        )
        self.assertFalse(validate_exact_ranged_candidate(source, clean))
        clean = fold_exact_signed_upper_pairs(source)
        object.__setattr__(clean, "folded_pairs", ((0.0, 1),))
        self.assertFalse(validate_exact_ranged_candidate(source, clean))
        poisoned = dict(candidate.receipt)
        poisoned["candidate_rows"] = float(poisoned["candidate_rows"])
        with self.assertRaises(ValueError):
            ExactRangedRowCandidate(
                candidate.A_cont,
                candidate.A_bin,
                candidate.lower,
                candidate.upper,
                candidate.row_tags,
                candidate.source_to_candidate,
                candidate.folded_pairs,
                poisoned,
                candidate.candidate_sha256,
            )

    def test_malformed_numeric_frames_fail_closed(self) -> None:
        base = _source()
        with self.assertRaises(ValueError):
            SignedUpperSource(
                base.A_cont.astype(np.float32),
                base.A_bin,
                base.upper,
                base.row_tags,
            )
        bad_upper = base.upper.copy()
        bad_upper[0] = np.nan
        with self.assertRaises(ValueError):
            SignedUpperSource(base.A_cont, base.A_bin, bad_upper, base.row_tags)
        noncanonical = base.A_cont.copy()
        noncanonical.has_sorted_indices = False
        with self.assertRaises(ValueError):
            SignedUpperSource(noncanonical, base.A_bin, base.upper, base.row_tags)
        explicit_zero = base.A_cont.copy()
        explicit_zero.data[0] = 0.0
        with self.assertRaises(ValueError):
            SignedUpperSource(explicit_zero, base.A_bin, base.upper, base.row_tags)

        duplicate = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2, 2, 2], dtype=np.int32),
            ),
            shape=base.A_cont.shape,
        )
        duplicate.has_sorted_indices = True
        duplicate.has_canonical_format = True
        with self.assertRaises(ValueError):
            SignedUpperSource(duplicate, base.A_bin, base.upper, base.row_tags)

    def test_live_candidate_csr_dtype_flags_and_digest_are_replayed(self) -> None:
        source = _source()
        candidate = fold_exact_signed_upper_pairs(source)
        poisoned = candidate.A_cont.copy()
        poisoned.indices = poisoned.indices.astype(np.int64)
        poisoned.indptr = poisoned.indptr.astype(np.int64)
        object.__setattr__(candidate, "A_cont", poisoned)
        self.assertFalse(validate_exact_ranged_candidate(source, candidate))

        candidate = fold_exact_signed_upper_pairs(source)
        poisoned = candidate.A_bin.copy()
        poisoned.has_sorted_indices = False
        poisoned.has_canonical_format = False
        object.__setattr__(candidate, "A_bin", poisoned)
        self.assertFalse(validate_exact_ranged_candidate(source, candidate))

        candidate = fold_exact_signed_upper_pairs(source)
        poisoned = candidate.A_cont.copy()
        poisoned.data[0] = np.nextafter(poisoned.data[0], np.inf)
        poisoned.data.setflags(write=False)
        poisoned.indices.setflags(write=False)
        poisoned.indptr.setflags(write=False)
        object.__setattr__(candidate, "A_cont", poisoned)
        self.assertFalse(validate_exact_ranged_candidate(source, candidate))

    def test_disconnected_authority_and_forbidden_route_firewall(self) -> None:
        source = _source()
        candidate = fold_exact_signed_upper_pairs(source)
        receipt = candidate.receipt
        self.assertIs(receipt["candidate_only"], True)
        self.assertIs(receipt["source_frame_retained_by_candidate"], False)
        self.assertIs(receipt["source_frame_required_for_replay"], True)
        self.assertIs(receipt["provenance_authority"], False)
        self.assertIs(receipt["authenticity_authority"], False)
        self.assertIs(receipt["proof_authority"], False)
        self.assertIs(receipt["verdict_authority"], False)
        self.assertIs(receipt["production_integration"], False)
        for key in (
            "triangle_relaxation_called",
            "branch_and_bound_called",
            "backward_called",
            "dual_called",
            "solver_called",
        ):
            self.assertIs(receipt[key], False)


if __name__ == "__main__":
    unittest.main()
