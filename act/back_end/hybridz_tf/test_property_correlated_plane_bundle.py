#!/usr/bin/env python3
# ===- test_property_correlated_plane_bundle.py - RC-MPH toy gates --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Toy-only exact gates for rival-separable correlated plane bundles."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import math
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from act.back_end.hybridz_tf import property_correlated_plane_bundle as bundle_impl
from act.back_end.hybridz_tf.property_correlated_plane_bundle import (
    AffineUpperPlane,
    PlaneBundleCandidate,
    RivalPlaneBundle,
    SparsePrefixFrame,
    check_bundle_candidate,
    exact_single_plane_support,
    propose_plane_bundle_dual,
    solve_plane_bundles_rival_separable,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _EqualString(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def _frame(*, correlated: bool = True, stable_ids=(101, 202)) -> SparsePrefixFrame:
    # Two materialized DAG paths.  The complete forward/reverse equality band
    # is the only fact proving that both stored variables are the same x.
    if correlated:
        A_eq = sp.csr_matrix([[1.0, -1.0]])
        b_eq = np.zeros(1, dtype=np.float64)
        eq_keys = ("dag_join:p_equals_q",)
    else:
        A_eq = sp.csr_matrix((0, 2), dtype=np.float64)
        b_eq = np.empty(0, dtype=np.float64)
        eq_keys = ()
    return SparsePrefixFrame(
        A_ub=sp.csr_matrix((0, 2), dtype=np.float64),
        b_ub=np.empty(0, dtype=np.float64),
        A_eq=A_eq,
        b_eq=b_eq,
        lb=np.asarray([-1.0, -1.0]),
        ub=np.asarray([1.0, 1.0]),
        stable_var_ids=tuple(stable_ids),
        stable_ub_row_keys=(),
        stable_eq_row_keys=eq_keys,
    )


def _plane(
    frame: SparsePrefixFrame,
    *,
    rival: int,
    plane_id: str,
    coefficients,
    intercept: Fraction,
) -> AffineUpperPlane:
    return AffineUpperPlane(
        plane_id=plane_id,
        rival_id=rival,
        property_digest=_sha(f"property:{rival}"),
        prefix_digest=frame.semantic_digest,
        stop_digest=_sha("ADD(p,q)->RELU"),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercept=float(intercept),
        producer_receipt_digest=_sha(f"producer:{rival}:{plane_id}"),
    )


def _bundles(frame: SparsePrefixFrame):
    # h=ReLU(p+q).  h>=0 and h>=p+q give both legal planes.
    r1 = (
        _plane(
            frame,
            rival=1,
            plane_id="r1:h_ge_0",
            coefficients=(1, 0),
            intercept=Fraction(-1, 4),
        ),
        _plane(
            frame,
            rival=1,
            plane_id="r1:h_ge_s",
            coefficients=(0, -1),
            intercept=Fraction(-1, 4),
        ),
    )
    r2 = (
        _plane(
            frame,
            rival=2,
            plane_id="r2:h_ge_0",
            coefficients=(Fraction(1, 2), 0),
            intercept=Fraction(-1, 8),
        ),
        _plane(
            frame,
            rival=2,
            plane_id="r2:h_ge_s",
            coefficients=(Fraction(-1, 2), -1),
            intercept=Fraction(-1, 8),
        ),
    )
    return tuple(
        RivalPlaneBundle(
            rival_id=planes[0].rival_id,
            property_digest=planes[0].property_digest,
            prefix_digest=frame.semantic_digest,
            planes=planes,
        )
        for planes in (r1, r2)
    )


def _fraction_graph_margin(x: Fraction, rival: int) -> Fraction:
    p = x
    q = x
    h = max(Fraction(0), p + q)
    if rival == 1:
        return p - h - Fraction(1, 4)
    if rival == 2:
        return Fraction(1, 2) * p - h - Fraction(1, 8)
    raise ValueError(rival)


def _primal_bundle_lp(frame: SparsePrefixFrame, bundle: RivalPlaneBundle) -> float:
    # Variables p,q,t.  Each row is t <= a_k z + beta_k.
    A_planes = np.asarray(
        [
            [-plane.coefficients[0], -plane.coefficients[1], 1.0]
            for plane in bundle.planes
        ],
        dtype=np.float64,
    )
    b_planes = np.asarray([plane.intercept for plane in bundle.planes])
    A_prefix = np.hstack(
        [
            frame.A_ub.toarray(),
            np.zeros((frame.A_ub.shape[0], 1), dtype=np.float64),
        ]
    )
    result = linprog(
        np.asarray([0.0, 0.0, -1.0]),
        A_ub=np.vstack([A_planes, A_prefix]),
        b_ub=np.concatenate([b_planes, frame.b_ub]),
        A_eq=np.hstack(
            [
                frame.A_eq.toarray(),
                np.zeros((frame.A_eq.shape[0], 1), dtype=np.float64),
            ]
        ),
        b_eq=frame.b_eq,
        bounds=[(-1.0, 1.0), (-1.0, 1.0), (None, None)],
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun)


def _relu_milp_upper(rival: int, *, correlated: bool = True) -> float:
    # Variables p,q,h,d with h=ReLU(p+q), p,q in [-1,1].
    rows = [
        [1.0, 1.0, -1.0, 0.0],   # h >= p+q
        [0.0, 0.0, -1.0, 0.0],   # h >= 0
        [-1.0, -1.0, 1.0, 2.0],  # h <= p+q+2(1-d)
        [0.0, 0.0, 1.0, -2.0],   # h <= 2d
    ]
    lower = [-np.inf, -np.inf, -np.inf, -np.inf]
    upper = [0.0, 0.0, 2.0, 0.0]
    if correlated:
        rows.extend([[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]])
        lower.extend([-np.inf, -np.inf])
        upper.extend([0.0, 0.0])
    objective = (
        np.asarray([-1.0, 0.0, 1.0, 0.0])
        if rival == 1
        else np.asarray([-0.5, 0.0, 1.0, 0.0])
    )
    constant = Fraction(-1, 4) if rival == 1 else Fraction(-1, 8)
    result = milp(
        objective,
        integrality=np.asarray([0, 0, 0, 1]),
        bounds=Bounds([-1, -1, 0, 0], [1, 1, 2, 1]),
        constraints=LinearConstraint(np.asarray(rows), lower, upper),
        options={"time_limit": 2.0},
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun + float(constant))


class CorrelatedPlaneBundleDecisiveToyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _frame()
        self.r1, self.r2 = _bundles(self.frame)

    def test_all_four_single_planes_are_strictly_positive(self):
        observed = tuple(
            exact_single_plane_support(self.frame, plane)
            for bundle in (self.r1, self.r2)
            for plane in bundle.planes
        )
        self.assertEqual(
            observed,
            (Fraction(3, 4), Fraction(3, 4), Fraction(3, 8), Fraction(11, 8)),
        )
        self.assertTrue(all(value > 0 for value in observed))

    def test_bundle_exactly_crosses_to_minus_quarter_and_eighth(self):
        first = check_bundle_candidate(
            self.frame,
            self.r1,
            propose_plane_bundle_dual(self.frame, self.r1),
        )
        second = check_bundle_candidate(
            self.frame,
            self.r2,
            propose_plane_bundle_dual(self.frame, self.r2),
        )
        self.assertEqual(first.exact_upper, Fraction(-1, 4))
        self.assertEqual(second.exact_upper, Fraction(-1, 8))
        self.assertEqual(
            tuple(Fraction(*pair) for pair in first.exact_plane_weights),
            (Fraction(1, 2), Fraction(1, 2)),
        )
        self.assertEqual(
            tuple(Fraction(*pair) for pair in second.exact_plane_weights),
            (Fraction(3, 4), Fraction(1, 4)),
        )
        self.assertFalse(first.proof_authority)
        self.assertFalse(first.verdict_authority)
        self.assertFalse(first.receipt["candidate_solver_status_has_authority"])
        self.assertFalse(first.receipt["plane_validity_authority"])
        self.assertFalse(first.receipt["prefix_provenance_authority"])

    def test_remove_plane_is_positive_and_k1_bundle_fails_closed(self):
        for bundle in (self.r1, self.r2):
            for plane in bundle.planes:
                self.assertGreater(exact_single_plane_support(self.frame, plane), 0)
            with self.assertRaises(ValueError):
                RivalPlaneBundle(
                    rival_id=bundle.rival_id,
                    property_digest=bundle.property_digest,
                    prefix_digest=bundle.prefix_digest,
                    planes=(bundle.planes[0],),
                )

    def test_delete_join_correlation_keeps_bundles_positive(self):
        loose = _frame(correlated=False)
        loose_r1, loose_r2 = _bundles(loose)
        first = check_bundle_candidate(
            loose, loose_r1, propose_plane_bundle_dual(loose, loose_r1)
        )
        second = check_bundle_candidate(
            loose, loose_r2, propose_plane_bundle_dual(loose, loose_r2)
        )
        self.assertEqual(first.exact_upper, Fraction(3, 4))
        self.assertEqual(second.exact_upper, Fraction(3, 8))

    def test_wrong_stable_id_and_stale_numeric_arrays_fail_closed(self):
        candidate = propose_plane_bundle_dual(self.frame, self.r1)
        wrong = _frame(stable_ids=(101, 303))
        with self.assertRaises(ValueError):
            check_bundle_candidate(wrong, self.r1, candidate)
        tampered_weights = candidate.plane_weights.copy()
        tampered_weights[:] = (1.0, 0.0)
        tampered_weights.setflags(write=False)
        with self.assertRaises(ValueError):
            check_bundle_candidate(
                self.frame,
                self.r1,
                replace(candidate, plane_weights=tampered_weights),
            )

    def test_rival_separable_batch_equals_scalar(self):
        batch = solve_plane_bundles_rival_separable(
            self.frame, (self.r1, self.r2)
        )
        scalar = tuple(
            check_bundle_candidate(
                self.frame,
                bundle,
                propose_plane_bundle_dual(self.frame, bundle),
            ).exact_upper
            for bundle in (self.r1, self.r2)
        )
        self.assertEqual(tuple(item.exact_upper for item in batch.checked), scalar)
        self.assertEqual(scalar, (Fraction(-1, 4), Fraction(-1, 8)))
        self.assertTrue(batch.all_nonpositive)
        self.assertEqual(batch.shared_prefix_scan_count, 1)
        self.assertFalse(batch.receipt["rival_simplex_shared"])
        self.assertTrue(batch.receipt["all_nonpositive_is_diagnostic_only"])
        self.assertFalse(batch.receipt["plane_validity_authority"])
        self.assertFalse(batch.receipt["prefix_provenance_authority"])

    def test_unsafe_rival_prevents_batch_safe_claim(self):
        planes = tuple(
            _plane(
                self.frame,
                rival=3,
                plane_id=f"unsafe:{index}",
                coefficients=(0, 0),
                intercept=Fraction(1, 8),
            )
            for index in range(2)
        )
        unsafe = RivalPlaneBundle(
            rival_id=3,
            property_digest=planes[0].property_digest,
            prefix_digest=self.frame.semantic_digest,
            planes=planes,
        )
        batch = solve_plane_bundles_rival_separable(
            self.frame, (self.r1, unsafe, self.r2)
        )
        self.assertEqual(batch.checked[1].exact_upper, Fraction(1, 8))
        self.assertFalse(batch.all_nonpositive)

    def test_fraction_graph_primal_lp_and_independent_milp_agree(self):
        for rival, bundle, expected in (
            (1, self.r1, Fraction(-1, 4)),
            (2, self.r2, Fraction(-1, 8)),
        ):
            graph_upper = max(
                _fraction_graph_margin(x, rival)
                for x in (Fraction(-1), Fraction(0), Fraction(1))
            )
            self.assertEqual(graph_upper, expected)
            self.assertAlmostEqual(_primal_bundle_lp(self.frame, bundle), float(expected))
            self.assertAlmostEqual(_relu_milp_upper(rival), float(expected))
            self.assertGreater(_relu_milp_upper(rival, correlated=False), 0.0)

    def test_point_width_and_jacobian_oracles(self):
        for x in (
            Fraction(-1),
            Fraction(-1, 2),
            Fraction(0),
            Fraction(1, 2),
            Fraction(1),
        ):
            for rival, bundle in ((1, self.r1), (2, self.r2)):
                margin = _fraction_graph_margin(x, rival)
                values = tuple(
                    _fraction(plane.coefficients[0]) * x
                    + _fraction(plane.coefficients[1]) * x
                    + _fraction(plane.intercept)
                    for plane in bundle.planes
                )
                self.assertLessEqual(margin, min(values))
        self.assertEqual(
            (_fraction_graph_margin(Fraction(-1, 2), 1)
             - _fraction_graph_margin(Fraction(-3, 4), 1)) / Fraction(1, 4),
            Fraction(1),
        )
        self.assertEqual(
            (_fraction_graph_margin(Fraction(3, 4), 1)
             - _fraction_graph_margin(Fraction(1, 2), 1)) / Fraction(1, 4),
            Fraction(-1),
        )
        self.assertEqual(
            (_fraction_graph_margin(Fraction(-1, 2), 2)
             - _fraction_graph_margin(Fraction(-3, 4), 2)) / Fraction(1, 4),
            Fraction(1, 2),
        )
        self.assertEqual(
            (_fraction_graph_margin(Fraction(3, 4), 2)
             - _fraction_graph_margin(Fraction(1, 2), 2)) / Fraction(1, 4),
            Fraction(-3, 2),
        )

        wide_planes = tuple(
            _plane(
                self.frame,
                rival=4,
                plane_id=f"wrong-width:{index}",
                coefficients=(0, 0, 0),
                intercept=Fraction(0),
            )
            for index in range(2)
        )
        wide_bundle = RivalPlaneBundle(
            rival_id=4,
            property_digest=wide_planes[0].property_digest,
            prefix_digest=self.frame.semantic_digest,
            planes=wide_planes,
        )
        with self.assertRaises(ValueError):
            propose_plane_bundle_dual(self.frame, wide_bundle)

    def test_original_ub_mu_and_free_eq_nu_are_replayed(self):
        capped = SparsePrefixFrame(
            A_ub=sp.csr_matrix([[1.0]]),
            b_ub=np.asarray([-0.5]),
            A_eq=sp.csr_matrix((0, 1), dtype=np.float64),
            b_eq=np.empty(0, dtype=np.float64),
            lb=np.asarray([-1.0]),
            ub=np.asarray([1.0]),
            stable_var_ids=(404,),
            stable_ub_row_keys=("cap:x_le_minus_half",),
            stable_eq_row_keys=(),
        )
        capped_planes = tuple(
            _plane(
                capped,
                rival=5,
                plane_id=f"cap:{index}",
                coefficients=(1,),
                intercept=Fraction(0),
            )
            for index in range(2)
        )
        capped_bundle = RivalPlaneBundle(
            rival_id=5,
            property_digest=capped_planes[0].property_digest,
            prefix_digest=capped.semantic_digest,
            planes=capped_planes,
        )
        capped_candidate = propose_plane_bundle_dual(capped, capped_bundle)
        capped_checked = check_bundle_candidate(
            capped, capped_bundle, capped_candidate
        )
        self.assertEqual(capped_checked.exact_upper, Fraction(-1, 2))
        self.assertEqual(
            tuple(Fraction(*pair) for pair in capped_checked.exact_prefix_ub_dual),
            (Fraction(1),),
        )

        source_mu = capped_candidate.prefix_ub_dual.copy()
        copied_candidate = replace(
            capped_candidate, prefix_ub_dual=source_mu
        )
        source_mu[0] = -10.0
        self.assertEqual(copied_candidate.prefix_ub_dual[0], 1.0)

        # The bytes-backed arrays cannot have writes re-enabled.  Even an
        # object.__setattr__ attack after private snapshot validation changes
        # only the external object; a negative mu at entry still fails closed.
        raced_candidate = replace(capped_candidate)
        with self.assertRaises(ValueError):
            raced_candidate.prefix_ub_dual.setflags(write=True)
        original_validate = bundle_impl._validate_candidate

        def validate_then_mutate_external(*args, **kwargs):
            original_validate(*args, **kwargs)
            object.__setattr__(
                raced_candidate,
                "prefix_ub_dual",
                np.asarray([-10.0]),
            )

        with patch.object(
            bundle_impl,
            "_validate_candidate",
            side_effect=validate_then_mutate_external,
        ):
            raced_checked = check_bundle_candidate(
                capped, capped_bundle, raced_candidate
            )
        self.assertEqual(raced_checked.exact_upper, Fraction(-1, 2))
        with self.assertRaises(ValueError):
            check_bundle_candidate(capped, capped_bundle, raced_candidate)
        object.__setattr__(
            raced_candidate,
            "prefix_ub_dual",
            capped_candidate.prefix_ub_dual,
        )
        self.assertEqual(
            check_bundle_candidate(
                capped, capped_bundle, raced_candidate
            ).exact_upper,
            Fraction(-1, 2),
        )

        # Reversing an equality row preserves the prefix set but requires the
        # equality multiplier to change sign; unlike mu, nu must remain free.
        reversed_eq = SparsePrefixFrame(
            A_ub=sp.csr_matrix((0, 2), dtype=np.float64),
            b_ub=np.empty(0, dtype=np.float64),
            A_eq=sp.csr_matrix([[-1.0, 1.0]]),
            b_eq=np.zeros(1, dtype=np.float64),
            lb=np.asarray([-1.0, -1.0]),
            ub=np.asarray([1.0, 1.0]),
            stable_var_ids=(101, 202),
            stable_ub_row_keys=(),
            stable_eq_row_keys=("dag_join:q_minus_p",),
        )
        reversed_r1, _ = _bundles(reversed_eq)
        reversed_candidate = propose_plane_bundle_dual(reversed_eq, reversed_r1)
        reversed_checked = check_bundle_candidate(
            reversed_eq, reversed_r1, reversed_candidate
        )
        self.assertEqual(reversed_checked.exact_upper, Fraction(-1, 4))
        self.assertLess(reversed_candidate.prefix_eq_dual[0], 0.0)

    def test_private_snapshots_block_validate_then_public_numeric_mutation(self):
        capped = SparsePrefixFrame(
            A_ub=sp.csr_matrix([[1.0]]),
            b_ub=np.asarray([-0.5]),
            A_eq=sp.csr_matrix((0, 1), dtype=np.float64),
            b_eq=np.empty(0, dtype=np.float64),
            lb=np.asarray([-1.0]),
            ub=np.asarray([1.0]),
            stable_var_ids=(707,),
            stable_ub_row_keys=("cap",),
            stable_eq_row_keys=(),
        )
        planes = tuple(
            _plane(
                capped,
                rival=7,
                plane_id=f"snapshot:{index}",
                coefficients=(1,),
                intercept=Fraction(0),
            )
            for index in range(2)
        )
        bundle = RivalPlaneBundle(
            rival_id=7,
            property_digest=planes[0].property_digest,
            prefix_digest=capped.semantic_digest,
            planes=planes,
        )
        candidate = propose_plane_bundle_dual(capped, bundle)
        original_validate = bundle_impl._validate_candidate

        attacks = []
        original_ub = capped.ub
        attacks.append(
            (
                lambda: object.__setattr__(capped, "ub", np.asarray([-1.0])),
                lambda: object.__setattr__(capped, "ub", original_ub),
            )
        )
        original_A_data = capped.A_ub.data
        attacks.append(
            (
                lambda: setattr(capped.A_ub, "data", np.asarray([-1.0])),
                lambda: setattr(capped.A_ub, "data", original_A_data),
            )
        )
        original_coefficients = planes[0].coefficients
        attacks.append(
            (
                lambda: object.__setattr__(
                    planes[0], "coefficients", np.asarray([-1.0])
                ),
                lambda: object.__setattr__(
                    planes[0], "coefficients", original_coefficients
                ),
            )
        )

        for attack, restore in attacks:
            def validate_then_attack(*args, **kwargs):
                original_validate(*args, **kwargs)
                attack()

            try:
                with patch.object(
                    bundle_impl,
                    "_validate_candidate",
                    side_effect=validate_then_attack,
                ):
                    checked = check_bundle_candidate(capped, bundle, candidate)
                self.assertEqual(checked.exact_upper, Fraction(-1, 2))
            finally:
                restore()

    def test_output_authority_and_recursive_immutability_fail_closed(self):
        checked = check_bundle_candidate(
            self.frame,
            self.r1,
            propose_plane_bundle_dual(self.frame, self.r1),
        )
        batch = solve_plane_bundles_rival_separable(
            self.frame, (self.r1, self.r2)
        )
        with self.assertRaises(ValueError):
            replace(checked, proof_authority=True)
        with self.assertRaises(ValueError):
            replace(batch, verdict_authority=True)
        with self.assertRaises(TypeError):
            checked.receipt["proof_authority"] = True
        with self.assertRaises(TypeError):
            batch.receipt["ordered_rival_ids"][0] = 99
        with self.assertRaises(ValueError):
            batch.candidates[0].plane_weights.setflags(write=True)

    def test_every_public_schema_and_output_binding_fails_closed(self):
        candidate = propose_plane_bundle_dual(self.frame, self.r1)
        checked = check_bundle_candidate(self.frame, self.r1, candidate)
        batch = solve_plane_bundles_rival_separable(
            self.frame, (self.r1, self.r2)
        )
        public_objects = (
            self.frame,
            self.r1.planes[0],
            self.r1,
            candidate,
            checked,
            batch,
        )
        for value in public_objects:
            with self.assertRaises(ValueError):
                replace(value, schema="act.wrong.v1")
            with self.assertRaises(ValueError):
                replace(value, schema=_EqualString(value.schema))

        with self.assertRaises(ValueError):
            replace(
                self.frame,
                semantic_digest=_EqualString(self.frame.semantic_digest),
            )
        with self.assertRaises(ValueError):
            replace(
                self.r1.planes[0],
                plane_digest=_EqualString(self.r1.planes[0].plane_digest),
            )
        with self.assertRaises(ValueError):
            replace(
                self.r1,
                bundle_digest=_EqualString(self.r1.bundle_digest),
            )

        with self.assertRaises(ValueError):
            replace(candidate, candidate_support=candidate.candidate_support + 1.0)
        with self.assertRaises(ValueError):
            replace(checked, exact_denominator=0)
        with self.assertRaises(ValueError):
            replace(checked, exact_numerator=checked.exact_numerator + 1)
        with self.assertRaises(ValueError):
            replace(checked, outward_upper=math.inf)
        with self.assertRaises(ValueError):
            replace(
                checked,
                exact_plane_weights=((1, 1), (1, 1)),
            )

        extra_receipt = dict(checked.receipt)
        extra_receipt["unexpected"] = False
        extra_receipt.pop("receipt_sha256")
        extra_receipt["receipt_sha256"] = bundle_impl._digest(extra_receipt)
        with self.assertRaises(ValueError):
            replace(checked, receipt=extra_receipt)

        missing_receipt = dict(checked.receipt)
        missing_receipt.pop("checker_source")
        missing_receipt.pop("receipt_sha256")
        missing_receipt["receipt_sha256"] = bundle_impl._digest(missing_receipt)
        with self.assertRaises(ValueError):
            replace(checked, receipt=missing_receipt)

        invalid_sha_receipt = dict(checked.receipt)
        invalid_sha_receipt["receipt_sha256"] = "g" * 64
        with self.assertRaises(ValueError):
            replace(checked, receipt=invalid_sha_receipt)

        bool_rival_receipt = dict(checked.receipt)
        bool_rival_receipt["rival_id"] = True
        bool_rival_receipt.pop("receipt_sha256")
        bool_rival_receipt["receipt_sha256"] = bundle_impl._digest(
            bool_rival_receipt
        )
        with self.assertRaises(ValueError):
            replace(checked, receipt=bool_rival_receipt)

        wrong_nominal = dict(checked.receipt)
        wrong_nominal["longdouble_nominal"] = "123"
        wrong_nominal.pop("receipt_sha256")
        wrong_nominal["receipt_sha256"] = bundle_impl._digest(wrong_nominal)
        with self.assertRaises(ValueError):
            replace(checked, receipt=wrong_nominal)

        wrong_batch_receipt = dict(batch.receipt)
        wrong_batch_receipt["ordered_candidate_digests"] = tuple(reversed(
            wrong_batch_receipt["ordered_candidate_digests"]
        ))
        wrong_batch_receipt.pop("receipt_sha256")
        wrong_batch_receipt["receipt_sha256"] = bundle_impl._digest(
            wrong_batch_receipt
        )
        with self.assertRaises(ValueError):
            replace(batch, receipt=wrong_batch_receipt)
        with self.assertRaises(ValueError):
            replace(batch, all_nonpositive=False)

        for key, poisoned in (
            ("ordered_rival_ids", (True, 2)),
            ("shared_prefix_scan_count", True),
        ):
            bool_batch_receipt = dict(batch.receipt)
            bool_batch_receipt[key] = poisoned
            bool_batch_receipt.pop("receipt_sha256")
            bool_batch_receipt["receipt_sha256"] = bundle_impl._digest(
                bool_batch_receipt
            )
            with self.assertRaises(ValueError):
                replace(batch, receipt=bool_batch_receipt)

        attacked_candidate = batch.candidates[0]
        object.__setattr__(attacked_candidate, "proof_authority", True)
        try:
            with self.assertRaises(ValueError):
                replace(
                    batch,
                    candidates=(attacked_candidate, batch.candidates[1]),
                )
        finally:
            object.__setattr__(attacked_candidate, "proof_authority", False)

    def test_snapshot_rejects_identity_coercion_under_an_old_digest(self):
        poisoned_ids = (
            np.asarray([101.5, 202.0]),
            np.asarray([True, False]),
            np.asarray([101, 202], dtype=np.int64),
        )
        for values in poisoned_ids:
            frame = _frame()
            bundle, _ = _bundles(frame)
            object.__setattr__(frame, "stable_var_ids", values)
            with self.assertRaises(ValueError):
                propose_plane_bundle_dual(frame, bundle)

        frame = _frame()
        bundle, _ = _bundles(frame)
        object.__setattr__(frame, "schema", _EqualString(frame.schema))
        with self.assertRaises(ValueError):
            propose_plane_bundle_dual(frame, bundle)

        frame = _frame()
        bundle, _ = _bundles(frame)
        object.__setattr__(
            frame,
            "semantic_digest",
            _EqualString(frame.semantic_digest),
        )
        with self.assertRaises(ValueError):
            propose_plane_bundle_dual(frame, bundle)

        frame = _frame()
        bundle, _ = _bundles(frame)
        object.__setattr__(
            frame,
            "stable_eq_row_keys",
            (_EqualString(frame.stable_eq_row_keys[0]),),
        )
        with self.assertRaises(ValueError):
            propose_plane_bundle_dual(frame, bundle)

        for field, replacement in (
            ("ub", _frame().ub.copy()),
            ("A_eq", sp.csr_matrix(_frame().A_eq, copy=True)),
        ):
            frame = _frame()
            bundle, _ = _bundles(frame)
            object.__setattr__(frame, field, replacement)
            with self.assertRaises(ValueError):
                propose_plane_bundle_dual(frame, bundle)

        frame = _frame()
        bundle, _ = _bundles(frame)
        plane = bundle.planes[0]
        object.__setattr__(plane, "schema", _EqualString(plane.schema))
        with self.assertRaises(ValueError):
            propose_plane_bundle_dual(frame, bundle)

        frame = _frame()
        bundle, _ = _bundles(frame)
        candidate = propose_plane_bundle_dual(frame, bundle)
        object.__setattr__(
            candidate, "schema", _EqualString(candidate.schema)
        )
        with self.assertRaises(ValueError):
            check_bundle_candidate(frame, bundle, candidate)

    def test_outward_rounding_overflow_fails_closed(self):
        maximum = float(np.finfo(np.float64).max)
        exact_maximum = Fraction(*maximum.as_integer_ratio())
        self.assertEqual(
            bundle_impl._outward_binary64(exact_maximum), maximum
        )
        self.assertEqual(
            bundle_impl._outward_binary64(exact_maximum - 1), maximum
        )
        with self.assertRaises(OverflowError):
            bundle_impl._outward_binary64(exact_maximum + 1)

    def test_sparse_lp_builder_avoids_dense_prefix_copy_and_scales_linearly(self):
        width = 20_000
        A_ub = sp.csr_matrix(
            (np.asarray([1.0]), (np.asarray([0]), np.asarray([0]))),
            shape=(1, width),
        )
        A_eq = sp.csr_matrix(
            (np.asarray([1.0]), (np.asarray([0]), np.asarray([width - 1]))),
            shape=(1, width),
        )
        frame = SparsePrefixFrame(
            A_ub=A_ub,
            b_ub=np.asarray([1.0]),
            A_eq=A_eq,
            b_eq=np.asarray([0.0]),
            lb=np.full(width, -1.0),
            ub=np.full(width, 1.0),
            stable_var_ids=tuple(range(width)),
            stable_ub_row_keys=("ub:0",),
            stable_eq_row_keys=("eq:0",),
        )
        coefficients = np.zeros(width, dtype=np.float64)
        coefficients[width // 2] = 1.0
        planes = tuple(
            _plane(
                frame,
                rival=6,
                plane_id=f"medium:{index}",
                coefficients=coefficients,
                intercept=Fraction(index, 8),
            )
            for index in range(2)
        )
        bundle = RivalPlaneBundle(
            rival_id=6,
            property_digest=planes[0].property_digest,
            prefix_digest=frame.semantic_digest,
            planes=planes,
        )
        with patch.object(
            sp.csr_matrix,
            "toarray",
            side_effect=AssertionError("dense prefix copy"),
        ), patch.object(
            bundle_impl.np,
            "vstack",
            side_effect=AssertionError("dense stack"),
        ), patch.object(
            bundle_impl.np,
            "hstack",
            side_effect=AssertionError("dense stack"),
        ):
            objective, equality, rhs, bounds = bundle_impl._build_candidate_lp(
                frame, bundle
            )
        self.assertTrue(sp.isspmatrix_csr(equality))
        self.assertEqual(equality.shape, (width + 1, 2 + 1 + 1 + 2 * width))
        self.assertEqual(objective.size, equality.shape[1])
        self.assertEqual(rhs.size, equality.shape[0])
        self.assertEqual(len(bounds), equality.shape[1])
        self.assertLessEqual(equality.nnz, 2 * width + 8)

    def test_stable_identity_inputs_reject_silent_coercion(self):
        kwargs = dict(
            A_ub=sp.csr_matrix((0, 1), dtype=np.float64),
            b_ub=np.empty(0, dtype=np.float64),
            A_eq=sp.csr_matrix((0, 1), dtype=np.float64),
            b_eq=np.empty(0, dtype=np.float64),
            lb=np.asarray([-1.0]),
            ub=np.asarray([1.0]),
            stable_ub_row_keys=(),
            stable_eq_row_keys=(),
        )
        for unsafe_ids in ((1.5,), (True,), (np.int64(1),)):
            with self.assertRaises(ValueError):
                SparsePrefixFrame(stable_var_ids=unsafe_ids, **kwargs)
        with self.assertRaises(ValueError):
            SparsePrefixFrame(
                stable_var_ids=(1,),
                stable_ub_row_keys=(object(),),
                **{key: value for key, value in kwargs.items()
                   if key != "stable_ub_row_keys"},
            )


class CorrelatedPlaneBundleSeededTests(unittest.TestCase):
    def test_small_seeded_fraction_outward_oracle(self):
        rng = np.random.default_rng(0x20260809)
        for _ in range(256):
            exact = Fraction(
                int(rng.integers(-1_000_000, 1_000_001)),
                int(rng.integers(1, 1_000_001)),
            )
            outward = bundle_impl._outward_binary64(exact)
            self.assertTrue(math.isfinite(outward))
            self.assertGreaterEqual(Fraction(*outward.as_integer_ratio()), exact)
            previous = math.nextafter(outward, -math.inf)
            self.assertLess(Fraction(*previous.as_integer_ratio()), exact)

    def test_small_seeded_dyadic_fraction_fuzz(self):
        rng = np.random.default_rng(20260809)
        frame = _frame()
        for case in range(32):
            alpha = Fraction(int(rng.integers(1, 16)), 8)
            if alpha >= 2:
                alpha = Fraction(15, 8)
            # Keep both individual planes strictly positive while the exact
            # bundle crosses zero.  alpha and 2-alpha are at least 1/8.
            threshold = Fraction(int(rng.integers(1, 4)), 64)
            rival = 100 + case
            planes = (
                _plane(
                    frame,
                    rival=rival,
                    plane_id=f"{case}:zero",
                    coefficients=(alpha, 0),
                    intercept=-threshold,
                ),
                _plane(
                    frame,
                    rival=rival,
                    plane_id=f"{case}:identity",
                    coefficients=(alpha - 1, -1),
                    intercept=-threshold,
                ),
            )
            bundle = RivalPlaneBundle(
                rival_id=rival,
                property_digest=planes[0].property_digest,
                prefix_digest=frame.semantic_digest,
                planes=planes,
            )
            checked = check_bundle_candidate(
                frame, bundle, propose_plane_bundle_dual(frame, bundle)
            )
            self.assertEqual(checked.exact_upper, -threshold)
            exact_graph = max(
                alpha * x - max(Fraction(0), 2 * x) - threshold
                for x in (Fraction(-1), Fraction(0), Fraction(1))
            )
            self.assertEqual(exact_graph, -threshold)
            self.assertGreater(exact_single_plane_support(frame, planes[0]), 0)
            self.assertGreater(exact_single_plane_support(frame, planes[1]), 0)


def _fraction(value) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


if __name__ == "__main__":
    unittest.main()
