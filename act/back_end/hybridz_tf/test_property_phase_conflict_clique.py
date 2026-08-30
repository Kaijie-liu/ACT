#!/usr/bin/env python3
"""Exact controlled-toy gates for candidate-only PC-PCC."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from act.back_end.hybridz_tf import (
    property_phase_conflict_clique as pc_pcc,
)
from act.back_end.hybridz_tf.adaptive_phase_forest import RivalSpec
from act.back_end.hybridz_tf.property_micro_rlt import (
    apply_property_micro_rlt,
    verify_property_micro_rlt_result,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PCPCCError,
    PCPCCResult,
    PhaseLiteral,
    run_pc_pcc_candidate,
    verify_exact_conflict_certificate,
    verify_pc_pcc_receipt,
    verify_pc_pcc_result,
    verify_pc_pcc_structural_result,
)
from act.back_end.solver.solver_hz import SparseHZono, hz_fresh_col_ids


_THRESHOLD = Fraction(9, 8)


def _edges(
    n_binary: int,
    missing=(),
):
    missing_set = {
        tuple(sorted((int(left), int(right))))
        for left, right in missing
    }
    return tuple(
        edge
        for edge in itertools.combinations(range(n_binary), 2)
        if edge not in missing_set
    )


def _stable_set_hz(
    n_binary: int,
    *,
    missing=(),
    phases=None,
) -> SparseHZono:
    if phases is None:
        phases = (1,) * n_binary
    phases = tuple(int(value) for value in phases)
    if (
        len(phases) != n_binary
        or any(value not in {-1, 1} for value in phases)
    ):
        raise ValueError("stable-set phases must be +/-1")
    graph_edges = _edges(n_binary, missing)
    rows = np.zeros(
        (len(graph_edges), n_binary), dtype=np.float64
    )
    for row, (left, right) in enumerate(graph_edges):
        # z_i=(1+p_i*s_i)/2, so z_l+z_r<=1 is
        # p_l*s_l+p_r*s_r<=0 for arbitrary signed literals.
        rows[row, left] = float(phases[left])
        rows[row, right] = float(phases[right])
    stable_ids = (
        hz_fresh_col_ids(n_binary, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return SparseHZono(
        c=np.asarray([n_binary / 2.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, 0), dtype=np.float64),
        Gb=sp.csr_matrix(
            0.5
            * np.asarray(phases, dtype=np.float64).reshape(1, -1)
        ),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, n_binary), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            (len(graph_edges), 0), dtype=np.float64
        ),
        Aub=sp.csr_matrix(rows, dtype=np.float64),
        ub=np.zeros(len(graph_edges), dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=stable_ids,
    )


def _two_row_continuous_cancellation_hz() -> SparseHZono:
    """x+s1<=0 and -x+s2<=0; only their sum conflicts at +/+."""

    continuous_id = (
        hz_fresh_col_ids(1, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    binary_ids = (
        hz_fresh_col_ids(2, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return SparseHZono(
        c=np.asarray([1.0], dtype=np.float64),
        Gc=sp.csr_matrix(np.zeros((1, 1), dtype=np.float64)),
        Gb=sp.csr_matrix(
            np.asarray([[0.5, 0.5]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, 1), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            np.asarray([[1.0], [-1.0]], dtype=np.float64)
        ),
        Aub=sp.csr_matrix(
            np.asarray(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            )
        ),
        ub=np.zeros(2, dtype=np.float64),
        col_ids=continuous_id,
        bcol_ids=binary_ids,
    )


def _complete_c49(
    n_binary: int,
    *,
    missing=(),
) -> SparseHZono:
    base = _stable_set_hz(n_binary, missing=missing)
    source_rows = tuple(range(base.n_ub))
    result = apply_property_micro_rlt(
        base,
        source_rows_by_binary={
            binary: source_rows for binary in range(n_binary)
        },
        max_binary_factors=n_binary,
        max_source_rows_per_binary=len(source_rows),
        max_product_factors=n_binary * n_binary,
        max_selected_row_nnz=(
            4 * n_binary * max(1, len(source_rows))
        ),
        max_requirement_scan_nnz=(
            4 * n_binary * max(1, len(source_rows))
        ),
    )
    if not verify_property_micro_rlt_result(result):
        raise AssertionError("complete C49 live audit failed")
    return result.hz


def _rivals() -> tuple[RivalSpec, ...]:
    scales = (Fraction(1), Fraction(3, 4), Fraction(1, 2))
    ids = (101, 503, 907)
    return tuple(
        RivalSpec(
            rival_id=rival_id,
            objective=(float(scale),),
            threshold=float(scale * _THRESHOLD),
            assert_digest=hashlib.sha256(
                f"pc-pcc-raw-assert-{rival_id}".encode("ascii")
            ).hexdigest(),
        )
        for rival_id, scale in zip(ids, scales)
    )


def _lp_upper(hz: SparseHZono) -> float:
    objective = np.concatenate(
        [
            np.asarray(hz.Gc.getrow(0).toarray()).reshape(-1),
            np.asarray(hz.Gb.getrow(0).toarray()).reshape(-1),
        ]
    )
    result = linprog(
        -objective,
        A_ub=sp.hstack([hz.Auc, hz.Aub], format="csr"),
        b_ub=hz.ub,
        A_eq=sp.hstack([hz.Ac, hz.Ab], format="csr"),
        b_eq=hz.b,
        bounds=[(-1.0, 1.0)] * (hz.n_cont + hz.n_bin),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(hz.c[0] - result.fun)


def _fraction_integer_upper(
    n_binary: int,
    *,
    missing=(),
) -> Fraction:
    graph_edges = _edges(n_binary, missing)
    best = Fraction(0)
    for active in itertools.product((0, 1), repeat=n_binary):
        if any(
            active[left] + active[right] > 1
            for left, right in graph_edges
        ):
            continue
        best = max(best, sum(active, 0))
    return best


def _milp_upper(hz: SparseHZono) -> float:
    # SciPy's integer variables with bounds [-1,1] also admit zero.  Convert
    # every exact HZ binary s in {-1,+1} to z=(s+1)/2 in {0,1}.
    binary_ones = np.ones(hz.n_bin, dtype=np.float64)
    objective = np.concatenate(
        [
            np.asarray(hz.Gc.getrow(0).toarray()).reshape(-1),
            2.0
            * np.asarray(
                hz.Gb.getrow(0).toarray()
            ).reshape(-1),
        ]
    )
    constant = float(
        hz.c[0]
        - np.asarray(
            hz.Gb.getrow(0).toarray()
        ).reshape(-1)
        @ binary_ones
    )
    matrix = sp.vstack(
        [
            sp.hstack([hz.Ac, 2.0 * hz.Ab], format="csr"),
            sp.hstack([hz.Auc, 2.0 * hz.Aub], format="csr"),
        ],
        format="csr",
    )
    lower = np.concatenate(
        [
            hz.b
            + np.asarray(hz.Ab @ binary_ones).reshape(-1),
            np.full(hz.n_ub, -np.inf, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            hz.b
            + np.asarray(hz.Ab @ binary_ones).reshape(-1),
            hz.ub
            + np.asarray(hz.Aub @ binary_ones).reshape(-1),
        ]
    )
    result = milp(
        -objective,
        integrality=np.concatenate(
            [
                np.zeros(hz.n_cont, dtype=np.int8),
                np.ones(hz.n_bin, dtype=np.int8),
            ]
        ),
        bounds=Bounds(
            np.concatenate(
                [
                    np.full(hz.n_cont, -1.0),
                    np.zeros(hz.n_bin),
                ]
            ),
            np.full(hz.n_cont + hz.n_bin, 1.0),
        ),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 5.0},
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(constant - result.fun)


def _clone_hz(hz: SparseHZono) -> SparseHZono:
    result = SparseHZono(
        c=np.array(hz.c, dtype=np.float64, copy=True),
        Gc=hz.Gc.copy(),
        Gb=hz.Gb.copy(),
        Ac=hz.Ac.copy(),
        Ab=hz.Ab.copy(),
        b=np.array(hz.b, dtype=np.float64, copy=True),
        Auc=hz.Auc.copy(),
        Aub=hz.Aub.copy(),
        ub=np.array(hz.ub, dtype=np.float64, copy=True),
        col_ids=np.array(hz.col_ids, dtype=np.int64, copy=True),
        bcol_ids=np.array(hz.bcol_ids, dtype=np.int64, copy=True),
    )
    for name, value in vars(hz).items():
        if "conditional" in name.lower():
            setattr(result, name, value)
    return result


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _reseal_receipt(receipt):
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def _reseal_certificate(certificate):
    placeholder = replace(certificate, certificate_digest="")
    return replace(
        placeholder,
        certificate_digest=pc_pcc._certificate_digest(placeholder),
    )


class PCPCCExactToyTests(unittest.TestCase):
    def test_mixed_signed_literals_emit_exact_clique_row(self) -> None:
        phases = (1, -1, 1, -1)
        parent = _stable_set_hz(4, phases=phases)
        rivals = _rivals()
        self.assertAlmostEqual(_lp_upper(parent), 2.0, places=10)
        self.assertAlmostEqual(_milp_upper(parent), 1.0, places=10)

        result = run_pc_pcc_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(
            tuple(literal.phase for literal in result.literals),
            phases,
        )
        self.assertEqual(len(result.certificates), 6)
        live_row = np.asarray(
            result.hz.Aub.getrow(parent.n_ub).toarray()
        ).reshape(-1)
        np.testing.assert_array_equal(
            live_row, np.asarray(phases, dtype=np.float64)
        )
        self.assertEqual(
            float(result.hz.ub[parent.n_ub]), -2.0
        )
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0, places=10)
        self.assertTrue(
            verify_pc_pcc_result(parent, rivals, result)
        )

    def test_k4_exact_discriminator_unknown_to_safe(self) -> None:
        ordinary = _stable_set_hz(4)
        lifted = _complete_c49(4)
        self.assertAlmostEqual(_lp_upper(ordinary), 2.0, places=10)
        self.assertAlmostEqual(
            _lp_upper(lifted), 4.0 / 3.0, places=10
        )
        self.assertEqual(_fraction_integer_upper(4), Fraction(1))
        self.assertAlmostEqual(_milp_upper(lifted), 1.0, places=10)

        rivals = _rivals()
        result = run_pc_pcc_candidate(
            lifted,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(result.status, "unknown_to_safe_candidate")
        self.assertFalse(result.proof_authority)
        self.assertIsNotNone(result.hz)
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0, places=10)
        self.assertLess(_lp_upper(result.hz), float(_THRESHOLD))
        self.assertTrue(
            verify_pc_pcc_result(lifted, rivals, result)
        )
        self.assertFalse(
            verify_pc_pcc_result(lifted, rivals, result)
        )
        self.assertTrue(
            verify_pc_pcc_structural_result(
                lifted, rivals, result
            )
        )

        receipt = result.receipt
        self.assertEqual(receipt["expected_pairs"], 6)
        self.assertEqual(receipt["highs_calls"], 6)
        self.assertEqual(receipt["certified_conflict_edges"], 6)
        self.assertEqual(receipt["cut_rows_added"], 1)
        self.assertEqual(receipt["phase_children_minted"], 0)
        self.assertFalse(receipt["branching_used"])
        self.assertTrue(receipt["closure"]["complete"])
        self.assertTrue(verify_pc_pcc_receipt(receipt))
        for certificate in result.certificates:
            self.assertTrue(
                verify_exact_conflict_certificate(
                    lifted,
                    certificate,
                    property_digest=receipt["property_digest"],
                )
            )

    def test_k7_c49_seven_thirds_to_one_without_children(
        self,
    ) -> None:
        lifted = _complete_c49(7)
        self.assertAlmostEqual(
            _lp_upper(lifted), 7.0 / 3.0, places=9
        )
        self.assertEqual(_fraction_integer_upper(7), Fraction(1))
        self.assertAlmostEqual(_milp_upper(lifted), 1.0, places=9)

        rivals = _rivals()
        result = run_pc_pcc_candidate(
            lifted,
            rivals,
            deadline=time.monotonic() + 15.0,
        )
        self.assertEqual(result.status, "unknown_to_safe_candidate")
        self.assertEqual(len(result.certificates), 21)
        self.assertEqual(result.receipt["expected_pairs"], 21)
        self.assertEqual(
            result.receipt["certified_conflict_edges"], 21
        )
        self.assertEqual(result.receipt["phase_children_minted"], 0)
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0, places=9)
        self.assertTrue(
            verify_pc_pcc_result(lifted, rivals, result)
        )

    def test_two_source_rows_are_needed_and_replayed_exactly(
        self,
    ) -> None:
        parent = _two_row_continuous_cancellation_hz()
        rivals = _rivals()
        result = run_pc_pcc_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 5.0,
        )
        self.assertIsNotNone(result.hz)
        self.assertEqual(len(result.certificates), 1)
        certificate = result.certificates[0]
        source_terms = [
            term
            for term in certificate.terms
            if term.kind in pc_pcc._SOURCE_KINDS
        ]
        self.assertEqual(len(source_terms), 2)
        self.assertEqual(len(certificate.source_row_digests), 2)
        self.assertEqual(certificate.contradiction, Fraction(-2))
        self.assertTrue(
            verify_exact_conflict_certificate(
                parent,
                certificate,
                property_digest=result.receipt[
                    "property_digest"
                ],
            )
        )
        self.assertTrue(
            verify_pc_pcc_result(parent, rivals, result)
        )
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0, places=10)

    def test_k4_minus_one_edge_never_forges_k4_clique(self) -> None:
        missing = ((2, 3),)
        lifted = _complete_c49(4, missing=missing)
        self.assertEqual(
            _fraction_integer_upper(4, missing=missing), Fraction(2)
        )
        self.assertAlmostEqual(_lp_upper(lifted), 2.0, places=9)
        self.assertAlmostEqual(_milp_upper(lifted), 2.0, places=9)

        rivals = _rivals()
        result = run_pc_pcc_candidate(
            lifted,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(result.status, "incomplete_conflict_graph")
        self.assertIsNone(result.hz)
        self.assertEqual(len(result.certificates), 5)
        self.assertEqual(result.receipt["expected_pairs"], 6)
        self.assertEqual(
            result.receipt["highs_feasible_or_unknown_pairs"], 1
        )
        self.assertFalse(result.receipt["complete_conflict_graph"])
        self.assertEqual(result.receipt["cut_rows_added"], 0)
        self.assertTrue(
            verify_pc_pcc_result(lifted, rivals, result)
        )


class PCPCCFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = _complete_c49(4)
        self.rivals = _rivals()
        self.result = run_pc_pcc_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        if not verify_pc_pcc_structural_result(
            self.parent, self.rivals, self.result
        ):
            raise AssertionError("test fixture failed live validation")

    def tearDown(self) -> None:
        # Revoke any fixture capability not deliberately consumed by a test.
        verify_pc_pcc_result(
            self.parent, self.rivals, self.result
        )

    def _forged_cut_result(
        self,
        *,
        updates,
        status=None,
        certificates=None,
        pair_records=None,
    ):
        receipt = dict(self.result.receipt)
        receipt.update(updates)
        if pair_records is not None:
            receipt["pair_records"] = list(pair_records)
        if certificates is None:
            certificates = self.result.certificates
        receipt = _reseal_receipt(receipt)
        forged_hz = _clone_hz(self.result.hz)
        setattr(
            forged_hz,
            "_pc_pcc_candidate_receipt",
            receipt,
        )
        return replace(
            self.result,
            status=(
                self.result.status
                if status is None
                else status
            ),
            hz=forged_hz,
            certificates=tuple(certificates),
            receipt=receipt,
        )

    def test_live_capability_is_exact_identity_and_single_use(
        self,
    ) -> None:
        copied = replace(self.result)
        self.assertFalse(
            verify_pc_pcc_result(
                self.parent, self.rivals, copied
            )
        )
        self.assertFalse(
            verify_pc_pcc_result(
                self.parent, self.rivals, self.result
            )
        )

        fresh = run_pc_pcc_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            verify_pc_pcc_result(
                self.parent, self.rivals, fresh
            )
        )
        self.assertFalse(
            verify_pc_pcc_result(
                self.parent, self.rivals, fresh
            )
        )

        same_content_rivals = tuple(list(self.rivals))
        wrong_binding = run_pc_pcc_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertIsNot(same_content_rivals, self.rivals)
        self.assertFalse(
            verify_pc_pcc_result(
                self.parent,
                same_content_rivals,
                wrong_binding,
            )
        )

    def test_resealed_status_caps_reorder_and_stopped_fail(
        self,
    ) -> None:
        status_forgery = self._forged_cut_result(
            updates={
                "status": "cut_candidate",
                "reason": "fabricated_no_gain",
            },
            status="cut_candidate",
        )
        self.assertTrue(
            verify_pc_pcc_receipt(status_forgery.receipt)
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, status_forgery
            )
        )

        caps = dict(self.result.receipt["caps"])
        caps.update(
            max_literals=1,
            max_pairs=1,
            max_highs_calls=1,
            max_exact_source_pairs=64,
        )
        cap_forgery = self._forged_cut_result(
            updates={"caps": caps}
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, cap_forgery
            )
        )

        order = tuple(
            range(len(self.result.certificates) - 1, -1, -1)
        )
        reordered_certificates = tuple(
            self.result.certificates[index] for index in order
        )
        reordered_records = tuple(
            self.result.receipt["pair_records"][index]
            for index in order
        )
        reorder_forgery = self._forged_cut_result(
            updates={
                "certificate_digests": [
                    certificate.certificate_digest
                    for certificate in reordered_certificates
                ]
            },
            certificates=reordered_certificates,
            pair_records=reordered_records,
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, reorder_forgery
            )
        )

        stopped_receipt = dict(self.result.receipt)
        stopped_receipt.update(
            status="stopped_without_cut",
            reason="literal_cap_exceeded",
            certificate_digests=[],
            pair_records=[],
            highs_calls=0,
            certified_conflict_edges=0,
            highs_feasible_or_unknown_pairs=0,
            exact_replay_rejected_pairs=0,
            unprocessed_pairs=stopped_receipt["expected_pairs"],
            complete_conflict_graph=False,
            cut_eligible=False,
            result_n_ub=None,
            cut_rows_added=0,
            cut_semantic_digest=None,
            post_cut_highs_property_uppers=[],
            deadline_respected=True,
        )
        stopped_receipt["closure"] = {
            key: True
            for key in stopped_receipt["closure"]
        }
        stopped_receipt = _reseal_receipt(stopped_receipt)
        stopped_forgery = replace(
            self.result,
            status="stopped_without_cut",
            hz=None,
            certificates=(),
            receipt=stopped_receipt,
        )
        self.assertTrue(
            verify_pc_pcc_receipt(stopped_receipt)
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, stopped_forgery
            )
        )

    def test_unkeyed_telemetry_is_diagnostic_but_live_mac_binds_it(
        self,
    ) -> None:
        receipt = self.result.receipt
        resealed = dict(receipt)
        resealed["pre_cut_highs_property_uppers"] = [
            999.0,
            999.0,
            999.0,
        ]
        resealed["post_cut_highs_property_uppers"] = [
            0.0,
            0.0,
            0.0,
        ]
        resealed = _reseal_receipt(resealed)
        receipt.clear()
        receipt.update(resealed)
        self.assertTrue(verify_pc_pcc_receipt(receipt))
        self.assertTrue(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, self.result
            )
        )
        self.assertFalse(
            verify_pc_pcc_result(
                self.parent, self.rivals, self.result
            )
        )

    def test_literal_binding_polarity_and_id_are_live_recomputed(
        self,
    ) -> None:
        certificate = self.result.certificates[0]
        left, right = certificate.literals
        forged_binding = replace(
            left,
            binding_digest="0" * 64,
        )
        forged_certificate = _reseal_certificate(
            replace(
                certificate,
                literals=(forged_binding, right),
            )
        )
        self.assertFalse(
            verify_exact_conflict_certificate(
                self.parent,
                forged_certificate,
                property_digest=self.result.receipt[
                    "property_digest"
                ],
            )
        )

        flipped = replace(left, phase=-left.phase)
        wrong_id = replace(
            left,
            stable_bcol_id=max(self.parent.bcol_ids) + 100,
        )
        for forged in (flipped, wrong_id):
            forged_certificate = _reseal_certificate(
                replace(
                    certificate,
                    literals=(forged, right),
                )
            )
            self.assertFalse(
                verify_exact_conflict_certificate(
                    self.parent,
                    forged_certificate,
                    property_digest=self.result.receipt[
                        "property_digest"
                    ],
                )
            )

        malformed_literal = replace(
            certificate,
            literals=(object(), right),
        )
        self.assertFalse(
            verify_exact_conflict_certificate(
                self.parent,
                malformed_literal,
                property_digest=self.result.receipt[
                    "property_digest"
                ],
            )
        )

    def test_missing_wrong_and_same_count_edge_replacements_fail(
        self,
    ) -> None:
        missing = replace(
            self.result,
            certificates=self.result.certificates[:-1],
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, missing
            )
        )

        first = self.result.certificates[0]
        source = first.terms[0]
        wrong_ray = _reseal_certificate(
            replace(
                first,
                terms=(
                    replace(source, numerator=source.numerator + 1),
                    *first.terms[1:],
                ),
            )
        )
        self.assertFalse(
            verify_exact_conflict_certificate(
                self.parent,
                wrong_ray,
                property_digest=self.result.receipt[
                    "property_digest"
                ],
            )
        )

        duplicate = replace(
            self.result,
            certificates=(
                self.result.certificates[0],
                self.result.certificates[0],
                *self.result.certificates[2:],
            ),
        )
        duplicate_receipt = dict(duplicate.receipt)
        duplicate_receipt["certificate_digests"] = [
            certificate.certificate_digest
            for certificate in duplicate.certificates
        ]
        duplicate = replace(
            duplicate,
            receipt=_reseal_receipt(duplicate_receipt),
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, duplicate
            )
        )

    def test_bad_internal_ray_is_rejected_before_any_cut(self) -> None:
        real_builder = pc_pcc._search_exact_bounded_farkas

        def bad_builder(*args, **kwargs):
            certificate = real_builder(*args, **kwargs)
            if certificate is None:
                return None
            first = certificate.terms[0]
            return _reseal_certificate(
                replace(
                    certificate,
                    terms=(
                        replace(
                            first, numerator=first.numerator + 1
                        ),
                        *certificate.terms[1:],
                    ),
                )
            )

        with patch.object(
            pc_pcc,
            "_search_exact_bounded_farkas",
            side_effect=bad_builder,
        ):
            rejected = run_pc_pcc_candidate(
                self.parent,
                self.rivals,
                deadline=time.monotonic() + 10.0,
            )
        self.assertEqual(
            rejected.status, "incomplete_conflict_graph"
        )
        self.assertIsNone(rejected.hz)
        self.assertEqual(
            rejected.receipt["exact_replay_rejected_pairs"], 6
        )
        self.assertEqual(
            rejected.receipt["certified_conflict_edges"], 0
        )

    def test_row_permutation_reruns_but_old_binding_fails(self) -> None:
        permuted = _clone_hz(self.parent)
        order = np.arange(permuted.n_ub - 1, -1, -1)
        permuted.Auc = permuted.Auc[order].tocsr()
        permuted.Aub = permuted.Aub[order].tocsr()
        permuted.ub = permuted.ub[order].copy()
        self.assertFalse(
            verify_pc_pcc_structural_result(
                permuted, self.rivals, self.result
            )
        )
        rerun = run_pc_pcc_candidate(
            permuted,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            verify_pc_pcc_result(permuted, self.rivals, rerun)
        )
        self.assertAlmostEqual(_lp_upper(rerun.hz), 1.0, places=9)

        binary_permuted = _clone_hz(self.parent)
        binary_order = np.asarray([2, 0, 3, 1], dtype=np.int64)
        binary_permuted.Gb = binary_permuted.Gb[
            :, binary_order
        ].tocsr()
        binary_permuted.Ab = binary_permuted.Ab[
            :, binary_order
        ].tocsr()
        binary_permuted.Aub = binary_permuted.Aub[
            :, binary_order
        ].tocsr()
        for matrix in (
            binary_permuted.Gb,
            binary_permuted.Ab,
            binary_permuted.Aub,
        ):
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
        binary_permuted.bcol_ids = binary_permuted.bcol_ids[
            binary_order
        ].copy()
        self.assertFalse(
            verify_pc_pcc_structural_result(
                binary_permuted, self.rivals, self.result
            )
        )
        id_rerun = run_pc_pcc_candidate(
            binary_permuted,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            verify_pc_pcc_result(
                binary_permuted, self.rivals, id_rerun
            )
        )
        self.assertAlmostEqual(_lp_upper(id_rerun.hz), 1.0, places=9)

    def test_parent_dense_csr_and_nonfinite_mutations_fail(self) -> None:
        dense = _clone_hz(self.parent)
        dense.c[0] += 0.25
        self.assertFalse(
            verify_pc_pcc_structural_result(
                dense, self.rivals, self.result
            )
        )

        sparse = _clone_hz(self.parent)
        sparse.Aub.data[0] += 0.25
        self.assertFalse(
            verify_pc_pcc_structural_result(
                sparse, self.rivals, self.result
            )
        )

        for field, value in (("c", np.nan), ("Aub", np.inf)):
            malformed = _clone_hz(self.parent)
            if field == "c":
                malformed.c[0] = value
            else:
                malformed.Aub.data[0] = value
            with self.assertRaises(PCPCCError):
                run_pc_pcc_candidate(
                    malformed,
                    self.rivals,
                    deadline=time.monotonic() + 2.0,
                )

    def test_deadline_caps_and_exact_term_cap_stop_without_cut(
        self,
    ) -> None:
        expired = run_pc_pcc_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() - 1.0,
        )
        self.assertEqual(expired.status, "stopped_without_cut")
        self.assertIsNone(expired.hz)
        self.assertFalse(expired.receipt["deadline_respected"])
        self.assertTrue(
            verify_pc_pcc_result(
                self.parent, self.rivals, expired
            )
        )

        capped = run_pc_pcc_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() + 5.0,
            max_literals=3,
        )
        self.assertEqual(capped.status, "stopped_without_cut")
        self.assertEqual(capped.receipt["reason"], "literal_cap_exceeded")
        self.assertIsNone(capped.hz)

        term_capped = run_pc_pcc_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() + 5.0,
            max_exact_terms=2,
        )
        self.assertEqual(
            term_capped.status, "incomplete_conflict_graph"
        )
        self.assertEqual(
            term_capped.receipt["exact_replay_rejected_pairs"], 6
        )
        self.assertIsNone(term_capped.hz)

    def test_slow_highs_crossing_absolute_deadline_discards_cut(
        self,
    ) -> None:
        def slow_candidate(_hz, _pair, *, deadline):
            while time.monotonic() < deadline + 0.002:
                pass
            return True

        with patch.object(
            pc_pcc,
            "_highs_pair_infeasible_candidate",
            side_effect=slow_candidate,
        ):
            stopped = run_pc_pcc_candidate(
                self.parent,
                self.rivals,
                deadline=time.monotonic() + 0.02,
            )
        self.assertEqual(stopped.status, "stopped_without_cut")
        self.assertEqual(
            stopped.receipt["reason"],
            "deadline_expired_in_pair_highs",
        )
        self.assertIsNone(stopped.hz)
        self.assertFalse(stopped.receipt["deadline_respected"])

    def test_cut_omission_forgery_and_conditional_replacement_fail(
        self,
    ) -> None:
        omitted = replace(self.result, hz=None)
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, omitted
            )
        )

        forged_hz = _clone_hz(self.result.hz)
        forged_hz.Aub.data[-1] += 1.0
        forged_receipt = dict(self.result.receipt)
        forged_receipt["cut_semantic_digest"] = (
            pc_pcc.sparse_hz_semantic_digest(forged_hz)
        )
        forged_receipt = _reseal_receipt(forged_receipt)
        setattr(
            forged_hz,
            "_pc_pcc_candidate_receipt",
            forged_receipt,
        )
        forged = replace(
            self.result, hz=forged_hz, receipt=forged_receipt
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                self.parent, self.rivals, forged
            )
        )

        conditional_parent = _clone_hz(self.parent)
        setattr(
            conditional_parent,
            "_test_conditional_state",
            {"stable_rows": (11, 13), "mode": "original"},
        )
        conditional_result = run_pc_pcc_candidate(
            conditional_parent,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        conditional_hz = _clone_hz(conditional_result.hz)
        setattr(
            conditional_hz,
            "_test_conditional_state",
            {"stable_rows": (11, 17), "mode": "original"},
        )
        conditional_receipt = dict(conditional_result.receipt)
        conditional_receipt["cut_semantic_digest"] = (
            pc_pcc.sparse_hz_semantic_digest(conditional_hz)
        )
        conditional_receipt = _reseal_receipt(conditional_receipt)
        setattr(
            conditional_hz,
            "_pc_pcc_candidate_receipt",
            conditional_receipt,
        )
        conditional_forgery = replace(
            conditional_result,
            hz=conditional_hz,
            receipt=conditional_receipt,
        )
        self.assertFalse(
            verify_pc_pcc_structural_result(
                conditional_parent,
                self.rivals,
                conditional_forgery,
            )
        )


if __name__ == "__main__":
    unittest.main()
