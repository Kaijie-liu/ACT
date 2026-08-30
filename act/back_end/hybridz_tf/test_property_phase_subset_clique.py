#!/usr/bin/env python3
"""Exact closure and omission-firewall toys for subset PC-PCC."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    consume_raw_vnnlib_top1_candidate,
    issue_raw_vnnlib_top1_candidate,
    validate_consumed_raw_vnnlib_rival_batch,
)
from act.back_end.hybridz_tf.property_phase_literal_groups import (
    derive_property_literal_groups_candidate,
)
from act.back_end.hybridz_tf.property_phase_subset_clique import (
    PropertyPhaseSubsetCliqueError,
    run_property_phase_subset_clique_candidate,
    verify_property_phase_subset_clique_result,
)
from act.back_end.hybridz_tf.test_property_phase_conflict_clique import (
    _clone_hz,
    _complete_c49,
    _lp_upper,
    _rivals,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    _highs_property_upper,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_fresh_col_ids,
)


def _append_zero_effect_binary(hz: SparseHZono) -> SparseHZono:
    """Add one unconstrained stable binary with zero output effect."""

    stable_id = int(
        hz_fresh_col_ids(1, device="cpu")
        .detach()
        .cpu()
        .numpy()[0]
    )
    result = SparseHZono(
        c=np.array(hz.c, dtype=np.float64, copy=True),
        Gc=hz.Gc.copy(),
        Gb=sp.hstack(
            [
                hz.Gb,
                sp.csr_matrix((hz.n_out, 1), dtype=np.float64),
            ],
            format="csr",
        ),
        Ac=hz.Ac.copy(),
        Ab=sp.hstack(
            [
                hz.Ab,
                sp.csr_matrix((hz.n_eq, 1), dtype=np.float64),
            ],
            format="csr",
        ),
        b=np.array(hz.b, dtype=np.float64, copy=True),
        Auc=hz.Auc.copy(),
        Aub=sp.hstack(
            [
                hz.Aub,
                sp.csr_matrix((hz.n_ub, 1), dtype=np.float64),
            ],
            format="csr",
        ),
        ub=np.array(hz.ub, dtype=np.float64, copy=True),
        col_ids=np.array(hz.col_ids, dtype=np.int64, copy=True),
        bcol_ids=np.concatenate(
            [
                np.array(hz.bcol_ids, dtype=np.int64, copy=True),
                np.asarray([stable_id], dtype=np.int64),
            ]
        ),
    )
    for name, value in vars(hz).items():
        if "conditional" in name.lower():
            setattr(result, name, value)
    return result


def _classification_parent(hz: SparseHZono) -> SparseHZono:
    """Embed one margin geometry as two competitors below true logit 1.1."""

    zero_continuous = sp.csr_matrix(
        (1, hz.n_cont), dtype=np.float64
    )
    zero_binary = sp.csr_matrix(
        (1, hz.n_bin), dtype=np.float64
    )
    result = SparseHZono(
        c=np.asarray(
            [1.1, float(hz.c[0]), float(hz.c[0])],
            dtype=np.float64,
        ),
        Gc=sp.vstack(
            [zero_continuous, hz.Gc, hz.Gc], format="csr"
        ),
        Gb=sp.vstack(
            [zero_binary, hz.Gb, hz.Gb], format="csr"
        ),
        Ac=hz.Ac.copy(),
        Ab=hz.Ab.copy(),
        b=np.array(hz.b, dtype=np.float64, copy=True),
        Auc=hz.Auc.copy(),
        Aub=hz.Aub.copy(),
        ub=np.array(hz.ub, dtype=np.float64, copy=True),
        col_ids=np.array(hz.col_ids, dtype=np.int64, copy=True),
        bcol_ids=np.array(
            hz.bcol_ids, dtype=np.int64, copy=True
        ),
    )
    for name, value in vars(hz).items():
        if "conditional" in name.lower():
            setattr(result, name, value)
    return result


def _two_groups_with_one_shared_physical_pair() -> SparseHZono:
    """Two complete signed cliques share the physical (+b0,+b1) pair."""

    binary_ids = (
        hz_fresh_col_ids(3, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    rows = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    return SparseHZono(
        c=np.asarray([1.5, 1.5], dtype=np.float64),
        Gc=sp.csr_matrix((2, 0), dtype=np.float64),
        Gb=sp.csr_matrix(
            np.asarray(
                [
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, -0.5],
                ],
                dtype=np.float64,
            )
        ),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, 3), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((5, 0), dtype=np.float64),
        Aub=sp.csr_matrix(rows),
        ub=np.zeros(5, dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=binary_ids,
    )


def _raw_top1_source() -> str:
    return """
    (set-logic QF_LRA)
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (declare-const Y_2 Real)
    (assert (>= X_0 0))
    (assert (<= X_0 1))
    (assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))
    """


class _AliasInt(int):
    """An int subclass with nonstandard equality semantics."""

    def __new__(cls, source: int, target: int):
        value = int.__new__(cls, source)
        value.target = target
        return value

    def __eq__(self, other):
        return int(other) == self.target

    def __ne__(self, other):
        return not self == other


class _AliasStr(str):
    """A str subclass with nonstandard equality semantics."""

    def __new__(cls, source: str, target: str):
        value = str.__new__(cls, source)
        value.target = target
        return value

    def __eq__(self, other):
        return str(other) == self.target

    def __ne__(self, other):
        return not self == other


def _run(parent: SparseHZono):
    rivals = _rivals()
    grouping = derive_property_literal_groups_candidate(
        parent, rivals
    )
    result = run_property_phase_subset_clique_candidate(
        parent,
        rivals,
        grouping,
        deadline=time.monotonic() + 10.0,
    )
    return rivals, grouping, result


class PropertyPhaseSubsetCliqueTests(unittest.TestCase):
    def test_raw_top1_to_group_to_exact_subset_cut_chain(
        self,
    ) -> None:
        parent = _classification_parent(_complete_c49(4))
        C = torch.tensor(
            [
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        live = {
            "kind": "TOP1_ROBUST",
            "C": C,
            "thresholds": torch.zeros(
                (1, 2), dtype=torch.float64
            ),
            "M": 2,
            "y_true": torch.tensor([0], dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "top1.vnnlib"
            path.write_text(
                _raw_top1_source().strip() + "\n",
                encoding="utf-8",
            )
            source_sha = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=source_sha,
                live_assert_params=live,
                deadline=time.monotonic() + 10.0,
            )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate,
                live_assert_params=live,
                deadline=time.monotonic() + 10.0,
            )
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(
                    batch
                )
            )
            rivals = batch.rivals

        grouping = derive_property_literal_groups_candidate(
            parent, rivals
        )
        result = run_property_phase_subset_clique_candidate(
            parent,
            rivals,
            grouping,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, result
            )
        )
        self.assertEqual(
            tuple(rival.rival_id for rival in rivals), (1, 2)
        )
        self.assertEqual(len(grouping.groups), 1)
        self.assertEqual(len(result.closures[0].literals), 4)
        for rival in rivals:
            before = _highs_property_upper(
                parent,
                rival,
                deadline=time.monotonic() + 10.0,
            )
            after = _highs_property_upper(
                result.hz,
                rival,
                deadline=time.monotonic() + 10.0,
            )
            self.assertGreater(before, 0.0)
            self.assertLess(after, 0.0)

    def test_complete_subset_with_zero_omission_tightens_to_one(
        self,
    ) -> None:
        parent = _append_zero_effect_binary(_complete_c49(4))
        rivals, grouping, result = _run(parent)

        self.assertEqual(result.status, "subset_cut_candidate")
        self.assertIsNotNone(result.hz)
        self.assertEqual(len(result.closures), 1)
        closure = result.closures[0]
        self.assertEqual(len(closure.literals), 4)
        self.assertEqual(len(closure.omitted_zero_bcol_ids), 1)
        self.assertEqual(len(closure.pair_records), 6)
        self.assertEqual(len(closure.certificates), 6)
        self.assertTrue(closure.complete)
        self.assertTrue(closure.cut_applied)
        self.assertTrue(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, result
            )
        )
        self.assertGreater(_lp_upper(parent), 1.0)
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0)

        live_row = np.asarray(
            result.hz.Aub.getrow(result.hz.n_ub - 1).toarray()
        ).reshape(-1)
        omitted_position = parent.n_bin - 1
        self.assertEqual(live_row[omitted_position], 0.0)
        self.assertEqual(np.count_nonzero(live_row), 4)
        self.assertEqual(result.hz.ub[-1], -2.0)
        self.assertFalse(result.proof_authority)

    def test_missing_edge_produces_no_subset_cut(self) -> None:
        parent = _append_zero_effect_binary(
            _complete_c49(4, missing=((0, 1),))
        )
        rivals, grouping, result = _run(parent)
        self.assertEqual(
            result.status, "no_complete_subset_clique"
        )
        self.assertIsNone(result.hz)
        self.assertEqual(len(result.closures), 1)
        closure = result.closures[0]
        self.assertFalse(closure.complete)
        self.assertFalse(closure.cut_applied)
        self.assertEqual(len(closure.pair_records), 6)
        self.assertEqual(len(closure.certificates), 5)
        self.assertTrue(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, result
            )
        )

    def test_k7_complete_closure_uses_one_persistent_model(
        self,
    ) -> None:
        parent = _complete_c49(7)
        rivals, grouping, result = _run(parent)
        self.assertEqual(len(result.closures), 1)
        self.assertEqual(len(result.closures[0].pair_records), 21)
        self.assertEqual(
            len(result.closures[0].certificates), 21
        )
        self.assertEqual(result.telemetry["model_builds"], 1)
        self.assertEqual(
            result.telemetry["oracle"]["solve_calls"], 21
        )
        self.assertTrue(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, result
            )
        )
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0)

    def test_two_groups_may_share_one_physical_signed_pair(
        self,
    ) -> None:
        parent = _two_groups_with_one_shared_physical_pair()
        rivals = (
            replace(
                _rivals()[0],
                rival_id=111,
                objective=(1.0, 0.0),
                threshold=1.1,
            ),
            replace(
                _rivals()[1],
                rival_id=222,
                objective=(0.0, 1.0),
                threshold=1.1,
            ),
        )
        grouping = derive_property_literal_groups_candidate(
            parent, rivals
        )
        self.assertEqual(len(grouping.groups), 2)
        result = run_property_phase_subset_clique_candidate(
            parent,
            rivals,
            grouping,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(len(result.closures), 2)
        self.assertTrue(
            all(
                closure.complete
                for closure in result.closures
            )
        )
        self.assertEqual(
            tuple(
                len(closure.certificates)
                for closure in result.closures
            ),
            (3, 3),
        )
        self.assertTrue(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, result
            )
        )

    def test_closure_certificate_and_telemetry_tampering_fail(
        self,
    ) -> None:
        parent = _append_zero_effect_binary(_complete_c49(4))
        rivals, grouping, result = _run(parent)
        closure = result.closures[0]

        missing_certificate = replace(
            closure,
            certificates=closure.certificates[:-1],
        )
        malformed = replace(
            result, closures=(missing_certificate,)
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, malformed
            )
        )

        false_complete = replace(
            closure, complete=False, cut_applied=False
        )
        malformed = replace(result, closures=(false_complete,))
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, malformed
            )
        )

        telemetry = dict(result.telemetry)
        telemetry["pair_count"] -= 1
        malformed = replace(result, telemetry=telemetry)
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent, rivals, grouping, malformed
            )
        )

    def test_nonstandard_equality_values_fail_closed(
        self,
    ) -> None:
        parent = _append_zero_effect_binary(_complete_c49(4))
        rivals, grouping, result = _run(parent)
        closure = result.closures[0]

        record = closure.pair_records[0]
        literal = record.literals[0]
        aliased_literal = replace(
            literal,
            stable_bcol_id=_AliasInt(
                literal.stable_bcol_id,
                literal.stable_bcol_id,
            ),
        )
        aliased_record = replace(
            record,
            literals=(aliased_literal, record.literals[1]),
        )
        malformed_closure = replace(
            closure,
            pair_records=(
                aliased_record,
                *closure.pair_records[1:],
            ),
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(result, closures=(malformed_closure,)),
            )
        )

        certificate = closure.certificates[0]
        aliased_certificate = replace(
            certificate,
            property_digest=_AliasStr(
                certificate.property_digest,
                certificate.property_digest,
            ),
        )
        malformed_closure = replace(
            closure,
            certificates=(
                aliased_certificate,
                *closure.certificates[1:],
            ),
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(result, closures=(malformed_closure,)),
            )
        )

        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(
                    result,
                    parent_semantic_digest=_AliasStr(
                        result.parent_semantic_digest,
                        result.parent_semantic_digest,
                    ),
                ),
            )
        )
        telemetry = dict(result.telemetry)
        telemetry["pair_count"] = _AliasInt(
            telemetry["pair_count"],
            telemetry["pair_count"],
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(result, telemetry=telemetry),
            )
        )

    def test_diagnostic_telemetry_tampering_fails_closed(
        self,
    ) -> None:
        parent = _complete_c49(4)
        rivals, grouping, result = _run(parent)

        telemetry = dict(result.telemetry)
        oracle = dict(telemetry["oracle"])
        oracle["solve_calls"] = -999
        telemetry["oracle"] = oracle
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(result, telemetry=telemetry),
            )
        )

        telemetry = dict(result.telemetry)
        grouping_caps = dict(telemetry["grouping_caps"])
        grouping_caps["extra"] = 1
        telemetry["grouping_caps"] = grouping_caps
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(result, telemetry=telemetry),
            )
        )

        closure = result.closures[0]
        record = replace(
            closure.pair_records[0],
            ray_nonzero_rows=10**9,
        )
        malformed_closure = replace(
            closure,
            pair_records=(
                record,
                *closure.pair_records[1:],
            ),
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                replace(result, closures=(malformed_closure,)),
            )
        )

    def test_parent_grouping_caps_and_deadline_fail_closed(
        self,
    ) -> None:
        parent = _append_zero_effect_binary(_complete_c49(4))
        rivals, grouping, result = _run(parent)

        changed = _clone_hz(parent)
        changed.Aub.data[0] = np.nextafter(
            changed.Aub.data[0], np.inf
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                changed, rivals, grouping, result
            )
        )
        with self.assertRaises(PropertyPhaseSubsetCliqueError):
            run_property_phase_subset_clique_candidate(
                parent,
                rivals,
                grouping,
                deadline=time.monotonic() + 10.0,
                max_total_pairs=5,
            )
        with self.assertRaises(PropertyPhaseSubsetCliqueError):
            run_property_phase_subset_clique_candidate(
                parent,
                rivals,
                grouping,
                deadline=time.monotonic() + 10.0,
                max_parent_variables=1,
            )
        with self.assertRaises(PropertyPhaseSubsetCliqueError):
            run_property_phase_subset_clique_candidate(
                parent,
                rivals,
                grouping,
                deadline=time.monotonic() + 10.0,
                max_parent_buffer_items=1,
            )
        with self.assertRaises(PropertyPhaseSubsetCliqueError):
            run_property_phase_subset_clique_candidate(
                parent,
                rivals,
                grouping,
                deadline=time.monotonic() - 1.0,
            )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                result,
                max_total_pairs=5,
            )
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                result,
                max_parent_variables=1,
            )
        )
        self.assertFalse(
            verify_property_phase_subset_clique_result(
                parent,
                rivals,
                grouping,
                result,
                max_parent_buffer_items=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
