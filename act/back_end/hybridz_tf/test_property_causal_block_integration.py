#!/usr/bin/env python3
# ===- test_property_causal_block_integration.py ---------------------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Full-frame gates for the proof-neutral PC-CBDE integration layer."""

from __future__ import annotations

from fractions import Fraction
import time
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from act.back_end.hybridz_tf.gpu_dual_candidates import (
    ConstraintRowTag,
    OriginalFrameLP,
)
from act.back_end.hybridz_tf.property_causal_block_integration import (
    property_causal_block_integration,
)
from act.back_end.solver.solver_hz import (
    _hz_independent_lp_lagrangian_upper,
)


def _shared_stall_fixture(
    *,
    wrong_copy: bool = False,
) -> tuple[OriginalFrameLP, np.ndarray, np.ndarray, np.ndarray]:
    """Build a genuine packet/property-separated exchange problem.

    Stable columns are ``[y, x, w]`` and the generated packet does not contain
    the property column ``y``:

      G: w - x/2 <= 0
      M: y + x/2 <= 0

    Here ``w`` is fixed to zero and ``y,x`` are in ``[-1,1]``.  With
    ``d_M=1`` as the warm start, the local dual support is

      ``|1-d_M| + |d_G-d_M|/2``.

    Thus a joint block exchange reaches zero, while deleting generated or
    bridge families gives respectively ``1/2`` and ``1``.  In the wrong-copy
    control, G consumes an interval-identical independent ``x_copy``; the CSR
    incidence path then disappears before optimization.
    """

    if wrong_copy:
        dense = np.asarray(
            [
                [0.0, 0.0, 1.0, -0.5],
                [1.0, 0.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        lb = np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float64)
        ub = np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float64)
        q = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    else:
        dense = np.asarray(
            [
                [0.0, -0.5, 1.0],
                [1.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        lb = np.asarray([-1.0, -1.0, 0.0], dtype=np.float64)
        ub = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
        q = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    tags = (
        ConstraintRowTag(
            0,
            "ub",
            "property_micro_rlt:generated:shared_packet",
            0,
        ),
        ConstraintRowTag(1, "ub", "bridge:shared_materialization", 0),
    )
    frame = OriginalFrameLP(
        A=sp.csr_matrix(dense),
        rl=np.full(2, -np.inf, dtype=np.float64),
        ru=np.zeros(2, dtype=np.float64),
        lb=lb,
        ub=ub,
        row_tags=tags,
    )
    warm = np.asarray([[0.0, 1.0]], dtype=np.float64)
    allowed = np.ones(2, dtype=np.bool_)
    return frame, q, warm, allowed


def _integrate_stall(
    frame: OriginalFrameLP,
    q: np.ndarray,
    warm: np.ndarray,
    allowed: np.ndarray,
):
    return property_causal_block_integration(
        frame,
        q,
        warm,
        incidence_packet_rows=(0,),
        optimization_packet_rows=(0,),
        source_rows=(),
        allowed_row_mask=allowed,
        row_tags=frame.row_tags,
        deadline=time.monotonic() + 5.0,
    )


def _checker_upper(
    frame: OriginalFrameLP,
    row_dual: np.ndarray,
) -> tuple[np.longdouble | None, dict]:
    objective = np.zeros((1, frame.n_variables), dtype=np.float64)
    objective[0, 0] = 1.0
    return _hz_independent_lp_lagrangian_upper(
        c=np.asarray([0.0], dtype=np.float64),
        Gc=sp.csr_matrix(objective),
        C_row=np.asarray([1.0], dtype=np.float64),
        threshold=0.0,
        A=frame.A,
        rl=frame.rl,
        ru=frame.ru,
        lb=frame.lb,
        ub=frame.ub,
        row_dual=row_dual,
    )


def _highs_upper(frame: OriginalFrameLP, rows: tuple[int, ...]) -> float:
    selected = np.asarray(rows, dtype=np.int64)
    result = linprog(
        -np.eye(1, frame.n_variables, 0, dtype=np.float64).reshape(-1),
        A_ub=frame.A[selected, :] if selected.size else None,
        b_ub=frame.ru[selected] if selected.size else None,
        bounds=list(zip(frame.lb, frame.ub)),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun)


def _atomic_add_fixture(
    *,
    wrong_reverse_copy: bool = False,
    production_rows: int = 4,
    production_columns: int = 3,
) -> tuple[OriginalFrameLP, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build packet(w)->source(x,w)->atomic ADD(y,x)->property(y)."""

    if production_rows < 4 or production_columns < 3:
        raise ValueError("atomic ADD fixture is too small")
    counts = np.full(production_rows, 12, dtype=np.int32)
    counts[:4] = np.asarray([1, 2, 2, 2], dtype=np.int32)
    if production_columns < 15 and production_rows > 4:
        raise ValueError("production fixture requires at least 15 columns")
    indptr = np.empty(production_rows + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.ones(int(indptr[-1]), dtype=np.float64)

    indices[0:1] = (2,)
    data[0:1] = (1.0,)
    indices[1:3] = (1, 2)
    data[1:3] = (1.0, -1.0)
    indices[3:5] = (0, 1)
    data[3:5] = (1.0, -1.0)
    indices[5:7] = (0, 2) if wrong_reverse_copy else (0, 1)
    data[5:7] = (-1.0, 1.0)
    if production_rows > 4:
        background = np.arange(3, 15, dtype=np.int32)
        indices[7:] = np.tile(background, production_rows - 4)

    tags = ["background:ordinary"] * production_rows
    tags[:4] = [
        "property_micro_rlt:generated:packet_w",
        "source:packet_to_x",
        "add_materialize:40:forward",
        "add_materialize:40:reverse",
    ]
    tags_tuple = tuple(tags)
    frame = OriginalFrameLP(
        A=sp.csr_matrix(
            (data, indices, indptr),
            shape=(production_rows, production_columns),
        ),
        rl=np.full(production_rows, -np.inf, dtype=np.float64),
        ru=np.zeros(production_rows, dtype=np.float64),
        lb=-np.ones(production_columns, dtype=np.float64),
        ub=np.ones(production_columns, dtype=np.float64),
        row_tags=tuple(
            ConstraintRowTag(row, "ub", tag, 0)
            for row, tag in enumerate(tags_tuple)
        ),
    )
    q = np.zeros((1, production_columns), dtype=np.float64)
    q[0, 0] = 1.0
    warm = np.zeros((1, production_rows), dtype=np.float64)
    return frame, q, warm, tags_tuple


class PropertyCausalBlockIntegrationTests(unittest.TestCase):
    def test_shared_stall_full_expansion_and_longdouble_checker(self) -> None:
        frame, q, warm, allowed = _shared_stall_fixture()

        result = _integrate_stall(frame, q, warm, allowed)

        self.assertTrue(result.success, (result.status, result.diagnostic))
        self.assertFalse(result.proof_authority)
        self.assertEqual(result.status, "candidate_ready_unchecked")
        np.testing.assert_array_equal(result.property_columns, [0])
        np.testing.assert_array_equal(result.cone_rows, [1])
        np.testing.assert_array_equal(result.local_rows, [0, 1])
        self.assertEqual(
            tuple(candidate.name for candidate in result.ablations),
            (
                "full",
                "without_generated",
                "without_bridge",
                "without_both",
            ),
        )

        expected = {
            "full": 0.0,
            "without_generated": 0.5,
            "without_bridge": 1.0,
            "without_both": 1.0,
        }
        expected_d = {
            "full": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "without_generated": np.asarray([[0.0, 1.0]], dtype=np.float64),
            "without_bridge": np.asarray([[0.0, 0.0]], dtype=np.float64),
            "without_both": np.asarray([[0.0, 0.0]], dtype=np.float64),
        }
        for name, exact in expected.items():
            candidate = result.ablation(name)
            self.assertFalse(candidate.proof_authority)
            self.assertFalse(candidate.optimizer.proof_authority)
            np.testing.assert_allclose(candidate.d, expected_d[name], atol=0.0)
            np.testing.assert_array_equal(candidate.row_dual, -candidate.d)
            np.testing.assert_allclose(
                candidate.candidate_support,
                [exact],
                atol=0.0,
            )
            upper, receipt = _checker_upper(frame, candidate.row_dual[0])
            self.assertEqual(receipt["status"], "verified_upper")
            self.assertIsNotNone(upper)
            self.assertAlmostEqual(float(upper), exact, places=14)

        self.assertLess(
            result.ablation("full").candidate_support[0],
            result.ablation("without_generated").candidate_support[0],
        )
        self.assertLess(
            result.ablation("full").candidate_support[0],
            result.ablation("without_bridge").candidate_support[0],
        )

        # Independent exact-rational values for the exposed multiplier states.
        support = lambda m, g: (
            abs(Fraction(1) - m)
            + abs(g - m) / 2
        )
        self.assertEqual(support(Fraction(1), Fraction(1)), Fraction(0))
        self.assertEqual(support(Fraction(1), Fraction(0)), Fraction(1, 2))
        self.assertEqual(support(Fraction(0), Fraction(0)), Fraction(1))

        # HiGHS primal optima agree for the four actual row-family deletions.
        self.assertEqual(_highs_upper(frame, (0, 1)), 0.0)
        self.assertEqual(_highs_upper(frame, (1,)), 0.5)
        self.assertEqual(_highs_upper(frame, (0,)), 1.0)
        self.assertEqual(_highs_upper(frame, ()), 1.0)

    def test_wrong_copy_and_deleted_or_tampered_path_fail_closed(self) -> None:
        wrong_frame, q, warm, allowed = _shared_stall_fixture(
            wrong_copy=True
        )
        wrong = _integrate_stall(wrong_frame, q, warm, allowed)
        self.assertFalse(wrong.success)
        self.assertEqual(wrong.status, "incidence_path_unavailable")
        self.assertEqual(wrong.ablations, ())

        frame, q, warm, allowed = _shared_stall_fixture()
        deleted = allowed.copy()
        deleted[1] = False
        missing = _integrate_stall(frame, q, warm, deleted)
        self.assertFalse(missing.success)
        self.assertEqual(missing.status, "incidence_path_unavailable")
        self.assertEqual(missing.ablations, ())

        tampered_A = frame.A.copy()
        tampered_A[1, 0] = 0.0
        tampered_A.eliminate_zeros()
        tampered = OriginalFrameLP(
            A=tampered_A,
            rl=frame.rl,
            ru=frame.ru,
            lb=frame.lb,
            ub=frame.ub,
            row_tags=frame.row_tags,
        )
        changed = _integrate_stall(tampered, q, warm, allowed)
        self.assertFalse(changed.success)
        self.assertEqual(changed.status, "incidence_path_unavailable")
        self.assertEqual(changed.ablations, ())

    def test_row_permutation_preserves_semantic_full_candidates(self) -> None:
        frame, q, warm, allowed = _shared_stall_fixture()
        baseline = _integrate_stall(frame, q, warm, allowed)
        self.assertTrue(baseline.success)

        permutation = np.asarray([1, 0], dtype=np.int64)
        tags = tuple(
            ConstraintRowTag(
                global_row=position,
                sense=frame.row_tags[int(original)].sense,
                block_tag=frame.row_tags[int(original)].block_tag,
                block_local_row=frame.row_tags[int(original)].block_local_row,
            )
            for position, original in enumerate(permutation)
        )
        permuted_frame = OriginalFrameLP(
            A=frame.A[permutation, :],
            rl=frame.rl[permutation],
            ru=frame.ru[permutation],
            lb=frame.lb,
            ub=frame.ub,
            row_tags=tags,
        )
        packet_row = int(np.flatnonzero(permutation == 0)[0])
        permuted = property_causal_block_integration(
            permuted_frame,
            q,
            warm[:, permutation],
            incidence_packet_rows=(packet_row,),
            optimization_packet_rows=(packet_row,),
            source_rows=(),
            allowed_row_mask=allowed[permutation],
            row_tags=tags,
            deadline=time.monotonic() + 5.0,
        )
        self.assertTrue(
            permuted.success,
            (permuted.status, permuted.diagnostic),
        )
        for name in (
            "full",
            "without_generated",
            "without_bridge",
            "without_both",
        ):
            restored = np.zeros_like(baseline.ablation(name).d)
            restored[:, permutation] = permuted.ablation(name).d
            np.testing.assert_array_equal(
                restored,
                baseline.ablation(name).d,
            )
            np.testing.assert_array_equal(
                permuted.ablation(name).candidate_support,
                baseline.ablation(name).candidate_support,
            )

    def test_atomic_add_source_family_and_wrong_reverse_copy(self) -> None:
        frame, q, warm, tags = _atomic_add_fixture(
            production_rows=5,
            production_columns=15,
        )
        allowed = np.ones(frame.n_rows, dtype=np.bool_)
        # A nonzero multiplier on a disconnected, unselected row exercises
        # local-q residualization and full-row expansion.  It must remain fixed
        # through every family ablation.
        warm[0, 4] = 0.25

        result = property_causal_block_integration(
            frame,
            q,
            warm,
            incidence_packet_rows=(0,),
            optimization_packet_rows=(0,),
            source_rows=(1, 4),
            allowed_row_mask=allowed,
            row_tags=tags,
            deadline=time.monotonic() + 5.0,
        )

        self.assertTrue(result.success, (result.status, result.diagnostic))
        np.testing.assert_array_equal(result.cone_rows, [1, 2, 3])
        np.testing.assert_array_equal(result.source_rows, [1])
        np.testing.assert_array_equal(result.ignored_source_rows, [4])
        np.testing.assert_array_equal(result.bridge_rows, [2, 3])
        np.testing.assert_array_equal(result.generated_rows, [0])
        self.assertEqual(
            result.ablation("full").enabled_families,
            ("bridge", "generated", "source"),
        )
        self.assertEqual(
            result.ablation("without_generated").enabled_families,
            ("bridge", "source"),
        )
        self.assertEqual(
            result.ablation("without_bridge").enabled_families,
            ("generated", "source"),
        )
        self.assertEqual(
            result.ablation("without_both").enabled_families,
            ("source",),
        )
        for candidate in result.ablations:
            self.assertEqual(candidate.d[0, 4], 0.25)
            upper, receipt = _checker_upper(frame, candidate.row_dual[0])
            self.assertEqual(receipt["status"], "verified_upper")
            self.assertIsNotNone(upper)
            self.assertAlmostEqual(
                float(upper),
                float(candidate.candidate_support[0]),
                places=13,
            )

        wrong_frame, wrong_q, wrong_warm, wrong_tags = _atomic_add_fixture(
            wrong_reverse_copy=True
        )
        wrong = property_causal_block_integration(
            wrong_frame,
            wrong_q,
            wrong_warm,
            incidence_packet_rows=(0,),
            optimization_packet_rows=(0,),
            source_rows=(1,),
            allowed_row_mask=np.ones(wrong_frame.n_rows, dtype=np.bool_),
            row_tags=wrong_tags,
            deadline=time.monotonic() + 5.0,
        )
        self.assertFalse(wrong.success)
        self.assertEqual(wrong.status, "incidence_path_unavailable")
        self.assertEqual(wrong.ablations, ())

    def test_single_objective_and_frame_tag_identity_are_fail_closed(self) -> None:
        frame, q, warm, allowed = _shared_stall_fixture()

        batched = property_causal_block_integration(
            frame,
            np.concatenate((q, q), axis=0),
            np.concatenate((warm, warm), axis=0),
            incidence_packet_rows=(0,),
            optimization_packet_rows=(0,),
            source_rows=(),
            allowed_row_mask=allowed,
            row_tags=frame.row_tags,
            deadline=time.monotonic() + 5.0,
        )
        self.assertFalse(batched.success)
        self.assertEqual(batched.status, "invalid_objective")
        self.assertEqual(batched.ablations, ())

        inconsistent_tags = (
            frame.row_tags[0],
            ConstraintRowTag(
                1,
                "ub",
                "bridge:different_metadata",
                0,
            ),
        )
        inconsistent = property_causal_block_integration(
            frame,
            q,
            warm,
            incidence_packet_rows=(0,),
            optimization_packet_rows=(0,),
            source_rows=(),
            allowed_row_mask=allowed,
            row_tags=inconsistent_tags,
            deadline=time.monotonic() + 5.0,
        )
        self.assertFalse(inconsistent.success)
        self.assertEqual(inconsistent.status, "invalid_tags")
        self.assertEqual(inconsistent.ablations, ())

    def test_small_production_shaped_deadline_and_caps(self) -> None:
        frame, q, warm, tags = _atomic_add_fixture(
            production_rows=8192,
            production_columns=2048,
        )
        allowed = np.ones(frame.n_rows, dtype=np.bool_)
        call = dict(
            incidence_packet_rows=(0,),
            optimization_packet_rows=(0,),
            source_rows=(1,),
            allowed_row_mask=allowed,
            row_tags=tags,
        )

        started = time.monotonic()
        result = property_causal_block_integration(
            frame,
            q,
            warm,
            deadline=started + 5.0,
            **call,
        )
        elapsed = time.monotonic() - started
        self.assertTrue(result.success, (result.status, result.diagnostic))
        np.testing.assert_array_equal(result.cone_rows, [1, 2, 3])
        self.assertLess(elapsed, 5.0)

        no_updates = property_causal_block_integration(
            frame,
            q,
            warm,
            deadline=time.monotonic() + 5.0,
            optimizer_max_updates=0,
            **call,
        )
        self.assertTrue(
            no_updates.success,
            (no_updates.status, no_updates.diagnostic),
        )
        self.assertTrue(
            all(item.optimizer.updates == 0 for item in no_updates.ablations)
        )

        row_cap = property_causal_block_integration(
            frame,
            q,
            warm,
            deadline=time.monotonic() + 5.0,
            selector_max_rows=2,
            **call,
        )
        self.assertFalse(row_cap.success)
        self.assertEqual(row_cap.status, "incidence_path_unavailable")
        self.assertEqual(row_cap.ablations, ())

        expired = property_causal_block_integration(
            frame,
            q,
            warm,
            deadline=time.monotonic() - 1.0,
            **call,
        )
        self.assertFalse(expired.success)
        self.assertEqual(expired.status, "deadline")
        self.assertTrue(expired.deadline_reached)
        self.assertEqual(expired.ablations, ())

        too_many_packet_rows = property_causal_block_integration(
            frame,
            q,
            warm,
            incidence_packet_rows=tuple(range(65)),
            optimization_packet_rows=tuple(range(65)),
            source_rows=(1,),
            allowed_row_mask=allowed,
            row_tags=tags,
            deadline=time.monotonic() + 5.0,
        )
        self.assertFalse(too_many_packet_rows.success)
        self.assertEqual(
            too_many_packet_rows.status,
            "optimization_packet_cap",
        )
        self.assertEqual(too_many_packet_rows.ablations, ())


if __name__ == "__main__":
    unittest.main()
