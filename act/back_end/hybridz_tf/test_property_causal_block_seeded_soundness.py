#!/usr/bin/env python3
# ===- test_property_causal_block_seeded_soundness.py ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Seeded soundness gates for incidence plus causal-block integration.

This file is intentionally independent of the historical C66 gate.  It uses
small dyadic DAGs, exact-rational primal oracles, row permutations, and paired
stable-ID/wrong-copy controls.  No candidate has proof authority until the
existing full-frame long-double checker accepts it.
"""

from __future__ import annotations

from dataclasses import dataclass
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


_SEED_COUNT = 256
_HIGHS_SEEDS = frozenset(range(0, _SEED_COUNT, 8))
_ABLATION_FAMILIES = {
    "full": frozenset(("generated", "bridge")),
    "without_generated": frozenset(("bridge",)),
    "without_bridge": frozenset(("generated",)),
    "without_both": frozenset(),
}


@dataclass(frozen=True)
class _SeedPair:
    shared: OriginalFrameLP
    wrong: OriginalFrameLP
    q: np.ndarray
    warm: np.ndarray
    allowed: np.ndarray
    packet_row: int
    canonical_to_permuted: tuple[int, ...]
    coefficient: Fraction
    objective_scale: Fraction
    radius: Fraction


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(2 ** exponent)
    return Fraction(1, 2 ** (-exponent))


def _seed_pair(seed: int) -> _SeedPair:
    """Return same-shape frames differing only in one stable column identity."""

    rng = np.random.default_rng(int(seed))
    coefficient = Fraction(int(rng.integers(1, 16)), 16)
    objective_scale = _power_of_two(int(rng.integers(-1, 2)))
    radius = _power_of_two(int(rng.integers(-1, 2)))
    permutation = np.asarray(rng.permutation(4), dtype=np.int64)

    # Stable columns are [y, x, w, x_copy, u, v].  The final two constraints
    # make a disconnected equality DAG component and exercise full expansion.
    shared_A = np.asarray(
        [
            [0.0, -float(coefficient), 1.0, 0.0, 0.0, 0.0],
            [1.0, float(coefficient), 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    wrong_A = shared_A.copy()
    wrong_A[0, 1] = 0.0
    wrong_A[0, 3] = -float(coefficient)

    radius64 = float(radius)
    lb = np.asarray(
        [-radius64, -radius64, 0.0, -radius64, -radius64, -radius64],
        dtype=np.float64,
    )
    ub = np.asarray(
        [radius64, radius64, 0.0, radius64, radius64, radius64],
        dtype=np.float64,
    )
    canonical_tags = (
        "property_micro_rlt:generated:seed_packet",
        "bridge:seed_shared_materialization",
        "background:seed_equality_forward",
        "background:seed_equality_reverse",
    )
    ordered_tags = tuple(
        ConstraintRowTag(
            global_row=position,
            sense="ub",
            block_tag=canonical_tags[int(canonical_row)],
            block_local_row=0,
        )
        for position, canonical_row in enumerate(permutation)
    )

    def frame(A: np.ndarray) -> OriginalFrameLP:
        return OriginalFrameLP(
            A=sp.csr_matrix(A[permutation, :]),
            rl=np.full(4, -np.inf, dtype=np.float64),
            ru=np.zeros(4, dtype=np.float64),
            lb=lb.copy(),
            ub=ub.copy(),
            row_tags=ordered_tags,
        )

    canonical_to_permuted = tuple(
        int(np.flatnonzero(permutation == canonical)[0])
        for canonical in range(4)
    )
    q = np.zeros((1, 6), dtype=np.float64)
    q[0, 0] = float(objective_scale)
    canonical_warm = np.asarray(
        [[0.0, float(objective_scale), 0.0, 0.0]],
        dtype=np.float64,
    )
    return _SeedPair(
        shared=frame(shared_A),
        wrong=frame(wrong_A),
        q=q,
        warm=canonical_warm[:, permutation],
        allowed=np.ones(4, dtype=np.bool_),
        packet_row=canonical_to_permuted[0],
        canonical_to_permuted=canonical_to_permuted,
        coefficient=coefficient,
        objective_scale=objective_scale,
        radius=radius,
    )


def _integrate(frame: OriginalFrameLP, pair: _SeedPair):
    return property_causal_block_integration(
        frame,
        pair.q,
        pair.warm,
        incidence_packet_rows=(pair.packet_row,),
        optimization_packet_rows=(pair.packet_row,),
        source_rows=(),
        allowed_row_mask=pair.allowed,
        row_tags=frame.row_tags,
        property_columns=(0,),
        optimizer_max_updates=4,
        optimizer_max_zero_gain_updates=2,
        optimizer_face_visit_cap=2,
        optimizer_frontier_topk=4,
        optimizer_nnz_cap=8,
    )


def _fraction_vertex_upper(
    *,
    coefficient: Fraction,
    objective_scale: Fraction,
    radius: Fraction,
    shared_id: bool,
    enabled_families: frozenset[str],
) -> Fraction:
    """Exact vertex oracle for the dyadic two-family DAG."""

    endpoints = (-radius, Fraction(0), radius)
    best = None
    for x in endpoints:
        for x_copy in endpoints:
            if "generated" in enabled_families:
                consumed = x if shared_id else x_copy
                if -coefficient * consumed > 0:
                    continue
            y_candidates = {-radius, radius}
            if "bridge" in enabled_families:
                y_candidates.add(-coefficient * x)
            for y in y_candidates:
                if y < -radius or y > radius:
                    continue
                if (
                    "bridge" in enabled_families
                    and y + coefficient * x > 0
                ):
                    continue
                value = objective_scale * y
                best = value if best is None else max(best, value)
    if best is None:
        raise AssertionError("dyadic vertex oracle found no feasible point")
    return best


def _fraction_to_longdouble(value: Fraction) -> np.longdouble:
    return (
        np.longdouble(value.numerator)
        / np.longdouble(value.denominator)
    )


def _checker_upper(
    frame: OriginalFrameLP,
    q: np.ndarray,
    row_dual: np.ndarray,
) -> tuple[np.longdouble | None, dict]:
    return _hz_independent_lp_lagrangian_upper(
        c=np.asarray([0.0], dtype=np.float64),
        Gc=sp.csr_matrix(np.asarray(q, dtype=np.float64)),
        C_row=np.asarray([1.0], dtype=np.float64),
        threshold=0.0,
        A=frame.A,
        rl=frame.rl,
        ru=frame.ru,
        lb=frame.lb,
        ub=frame.ub,
        row_dual=row_dual,
    )


def _highs_upper(
    frame: OriginalFrameLP,
    q: np.ndarray,
    rows: tuple[int, ...],
) -> float:
    selected = np.asarray(rows, dtype=np.int64)
    result = linprog(
        -np.asarray(q, dtype=np.float64).reshape(-1),
        A_ub=frame.A[selected, :] if selected.size else None,
        b_ub=frame.ru[selected] if selected.size else None,
        bounds=list(zip(frame.lb, frame.ub)),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(-result.fun)


class PropertyCausalBlockSeededSoundnessTests(unittest.TestCase):
    def test_256_seeded_shared_id_wrong_copy_pairs(self) -> None:
        started = time.perf_counter()
        highs_checks = 0
        checked_candidates = 0

        for seed in range(_SEED_COUNT):
            with self.subTest(seed=seed):
                pair = _seed_pair(seed)
                shared = _integrate(pair.shared, pair)
                wrong = _integrate(pair.wrong, pair)

                self.assertTrue(
                    shared.success,
                    (shared.status, shared.diagnostic),
                )
                self.assertEqual(
                    shared.status,
                    "candidate_ready_unchecked",
                )
                self.assertFalse(shared.proof_authority)
                np.testing.assert_array_equal(
                    shared.cone_rows,
                    np.asarray(
                        [pair.canonical_to_permuted[1]],
                        dtype=np.int64,
                    ),
                )

                # The copied stable coordinate destroys CSR incidence.  No
                # partial optimizer candidate may escape this fail-closed gate.
                self.assertFalse(wrong.success)
                self.assertEqual(
                    wrong.status,
                    "incidence_path_unavailable",
                )
                self.assertEqual(wrong.ablations, ())
                wrong_exact = _fraction_vertex_upper(
                    coefficient=pair.coefficient,
                    objective_scale=pair.objective_scale,
                    radius=pair.radius,
                    shared_id=False,
                    enabled_families=frozenset(
                        ("generated", "bridge")
                    ),
                )
                self.assertGreater(wrong_exact, 0)

                supports: dict[str, Fraction] = {}
                checked_uppers: dict[str, np.longdouble] = {}
                for name, families in _ABLATION_FAMILIES.items():
                    candidate = shared.ablation(name)
                    checked_candidates += 1
                    self.assertEqual(
                        candidate.d.shape,
                        (1, pair.shared.n_rows),
                    )
                    self.assertTrue(np.all(np.isfinite(candidate.d)))
                    # Integration exposes maximization multipliers d and the
                    # checker's HiGHS/minimization convention row_dual=-d.
                    self.assertTrue(np.all(candidate.d >= 0.0))
                    self.assertTrue(np.all(candidate.row_dual <= 0.0))
                    np.testing.assert_array_equal(
                        candidate.row_dual,
                        -candidate.d,
                    )

                    exact = _fraction_vertex_upper(
                        coefficient=pair.coefficient,
                        objective_scale=pair.objective_scale,
                        radius=pair.radius,
                        shared_id=True,
                        enabled_families=families,
                    )
                    supports[name] = exact
                    upper, receipt = _checker_upper(
                        pair.shared,
                        pair.q,
                        candidate.row_dual[0],
                    )
                    self.assertEqual(
                        receipt["status"],
                        "verified_upper",
                    )
                    self.assertEqual(
                        receipt["illegal_sign_projected"],
                        0,
                    )
                    self.assertEqual(
                        receipt["nonfinite_dual_zeroed"],
                        0,
                    )
                    self.assertIsNotNone(upper)
                    checked_uppers[name] = np.longdouble(upper)
                    self.assertGreaterEqual(
                        np.longdouble(upper),
                        _fraction_to_longdouble(exact),
                    )
                    self.assertAlmostEqual(
                        float(candidate.candidate_support[0]),
                        float(exact),
                        places=13,
                    )

                self.assertLess(supports["full"], supports["without_generated"])
                self.assertLess(
                    supports["without_generated"],
                    supports["without_bridge"],
                )
                self.assertEqual(
                    supports["without_bridge"],
                    supports["without_both"],
                )

                if seed in _HIGHS_SEEDS:
                    background = (
                        pair.canonical_to_permuted[2],
                        pair.canonical_to_permuted[3],
                    )
                    family_rows = {
                        "generated": pair.canonical_to_permuted[0],
                        "bridge": pair.canonical_to_permuted[1],
                    }
                    for name, families in _ABLATION_FAMILIES.items():
                        rows = tuple(
                            sorted(
                                background
                                + tuple(
                                    family_rows[family]
                                    for family in sorted(families)
                                )
                            )
                        )
                        optimum = _highs_upper(
                            pair.shared,
                            pair.q[0],
                            rows,
                        )
                        self.assertAlmostEqual(
                            optimum,
                            float(supports[name]),
                            places=10,
                        )
                        self.assertGreaterEqual(
                            checked_uppers[name],
                            np.longdouble(optimum),
                        )
                        highs_checks += 1
                    wrong_optimum = _highs_upper(
                        pair.wrong,
                        pair.q[0],
                        tuple(range(pair.wrong.n_rows)),
                    )
                    self.assertAlmostEqual(
                        wrong_optimum,
                        float(wrong_exact),
                        places=10,
                    )
                    highs_checks += 1

        elapsed = time.perf_counter() - started
        self.assertEqual(checked_candidates, 4 * _SEED_COUNT)
        self.assertEqual(highs_checks, 5 * len(_HIGHS_SEEDS))
        print(
            "seeded PC-CBDE soundness gate: "
            f"seeds={_SEED_COUNT} "
            f"candidates={checked_candidates} "
            f"HiGHS={highs_checks} "
            f"wall={elapsed:.6f}s"
        )
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
