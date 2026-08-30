#!/usr/bin/env python3
# ===- test_property_micro_rlt.py ---------------------------------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# ===----------------------------------------------------------------===#
"""Soundness and tightness gates for the bounded property micro-RLT lift."""

from __future__ import annotations

import copy
from fractions import Fraction
import itertools
import math
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.hybridz_tf import property_micro_rlt as micro_rlt
from act.back_end.hybridz_tf.property_micro_rlt import (
    PropertyMicroRLTError,
    apply_property_micro_rlt,
    verify_property_micro_rlt_receipt,
    verify_property_micro_rlt_result,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_fresh_col_ids,
)


_SELECTION = {
    0: [0, 1, 2, 3],
    1: [0, 3, 4, 5],
}


def _binary_product_toy() -> SparseHZono:
    """One row whose two multiplication orientations share ``s0*s1``."""

    stable_ids = (
        hz_fresh_col_ids(3, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return SparseHZono(
        c=np.array([0.0], dtype=np.float64),
        Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
        Gb=sp.csr_matrix((1, 2), dtype=np.float64),
        Ac=sp.csr_matrix((0, 1), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix([[0.0]], dtype=np.float64),
        Aub=sp.csr_matrix([[2.0, 3.0]], dtype=np.float64),
        ub=np.array([4.0], dtype=np.float64),
        col_ids=stable_ids[:1],
        bcol_ids=stable_ids[1:],
    )


def _duplicate_relu_toy() -> SparseHZono:
    """Two exact ReLUs of one shared x, with y_i=(1+t_i)/2.

    For binary phase ``s_i=-1``, the three rows below impose x <= 0 and
    t_i=-1.  For ``s_i=1``, they impose x >= 0 and t_i=2*x-1.  Thus the
    integer set has y_1=y_2=ReLU(x), while independently relaxing the two
    phase factors permits max(y_2-y_1)=1/2.
    """

    stable_ids = (
        hz_fresh_col_ids(5, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return SparseHZono(
        c=np.array([0.5, 0.5], dtype=np.float64),
        Gc=sp.csr_matrix(
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=np.float64,
        ),
        Gb=sp.csr_matrix((2, 2), dtype=np.float64),
        Ac=sp.csr_matrix((0, 3), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            [
                [1.0, -0.5, 0.0],
                [-1.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.0, -0.5],
                [-1.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
            ],
            dtype=np.float64,
        ),
        Aub=sp.csr_matrix(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [-0.5, 0.0],
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, -0.5],
            ],
            dtype=np.float64,
        ),
        ub=np.array(
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            dtype=np.float64,
        ),
        col_ids=stable_ids[:3],
        bcol_ids=stable_ids[3:],
    )


def _scale_4100_factor_toy() -> SparseHZono:
    """Six sparse rows with the real iid2 requirement envelope.

    Each directed four-row packet has 6,150 selected nonzeros and a union of
    2,050 continuous factors.  The two packets together therefore reproduce
    the observed 12,300/4,100 requirement pair without a large network.
    """

    width = 2050
    half = width // 2

    def row(columns: int) -> sp.csr_matrix:
        return sp.csr_matrix(
            (
                np.ones(columns, dtype=np.float64),
                (
                    np.zeros(columns, dtype=np.int64),
                    np.arange(columns, dtype=np.int64),
                ),
            ),
            shape=(1, width),
            dtype=np.float64,
        )

    Auc = sp.vstack(
        [
            row(width),
            row(half),
            row(half),
            row(width),
            row(half),
            row(half),
        ],
        format="csr",
    )
    stable_ids = (
        hz_fresh_col_ids(width + 2, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return SparseHZono(
        c=np.asarray([0.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, width), dtype=np.float64),
        Gb=sp.csr_matrix((1, 2), dtype=np.float64),
        Ac=sp.csr_matrix((0, width), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=Auc,
        Aub=sp.csr_matrix((6, 2), dtype=np.float64),
        ub=np.full(6, 4096.0, dtype=np.float64),
        col_ids=stable_ids[:width],
        bcol_ids=stable_ids[width:],
    )


def _factor_support(
    hz: SparseHZono,
    *,
    continuous: np.ndarray,
    binary: np.ndarray | None = None,
    constant: float = 0.0,
    phases: tuple[int, ...] | None = None,
    keep_upper: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Maximize one affine factor expression over the stored relaxation."""

    continuous = np.asarray(continuous, dtype=np.float64).reshape(-1)
    if continuous.size != hz.n_cont:
        raise AssertionError("continuous objective width mismatch")
    if binary is None:
        binary = np.zeros(hz.n_bin, dtype=np.float64)
    binary = np.asarray(binary, dtype=np.float64).reshape(-1)
    if binary.size != hz.n_bin:
        raise AssertionError("binary objective width mismatch")

    A_ub = sp.hstack([hz.Auc, hz.Aub], format="csr")
    b_ub = np.asarray(hz.ub, dtype=np.float64)
    if keep_upper is not None:
        keep_upper = np.asarray(keep_upper, dtype=bool)
        A_ub = A_ub[keep_upper]
        b_ub = b_ub[keep_upper]
    A_eq = sp.hstack([hz.Ac, hz.Ab], format="csr")
    objective = np.concatenate([continuous, binary])
    bounds = [(-1.0, 1.0)] * (hz.n_cont + hz.n_bin)
    if phases is not None:
        if len(phases) != hz.n_bin:
            raise AssertionError("phase width mismatch")
        for offset, phase in enumerate(phases):
            bounds[hz.n_cont + offset] = (float(phase), float(phase))

    result = linprog(
        -objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=hz.b,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise AssertionError(
            f"toy LP unexpectedly failed: {result.status}: {result.message}"
        )
    return float(constant - result.fun), np.asarray(
        result.x, dtype=np.float64
    )


def _output_support(
    hz: SparseHZono,
    output_direction: np.ndarray,
    *,
    phases: tuple[int, ...] | None = None,
    keep_upper: np.ndarray | None = None,
) -> float:
    output_direction = np.asarray(
        output_direction, dtype=np.float64
    ).reshape(-1)
    return _factor_support(
        hz,
        continuous=np.asarray(
            output_direction @ hz.Gc, dtype=np.float64
        ).reshape(-1),
        binary=np.asarray(
            output_direction @ hz.Gb, dtype=np.float64
        ).reshape(-1),
        constant=float(output_direction @ hz.c),
        phases=phases,
        keep_upper=keep_upper,
    )[0]


def _exact_product_extension(
    *,
    receipt: dict,
    old_continuous: np.ndarray,
    phases: tuple[int, ...],
) -> np.ndarray:
    result = np.zeros(
        int(receipt["result_n_cont"]), dtype=np.float64
    )
    result[: int(receipt["base_n_cont"])] = old_continuous
    for record in receipt["product_records"]:
        selected = phases[int(record["binary_position"])]
        factor_position = int(record["factor_position"])
        if record["factor_kind"] == "continuous":
            factor = old_continuous[factor_position]
        elif record["factor_kind"] == "binary":
            factor = phases[factor_position]
        else:
            raise AssertionError("unknown product factor kind")
        result[int(record["aux_continuous_position"])] = (
            selected * factor
        )
    return result


class PropertyMicroRLTToyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = _duplicate_relu_toy()
        self.lift = apply_property_micro_rlt(
            self.base,
            source_rows_by_binary=_SELECTION,
        )

    def test_duplicate_relu_relaxation_gap_closes(self) -> None:
        direction = np.array([-1.0, 1.0], dtype=np.float64)
        before = _output_support(self.base, direction)
        after = _output_support(self.lift.hz, direction)

        self.assertAlmostEqual(before, 0.5, places=10)
        self.assertAlmostEqual(after, 0.0, places=10)
        self.assertEqual(self.lift.receipt["new_product_factors"], 6)
        self.assertEqual(self.lift.receipt["new_product_hull_rows"], 24)
        self.assertEqual(self.lift.receipt["new_rlt_rows"], 16)
        self.assertEqual(self.lift.receipt["new_upper_rows"], 40)
        self.assertEqual(self.lift.hz.n_ub, 46)

    def test_gain_exists_only_before_exact_phase_enumeration(self) -> None:
        direction = np.array([-1.0, 1.0], dtype=np.float64)
        self.assertEqual(
            self.lift.receipt["intended_consumer"],
            "parent_binary_relaxation_before_exact_phase_enumeration",
        )
        self.assertTrue(
            self.lift.receipt["fixed_phase_rows_are_redundant"]
        )
        self.assertAlmostEqual(
            _output_support(self.base, direction),
            0.5,
            places=10,
        )
        self.assertAlmostEqual(
            _output_support(self.lift.hz, direction),
            0.0,
            places=10,
        )
        for phases in itertools.product((-1, 1), repeat=2):
            with self.subTest(phases=phases):
                before = _output_support(
                    self.base, direction, phases=phases
                )
                after = _output_support(
                    self.lift.hz, direction, phases=phases
                )
                self.assertAlmostEqual(before, 0.0, places=10)
                self.assertAlmostEqual(after, before, places=10)

    def test_one_necessary_rlt_side_is_a_decisive_negative_control(
        self,
    ) -> None:
        direction = np.array([-1.0, 1.0], dtype=np.float64)
        row_name = "rlt[1,0].plus"
        local_row = self.lift.receipt["generated_row_names"].index(
            row_name
        )
        keep = np.ones(self.lift.hz.n_ub, dtype=bool)
        keep[self.base.n_ub + local_row] = False

        complete = _output_support(self.lift.hz, direction)
        missing_side = _output_support(
            self.lift.hz,
            direction,
            keep_upper=keep,
        )
        self.assertAlmostEqual(complete, 0.0, places=10)
        self.assertAlmostEqual(missing_side, 0.5, places=10)

    def test_every_exact_phase_preserves_the_base_projection(self) -> None:
        receipt = self.lift.receipt
        generated_start = self.base.n_ub
        generated_Auc = self.lift.hz.Auc[generated_start:]
        generated_Aub = self.lift.hz.Aub[generated_start:]
        generated_ub = self.lift.hz.ub[generated_start:]
        product_by_aux = {
            int(record["aux_continuous_position"]): record
            for record in receipt["product_records"]
        }

        # The lifted system retains every base row.  Conversely, with phases
        # fixed, substitute v=s*q into every generated row and maximize its
        # violation over the entire base phase polytope.  Nonpositive maxima
        # prove that every base point has a valid exact product extension.
        for phases in itertools.product((-1, 1), repeat=2):
            for row in range(generated_Auc.shape[0]):
                continuous = np.asarray(
                    generated_Auc[row, : self.base.n_cont].toarray()
                ).reshape(-1)
                constant = (
                    float(
                        generated_Aub[row].toarray().reshape(-1)
                        @ np.asarray(phases)
                    )
                    - float(generated_ub[row])
                )
                for aux in range(
                    self.base.n_cont, self.lift.hz.n_cont
                ):
                    coefficient = float(generated_Auc[row, aux])
                    if coefficient == 0.0:
                        continue
                    record = product_by_aux[aux]
                    selected = phases[
                        int(record["binary_position"])
                    ]
                    factor_position = int(record["factor_position"])
                    if record["factor_kind"] == "continuous":
                        continuous[factor_position] += (
                            coefficient * selected
                        )
                    else:
                        constant += (
                            coefficient
                            * selected
                            * phases[factor_position]
                        )
                maximum_violation, _ = _factor_support(
                    self.base,
                    continuous=continuous,
                    constant=constant,
                    phases=phases,
                )
                self.assertLessEqual(
                    maximum_violation,
                    2.0e-9,
                    msg=(
                        f"phase={phases}, generated_row={row}, "
                        f"name={receipt['generated_row_names'][row]}"
                    ),
                )

            # A direct solver-level cross-check covers output objectives and
            # all original continuous-factor coordinate directions.
            directions = [
                np.array([-1.0, 1.0]),
                np.array([1.0, -1.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
            ]
            for direction in directions:
                self.assertAlmostEqual(
                    _output_support(
                        self.base, direction, phases=phases
                    ),
                    _output_support(
                        self.lift.hz, direction, phases=phases
                    ),
                    places=9,
                )
            for old_factor in range(self.base.n_cont):
                for sign in (-1.0, 1.0):
                    base_objective = np.zeros(
                        self.base.n_cont, dtype=np.float64
                    )
                    base_objective[old_factor] = sign
                    lifted_objective = np.zeros(
                        self.lift.hz.n_cont, dtype=np.float64
                    )
                    lifted_objective[: self.base.n_cont] = (
                        base_objective
                    )
                    self.assertAlmostEqual(
                        _factor_support(
                            self.base,
                            continuous=base_objective,
                            phases=phases,
                        )[0],
                        _factor_support(
                            self.lift.hz,
                            continuous=lifted_objective,
                            phases=phases,
                        )[0],
                        places=9,
                    )

    def test_every_generated_row_is_sound_on_true_relu_points(
        self,
    ) -> None:
        receipt = self.lift.receipt
        phase_points = []
        for x in np.linspace(-1.0, 1.0, 33):
            t = 2.0 * max(0.0, float(x)) - 1.0
            if x < 0.0:
                phases = [(-1, -1)]
            elif x > 0.0:
                phases = [(1, 1)]
            else:
                phases = list(
                    itertools.product((-1, 1), repeat=2)
                )
            for phase in phases:
                phase_points.append(
                    (
                        np.array([x, t, t], dtype=np.float64),
                        phase,
                    )
                )

        worst = -np.inf
        for old_continuous, phases in phase_points:
            continuous = _exact_product_extension(
                receipt=receipt,
                old_continuous=old_continuous,
                phases=phases,
            )
            lhs = (
                self.lift.hz.Auc @ continuous
                + self.lift.hz.Aub @ np.asarray(phases)
            )
            violation = np.asarray(
                lhs - self.lift.hz.ub, dtype=np.float64
            )
            worst = max(worst, float(np.max(violation)))
        self.assertLessEqual(worst, 2.0e-12)

    def test_binary_product_is_shared_across_both_orientations(
        self,
    ) -> None:
        base = _binary_product_toy()
        lift = apply_property_micro_rlt(
            base,
            source_rows_by_binary={1: [0], 0: [0]},
        )
        receipt = lift.receipt
        self.assertTrue(verify_property_micro_rlt_result(lift))
        self.assertEqual(receipt["new_product_factors"], 1)
        self.assertEqual(receipt["new_product_hull_rows"], 4)
        self.assertEqual(receipt["new_rlt_rows"], 4)
        self.assertEqual(receipt["new_upper_rows"], 8)

        record = receipt["product_records"][0]
        self.assertEqual(record["factor_kind"], "binary")
        self.assertEqual(record["unordered_binary_pair"], [0, 1])
        self.assertTrue(record["commutative"])
        self.assertEqual(
            record["orientation_uses"],
            [
                {
                    "selected_binary_position": 0,
                    "other_binary_position": 1,
                    "orientation_sign": 1,
                },
                {
                    "selected_binary_position": 1,
                    "other_binary_position": 0,
                    "orientation_sign": 1,
                },
            ],
        )

        aux = int(record["aux_continuous_position"])
        generated_start = base.n_ub
        names = receipt["generated_row_names"]
        expected = {
            "rlt[0,0].plus": (
                np.array([-2.0, 3.0]),
                3.0,
                2.0,
            ),
            "rlt[0,0].minus": (
                np.array([6.0, 3.0]),
                -3.0,
                6.0,
            ),
            "rlt[1,0].plus": (
                np.array([2.0, -1.0]),
                2.0,
                1.0,
            ),
            "rlt[1,0].minus": (
                np.array([2.0, 7.0]),
                -2.0,
                7.0,
            ),
        }
        for name, (binary, product, rhs) in expected.items():
            row = generated_start + names.index(name)
            np.testing.assert_array_equal(
                lift.hz.Aub[row].toarray().reshape(-1),
                binary,
            )
            self.assertEqual(float(lift.hz.Auc[row, aux]), product)
            self.assertEqual(float(lift.hz.ub[row]), rhs)

        for phases in itertools.product((-1, 1), repeat=2):
            if 2 * phases[0] + 3 * phases[1] > 4:
                continue
            old_continuous = np.array([0.25], dtype=np.float64)
            continuous = _exact_product_extension(
                receipt=receipt,
                old_continuous=old_continuous,
                phases=phases,
            )
            lhs = (
                lift.hz.Auc @ continuous
                + lift.hz.Aub @ np.asarray(phases)
            )
            self.assertLessEqual(
                float(np.max(lhs - lift.hz.ub)), 1.0e-12
            )

    def test_fraction_storage_guard_covers_non_dyadic_and_extreme_rows(
        self,
    ) -> None:
        cases = [
            (
                {
                    0: Fraction.from_float(0.1)
                    + Fraction.from_float(0.2),
                    1: Fraction.from_float(-0.3)
                    + Fraction.from_float(0.05),
                },
                {
                    0: Fraction.from_float(0.7)
                    - Fraction.from_float(0.2)
                },
                Fraction.from_float(0.9)
                - Fraction.from_float(0.1),
            ),
            (
                {
                    0: Fraction.from_float(
                        math.nextafter(1.0e200, math.inf)
                    )
                    - Fraction.from_float(1.0e200),
                    1: Fraction.from_float(1.0e-200)
                    + Fraction.from_float(1.0e-216),
                },
                {0: Fraction.from_float(-1.0e150)},
                Fraction.from_float(1.0e200)
                + Fraction.from_float(1.0e184),
            ),
            (
                {
                    # Exactly half of the least positive binary64 subnormal
                    # rounds to zero.  The lost coefficient mass must move to
                    # the outward RHS, not silently disappear.
                    0: Fraction(1, 2**1075),
                },
                {},
                Fraction(0),
            ),
            (
                {
                    # Midpoint between 1 and its successor exercises a
                    # half-ULP coefficient tie.
                    0: Fraction.from_float(1.0)
                    + (
                        Fraction.from_float(
                            math.nextafter(1.0, math.inf)
                        )
                        - Fraction.from_float(1.0)
                    )
                    / 2,
                    1: -Fraction(1, 2**1075),
                },
                {0: Fraction(1, 3)},
                Fraction(1, 4),
            ),
        ]
        for index, (cont, binary, rhs) in enumerate(cases):
            stored_cont, stored_binary, stored_rhs = (
                micro_rlt._store_upper_row(
                    cont=cont,
                    binary=binary,
                    rhs=rhs,
                    n_cont=2,
                    n_bin=1,
                    row_name=f"rounding_case_{index}",
                )
            )
            coefficient_error = sum(
                (
                    abs(Fraction.from_float(stored_cont[column]) - value)
                    for column, value in cont.items()
                ),
                Fraction(0),
            ) + sum(
                (
                    abs(
                        Fraction.from_float(stored_binary[column])
                        - value
                    )
                    for column, value in binary.items()
                ),
                Fraction(0),
            )
            self.assertGreaterEqual(
                Fraction.from_float(stored_rhs),
                rhs + coefficient_error,
            )
            self.assertTrue(math.isfinite(stored_rhs))

            for factors in itertools.product((-1, 1), repeat=3):
                exact_lhs = sum(
                    value * factors[column]
                    for column, value in cont.items()
                ) + sum(
                    value * factors[2 + column]
                    for column, value in binary.items()
                )
                if exact_lhs > rhs:
                    continue
                stored_lhs = sum(
                    Fraction.from_float(value) * factors[column]
                    for column, value in stored_cont.items()
                ) + sum(
                    Fraction.from_float(value)
                    * factors[2 + column]
                    for column, value in stored_binary.items()
                )
                self.assertLessEqual(
                    stored_lhs, Fraction.from_float(stored_rhs)
                )

    def test_global_aux_ids_are_fresh_and_collision_fails_closed(
        self,
    ) -> None:
        base_ids = {
            int(value)
            for value in np.concatenate(
                [self.base.col_ids, self.base.bcol_ids]
            )
        }
        result_ids = [
            int(value) for value in self.lift.hz.col_ids
        ]
        aux_ids = result_ids[self.base.n_cont :]
        self.assertEqual(len(aux_ids), len(set(aux_ids)))
        self.assertTrue(base_ids.isdisjoint(aux_ids))

        n_aux = int(self.lift.receipt["new_product_factors"])
        colliding = torch.tensor(
            [
                int(self.base.col_ids[0]),
                *range(10_000_000, 10_000_000 + n_aux - 1),
            ],
            dtype=torch.long,
        )
        with mock.patch.object(
            micro_rlt,
            "hz_fresh_col_ids",
            return_value=colliding,
        ):
            with self.assertRaises(PropertyMicroRLTError):
                apply_property_micro_rlt(
                    self.base,
                    source_rows_by_binary=_SELECTION,
                )

    def test_live_result_validator_rejects_material_and_digest_tamper(
        self,
    ) -> None:
        self.assertTrue(verify_property_micro_rlt_result(self.lift))

        mutations = []
        changed_rhs = copy.deepcopy(self.lift)
        changed_rhs.hz.ub[-1] = np.nextafter(
            changed_rhs.hz.ub[-1], math.inf
        )
        mutations.append(changed_rhs)

        changed_matrix = copy.deepcopy(self.lift)
        changed_matrix.hz.Auc.data[-1] = np.nextafter(
            changed_matrix.hz.Auc.data[-1], math.inf
        )
        mutations.append(changed_matrix)

        changed_col_ids = copy.deepcopy(self.lift)
        changed_col_ids.hz.col_ids[-1] += 10_000_000
        mutations.append(changed_col_ids)

        changed_binary_ids = copy.deepcopy(self.lift)
        changed_binary_ids.hz.bcol_ids[:] = (
            changed_binary_ids.hz.bcol_ids[::-1]
        )
        mutations.append(changed_binary_ids)

        changed_output = copy.deepcopy(self.lift)
        output = changed_output.hz.Gc.tolil(copy=True)
        output[0, self.base.n_cont] = 1.0
        changed_output.hz.Gc = output.tocsr()
        mutations.append(changed_output)

        changed_equality = copy.deepcopy(self.lift)
        equality = changed_equality.hz.Ac.tolil(copy=True)
        if equality.shape[0] == 0:
            equality = sp.csr_matrix(
                ([1.0], ([0], [self.base.n_cont])),
                shape=(1, changed_equality.hz.n_cont),
                dtype=np.float64,
            )
            changed_equality.hz.Ac = equality
        else:
            equality[0, self.base.n_cont] = 1.0
            changed_equality.hz.Ac = equality.tocsr()
        mutations.append(changed_equality)

        for changed in mutations:
            with self.subTest(kind=id(changed)):
                self.assertFalse(
                    verify_property_micro_rlt_result(changed)
                )

        changed_dimensions = copy.deepcopy(self.lift)
        changed_dimensions.receipt["result_n_cont"] += 1
        changed_dimensions.receipt["receipt_sha256"] = (
            micro_rlt._canonical_sha256(
                {
                    key: value
                    for key, value in changed_dimensions.receipt.items()
                    if key != "receipt_sha256"
                }
            )
        )
        changed_dimensions.hz._property_micro_rlt_receipt = dict(
            changed_dimensions.receipt
        )
        self.assertTrue(
            verify_property_micro_rlt_receipt(
                changed_dimensions.receipt
            )
        )
        self.assertFalse(
            verify_property_micro_rlt_result(changed_dimensions)
        )

    def test_receipt_is_deterministic_and_tamper_evident(self) -> None:
        receipt = self.lift.receipt
        self.assertTrue(verify_property_micro_rlt_receipt(receipt))
        self.assertTrue(verify_property_micro_rlt_result(self.lift))
        self.assertEqual(receipt["binary_factor_encoding"], "signed_pm1")
        self.assertEqual(
            receipt["solver_relaxation_encoding"],
            "z_in_0_1_with_signed_factor_s_equals_2z_minus_1",
        )
        self.assertEqual(
            micro_rlt._csr_sha256(self.lift.hz.Auc),
            receipt["result_upper_csr_sha256"],
        )
        self.assertEqual(
            micro_rlt._csr_sha256(self.lift.hz.Aub),
            receipt["result_upper_binary_csr_sha256"],
        )

        reordered = apply_property_micro_rlt(
            self.base,
            source_rows_by_binary={
                1: [5, 4, 3, 0],
                0: [3, 2, 1, 0],
            },
        )
        # Global stable IDs are intentionally process-order dependent.  Every
        # structural field remains deterministic after excluding the fresh-ID
        # digest and the receipt hash which transitively binds it.
        structural = dict(receipt)
        reordered_structural = dict(reordered.receipt)
        for payload in (structural, reordered_structural):
            payload.pop("result_continuous_col_ids_sha256")
            payload.pop("receipt_sha256")
        self.assertEqual(structural, reordered_structural)
        self.assertEqual(
            micro_rlt._csr_sha256(self.lift.hz.Auc),
            micro_rlt._csr_sha256(reordered.hz.Auc),
        )

        for key, replacement in (
            ("new_rlt_rows", 15),
            ("integer_set_equivalent", False),
            ("result_upper_csr_sha256", "0" * 64),
        ):
            with self.subTest(key=key):
                tampered = copy.deepcopy(receipt)
                tampered[key] = replacement
                self.assertFalse(
                    verify_property_micro_rlt_receipt(tampered)
                )
        missing_hash = copy.deepcopy(receipt)
        del missing_hash["receipt_sha256"]
        self.assertFalse(
            verify_property_micro_rlt_receipt(missing_hash)
        )
        self.assertFalse(verify_property_micro_rlt_receipt({}))

    def test_invalid_requests_and_caps_fail_closed(self) -> None:
        invalid_requests = [
            ({}, {}),
            ({2: [0]}, {}),
            ({True: [0]}, {}),
            ({0: [0, 0]}, {}),
            ({0: [6]}, {}),
            (_SELECTION, {"max_binary_factors": 1}),
            ({0: [0, 1, 2, 3]}, {"max_source_rows_per_binary": 3}),
            (_SELECTION, {"max_product_factors": 5}),
            ({0: [0]}, {"max_selected_row_nnz": 1}),
            ({0: [0]}, {"max_binary_factors": 0}),
            ({0: [0]}, {"max_requirement_scan_nnz": 0}),
            ({0: [True]}, {}),
            ({0: [np.bool_(True)]}, {}),
            ({0: [0.0]}, {}),
            ({0: [np.float64(0.0)]}, {}),
            ({0: ["0"]}, {}),
            ({0: [None]}, {}),
            ({0: None}, {}),
            ({0: 0}, {}),
        ]
        for selection, caps in invalid_requests:
            with self.subTest(selection=selection, caps=caps):
                with self.assertRaises(PropertyMicroRLTError):
                    apply_property_micro_rlt(
                        self.base,
                        source_rows_by_binary=selection,
                        **caps,
                    )

        with self.assertRaisesRegex(
            PropertyMicroRLTError,
            r"product-factor cap exceeded: required=6, cap=5",
        ) as product_error:
            apply_property_micro_rlt(
                self.base,
                source_rows_by_binary=_SELECTION,
                max_product_factors=5,
            )
        self.assertEqual(
            product_error.exception.product_factors_required, 6
        )
        self.assertEqual(
            product_error.exception.selected_source_row_nnz_required,
            18,
        )
        with self.assertRaisesRegex(
            PropertyMicroRLTError,
            r"source-row nnz cap exceeded: required=2, cap=1",
        ) as source_error:
            apply_property_micro_rlt(
                self.base,
                source_rows_by_binary={0: [0]},
                max_selected_row_nnz=1,
            )
        self.assertEqual(
            source_error.exception.selected_source_row_nnz_required,
            2,
        )
        self.assertEqual(
            source_error.exception.product_factors_required, 2
        )
        with mock.patch.object(
            micro_rlt,
            "_continuous_product_key",
            wraps=micro_rlt._continuous_product_key,
        ) as product_key:
            with self.assertRaisesRegex(
                PropertyMicroRLTError,
                r"requirement-scan nnz cap exceeded.*raw_required=2, "
                r"cap=1",
            ) as scan_error:
                apply_property_micro_rlt(
                    self.base,
                    source_rows_by_binary={0: [0]},
                    max_requirement_scan_nnz=1,
                )
        self.assertIsNone(
            scan_error.exception.selected_source_row_nnz_required
        )
        self.assertIsNone(
            scan_error.exception.product_factors_required
        )
        product_key.assert_not_called()

        with self.assertRaises(TypeError):
            apply_property_micro_rlt(
                object(),
                source_rows_by_binary={0: [0]},
            )

    def test_real_requirement_scale_builds_transactionally(self) -> None:
        base = _scale_4100_factor_toy()
        full = apply_property_micro_rlt(
            base,
            source_rows_by_binary=_SELECTION,
            max_binary_factors=2,
            max_source_rows_per_binary=4,
            max_product_factors=4100,
            max_selected_row_nnz=12300,
            max_requirement_scan_nnz=12300,
        )
        self.assertEqual(
            full.receipt["selected_source_row_nnz"], 12300
        )
        self.assertEqual(full.receipt["new_product_factors"], 4100)
        self.assertEqual(full.receipt["new_upper_rows"], 16416)
        self.assertEqual(full.hz.n_cont, base.n_cont + 4100)
        self.assertEqual(full.hz.n_ub, base.n_ub + 16416)
        self.assertTrue(verify_property_micro_rlt_result(full))

        directed = apply_property_micro_rlt(
            base,
            source_rows_by_binary={1: _SELECTION[1]},
            max_binary_factors=1,
            max_source_rows_per_binary=4,
            max_product_factors=2050,
            max_selected_row_nnz=6150,
            max_requirement_scan_nnz=6150,
        )
        self.assertEqual(
            directed.receipt["selected_source_row_nnz"], 6150
        )
        self.assertEqual(
            directed.receipt["new_product_factors"], 2050
        )
        self.assertEqual(directed.receipt["new_upper_rows"], 8208)
        self.assertTrue(verify_property_micro_rlt_result(directed))

        no_upper = SparseHZono(
            c=self.base.c,
            Gc=self.base.Gc,
            Gb=self.base.Gb,
            Ac=self.base.Ac,
            Ab=self.base.Ab,
            b=self.base.b,
        )
        with self.assertRaises(PropertyMicroRLTError):
            apply_property_micro_rlt(
                no_upper,
                source_rows_by_binary={0: [0]},
            )

        nonfinite_Auc = self.base.Auc.copy()
        nonfinite_Auc.data[0] = np.nan
        nonfinite = SparseHZono(
            c=self.base.c,
            Gc=self.base.Gc,
            Gb=self.base.Gb,
            Ac=self.base.Ac,
            Ab=self.base.Ab,
            b=self.base.b,
            Auc=nonfinite_Auc,
            Aub=self.base.Aub,
            ub=self.base.ub,
        )
        with self.assertRaises(PropertyMicroRLTError):
            apply_property_micro_rlt(
                nonfinite,
                source_rows_by_binary={0: [0]},
            )


if __name__ == "__main__":
    unittest.main()
