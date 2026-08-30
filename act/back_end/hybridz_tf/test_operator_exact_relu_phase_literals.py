#!/usr/bin/env python3
"""Controlled audits for Operator-HZ exact-ReLU phase-literal selection."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import itertools
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.adaptive_phase_forest import RivalSpec
from act.back_end.hybridz_tf import (
    operator_exact_relu_phase_literals as phase_adapter,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseLiteralError,
    OperatorExactReLUPhaseSelectionCaps,
    derive_operator_exact_relu_property_phase_literals,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    build_operator_hz,
)
from act.back_end.solver.solver_hz import SparseHZono


_DTYPE = torch.float64


def _layer(
    layer_id: int,
    kind: str,
    params: Mapping[str, Any] | None = None,
    *,
    width: int,
) -> Any:
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[
            (int(layer_id), row) for row in range(int(width))
        ],
    )


def _dense(
    layer_id: int,
    weight: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> Any:
    weight_array = np.asarray(weight, dtype=np.float64)
    bias_array = np.asarray(bias, dtype=np.float64)
    if (
        weight_array.ndim != 2
        or bias_array.ndim != 1
        or weight_array.shape[0] != bias_array.size
    ):
        raise AssertionError("malformed dense toy")
    return _layer(
        layer_id,
        "DENSE",
        {
            "weight": torch.tensor(weight_array, dtype=_DTYPE),
            "bias": torch.tensor(bias_array, dtype=_DTYPE),
            "in_features": int(weight_array.shape[1]),
            "out_features": int(weight_array.shape[0]),
        },
        width=int(weight_array.shape[0]),
    )


def _k4_corner_build() -> OperatorHZBuild:
    """Four exact ReLUs with mutually exclusive positive corner phases."""

    lower = torch.tensor([[-1.0, -1.0]], dtype=_DTYPE)
    upper = torch.tensor([[1.0, 1.0]], dtype=_DTYPE)
    layers = [
        _layer(
            0,
            "INPUT",
            {"shape": (1, 2)},
            width=2,
        ),
        _layer(
            1,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": lower, "ub": upper},
            width=2,
        ),
        _dense(
            2,
            (
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ),
            (-1.5, -1.5, -1.5, -1.5),
        ),
        _layer(3, "RELU", width=4),
        _dense(
            4,
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 1.0),
                (0.5, 0.5, 0.5, 0.5),
            ),
            (0.75, 0.0, 0.0),
        ),
        _layer(5, "ASSERT", width=3),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [3],
        5: [4],
    }
    succs = {layer.id: [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {}
    for layer in layers:
        width = len(layer.out_vars)
        if layer.kind in {"INPUT", "INPUT_SPEC"}:
            fact_lower = lower.clone()
            fact_upper = upper.clone()
        else:
            # These facts are deliberately non-informative; Operator-HZ must
            # derive the exact Big-M cube from its own live expression.
            fact_lower = torch.full(
                (1, width), -1.0e30, dtype=_DTYPE
            )
            fact_upper = torch.full(
                (1, width), 1.0e30, dtype=_DTYPE
            )
        facts[layer.id] = Fact(
            Bounds(fact_lower, fact_upper), ConSet()
        )
    return build_operator_hz(
        net,
        facts,
        facts,
        exact_budget=4,
        materialize_add=True,
    )


def _rivals() -> tuple[RivalSpec, RivalSpec]:
    return (
        RivalSpec(
            rival_id=10,
            objective=(-1.0, 1.0, 0.0),
            threshold=0.0,
            assert_digest="a" * 64,
        ),
        RivalSpec(
            rival_id=20,
            objective=(-1.0, 0.0, 1.0),
            threshold=0.0,
            assert_digest="b" * 64,
        ),
    )


def _canonical_csr(value: Any) -> sp.csr_matrix:
    result = sp.csr_matrix(value, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def _clone_hz(
    source: SparseHZono,
    **overrides: Any,
) -> SparseHZono:
    row_tags = tuple(
        overrides.pop(
            "row_tags",
            source._solver_constraint_row_tags,
        )
    )
    values = {
        "c": source.c.copy(),
        "Gc": source.Gc.copy(),
        "Gb": source.Gb.copy(),
        "Ac": source.Ac.copy(),
        "Ab": source.Ab.copy(),
        "b": source.b.copy(),
        "Auc": source.Auc.copy(),
        "Aub": source.Aub.copy(),
        "ub": source.ub.copy(),
        "col_ids": source.col_ids.copy(),
        "bcol_ids": source.bcol_ids.copy(),
    }
    values.update(overrides)
    result = SparseHZono(**values)
    setattr(
        result,
        "_solver_constraint_row_tags",
        row_tags,
    )
    return result


def _replace_hz(
    build: OperatorHZBuild,
    hz: SparseHZono,
) -> OperatorHZBuild:
    return replace(build, hz=hz)


def _relaxed_margin_upper(
    hz: SparseHZono,
    objective: Sequence[float],
) -> float:
    objective_array = np.asarray(objective, dtype=np.float64)
    factor_objective = np.concatenate(
        [
            np.asarray(objective_array @ hz.Gc).reshape(-1),
            np.asarray(objective_array @ hz.Gb).reshape(-1),
        ]
    )
    upper = sp.hstack([hz.Auc, hz.Aub], format="csr")
    equality = sp.hstack([hz.Ac, hz.Ab], format="csr")
    result = linprog(
        -factor_objective,
        A_ub=upper if hz.n_ub else None,
        b_ub=hz.ub if hz.n_ub else None,
        A_eq=equality if hz.n_eq else None,
        b_eq=hz.b if hz.n_eq else None,
        bounds=[(-1.0, 1.0)] * (hz.n_cont + hz.n_bin),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(np.dot(objective_array, hz.c) - result.fun)


def _exact_k4_margin_upper() -> Fraction:
    result = []
    for x1, x2 in itertools.product(
        (Fraction(-1), Fraction(1)), repeat=2
    ):
        preactivations = (
            x1 + x2 - Fraction(3, 2),
            x1 - x2 - Fraction(3, 2),
            -x1 + x2 - Fraction(3, 2),
            -x1 - x2 - Fraction(3, 2),
        )
        competitor = sum(
            (max(Fraction(0), value) for value in preactivations),
            Fraction(0),
        )
        result.append(competitor - Fraction(3, 4))
    return max(result)


class OperatorExactReLUPhaseLiteralTest(unittest.TestCase):
    def setUp(self) -> None:
        self.build = _k4_corner_build()
        self.rivals = _rivals()

    def test_k4_operator_path_selects_four_active_literals(self):
        hz = self.build.hz
        self.assertEqual(hz.n_bin, 4)
        self.assertEqual(hz.Gb.nnz, 0)
        self.assertAlmostEqual(
            _relaxed_margin_upper(hz, self.rivals[0].objective),
            0.25,
            places=10,
        )
        self.assertEqual(
            _exact_k4_margin_upper(),
            Fraction(-1, 4),
        )

        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )
        self.assertEqual(result.status, "selected")
        self.assertEqual(len(result.mappings), 4)
        self.assertEqual(len(result.literals), 4)
        self.assertFalse(result.zero_omissions)
        self.assertEqual(
            tuple(literal.phase for literal in result.literals),
            (1, 1, 1, 1),
        )
        self.assertEqual(
            tuple(
                literal.stable_bcol_id
                for literal in result.literals
            ),
            tuple(
                mapping.stable_bcol_id
                for mapping in result.mappings
            ),
        )
        self.assertTrue(
            verify_operator_exact_relu_property_phase_selection(
                self.build, self.rivals, result
            )
        )
        self.assertFalse(result.proof_authority)
        self.assertFalse(hasattr(result, "hz"))

    def test_fraction_coefficients_replay_stored_f64_exactly(self):
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )
        for mapping in result.mappings:
            column = self.build.hz.Gc.getcol(
                mapping.output_continuous_position
            ).tocoo()
            stored = tuple(
                (int(row), float(value))
                for row, value in zip(
                    column.row.tolist(), column.data.tolist()
                )
                if value != 0.0
            )
            expected = []
            for rival in self.rivals:
                coefficient = sum(
                    (
                        Fraction.from_float(rival.objective[row])
                        * Fraction.from_float(value)
                        for row, value in stored
                    ),
                    Fraction(0),
                )
                expected.append(coefficient)
            self.assertEqual(
                tuple(
                    coefficient.value
                    for coefficient in mapping.rival_coefficients
                ),
                tuple(expected),
            )
            self.assertGreater(expected[0], 0)
            self.assertGreater(expected[1], 0)
        self.assertEqual(
            result.arithmetic,
            "Fraction.from_float_exact_dyadic",
        )

    def test_exact_cancellation_is_an_explicit_zero_omission(self):
        cancellation = (
            RivalSpec(
                rival_id=30,
                objective=(0.0, -0.5, 1.0),
                threshold=0.0,
                assert_digest="c" * 64,
            ),
        )
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, cancellation
            )
        )
        self.assertEqual(
            result.status,
            "selected_with_exact_zero_omissions",
        )
        self.assertFalse(result.literals)
        self.assertEqual(len(result.zero_omissions), 4)
        self.assertTrue(
            all(
                coefficient.value == 0
                for omission in result.zero_omissions
                for coefficient in omission.rival_coefficients
            )
        )
        self.assertTrue(
            all(
                omission.proof_authority is False
                for omission in result.zero_omissions
            )
        )

    def test_negative_direct_effect_selects_inactive_literals(self):
        inactive = (
            RivalSpec(
                rival_id=40,
                objective=(1.0, -1.0, 0.0),
                threshold=0.0,
                assert_digest="d" * 64,
            ),
        )
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, inactive
            )
        )
        self.assertEqual(
            tuple(item.phase for item in result.literals),
            (-1, -1, -1, -1),
        )

    def test_column_and_upper_row_permutations_preserve_stable_choice(self):
        baseline = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )
        hz = self.build.hz
        continuous_permutation = np.arange(hz.n_cont - 1, -1, -1)
        binary_permutation = np.asarray((2, 0, 3, 1))
        upper_permutation = np.asarray(
            (8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3)
        )
        tags = tuple(hz._solver_constraint_row_tags)
        permuted_tags = (
            tags[: hz.n_eq]
            + tuple(tags[hz.n_eq + row] for row in upper_permutation)
        )
        permuted = _clone_hz(
            hz,
            Gc=_canonical_csr(
                hz.Gc[:, continuous_permutation]
            ),
            Gb=_canonical_csr(hz.Gb[:, binary_permutation]),
            Ac=_canonical_csr(
                hz.Ac[:, continuous_permutation]
            ),
            Ab=_canonical_csr(hz.Ab[:, binary_permutation]),
            Auc=_canonical_csr(
                hz.Auc[
                    upper_permutation, :
                ][:, continuous_permutation]
            ),
            Aub=_canonical_csr(
                hz.Aub[
                    upper_permutation, :
                ][:, binary_permutation]
            ),
            ub=hz.ub[upper_permutation].copy(),
            col_ids=hz.col_ids[continuous_permutation].copy(),
            bcol_ids=hz.bcol_ids[binary_permutation].copy(),
            row_tags=permuted_tags,
        )
        permuted_build = _replace_hz(self.build, permuted)
        candidate = (
            derive_operator_exact_relu_property_phase_literals(
                permuted_build, self.rivals
            )
        )
        self.assertEqual(
            tuple(
                (item.stable_bcol_id, item.phase)
                for item in candidate.literals
            ),
            tuple(
                (item.stable_bcol_id, item.phase)
                for item in baseline.literals
            ),
        )
        self.assertEqual(
            {
                (
                    item.stable_bcol_id,
                    item.stable_output_col_id,
                )
                for item in candidate.mappings
            },
            {
                (
                    item.stable_bcol_id,
                    item.stable_output_col_id,
                )
                for item in baseline.mappings
            },
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                permuted_build, self.rivals, baseline
            )
        )
        self.assertTrue(
            verify_operator_exact_relu_property_phase_selection(
                permuted_build, self.rivals, candidate
            )
        )

    def test_missing_direct_suffix_column_fails_closed(self):
        baseline = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )
        position = baseline.mappings[0].output_continuous_position
        generator = self.build.hz.Gc.tolil(copy=True)
        generator[:, position] = 0.0
        generator = _canonical_csr(generator)
        early = _clone_hz(self.build.hz, Gc=generator)
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "no_direct_suffix_coefficient",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, early), self.rivals
            )

    def test_property_upper_output_fails_closed(self):
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "property_upper_output_unsupported",
        ):
            derive_operator_exact_relu_property_phase_literals(
                replace(self.build, property_upper_output=True),
                self.rivals,
            )

    def test_noncanonical_csr_is_recomputed_not_trusted(self):
        bad = _clone_hz(self.build.hz)
        start = int(bad.Auc.indptr[0])
        self.assertGreaterEqual(
            int(bad.Auc.indptr[1]) - start, 2
        )
        bad.Auc.indices[start], bad.Auc.indices[start + 1] = (
            bad.Auc.indices[start + 1],
            bad.Auc.indices[start],
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "parent_semantics_malformed",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, bad), self.rivals
            )

    def test_duplicate_branch_row_is_ambiguous(self):
        hz = self.build.hz
        x_row = 4
        tags = tuple(hz._solver_constraint_row_tags)
        duplicate = _clone_hz(
            hz,
            Auc=_canonical_csr(
                sp.vstack(
                    [hz.Auc, hz.Auc.getrow(x_row)],
                    format="csr",
                )
            ),
            Aub=_canonical_csr(
                sp.vstack(
                    [hz.Aub, hz.Aub.getrow(x_row)],
                    format="csr",
                )
            ),
            ub=np.concatenate(
                [hz.ub, np.asarray([hz.ub[x_row]])]
            ),
            row_tags=tags
            + (tags[hz.n_eq + x_row],),
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "row_count_not_two",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, duplicate), self.rivals
            )

    def test_missing_or_malformed_row_tags_fail_closed(self):
        missing = _clone_hz(self.build.hz)
        missing._solver_constraint_row_tags = (
            missing._solver_constraint_row_tags[:-1]
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "row_tags_malformed",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, missing), self.rivals
            )

        malformed = _clone_hz(self.build.hz)
        tags = list(malformed._solver_constraint_row_tags)
        tags[-1] = "relu_exact_zero_branch:03"
        malformed._solver_constraint_row_tags = tuple(tags)
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "layer_id_noncanonical",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, malformed), self.rivals
            )

    def test_scale_and_stable_id_tampering_fail_closed(self):
        scale = _clone_hz(self.build.hz)
        zero_row = 8
        start = int(scale.Aub.indptr[zero_row])
        for _ in range(8):
            scale.Aub.data[start] = np.nextafter(
                scale.Aub.data[start], -np.inf
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "scale_mismatch",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, scale), self.rivals
            )

        stable = _clone_hz(self.build.hz)
        stable.bcol_ids[1] = stable.bcol_ids[0]
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "parent_semantics_malformed",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, stable), self.rivals
            )

    def test_ordered_rival_sign_disagreement_fails_closed(self):
        disagree = (
            self.rivals[0],
            RivalSpec(
                rival_id=50,
                objective=(1.0, -1.0, 0.0),
                threshold=0.0,
                assert_digest="e" * 64,
            ),
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "rivals_disagree",
        ):
            derive_operator_exact_relu_property_phase_literals(
                self.build, disagree
            )

    def test_verifier_rejects_alias_int_and_alias_str_fields(self):
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )

        class AliasInt(int):
            def __eq__(self, other):
                raise AssertionError("AliasInt equality was invoked")

        class AliasStr(str):
            def __eq__(self, other):
                raise AssertionError("AliasStr equality was invoked")

        first_mapping = result.mappings[0]
        aliased_mapping = replace(
            first_mapping,
            stable_bcol_id=AliasInt(
                first_mapping.stable_bcol_id
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=(
                        aliased_mapping,
                        *result.mappings[1:],
                    ),
                ),
            )
        )

        first_literal = result.literals[0]
        aliased_literal_phase = replace(
            first_literal,
            phase=AliasInt(first_literal.phase),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    literals=(
                        aliased_literal_phase,
                        *result.literals[1:],
                    ),
                ),
            )
        )
        aliased_literal = replace(
            first_literal,
            binding_digest=AliasStr(
                first_literal.binding_digest
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    literals=(
                        aliased_literal,
                        *result.literals[1:],
                    ),
                ),
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    arithmetic=AliasStr(result.arithmetic),
                ),
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    caps=replace(
                        result.caps,
                        max_binaries=AliasInt(
                            result.caps.max_binaries
                        ),
                    ),
                ),
            )
        )

        aliased_rivals = (
            replace(
                self.rivals[0],
                assert_digest=AliasStr(
                    self.rivals[0].assert_digest
                ),
            ),
            self.rivals[1],
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "rival_binding_contract_invalid",
        ):
            derive_operator_exact_relu_property_phase_literals(
                self.build, aliased_rivals
            )

        aliased_tags = _clone_hz(self.build.hz)
        tags = list(aliased_tags._solver_constraint_row_tags)
        tags[0] = AliasStr(tags[0])
        aliased_tags._solver_constraint_row_tags = tuple(tags)
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "row_tags_malformed",
        ):
            derive_operator_exact_relu_property_phase_literals(
                _replace_hz(self.build, aliased_tags),
                self.rivals,
            )

    def test_verifier_never_invokes_equality_gadgets(self):
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )

        class EvilEquality:
            calls = 0

            def __eq__(self, other):
                type(self).calls += 1
                raise AssertionError("candidate __eq__ was invoked")

            def __ne__(self, other):
                type(self).calls += 1
                raise AssertionError("candidate __ne__ was invoked")

        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=tuple(
                        EvilEquality()
                        for _ in result.mappings
                    ),
                ),
            )
        )
        self.assertEqual(EvilEquality.calls, 0)

        first_mapping = result.mappings[0]
        nested = replace(
            first_mapping,
            rival_coefficients=tuple(
                EvilEquality()
                for _ in first_mapping.rival_coefficients
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=(
                        nested,
                        *result.mappings[1:],
                    ),
                ),
            )
        )
        self.assertEqual(EvilEquality.calls, 0)
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                EvilEquality(),
            )
        )
        self.assertEqual(EvilEquality.calls, 0)

    def test_fraction_tuple_and_phase_fields_are_exact_primitives(self):
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )
        mapping = result.mappings[0]
        coefficient = mapping.rival_coefficients[0]
        noncanonical = replace(
            coefficient,
            numerator=2 * coefficient.numerator,
            denominator=2 * coefficient.denominator,
        )
        malformed_mapping = replace(
            mapping,
            rival_coefficients=(
                noncanonical,
                *mapping.rival_coefficients[1:],
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=(
                        malformed_mapping,
                        *result.mappings[1:],
                    ),
                ),
            )
        )

        oversized = replace(
            coefficient,
            numerator=1 << 9000,
            denominator=1,
        )
        oversized_mapping = replace(
            mapping,
            rival_coefficients=(
                oversized,
                *mapping.rival_coefficients[1:],
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=(
                        oversized_mapping,
                        *result.mappings[1:],
                    ),
                ),
            )
        )

        class AliasTuple(tuple):
            pass

        tuple_alias = replace(
            mapping,
            rival_coefficients=AliasTuple(
                mapping.rival_coefficients
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=(
                        tuple_alias,
                        *result.mappings[1:],
                    ),
                ),
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(
                    result,
                    mappings=(
                        replace(
                            mapping,
                            selected_phase=np.int64(1),
                        ),
                        *result.mappings[1:],
                    ),
                ),
            )
        )

    def test_resource_caps_deadline_and_cap_binding_fail_closed(self):
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "binary_cap_exceeded",
        ):
            derive_operator_exact_relu_property_phase_literals(
                self.build,
                self.rivals,
                max_binaries=3,
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "rival_sequence_or_cap_invalid",
        ):
            derive_operator_exact_relu_property_phase_literals(
                self.build,
                self.rivals,
                max_rivals=1,
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "work_cap_exceeded",
        ):
            derive_operator_exact_relu_property_phase_literals(
                self.build,
                self.rivals,
                max_work_items=1,
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseLiteralError,
            "timeout_out_of_range",
        ):
            derive_operator_exact_relu_property_phase_literals(
                self.build,
                self.rivals,
                timeout_seconds=61.0,
            )

        caps = {
            "max_rivals": 2,
            "max_binaries": 4,
            "max_work_items": 1000,
            "timeout_seconds": 1.0,
        }
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build,
                self.rivals,
                **caps,
            )
        )
        self.assertEqual(
            type(result.caps),
            OperatorExactReLUPhaseSelectionCaps,
        )
        self.assertTrue(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                result,
                **caps,
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                result,
                **{**caps, "max_binaries": 5},
            )
        )

        calls = [0]

        def expiring_clock():
            calls[0] += 1
            return 0.0 if calls[0] <= 2 else 10.0

        with patch.object(
            phase_adapter.time,
            "monotonic",
            side_effect=expiring_clock,
        ):
            with self.assertRaisesRegex(
                OperatorExactReLUPhaseLiteralError,
                "deadline_expired",
            ):
                derive_operator_exact_relu_property_phase_literals(
                    self.build,
                    self.rivals,
                    timeout_seconds=1.0,
                )

        calls[0] = 0
        baseline = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )
        with patch.object(
            phase_adapter.time,
            "monotonic",
            side_effect=expiring_clock,
        ):
            self.assertFalse(
                verify_operator_exact_relu_property_phase_selection(
                    self.build,
                    self.rivals,
                    baseline,
                    timeout_seconds=1.0,
                )
            )

    def test_parent_tag_rival_and_result_mutation_are_rejected(self):
        result = (
            derive_operator_exact_relu_property_phase_literals(
                self.build, self.rivals
            )
        )

        mutated_parent = _clone_hz(self.build.hz)
        mutated_parent.ub[0] = np.nextafter(
            mutated_parent.ub[0], np.inf
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                _replace_hz(self.build, mutated_parent),
                self.rivals,
                result,
            )
        )

        mutated_tags = _clone_hz(self.build.hz)
        tags = list(mutated_tags._solver_constraint_row_tags)
        tags[0] = "relu_exact_lower:4"
        mutated_tags._solver_constraint_row_tags = tuple(tags)
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                _replace_hz(self.build, mutated_tags),
                self.rivals,
                result,
            )
        )

        mutated_rivals = (
            replace(self.rivals[0], threshold=1.0),
            self.rivals[1],
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build, mutated_rivals, result
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build, tuple(reversed(self.rivals)), result
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(result, selection_digest="0" * 64),
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                self.build,
                self.rivals,
                replace(result, proof_authority=True),
            )
        )


if __name__ == "__main__":
    unittest.main()
