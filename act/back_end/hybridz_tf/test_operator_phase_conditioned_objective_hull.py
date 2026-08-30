#!/usr/bin/env python3
"""Exact toy gates for the isolated phase-conditioned objective hull core."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json
import math
import unittest

from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    ExactHZLinearRow,
    ExternalCertifiedEmptyPattern,
    PhaseConditionedObjectiveHullError,
    bind_external_certified_empty_pattern,
    bind_external_pattern_upper_bound,
    build_objective_binding,
    build_phase_conditioned_objective_hull,
    evaluate_exact_hz_row_lhs,
    outward_float64,
    verify_external_certified_empty_pattern,
    verify_external_pattern_upper_bound,
    verify_objective_binding,
    verify_phase_conditioned_objective_hull,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _objective_value(binding, continuous, binary) -> Fraction:
    value = binding.center
    value += sum(
        coefficient * continuous[stable_id]
        for stable_id, coefficient in binding.continuous_terms
    )
    value += sum(
        coefficient * binary[stable_id]
        for stable_id, coefficient in binding.binary_terms
    )
    return value


def _exact_fraction_pattern_oracle(
    *, binding, assignments, feasible_continuous_points
):
    """Independent finite enumeration oracle used only by these toys."""

    binary = dict(assignments)
    records = []
    values = []
    for point in feasible_continuous_points:
        exact_point = {
            int(stable_id): value
            for stable_id, value in point.items()
        }
        if any(type(value) is not Fraction for value in exact_point.values()):
            raise AssertionError("toy oracle only accepts exact Fractions")
        value = _objective_value(binding, exact_point, binary)
        values.append(value)
        records.append(
            {
                "continuous": [
                    [stable_id, _text(exact_point[stable_id])]
                    for stable_id in sorted(exact_point)
                ],
                "objective": _text(value),
            }
        )
    if not values:
        raise AssertionError("toy phase must have an exact feasible point")
    upper = max(values)
    payload = {
        "schema": "act.test.exact_fraction_pattern_enumeration.v1",
        "parent": binding.parent_semantic_digest,
        "objective": binding.objective_binding_sha256,
        "assignments": [list(item) for item in sorted(assignments)],
        "points": records,
        "upper": _text(upper),
    }
    return upper, _canonical_sha(payload)


def _certified_bounds(
    *,
    binding,
    stable_ids,
    feasible_points_by_pattern,
    reverse_bounds=False,
    reverse_assignments=False,
):
    patterns = tuple(itertools.product((-1, 1), repeat=len(stable_ids)))
    bounds = []
    for pattern in patterns:
        assignments = tuple(zip(stable_ids, pattern))
        upper, certificate_sha = _exact_fraction_pattern_oracle(
            binding=binding,
            assignments=assignments,
            feasible_continuous_points=feasible_points_by_pattern[pattern],
        )
        supplied_assignments = (
            tuple(reversed(assignments)) if reverse_assignments else assignments
        )
        bounds.append(
            bind_external_pattern_upper_bound(
                assignments=supplied_assignments,
                upper_exact=upper,
                upper_stored=outward_float64(upper),
                parent_semantic_digest=binding.parent_semantic_digest,
                objective_binding_sha256=binding.objective_binding_sha256,
                certificate_schema=(
                    "act.test.exact_fraction_pattern_enumeration.v1"
                ),
                certificate_sha256=certificate_sha,
                upstream_proof_authority=True,
                independently_certified=True,
            )
        )
    if reverse_bounds:
        bounds.reverse()
    return tuple(bounds)


def _structural_bounds(*, binding, stable_ids, upper_by_pattern):
    """Complete toy bounds; their opaque proofs remain non-authoritative."""

    result = []
    for pattern in itertools.product((-1, 1), repeat=len(stable_ids)):
        exact = upper_by_pattern[pattern]
        if type(exact) is not Fraction:
            raise AssertionError("toy structural bounds require Fractions")
        result.append(
            bind_external_pattern_upper_bound(
                assignments=tuple(zip(stable_ids, pattern)),
                upper_exact=exact,
                upper_stored=outward_float64(exact),
                parent_semantic_digest=binding.parent_semantic_digest,
                objective_binding_sha256=(
                    binding.objective_binding_sha256
                ),
                certificate_schema="act.test.structural_pattern_bound.v1",
                certificate_sha256=_sha(
                    f"structural-bound:{stable_ids}:{pattern}:{exact}"
                ),
                upstream_proof_authority=True,
                independently_certified=True,
            )
        )
    return tuple(result)


def _empty_evidence(
    *,
    parent,
    stable_ids,
    pattern,
    witness,
    source_bundle_label="shared-empty-bundle",
):
    assignments = tuple(zip(stable_ids, pattern))
    witness = tuple(sorted(witness))
    return bind_external_certified_empty_pattern(
        assignments=assignments,
        witness_literals=witness,
        parent_semantic_digest=parent,
        property_digest=_sha(f"empty-property:{parent}"),
        selection_digest=_sha(f"empty-selection:{parent}:{stable_ids}"),
        operator_row_tag_digest=_sha(f"empty-row-tags:{parent}"),
        ordered_source_frame_sha256=_sha(
            f"empty-source-frame:{parent}"
        ),
        source_bundle_sha256=_sha(
            f"empty-source-bundle:{parent}:{source_bundle_label}"
        ),
        coverage_sha256=_sha(
            f"empty-coverage:{parent}:{stable_ids}:{pattern}:{witness}"
        ),
        source_record_sha256=_sha(
            f"empty-record:{parent}:{witness}"
        ),
        local_row_map_sha256=_sha(
            f"empty-local-row-map:{parent}:{witness}"
        ),
        certificate_schema="act.test.exact_empty_pattern.v1",
        certificate_sha256=_sha(
            f"empty-certificate:{parent}:{witness}"
        ),
        eta_fixed_value=-1,
        upstream_exact_replay_authority=True,
        independently_exact_certified=True,
    )


def _independent_cube_points(continuous_ids):
    if not continuous_ids:
        return ({},)
    return tuple(
        {
            stable_id: Fraction(sign)
            for stable_id, sign in zip(continuous_ids, signs)
        }
        for signs in itertools.product((-1, 1), repeat=len(continuous_ids))
    )


def _build_cube_toy(k: int, *, stable_ids=None):
    if stable_ids is None:
        stable_ids = tuple(100 + 3 * index for index in range(k))
    stable_ids = tuple(stable_ids)
    parent = _sha(f"pc-oh-cube-parent-k{k}")
    binding = build_objective_binding(
        objective_id=f"cube-objective-k{k}",
        parent_semantic_digest=parent,
        center=Fraction(3, 7),
        continuous_terms=((900, Fraction(2, 7)),),
        binary_terms=tuple(
            (stable_id, Fraction(index + 1, 11))
            for index, stable_id in enumerate(sorted(stable_ids))
        ),
    )
    patterns = tuple(itertools.product((-1, 1), repeat=k))
    points = {
        pattern: _independent_cube_points((900,)) for pattern in patterns
    }
    bounds = _certified_bounds(
        binding=binding,
        stable_ids=tuple(sorted(stable_ids)),
        feasible_points_by_pattern=points,
    )
    baseline_exact = max(bound.upper_exact for bound in bounds)
    descriptor = build_phase_conditioned_objective_hull(
        stable_bit_ids=stable_ids,
        pattern_bounds=bounds,
        objective_binding=binding,
        parent_semantic_digest=parent,
        baseline_upper_stored=outward_float64(baseline_exact),
    )
    return descriptor, binding, points


def _solve_square_fraction(matrix, rhs):
    n = len(rhs)
    work = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            scale = work[row][column]
            if scale:
                work[row] = [
                    left - scale * right
                    for left, right in zip(work[row], work[column])
                ]
    return tuple(work[row][-1] for row in range(n))


def _exact_lambda_vertex_envelope(patterns, uppers, target_bits):
    """Exact vertex enumeration for max sum U_p lambda_p at fixed marginals."""

    rank = len(target_bits) + 1
    rhs = (Fraction(1), *target_bits)
    vertices = []
    for support in itertools.combinations(range(len(patterns)), rank):
        matrix = [
            [Fraction(1) for _ in support],
            *[
                [Fraction(patterns[column][bit]) for column in support]
                for bit in range(len(target_bits))
            ],
        ]
        solution = _solve_square_fraction(matrix, rhs)
        if solution is None or any(value < 0 for value in solution):
            continue
        if sum(solution) != 1:
            continue
        if any(
            sum(
                solution[offset] * patterns[column][bit]
                for offset, column in enumerate(support)
            )
            != target_bits[bit]
            for bit in range(len(target_bits))
        ):
            continue
        vertices.append(
            sum(
                solution[offset] * uppers[column]
                for offset, column in enumerate(support)
            )
        )
    if not vertices:
        raise AssertionError("fixed marginal lambda polytope has no vertex")
    return max(vertices)


def _exact_active_lambda_vertex_envelope(
    patterns, uppers, target_bits, *, empty_patterns=()
):
    """Independent Fraction oracle supporting reduced/degenerate hulls."""

    empty = frozenset(empty_patterns)
    active = tuple(
        index for index, pattern in enumerate(patterns)
        if pattern not in empty
    )
    equation_rows = (
        tuple(Fraction(1) for _ in patterns),
        *tuple(
            tuple(Fraction(pattern[bit]) for pattern in patterns)
            for bit in range(len(target_bits))
        ),
    )
    rhs = (Fraction(1), *target_bits)
    vertices = []
    rank = len(rhs)
    for support_size in range(1, min(rank, len(active)) + 1):
        for support in itertools.combinations(active, support_size):
            for selected_rows in itertools.combinations(
                range(rank), support_size
            ):
                matrix = tuple(
                    tuple(equation_rows[row][column] for column in support)
                    for row in selected_rows
                )
                selected_rhs = tuple(rhs[row] for row in selected_rows)
                solution = _solve_square_fraction(matrix, selected_rhs)
                if solution is None or any(value < 0 for value in solution):
                    continue
                if any(
                    sum(
                        solution[offset] * equation_rows[row][column]
                        for offset, column in enumerate(support)
                    )
                    != rhs[row]
                    for row in range(rank)
                ):
                    continue
                vertices.append(
                    sum(
                        solution[offset] * uppers[column]
                        for offset, column in enumerate(support)
                    )
                )
    return None if not vertices else max(vertices)


class PhaseConditionedObjectiveHullToyTests(unittest.TestCase):
    def test_k1_through_k4_one_hot_point_jacobian_and_row_counts(self):
        for k in range(1, 5):
            with self.subTest(k=k):
                descriptor, binding, points = _build_cube_toy(k)
                self.assertFalse(descriptor.proof_authority)
                self.assertFalse(descriptor.verdict_authority)
                self.assertFalse(descriptor.empty_pattern_evidence)
                self.assertTrue(
                    verify_phase_conditioned_objective_hull(
                        descriptor,
                        live_parent_semantic_digest=(
                            descriptor.parent_semantic_digest
                        ),
                        live_objective_binding=binding,
                    )
                )
                self.assertEqual(len(descriptor.patterns), 2**k)
                self.assertEqual(len(descriptor.eta_columns), 2**k)
                self.assertEqual(len(descriptor.equality_rows), k + 1)
                self.assertEqual(len(descriptor.upper_rows), 1)

                upper_by_pattern = {
                    tuple(value for _, value in bound.assignments): (
                        Fraction.from_float(bound.upper_stored)
                    )
                    for bound in descriptor.pattern_bounds
                }
                baseline = Fraction.from_float(
                    descriptor.baseline_upper_stored
                )
                for pattern_index, pattern in enumerate(descriptor.patterns):
                    eta = {
                        index: Fraction(1 if index == pattern_index else -1)
                        for index in range(len(descriptor.patterns))
                    }
                    binary = {
                        stable_id: Fraction(phase)
                        for stable_id, phase in zip(
                            descriptor.stable_bit_ids, pattern
                        )
                    }
                    continuous = {900: Fraction(1)}
                    for row in descriptor.equality_rows:
                        self.assertEqual(
                            evaluate_exact_hz_row_lhs(
                                row,
                                continuous_values=continuous,
                                binary_values=binary,
                                eta_values=eta,
                            ),
                            row.rhs,
                        )
                    upper_row = descriptor.upper_rows[0]
                    lhs = evaluate_exact_hz_row_lhs(
                        upper_row,
                        continuous_values=continuous,
                        binary_values=binary,
                        eta_values=eta,
                    )
                    exact_objective = _objective_value(
                        binding, continuous, binary
                    )
                    self.assertEqual(
                        upper_row.rhs - lhs,
                        upper_by_pattern[pattern] - exact_objective,
                    )
                    self.assertLessEqual(exact_objective, upper_by_pattern[pattern])
                    self.assertLessEqual(upper_by_pattern[pattern], baseline)

                for row in (*descriptor.equality_rows, *descriptor.upper_rows):
                    all_groups = (
                        (row.parent_continuous_terms, "continuous"),
                        (row.parent_binary_terms, "binary"),
                        (row.eta_terms, "eta"),
                    )
                    base_maps = {
                        "continuous": {
                            stable_id: Fraction(0)
                            for stable_id, _ in row.parent_continuous_terms
                        },
                        "binary": {
                            stable_id: Fraction(0)
                            for stable_id, _ in row.parent_binary_terms
                        },
                        "eta": {
                            stable_id: Fraction(0)
                            for stable_id, _ in row.eta_terms
                        },
                    }
                    base = evaluate_exact_hz_row_lhs(
                        row,
                        continuous_values=base_maps["continuous"],
                        binary_values=base_maps["binary"],
                        eta_values=base_maps["eta"],
                    )
                    for terms, group in all_groups:
                        for stable_id, coefficient in terms:
                            shifted = {
                                key: dict(value) for key, value in base_maps.items()
                            }
                            shifted[group][stable_id] = Fraction(1)
                            value = evaluate_exact_hz_row_lhs(
                                row,
                                continuous_values=shifted["continuous"],
                                binary_values=shifted["binary"],
                                eta_values=shifted["eta"],
                            )
                            self.assertEqual(value - base, coefficient)

    def test_exact_vertex_oracle_integer_patterns_and_baseline_sandwich(self):
        for k in range(1, 4):
            descriptor, binding, points = _build_cube_toy(k)
            uppers = tuple(
                Fraction.from_float(bound.upper_stored)
                for bound in descriptor.pattern_bounds
            )
            baseline = Fraction.from_float(descriptor.baseline_upper_stored)
            for pattern, upper in zip(descriptor.patterns, uppers):
                envelope = _exact_lambda_vertex_envelope(
                    descriptor.patterns,
                    uppers,
                    tuple(Fraction(value) for value in pattern),
                )
                exact = max(
                    _objective_value(
                        binding,
                        point,
                        dict(zip(descriptor.stable_bit_ids, pattern)),
                    )
                    for point in points[pattern]
                )
                self.assertEqual(envelope, upper)
                self.assertLessEqual(exact, envelope)
                self.assertLessEqual(envelope, baseline)

    def test_correlated_positive_control_and_independent_negative_control(self):
        parent = _sha("pc-oh-controls-parent")
        bit = 17
        binding = build_objective_binding(
            objective_id="one-plus-correlated-x",
            parent_semantic_digest=parent,
            center=Fraction(1),
            continuous_terms=((71, Fraction(1)),),
        )
        correlated_points = {
            (-1,): ({71: Fraction(-1)},),
            (1,): ({71: Fraction(1)},),
        }
        correlated_bounds = _certified_bounds(
            binding=binding,
            stable_ids=(bit,),
            feasible_points_by_pattern=correlated_points,
        )
        correlated = build_phase_conditioned_objective_hull(
            stable_bit_ids=(bit,),
            pattern_bounds=correlated_bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=2.0,
        )
        correlated_envelope = _exact_lambda_vertex_envelope(
            correlated.patterns,
            tuple(
                Fraction.from_float(bound.upper_stored)
                for bound in correlated.pattern_bounds
            ),
            (Fraction(0),),
        )
        self.assertEqual(correlated_envelope, Fraction(1))
        self.assertLess(correlated_envelope, Fraction(2))

        independent_points = {
            (-1,): ({71: Fraction(1)},),
            (1,): ({71: Fraction(1)},),
        }
        independent_bounds = _certified_bounds(
            binding=binding,
            stable_ids=(bit,),
            feasible_points_by_pattern=independent_points,
        )
        independent = build_phase_conditioned_objective_hull(
            stable_bit_ids=(bit,),
            pattern_bounds=independent_bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=2.0,
        )
        independent_envelope = _exact_lambda_vertex_envelope(
            independent.patterns,
            tuple(
                Fraction.from_float(bound.upper_stored)
                for bound in independent.pattern_bounds
            ),
            (Fraction(0),),
        )
        self.assertEqual(independent_envelope, Fraction(2))
        self.assertEqual(
            independent_envelope,
            Fraction.from_float(independent.baseline_upper_stored),
        )

    def test_fraction_lambda_eta_equivalence_for_relaxed_mixture(self):
        parent = _sha("pc-oh-fraction-mixture-parent")
        stable_ids = (3, 9)
        binding = build_objective_binding(
            objective_id="fraction-mixture-objective",
            parent_semantic_digest=parent,
            center=Fraction(0),
            continuous_terms=((77, Fraction(1)),),
        )
        exact_by_pattern = {
            (-1, -1): Fraction(-3, 4),
            (-1, 1): Fraction(-1, 4),
            (1, -1): Fraction(1, 4),
            (1, 1): Fraction(3, 4),
        }
        points = {
            pattern: ({77: upper},)
            for pattern, upper in exact_by_pattern.items()
        }
        bounds = _certified_bounds(
            binding=binding,
            stable_ids=stable_ids,
            feasible_points_by_pattern=points,
        )
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=0.75,
        )
        lambdas = (
            Fraction(1, 10),
            Fraction(2, 10),
            Fraction(3, 10),
            Fraction(4, 10),
        )
        eta = {
            index: 2 * value - 1 for index, value in enumerate(lambdas)
        }
        binary = {
            stable_id: sum(
                lambdas[index] * pattern[bit_offset]
                for index, pattern in enumerate(descriptor.patterns)
            )
            for bit_offset, stable_id in enumerate(stable_ids)
        }
        stored_uppers = tuple(
            Fraction.from_float(bound.upper_stored)
            for bound in descriptor.pattern_bounds
        )
        mixture_upper = sum(
            weight * upper for weight, upper in zip(lambdas, stored_uppers)
        )
        continuous = {77: mixture_upper}
        for row in descriptor.equality_rows:
            self.assertEqual(
                evaluate_exact_hz_row_lhs(
                    row,
                    continuous_values=continuous,
                    binary_values=binary,
                    eta_values=eta,
                ),
                row.rhs,
            )
        upper = descriptor.upper_rows[0]
        self.assertEqual(
            evaluate_exact_hz_row_lhs(
                upper,
                continuous_values=continuous,
                binary_values=binary,
                eta_values=eta,
            ),
            upper.rhs,
        )

    def test_forbidden_corner_and_large_empty_bound_are_eliminated_exactly(self):
        parent = _sha("pc-oh-forbidden-corner-parent")
        stable_ids = (11, 19)
        binding = build_objective_binding(
            objective_id="forbidden-corner-objective",
            parent_semantic_digest=parent,
            center=Fraction(0),
        )
        huge = Fraction.from_float(1.0e300)
        upper_by_pattern = {
            (-1, -1): Fraction(0),
            (-1, 1): Fraction(0),
            (1, -1): Fraction(0),
            (1, 1): huge,
        }
        bounds = _structural_bounds(
            binding=binding,
            stable_ids=stable_ids,
            upper_by_pattern=upper_by_pattern,
        )
        evidence = _empty_evidence(
            parent=parent,
            stable_ids=stable_ids,
            pattern=(1, 1),
            witness=((11, 1), (19, 1)),
        )
        self.assertTrue(verify_external_certified_empty_pattern(evidence))
        baseline = 1.0e300
        without_empty = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=baseline,
        )
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=baseline,
            empty_pattern_evidence=(evidence,),
        )

        self.assertEqual(len(descriptor.pattern_bounds), 4)
        self.assertEqual(len(descriptor.patterns), 4)
        self.assertEqual(descriptor.empty_pattern_evidence, (evidence,))
        empty_index = descriptor.patterns.index((1, 1))
        self.assertEqual(descriptor.eta_columns[empty_index].lower, -1)
        self.assertEqual(descriptor.eta_columns[empty_index].upper, -1)
        self.assertEqual(
            descriptor.equality_rows[-1],
            ExactHZLinearRow(
                name=f"certified_empty_eta_fix:{empty_index}",
                sense="eq",
                parent_continuous_terms=(),
                parent_binary_terms=(),
                eta_terms=((empty_index, Fraction(1)),),
                rhs=Fraction(-1),
            ),
        )
        objective_row = descriptor.upper_rows[0]
        self.assertNotIn(empty_index, dict(objective_row.eta_terms))
        self.assertEqual(objective_row.eta_terms, ())
        self.assertEqual(objective_row.rhs, Fraction(0))
        self.assertEqual(
            descriptor.receipt[
                "objective_row_empty_terms_eliminated_exactly"
            ],
            True,
        )

        stored = tuple(
            Fraction.from_float(bound.upper_stored) for bound in bounds
        )
        unrestricted = _exact_active_lambda_vertex_envelope(
            without_empty.patterns,
            stored,
            (Fraction(0), Fraction(0)),
        )
        restricted = _exact_active_lambda_vertex_envelope(
            descriptor.patterns,
            stored,
            (Fraction(0), Fraction(0)),
            empty_patterns=((1, 1),),
        )
        self.assertEqual(unrestricted, huge / 2)
        self.assertEqual(restricted, Fraction(0))
        self.assertTrue(
            verify_phase_conditioned_objective_hull(
                descriptor,
                live_parent_semantic_digest=parent,
                live_objective_binding=binding,
            )
        )

    def test_k4_pair_empty_coverage_retains_sixteen_slots_and_fixes_eleven(self):
        parent = _sha("pc-oh-k4-eleven-empty-parent")
        stable_ids = (2, 5, 7, 13)
        binding = build_objective_binding(
            objective_id="k4-eleven-empty-objective",
            parent_semantic_digest=parent,
            center=Fraction(0),
        )
        patterns = tuple(itertools.product((-1, 1), repeat=4))
        empty_patterns = tuple(
            pattern for pattern in patterns if pattern.count(1) >= 2
        )
        active_patterns = tuple(
            pattern for pattern in patterns if pattern.count(1) <= 1
        )
        upper_by_pattern = {
            pattern: (
                Fraction(100)
                if pattern in empty_patterns
                else Fraction(index + 1, 10)
            )
            for index, pattern in enumerate(patterns)
        }
        bounds = _structural_bounds(
            binding=binding,
            stable_ids=stable_ids,
            upper_by_pattern=upper_by_pattern,
        )
        evidence = []
        for pattern in empty_patterns:
            positive_ids = tuple(
                stable_id
                for stable_id, phase in zip(stable_ids, pattern)
                if phase == 1
            )
            witness = tuple((stable_id, 1) for stable_id in positive_ids[:2])
            evidence.append(
                _empty_evidence(
                    parent=parent,
                    stable_ids=stable_ids,
                    pattern=pattern,
                    witness=witness,
                )
            )
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=100.0,
            empty_pattern_evidence=tuple(reversed(evidence)),
        )

        self.assertEqual(len(descriptor.patterns), 16)
        self.assertEqual(len(descriptor.pattern_bounds), 16)
        self.assertEqual(len(descriptor.eta_columns), 16)
        self.assertEqual(len(descriptor.empty_pattern_evidence), 11)
        self.assertEqual(len(descriptor.equality_rows), 16)
        self.assertEqual(descriptor.receipt["certified_empty_patterns"], 11)
        self.assertEqual(
            descriptor.receipt["not_certified_empty_patterns"], 5
        )
        fixed_indices = {
            index
            for index, column in enumerate(descriptor.eta_columns)
            if column.lower == column.upper == -1
        }
        self.assertEqual(
            fixed_indices,
            {
                descriptor.patterns.index(pattern)
                for pattern in empty_patterns
            },
        )
        objective_indices = set(dict(descriptor.upper_rows[0].eta_terms))
        self.assertEqual(
            objective_indices,
            {
                descriptor.patterns.index(pattern)
                for pattern in active_patterns
            },
        )

        stored = tuple(
            Fraction.from_float(bound.upper_stored)
            for bound in descriptor.pattern_bounds
        )
        for pattern in patterns:
            envelope = _exact_active_lambda_vertex_envelope(
                descriptor.patterns,
                stored,
                tuple(Fraction(value) for value in pattern),
                empty_patterns=empty_patterns,
            )
            if pattern in empty_patterns:
                self.assertIsNone(envelope)
            else:
                self.assertEqual(
                    envelope,
                    stored[descriptor.patterns.index(pattern)],
                )
        self.assertTrue(
            verify_phase_conditioned_objective_hull(
                descriptor,
                live_parent_semantic_digest=parent,
                live_objective_binding=binding,
            )
        )

    def test_mixed_sign_empty_and_triple_only_negative_control(self):
        parent = _sha("pc-oh-mixed-sign-empty-parent")
        stable_ids = (31, 47)
        binding = build_objective_binding(
            objective_id="mixed-sign-empty-objective",
            parent_semantic_digest=parent,
            center=Fraction(0),
        )
        bounds = _structural_bounds(
            binding=binding,
            stable_ids=stable_ids,
            upper_by_pattern={
                pattern: Fraction(1)
                for pattern in itertools.product((-1, 1), repeat=2)
            },
        )
        evidence = _empty_evidence(
            parent=parent,
            stable_ids=stable_ids,
            pattern=(-1, 1),
            witness=((31, -1), (47, 1)),
        )
        mixed = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=1.0,
            empty_pattern_evidence=(evidence,),
        )
        mixed_index = mixed.patterns.index((-1, 1))
        self.assertEqual(mixed.eta_columns[mixed_index].upper, -1)
        self.assertNotIn(mixed_index, dict(mixed.upper_rows[0].eta_terms))

        triple_parent = _sha("pc-oh-triple-only-negative-parent")
        triple_ids = (3, 5, 9)
        triple_binding = build_objective_binding(
            objective_id="triple-only-negative-objective",
            parent_semantic_digest=triple_parent,
            center=Fraction(0),
        )
        triple_bounds = _structural_bounds(
            binding=triple_binding,
            stable_ids=triple_ids,
            upper_by_pattern={
                pattern: Fraction(1)
                for pattern in itertools.product((-1, 1), repeat=3)
            },
        )
        triple = build_phase_conditioned_objective_hull(
            stable_bit_ids=triple_ids,
            pattern_bounds=triple_bounds,
            objective_binding=triple_binding,
            parent_semantic_digest=triple_parent,
            baseline_upper_stored=1.0,
        )
        triple_index = triple.patterns.index((1, 1, 1))
        self.assertEqual(triple.eta_columns[triple_index].upper, 1)
        self.assertFalse(triple.empty_pattern_evidence)
        self.assertEqual(triple.receipt["certified_empty_patterns"], 0)

    def test_empty_evidence_canonicalization_and_tamper_guards(self):
        parent = _sha("pc-oh-empty-evidence-tamper-parent")
        stable_ids = (3, 7, 11)
        binding = build_objective_binding(
            objective_id="empty-evidence-tamper-objective",
            parent_semantic_digest=parent,
            center=Fraction(0),
        )
        patterns = tuple(itertools.product((-1, 1), repeat=3))
        bounds = _structural_bounds(
            binding=binding,
            stable_ids=stable_ids,
            upper_by_pattern={pattern: Fraction(2) for pattern in patterns},
        )
        first = _empty_evidence(
            parent=parent,
            stable_ids=stable_ids,
            pattern=(-1, 1, 1),
            witness=((7, 1), (11, 1)),
        )
        self.assertIsInstance(first, ExternalCertifiedEmptyPattern)
        second = _empty_evidence(
            parent=parent,
            stable_ids=stable_ids,
            pattern=(1, 1, -1),
            witness=((3, 1), (7, 1)),
        )
        kwargs = dict(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=2.0,
        )
        canonical = build_phase_conditioned_objective_hull(
            empty_pattern_evidence=(first, second), **kwargs
        )
        reordered = build_phase_conditioned_objective_hull(
            empty_pattern_evidence=(second, first), **kwargs
        )
        self.assertEqual(canonical, reordered)

        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "duplicate_empty_pattern_evidence",
        ):
            build_phase_conditioned_objective_hull(
                empty_pattern_evidence=(first, first), **kwargs
            )
        mismatched_source = _empty_evidence(
            parent=parent,
            stable_ids=stable_ids,
            pattern=(1, 1, -1),
            witness=((3, 1), (7, 1)),
            source_bundle_label="different-empty-bundle",
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "empty_evidence_source_join_mismatch",
        ):
            build_phase_conditioned_objective_hull(
                empty_pattern_evidence=(first, mismatched_source), **kwargs
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "empty_witness_phase_not_in_pattern",
        ):
            _empty_evidence(
                parent=parent,
                stable_ids=stable_ids,
                pattern=(-1, 1, 1),
                witness=((3, 1), (7, 1)),
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "empty_witness_id_not_in_pattern",
        ):
            _empty_evidence(
                parent=parent,
                stable_ids=stable_ids,
                pattern=(-1, 1, 1),
                witness=((7, 1), (999, 1)),
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "empty_eta_fixed_value_must_be_minus_one",
        ):
            bind_external_certified_empty_pattern(
                assignments=first.assignments,
                witness_literals=first.witness_literals,
                parent_semantic_digest=first.parent_semantic_digest,
                property_digest=first.property_digest,
                selection_digest=first.selection_digest,
                operator_row_tag_digest=first.operator_row_tag_digest,
                ordered_source_frame_sha256=(
                    first.ordered_source_frame_sha256
                ),
                source_bundle_sha256=first.source_bundle_sha256,
                coverage_sha256=first.coverage_sha256,
                source_record_sha256=first.source_record_sha256,
                local_row_map_sha256=first.local_row_map_sha256,
                certificate_schema=first.certificate_schema,
                certificate_sha256=first.certificate_sha256,
                eta_fixed_value=0,
                upstream_exact_replay_authority=True,
                independently_exact_certified=True,
            )

        tampered = replace(first, eta_fixed_value=0)
        self.assertFalse(verify_external_certified_empty_pattern(tampered))
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "empty_pattern_evidence_0_invalid",
        ):
            build_phase_conditioned_objective_hull(
                empty_pattern_evidence=(tampered,), **kwargs
            )
        descriptor_tamper = replace(
            canonical,
            empty_pattern_evidence=(
                replace(first, certificate_sha256="0" * 64),
                second,
            ),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "empty_pattern_evidence_0_invalid",
        ):
            verify_phase_conditioned_objective_hull(
                descriptor_tamper,
                live_parent_semantic_digest=parent,
                live_objective_binding=binding,
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "descriptor_verdict_authority_must_be_false",
        ):
            verify_phase_conditioned_objective_hull(
                replace(canonical, verdict_authority=True),
                live_parent_semantic_digest=parent,
                live_objective_binding=binding,
            )

    def test_pattern_and_input_permutation_canonicalization(self):
        stable_ids = (41, 7, 23)
        canonical_ids = tuple(sorted(stable_ids))
        parent = _sha("pc-oh-permutation-parent")
        binding = build_objective_binding(
            objective_id="permutation-objective",
            parent_semantic_digest=parent,
            center=Fraction(1, 3),
            binary_terms=tuple(
                (stable_id, Fraction(index + 1, 13))
                for index, stable_id in enumerate(canonical_ids)
            ),
        )
        points = {
            pattern: ({},)
            for pattern in itertools.product((-1, 1), repeat=3)
        }
        left_bounds = _certified_bounds(
            binding=binding,
            stable_ids=canonical_ids,
            feasible_points_by_pattern=points,
        )
        right_bounds = _certified_bounds(
            binding=binding,
            stable_ids=canonical_ids,
            feasible_points_by_pattern=points,
            reverse_bounds=True,
            reverse_assignments=True,
        )
        baseline = outward_float64(
            max(bound.upper_exact for bound in left_bounds)
        )
        left = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=left_bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=baseline,
        )
        right = build_phase_conditioned_objective_hull(
            stable_bit_ids=tuple(reversed(stable_ids)),
            pattern_bounds=right_bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=baseline,
        )
        self.assertEqual(left, right)
        self.assertEqual(left.representation_sha256, right.representation_sha256)
        self.assertEqual(
            left.receipt["receipt_sha256"], right.receipt["receipt_sha256"]
        )

    def test_strict_binding_and_external_bound_checks(self):
        descriptor, binding, _ = _build_cube_toy(2)
        self.assertTrue(verify_objective_binding(binding))
        self.assertTrue(
            all(
                verify_external_pattern_upper_bound(bound)
                for bound in descriptor.pattern_bounds
            )
        )
        self.assertFalse(
            descriptor.receipt["upstream_certificates_replayed"]
        )
        self.assertFalse(descriptor.receipt["proof_authority"])

        with self.assertRaises(PhaseConditionedObjectiveHullError):
            bind_external_pattern_upper_bound(
                assignments=((1, -1),),
                upper_exact=Fraction(0),
                upper_stored=math.nan,
                parent_semantic_digest=binding.parent_semantic_digest,
                objective_binding_sha256=binding.objective_binding_sha256,
                certificate_schema="toy",
                certificate_sha256=_sha("nan"),
                upstream_proof_authority=True,
                independently_certified=True,
            )
        with self.assertRaises(PhaseConditionedObjectiveHullError):
            bind_external_pattern_upper_bound(
                assignments=((1, -1),),
                upper_exact=Fraction(1, 10),
                upper_stored=0.0,
                parent_semantic_digest=binding.parent_semantic_digest,
                objective_binding_sha256=binding.objective_binding_sha256,
                certificate_schema="toy",
                certificate_sha256=_sha("not-outward"),
                upstream_proof_authority=True,
                independently_certified=True,
            )
        with self.assertRaises(PhaseConditionedObjectiveHullError):
            bind_external_pattern_upper_bound(
                assignments=((1, -1),),
                upper_exact=Fraction(0),
                upper_stored=0.0,
                parent_semantic_digest=binding.parent_semantic_digest,
                objective_binding_sha256=binding.objective_binding_sha256,
                certificate_schema="toy",
                certificate_sha256=_sha("no-authority"),
                upstream_proof_authority=False,
                independently_certified=True,
            )

    def test_missing_duplicate_wrong_pattern_and_certificate_fail_closed(self):
        descriptor, binding, _ = _build_cube_toy(2)
        kwargs = dict(
            stable_bit_ids=descriptor.stable_bit_ids,
            objective_binding=binding,
            parent_semantic_digest=descriptor.parent_semantic_digest,
            baseline_upper_stored=descriptor.baseline_upper_stored,
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "incomplete_pattern_cover"
        ):
            build_phase_conditioned_objective_hull(
                pattern_bounds=descriptor.pattern_bounds[:-1], **kwargs
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "duplicate_pattern_bound"
        ):
            build_phase_conditioned_objective_hull(
                pattern_bounds=(
                    *descriptor.pattern_bounds[:-1],
                    descriptor.pattern_bounds[0],
                ),
                **kwargs,
            )
        wrong_ids = bind_external_pattern_upper_bound(
            assignments=((101, -1), (999, -1)),
            upper_exact=Fraction(0),
            upper_stored=0.0,
            parent_semantic_digest=descriptor.parent_semantic_digest,
            objective_binding_sha256=binding.objective_binding_sha256,
            certificate_schema="toy",
            certificate_sha256=_sha("wrong-ids"),
            upstream_proof_authority=True,
            independently_certified=True,
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "wrong_stable_ids"
        ):
            build_phase_conditioned_objective_hull(
                pattern_bounds=(wrong_ids, *descriptor.pattern_bounds[1:]),
                **kwargs,
            )
        duplicate_certificate = replace(
            descriptor.pattern_bounds[1],
            certificate_sha256=descriptor.pattern_bounds[0].certificate_sha256,
        )
        # Rebinding is required to make the source descriptor internally
        # consistent; the PCOH full-set gate must still reject shared proof.
        duplicate_certificate = bind_external_pattern_upper_bound(
            assignments=duplicate_certificate.assignments,
            upper_exact=duplicate_certificate.upper_exact,
            upper_stored=duplicate_certificate.upper_stored,
            parent_semantic_digest=duplicate_certificate.parent_semantic_digest,
            objective_binding_sha256=(
                duplicate_certificate.objective_binding_sha256
            ),
            certificate_schema=duplicate_certificate.certificate_schema,
            certificate_sha256=duplicate_certificate.certificate_sha256,
            upstream_proof_authority=True,
            independently_certified=True,
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError,
            "pattern_certificates_not_independent",
        ):
            build_phase_conditioned_objective_hull(
                pattern_bounds=(
                    descriptor.pattern_bounds[0],
                    duplicate_certificate,
                    *descriptor.pattern_bounds[2:],
                ),
                **kwargs,
            )

    def test_stale_parent_objective_and_baseline_regression_fail_closed(self):
        descriptor, binding, _ = _build_cube_toy(1)
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "stale_parent"
        ):
            build_phase_conditioned_objective_hull(
                stable_bit_ids=descriptor.stable_bit_ids,
                pattern_bounds=descriptor.pattern_bounds,
                objective_binding=binding,
                parent_semantic_digest=_sha("different-parent"),
                baseline_upper_stored=descriptor.baseline_upper_stored,
            )
        other_binding = build_objective_binding(
            objective_id="different-objective",
            parent_semantic_digest=descriptor.parent_semantic_digest,
            center=Fraction(0),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "stale_objective"
        ):
            build_phase_conditioned_objective_hull(
                stable_bit_ids=descriptor.stable_bit_ids,
                pattern_bounds=descriptor.pattern_bounds,
                objective_binding=other_binding,
                parent_semantic_digest=descriptor.parent_semantic_digest,
                baseline_upper_stored=descriptor.baseline_upper_stored,
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "regresses_baseline"
        ):
            build_phase_conditioned_objective_hull(
                stable_bit_ids=descriptor.stable_bit_ids,
                pattern_bounds=descriptor.pattern_bounds,
                objective_binding=binding,
                parent_semantic_digest=descriptor.parent_semantic_digest,
                baseline_upper_stored=-1e100,
            )

    def test_source_and_result_tampering_fail_closed(self):
        descriptor, binding, _ = _build_cube_toy(2)
        tampered_bound = replace(
            descriptor.pattern_bounds[0],
            upper_exact=descriptor.pattern_bounds[0].upper_exact - 1,
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "pattern_bound_0_invalid"
        ):
            build_phase_conditioned_objective_hull(
                stable_bit_ids=descriptor.stable_bit_ids,
                pattern_bounds=(tampered_bound, *descriptor.pattern_bounds[1:]),
                objective_binding=binding,
                parent_semantic_digest=descriptor.parent_semantic_digest,
                baseline_upper_stored=descriptor.baseline_upper_stored,
            )

        row = descriptor.equality_rows[0]
        row_tamper = replace(row, rhs=row.rhs + 1)
        result_tamper = replace(
            descriptor,
            equality_rows=(row_tamper, *descriptor.equality_rows[1:]),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "equality_rows"
        ):
            verify_phase_conditioned_objective_hull(
                result_tamper,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                live_objective_binding=binding,
            )

        receipt = dict(descriptor.receipt)
        receipt["integer_phase_forces_one_hot"] = False
        receipt_tamper = replace(descriptor, receipt=receipt)
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "receipt"
        ):
            verify_phase_conditioned_objective_hull(
                receipt_tamper,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                live_objective_binding=binding,
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "proof_authority"
        ):
            verify_phase_conditioned_objective_hull(
                replace(descriptor, proof_authority=True),
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                live_objective_binding=binding,
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullError, "descriptor_stale_parent"
        ):
            verify_phase_conditioned_objective_hull(
                descriptor,
                live_parent_semantic_digest=_sha("stale-live-parent"),
                live_objective_binding=binding,
            )

    def test_dimension_id_and_coefficient_guards(self):
        descriptor, binding, _ = _build_cube_toy(1)
        for ids in ((), (1, 2, 3, 4, 5), (1, 1)):
            with self.subTest(ids=ids):
                with self.assertRaises(PhaseConditionedObjectiveHullError):
                    build_phase_conditioned_objective_hull(
                        stable_bit_ids=ids,
                        pattern_bounds=descriptor.pattern_bounds,
                        objective_binding=binding,
                        parent_semantic_digest=descriptor.parent_semantic_digest,
                        baseline_upper_stored=descriptor.baseline_upper_stored,
                    )
        with self.assertRaises(PhaseConditionedObjectiveHullError):
            build_objective_binding(
                objective_id="bad-nan",
                parent_semantic_digest=descriptor.parent_semantic_digest,
                center=math.nan,
            )
        with self.assertRaises(PhaseConditionedObjectiveHullError):
            build_objective_binding(
                objective_id="duplicate-term",
                parent_semantic_digest=descriptor.parent_semantic_digest,
                center=0,
                continuous_terms=((1, 1), (1, 2)),
            )
        with self.assertRaises(PhaseConditionedObjectiveHullError):
            build_objective_binding(
                objective_id="explicit-zero",
                parent_semantic_digest=descriptor.parent_semantic_digest,
                center=0,
                continuous_terms=((1, 0),),
            )

    def test_exact_row_coefficients_match_lambda_eta_derivation(self):
        parent = _sha("pc-oh-explicit-row-parent")
        binding = build_objective_binding(
            objective_id="explicit-row",
            parent_semantic_digest=parent,
            center=Fraction(5, 3),
            continuous_terms=((10, Fraction(7, 5)),),
            binary_terms=((20, Fraction(-2, 9)),),
        )
        stable_ids = (20,)
        points = {
            (-1,): ({10: Fraction(-1)}, {10: Fraction(1)}),
            (1,): ({10: Fraction(-1)}, {10: Fraction(1)}),
        }
        bounds = _certified_bounds(
            binding=binding,
            stable_ids=stable_ids,
            feasible_points_by_pattern=points,
        )
        baseline = outward_float64(max(bound.upper_exact for bound in bounds))
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=baseline,
        )
        self.assertEqual(
            descriptor.equality_rows[0],
            ExactHZLinearRow(
                name="lambda_normalization",
                sense="eq",
                parent_continuous_terms=(),
                parent_binary_terms=(),
                eta_terms=((0, Fraction(1)), (1, Fraction(1))),
                rhs=Fraction(0),
            ),
        )
        self.assertEqual(
            descriptor.equality_rows[1],
            ExactHZLinearRow(
                name="stable_bit_link:20",
                sense="eq",
                parent_continuous_terms=(),
                parent_binary_terms=((20, Fraction(2)),),
                eta_terms=((0, Fraction(1)), (1, Fraction(-1))),
                rhs=Fraction(0),
            ),
        )
        stored = tuple(
            Fraction.from_float(bound.upper_stored)
            for bound in descriptor.pattern_bounds
        )
        upper = descriptor.upper_rows[0]
        self.assertEqual(upper.parent_continuous_terms, binding.continuous_terms)
        self.assertEqual(upper.parent_binary_terms, binding.binary_terms)
        self.assertEqual(
            dict(upper.eta_terms),
            {
                index: -Fraction(1, 2) * value
                for index, value in enumerate(stored)
                if value != 0
            },
        )
        self.assertEqual(
            upper.rhs,
            Fraction(1, 2) * sum(stored) - binding.center,
        )


if __name__ == "__main__":
    unittest.main()
