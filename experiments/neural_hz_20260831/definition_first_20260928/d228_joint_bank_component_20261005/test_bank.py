"""Sixteen preregistered mathematical tests; no native or model qualification.

The reference enumerates the sixteen labelled cells only in the test process.
It neither supplies bounds to the candidate nor invokes an external solver.
This file must not be imported, collected, or executed before its freeze.
"""

from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations, product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005.bank import (
    Binding, Budget, Rejected, Spec, build,
)


_NAMES = (
    "default_off_and_foreign_frame",
    "spec_types_and_bounds_fail_closed",
    "no_residual_matches_independent_phase_cells",
    "one_residual_matches_independent_phase_cells",
    "two_residuals_match_independent_phase_cells",
    "interior_zero_intersection_retained",
    "every_vertex_is_true_graph",
    "signed_rows_and_zero_labels",
    "integer_phase_compatible_mixture_is_true_graph",
    "physical_and_phase_support",
    "shared_bank_control_and_d062_reference",
    "source_binding_without_vertex_source_atoms",
    "constant_residual_and_dependent_zero_planes",
    "compiled_counts_and_nnz",
    "shared_budget_and_bit_limit",
    "asymmetric_permutations_and_summary",
)
_EVIDENCE = {}
_CASES = {}
_REFERENCES = {}


def _record(number, **data):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = data


def _spec(residual_count=0):
    if residual_count == 1:
        return Spec(
            parent_bounds=((F(-3, 2), F(5, 4)), (F(-2, 3), F(7, 6))),
            residual_bounds=((F(-1, 5), F(2, 7)),),
            coefficients=((F(1, 7), F(-1, 9), F(2, 3), F(-5, 4), F(3, 5)),
                          (F(-2, 11), F(1, 8), F(-3, 4), F(7, 6), F(-2, 5))),
            biases=(F(1, 10), F(-1, 6)),
        )
    if residual_count == 2:
        return Spec(
            parent_bounds=((-1, 1), (-1, 1)),
            residual_bounds=((F(-1, 5), F(1, 4)), (F(-1, 6), F(1, 3))),
            coefficients=((0, 0, 1, 1, 1, 0), (0, 0, 1, -1, 0, 1)),
            biases=(F(-9, 10), F(-1, 5)),
        )
    assert residual_count == 0
    return Spec(
        parent_bounds=((-1, 1), (-1, 1)), residual_bounds=(),
        coefficients=((0, 0, 1, 1), (0, 0, 1, -1)),
        biases=(F(-9, 10), F(-1, 5)),
    )


def _binding(spec, frame):
    residual_ids = tuple("v%d" % i for i in range(len(spec.residual_bounds)))
    return Binding(
        frame=frame,
        physical_ids=("f1", "f2", "q1", "q2") + residual_ids + ("y1", "y2"),
        phase_ids=("alpha1", "alpha2", "gamma1", "gamma2"),
    )


def _make(spec, budget=None):
    frame = object()
    binding = _binding(spec, frame)
    return build(spec, binding, enabled=True, budget=budget), frame, binding


def _case(residual_count):
    if residual_count not in _CASES:
        spec = _spec(residual_count)
        bank, frame, binding = _make(spec)
        _CASES[residual_count] = spec, bank, frame, binding
    return _CASES[residual_count]


def _dot(a, b):
    assert len(a) == len(b)
    return sum((F(x) * F(y) for x, y in zip(a, b)), F(0))


def _graph(spec, sources):
    sources = tuple(F(value) for value in sources)
    assert len(sources) == 2 + len(spec.residual_bounds)
    base = sources[:2] + (max(F(0), sources[0]), max(F(0), sources[1])) + sources[2:]
    h = tuple(_dot(row, base) + bias for row, bias in zip(spec.coefficients, spec.biases))
    return base + tuple(max(F(0), value) for value in h)


def _preactivations(spec, physical):
    base = physical[:-2]
    return physical[:2] + tuple(
        _dot(row, base) + bias for row, bias in zip(spec.coefficients, spec.biases)
    )


def _signs(spec, physical):
    return tuple(1 if value > 0 else -1 if value < 0 else 0
                 for value in _preactivations(spec, physical))


def _legal_phases(spec, physical):
    return product(*(((-1, 1) if value == 0 else (1,) if value > 0 else (-1,))
                     for value in _preactivations(spec, physical)))


def _unit(size, index):
    return tuple(F(int(i == index)) for i in range(size))


def _average(points, weights):
    assert len(points) == len(weights) and sum(weights, F(0)) == 1
    return tuple(sum((weight * point[i] for point, weight in zip(points, weights)), F(0))
                 for i in range(len(points[0])))


def _normal_equation(row, rhs):
    row = tuple(F(value) for value in row)
    first = next((value for value in row if value), None)
    if first is None:
        return row, F(rhs)
    return tuple(value / first for value in row), F(rhs) / first


def _linear_solution(equations):
    """Independent exact Gauss--Jordan elimination, with no candidate calls."""
    size = len(equations)
    matrix = [list(row) + [rhs] for row, rhs in equations]
    for column in range(size):
        pivot = next((i for i in range(column, size) if matrix[i][column]), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for i in range(size):
            if i == column or not matrix[i][column]:
                continue
            factor = matrix[i][column]
            matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[column])]
    return tuple(matrix[i][-1] for i in range(size))


def _reference_vertices(spec):
    """All labelled-cell vertices from independent H constraints.

    Parent guards are absorbed into their box bounds. Each of the sixteen
    cells then has box inequalities and its two actual child guards. We use
    a generic active-constraint enumeration, not the candidate's face routine.
    Repeated equations across child labels are cached only in this reference.
    """
    if spec in _REFERENCES:
        return _REFERENCES[spec]
    dimension = 2 + len(spec.residual_bounds)
    assert all(lo < hi for lo, hi in spec.residual_bounds)
    solutions = {}
    vertices = set()
    for parent_bits in product((0, 1), repeat=2):
        bounds = tuple((F(0), F(hi)) if active else (F(lo), F(0))
                       for (lo, hi), active in zip(spec.parent_bounds, parent_bits))
        bounds += tuple((F(lo), F(hi)) for lo, hi in spec.residual_bounds)
        child_forms = tuple(
            (F(row[0]) + parent_bits[0] * F(row[2]),
             F(row[1]) + parent_bits[1] * F(row[3]))
            + tuple(F(value) for value in row[4:])
            for row in spec.coefficients
        )
        for child_bits in product((0, 1), repeat=2):
            constraints = []
            for coordinate, (lo, hi) in enumerate(bounds):
                unit = _unit(dimension, coordinate)
                constraints.append((tuple(-value for value in unit), -lo, coordinate))
                constraints.append((unit, hi, coordinate))
            for row, bias, active in zip(child_forms, spec.biases, child_bits):
                sign = 1 if active else -1
                constraints.append((tuple(-sign * value for value in row),
                                    sign * F(bias), None))
            for chosen in combinations(range(len(constraints)), dimension):
                bound_coordinates = [constraints[i][2] for i in chosen
                                     if constraints[i][2] is not None]
                if len(set(bound_coordinates)) != len(bound_coordinates):
                    continue
                equations = tuple(sorted(_normal_equation(constraints[i][0], constraints[i][1])
                                         for i in chosen))
                if equations not in solutions:
                    solutions[equations] = _linear_solution(equations)
                point = solutions[equations]
                if point is None:
                    continue
                if all(_dot(row, point) <= rhs for row, rhs, _ in constraints):
                    vertices.add(_graph(spec, point))
    result = tuple(sorted(vertices))
    assert result
    _REFERENCES[spec] = result
    return result


def _assert_true_vertices(spec, bank):
    bounds = spec.parent_bounds + spec.residual_bounds
    assert tuple(bank.vertices) == tuple(sorted(set(bank.vertices)))
    assert len(bank.signs) == len(bank.vertices)
    for vertex, signs in zip(bank.vertices, bank.signs):
        assert len(vertex) == 6 + len(spec.residual_bounds)
        sources = vertex[:2] + vertex[4:-2]
        assert all(F(lo) <= value <= F(hi) for value, (lo, hi) in zip(sources, bounds))
        assert tuple(vertex) == _graph(spec, sources)
        assert tuple(signs) == _signs(spec, vertex)


def test_01_default_off_and_foreign_frame():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError("disabled build inspected its input")

    with pytest.raises(Rejected):
        build(Unreadable(), Unreadable())
    with pytest.raises(Rejected):
        build(Unreadable(), Unreadable(), enabled=1)
    spec = _spec()
    bank, frame, binding = _make(spec)
    bank.support((0,) * 6, frame=frame)
    compiled = bank.compile(frame=frame)
    assert compiled.physical_ids == binding.physical_ids
    assert compiled.phase_ids == binding.phase_ids
    with pytest.raises(Rejected):
        bank.support((0,) * 6, frame=object())
    with pytest.raises(Rejected):
        bank.compile(frame=object())
    _record(1, disabled_reads_input=False, foreign_frames_rejected=True,
            physical_ids=list(binding.physical_ids), phase_ids=list(binding.phase_ids))


def test_02_spec_types_and_bounds_fail_closed():
    spec = _spec()
    invalid = (
        {"parent_bounds": ((-1, 0), (-1, 1))},
        {"parent_bounds": ((1, -1), (-1, 1))},
        {"parent_bounds": ((-1, 1),)},
        {"parent_bounds": ((-1.0, 1), (-1, 1))},
        {"residual_bounds": ((1, 0),)},
        {"residual_bounds": ((-1, 1),) * 3},
        {"coefficients": ((0, 0, 1), (0, 0, 1))},
        {"coefficients": ((0, 0, True, 1), (0, 0, 1, -1))},
        {"coefficients": ((0, 0, 1.0, 1), (0, 0, 1, -1))},
        {"coefficients": ([0, 0, 1, 1], (0, 0, 1, -1))},
        {"biases": (0,)},
        {"biases": (False, 0)},
    )
    for changes in invalid:
        with pytest.raises(Rejected):
            _make(replace(spec, **changes))
    frame = object()
    binding = _binding(spec, frame)
    for changes in ({"phase_ids": ("alpha1",) * 4},
                    {"physical_ids": binding.physical_ids[:-1]},
                    {"phase_ids": ("f1", "alpha2", "gamma1", "gamma2")}):
        with pytest.raises(Rejected):
            build(spec, replace(binding, **changes), enabled=True)
    bank, frame, _ = _make(spec)
    for coefficients in ((0,) * 5, (0, 0, 0, 0, False, 1), (0, 0, 0, 0, 0.5, 1)):
        with pytest.raises(Rejected):
            bank.support(coefficients, frame=frame)
    _record(2, invalid_specs=len(invalid), malformed_bindings=3, malformed_queries=3)


def test_03_no_residual_matches_independent_phase_cells():
    spec, bank, _, _ = _case(0)
    reference = _reference_vertices(spec)
    assert tuple(bank.vertices) == reference
    assert len(bank.vertices) <= 37
    _record(3, residual_dimensions=0, labelled_cells=16, vertices=len(reference),
            original_grid_count=bank.original_grid_count, candidate_count=bank.candidate_count,
            independent_reference_equal=True, reference_supplies_candidate_bounds=False)


def test_04_one_residual_matches_independent_phase_cells():
    spec, bank, _, _ = _case(1)
    reference = _reference_vertices(spec)
    assert tuple(bank.vertices) == reference
    assert len(bank.vertices) <= 104
    _record(4, residual_dimensions=1, labelled_cells=16, vertices=len(reference),
            original_grid_count=bank.original_grid_count, candidate_count=bank.candidate_count,
            independent_reference_equal=True, asymmetric_mixed_coefficients=True)


def test_05_two_residuals_match_independent_phase_cells():
    spec, bank, _, _ = _case(2)
    reference = _reference_vertices(spec)
    assert tuple(bank.vertices) == reference
    assert len(bank.vertices) <= 277
    _record(5, residual_dimensions=2, labelled_cells=16, vertices=len(reference),
            original_grid_count=bank.original_grid_count, candidate_count=bank.candidate_count,
            independent_reference_equal=True, actual_residual_coordinates_retained=True)


def test_06_interior_zero_intersection_retained():
    spec, bank, frame, _ = _case(0)
    required = (F(11, 20), F(7, 20), F(11, 20), F(7, 20), F(0), F(0))
    assert required in bank.vertices
    index = bank.vertices.index(required)
    assert bank.signs[index] == (1, 1, 0, 0)
    compiled = bank.compile(frame=frame)
    for child_phases in product((-1, 1), repeat=2):
        assert compiled.satisfied(required, (1, 1) + child_phases, _unit(len(bank.vertices), index))
    assert all(value not in (F(-1), F(0), F(1)) for value in required[:2])
    _record(6, required_source=["11/20", "7/20"], interior_face_vertex=True,
            all_four_zero_child_labels=True)


def test_07_every_vertex_is_true_graph():
    total = 0
    for residual_count in range(3):
        spec, bank, _, _ = _case(residual_count)
        _assert_true_vertices(spec, bank)
        total += len(bank.vertices)
    _record(7, checked_vertices=total, exact_original_relu_graph=True,
            actual_network_input_witness=False)


def test_08_signed_rows_and_zero_labels():
    spec = replace(_spec(), coefficients=((0, 0, 1, -1), (1, 1, 0, 0)), biases=(0, 0))
    bank, frame, _ = _make(spec)
    compiled = bank.compile(frame=frame)
    origin = (F(0),) * 6
    index = bank.vertices.index(origin)
    for phases in product((-1, 1), repeat=4):
        assert compiled.satisfied(origin, phases, _unit(len(bank.vertices), index))
    wrong_labels_checked = 0
    for index, vertex in enumerate(bank.vertices):
        phases = next(_legal_phases(spec, vertex))
        weights = _unit(len(bank.vertices), index)
        assert compiled.satisfied(vertex, phases, weights)
        values = tuple(vertex) + tuple(F(value) for value in phases) + weights
        assert len(values) == compiled.n_columns
        assert all(_dot(tuple(coefficient for _, coefficient in row.terms),
                        tuple(values[column] for column, _ in row.terms)) == row.rhs
                   for row in compiled.eq)
        assert all(_dot(tuple(coefficient for _, coefficient in row.terms),
                        tuple(values[column] for column, _ in row.terms)) <= row.rhs
                   for row in compiled.le)
        for phase_index, sign in enumerate(_signs(spec, vertex)):
            if sign:
                wrong = list(phases)
                wrong[phase_index] = -sign
                assert not compiled.satisfied(vertex, tuple(wrong), weights)
                wrong_labels_checked += 1
    assert wrong_labels_checked
    _record(8, signed_original_phases=4, origin_legal_labels=16,
            wrong_nonzero_labels_rejected=wrong_labels_checked)


def test_09_integer_phase_compatible_mixture_is_true_graph():
    spec, bank, frame, _ = _case(0)
    compiled = bank.compile(frame=frame)
    checked = 0
    for phases in product((-1, 1), repeat=4):
        compatible = [i for i, signs in enumerate(bank.signs)
                      if all(sign == 0 or sign == phase for sign, phase in zip(signs, phases))]
        if len(compatible) < 2:
            continue
        i, j = compatible[0], compatible[-1]
        weights = tuple(F(1, 3) if k == i else F(2, 3) if k == j else F(0)
                        for k in range(len(bank.vertices)))
        physical = _average((bank.vertices[i], bank.vertices[j]), (F(1, 3), F(2, 3)))
        assert compiled.satisfied(physical, phases, weights)
        assert physical == _graph(spec, physical[:2])
        assert all(sign == 0 or sign == phase
                   for sign, phase in zip(_signs(spec, physical), phases))
        checked += 1
    assert checked >= 4
    _record(9, compatible_integer_phase_mixtures=checked,
            original_bits_retained=True, integer_graph_exact=True)


def test_10_physical_and_phase_support():
    spec, bank, frame, _ = _case(1)
    width = len(bank.vertices[0])
    queries = (
        ((0,) * width, (0, 0, 0, 0)),
        ((1, -2, 3, -1, F(2, 3), 2, -3), (0, 0, 0, 0)),
        ((0, 0, -1, 2, F(-3, 5), -2, 1), (1, -2, 3, -4)),
        ((0,) * width, (-3, 1, 0, 2)),
    )
    support_values = []
    for physical_coefficients, phase_coefficients in queries:
        result = bank.support(physical_coefficients, phase_coefficients, frame=frame)
        reference = max(_dot(physical_coefficients, vertex) + _dot(phase_coefficients, phases)
                        for vertex in _reference_vertices(spec)
                        for phases in _legal_phases(spec, vertex))
        assert result.value == reference
        assert result.physical == bank.vertices[result.vertex_index]
        assert result.phases in tuple(_legal_phases(spec, result.physical))
        assert result.value == (_dot(physical_coefficients, result.physical)
                                + _dot(phase_coefficients, result.phases))
        support_values.append(str(result.value))
    tied = bank.support((0,) * width, frame=frame)
    assert tied.vertex_index == 0
    assert all(phase == (-1 if sign == 0 else sign)
               for sign, phase in zip(bank.signs[0], tied.phases))
    _record(10, complete_physical_and_signed_phase_queries=len(queries),
            support_values=support_values,
            exact_independent_reference=True, deterministic_tie=True)


def test_11_shared_bank_control_and_d062_reference():
    r, s, e1, e2 = F(3, 2), F(4, 3), F(1, 10), F(1, 5)
    spec = replace(_spec(), coefficients=((0, 0, 1, r), (0, 0, 1, -s)),
                   biases=(-r-e1, -e2))
    bank, frame, _ = _make(spec)
    diagonal = _average((_graph(spec, (0, 0)), _graph(spec, (1, 1))), (F(1, 2),) * 2)
    opposite = _average((_graph(spec, (1, 0)), _graph(spec, (0, 1))), (F(1, 2),) * 2)
    assert diagonal[:4] == opposite[:4] == (F(1, 2),) * 4
    assert diagonal[-2] == (1-e1)/2
    assert opposite[-1] == (1-e2)/2
    fake = diagonal[:4] + (diagonal[-2], opposite[-1])
    direction = (0, 0, -1, 0, 1, 1)
    gap = _dot(direction, fake)
    assert gap == F(7, 20)
    assert bank.support(direction, frame=frame).value == 0
    old_d062_plane_supports = (F(0), -e1, -e2, 1-min(r, s)-e1-e2)
    assert max(old_d062_plane_supports) == 0
    tau = gap / 2
    assert max(F(0), -tau) == 0 and max(F(0), gap-tau) == F(7, 40)
    _record(11, independent_complete_blocks_allow_physical_gap=str(gap),
            shared_support="0", existing_d062_support="0", child_false_value=str(gap-tau),
            exceeds_d062=False, d214_fiber_evaluated_on_this_control=False, formal_gain=0)


def test_12_source_binding_without_vertex_source_atoms():
    spec = Spec(
        parent_bounds=((F(-1, 2), F(3, 2)), (F(-3, 4), F(5, 4))), residual_bounds=(),
        coefficients=((0, 0, 1, -1), (0, 0, F(3, 10), F(1, 2))),
        biases=(F(1, 10), F(-1, 5)),
    )
    bank, frame, _ = _make(spec)
    original_input = (F(1), F(0))
    f = (original_input[0]+original_input[1]-F(1, 2),
         original_input[0]-original_input[1]+F(1, 4))
    target = _graph(spec, f)
    assert f == (F(1, 2), F(5, 4)) and target[-2:] == (F(0), F(23, 40))
    assert target not in bank.vertices
    left = _graph(spec, (0, F(5, 4)))
    right = _graph(spec, (F(23, 20), F(5, 4)))
    i, j = bank.vertices.index(left), bank.vertices.index(right)
    weights = tuple(F(13, 23) if k == i else F(10, 23) if k == j else F(0)
                    for k in range(len(bank.vertices)))
    assert _average((left, right), (F(13, 23), F(10, 23))) == target
    assert bank.compile(frame=frame).satisfied(target, (1, 1, -1, 1), weights)

    def decode(physical):
        return ((physical[0]+physical[1]+F(1, 4))/2,
                (physical[0]-physical[1]+F(3, 4))/2)

    assert decode(target) == original_input
    assert decode(left) == (F(3, 4), F(-1, 4))
    assert decode(right) == (F(53, 40), F(13, 40))
    assert any(value < 0 or value > 1 for value in decode(left))
    assert any(value < 0 or value > 1 for value in decode(right))
    _record(12, original_input=["1", "0"], actual_source_binding_preserved=True,
            weights=["13/23", "10/23"], native_full_source_qualified=False,
            local_vertices_are_original_input_atoms=False,
            naive_conditioned_input_atom_lift_would_remove_true_point=True)


def test_13_constant_residual_and_dependent_zero_planes():
    spec = Spec(
        parent_bounds=((-1, 1), (-1, 1)),
        residual_bounds=((F(1, 3), F(1, 3)), (F(-1, 2), F(-1, 2))),
        coefficients=((0, 0, 1, 1, 1, 0), (0, 0, 2, 2, 2, 0)),
        biases=(F(-9, 10), F(-9, 5)),
    )
    bank, frame, _ = _make(spec)
    _assert_true_vertices(spec, bank)
    assert len(bank.vertices) <= 21
    assert all(vertex[4:6] == (F(1, 3), F(-1, 2)) and vertex[-1] == 2*vertex[-2]
               for vertex in bank.vertices)
    zero = replace(spec, coefficients=((0,) * 6, (0,) * 6), biases=(0, 0))
    zero_bank, zero_frame, _ = _make(zero)
    assert len(zero_bank.vertices) == 9
    compiled = zero_bank.compile(frame=zero_frame)
    for i, vertex in enumerate(zero_bank.vertices):
        parent_phases = tuple(1 if value > 0 else -1 for value in vertex[:2])
        for child_phases in product((-1, 1), repeat=2):
            assert compiled.satisfied(vertex, parent_phases+child_phases,
                                      _unit(len(zero_bank.vertices), i))
    assert bank.compile(frame=frame).n_columns == 12 + len(bank.vertices)
    _record(13, constant_residual_coordinates=2, dependent_plane_vertices=len(bank.vertices),
            identically_zero_plane_vertices=9, zero_labels_not_removed=True)


def test_14_compiled_counts_and_nnz():
    counts = []
    for residual_count, maximum_n, maximum_nnz in ((0, 37, 458), (1, 104, 1367), (2, 277, 3894)):
        _, bank, frame, binding = _case(residual_count)
        compiled = bank.compile(frame=frame)
        size = len(bank.vertices)
        nnz = sum(len(row.terms) for row in compiled.eq + compiled.le)
        assert size <= maximum_n
        assert compiled.n_lambda == size
        assert compiled.n_columns == 10+residual_count+size
        assert len(compiled.eq) == 7+residual_count
        assert len(compiled.le) == size+8
        assert nnz <= (12+residual_count)*size+14+residual_count <= maximum_nnz
        assert compiled.physical_ids == binding.physical_ids
        assert compiled.phase_ids == binding.phase_ids
        assert len(compiled.phase_ids) == 4
        assert all(tuple(sorted(row.terms)) == row.terms
                   and len({column for column, _ in row.terms}) == len(row.terms)
                   and all(0 <= column < compiled.n_columns and coefficient != 0
                           for column, coefficient in row.terms)
                   for row in compiled.eq+compiled.le)
        counts.append({"residuals": residual_count, "vertices": size, "eq": len(compiled.eq),
                       "le": len(compiled.le), "nnz": nnz, "total_columns": compiled.n_columns})
    _record(14, local_counts=counts, original_hz_and_evidence_cost_excluded=True,
            terminal_or_gpu_performance_claim=False)


def test_15_shared_budget_and_bit_limit():
    budget = Budget()
    spec = _spec()
    first, frame, _ = _make(spec, budget)
    assert first.budget is budget
    built_work, built_entries = budget.work, budget.entries
    assert built_work > 0 and built_entries > 0
    first.support((0, 0, 0, 0, 1, -1), frame=frame)
    assert budget.work > built_work
    first.compile(frame=frame)
    assert budget.entries > built_entries
    before_second = budget.work, budget.entries
    second, _, _ = _make(spec, budget)
    assert second.budget is budget
    assert budget.work > before_second[0] and budget.entries > before_second[1]
    for limits in ({"max_work": 0}, {"max_branch": 0}, {"max_entries": 0}):
        exhausted = Budget(**limits)
        with pytest.raises(Rejected):
            _make(spec, exhausted)
        assert exhausted.failed
        with pytest.raises(Rejected):
            _make(spec, exhausted)
        assert exhausted.failed
    with pytest.raises(Rejected):
        Budget(max_bits=513)
    with pytest.raises(Rejected):
        _make(replace(spec, biases=(F(1 << 512), 0)))
    _record(15, cumulative_work=budget.work, cumulative_entries=budget.entries,
            shared_budget_identity=True, exhausted_budget_latched=True,
            oversize_512_bit_input_rejected=True)


def test_16_asymmetric_permutations_and_summary():
    spec, bank, _, _ = _case(1)
    swapped_parents = replace(
        spec, parent_bounds=tuple(reversed(spec.parent_bounds)),
        coefficients=tuple((row[1], row[0], row[3], row[2])+row[4:]
                           for row in spec.coefficients),
    )
    parent_bank, _, _ = _make(swapped_parents)
    parent_permutation = (1, 0, 3, 2, 4, 5, 6)
    assert {tuple(vertex[i] for i in parent_permutation) for vertex in parent_bank.vertices} == set(bank.vertices)
    swapped_children = replace(spec, coefficients=tuple(reversed(spec.coefficients)),
                               biases=tuple(reversed(spec.biases)))
    child_bank, _, _ = _make(swapped_children)
    assert {vertex[:-2]+(vertex[-1], vertex[-2]) for vertex in child_bank.vertices} == set(bank.vertices)
    _assert_true_vertices(swapped_parents, parent_bank)
    _assert_true_vertices(swapped_children, child_bank)
    _record(16, ordinary_asymmetric_mixed_spec=True, parent_permutation_exact=True,
            child_permutation_exact=True)
    expected = (Path(__file__).resolve().parents[2]
                / "results/d228_joint_bank_component_20261005_v1")
    run = Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"])
    assert run == expected and run.is_dir() and not run.is_symlink()
    missing = [name for name in _NAMES if name not in _EVIDENCE]
    payload = {
        "schema": "d228_joint_bank_component_mathematical_tests_v1",
        "scope": "default-off local mathematical primitive; not a new capability claim",
        "required_tests": list(_NAMES), "completed_tests": _EVIDENCE,
        "missing_tests": missing, "independent_reference_uses_phase_cells": True,
        "candidate_uses_test_reference_or_external_solver": False,
        "actual_model_run": False, "native_qualified": False, "gpu_qualified": False,
        "formal_gain": 0, "independent_e0_gain": 0,
    }
    with (run / "summary.json").open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    assert not missing
