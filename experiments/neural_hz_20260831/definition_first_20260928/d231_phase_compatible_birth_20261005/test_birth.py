"""Twelve frozen mathematical tests, not native/model/runtime qualification.

The independent two-dimensional oracle enumerates fixed labelled cells only
in the TEST process.  It supplies no points, bounds, or phases to candidate
construction.  No solver, original model, or numerical search is used.
Do not import, collect, compile, or execute this file before its freeze.
"""

from fractions import Fraction as F
from itertools import combinations, product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as joint
from experiments.neural_hz_20260831.definition_first_20260928.d231_phase_compatible_birth_20261005 import birth


ZERO, ONE = F(0), F(1)
RUN = (Path(__file__).resolve().parents[2]
       / "results/d231_phase_compatible_birth_20261005_v1")
_NAMES = (
    "default_off_and_frame_identity",
    "mixed_fifth_gate_strict_gap",
    "all_vertices_remain_true_graph",
    "cut_completeness_independent_geometry",
    "integer_compatible_mixture_is_graph",
    "zero_labels_and_signed_support",
    "repeated_crossing_birth",
    "common_source_residual_control",
    "resource_failure_is_sticky",
    "exact_arithmetic_and_invalid_inputs",
    "compiled_rows_and_counterexample",
    "original_bank_immutable_and_record",
)
_EVIDENCE, _CASES = {}, {}
H = (F(1, 4), F(1, 8), ZERO, ZERO, ONE, F(-1, 2))
H_BIAS = F(-1, 8)
F_WEIGHTS = (F(-15, 16), F(-31, 32), F(2), F(2),
             F(1, 4), F(-1, 8), F(-1, 2))
F_BIAS = F(-1, 32)


def _record(number, **values):
    name = _NAMES[number-1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    """Independent exclusive writer; moving evidence does not change math."""
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _dot(left, right):
    assert len(left) == len(right)
    return sum((F(a)*F(b) for a, b in zip(left, right)), ZERO)


def _average(points, weights):
    assert len(points) == len(weights) and sum(weights, ZERO) == ONE
    return tuple(sum((w*p[i] for w, p in zip(weights, points)), ZERO)
                 for i in range(len(points[0])))


def _unit(size, index):
    return tuple(F(int(i == index)) for i in range(size))


def _spec():
    return joint.Spec(((-1, 1), (-1, 1)), (),
                      ((0, 0, 1, F(1, 2)), (0, 0, F(1, 2), -1)),
                      (F(-3, 4), F(-1, 4)))


def _make(spec, budget=None):
    frame = object()
    residual_ids = tuple("source:v%d" % i for i in range(len(spec.residual_bounds)))
    identity = joint.Binding(
        frame, ("source:x1", "source:x2", "q1", "q2") + residual_ids + ("y1", "y2"),
        ("original:alpha1", "original:alpha2", "original:gamma1", "original:gamma2"),
    )
    result = joint.build(spec, identity, enabled=True, budget=budget)
    return result, frame


def _base():
    if "base" not in _CASES:
        bank, frame = _make(_spec())
        _CASES["base"] = bank, frame
        _CASES["base_snapshot"] = (bank.spec, bank.vertices, bank.signs,
                                    bank.binding.physical_ids, bank.binding.phase_ids)
    return _CASES["base"]


def _case():
    if "grown" not in _CASES:
        base, frame = _base()
        old = birth.from_bank(base, enabled=True, frame=frame)
        new = birth.advance(old, H, H_BIAS, "original:r", "original:rho",
                            enabled=True, frame=frame)
        _CASES["grown"] = old, new, frame
        _CASES["old_snapshot"] = (old.vertices, old.signs, old.physical_ids, old.phase_ids)
        _CASES["new_snapshot"] = (new.vertices, new.signs, new.physical_ids, new.phase_ids)
    return _CASES["grown"]


def _tiny(budget=None):
    spec = joint.Spec(((-1, 1), (-1, 1)), (), ((0, 0, 0, 0),)*2, (0, 0))
    base, frame = _make(spec, budget)
    return base, frame


def _old_graph(x1, x2):
    q1, q2 = max(ZERO, x1), max(ZERO, x2)
    y1 = max(ZERO, q1+q2/2-F(3, 4))
    y2 = max(ZERO, q1/2-q2-F(1, 4))
    return (x1, x2, q1, q2, y1, y2)


def _graph(x1, x2):
    old = _old_graph(x1, x2)
    return old + (max(ZERO, _dot(H, old)+H_BIAS),)


def _pre(physical):
    x1, x2, q1, q2, y1, y2 = physical[:6]
    return (x1, x2, q1+q2/2-F(3, 4), q1/2-q2-F(1, 4),
            _dot(H, physical[:6])+H_BIAS)


def _sign(values):
    return tuple(1 if value > ZERO else -1 if value < ZERO else 0 for value in values)


def _legal(signs):
    return product(*(((-1, 1) if sign == 0 else (sign,)) for sign in signs))


def _lambda(bank, weighted_points):
    result = [ZERO]*len(bank.vertices)
    for point, weight in weighted_points:
        result[bank.vertices.index(point)] += weight
    assert sum(result, ZERO) == ONE
    return tuple(result)


def _aadd(*scaled, bias=ZERO):
    """Test-only affine arithmetic, order (constant,x1,x2)."""
    return (F(bias)+sum((scale*form[0] for scale, form in scaled), ZERO),
            sum((scale*form[1] for scale, form in scaled), ZERO),
            sum((scale*form[2] for scale, form in scaled), ZERO))


def _independent_geometry():
    """Exact fixed five-gate cell vertices; no candidate routines or outputs."""
    if "oracle" in _CASES:
        return _CASES["oracle"]
    vertices = set()
    f1, f2, zero = (ZERO, ONE, ZERO), (ZERO, ZERO, ONE), (ZERO, ZERO, ZERO)
    for bits in product((-1, 1), repeat=5):
        q1, q2 = f1 if bits[0] > 0 else zero, f2 if bits[1] > 0 else zero
        g1 = _aadd((ONE, q1), (F(1, 2), q2), bias=F(-3, 4))
        g2 = _aadd((F(1, 2), q1), (-ONE, q2), bias=F(-1, 4))
        y1, y2 = g1 if bits[2] > 0 else zero, g2 if bits[3] > 0 else zero
        h = _aadd((ONE, y1), (F(-1, 2), y2), (F(1, 4), f1),
                  (F(1, 8), f2), bias=H_BIAS)
        rows = [(ONE, ZERO, ONE), (-ONE, ZERO, ONE),
                (ZERO, ONE, ONE), (ZERO, -ONE, ONE)]
        rows += [(-bit*form[1], -bit*form[2], bit*form[0])
                 for bit, form in zip(bits, (f1, f2, g1, g2, h))]
        for first, second in combinations(rows, 2):
            a, b, c = first
            d, e, f = second
            determinant = a*e-d*b
            if determinant == ZERO:
                continue
            x1, x2 = (c*e-f*b)/determinant, (a*f-d*c)/determinant
            if all(a*x1+b*x2 <= c for a, b, c in rows):
                vertices.add(_graph(x1, x2))
    assert vertices
    result = tuple(sorted(vertices))
    _CASES["oracle"] = result
    return result


def test_01_default_off_and_frame_identity():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError("disabled operation inspected its argument")

    with pytest.raises(birth.Rejected):
        birth.from_bank(Unreadable(), frame=object())
    with pytest.raises(birth.Rejected):
        birth.advance(Unreadable(), (), 0, "r", "rho", frame=object())
    with pytest.raises(birth.Rejected):
        birth.from_bank(Unreadable(), enabled=1, frame=object())
    base, frame = _base()
    with pytest.raises(birth.Rejected):
        birth.from_bank(base, enabled=True, frame=object())
    old, new, same_frame = _case()
    assert same_frame is frame and old.frame is frame and new.frame is frame
    assert old.budget is base.budget and new.budget is base.budget
    for operation in (
        lambda: new.support((0,)*7, frame=object()),
        lambda: new.compile(frame=object()),
        lambda: birth.advance(old, H, H_BIAS, "r2", "rho2", enabled=True, frame=object()),
    ):
        with pytest.raises(birth.Rejected):
            operation()
    _record(1, default_off=True, foreign_frame_rejected=True, shared_budget=True)


def test_02_mixed_fifth_gate_strict_gap():
    old, new, frame = _case()
    support = new.support(F_WEIGHTS, frame=frame)
    assert support.value+F_BIAS == F(63, 32)
    corners = tuple(_graph(F(x), F(y)) for x, y in product((-1, 1), repeat=2))
    assert tuple(_pre(p)[-1] for p in corners) == (F(-1, 2), F(-1, 4), F(1, 8), ONE)
    assert max(_dot(F_WEIGHTS, point)+F_BIAS for point in corners) == F(63, 32)
    assert ONE-F(1, 4)*F(3, 2) > ZERO and ONE-F(1, 4)*F(9, 8) > ZERO
    left, right = _old_graph(-ONE, -ONE), _old_graph(ONE, ONE)
    mean = _average((left, right), (F(2, 3), F(1, 3)))
    assert mean == (F(-1, 3), F(-1, 3), F(1, 3), F(1, 3), F(1, 4), ZERO)
    assert _dot(H, mean)+H_BIAS == ZERO
    false_point = mean+(ZERO,)
    assert _dot(F_WEIGHTS, false_point)+F_BIAS == F(2)
    assert F(2)-(support.value+F_BIAS) == F(1, 32)
    assert new.physical_ids == old.physical_ids+("original:r",)
    assert new.phase_ids == old.phase_ids+("original:rho",)
    _record(2, new_upper="63/32", complete_old_hull_then_new_graph_lower="2",
            physical_gap="1/32", live_sources=True, both_old_children_consumed=True,
            not_a_concrete_adv=True, no_all_old_methods_dominance_claim=True)


def test_03_all_vertices_remain_true_graph():
    old, new, _ = _case()
    assert tuple(new.vertices) == tuple(sorted(set(new.vertices)))
    assert len(new.signs) == len(new.vertices)
    for point, signs in zip(new.vertices, new.signs):
        assert -ONE <= point[0] <= ONE and -ONE <= point[1] <= ONE
        assert point == _graph(point[0], point[1])
        assert tuple(signs) == _sign(_pre(point))
    lifted_old = {p+(max(ZERO, _dot(H, p)+H_BIAS),) for p in old.vertices}
    assert lifted_old <= set(new.vertices)
    assert len(new.vertices) <= len(old.vertices)+len(old.vertices)**2//4
    _record(3, old_vertices=len(old.vertices), new_vertices=len(new.vertices),
            every_point_true_graph=True, all_old_points_retained=True)


def test_04_cut_completeness_independent_geometry():
    _, new, frame = _case()
    oracle = _independent_geometry()
    assert set(oracle) <= set(new.vertices)
    directions = (F_WEIGHTS, (1, -2, 1, 0, -1, 2, -1),
                  (0, 1, -1, 2, 1, -1, 1), _unit(7, 6))
    for direction in directions:
        assert new.support(tuple(direction), frame=frame).value == max(
            _dot(direction, point) for point in oracle)
    assert any(point[-1] == ZERO and _pre(point)[-1] == ZERO
               and point[:6] not in _case()[0].vertices for point in oracle)
    _record(4, independent_cell_vertices=len(oracle), fixed_test_cells=32,
            candidate_reads_reference=False, complete_vertex_inclusion=True,
            directions_checked=len(directions))


def test_05_integer_compatible_mixture_is_graph():
    _, new, frame = _case()
    checked = 0
    for (left, left_sign), (right, right_sign) in combinations(zip(new.vertices, new.signs), 2):
        if any(a*b == -1 for a, b in zip(left_sign, right_sign)):
            continue
        mean = _average((left, right), (F(1, 3), F(2, 3)))
        assert mean == _graph(mean[0], mean[1])
        checked += 1
    assert checked > 0
    points = (_graph(-ONE, -ONE), _graph(-ONE, ZERO))
    mean = _average(points, (F(1, 3), F(2, 3)))
    weights = _lambda(new, tuple(zip(points, (F(1, 3), F(2, 3)))))
    assert new.compile(frame=frame).satisfied(mean, (-1,)*5, weights)
    _record(5, compatible_pairs_checked=checked, integer_phases_preserve_graph=True,
            nontrivial_shared_lambda_checked=True)


def test_06_zero_labels_and_signed_support():
    _, new, frame = _case()
    compiled = new.compile(frame=frame)
    zero_index = next(i for i, signs in enumerate(new.signs) if signs[-1] == 0)
    zero_point, zero_signs = new.vertices[zero_index], new.signs[zero_index]
    weights = tuple(F(int(i == zero_index)) for i in range(len(new.vertices)))
    count = 0
    for phases in _legal(zero_signs):
        assert compiled.satisfied(zero_point, phases, weights)
        count += 1
    assert count >= 2
    physical = (F(1, 7), F(-2, 9), ZERO, F(1, 3), F(-1, 4), ZERO, F(2, 5))
    phase = (F(1, 2), F(-3, 4), F(1, 5), F(-1, 3), F(2, 3))
    expected = max(_dot(physical, point)+_dot(phase, phases)
                   for point, signs in zip(new.vertices, new.signs)
                   for phases in _legal(signs))
    result = new.support(physical, phase, frame=frame)
    assert result.value == expected
    assert result.physical == new.vertices[result.vertex_index]
    assert tuple(result.phases) in tuple(_legal(new.signs[result.vertex_index]))
    assert _dot(physical, result.physical)+_dot(phase, result.phases) == result.value
    _record(6, zero_phase_assignments=count, signed_support=str(expected),
            old_zero_labels_preserved=True)


def test_07_repeated_crossing_birth():
    _, new, frame = _case()
    before = new.budget.work
    repeated = birth.advance(new, _unit(7, 6), F(-1, 2), "original:s", "original:sigma",
                             enabled=True, frame=frame)
    assert repeated.budget is new.budget and repeated.budget.work > before
    assert repeated.physical_dim == 8 and repeated.phase_count == 6
    assert repeated.physical_ids == new.physical_ids+("original:s",)
    assert repeated.phase_ids == new.phase_ids+("original:sigma",)
    for point, signs in zip(repeated.vertices, repeated.signs):
        assert point[:7] == _graph(point[0], point[1])
        assert point[-1] == max(ZERO, point[6]-F(1, 2))
        assert tuple(signs) == _sign(_pre(point)+(point[6]-F(1, 2),))
    assert repeated.support(_unit(8, 7), frame=frame).value == F(1, 2)
    _record(7, repeated_vertices=len(repeated.vertices), original_bits=6,
            crossing_pre_bounds=["-1/2", "1/2"], exact_output_upper="1/2",
            unbounded_depth_or_affordability_claim=False)


def test_08_common_source_residual_control():
    spec = joint.Spec(((-1, 1), (-1, 1)),
                      ((F(-1, 4), F(1, 4)), (F(-1, 8), F(1, 8))),
                      ((0, 0, 1, F(1, 2), 1, 0),
                       (0, 0, F(1, 2), 1, -1, 1)),
                      (F(-5, 4), F(-5, 4)))
    base, frame = _make(spec)
    old = birth.from_bank(base, enabled=True, frame=frame)
    readout = (ZERO, ZERO, F(-1, 4), F(-1, 4), ZERO, ZERO, ONE, ONE)
    assert old.support(readout, frame=frame).value == F(1, 8)
    new = birth.advance(old, readout, F(-11, 80), "original:stable", "original:stable_phase",
                        enabled=True, frame=frame)
    assert len(new.vertices) == len(old.vertices)
    assert all(p[-1] == ZERO and signs[-1] == -1 for p, signs in zip(new.vertices, new.signs))
    assert new.support(_unit(9, 8), frame=frame).value == ZERO
    assert new.budget is base.budget
    _record(8, fixed_sources=["z", "t"], residual_matrix_retained=True,
            old_readout_upper="1/8", next_output_upper="0",
            restored_old_reference_only=True, no_source_basis_search=True)


def test_09_resource_failure_is_sticky():
    reference, _ = _tiny()
    work_to_build = reference.budget.work
    limited = joint.Budget(max_work=work_to_build+1)
    base, frame = _tiny(limited)
    assert limited.work == work_to_build
    with pytest.raises(birth.Rejected):
        birth.from_bank(base, enabled=True, frame=frame)
    assert limited.failed
    charged = (limited.work, limited.entries)
    with pytest.raises(birth.Rejected):
        birth.from_bank(base, enabled=True, frame=frame)
    assert limited.failed and (limited.work, limited.entries) == charged
    _record(9, sticky_failure=True, no_partial_result=True,
            deterministic_tiny_build_work=work_to_build)


def test_10_exact_arithmetic_and_invalid_inputs():
    base, frame = _tiny()
    old = birth.from_bank(base, enabled=True, frame=frame)
    direction = (F(1, 3), F(1, 5), ZERO, ZERO, ZERO, ZERO)
    new = birth.advance(old, direction, F(-1, 7), "rational:r", "rational:rho",
                        enabled=True, frame=frame)
    assert new.support(_unit(7, 6), frame=frame).value == F(41, 105)
    assert all(isinstance(value, F) for point in new.vertices for value in point)
    invalid = (
        ((True, 0, 0, 0, 0, 0), 0, "bad:r", "bad:rho"),
        ((0.5, 0, 0, 0, 0, 0), 0, "bad:r", "bad:rho"),
        ((0,)*5, 0, "bad:r", "bad:rho"),
        ([0]*6, 0, "bad:r", "bad:rho"),
        ((0,)*6, False, "bad:r", "bad:rho"),
        ((0,)*6, 0.5, "bad:r", "bad:rho"),
        ((0,)*6, 0, old.physical_ids[0], "bad:rho"),
        ((0,)*6, 0, "bad:r", old.phase_ids[0]),
        ((0,)*6, 0, "same", "same"),
    )
    for coefficients, bias, physical_id, phase_id in invalid:
        with pytest.raises(birth.Rejected):
            birth.advance(old, coefficients, bias, physical_id, phase_id,
                          enabled=True, frame=frame)
    # Independent lifetime: the ordinary hard 512-bit fail-closed check must
    # not poison any succeeding fixture or invite precision rescue.
    bit_base, bit_frame = _tiny()
    bit_old = birth.from_bank(bit_base, enabled=True, frame=bit_frame)
    with pytest.raises(birth.Rejected):
        birth.advance(bit_old, (0,)*6, 1 << 512, "too_big:r", "too_big:rho",
                      enabled=True, frame=bit_frame)
    _record(10, rational_upper="41/105", invalid_inputs_rejected=len(invalid),
            hard_bits=512, no_float_or_bool_coercion=True)


def test_11_compiled_rows_and_counterexample():
    _, new, frame = _case()
    compiled = new.compile(frame=frame)
    n, d, b = len(new.vertices), new.physical_dim, new.phase_count
    assert compiled.physical_ids == new.physical_ids and compiled.phase_ids == new.phase_ids
    assert compiled.n_lambda == n and compiled.n_columns == d+b+n
    assert len(compiled.eq) == d+1 and len(compiled.le) == n+2*b
    nnz = sum(len(row.terms) for row in compiled.eq+compiled.le)
    assert nnz <= (d+b+2)*n+d+2*b
    left, right = _graph(-ONE, -ONE), _graph(ONE, ONE)
    weights = _lambda(new, ((left, F(2, 3)), (right, F(1, 3))))
    actual_mean = _average((left, right), (F(2, 3), F(1, 3)))
    phases = _average((_sign(_pre(left)), _sign(_pre(right))), (F(2, 3), F(1, 3)))
    assert compiled.satisfied(actual_mean, phases, weights)
    false_physical = actual_mean[:6]+(ZERO,)
    false_phases = phases[:4]+(ZERO,)
    assert _dot(H, false_physical[:6])+H_BIAS == ZERO
    assert not compiled.satisfied(false_physical, false_phases, weights)
    # The support certificate excludes EVERY lambda extension, not just the
    # displayed corner weights: its full physical F is 2 > 63/32.
    assert _dot(F_WEIGHTS, false_physical) > new.support(F_WEIGHTS, frame=frame).value
    _record(11, variables=compiled.n_columns, lambda_columns=n,
            eq=len(compiled.eq), le=len(compiled.le), nnz=nnz,
            nnz_bound=(d+b+2)*n+d+2*b, false_point_all_lambdas_excluded_by_support=True)


def test_12_original_bank_immutable_and_record():
    base, frame = _base()
    old, new, same_frame = _case()
    assert frame is same_frame
    assert (base.spec, base.vertices, base.signs, base.binding.physical_ids,
            base.binding.phase_ids) == _CASES["base_snapshot"]
    assert (old.vertices, old.signs, old.physical_ids, old.phase_ids) == _CASES["old_snapshot"]
    assert (new.vertices, new.signs, new.physical_ids, new.phase_ids) == _CASES["new_snapshot"]
    assert not base.budget.failed
    _record(12, old_geometry_immutable=True, original_ids_preserved=True,
            main_lifetime_work=base.budget.work, main_lifetime_entries=base.budget.entries,
            cost_scope="shared logical Budget, not complete Python/native/GPU cost")
    assert tuple(_EVIDENCE) == _NAMES
    _record_file("summary.json", {
        "component": "phase_compatible_birth", "test_count": 12,
        "records": _EVIDENCE, "mathematical_component_only": True,
        "ordinary_model_executed": False, "gpu_executed": False,
        "native_binding_or_installation": False, "validated_adv": False,
        "formal_score_changed": False, "publication_novelty_claim": False,
    })
