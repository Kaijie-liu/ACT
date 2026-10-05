"""Opt-in exact-rational signed readout envelopes on the original D112 H.

This component does not bind irrational BN coefficients or native model ports.
It authenticates the original exact gates with D119, keeps every original
column/phase/predicate, and derives both directions from the SAME signed kernel.
The retained-entry preflight is not a complete physical/whole-work certificate.
"""
from dataclasses import replace
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct


Form = ef.Form
System = ef.System
Gate = ct.Gate
MAX_ENTRIES = ct.MAX_ENTRIES
ZERO, ONE = F(0), F(1)


def append_mixed_consumer(system, gates, knots, readout, *, frames,
                          readout_frame, child=None, enabled=False,
                          max_entries=MAX_ENTRIES):
    """Install two signed envelopes, and optionally an original ReLU consumer.

    Weights and the COMPLETE rest are extracted from the actual readout Form;
    there is no caller-supplied rest or independent witness-bound parameter.
    If child is present, its authenticated original preactivation must equal
    readout.  ReLU's upper secant uses the derived envelope's own box.
    """
    if enabled is False:
        return system, {'enabled': False}
    if enabled is not True:
        raise ValueError('strict boolean opt-in required')
    old_entries, binary = ct._schema(system, max_entries)
    if (type(gates) is not tuple or len(gates) < 3
            or type(knots) is not tuple or len(knots) != len(gates)
            or type(frames) is not tuple or len(frames) != len(gates)):
        raise ValueError('group shape')
    if (type(readout_frame) is not int or readout_frame != system.frame
            or any(type(frame) is not int or frame != system.frame
                   for frame in frames)):
        raise ValueError('source/readout frame mismatch')
    previous = F(-1)
    for knot in knots:
        knot = ct._fraction(knot)
        if not previous < knot <= ONE or knot < ZERO:
            raise ValueError('strict increasing knots required')
        previous = knot
    if knots[0] != ZERO or knots[-1] != ONE:
        raise ValueError('endpoint knots required')
    ct._form(system, readout)
    original_rows = set(system.le)
    output_order, outputs, phases = [], set(), set()
    for gate in gates:
        output = ct._gate(system, gate, binary, original_rows)
        if output in outputs or gate.bit in phases:
            raise ValueError('distinct original gates required')
        output_order.append(output)
        outputs.add(output)
        phases.add(gate.bit)
    if child is not None:
        child_output = ct._gate(system, child, binary, original_rows)
        if child_output in outputs or child.bit in phases:
            raise ValueError('distinct original child required')
        if child.g != readout:
            raise ValueError('child preactivation must equal the complete readout')

    first, last = gates[0], gates[-1]
    # Both envelopes use only rest, two original endpoint output columns, f/h,
    # and the residual supports.  Reserve all resulting LE entries BEFORE any
    # residual/envelope construction.  Biases count even for a zero row.
    residual_support = sum(len(gate.g.terms) + len(first.g.terms)
                           + len(last.g.terms) for gate in gates[1:-1])
    envelope_support = (len(readout.terms) + 2 + len(first.g.terms)
                        + len(last.g.terms) + residual_support)
    new_row_count = 2 + int(child is not None)
    new_nnz_upper = 2 * (len(readout.terms) + envelope_support)
    if child is not None:
        new_nnz_upper += 1 + envelope_support
    entry_upper = old_entries + new_row_count + 2 * new_nnz_upper
    if entry_upper > max_entries:
        raise ValueError('entry cap')

    readout_terms = dict(readout.terms)
    weights = tuple(readout_terms.get(column, ZERO) for column in output_order)
    rest = ct._linear(((ONE, readout),) + tuple(
        (-weight, gate.q) for weight, gate in zip(weights, gates)))
    kernel_lower, kernel_upper = ct.support(knots[1:-1], weights[1:-1])
    first_weight, last_weight = weights[0], weights[-1]
    for knot, weight in zip(knots[1:-1], weights[1:-1]):
        first_weight = ct._add(first_weight, ct._mul(weight, ct._add(ONE, -knot)))
        last_weight = ct._add(last_weight, ct._mul(weight, knot))
    affine = ct._linear(((first_weight, first.q), (last_weight, last.q)))
    amplitude = ct._linear(((F(2), first.q), (F(2), last.q),
                            (-ONE, first.g), (-ONE, last.g)))
    upper_parts = [(ONE, rest), (ONE, affine), (-kernel_lower, amplitude)]
    lower_parts = [(ONE, rest), (ONE, affine), (-kernel_upper, amplitude)]
    residual_boxes, weighted_boxes = [], []
    positive_secants, negative_secants = [], []
    for gate, knot, weight in zip(gates[1:-1], knots[1:-1], weights[1:-1]):
        residual = ct._linear(((ONE, gate.g),
                               (-ct._add(ONE, -knot), first.g),
                               (-knot, last.g)))
        residual_boxes.append(ct._box(system, residual))
        weighted = ct._linear(((weight, residual),))
        weighted_boxes.append(ct._box(system, weighted))
        positive = ct._secant(system, weighted)
        negative = ct._secant(system, ct._linear(((-ONE, weighted),)))
        positive_secants.append(positive)
        negative_secants.append(negative)
        upper_parts.append((ONE, positive))
        lower_parts.append((-ONE, negative))
    upper, lower = ct._linear(upper_parts), ct._linear(lower_parts)
    rows = (ct._linear(((ONE, readout), (-ONE, upper))),
            ct._linear(((ONE, lower), (-ONE, readout))))
    upper_box = ct._box(system, upper)
    child_upper, child_branch = None, None
    if child is not None:
        child_upper = ct._secant(system, upper)
        child_branch = ('nonpositive' if upper_box[1] <= ZERO else
                        'nonnegative' if upper_box[0] >= ZERO else 'crossing')
        rows += (ct._linear(((ONE, child.q), (-ONE, child_upper))),)
    result = replace(system, le=system.le + rows)
    if ef.entries(result) > min(max_entries, entry_upper):
        raise ValueError('entry cap')
    return result, ct._receipt(
        system, result, rows, readout=readout, weights=weights, rest=rest,
        kernel_lower=kernel_lower, kernel_upper=kernel_upper,
        affine=affine, amplitude=amplitude, lower=lower, upper=upper,
        residual_boxes=tuple(residual_boxes), weighted_boxes=tuple(weighted_boxes),
        positive_secants=tuple(positive_secants), negative_secants=tuple(negative_secants),
        upper_box=upper_box, child_upper=child_upper, child_branch=child_branch,
        entry_upper=entry_upper, gpu_qualified=False, solver_calls=0)
