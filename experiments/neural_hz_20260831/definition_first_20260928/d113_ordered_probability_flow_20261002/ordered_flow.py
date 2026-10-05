"""Opt-in source-ordered Softmax flow lowering; no solver or model calls.

Actual p/q Softmax and common-value bindings remain caller proof obligations.
The coefficient source order is certified from the retained common frame's
box, not from a sample, an LP state, or the order of output value weights.
"""
from dataclasses import replace
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef


def append_ordered_flow(system, s, t, p, q, values, yp, yq, prob_bounds,
                        *, frames, enabled=False, max_entries=64_000_000):
    if enabled is False:
        return system, {'enabled': False}
    if enabled is not True or type(max_entries) is not int or not 0 < max_entries <= 64_000_000:
        raise ValueError('invalid opt-in or entry cap')
    if type(frames) is not tuple or len(frames) != 7 or any(type(f) is not int or f != system.frame for f in frames):
        raise ValueError('source/frame mismatch')
    n, channels = len(s), len(yp)
    if (n < 2 or any(len(rows) != n for rows in (t, p, q, values))
            or not channels or len(yq) != channels or any(len(row) != channels for row in values)
            or len(prob_bounds) != 2 or any(len(row) != n for row in prob_bounds)):
        raise ValueError('flow shape mismatch')
    for form in (*s, *t, *p, *q, *yp, *yq, *(v for row in values for v in row)):
        ef.check_form(system, form)
    for bounds in prob_bounds:
        for lo, hi in bounds:
            if not 0 < ef.rational(lo) <= ef.rational(hi) <= 1:
                raise ValueError('positive probability bounds required')
    extra_columns = (n-1)*(1+channels)
    extra_rows = 2 + (n-1) + channels + 4*n + 8*(n-1)*channels
    entry_upper = (ef.entries(system) + 2*extra_columns
                   + extra_rows*(1+2*(len(system.bounds)+extra_columns)))
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    d = tuple(a-b for a, b in zip(s, t))
    intervals = tuple(ef.box(system, row) for row in d)
    order = tuple(sorted(range(n), key=lambda i: (sum(intervals[i]), i)))
    order_upper = tuple(ef.box(system, d[i]-d[j])[1]
                        for i, j in zip(order, order[1:]))
    if any(bound > 0 for bound in order_upper):
        raise ValueError('source-wide score-change order is not certified')
    original = system
    system = replace(system, eq=system.eq + (sum(p, ef.Form())-1, sum(q, ef.Form())-1))
    for i in range(n):
        pl, pu = prob_bounds[0][i]
        ql, qu = prob_bounds[1][i]
        system = replace(system, le=system.le + (pl-p[i], p[i]-pu, ql-q[i], q[i]-qu))
    prefix_upper, tail_upper = F(0), sum((prob_bounds[0][i][1]-prob_bounds[1][i][0] for i in order), F(0))
    previous, flows, flow_indices = ef.Form(), [], []
    for i in order[:-1]:
        prefix_upper = ef.rational(prefix_upper + prob_bounds[1][i][1]-prob_bounds[0][i][0])
        tail_upper = ef.rational(tail_upper - prob_bounds[0][i][1]+prob_bounds[1][i][0])
        upper = min(F(1), prefix_upper, tail_upper)
        if upper < 0:
            raise ValueError('inconsistent certified flow bounds')
        flow_indices.append(len(system.bounds))
        system, flow = ef.variable(system, F(0), upper)
        system = replace(system, eq=system.eq + (flow-previous+p[i]-q[i],))
        previous = flow
        flows.append(flow)
    for channel in range(channels):
        total = ef.Form()
        for flow, left, right in zip(flows, order, order[1:]):
            system, term = ef.product(system, flow, values[right][channel]-values[left][channel])
            total = total + term
        system = replace(system, eq=system.eq + (yp[channel]-yq[channel]-total,))
    if ef.entries(system) > max_entries:
        raise ValueError('entry cap')
    assert system.bounds[:len(original.bounds)] == original.bounds
    assert system.binary == original.binary
    assert system.eq[:len(original.eq)] == original.eq
    assert system.le[:len(original.le)] == original.le
    return system, dict(enabled=True, order=order,
        order_upper_bounds=tuple(str(x) for x in order_upper), flow_indices=tuple(flow_indices),
        old_columns=len(original.bounds), new_columns=len(system.bounds)-len(original.bounds),
        new_eq=len(system.eq)-len(original.eq), new_le=len(system.le)-len(original.le),
        entries=ef.entries(system), exact_rational_rows=True,
        native_binding_qualified=False, complete_physical_qualification=False)
