"""Four fixed mathematical entries; never a model/verifier experiment."""
from fractions import Fraction as F
from itertools import combinations, product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d057_triangle_source_census_20260930 import window_relation as wr

ZERO, ONE, HALF = F(0), F(1), F(1, 2)


def _point(x):
    value = F(x)
    return value, value


def _control(count=3):
    coefficients = ((ONE, F(-1, 3)), (ONE, ONE), (F(-1, 3), ONE),
                    (F(7, 10), F(-1, 5)), (F(-2, 5), F(4, 5)))
    biases = (F(-3, 10), F(-9, 10), F(-3, 10), F(-1, 5), F(-1, 5))
    weights = tuple(tuple(_point(a) for a in row) for row in coefficients[:count])
    return weights, tuple(_point(b) for b in biases[:count]), ((ZERO, ONE),) * 2, (7, 2), (19, 3, 11, 29, 5)[:count]


def _call(data, budget=None):
    return wr.compile_window(*data, enabled=True, budget=budget)


def _plain(value):
    kind = type(value)
    assert kind in (dict, tuple, str, int, bool, F, type(None))
    if kind is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _plain(item)
    elif kind is tuple:
        for item in value:
            _plain(item)
    elif kind is F:
        assert max(abs(value.numerator).bit_length(), value.denominator.bit_length()) <= 512


def _hold(rows, assignments):
    return all(sum((a * assignments[token] for token, a in values + phases), ZERO) <= rhs
               for values, phases, rhs in rows)


def _direct(data, channels):
    """Independent fresh D056 call, not the bridge's composition helper."""
    weights, biases, bounds, ordinals, outputs = data
    ms, ss, pt = wr.ms, wr.ss, wr.pt
    frame = ms.make_frame('independent full D056 replay', enabled=True)
    roles, phase_roles, entries, source_values = {}, {}, [], []
    for ordinal, (lo, hi) in zip(ordinals, bounds):
        if ordinal is None:
            source_values.append(None)
        else:
            value = ms.make_value(frame, 'original source', enabled=True)
            source_values.append(value)
            roles[value] = ('source', ordinal)
            entries.append((value, ordinal, max(ZERO, lo), max(ZERO, hi)))
    context = ms.source_context(frame, tuple(entries), enabled=True)
    forms, qs, bits = [], [], []
    for channel in channels:
        q = ms.make_value(frame, 'original output', enabled=True)
        bit = ms.make_phase(q, outputs[channel], 'original phase', enabled=True)
        roles[q], phase_roles[bit] = ('output', outputs[channel]), ('phase', outputs[channel])
        terms = tuple((context._by_value[v], lo, hi) for v, (lo, hi) in
                      zip(source_values, weights[channel]) if v is not None)
        forms.append(ss.affine(context, biases[channel], terms, enabled=True))
        qs.append(q)
        bits.append(bit)
    try:
        result = pt.generate(tuple(forms), tuple(qs), tuple(bits), enabled=True)
        rows = tuple((tuple((roles[v], a) for v, a in values),
                      tuple((phase_roles[b], a) for b, a in phases), rhs)
                     for values, phases, rhs in result.rows)
        deltas = tuple(pair.delta for pair in result.pairs)
        return result.status, result.bounds, rows, deltas
    finally:
        frame._source_phases.clear()
        frame._phases.clear()
        frame._values.clear()


def test_complete_window_control():
    data = _control()
    budget = wr.WorkBudget(enabled=True)
    result = _call(data, budget)
    assert result['schema'] == 'd057_window_v1'
    assert result['ordinary_bounds'] == ((F(-19, 30), F(7, 10)),
                                          (F(-9, 10), F(11, 10)),
                                          (F(-19, 30), F(7, 10)))
    assert result['open_channels'] == (1, 2, 0)
    assert result['total_triangles'] == result['candidate_triangles'] == 1
    assert result['pair_proof_count'] == 3 and result['pair_row_count'] == 6
    assert result['odd_triangles'] == result['row_count'] == 1 and result['nnz'] == 8
    triangle = result['triangles'][0]
    assert triangle['pair_indices'] == (0, 1, 2)
    assert triangle['cube_midpoint_gap'] == F(1, 3)
    values, phases, rhs = triangle['rows'][0]
    assert dict(values) == {('source', 7): -ONE, ('source', 2): -ONE,
                            ('output', 19): F(2), ('output', 3): F(2), ('output', 11): F(2)}
    assert dict(phases) == {('phase', 19): F(-2, 5), ('phase', 3): F(-13, 15), ('phase', 11): F(-2, 5)}
    assert rhs == ZERO
    old = {('source', 7): HALF, ('source', 2): HALF,
           ('output', 19): F(1, 3), ('output', 3): F(2, 5), ('output', 11): F(1, 3),
           ('phase', 19): HALF, ('phase', 3): HALF, ('phase', 11): HALF}
    assert all(_hold(pair['rows'], old) for pair in result['pairs'])
    assert sum((a * old[token] for token, a in values + phases), ZERO) - rhs == F(3, 10)
    assert result['work_used'] == budget.used
    # Whole fixed four/five-gate populations, not just the old one-triangle case.
    for count in (4, 5):
        current = _control(count)
        complete = _call(current)
        expected = tuple(combinations(complete['open_channels'], 3))
        assert tuple(t['channels'] for t in complete['triangles']) == expected
        assert complete['pair_proof_count'] == count * (count - 1) // 2
        assert complete['candidate_triangles'] == count * (count - 1) * (count - 2) // 6
        for entry in complete['triangles']:
            status, bounds, rows, deltas = _direct(current, entry['channels'])
            assert (entry['status'], entry['bounds'], entry['rows']) == (status, bounds, rows)
            assert tuple(complete['pairs'][i]['delta'] for i in entry['pair_indices']) == deltas
    _plain(result)


def test_full_population_and_zero_phases():
    bounds, ordinals = ((-ONE, ONE), (ZERO, F(2)), (ZERO, ZERO)), (0, 9, None)
    padding = (F(-3), F(7))
    rows = ((_point(0), _point(0), padding),) * 3 + (
            (_point(1), _point(0), padding), (_point(-1), _point(0), padding))
    biases = tuple(_point(b) for b in (1, -1, 0, 0, 0))
    outputs = (100, 101, 102, 103, 104)
    result = _call((rows, biases, bounds, ordinals, outputs))
    assert result['activation_bounds'] == ((ZERO, ONE), (ZERO, F(2)), (ZERO, ZERO))
    assert result['canonical_slots'] == 3 and result['receiver_canonical_slots'] == 15
    assert result['real_slots'] == 2 and result['padding_slots'] == 1
    assert result['ordinary_bounds'] == ((_point(1)), (_point(-1)), (_point(0)), (ZERO, ONE), (-ONE, ZERO))
    assert result['channel_states'] == ('A', 'I', 'O', 'O', 'O')
    assert result['total_triangles'] == 10 and result['stable_skipped_triangles'] == 9
    assert result['candidate_triangles'] == 1 and result['no_odd_triangles'] == 1
    assert result['row_count'] == 0 and result['original_bits_deleted'] == 0
    all_stable = _call(((rows[0],) * 5, (_point(1),) * 5, bounds, ordinals, outputs))
    assert all_stable['stable_skipped_triangles'] == 10
    assert all_stable['candidate_triangles'] == 0 and all_stable['pairs'] == all_stable['triangles'] == ()
    zero_weights = tuple(tuple(_point(a) for a in row) for row in ((1, -1, 0), (0, 1, -1), (-1, 0, 1)))
    zero = _call((zero_weights, (_point(0),) * 3, ((ZERO, ONE),) * 3, (1, 2, 3), (10, 11, 12)))
    assert zero['channel_states'] == ('O',) * 3 and zero['odd_triangles'] == 1
    for bits in product((ZERO, ONE), repeat=3):
        assignments = {('source', i): ZERO for i in (1, 2, 3)}
        assignments.update({('output', i): ZERO for i in (10, 11, 12)})
        assignments.update((('phase', i), bit) for i, bit in zip((10, 11, 12), bits))
        assert all(_hold(t['rows'], assignments) for t in zero['triangles'])


def test_plain_identity_and_intervals():
    positive, negative, padding = (F(9, 10), F(11, 10)), (F(-2, 5), F(-1, 4)), (F(-3), F(4))
    weights = ((positive, negative, padding), (positive, positive, padding), (negative, positive, padding))
    biases = ((F(-31, 100), F(-29, 100)), (F(-91, 100), F(-89, 100)), (F(-31, 100), F(-29, 100)))
    bounds = ((-ONE, F(2)), (ZERO, ONE), (ZERO, ZERO))
    sources, outputs = (5, 17, None), (5, 17, 99)
    result = _call((weights, biases, bounds, sources, outputs))
    assert result['odd_triangles'] == 1
    _plain(result)
    row_roles = {token[0] for t in result['triangles'] for row in t['rows'] for token, _ in row[0]}
    assert row_roles == {'source', 'output'}
    for choice in (0, 1, 2):
        actual_weights = tuple(tuple((lo + hi) / 2 if choice == 2 else (lo, hi)[(choice + i + j) % 2]
                                      for j, (lo, hi) in enumerate(row)) for i, row in enumerate(weights))
        actual_biases = tuple((lo + hi) / 2 if choice == 2 else (lo, hi)[(choice + i) % 2]
                             for i, (lo, hi) in enumerate(biases))
        for first, second in product((ZERO, ONE, F(2)), (ZERO, HALF, ONE)):
            activation = (first, second, ZERO)
            pre = tuple(bias + sum((a * x for a, x in zip(row, activation)), ZERO)
                        for bias, row in zip(actual_biases, actual_weights))
            assignments = {('source', 5): first, ('source', 17): second}
            assignments.update((('output', ordinal), max(ZERO, value)) for ordinal, value in zip(outputs, pre))
            legal = tuple((ZERO, ONE) if value == ZERO else ((ONE,) if value > ZERO else (ZERO,)) for value in pre)
            for bits in product(*legal):
                assignments.update((('phase', ordinal), bit) for ordinal, bit in zip(outputs, bits))
                assert all(_hold(t['rows'], assignments) for t in result['triangles'])
    # A real zero source is not padding: its parameter terms remain in proof.
    real_zero = _call((weights, biases, bounds, (5, 17, 23), outputs))
    assert real_zero['ordinary_bounds'] == result['ordinary_bounds']
    assert real_zero['real_slots'] == 3 and real_zero['padding_slots'] == 0
    assert any(ordinal == 23 for p in real_zero['pairs'] for ordinal, _, _ in p['f']['terms'])
    assert tuple(t['rows'] for t in real_zero['triangles']) == tuple(t['rows'] for t in result['triangles'])
    order = (2, 0, 1)
    permuted = _call((tuple(weights[i] for i in order), tuple(biases[i] for i in order),
                      bounds, sources, tuple(outputs[i] for i in order)))
    assert tuple(t['rows'] for t in permuted['triangles']) == tuple(t['rows'] for t in result['triangles'])
    assert result['transient_numeric_entries'] == wr.transient_entries(3, 3, 3)


def test_fail_closed_budget_and_default_off():
    assert wr.compile_window(object(), object(), object(), object(), object()) is None
    data = _control()
    with pytest.raises(wr.KernelError):
        wr.compile_window(*data, enabled=1)
    with pytest.raises(wr.BudgetExceeded):
        _call(data, wr.WorkBudget(enabled=True, limit=0))
    with pytest.raises(wr.KernelDisabled):
        _call(data, wr.WorkBudget())
    with pytest.raises(wr.KernelError):
        _call(data, wr.WorkBudget(enabled=True, max_bits=64))
    weights, biases, bounds, sources, outputs = data
    invalid = ((weights, biases, bounds, (7, 7), outputs),
               (weights, biases, bounds, sources, (19, 19, 11)),
               (weights, biases, bounds, (None, 2), outputs),
               ((weights[0][:-1], *weights[1:]), biases, bounds, sources, outputs),
               (weights, biases, bounds, (1 << 512, 2), outputs),
               (weights, biases, bounds, sources, list(outputs)))
    for current in invalid:
        with pytest.raises(wr.KernelError):
            _call(current)
    bad_weights = (((1.0, 1.0), weights[0][1]), *weights[1:])
    with pytest.raises(wr.KernelError):
        _call((bad_weights, biases, bounds, sources, outputs))
    enormous = (((F(1 << 512), F(1 << 512)), weights[0][1]), *weights[1:])
    with pytest.raises(wr.KernelError):
        _call((enormous, biases, bounds, sources, outputs))
    arithmetic_overflow = (((F(1 << 511), F(1 << 511)), weights[0][1]), *weights[1:])
    with pytest.raises(wr.KernelError):
        _call((arithmetic_overflow, biases, ((ZERO, F(2)), bounds[1]), sources, outputs))
    malformed = wr.WorkBudget(enabled=True)
    malformed.used = malformed.limit + 1
    with pytest.raises(wr.KernelError):
        _call(data, malformed)
    with pytest.raises(wr.KernelError):
        wr.transient_entries(2, 3, 4)
