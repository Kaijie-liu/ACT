"""Exact, solver-free check of a bounded sigmoid range proof.

This proves a transcendental enclosure given an independently justified margin
interval. It does NOT justify the supplied network/route margin bounds.
"""
from fractions import Fraction as F
import re

SETTINGS = {'input_bits': 24, 'work_bits': 40, 'output_bits': 24,
            'degree': 16, 'max_abs_margin': 16}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    require(isinstance(value, str) and len(value) <= 2500, 'rational admission')
    require(re.fullmatch(r'-?\d+(?:/[1-9]\d*)?', value) is not None, 'rational syntax')
    # Individual tokens are bounded before integer allocation.
    require(all(len(t.lstrip('-')) <= 1234 for t in value.split('/')), 'rational token cap')
    q = F(value)
    require(max(q.numerator.bit_length(), q.denominator.bit_length()) <= 4096,
            'rational bit cap')
    require(str(q) == value, 'noncanonical rational')
    return q


def identity_context(context):
    require(isinstance(context, dict) and set(context) == {
        'request_id', 'ordered_pair', 'margin_lower_proof', 'margin_negative_upper_proof'},
        'scope fields')
    for key in ('request_id', 'margin_lower_proof', 'margin_negative_upper_proof'):
        require(isinstance(context[key], str) and re.fullmatch('[0-9a-f]{64}', context[key]),
                'scope hash')
    p = context['ordered_pair']
    require(isinstance(p, list) and len(p) == 2 and
            all(type(v) is int and v >= 0 for v in p) and p[0] < p[1], 'ordered pair')


def check(proof, *, expected_context, expected_margin):
    identity_context(expected_context)
    require(set(proof) == {'schema', 'settings', 'context', 'margin', 'endpoints', 'gate'},
            'proof fields')
    require(proof['schema'] == 'RATIONAL_SIGMOID_RANGE_V1', 'schema')
    require(proof['settings'] == SETTINGS and all(type(v) is int for v in proof['settings'].values()),
            'settings')
    identity_context(proof['context'])
    require(proof['context'] == expected_context, 'context mismatch')
    require(proof['margin'] == expected_margin and len(expected_margin) == 2, 'margin binding')
    lower, upper = map(rational, expected_margin)
    require(-16 <= lower <= upper <= 16, 'supported margin range')
    require(isinstance(proof['endpoints'], list) and len(proof['endpoints']) == 2, 'endpoints')
    verified = []
    for side, (original, item) in enumerate(zip((lower, upper), proof['endpoints'])):
        require(set(item) == {'rounded', 'halvings', 'series', 'tail', 'exp_steps', 'gate'},
                'endpoint fields')
        rounded = rational(item['rounded'])
        unit = F(1, 2**24)
        require((rounded <= original < rounded + unit) if side == 0 else
                (rounded - unit < original <= rounded), 'input rounded inward')
        require((rounded * 2**24).denominator == 1, 'input grid')
        k = item['halvings']
        require(type(k) is int and 0 <= k <= 5, 'halving cap')
        t = abs(rounded) / 2**k
        require(t <= F(1, 2), 'Taylor domain')
        # Positive series plus a geometric majorant for the positive tail.
        # Subsequent term ratios are <= t/18 after the first omitted term.
        term, total = F(1), F(1)
        for j in range(1, 17):
            term = term * t / j
            total += term
        remainder = (term * t / 17) / (1 - t / 18)
        require(rational(item['series']) == total and rational(item['tail']) == remainder,
                'invalid Taylor evidence')
        intervals = item['exp_steps']
        require(isinstance(intervals, list) and len(intervals) == k + 1, 'exp steps')
        prev_lo, prev_hi = total, total + remainder
        for step, interval in enumerate(intervals):
            require(isinstance(interval, list) and len(interval) == 2, 'exp interval')
            lo, hi = map(rational, interval)
            require(0 < lo <= prev_lo <= prev_hi <= hi, 'exp enclosure inward')
            require(all((x*2**40).denominator == 1 for x in (lo, hi)), 'work grid')
            require(prev_lo-lo < F(1, 2**40) and hi-prev_hi < F(1, 2**40),
                    'noncanonical outward rounding')
            if step < k:
                prev_lo, prev_hi = lo*lo, hi*hi
        if rounded >= 0:
            exact_lo, exact_hi = lo/(1+lo), hi/(1+hi)
        else:
            exact_lo, exact_hi = 1/(1+hi), 1/(1+lo)
        require(isinstance(item['gate'], list) and len(item['gate']) == 2, 'endpoint gate')
        gl, gu = map(rational, item['gate'])
        require(0 <= gl <= exact_lo <= exact_hi <= gu <= 1, 'sigmoid enclosure inward')
        require(all((x*2**24).denominator == 1 for x in (gl, gu)), 'gate grid')
        require(exact_lo-gl < unit and gu-exact_hi < unit, 'gate rounding')
        verified.append((gl, gu))
    result = [str(verified[0][0]), str(verified[1][1])]
    require(proof['gate'] == result, 'final gate mismatch')
    return {'status': 'CHECKED_SIGMOID_ENCLOSURE_GIVEN_MARGIN', 'gate': result,
            'scope': 'Supplied checked margin only; NOT complete MoE or network-lowering proof.'}
