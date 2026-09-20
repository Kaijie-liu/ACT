"""Deterministic bounded proof producer; independent checker is separate."""
from fractions import Fraction as F
from checked_gate.checker import SETTINGS, identity_context, rational


def down(q, bits):
    return F((q * (1 << bits)).numerator // (q * (1 << bits)).denominator, 1 << bits)


def up(q, bits):
    return -down(-q, bits)


def endpoint(x, side):
    x = down(x, 24) if side == 0 else up(x, 24)
    k = 0
    while abs(x) / (1 << k) > F(1, 2):
        k += 1
    t = abs(x) / (1 << k)
    terms = [F(1)]
    for j in range(1, 18):
        terms.append(terms[-1] * t / j)
    total = sum(terms[:17], F(0))
    tail = terms[17] * 18 / (18 - t)
    lo, hi = down(total, 40), up(total+tail, 40)
    steps = [[str(lo), str(hi)]]
    for _ in range(k):
        lo, hi = down(lo*lo, 40), up(hi*hi, 40)
        steps.append([str(lo), str(hi)])
    gate = (lo/(1+lo), hi/(1+hi)) if x >= 0 else (1/(1+hi), 1/(1+lo))
    return {'rounded': str(x), 'halvings': k, 'series': str(total), 'tail': str(tail),
            'exp_steps': steps, 'gate': [str(down(gate[0], 24)), str(up(gate[1], 24))]}


def propose(context, margin):
    identity_context(context)
    if not isinstance(margin, list) or len(margin) != 2:
        raise ValueError('margin shape')
    lo, hi = map(rational, margin)
    if not -16 <= lo <= hi <= 16:
        raise ValueError('unsupported margin; no fallback or range clipping')
    ends = [endpoint(lo, 0), endpoint(hi, 1)]
    return {'schema': 'RATIONAL_SIGMOID_RANGE_V1', 'settings': dict(SETTINGS),
            'context': {**context, 'ordered_pair': list(context['ordered_pair'])},
            'margin': list(margin), 'endpoints': ends,
            'gate': [ends[0]['gate'][0], ends[1]['gate'][1]]}
