"""Request-level contract for conditional weighted top-2 evidence, version 1."""
import itertools
from fractions import Fraction
from act.back_end.solver.lp_certificate import identity, rational

TRUSTED = ['network_input_to_HZ_and_ordered_expert_sources',
           'membership_and_pair_guard_lowering', 'route_infeasibility_exclusions']
THRESHOLD = 1e-7


def validate_request(r):
    if r.get('schema') != 'WEIGHTED_TOP2_REQUEST_V1' or r.get('top_k') != 2 or r.get('tie_policy') != 'ANY_LEGAL_TOPK':
        raise ValueError('unsupported request semantics')
    if r.get('gate') != 'selected_softmax' or r.get('mode') != 'eval':
        raise ValueError('unsupported gate/BN mode')
    if any(type(r.get(k)) is not int or r[k] < 2 for k in ('experts', 'classes')):
        raise ValueError('invalid dimensions')
    if type(r.get('clean_prediction')) is not int or not 0 <= r['clean_prediction'] < r['classes']:
        raise ValueError('invalid clean class')
    if 'epsilon' not in r or rational(r['epsilon'])<0:
        raise ValueError('invalid radius')
    if not r.get('properties') or not isinstance(r['properties'], list):
        raise ValueError('nonempty property list required')
    for p in r['properties']:
        if set(p) != {'q', 'constant'} or not isinstance(p['q'], list) or len(p['q']) != r['classes']:
            raise ValueError('invalid property shape')
        for v in p['q'] + [p['constant']]: rational(v)
    for k in ('model_state','center','lower','upper'):
        if not r.get(k): raise ValueError('missing model/input identity')
    return identity(r)


def classification_properties(classes, label):
    return [{'q': [int(i == label)-int(i == c) for i in range(classes)], 'constant': 0}
            for c in range(classes) if c != label]


def route_pairs(r, routes):
    if type(routes['exact']) is not bool:
        raise ValueError('route completeness must be boolean')
    expected = list(itertools.combinations(range(r['experts']), 2))
    groups = []
    for kind in ('feasible', 'infeasible', 'unresolved'):
        pairs = routes[kind]
        if not isinstance(pairs, list): raise ValueError('route list required')
        for pair in pairs:
            if len(pair) != 2 or any(type(v) is not int for v in pair): raise ValueError('invalid pair')
        groups.extend(tuple(p) for p in pairs)
    if sorted(groups) != expected: raise ValueError('route partition missing/duplicate/noncanonical')
    return sorted(tuple(p) for p in routes['feasible'])


def gate_envelope(lower, negative_upper):
    if lower is not None and negative_upper is not None and lower > -negative_upper:
        raise ValueError('inconsistent checked order bounds')
    return [str(Fraction(1,2) if lower is not None and lower >= 0 else Fraction(0)),
            str(Fraction(1,2) if negative_upper is not None and negative_upper >= 0 else Fraction(1))]


def interval_lp(bounds, prop, classes):
    lo, hi = bounds['lower'], bounds['upper']
    if len(lo) != classes or len(hi) != classes or any(rational(a)>rational(b) for a,b in zip(lo,hi)):
        raise ValueError('invalid expert interval')
    return {'c': prop['q'], 'offset': prop['constant'], 'lower': lo, 'upper': hi}


def interval_certificate(lp):
    value = rational(lp['offset']) + sum((min(rational(q)*rational(a), rational(q)*rational(b))
                                        for q,a,b in zip(lp['c'],lp['lower'],lp['upper'])), Fraction(0))
    return {'lp_sha256': identity(lp), 'inequality_dual': [], 'equality_dual': [],
            'claimed_lower_bound': str(value)}


def pair_key(pair):
    return f's{pair[0]}_{pair[1]}'
