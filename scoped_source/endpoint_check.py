"""Exact endpoint aggregation GIVEN guarded LPs, output forms and gate ranges.

No claim that those input LPs enclose a network, or that the supplied ranges
enclose its gates. Expected request identity must be bound outside the proof.
No constructor/solver import; this is not the production/hard-budget entry.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations

from scoped_source.graph import clock
from scoped_source.sparse_check import check_bound
from source_enclosure.format import identity
from upstream_source.checker import csr

THRESHOLD = F(1, 10_000_000)


def exact(value):
    if type(value) is not str:
        raise ValueError('canonical rational strings required')
    try:
        result = F(value)
    except (ValueError, ZeroDivisionError):
        raise ValueError('finite rational required') from None
    if str(result) != value:
        raise ValueError('canonical rational required')
    return result


def validate(request, expected, tick):
    tick()
    if (set(request) != {'schema', 'experts', 'classes', 'label', 'duties'} or
            request['schema'] != 'GIVEN_LP_TOP2_ENDPOINT_REQUEST_V1' or identity(request) != expected):
        raise ValueError('external request binding')
    e, c, y = (request[k] for k in ('experts', 'classes', 'label'))
    if (type(e) is not int or not 2 <= e <= 64 or type(c) is not int or c < 2 or
            type(y) is not int or not 0 <= y < c):
        raise ValueError('request dimensions')
    required = [(list(p), k) for p in combinations(range(e), 2) for k in range(c) if k != y]
    duties = request['duties']
    if len(duties) != len(required):
        raise ValueError('all unordered pair and property duties required')
    for duty, key in zip(duties, required):
        tick()
        if (set(duty) != {'pair', 'competitor', 'variables', 'base', 'a', 'b', 'gate'} or
                (duty['pair'], duty['competitor']) != key or
                type(duty['competitor']) is not int or
                any(type(v) is not int for v in duty['pair'])):
            raise ValueError('duty identity/order')
        names, lp = duty['variables'], duty['base']
        if (not names or any(type(v) is not str or not v for v in names) or
                len(set(names)) != len(names)):
            raise ValueError('variable identity/alias')
        if set(lp) != {'matrix_format', 'offset', 'c', 'lower', 'upper', 'A', 'b', 'E', 'h'}:
            raise ValueError('base LP fields')
        n = len(names)
        if (lp['matrix_format'] != 'csr_v1' or len(lp['c']) != n or
                any(exact(v) != 0 for v in lp['c']) or exact(lp['offset']) != 0 or
                len(lp['lower']) != n or len(lp['upper']) != n):
            raise ValueError('base LP dimensions/objective')
        if any(exact(a) > exact(b) for a, b in zip(lp['lower'], lp['upper'])):
            raise ValueError('finite ordered variable box')
        for matrix, rhs in [('A', 'b'), ('E', 'h')]:
            csr(lp[matrix], [len(lp[rhs]), n])
            for v in lp[matrix]['data'] + lp[rhs]:
                exact(v)
        for side in ('a', 'b'):
            form = duty[side]
            if set(form) != {'c', 'offset'} or len(form['c']) != n:
                raise ValueError('affine property dimensions')
            for v in form['c'] + [form['offset']]:
                exact(v)
        if len(duty['gate']) != 2 or not 0 <= exact(duty['gate'][0]) <= exact(duty['gate'][1]) <= 1:
            raise ValueError('normalized gate interval')
    tick()
    return duties


def check(request, proof, *, expected_request_sha256, deadline):
    tick = clock(deadline)
    duties = validate(request, expected_request_sha256, tick)
    if (set(proof) != {'schema', 'request_sha256', 'duties'} or
            proof['schema'] != 'GIVEN_LP_ENDPOINT_PROOF_V1' or
            proof['request_sha256'] != expected_request_sha256 or
            len(proof['duties']) != len(duties)):
        raise ValueError('proof binding or coverage')
    results = []; checked = 0; missing = 0
    for duty, record in zip(duties, proof['duties']):
        tick()
        if (set(record) != {'pair', 'competitor', 'endpoints'} or
                record['pair'] != duty['pair'] or record['competitor'] != duty['competitor'] or
                type(record['competitor']) is not int or
                any(type(v) is not int for v in record['pair'])):
            raise ValueError('proof duty binding')
        # The known gate interval, not the received proof, determines coverage.
        weights = sorted(set(map(exact, duty['gate'])))
        if len(record['endpoints']) != len(weights):
            raise ValueError('missing/extra endpoint')
        bounds = []
        for weight, endpoint in zip(weights, record['endpoints']):
            tick()
            if (set(endpoint) != {'weight', 'lp_sha256', 'certificate'} or
                    exact(endpoint['weight']) != weight):
                raise ValueError('endpoint identity')
            lp = deepcopy(duty['base'])
            # Independent convex combination, not the producer's u + t*d code.
            lp['c'] = [str(weight*exact(a)+(1-weight)*exact(b))
                       for a, b in zip(duty['a']['c'], duty['b']['c'])]
            lp['offset'] = str(weight*exact(duty['a']['offset']) +
                               (1-weight)*exact(duty['b']['offset']))
            if endpoint['lp_sha256'] != identity(lp):
                raise ValueError('endpoint LP identity')
            certificate = endpoint['certificate']
            if certificate is None:
                missing += 1; bounds.append(None)
            else:
                value = F(check_bound(lp, certificate)['checked_lower_bound'])
                tick(); bounds.append(value); checked += 1
        lower = None if None in bounds else min(bounds)
        results.append({'pair': duty['pair'], 'competitor': duty['competitor'],
                        'endpoint_bounds': [None if v is None else str(v) for v in bounds],
                        'lower_bound': None if lower is None else str(lower),
                        'positive': lower is not None and lower > THRESHOLD})
    tick()
    positive = sum(r['positive'] for r in results)
    return {'status': 'CHECKED_POSITIVE_GIVEN_BASE_AND_GATE' if positive == len(duties)
            else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE',
            'request_sha256': expected_request_sha256, 'required': len(duties),
            'positive': positive, 'endpoint_bounds_checked': checked, 'missing': missing,
            'duties': results, 'source_complete': False, 'deployed_float_SAFE': False,
            'remaining_trust': ['given_guarded_LP_contains_declared_network',
                                'given_affine_properties_correspond_to_experts',
                                'given_gate_intervals_cover_selected_softmax',
                                'parser_and_exact_checker_implementation'],
            'hard_budget_supervision': False}
