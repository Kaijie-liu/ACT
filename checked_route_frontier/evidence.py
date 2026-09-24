"""Exact all-original-obligation aggregation; no denominator shrink or solver trust."""
from fractions import Fraction as F

from checked_route_frontier.check import check
from scoped_proof.evidence import POSITIVE, bind_source, context, lower_bound, reconstruct_lp, roster
from scoped_source.graph import clock
from source_enclosure.format import identity


def aggregate(scope, doc, bundle, candidates, *, invocation, proposal_complete, deadline):
    tick = clock(deadline)
    if type(proposal_complete) is not bool or not isinstance(invocation, str) or not invocation:
        raise ValueError('proposal/run identity')
    doc_hash = bind_source(doc, scope)
    construction = check(doc, bundle, expected_source_sha256=doc_hash, deadline=deadline)
    expected = roster(scope)
    frontier = {tuple(row['pair']): row for row in construction['frontier']['pairs']}
    pairs = {tuple(row['pair']): row for row in bundle['pairs']}
    required_indices = {i for i, row in enumerate(expected) if tuple(row['pair']) in pairs}
    if type(candidates) is not dict or any(type(i) is not int or i not in required_indices for i in candidates):
        raise ValueError('unexpected or excluded output candidate index')
    bundle_hash = identity(bundle)
    rows, missing, nonpositive = [], [], []
    for index, obligation in enumerate(expected):
        tick()
        pair = tuple(obligation['pair'])
        base_row = {'index': index, **obligation}
        if frontier[pair]['status'] != 'RETAINED':
            rows.append({**base_row, 'status': 'DISCHARGED_BY_CHECKED_ROUTE_EXCLUSION',
                         'exclusion': frontier[pair]})
            continue
        entry = pairs[pair]
        row = next(v for v in entry['obligations']['rows'] if v['competitor'] == obligation['competitor'])
        lp = reconstruct_lp(entry['base'], row)
        candidate = candidates.get(index)
        if candidate is None:
            missing.append(index)
            rows.append({**base_row, 'status': 'MISSING'})
            continue
        want = context(scope, doc_hash, bundle_hash, invocation, index, obligation, lp)
        if candidate['schema'] != 'SCOPED_LP_CANDIDATE_V1' or candidate['context'] != want:
            raise ValueError('output candidate source/property/run binding')
        certificate = candidate['certificate']
        if certificate is None:
            if candidate['status'] != 'NO_CANDIDATE':
                raise ValueError('candidate status')
            missing.append(index)
            rows.append({**base_row, 'status': 'NO_CANDIDATE'})
            continue
        if candidate['status'] != 'CANDIDATE':
            raise ValueError('candidate status')
        bound = lower_bound(lp, certificate)
        positive = F(bound['checked_lower_bound']) > 0
        if not positive:
            nonpositive.append(index)
        rows.append({**base_row, 'status': 'POSITIVE_BOUND' if positive else 'NONPOSITIVE_BOUND', **bound})
    tick()
    closed = proposal_complete and not missing and not nonpositive
    return {'schema': 'CHECKED_ROUTE_REQUEST_V1', 'status': POSITIVE if closed else 'NOT_CLOSED',
            'source_sha256': doc_hash, 'bundle_sha256': bundle_hash, 'request_sha256': identity(scope),
            'invocation': invocation, 'required': len(expected), 'rows': rows,
            'discharged_by_exclusion': len(expected) - len(required_indices),
            'required_output_bounds': len(required_indices),
            'checked_bounds': len(required_indices) - len(missing),
            'positive_bounds': len(required_indices) - len(missing) - len(nonpositive),
            'missing': missing, 'nonpositive': nonpositive, 'proposal_complete': proposal_complete,
            'complete_output_positive_proof': closed, 'source_check': construction,
            'route_changing_established': False, 'native_float_proof': False,
            'trusted': construction['trusted'],
            'scope': 'Declared real top2 graph; all pairs excluded by checked strict margin or all outputs checked.'}
