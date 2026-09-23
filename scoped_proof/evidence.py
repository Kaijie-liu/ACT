"""Exact candidate acceptance and full obligation aggregation, stdlib only.

No solver status licenses acceptance. Network/guard/LP construction is checked
separately on the same new source; missing/nonpositive bounds remain NOT_CLOSED.
"""
from fractions import Fraction as F
from itertools import combinations
from source_enclosure.format import identity, sparse
from upstream_source.checker import csr, rational
from scoped_source.check import check as check_source
from scoped_proof.io import tick


POSITIVE = 'CHECKED_DECLARED_REAL_GRAPH_REQUEST'


def roster(scope):
    e, c, y = scope['experts'], scope['classes'], scope['label']
    if type(e) is not int or not 2 <= e <= 64 or type(c) is not int or c < 2 or type(y) is not int or not 0 <= y < c:
        raise ValueError('request dimensions')
    return [{'pair': list(p), 'label': y, 'competitor': k}
        for p in combinations(range(e), 2) for k in range(c) if k != y]


def bind_source(doc, scope):
    expected = {k: scope[k] for k in ('experts','classes','label','radius','margin','clip','center','model_state')}
    expected.update(top_k=2, gate='SELECTED_SOFTMAX', tie_policy='ANY_LEGAL_TOPK', training=False)
    if doc['request'] != expected: raise ValueError('source not bound to externally frozen request')
    return identity(doc)


def reconstruct_lp(base, row):
    """Independent materialization from checked shared base/obligation, not producer adapter."""
    n = base['variables']
    objective = csr(row['objective'], [1, n])[0]
    inequalities = csr(base['A'], [len(base['b']), n]) + csr(row['A_extra'], [4, n])
    return {'matrix_format': 'csr_v1', 'c': [str(objective.get(j, F(0))) for j in range(n)],
        'offset': row['offset'], 'A': sparse(inequalities, n), 'b': base['b']+row['b_extra'],
        'E': sparse(csr(base['E'], [len(base['h']), n]), n), 'h': base['h'],
        'lower': [-1]*(n-2)+[v[0] for v in row['extra_bounds']],
        'upper': [1]*(n-2)+[v[1] for v in row['extra_bounds']]}


def lower_bound(lp, certificate):
    if certificate['lp_sha256'] != identity(lp): raise ValueError('new LP identity mismatch')
    c, lo, hi = ([rational(x) for x in lp[k]] for k in ('c','lower','upper'))
    if not c or len(lo) != len(c) or len(hi) != len(c) or any(a > b for a,b in zip(lo,hi)):
        raise ValueError('finite LP variable bounds')
    residual = c[:]; bound = rational(lp['offset'])
    for matrix, rhs, name, signed in [('A','b','inequality_dual',True), ('E','h','equality_dual',False)]:
        rows = csr(lp[matrix], [len(lp[rhs]), len(c)])
        dual = list(map(rational, certificate[name]))
        if len(dual) != len(rows) or (signed and any(v > 0 for v in dual)):
            raise ValueError('invalid dual length/sign')
        for row, b, d in zip(rows, lp[rhs], dual):
            bound += rational(b)*d
            for j,v in row.items(): residual[j] -= v*d
    correction = sum((v*(a if v >= 0 else b) for v,a,b in zip(residual,lo,hi)), F(0))
    bound += correction
    return {'checked_lower_bound': str(bound), 'residual_box_term': str(correction),
        'nonzero_residual_coordinates': sum(v != 0 for v in residual),
        'lp_sha256': identity(lp)}


def context(scope, doc_hash, bundle_hash, invocation, index, obligation, lp):
    return {'request_sha256': identity(scope), 'source_sha256': doc_hash, 'bundle_sha256': bundle_hash,
        'invocation': invocation, 'index': index, **obligation, 'lp_sha256': identity(lp)}


def aggregate(scope, doc, bundle, candidates, *, invocation, proposal_complete, deadline):
    if type(proposal_complete) is not bool: raise ValueError('boolean proposal completion required')
    tick(deadline); doc_hash = bind_source(doc, scope)
    source = check_source(doc, bundle, expected_source_sha256=doc_hash, deadline=deadline)
    expected = roster(scope); bundle_hash = identity(bundle)
    if any(type(k) is not int or not 0 <= k < len(expected) for k in candidates):
        raise ValueError('unexpected candidate/duplicate index')
    by_pair = {tuple(p['pair']): p for p in bundle['pairs']}
    results = []; missing = []; nonpositive = []
    for i, obligation in enumerate(expected):
        tick(deadline); entry = by_pair[tuple(obligation['pair'])]
        row = next(r for r in entry['obligations']['rows'] if r['competitor'] == obligation['competitor'])
        lp = reconstruct_lp(entry['base'], row)
        candidate = candidates.get(i)
        if candidate is None:
            missing.append(i); results.append({'index': i, **obligation, 'status': 'MISSING'}); continue
        want = context(scope, doc_hash, bundle_hash, invocation, i, obligation, lp)
        if candidate['schema'] != 'SCOPED_LP_CANDIDATE_V1' or candidate['context'] != want:
            raise ValueError('candidate wrong request/source/property/run binding')
        cert = candidate['certificate']
        if cert is None:
            if candidate['status'] != 'NO_CANDIDATE': raise ValueError('missing claimed candidate')
            missing.append(i); results.append({'index': i, **obligation, 'status': 'NO_CANDIDATE'}); continue
        if candidate['status'] != 'CANDIDATE': raise ValueError('candidate status')
        bound = lower_bound(lp, cert); positive = F(bound['checked_lower_bound']) > 0
        if not positive: nonpositive.append(i)
        results.append({'index': i, **obligation, 'status': 'POSITIVE_BOUND' if positive else 'NONPOSITIVE_BOUND', **bound})
    tick(deadline)
    closed = proposal_complete and not missing and not nonpositive
    return {'schema': 'SCOPED_REQUEST_CHECK_V1', 'status': POSITIVE if closed else 'NOT_CLOSED',
        'request_sha256': identity(scope), 'source_sha256': doc_hash, 'bundle_sha256': bundle_hash,
        'invocation': invocation, 'required': len(expected), 'checked_bounds': len(expected)-len(missing),
        'positive_bounds': len(expected)-len(missing)-len(nonpositive), 'missing': missing,
        'nonpositive': nonpositive, 'proposal_complete': proposal_complete, 'rows': results,
        'source_check': source, 'complete_output_positive_proof': closed, 'native_float_proof': False,
        'route_changing_established': False,
        'trusted': ['declared graph/program correspondence', 'stored-center preprocessing', 'checker implementation/runtime'],
        'scope': 'new declared real graph and all guarded output LP bounds; no solver optimality or native-float claim'}
