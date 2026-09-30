"""Untrusted endpoint LP builder. No source/gate validity or timing claim."""
from copy import deepcopy
from scoped_source.graph import clock
from scoped_source.endpoint_check import validate, exact
from source_enclosure.format import identity


def build(request, *, expected_request_sha256, deadline, proposer=None):
    tick = clock(deadline)
    duties = validate(request, expected_request_sha256, tick)
    records = []
    for duty in duties:
        tick(); endpoints = []
        for weight in sorted(set(map(exact, duty['gate']))):
            tick(); lp = deepcopy(duty['base'])
            lp['c'] = [str(exact(b)+weight*(exact(a)-exact(b)))
                       for a, b in zip(duty['a']['c'], duty['b']['c'])]
            lp['offset'] = str(exact(duty['b']['offset']) + weight *
                               (exact(duty['a']['offset'])-exact(duty['b']['offset'])))
            certificate = None if proposer is None else proposer(deepcopy(duty), weight, deepcopy(lp))
            tick()
            endpoints.append({'weight': str(weight), 'lp_sha256': identity(lp), 'certificate': certificate})
        records.append({'pair': duty['pair'][:], 'competitor': duty['competitor'], 'endpoints': endpoints})
    tick()
    return {'schema': 'GIVEN_LP_ENDPOINT_PROOF_V1', 'request_sha256': expected_request_sha256,
            'duties': records}
