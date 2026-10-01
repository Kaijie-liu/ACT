"""Complete endpoint obligations on original HZ blocks, no joint materialization."""
from itertools import combinations

from act.back_end.moe.check_block_support import validate_batch, check_batch
from act.back_end.moe.check_hz_endpoints import THRESHOLD
from scoped_source.rowwise_bound import clock, identity, rational

SCHEMA = 'BLOCK_HZ_ENDPOINT_REQUEST_V1'
PROOF = 'BLOCK_HZ_ENDPOINT_PROOF_V1'


def validate_request(request, *, expected_request_sha256, deadline):
    tick = clock(deadline)
    if (identity(request) != expected_request_sha256 or set(request) !=
            {'schema', 'experts', 'classes', 'context', 'properties', 'pairs'} or request['schema'] != SCHEMA):
        raise ValueError('block endpoint request binding')
    e, c, props, ctx = (request[k] for k in ('experts','classes','properties','context'))
    if (type(e) is not int or not 2<=e<=4 or type(c) is not int or not 1<=c<=64
            or type(props) is not list or not 1<=len(props)<=4 or type(ctx) is not dict
            or any(type(ctx.get(k)) is not str or not ctx[k] for k in ('request','domain','guard'))):
        raise ValueError('request dimensions/context')
    ids = set()
    for p in props:
        if (set(p) != {'id','q','offset'} or type(p['id']) is not str or not p['id']
                or p['id'] in ids or len(p['q']) != c): raise ValueError('property roster')
        ids.add(p['id'])
        for v in p['q'] + [p['offset']]:
            if type(v) is not str or str(rational(v)) != v: raise ValueError('canonical property')
    roster = [list(p) for p in combinations(range(e),2)]
    if (type(request['pairs']) is not list or [p['pair'] for p in request['pairs']] != roster
            or any(type(v) is not int for p in request['pairs'] for v in p['pair'])):
        raise ValueError('all unordered pairs required in order')
    for record in request['pairs']:
        tick()
        if set(record) != {'pair','gate','batch'}: raise ValueError('block pair fields')
        gate, batch = record['gate'], record['batch']; guard_hash = identity(batch['sources']['entry'])
        if (set(gate) != {'pair','domain','guard_sha256','bounds','premise'} or gate['pair'] != record['pair']
                or gate['domain'] != ctx['domain'] or gate['guard_sha256'] != guard_hash
                or gate['premise'] != 'SUPPLIED_GATE_RANGE_NOT_INDEPENDENTLY_PROVED' or len(gate['bounds']) != 2):
            raise ValueError('gate premise binding')
        lo, hi = map(rational, gate['bounds'])
        if not 0<=lo<=hi<=1 or gate['bounds'] != [str(lo),str(hi)]: raise ValueError('gate range')
        expected_ctx = {'request':ctx['request'],'domain':ctx['domain'],'guard':guard_hash,
                        'caller_context':identity(ctx),'pair':record['pair'],'relation_mode':'shared_input'}
        if batch['context'] != expected_ctx: raise ValueError('batch context binding')
        expected = [(f'p{pi}:e{wi}', [str(t*rational(v)) for v in p['q']]+
                    [str((1-t)*rational(v)) for v in p['q']], p['offset'], 'min')
                    for pi,p in enumerate(props) for wi,t in enumerate(sorted({lo,hi}))]
        if [(q['id'],q['q'],q['offset'],q['side']) for q in batch['queries']] != expected:
            raise ValueError('complete endpoint/property/weight coverage')
        if len(batch['sources']['a']['c']) != c: raise ValueError('class dimension binding')
        validate_batch(batch, expected_batch_sha256=identity(batch), deadline=deadline)
    if identity(request) != expected_request_sha256: raise ValueError('request changed while checking')
    tick(); return request['pairs']


def check_request(request, proof, *, expected_request_sha256, deadline):
    tick = clock(deadline); anchor = identity(proof)
    pairs = validate_request(request, expected_request_sha256=expected_request_sha256, deadline=deadline)
    if (set(proof) != {'schema','request_sha256','pairs'} or proof['schema'] != PROOF
            or proof['request_sha256'] != expected_request_sha256 or len(proof['pairs']) != len(pairs)):
        raise ValueError('proof request coverage')
    results = []; missing = checked = 0
    for record,saved in zip(pairs, proof['pairs']):
        if set(saved) != {'pair','candidates'} or saved['pair'] != record['pair']: raise ValueError('proof pair identity')
        weights = sorted(set(map(rational,record['gate']['bounds'])))
        if saved['candidates'] is None:
            values = None; missing += len(weights)*len(request['properties'])
        else:
            batch = record['batch']
            values = check_batch(batch,saved['candidates'],expected_batch_sha256=identity(batch),deadline=deadline)['results']
            checked += len(values)
        for pi,p in enumerate(request['properties']):
            bounds = None if values is None else [rational(values[pi*len(weights)+i]['bound']) for i in range(len(weights))]
            lower = None if bounds is None else min(bounds)
            results.append({'pair':record['pair'],'property':p['id'],'weights':list(map(str,weights)),
                            'covered_gate_end_labels':[0,1] if bounds else [],
                            'bounds':None if bounds is None else list(map(str,bounds)),
                            'lower_bound':None if lower is None else str(lower),
                            'positive':lower is not None and lower>THRESHOLD})
    positive = sum(r['positive'] for r in results)
    if identity(request) != expected_request_sha256 or identity(proof) != anchor: raise ValueError('request reception pollution')
    tick(); return {'status':'CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE' if positive==len(results)
                    else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE',
                    'required':len(results),'positive':positive,'checked_endpoints':checked,
                    'missing_endpoints':missing,'results':results}
