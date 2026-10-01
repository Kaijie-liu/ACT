"""Endpoint preparation and proposals directly on common/entry/template HZs."""
from copy import deepcopy
from itertools import combinations

from act.back_end.moe.block_support import prepare_batch, propose_batch, snapshot
from act.back_end.moe.check_block_endpoints import SCHEMA, PROOF, validate_request
from scoped_source.rowwise_bound import clock, identity, rational


def prepare_request(pairs, properties, *, experts, classes, context, deadline):
    tick = clock(deadline)
    if type(experts) is not int or not 2<=experts<=4: raise ValueError('finite expert capacity')
    if [p['pair'] for p in pairs] != [list(p) for p in combinations(range(experts),2)]:
        raise ValueError('complete pair roster')
    props = [{'id':p['id'],'q':[str(rational(v)) for v in p['q']],
              'offset':str(rational(p['offset']))} for p in properties]
    request = {'schema':SCHEMA,'experts':experts,'classes':classes,'context':deepcopy(context),'properties':props,'pairs':[]}
    for p in pairs:
        gate = list(map(rational,p['gate']))
        if len(gate)!=2 or not 0<=gate[0]<=gate[1]<=1: raise ValueError('gate range')
        guard_hash = identity(snapshot(p['entry'],tick)); queries = []
        for pi,prop in enumerate(props):
            for wi,t in enumerate(sorted(set(gate))):
                q = [t*rational(v) for v in prop['q']] + [(1-t)*rational(v) for v in prop['q']]
                queries.append({'id':f'p{pi}:e{wi}','q':list(map(str,q)),'offset':prop['offset'],'side':'min'})
        ctx = {'request':context['request'],'domain':context['domain'],'guard':guard_hash,
               'caller_context':identity(context),'pair':p['pair'],'relation_mode':'shared_input'}
        batch = prepare_batch(p['common'],p['entry'],p['a'],p['b'],queries,context=ctx,deadline=deadline)
        request['pairs'].append({'pair':list(p['pair']),'batch':batch,
            'gate':{'pair':list(p['pair']),'domain':context['domain'],'guard_sha256':guard_hash,
                    'bounds':list(map(str,gate)),'premise':'SUPPLIED_GATE_RANGE_NOT_INDEPENDENTLY_PROVED'}})
    validate_request(request,expected_request_sha256=identity(request),deadline=deadline); tick(); return request


def propose_request(request, *, expected_request_sha256, deadline, device='cpu'):
    tick = clock(deadline)
    if device!='cpu': raise ValueError('CPU block controls only')
    pairs = validate_request(request,expected_request_sha256=expected_request_sha256,deadline=deadline)
    proof = {'schema':PROOF,'request_sha256':expected_request_sha256,'pairs':[]}
    for p in pairs:
        b = p['batch']; candidate = propose_batch(b,expected_batch_sha256=identity(b),deadline=deadline)
        proof['pairs'].append({'pair':list(p['pair']),'candidates':candidate})
    if identity(request)!=expected_request_sha256: raise ValueError('proposal request pollution')
    tick(); return proof
