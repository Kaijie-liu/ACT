"""Independent pair-map and endpoint reception for supplied guarded HZs.

No pair builder, exporter, numerical optimizer, SciPy or torch is used here.
Network/guard lowering and gate coverage remain explicit premises. ACT imports
are not a portable solver-free entry; deadlines are cooperative.
"""
from fractions import Fraction as F
from itertools import combinations

from act.back_end.moe.check_batched_support import check_batch, validated_records
from scoped_source.rowwise_bound import clock, identity, rational, rows

SCHEMA = 'GUARDED_HZ_ENDPOINT_REQUEST_V1'
PROOF = 'GUARDED_HZ_ENDPOINT_PROOF_V1'
THRESHOLD = F(1, 10_000_000)


def _source(src, tick):
    if set(src) != {'c','b','ub','frame_id','exact','Gc','Gb','Ac','Ab','Auc','Aub'}:
        raise ValueError('complete original HZ source required')
    if type(src['frame_id']) is not int or type(src['exact']) is not bool:
        raise ValueError('explicit frame and exact flag')
    nc, nb = src['Gc']['shape'][1], src['Gb']['shape'][1]
    if (type(nc) is not int or type(nb) is not int or min(nc, nb) < 0
            or not 1 <= nc+nb <= 128 or not 1 <= len(src['c']) <= 128
            or len(src['b'])+len(src['ub']) > 256):
        raise ValueError('finite source capacity')
    out = {k: [rational(v) for v in src[k]] for k in ('c','b','ub')}
    for k, nr, width in [('Gc',len(out['c']),nc),('Gb',len(out['c']),nb),
                         ('Ac',len(out['b']),nc),('Ab',len(out['b']),nb),
                         ('Auc',len(out['ub']),nc),('Aub',len(out['ub']),nb)]:
        out[k] = [{j: v for j, v in row if v} for row in rows(src[k], (nr,width), tick)]
    return out, nc, nb


def check_pair_map(record, tick):
    """Reconstruct disjoint private blocks and every retained row, independently."""
    sources = record['sources']
    if set(sources) != {'entry','a','b'}:
        raise ValueError('three source snapshots required')
    parsed = {k: _source(v, tick) for k,v in sources.items()}
    entry, sc, sb = parsed['entry']; a, ac, ab = parsed['a']; b, bc, bb = parsed['b']
    frame = sources['entry']['frame_id']
    for name in ('a','b'):
        value, nc, nb = parsed[name]
        if nc < sc or nb < sb or sources[name]['frame_id'] != frame:
            raise ValueError('source frame/shared prefix')
        for ckey,bkey,rhs in [('Ac','Ab','b'),('Auc','Aub','ub')]:
            count = len(entry[rhs])
            if (value[rhs][:count] != entry[rhs] or value[ckey][:count] != entry[ckey]
                    or value[bkey][:count] != entry[bkey]):
                raise ValueError('shared constraints changed/lost/private prefix pollution')
    if len(a['c']) != len(b['c']):
        raise ValueError('expert output dimensions')
    mode = record['relation_mode']
    if mode == 'shared_input':
        nc, nb = ac+bc-sc, ab+bb-sb
        cm = {'entry':list(range(sc)), 'a':list(range(ac)),
              'b':list(range(sc))+list(range(ac,nc))}
        bm = {'entry':list(range(sb)), 'a':list(range(ab)),
              'b':list(range(sb))+list(range(ab,nb))}
        parts = [('entry',0),('a',None),('b',None)]
    elif mode == 'independent_inputs':
        nc, nb = ac+bc, ab+bb
        cm = {'a':list(range(ac)), 'b':list(range(ac,nc))}
        bm = {'a':list(range(ab)), 'b':list(range(ab,nb))}
        parts = [('a',0),('b',0)]
    else:
        raise ValueError('unknown relation mode')
    joint = record['batch']['source']; actual,jc,jb = _source(joint,tick)
    if ((jc,jb)!=(nc,nb) or actual['c']!=a['c']+b['c']
            or (mode=='shared_input' and (joint['frame_id']!=frame or
                joint['exact']!=all(s['exact'] for s in sources.values())))
            or (mode=='independent_inputs' and (joint['frame_id']==frame or joint['exact']))):
        raise ValueError('joint frame/shape/center')
    def remap(source, key, mapping, start=0):
        return [{mapping[j]:v for j,v in row.items()} for row in parsed[source][0][key][start:]]
    for key,mapping in [('Gc',cm),('Gb',bm)]:
        if actual[key] != remap('a',key,mapping['a'])+remap('b',key,mapping['b']):
            raise ValueError('private output alias or mapping')
    for ckey,bkey,rhs in [('Ac','Ab','b'),('Auc','Aub','ub')]:
        expected_c=[]; expected_b=[]; expected_rhs=[]
        for name,start in parts:
            start = len(entry[rhs]) if start is None else start
            expected_c += remap(name,ckey,cm[name],start)
            expected_b += remap(name,bkey,bm[name],start)
            expected_rhs += parsed[name][0][rhs][start:]
        if (actual[ckey]!=expected_c or actual[bkey]!=expected_b or actual[rhs]!=expected_rhs):
            raise ValueError('joint constraint omission or column mapping')
    return len(a['c'])


def validate_request(request, *, expected_request_sha256, deadline):
    tick=clock(deadline)
    if (identity(request)!=expected_request_sha256 or set(request)!=
            {'schema','experts','classes','properties','context','pairs'} or request['schema']!=SCHEMA):
        raise ValueError('caller endpoint request binding')
    e,c=request['experts'],request['classes']; props=request['properties']; ctx=request['context']
    if (type(e) is not int or not 2<=e<=4 or type(c) is not int or not 1<=c<=64
            or type(props) is not list or not 1<=len(props)<=4 or type(ctx) is not dict
            or any(type(ctx.get(k)) is not str or not ctx[k] for k in ('request','domain','guard'))):
        raise ValueError('request dimensions/context')
    ids=set()
    for p in props:
        if (set(p)!={'id','q','offset'} or type(p['id']) is not str or not p['id']
                or p['id'] in ids or len(p['q'])!=c): raise ValueError('property roster')
        ids.add(p['id'])
        for v in p['q']+[p['offset']]:
            if type(v) is not str or str(rational(v))!=v: raise ValueError('exact property coefficients')
    roster=[list(p) for p in combinations(range(e),2)]
    if (type(request['pairs']) is not list or [p['pair'] for p in request['pairs']]!=roster
            or any(type(v) is not int for p in request['pairs'] for v in p['pair'])):
        raise ValueError('all unordered pairs required, in order')
    for record in request['pairs']:
        tick()
        if set(record)!={'pair','gate','sources','relation_mode','batch'}: raise ValueError('pair fields')
        if check_pair_map(record,tick)!=c: raise ValueError('pair/class binding')
        gate=record['gate']; source_hash=identity(record['sources']['entry'])
        if (set(gate)!={'pair','domain','guard_sha256','bounds','premise'} or gate['pair']!=record['pair']
                or gate['domain']!=ctx['domain'] or gate['guard_sha256']!=source_hash
                or gate['premise']!='SUPPLIED_GATE_RANGE_NOT_INDEPENDENTLY_PROVED'
                or len(gate['bounds'])!=2): raise ValueError('gate premise binding')
        low,high=(rational(v) for v in gate['bounds'])
        if not 0<=low<=high<=1: raise ValueError('gate interval')
        if gate['bounds']!=[str(low),str(high)]: raise ValueError('canonical gate range')
        batch=record['batch']
        expected_context={'request':ctx['request'],'domain':ctx['domain'],'guard':source_hash,
                          'caller_context':identity(ctx),'pair':record['pair'],'relation_mode':record['relation_mode']}
        if batch['context']!=expected_context: raise ValueError('batch context binding')
        # Independently form t*A+(1-t)*B, with the property constant once.
        expected=[]
        for pi,p in enumerate(props):
            for wi,t in enumerate(sorted({low,high})):
                q=[str(t*rational(v)) for v in p['q']]+[str((1-t)*rational(v)) for v in p['q']]
                expected.append((f'p{pi}:e{wi}',q,p['offset'],'min'))
        actual=[(q['id'],q['q'],q['offset'],q['side']) for q in batch['queries']]
        if actual!=expected: raise ValueError('endpoint property/weight/coverage')
        validated_records(batch,expected_batch_sha256=identity(batch),deadline=deadline)
    tick()
    if identity(request)!=expected_request_sha256: raise ValueError('request changed while checking')
    return request['pairs']


def check_request(request, proof, *, expected_request_sha256, deadline):
    tick=clock(deadline); proof_hash=identity(proof)
    pairs=validate_request(request,expected_request_sha256=expected_request_sha256,deadline=deadline)
    if (set(proof)!={'schema','request_sha256','pairs'} or proof['schema']!=PROOF
            or proof['request_sha256']!=expected_request_sha256 or len(proof['pairs'])!=len(pairs)):
        raise ValueError('proof request coverage')
    results=[]; checked=0; missing=0
    for record,saved in zip(pairs,proof['pairs']):
        tick()
        if set(saved)!={'pair','candidates'} or saved['pair']!=record['pair']: raise ValueError('proof pair identity')
        weights=sorted(set(map(rational,record['gate']['bounds'])))
        if saved['candidates'] is None:
            values=None; missing+=len(weights)*len(request['properties'])
        else:
            batch=record['batch']
            values=check_batch(batch,saved['candidates'],expected_batch_sha256=identity(batch),deadline=deadline)['results']
            checked+=len(values)
        for pi,p in enumerate(request['properties']):
            bounds=None if values is None else [rational(values[pi*len(weights)+i]['bound']) for i in range(len(weights))]
            lower=None if bounds is None else min(bounds)
            results.append({'pair':record['pair'],'property':p['id'],
                            'weights':list(map(str,weights)),'covered_gate_end_labels':[0,1] if bounds else [],
                            'bounds':None if bounds is None else list(map(str,bounds)),
                            'lower_bound':None if lower is None else str(lower),
                            'positive':lower is not None and lower>THRESHOLD})
    positive=sum(r['positive'] for r in results); tick()
    if identity(request)!=expected_request_sha256 or identity(proof)!=proof_hash: raise ValueError('reception input pollution')
    return {'status':'CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE' if positive==len(results)
            else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE',
            'request_sha256':expected_request_sha256,'proof_sha256':proof_hash,'required':len(results),
            'positive':positive,'checked_endpoints':checked,'missing_endpoints':missing,'results':results,
            'source_complete':False,'deployed_float_SAFE':False,'hard_budget_supervision':False,
            'remaining_trust':['network_and_guard_lowering_to_supplied_HZ','factor_provenance_of_supplied_HZ',
                               'supplied_gate_range_covers_actual_weight','exact_checker_implementation']}
