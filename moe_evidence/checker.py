"""Independent complete request check; no generator, solver or model imports."""
from act.back_end.solver.lp_certificate import check, identity, rational
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.check_rational_mccormick import check_construction
from moe_evidence.schema import TRUSTED, THRESHOLD, validate_request, route_pairs, gate_envelope, interval_lp, pair_key


def check_manifest(m, expected_request, load, *, tick=lambda: None):
    """Tick is an optional monotonic deadline guard; outer watchdog is required.

    Checks bindings and necessary obligations independently. Route exclusions
    and source/guard lowering remain explicitly trusted, even on success.
    """
    tick(); rid = validate_request(expected_request)
    if (m['schema'] != 'WEIGHTED_TOP2_EVIDENCE_V1' or m['request'] != expected_request
            or m['request_id'] != rid or m['trusted_base'] != TRUSTED
            or rational(m['positive_threshold']) != rational(THRESHOLD)):
        raise ValueError('request/assumption/policy binding mismatch')
    r = expected_request; pairs = route_pairs(r, m['routes']); props = r['properties']
    required = {(p,i) for p in pairs for i in range(len(props))}
    rows = m['obligations']
    if type(m['generation_complete']) is not bool or any(type(v['property_index']) is not int for v in rows):
        raise ValueError('invalid completeness/property index')
    if len(rows) != len(required) or {(tuple(v['pair']),v['property_index']) for v in rows} != required:
        raise ValueError('missing or duplicate necessary obligation')
    result = {'required_obligations':len(required), 'positive_obligations':0, 'obligations':[],
              'trusted_base':TRUSTED, 'deployed_float_SAFE':False, 'production_gate_changed':False,
              'route_pairs': [list(p) for p in pairs]}
    if not m['routes']['exact'] or m['routes']['unresolved'] or not pairs:
        return {**result, 'status':'UNKNOWN_ROUTE_COVERAGE'}
    snapshot = load(m['common_facts']); si = snapshot['identity']; scope = snapshot['scope']
    if (snapshot['request_id'] != rid or si != {k:r[k] for k in ('model_state','center','lower','upper')}
            or snapshot['pairs'] != [list(p) for p in pairs]
            or scope['request_id'] != rid or scope['gate'] != r['gate'] or scope['tie_policy'] != r['tie_policy']):
        raise ValueError('common facts refer to different request/domain')
    branches = {b['expert']:b['interval'] for b in snapshot['branches']}
    if len(branches)!=len(snapshot['branches']) or set(branches)!={e for p in pairs for e in p}:
        raise ValueError('membership source inventory mismatch')
    expected_context = {pair_key(p) for p in pairs if any(v['pair']==list(p) and v['kind']=='residual' for v in rows)}
    if set(m['contexts']) != expected_context: raise ValueError('pair context inventory mismatch')
    used_supports = set(); contexts = {}

    def support(key, pair, index, kind, q, source):
        tick(); used_supports.add(key)
        item = m['supports'][key]
        if (item['request_id']!=rid or item['pair']!=list(pair) or item['property_index']!=index
                or item['kind']!=kind or item['source_sha256']!=identity(source)):
            raise ValueError('support scope/source/property mismatch')
        record = load(item['export'])
        if [rational(v) for v in record['q']]!=[rational(v) for v in q] or rational(record['offset'])!=0:
            raise ValueError('wrong support objective')
        if item['status'] not in ('PENDING','UNAVAILABLE','PROPOSED'): raise ValueError('bad support status')
        if item['status']!='PROPOSED' and item['certificate'] is not None: raise ValueError('unexpected certificate')
        cert = load(item['certificate']) if item['status']=='PROPOSED' else None
        checked = check_export(record,cert,expected_source_sha256=identity(source)); tick()
        value = rational(checked['bound']['checked_lower_bound']) if cert else None
        return value

    def context(pair):
        key=pair_key(pair)
        if key not in contexts:
            c=m['contexts'][key]
            if c['pair']!=list(pair) or c['expert_order']!=list(pair) or c['request_id']!=rid:
                raise ValueError('wrong ordered pair/source')
            joint,router=load(c['joint_source']),load(c['router_source'])
            for src,width in ((joint,2*r['classes']),(router,r['experts'])):
                if len(src['c'])!=width or str(src['frame_id'])!=str(scope['frame_id']):
                    raise ValueError('source width/frame mismatch')
            q=[0]*r['experts'];q[pair[0]]=1;q[pair[1]]=-1
            gl=support(key+'_gate_lo',pair,None,'router_order',q,router)
            gu=support(key+'_gate_hi',pair,None,'router_order',[-v for v in q],router)
            contexts[key]=(joint,gate_envelope(gl,gu))
        return contexts[key]

    for row in rows:
        tick(); pair=tuple(row['pair']);i=row['property_index'];prop=props[i];value=None
        if row['property']!=prop: raise ValueError('property changed')
        if row['kind']=='reused':
            facts=row['facts']
            if len(facts)!=2: raise ValueError('two expert facts required')
            values=[]
            for expert,fact in zip(pair,facts):
                if (fact['expert']!=expert or fact['request_id']!=rid or fact['scope']!=scope
                        or fact['property_index']!=i or fact['guard']!='TOP2_MEMBERSHIP'
                        or fact['interval']!=branches[expert]): raise ValueError('invalid scoped reuse')
                lp=interval_lp(branches[expert],prop,r['classes'])
                checked=check(lp,fact['certificate']);v=rational(checked['checked_lower_bound'])
                if v<=rational(THRESHOLD): raise ValueError('nonpositive reused fact')
                values.append(v)
            value=min(values);state='CHECKED_REUSED_POSITIVE'
        elif row['kind']=='pending': state='MISSING_EVIDENCE'
        elif row['kind']=='residual':
            joint,gate=context(pair);base=pair_key(pair)+f'_p{i}'
            q=prop['q'];qd=q+[str(-rational(v)) for v in q]
            lo=support(base+'_lo',pair,i,'difference',qd,joint)
            neg=support(base+'_hi',pair,i,'difference',[str(-rational(v)) for v in qd],joint)
            if lo is None or neg is None:
                if row['weighted_status']!='RANGE_UNAVAILABLE' or row['weighted'] is not None:
                    raise ValueError('weighted proof lacks checked range')
                state='MISSING_EVIDENCE'
            else:
                if lo > -neg: raise ValueError('inconsistent checked difference bounds')
                diff=[str(lo),str(-neg)]
                if row['weighted_status']=='RANGE_UNAVAILABLE':
                    if row['weighted'] is not None or row['certificate'] is not None:
                        raise ValueError('unexpected unattempted weighted certificate')
                    result['obligations'].append({'pair':list(pair),'property_index':i,
                        'state':'MISSING_EVIDENCE','lower_bound':None})
                    tick();continue
                if row['gate_bounds']!=gate or row['difference_bounds']!=diff: raise ValueError('range changed/inward rounded')
                if row['weighted_status'] not in ('PROPOSED','UNAVAILABLE'): raise ValueError('bad weighted state')
                rec=load(row['weighted']);cert=load(row['certificate']) if row['weighted_status']=='PROPOSED' else None
                checked=check_construction(rec,cert,source_hash=identity(joint),q=q,offset=prop['constant'],gate=gate,difference=diff)
                value=rational(checked['bound']['checked_lower_bound']) if cert else None
                state=('MISSING_EVIDENCE' if value is None else 'CHECKED_RATIONAL_POSITIVE'
                       if value>rational(THRESHOLD) else 'CHECKED_NONPOSITIVE_OR_BELOW_THRESHOLD')
        else: raise ValueError('unknown obligation kind')
        tick(); result['obligations'].append({'pair':list(pair),'property_index':i,'state':state,
                                             'lower_bound':str(value) if value is not None else None})
    if set(m['supports'])!=used_supports: raise ValueError('extra/missing support obligations')
    positive=[x for x in result['obligations'] if x['state'] in ('CHECKED_REUSED_POSITIVE','CHECKED_RATIONAL_POSITIVE')]
    missing=sum(x['state']=='MISSING_EVIDENCE' for x in result['obligations'])
    result.update(positive_obligations=len(positive), missing_obligations=missing,
                  nonpositive_obligations=len(rows)-missing-len(positive))
    result['status']=('CHECKED_CONDITIONAL' if len(positive)==len(required) and m['generation_complete']
                      else 'UNKNOWN_MISSING_EVIDENCE' if missing or not m['generation_complete'] else 'UNKNOWN_NONPOSITIVE')
    result['minimum_lower_bound']=str(min(rational(x['lower_bound']) for x in positive)) if result['status']=='CHECKED_CONDITIONAL' else None
    tick(); return result
