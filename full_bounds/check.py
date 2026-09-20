"""Independent exact dual checking on new, source-checked output LPs."""
from fractions import Fraction as F
from act.back_end.solver.lp_certificate import identity, rational
from act.back_end.solver.sparse_lp_certificate import evaluate, rows


def program(base, obligation):
    """Independent shared-base expansion after parent construction validation."""
    n=base['variables']; objective=list(rows(obligation['objective'],n))
    if len(objective)!=1:raise ValueError('one objective required')
    c=['0']*n
    for j,v in objective[0]:c[j]=str(v)
    a,b=base['A'],obligation['A_extra']
    if b['shape']!=[4,n] or a['shape'][1]!=n:raise ValueError('product matrix dimensions')
    return {'matrix_format':'csr_v1','c':c,'offset':obligation['offset'],
        'A':{'shape':[a['shape'][0]+4,n],'data':a['data']+b['data'],'indices':a['indices']+b['indices'],
             'indptr':a['indptr']+[len(a['data'])+p for p in b['indptr'][1:]]},
        'b':base['b']+obligation['b_extra'],'E':base['E'],'h':base['h'],
        'lower':[-1]*(n-2)+[v[0] for v in obligation['extra_bounds']],
        'upper':[1]*(n-2)+[v[1] for v in obligation['extra_bounds']]}


def check_one(lp,candidate):
    if set(candidate)!={'lp_sha256','inequality_dual','equality_dual'}:
        raise ValueError('candidate multipliers only; no unverified bound claim')
    value,residual=evaluate(lp,candidate) # unchanged independent rational checker
    dual=rational(lp['offset'])
    for key,rhs in [('inequality_dual','b'),('equality_dual','h')]:
        dual+=sum((rational(x)*rational(y) for x,y in zip(candidate[key],lp[rhs])),F(0))
    return {'status':'CHECKED_POSITIVE_BOUND' if value>0 else 'CHECKED_NONPOSITIVE_BOUND',
        'lp_sha256':identity(lp),'checked_lower_bound':str(value),'dual_constant':str(dual),
        'residual_box_correction':str(value-dual),
        'residual_l1':str(sum(map(abs,residual),F(0))),
        'nonzero_residual_coordinates':sum(v!=0 for v in residual)}


def aggregate(base,obligations,manifest,outcomes,read):
    request_id=identity(manifest['request']);parent_id=manifest['parent_manifest_sha256']
    expected=[r['competitor'] for r in obligations['rows']]
    if ([r['competitor'] for r in outcomes]!=expected or len(set(expected))!=len(expected) or not expected):
        raise ValueError('complete ordered outcome inventory')
    checked=[]
    for obligation,entry in zip(obligations['rows'],outcomes):
        k=entry['competitor'];file=entry['file']
        if file is None:
            checked.append({'competitor':k,'status':'NO_PUBLISHED_CANDIDATE'});continue
        if file!=f'candidate_{k}.json':raise ValueError('candidate path binding')
        record=read(file)
        if (record['schema']!='NEW_OUTPUT_DUAL_CANDIDATE_V1' or record['competitor']!=k or
                record['request_id']!=request_id or record['parent_manifest_sha256']!=parent_id or
                record['pair']!=manifest['pair'] or record['source_sha256']!=obligations['source_sha256'] or
                record['base_sha256']!=obligations['base_sha256']):raise ValueError('candidate source/request/property binding')
        lp=program(base,obligation)
        if record['lp_sha256']!=identity(lp):raise ValueError('new LP identity drift')
        if record['candidate'] is None:
            checked.append({'competitor':k,'status':'NO_MULTIPLIER_CANDIDATE','lp_sha256':record['lp_sha256']});continue
        result=check_one(lp,record['candidate'])
        checked.append({'competitor':k,**result})
    positives=sum(r['status']=='CHECKED_POSITIVE_BOUND' for r in checked)
    nonpositives=sum(r['status']=='CHECKED_NONPOSITIVE_BOUND' for r in checked)
    missing=len(expected)-positives-nonpositives;complete=positives==len(expected)
    return {'status':'CHECKED_POSITIVE_DECLARED_REAL_MOE' if complete else
        'UNKNOWN_MISSING_BOUND_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE_BOUNDS',
        'required_obligations':len(expected),'positive_bounds':positives,'nonpositive_bounds':nonpositives,
        'missing_bounds':missing,'rows':checked,'complete_declared_real_output_proof':complete,
        'minimum_available_bound':str(min((F(r['checked_lower_bound']) for r in checked if 'checked_lower_bound' in r),default=F(0)))
            if positives+nonpositives else None,
        'old_LP_certificates_used':0,'production_verdict_changed':False,'deployed_floating_point_proof':False}
