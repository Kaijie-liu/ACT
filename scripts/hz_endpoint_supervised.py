"""Fixed CPU endpoint policy for the existing bounded lifecycle engine.

Execution completion, obligation coverage and positivity are separate fields.
No portable/source-complete/native-float/GPU guarantee is added here.
"""
from fractions import Fraction as F
import time

from scoped_proof.io import ROOT, load, sha, tick
from source_enclosure.format import identity
from scripts.hz_propagation_supervised import finite, required_hash
from scripts.hz_device_lifecycle import cleanup_confirmed
from scripts.run_hz_endpoint_controls import FILES as ENDPOINT_FILES
from scripts.hz_device_lifecycle import FILES as LIFECYCLE_FILES

CONFIG='configs/hz_endpoint_supervision_20261001.json'
CONFIG_SHA='87310481d2c9fd2fa2336974442d5695d88546a13a12dd586895daa4bb1f4e45'
REPORT_SHA='73e7bee251b2580676f1d713b3e932040d9edffd2b54ac25e1446411a4a7ee61'
DONE='CHECKED_ENDPOINT_CPU_EXECUTION'
WORKER='scripts.hz_endpoint_worker'
LIMIT=4*2**20
FILES=tuple(dict.fromkeys((*ENDPOINT_FILES,*LIFECYCLE_FILES,CONFIG,
    'docs/hz_endpoint_supervision_design_20261001.md','docs/hz_endpoint_controls_20261001_r4.json',
    'scoped_proof/device_lifecycle.py','scripts/hz_endpoint_supervised.py',
    'scripts/hz_endpoint_worker.py','scripts/test_hz_endpoint_supervised.py',
    'scripts/run_hz_endpoint_supervision.py')))


def protocol():
    cfg=load(ROOT/CONFIG,CONFIG_SHA)
    load(ROOT/cfg['reference_protocol'],cfg['reference_protocol_sha256'])
    load(ROOT/cfg['reference_report'],REPORT_SHA)
    return cfg


def sources(): return {name:sha(ROOT/name) for name in FILES}


def spec(case='separation',control=''):
    cfg=protocol()
    if case not in cfg['cases'] or control not in ('',*cfg['faults']) or control and case!='separation':
        raise ValueError('frozen tiny CPU roster')
    return {'schema':'HZ_ENDPOINT_CPU_SPEC_V1','case':case,'control':control,
            'protocol_sha256':CONFIG_SHA,**cfg['cases'][case]}


def validate_spec(s,permit):
    if permit is not None or s!=spec(s['case'],s['control']): raise ValueError('frozen spec binding')


def budget(s): return protocol()['faults'][s['control']][0] if s['control'] else 30


def cutoff(inv,phase,begin):
    if phase=='admit': return min(inv['work'],inv['start']+2)
    if phase=='produce': return inv['work']-min(5.,inv['budget']/4)
    if phase=='release': return min(inv['work'],begin+2)
    return inv['work']


def visible(inv,phase): return ''


def basis(request):
    return {**{k:request[k] for k in ('experts','classes','properties','context')},
            'pairs':[{k:p[k] for k in ('pair','sources','gate','relation_mode')} for p in request['pairs']]}


def validate_basis(request,s):
    if identity(basis(request))!=s['basis_sha256']: raise ValueError('frozen source/gate/property basis')


def admitted(root,inv,stage):
    if load(root/'admit.json',required_hash(stage['output_sha256']),LIMIT)!={
            'status':'CPU_NO_CUDA','invocation':inv['invocation']}:
        raise ValueError('CPU admission identity')
    return True


def produced(root,inv,stage):
    payload(root,inv,stage['output_sha256']); return True


def payload(root,inv,digest):
    p=load(root/'produce.json',required_hash(digest),LIMIT)
    if (set(p)!={'invocation','spec_sha256','request','proof','request_file_sha256','prefixes','cost_seconds'}
            or p['invocation']!=inv['invocation'] or p['spec_sha256']!=identity(inv['spec'])):
        raise ValueError('producer identity/schema')
    validate_basis(p['request'],inv['spec'])
    if load(root/'prefix_request.json',required_hash(p['request_file_sha256']),LIMIT)!=p['request']:
        raise ValueError('original request snapshot')
    stage=load(root/'produce_stage.json',limit=LIMIT)
    if stage.get('output_sha256')!=digest: raise ValueError('parent producer anchor')
    costs=p['cost_seconds']
    if set(costs)!={'create','prepare','propose'}: raise ValueError('required producer costs')
    for value in costs.values(): finite(value)
    if sum(costs.values())>stage['seconds']: raise ValueError('nested costs exceed producer')
    validate_prefixes(root,p['request'],p['prefixes'],p['proof'])
    return p


def validate_prefixes(root,request,refs,proof):
    expected=[p['pair'] for p in request['pairs']]
    previous=None
    if (not refs or len(refs)>len(expected)+1
            or list(refs)!=[f'prefix_{i:02}.json' for i in range(len(refs))]):
        raise ValueError('append-only prefix sequence')
    for i,(name,digest) in enumerate(refs.items()):
        prefix=load(root/name,required_hash(digest),LIMIT)
        if (set(prefix)!={'schema','request_sha256','pairs'}
                or prefix['schema']!='GUARDED_HZ_ENDPOINT_PROOF_V1'
                or prefix['request_sha256']!=identity(request)
                or [p['pair'] for p in prefix['pairs']]!=expected
                or any(set(p)!={'pair','candidates'} for p in prefix['pairs'])
                or any((p['candidates'] is not None)!=(j<i) for j,p in enumerate(prefix['pairs']))):
            raise ValueError('partial prefix coverage')
        if previous and prefix['pairs'][:i-1]!=previous['pairs'][:i-1]:
            raise ValueError('completed candidate prefix changed')
        previous=prefix
    if previous!=proof: raise ValueError('final proof differs from prefix')
    actual={p.name for p in root.glob('prefix_*.json')}-{'prefix_request.json'}
    if actual!=set(refs): raise ValueError('unaccounted prefix')


def check(root,inv,digest,deadline):
    p=payload(root,inv,digest); tick(deadline)
    from act.back_end.moe.check_hz_endpoints import check_request
    aggregate=check_request(p['request'],p['proof'],expected_request_sha256=identity(p['request']),deadline=deadline)
    flat=aggregate_rows(p,aggregate)
    tick(deadline)
    return {'invocation':inv['invocation'],'payload_sha256':digest,
            'result':{'results':flat,'aggregation':aggregate}}


def aggregate_rows(payload,agg):
    """Stdlib-only receiver validation; exact LP checking happens separately."""
    request,proof=payload['request'],payload['proof']; flat=[]; positive=missing=0; rows=[]
    if (agg['request_sha256']!=identity(request) or agg['proof_sha256']!=identity(proof)
            or any(agg[k] is not False for k in ('source_complete','deployed_float_SAFE','hard_budget_supervision'))):
        raise ValueError('checker source/guarantee binding')
    if len(agg['results'])!=len(request['pairs'])*len(request['properties']): raise ValueError('duty count')
    for pair,part in zip(request['pairs'],proof['pairs']):
        weights=list(map(str,sorted(set(map(F,pair['gate']['bounds'])))))
        for prop in request['properties']:
            r=agg['results'][len(rows)]; values=r['bounds']
            if (r['pair']!=pair['pair'] or r['property']!=prop['id'] or r['weights']!=weights
                    or (values is None)!=(part['candidates'] is None)):
                raise ValueError('duty pair/property/availability')
            if values is None:
                lower=None; missing+=len(weights)
            else:
                if len(values)!=len(weights): raise ValueError('all distinct endpoints required')
                lower=min(map(F,values))
                flat.extend({'pair':pair['pair'],'property':prop['id'],'weight':w,'bound':v}
                            for w,v in zip(weights,values))
            good=lower is not None and lower>F(1,10_000_000)
            if (r['lower_bound']!=(None if lower is None else str(lower)) or r['positive'] is not good
                    or r['covered_gate_end_labels']!=([0,1] if values is not None else [])):
                raise ValueError('endpoint lower bound/threshold/coverage')
            positive+=good; rows.append(r)
    status=('CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE' if positive==len(rows)
            else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE')
    if (agg['required']!=len(rows) or agg['positive']!=positive or agg['checked_endpoints']!=len(flat)
            or agg['missing_endpoints']!=missing or agg['status']!=status):
        raise ValueError('full aggregation counts/status')
    return flat


def receive(root,inv,inputs):
    p=payload(root,inv,inputs['produce']); checked=load(root/'check.json',required_hash(inputs['check']),LIMIT)
    if (set(checked)!={'invocation','payload_sha256','result'} or checked['invocation']!=inv['invocation']
            or checked['payload_sha256']!=inputs['produce'] or set(checked['result'])!={'results','aggregation'}):
        raise ValueError('parent checker output binding')
    agg=checked['result']['aggregation']
    if checked['result']['results']!=aggregate_rows(p,agg): raise ValueError('flattened endpoint identity')
    return {'status':DONE,'invocation':inv['invocation'],'inputs':inputs,'checked':checked['result'],
            'obligations_complete':agg['missing_endpoints']==0,
            'positive':agg['status']=='CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE',
            'complete_moe_proof':False,'cuda_execution':False}


def release_gate(root,inv,records):
    p,r=records['produce'],records['release']
    if not cleanup_confirmed(p) or not cleanup_confirmed(r) or r['status']!='COMPLETED':
        return {'status':'CPU_RELEASE_UNCONFIRMED','release_confirmed':False}
    if load(root/'release.json',required_hash(r['output_sha256']),LIMIT)!={
            'status':'CPU_NO_CUDA_TO_RELEASE','invocation':inv['invocation'],
            'producer_sha256':sha(root/'produce_stage.json')}:
        raise ValueError('CPU cleanup receipt identity')
    return {'status':'CPU_NO_CUDA_TO_RELEASE','release_confirmed':True,'next_admission_established':False}


def decision(root,inv,stages):
    records={s['phase']:s for s in stages}
    if any(not cleanup_confirmed(s) for s in stages): return 'CLEANUP_INCOMPLETE',None,None
    release=release_gate(root,inv,records) if 'release' in records else None
    for s in stages:
        if s['status']!='COMPLETED': return s['status'],release,None
    if 'admit' in records: admitted(root,inv,records['admit'])
    if 'produce' in records and not (release or {}).get('release_confirmed'):
        return 'CPU_RELEASE_UNCONFIRMED',release,None
    if list(records)!=['admit','produce','release','check','receive']: return 'ERROR',release,None
    inputs={k:records[k]['output_sha256'] for k in ('produce','release','check')}
    accepted=receive(root,inv,inputs)
    if load(root/'receive.json',records['receive']['output_sha256'],LIMIT)!=accepted:
        raise ValueError('receiver receipt identity')
    return DONE,release,accepted


def recheck_prefix(root,inv,deadline):
    """Offline-only check; never changes the online execution status."""
    path=root/'prefix_request.json'
    if not path.exists(): return None
    request=load(path,limit=LIMIT); validate_basis(request,inv['spec'])
    refs={p.name:sha(p) for p in sorted(root.glob('prefix_*.json')) if p.name!='prefix_request.json'}
    if not refs: return None
    proof=load(root/next(reversed(refs)),limit=LIMIT); validate_prefixes(root,request,refs,proof)
    from act.back_end.moe.check_hz_endpoints import check_request
    return check_request(request,proof,expected_request_sha256=identity(request),deadline=deadline)
