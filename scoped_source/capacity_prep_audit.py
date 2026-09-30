"""Read-only preparation accounting, requiring an observed completed API call.

No model, checkpoint, dataset, inference or solver is loaded. Source-byte
reconstruction is an identity check and never a source-to-output proof.
"""
import math
from pathlib import Path
import time
from scoped_proof.io import load,sha,tick
from scoped_source.capacity_intake import sources,PROTOCOL_SHA
from scoped_source.factored_io import required_hash
from source_enclosure.format import identity

STATES=('IDENTITY_PREPARED','ERROR','TIMEOUT','RESOURCE_LIMIT')


def nonnegative(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative cost required')
    return value


def audit(root,observed,*,expected_sources=None,recheck=True):
    from scoped_source.capacity_prepare import validate_spec,receive
    root=Path(root)
    if observed is None: raise ValueError('observed API return required')
    if (set(observed)!={'status','seconds','root','invocation','finish_sha256','real_capacity_admitted'} or
        observed['root']!=str(root) or observed['status'] not in STATES or
        observed['real_capacity_admitted'] is not False): raise ValueError('completed call identity')
    elapsed=nonnegative(observed['seconds'])
    finish=load(root/'finish.json',required_hash(observed['finish_sha256']))
    terminal=load(root/'terminal.json',required_hash(finish['terminal_sha256']))
    inv=load(root/'invocation.json',required_hash(terminal['invocation_sha256'])); spec=load(root/'spec.json')
    validate_spec(spec)
    expected_sources=sources() if expected_sources is None else expected_sources
    if (inv['schema']!='H2_CAPACITY_PREP_INVOCATION_V1' or
        terminal['schema']!='H2_CAPACITY_PREP_TERMINAL_V1' or finish['schema']!='H2_CAPACITY_PREP_FINISH_V1' or
        inv['sources']!=expected_sources or terminal['proof_status']!='NOT_A_PROOF' or
        any(x['invocation']!=observed['invocation'] for x in (finish,terminal,inv)) or
        inv['spec_sha256']!=identity(spec) or terminal['spec_sha256']!=identity(spec) or
        finish['status']!=terminal['status'] or terminal['status'] not in STATES):
        raise ValueError('bound preparation chain')
    budget=nonnegative(inv['budget']); start=nonnegative(inv['start'])
    if not 0<budget<=300 or type(inv['rss_limit']) is not int or not 0<inv['rss_limit']<=2*2**30:
        raise ValueError('frozen preparation resource limits')
    work=start+budget-min(1.,budget/5)
    if (inv['deadline']!=start+budget or inv['work_deadline']!=work or
        inv['produce_deadline']!=work-min(60.,budget/3)):
        raise ValueError('registered deadline allocation')
    before=nonnegative(terminal['seconds_before_publication']); finish_time=nonnegative(finish['seconds_before_finish'])
    if not before<=finish_time<=elapsed: raise ValueError('publication accounting')
    stages=terminal['stages']; end=0.; total=0.
    if len(stages)>2: raise ValueError('extra stages')
    for i,stage in enumerate(stages):
        phase=('prepare','receive')[i]
        if stage['phase']!=phase or load(root/(phase+'_stage.json'))!=stage: raise ValueError('stage inventory')
        a=nonnegative(stage['start_seconds']); b=nonnegative(stage['end_seconds']); seconds=nonnegative(stage['seconds'])
        nonnegative(stage['sampled_peak_rss'])
        if not end<=a<=b<=before or seconds>b-a+1e-6: raise ValueError('stage accounting')
        if stage['status'] not in ('COMPLETED','ERROR','TIMEOUT','RESOURCE_LIMIT'): raise ValueError('stage status')
        if stage['pid'] is not None:
            expected=inv['produce_deadline'] if phase=='prepare' else inv['work_deadline']
            if stage['deadline_monotonic']!=expected or stage['cleanup_included'] is not True:
                raise ValueError('stage deadline/cleanup')
        if i==0 and len(stages)==2 and stage['status']!='COMPLETED': raise ValueError('continuation after failed preparation')
        total+=seconds; end=b
    overhead=nonnegative(terminal['overhead_seconds'])
    if abs(before-total-overhead)>1e-7: raise ValueError('complete API accounting')
    timed_out=(root/'publication_timeout.json').exists() or elapsed>=budget
    expected_status='TIMEOUT' if timed_out else terminal['status']
    if observed['status']!=expected_status: raise ValueError('final timeout precedence')
    prepared=expected_status=='IDENTITY_PREPARED'; accepted=terminal['accepted']
    if terminal['status']=='IDENTITY_PREPARED':
        if (len(stages)!=2 or any(s['status']!='COMPLETED' or s['returncode']!=0 for s in stages) or
            before>=work-start or terminal['error'] is not None or accepted is None):
            raise ValueError('preparation cannot accept partial/late stages')
        received=load(root/'received.json',required_hash(terminal['receipt_sha256']),limit=2**20)
        candidate=load(root/'prepared.json',required_hash(terminal['candidate_sha256']),limit=2**20)
        if (received!=accepted or received['candidate']!=candidate or
            received['candidate_sha256']!=terminal['candidate_sha256']):
            raise ValueError('actual parent-anchored reception')
        if recheck:
            deadline=time.monotonic()+300
            rebuilt=receive(root,spec,inv,deadline,terminal['candidate_sha256'])
            if rebuilt!=accepted: raise ValueError('independent source preparation recheck')
    elif accepted is not None: raise ValueError('failed preparation contains accepted identity')
    return {'schema':'H2_CAPACITY_PREP_AUDIT_V1','status':expected_status,'identity_prepared':prepared,
        'case':spec['case'],'control':spec['control'],'budget_seconds':budget,'api_seconds':elapsed,
        'stage_seconds':total,'parent_and_publication_seconds':elapsed-total,
        'sampled_peak_rss':max((s['sampled_peak_rss'] for s in stages),default=0),
        'terminal_sha256':finish['terminal_sha256'],'invocation_sha256':terminal['invocation_sha256'],
        'source_identity':accepted if prepared else None,'proof_status':'NOT_A_PROOF',
        'source_rechecked':bool(recheck and terminal['status']=='IDENTITY_PREPARED'),
        'real_capacity_admitted':False,'includes':'creation, imports, capture, validation, chunking, source recheck, reception, cleanup, final publication',
        'excludes':'later administrative archival/audit; preparation is separately charged from future verification'}
