"""Re-read parent anchors, lifetime decisions and exact mathematical results.

This is an execution/evidence audit, not independent verification of operating
system process inventories or physical CUDA driver correctness.
"""
from pathlib import Path
import time

from scoped_proof.io import PYTHON, load, sha
from scripts import hz_device_lifecycle as life
from scripts.hz_propagation_supervised import finite, required_hash


def audit(root, *, observation=None, recheck=False):
    root=Path(root); inv_sha=sha(root/'invocation.json'); inv=life.bind(root,inv_sha)
    finish=load(root/'finish.json'); term=load(root/'terminal.json',required_hash(finish['terminal_sha256']))
    names={p.name for p in root.iterdir() if p.is_file()}-{'terminal.json','finish.json'}
    if (term['invocation_sha256']!=inv_sha or set(term['inventory'])!=names
            or any(sha(root/name)!=digest for name,digest in term['inventory'].items())):
        raise ValueError('complete parent inventory including failed prefixes')
    for value in (term['seconds'],term['stage_seconds'],term['parent_seconds'],finish['seconds']): finite(value)
    stages=term['stages']; phases=[s['phase'] for s in stages]
    if not stages or phases!=list(life.PHASES[:len(stages)]): raise ValueError('ordered phase prefix')
    if {p.name for p in root.glob('*_stage.json')}!={phase+'_stage.json' for phase in phases}:
        raise ValueError('phase receipt completeness')
    previous=0.; anchors={}
    for i,s in enumerate(stages):
        phase=s['phase']; inv2,plan=life.validate_plan(root,inv_sha,s['plan_sha256'])
        if inv2!=inv or load(root/(phase+'_stage.json'))!=s or plan['phase']!=phase:
            raise ValueError('parent stage/plan binding')
        for field in ('seconds','execution_seconds','cleanup_seconds','start_seconds','end_seconds','sampled_peak_rss'):
            finite(s[field])
        if (s['start_seconds']<previous or s['end_seconds']<s['start_seconds']+s['seconds']-1e-8
                or abs(plan['begin']-inv['start']-s['start_seconds'])>1e-8
                or abs(s['seconds']-s['execution_seconds']-s['cleanup_seconds'])>1e-8
                or s['run_deadline']!=plan['run_deadline'] or s['cleanup_deadline']!=plan['cleanup_deadline']
                or s['cpu_threads']!=1 or s['cuda_visible_devices']!='' or s['executable']!=PYTHON
                or s['escaped_descendants_or_driver_cleanup'] is not False):
            raise ValueError('charged execution/environment contract')
        expected=({} if phase=='admit' else {'admit':anchors['admit']} if phase=='produce' else
                  {'produce':sha(root/'produce_stage.json')} if phase=='release' else
                  {p:anchors[p] for p in ('produce','release')+ (('check',) if phase=='receive' else ())})
        if plan['inputs']!=expected: raise ValueError('parent launch inputs')
        if 'output_sha256' in s:
            load(root/(phase+'.json'),required_hash(s['output_sha256']),life.LIMIT)
            anchors[phase]=s['output_sha256']
        if s['status']=='COMPLETED':
            finite(s['exit_observed_at'])
            if (s['cleanup_status']!='LEADER_REAPED_NO_LIVE_GROUP' or s['remaining_group']['live']
                    or s['descendant_on_leader_exit'] is not False or type(s['pid']) is not int or s['pid']<=0
                    or not plan['begin']<=s['exit_observed_at']<plan['run_deadline']
                    or inv['start']+s['end_seconds']>=plan['cleanup_deadline']
                    or s['returncode']!=0 or s['sampled_peak_rss']>inv['rss_limit']):
                raise ValueError('unresolved successful phase')
            required_hash(s.get('output_sha256'))
        elif i!=len(stages)-1 and not (phase=='produce' and phases[i+1:]==['release']):
            raise ValueError('continued work after failure')
        stub=s.get('cleanup_unconfirmed_stub',False)
        if stub and not (phase=='produce' and inv['spec']['control']=='cleanup_unconfirmed'):
            raise ValueError('unexpected cleanup simulation')
        actual=dict(s); actual.pop('cleanup_unconfirmed_stub',None)
        if s['cleanup_status']!='CLEANUP_INCOMPLETE' and not life.cleanup_confirmed(actual):
            raise ValueError('contradictory cleanup record, including failed producer')
        if s['cleanup_status']=='CLEANUP_INCOMPLETE' or stub:
            if i!=len(stages)-1: raise ValueError('continued after pending cleanup')
        previous=s['end_seconds']
    if (term['seconds']<previous or finish['seconds']<term['seconds']
            or abs(term['stage_seconds']-sum(s['seconds'] for s in stages))>1e-8
            or abs(term['seconds']-term['stage_seconds']-term['parent_seconds'])>1e-8):
        raise ValueError('whole cost conservation')
    status,release,accepted=life.decision(root,inv,stages)
    status=life.finalize_status(status,inv['start']+term['seconds'],inv['work'])
    if (term['status']!=status or term['release']!=release
            or term['accepted']!=(accepted if status==life.DONE else None)):
        raise ValueError('terminal decision/acceptance')
    final_status=life.finalize_status(status,inv['start']+finish['seconds'],inv['deadline'])
    if finish['status']!=final_status: raise ValueError('publication deadline')
    bounds=0
    if status==life.DONE and recheck:
        recomputed=life.mathematical_check(root,inv,anchors['produce'],time.monotonic()+30)
        if recomputed!=load(root/'check.json',anchors['check'],life.LIMIT): raise ValueError('exact bound recheck')
        bounds=len(recomputed['result']['results'])
    reported=None
    if observation is not None:
        r=observation['result']
        for v in (observation['begin'],observation['end'],r['seconds']): finite(v)
        if (r['invocation_sha256']!=inv_sha or r['finish_sha256']!=sha(root/'finish.json')
                or observation['begin']>inv['start'] or observation['end']-inv['start']<r['seconds']
                or r['seconds']<finish['seconds'] or r['physical_cuda'] is not False
                or r['complete_moe_proof'] is not False):
            raise ValueError('externally anchored API cost/identity')
        reported=life.finalize_status(final_status,observation['end'],inv['deadline'])
        if r['status']!=reported: raise ValueError('late API acceptance')
    return {'status':'PASS','execution_status':reported,'budget_acceptance_observed':reported==life.DONE,
            'bounds_rechecked':bounds,'release':release,'physical_cuda':False,'complete_moe_proof':False}
