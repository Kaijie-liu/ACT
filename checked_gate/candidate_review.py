"""Fresh exact proof recheck and independent lifecycle/cost audit, no optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from checked_gate.bootstrap import ROOT, setup


def review(root):
    start=time.monotonic()
    setup(checker=True)
    # IO/identity helpers and the check_saved adapter do not call work/propose.
    from checked_gate.candidate_worker import load, check_saved
    freeze=load(ROOT/'docs/checked_gate_candidate_v2_freeze.json')
    execution=load(root/'execution.json'); terminal=load(root/'terminal.json')
    pub=load(root/'publication.json')
    for name,digest in freeze['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen source identity drift')
    if execution['freeze_sha256']!=hashlib.sha256((ROOT/'docs/checked_gate_candidate_v2_freeze.json').read_bytes()).hexdigest():
        raise ValueError('protocol identity changed')
    if pub['terminal_sha256']!=hashlib.sha256((root/'terminal.json').read_bytes()).hexdigest():
        raise ValueError('terminal changed')
    if (execution['request_id']!=freeze['request_id'] or
            (execution['total_seconds'],terminal['total_seconds'])!=(300,300)):
        raise ValueError('request/budget mismatch')
    for name,meta in terminal['artifacts'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError('artifact escape')
        raw=p.read_bytes()
        if len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:
            raise ValueError('partial/final artifact changed')
    # Exact inventory: no successful-looking artifact may be silently omitted.
    expected={p.name for p in root.iterdir() if p.is_file()}-{
        'terminal.json','publication.json','publication_timeout.json','review.json'}
    if set(terminal['artifacts'])!=expected: raise ValueError('terminal inventory incomplete')
    stages=terminal['stages']; previous=0.; stopped=False
    caps={'validate':80,'propose':120,'check':80}
    for i,s in enumerate(stages):
        if i>=3 or s['phase']!=('validate','propose','check')[i] or stopped:
            raise ValueError('stage ordering/retry')
        if (s!=load(root/(s['phase']+'_stage.json')) or s['cap_seconds']!=caps[s['phase']] or
                not previous<=s['start_seconds']<=s['end_seconds']<=terminal['elapsed_seconds_before_terminal'] or
                not 0<=s['seconds']<=s['end_seconds']-s['start_seconds']+.01):
            raise ValueError('phase costs/identity inconsistent')
        if s['state']=='COMPLETED' and (s['return_code']!=0 or not s['started']):
            raise ValueError('invalid completed phase')
        if s['state'] not in ('COMPLETED','ERROR','TIMEOUT'): raise ValueError('phase state')
        previous=s['end_seconds']; stopped=s['state']!='COMPLETED'
    if not terminal['elapsed_seconds_before_terminal']<=pub['elapsed_seconds']:
        raise ValueError('publication cost missing')
    expired=(pub['elapsed_seconds']>=300 or (root/'publication_timeout.json').exists())
    checked=None
    if expired: expected_status='TIMEOUT'
    elif terminal['error'] is not None: expected_status='ERROR'
    elif len(stages)==3 and not stopped:
        checked=check_saved(root)
        if checked!=load(root/'checked.json'): raise ValueError('fresh mathematical check disagrees')
        expected_status=checked['status']
    else:
        if not stages or not stopped: raise ValueError('missing terminal failure')
        expected_status=stages[-1]['state']
    complete=expected_status=='CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING'
    if not expired and (terminal['status']!=expected_status or terminal['complete_conditional_request']!=complete):
        raise ValueError('terminal scientific status mismatch')
    if terminal['complete_strict_network_certificate'] or terminal['production_verdict_changed']:
        raise ValueError('claim upgrade outside scope')
    costs={phase:None for phase in caps}
    for s in stages:
        costs[s['phase']]={'seconds':s['end_seconds']-s['start_seconds'],
                          'censored':s['state']=='TIMEOUT','state':s['state']}
    return {'status':'PASS','issues':[],'execution_head':execution['head'],
        'terminal_sha256':pub['terminal_sha256'],'outcome':expected_status,
        'complete_conditional_request':complete,'check':checked,
        'candidate_calls':int((root/'proposal_entered.json').exists()),
        'phase_costs':costs,'complete_stored_source_seconds':pub['elapsed_seconds'],
        'unattributed_control_publication_seconds':pub['elapsed_seconds']-sum(
            s['end_seconds']-s['start_seconds'] for s in stages),
        'artifact_count':len(terminal['artifacts']),
        'stored_artifact_bytes':sum(v['bytes'] for v in terminal['artifacts'].values()),
        'postterminal_review_seconds':time.monotonic()-start,
        'scope':'Postselected first-family supplied-source diagnostic; no new model propagation, no high-accuracy or cross-family strict claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root.resolve();result=review(root)
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with (root/'review.json').open('x') as f:f.write(raw)
    print(raw)
