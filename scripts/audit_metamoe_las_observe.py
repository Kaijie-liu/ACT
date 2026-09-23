"""Independent saved-only review of the bounded schema failure observation."""
import argparse
import json
from pathlib import Path
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import metamoe_las_diagnostic as control
from audit_metamoe_current_assignment import inventory
from audit_metamoe_csr_paired_r4 import check_candidate
from audit_metamoe_checked_paired import check_author
from metamoe_csr_paired_r4 import collect_terminal


def audit():
    start=time.monotonic();path=control.CONFIG;cfg=json.loads(path.read_text());control.validate(cfg)
    root=Path(cfg['output_root']);folder=root/'mnist_7_author'
    s=json.loads((root/'summary.json').read_text());r=s['receipt']
    result=json.loads((folder/'result.json').read_text())
    if (s['config_sha256']!=sha256(path) or s['terminal']!=collect_terminal(folder,r) or
            s['terminal']['status']!='ERROR' or r['status']!='COMPLETED' or
            r['deadline_seconds']!=300 or r['execution_including_preflight_seconds']>=300 or
            s['batch_seconds']<r['total_with_postflight_seconds']):
        raise ValueError('terminal or cost mismatch')
    if r!=json.loads((folder/'receipt.json').read_text()):raise ValueError('receipt mismatch')
    for name in ('stdout','stderr'):
        if r[name+'_sha256']!=sha256(folder/f'{name}.txt'):raise ValueError('stream drift')
    check_candidate(result,'author');check_author(folder,cfg,cfg['requests'][0],result)
    old=Path(json.loads(control.PARENT.read_text())['output_root'])/'mnist_7_author'
    if (folder/'request.vnnlib').read_bytes()!=(old/'request.vnnlib').read_bytes():
        raise ValueError('changed physical box or property')
    launch=json.loads((folder/'backend_launch.json').read_text())
    if launch['config_sha256']!=sha256(path) or not launch['observation_only']:
        raise ValueError('observation identity')
    events=[json.loads(l) for l in (folder/'las_events.jsonl').read_text().splitlines()]
    initial=next(e for e in events if e['event']=='DOMAIN_INIT')
    last=next(e for e in reversed(events) if e['event']=='DOMAIN_ADD')
    missing=sorted(set(last['stored_keys'])-set(last['returned_lAs']))
    if (set(last['stored_keys'])!=set(initial['lAs']) or not missing or
        any(initial['lAs'][k]['nonzero']!=0 for k in missing) or
        any(not n['lA_none'] for n in last['nodes'] if n['name'] in missing) or
        'assert len(self.all_lAs) == len(bounds[\'lAs\'])' not in (folder/'backend.stderr').read_text()):
        raise ValueError('failed schema diagnosis not reproduced')
    return {'audit':'PASS','issues':0,'config_sha256':sha256(path),'summary_sha256':sha256(root/'summary.json'),
        'initial_keys':sorted(initial['lAs']),'returned_keys':sorted(last['returned_lAs']),
        'missing_initially_zero_keys':missing,'C':last['C'],'events':events,
        'result':result,'receipt':r,'batch_seconds':s['batch_seconds'],'files':inventory(root),
        'audit_seconds':time.monotonic()-start,'new_solves':0,'automatic_followup':False,
        'scope':'Reproduced metadata schema mismatch; not an independent bound/source proof.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();v=audit();write(a.output,v);print(v['audit'],v['missing_initially_zero_keys'])
