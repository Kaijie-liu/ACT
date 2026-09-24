"""No solver: audit every registered terminal, including not-started entries."""
import argparse
import json
from pathlib import Path
import time
from scoped_proof.io import load, sha
from source_enclosure.format import identity
from scoped_parse_proof.audit import audit


def review(root):
    root=Path(root);started=time.monotonic();launch=load(root/'launch.json');cfg=launch['config']
    if identity(cfg)!=launch['config_sha256'] or cfg['output']!=str(root):
        raise ValueError('batch configuration identity')
    execution=load(root/'execution.json')
    if execution['config_sha256']!=launch['config_sha256'] or execution['no_retries'] is not True:
        raise ValueError('batch identity/retry policy')
    if [r['id'] for r in execution['rows']]!=[c['id'] for c in cfg['calls']]:
        raise ValueError('missing/duplicated/reordered terminal')
    rows=[];signatures=[]
    for call,row in zip(cfg['calls'],execution['rows']):
        if load(root/(call['id']+'_batch_terminal.json'))!=row:raise ValueError('batch terminal identity')
        path=root/call['id']
        if row['status']=='NOT_STARTED_RESOURCE':
            if row['launched'] or row['seconds'] is not None or path.exists():raise ValueError('not-started identity/cost')
            rows.append({'id':call['id'],'status':row['status'],'cost_seconds':None});continue
        if row['status']=='SUPERVISOR_ERROR':
            raise ValueError('incomplete supervisor receipt: preserve but cannot pass full cost audit')
        spec=load(path/'spec.json')
        if spec!={**cfg['common'],'construction_policy':call['construction_policy']}:
            raise ValueError('different request or arm')
        checked=audit(path)
        if row['status']!=checked['effective_status'] or not row['launched'] or row['seconds']<checked['end_to_end_seconds']:
            raise ValueError('request/outer terminal mismatch')
        complete=checked['construction_receipt_checked']
        if complete:signatures.append((sha(path/'source.json'),sha(path/'construction.json')))
        rows.append({'id':call['id'],'status':row['status'],'cost_seconds':row['seconds'],
            'audit':checked, 'files':{str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}})
    if len(set(signatures))>1:raise ValueError('arms built different sources/matrices')
    return {'audit':'PASS','issues':0,'rows':rows,'required_terminals':len(cfg['calls']),
        'complete_constructions_compared':len(signatures),
        'all_constructions_identical':len(signatures)==len(cfg['calls']) and len(set(signatures))==1,
        'observed_call_seconds':sum(r['cost_seconds'] for r in rows if r['cost_seconds'] is not None),
        'missing_costs':sum(r['cost_seconds'] is None for r in rows),
        'separate_audit_seconds':time.monotonic()-started,'new_solver_calls':0,
        'scope':'same-object execution/cost and exact evidence recheck, not native floating-point proof'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(review(a.root)))
