"""Keep both original calls and all original duties in the terminal denominator."""
import argparse
import json
import math
from pathlib import Path
import time

from residual_proof.audit import audit
from scoped_proof.io import load, sha
from scoped_proof.evidence import POSITIVE
from source_enclosure.format import identity


def review(root):
    root=Path(root); started=time.monotonic()
    launch=load(root/'launch.json'); cfg=launch['config']
    if identity(cfg)!=launch['config_sha256'] or cfg['output']!=str(root):
        raise ValueError('batch identity')
    execution=load(root/'execution.json')
    if (execution['config_sha256']!=launch['config_sha256'] or execution['no_retries'] is not True or
            [r['id'] for r in execution['rows']] != [c['id'] for c in cfg['calls']]):
        raise ValueError('missing/duplicate/reordered terminal')
    if [c['id'] for c in cfg['calls']] != ['pairwise','shared']:
        raise ValueError('frozen two-arm roster')
    rows=[]; signatures={}; missing=0; issues=[]
    for call,row in zip(cfg['calls'],execution['rows']):
        if load(root/(call['id']+'_batch_terminal.json'))!=row: raise ValueError('changed batch terminal')
        if row['launched'] is False:
            if row['status']!='NOT_STARTED_RESOURCE' or row['seconds'] is not None or (root/call['id']).exists():
                raise ValueError('invalid not-started status')
            missing+=1; rows.append({'id':call['id'],'status':row['status'],'seconds':None,'positive':False}); continue
        if row['launched'] is not True:raise ValueError('invalid launch flag')
        if row['status']=='SUPERVISOR_ERROR':
            if not isinstance(row['seconds'],(int,float)) or not math.isfinite(row['seconds']) or row['seconds']<0:
                raise ValueError('missing supervisor exception time')
            issues.append(call['id']+': missing completed supervisor cost/proof receipt')
            rows.append({'id':call['id'],'status':row['status'],'seconds':row['seconds'],'positive':False})
            continue
        folder=root/call['id']
        if load(folder/'spec.json')!={**cfg['common'],'proof_policy':call['proof_policy']}:
            raise ValueError('call policy/scope changed')
        checked=audit(folder)
        inv=load(folder/'invocation.json')
        if (inv['budget_seconds']!=cfg['common']['limits']['whole_pipeline_seconds'] or
                inv['rss_limit']!=cfg['common']['limits']['sampled_group_rss_bytes']):
            raise ValueError('actual resource contract differs')
        if checked['effective_status']!=row['status'] or not isinstance(row['seconds'],(int,float)) or row['seconds']<checked['end_to_end_seconds']:
            raise ValueError('call status/cost mismatch')
        rows.append({'id':call['id'],'status':row['status'],'seconds':row['seconds'],
                     'positive':row['status']==POSITIVE,'audit':checked})
        if (folder/'source.json').exists(): signatures[call['id']]=sha(folder/'source.json')
    comparable=len(signatures)==len(cfg['calls'])
    if comparable and len(set(signatures.values()))!=1: raise ValueError('same object source differs')
    matrices=None
    if all((root/c['id']/'construction.json').exists() for c in cfg['calls']):
        old=load(root/'pairwise/construction.json');new=load(root/'shared/construction.json')
        if [p['pair'] for p in old['pairs']]!=[p['pair'] for p in new['pairs']]:raise ValueError('retained route sets differ')
        by_pair={tuple(p['pair']):identity(p) for p in old['pairs']}
        if any(by_pair.get(tuple(p['pair']))!=identity(p) for p in new['pairs']):
            raise ValueError('retained matrix/guard identity differs')
        matrices={'compared_retained_pairs':len(new['pairs']),'all_identical':True}
    positives={r['id'] for r in rows if r['positive']}
    return {'audit':'INCOMPLETE' if issues else 'PASS','issues':len(issues),'issue_details':issues,
        'required_terminals':len(cfg['calls']),'rows':rows,
        'missing_costs':missing,'observed_call_seconds':sum(r['seconds'] for r in rows if r['seconds'] is not None),
        'same_source_verified':comparable,'source_signatures':signatures,
        'retained_matrix_differential':matrices,
        'shared_only_positive':'shared' in positives and 'pairwise' not in positives,
        'pairwise_only_positive':'pairwise' in positives and 'shared' not in positives,
        'new_solves':0,'route_changing_claim':False,'native_float_claim':False,
        'separate_audit_seconds':time.monotonic()-started}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(review(a.root.resolve())))
