"""Structural/request identity audit and dynamic witness replay, not SAFE proof."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
from statistics import mean,median
from act.pipeline.moe.experiment1 import _sha256
from act.pipeline.moe.paired_followup import save
from act.pipeline.moe.schedule_confirmation import inspect_row
from act.pipeline.moe.external_pair_comparison import artifacts,jobs,CFG,CFG_HASH,ARMS


def route_inventory(record):
    groups=[record[k] for k in ('feasible','infeasible','unresolved')]
    if sorted(tuple(p) for g in groups for p in g)!=list(itertools.combinations(range(8),2)):
        raise ValueError('missing/duplicate/illegal route')
    branches=record['branches']
    if len(branches)!=28 or {tuple(b['route_set']) for b in branches}!=set(itertools.combinations(range(8),2)):
        raise ValueError('route branch inventory mismatch')
    for kind,group in zip(('feasible','infeasible','unresolved'),groups):
        for pair in group:
            branch=next(b for b in branches if b['route_set']==pair)
            state=branch['feasibility']
            if (kind=='unresolved' and state in ('feasible','infeasible')) or (kind!='unresolved' and state!=kind):
                raise ValueError('route feasibility metadata mismatch')
    return bool(record['exact'] and not record['unresolved'] and record['feasible'])


def external_result(root,row,request):
    directory=root/row['job_id']
    for name,key in [('routes.json','routes_sha256'),('external.json','external_sha256')]:
        if (directory/name).exists()!=(key in row):raise ValueError('omitted external artifact')
        if key in row and _sha256(directory/name)!=row[key]:raise ValueError('external artifact hash changed')
    route=json.loads((directory/'routes.json').read_text()) if 'routes_sha256' in row else None
    complete=False
    if route is not None:
        if route['request']!=request:raise ValueError('route request differs')
        complete=route_inventory(route['routes'])
    if row['outer_timeout']:
        if row['status']!='TIMEOUT' or row['package'] is not None:raise ValueError('late result promoted')
        return {'complete':False,'pairs':route['routes']['feasible'] if complete else None,'replayed':False}
    if row['return_code']!=0 or row['status'] not in ('POSITIVE','UNSAFE','UNKNOWN'):raise ValueError('external worker did not finish')
    result=json.loads((directory/'external.json').read_text())
    if result['status']!=row['status']:raise ValueError('external terminal mismatch')
    if not complete:
        if row['status']!='UNKNOWN' or result['reason']!='INCOMPLETE_ROUTE_COVERAGE':raise ValueError('incomplete enumeration promoted')
        return {'complete':True,'pairs':None,'replayed':False}
    from act.pipeline.moe.check_request_lp import property_row
    if result['model_state']!=request['subject']['model_state'] or result['tensor_identity']!={k:request['sample'][k] for k in ('center','lower','upper')}:
        raise ValueError('cross-environment tensor/model mismatch')
    if result['C']!=[[property_row(10,request['sample']['label'],i) for i in range(9)]]:raise ValueError('property lowering mismatch')
    env=result['environment']
    from act.pipeline.moe.external_compatibility import TOOL
    if (env['device']!='cpu' or env['dtype']!='float64' or env['threads']!=1 or
        not Path(env['auto_lirpa_file']).resolve().is_relative_to(TOOL/'auto_LiRPA') or
        result['backend']!={'method':'CROWN','bound_opts':{'conv_mode':'matrix'}} or result['formal_SAFE'] is not False):
        raise ValueError('backend semantics drift')
    if row['status']=='UNSAFE':
        import torch
        from act.pipeline.moe.external_pair_worker import load
        model,tensors=load(request)
        if _sha256(directory/'witness.pt')!=result['witness_sha256']:raise ValueError('witness identity changed')
        x=torch.load(directory/'witness.pt',map_location='cpu',weights_only=True)
        if x.shape!=tensors['center'].shape or x.dtype!=torch.float64 or not torch.isfinite(x).all() or not torch.all((x>=tensors['lower'])&(x<=tensors['upper'])):
            raise ValueError('witness outside represented domain')
        with torch.no_grad():pred=int(model(x).argmax())
        if pred==request['sample']['label']:raise ValueError('not a dynamic-model violation')
        return {'complete':True,'pairs':route['routes']['feasible'],'replayed':True}
    pairs=result['pairs']
    if [p['pair'] for p in pairs]!=route['routes']['feasible']:raise ValueError('missing/duplicate pair bound')
    for pair in pairs:
        if len(pair['lower'])!=9 or len(pair['upper'])!=9:raise ValueError('incomplete output margins')
        if any(not math.isfinite(v) for v in pair['lower']+pair['upper']) or any(l>u for l,u in zip(pair['lower'],pair['upper'])):
            raise ValueError('invalid numerical bounds')
        if not 0<=pair['concrete_max_error']<=1e-10 or not 0<=pair['lowered_max_error']<=1e-10:raise ValueError('conformance failed')
    positive=bool(pairs) and all(v>1e-7 for p in pairs for v in p['lower'])
    if (row['status']=='POSITIVE')!=positive:raise ValueError('positive aggregation mismatch')
    return {'complete':True,'pairs':route['routes']['feasible'],'replayed':False}


def audit(root):
    root=Path(root).resolve();runtime=json.loads((root/'runtime.json').read_text());selection=artifacts()
    if selection!=runtime['selection'] or _sha256(Path(runtime['config_path']))!=runtime['config_sha256']:raise ValueError('selection/config drift')
    configs={'adaptive':json.loads(CFG.read_text())}
    if runtime['config']['methods']['adaptive']!={'path':str(CFG),'sha256':CFG_HASH}:raise ValueError('method identity changed')
    rows=[json.loads(s) for s in (root/'rows.jsonl').read_text().splitlines()];expected=jobs(selection,runtime['smoke'])
    if len(rows)!=len(expected) or any(any(r[k]!=j[k] for k in j) for r,j in zip(rows,expected)):raise ValueError('schedule incomplete/reordered')
    details={};complete={a:0 for a in ARMS};replayed=0
    import torch
    from act.pipeline.moe.staged_verifier import _tensor_identity
    for rank,sample in enumerate(selection['smoke_samples' if runtime['smoke'] else 'samples']):
        ref=runtime['tensors'][str(rank)];path=Path(ref['path']).resolve()
        if not path.is_relative_to(root/'inputs') or _sha256(path)!=ref['sha256']:raise ValueError('input artifact drift')
        tensors=torch.load(path,map_location='cpu',weights_only=True)
        if set(tensors)!={'center','lower','upper'} or any(_tensor_identity(t)!=sample[k] for k,t in tensors.items()):
            raise ValueError('materialized input differs from frozen selection')
    for row in rows:
        directory=root/row['job_id'];sample=selection['smoke_samples' if runtime['smoke'] else 'samples'][row['rank']]
        expected_request={'subject':selection['models'][row['model']],'sample':sample,'epsilon':selection['request']['epsilon'],
            'method':row['method'],'tensors':runtime['tensors'][str(row['rank'])],'head':runtime['git_head'],
            'config':{'path':str(CFG),'sha256':CFG_HASH}}
        if json.loads((directory/'terminal.json').read_text())!=row or _sha256(directory/'request.json')!=row['request_sha256']:
            raise ValueError('terminal/request hash mismatch')
        request=json.loads((directory/'request.json').read_text())
        if request!=expected_request or _sha256(Path(request['tensors']['path']))!=request['tensors']['sha256']:raise ValueError('frozen request drift')
        if row['budget_seconds']!=300 or not math.isfinite(row['wall_seconds']) or row['wall_seconds']<0 or (row['wall_seconds']>300 and not row['outer_timeout']):
            raise ValueError('invalid timing/deadline accounting')
        level='HZ_POLICY_ACCEPTED' if row['method']=='adaptive' else 'CROWN_NUMERICAL_FILTER'
        if row['evidence_level']!=level:raise ValueError('evidence level changed')
        if row['method']=='adaptive':
            detail=inspect_row(root,row,runtime,selection,configs)
            pairs=None
            if row['package']:
                e=json.loads((Path(row['package'])/'evidence.json').read_text())
                if e['route_coverage']['route_sets_exact']:pairs=e['route_coverage']['feasible_route_sets']
            d={'complete':detail['package'],'pairs':pairs,'replayed':detail['replayed']}
        else:d=external_result(root,row,request)
        complete[row['method']]+=int(d['complete']);replayed+=int(d['replayed']);details[row['model'],row['rank'],row['method']]=d
    by={(r['model'],r['rank'],r['method']):r for r in rows};n=1 if runtime['smoke'] else 10;models={}
    for model in sorted(selection['models']):
        methods={}
        for arm in ARMS:
            values=[by[model,i,arm] for i in range(n)]
            methods[arm]={'states':dict(Counter(r['status'] for r in values)),
                'evidence_level':values[0]['evidence_level'],'mean_observed_seconds':mean(r['wall_seconds'] for r in values)}
        contrasts={}
        for title,accepted in [('positive',{'SAFE','POSITIVE'}),('decided',{'SAFE','POSITIVE','UNSAFE'})]:
            a={i for i in range(n) if by[model,i,'adaptive']['status'] in accepted};b={i for i in range(n) if by[model,i,'crown']['status'] in accepted}
            contrasts[title]={'intersection':sorted(a&b),'adaptive_only':sorted(a-b),'crown_only':sorted(b-a)}
        strata=[]
        for i in range(n):
            a,b=[details[model,i,arm]['pairs'] for arm in ARMS]
            if a is not None and b is not None and a!=b:raise ValueError('complete route inventories disagree')
            pairs=a if a is not None else b
            statuses={by[model,i,arm]['status'] for arm in ARMS}
            if 'UNSAFE' in statuses and statuses & {'SAFE','POSITIVE'}:raise ValueError('positive/dynamic-witness conflict')
            strata.append({'rank':i,'dataset_index':by[model,i,'adaptive']['dataset_index'],'pair_count':len(pairs) if pairs is not None else None,
                           'adaptive':by[model,i,'adaptive']['status'],'crown':by[model,i,'crown']['status']})
        delta=[by[model,i,'adaptive']['wall_seconds']-by[model,i,'crown']['wall_seconds'] for i in range(n)]
        models[model]={'methods':methods,'contrasts':contrasts,'rows':strata,'mean_paired_seconds':mean(delta),'median_paired_seconds':median(delta)}
    return {'status':'PASS','issues':[],'rows':len(rows),'complete_per_arm':complete,'unsafe_replayed':replayed,'models':models,
            'runtime_sha256':_sha256(root/'runtime.json'),'rows_sha256':_sha256(root/'rows.jsonl'),
            'scope':'Observed-cohort hybrid external path, complete request costs. Numerical positive filters differ from HZ-policy SAFE; no independent SAFE reproof or standalone complete alpha-beta-CROWN comparison.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root);save(a.output,result);print(json.dumps(result,indent=2))
