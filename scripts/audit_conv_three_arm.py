"""Independent E4 terminal audit; does not independently re-prove HZ/CROWN bounds."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from act.pipeline.moe.experiment1 import _sha256
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.schedule_confirmation import inspect_row, expected_identity
from act.pipeline.moe.common_fact_snapshot import check_snapshot, fact_view
from scripts.conv_three_arm_contract import (ARMS, ROOT, selection, read, request_for,
    wrapper_hashes, SELECTION_HASH, PROTOCOL_HASH, gate)


def route_inventory(record):
    universe = list(itertools.combinations(range(4), 2))
    groups = [record[k] for k in ('feasible', 'infeasible', 'unresolved')]
    if sorted(tuple(p) for g in groups for p in g) != universe:
        raise ValueError('missing/duplicate/illegal E4 route')
    branches = record['branches']
    if len(branches) != 6 or {tuple(b['route_set']) for b in branches} != set(universe):
        raise ValueError('E4 route branch inventory mismatch')
    for kind, group in zip(('feasible', 'infeasible', 'unresolved'), groups):
        for pair in group:
            state = next(b for b in branches if b['route_set']==pair)['feasibility']
            if (kind == 'unresolved' and state in ('feasible', 'infeasible')) or (kind != 'unresolved' and state != kind):
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
    if (not env['python'].startswith('3.11.16 ') or env['torch']!='2.11.0+cu130' or
        env['device']!='cpu' or env['dtype']!='float64' or env['threads']!=1 or
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


def terminal_contract(row):
    if (row['budget_seconds'] != 300 or not math.isfinite(row['wall_seconds'])
            or row['wall_seconds'] < 0 or (row['wall_seconds'] > 300 and not row['outer_timeout'])):
        raise ValueError('invalid deadline accounting')
    if row['outer_timeout'] and (row['status'] != 'TIMEOUT' or row['package'] is not None):
        raise ValueError('late result promoted')
    if row['status'] == 'ERROR' and row['package'] is not None:
        raise ValueError('error promoted to package')
    if not row['outer_timeout'] and row['status'] != 'ERROR' and row['return_code'] != 0:
        raise ValueError('failed worker promoted')


def audit(root):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT/'data/moe/results'):
        raise ValueError('invalid run root')
    runtime = read(root/'runtime.json'); value = selection()
    if (runtime['schema'] != 'conv_smoke_execution_v1' or runtime['smoke'] is not True
            or runtime['full_authorized'] is not False or runtime['selection'] != value
            or runtime['wrapper_sha256'] != wrapper_hashes()
            or runtime['selection_sha256'] != SELECTION_HASH or runtime['protocol_sha256'] != PROTOCOL_HASH
            or runtime['config']['methods'] != value['identities']['method_configs']):
        raise ValueError('execution identity drift')
    # Pin actual materialized values independently of the worker status, including
    # requests that die before any model or tensor check executes.
    import torch
    from act.pipeline.moe.external_pair_worker import load
    for job in value['smoke_jobs'][::3]:
        load(request_for(value, job, runtime['git_head']))
    rows = [json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()] if (root/'rows.jsonl').exists() else []
    expected = value['smoke_jobs']; end = read(root/'run_terminal.json')
    if (len(rows) > 6 or any(any(r.get(k) != j[k] for k in j) for r,j in zip(rows, expected))
            or end['completed_job_ids'] != [r['job_id'] for r in rows]
            or end['unattempted'] != expected[len(rows):] or end['full_started'] is not False):
        raise ValueError('terminal roster incomplete/reordered/hidden')
    if end['state'] not in ('EXECUTION_COMPLETED','EXECUTION_ERROR'):
        raise ValueError('invalid run terminal')
    if end['state']=='EXECUTION_COMPLETED' and (len(rows)!=6 or any(r['status']=='ERROR' for r in rows)):
        raise ValueError('incomplete run called complete')
    if end['state']=='EXECUTION_ERROR' and not end['error']:
        raise ValueError('missing failure explanation')
    if any(r['status']=='ERROR' for r in rows[:-1]):
        raise ValueError('execution continued after ERROR')
    configs={a:read(v['path']) for a,v in value['identities']['method_configs'].items()}
    # Adapt ONLY the generic audit's subject lookup, not its result or semantics.
    generic_selection={**value, 'models':{'conv':value['subject']}, 'request':{'epsilon':2/255}}
    complete={a:0 for a in ARMS}; details={}; replays=0
    for row in rows:
        directory=root/row['job_id']; request=request_for(value,row,runtime['git_head'])
        if read(directory/'terminal.json')!=row or _sha256(directory/'request.json')!=row['request_sha256'] or read(directory/'request.json')!=request:
            raise ValueError('terminal/request binding differs')
        terminal_contract(row)
        expected_level='CROWN_NUMERICAL_FILTER' if row['method']=='crown' else 'HZ_POLICY_ACCEPTED'
        if row['evidence_level']!=expected_level:
            raise ValueError('evidence level changed')
        wait=row['resource_wait']; state=wait['at_launch']
        if (not math.isfinite(wait['seconds']) or not 0<=wait['seconds']<=86400
                or state['available_ram_gib']<16 or state['free_disk_gib']<5 or not 0<=state['load_per_core']<=.5):
            raise ValueError('resource gate violated')
        for filename,key in [('common_facts.json','snapshot_sha256'),('routes.json','routes_sha256'),('external.json','external_sha256')]:
            path=directory/filename
            if path.exists()!=(row.get(key) is not None) or (path.exists() and _sha256(path)!=row[key]):
                raise ValueError('omitted or changed partial artifact')
        facts=None
        if row['status']=='ERROR':
            if row['return_code']==0 and not row.get('error'):
                raise ValueError('unexplained execution error')
            if row.get('snapshot_sha256'):
                snap=read(directory/'common_facts.json'); cfg=configs[row['method']]
                identity=expected_identity(generic_selection,{**row,'model':'conv'},cfg,True)
                check_snapshot(snap,expected_identity=identity,expected_config=cfg)
                if snap['payload']['completion_elapsed_seconds']>row['wall_seconds']:
                    raise ValueError('snapshot after error termination')
                facts=fact_view(snap)
            if row.get('routes_sha256'):
                route=read(directory/'routes.json')
                if route['request']!=request:
                    raise ValueError('partial external request mismatch')
                route_inventory(route['routes'])
            d={'complete':False,'pairs':facts['pairs'] if facts else None,'replayed':False}
        elif row['method']=='crown':
            d=external_result(root,row,request)
        else:
            checked=inspect_row(root,{**row,'model':'conv'},runtime,generic_selection,configs)
            facts=checked['facts']; pairs=facts['pairs'] if facts else None
            if row['package']:
                evidence=read(Path(row['package'])/'evidence.json')
                coverage=evidence['route_coverage']
                if any(i not in range(4) for i in (coverage.get('candidate_experts') or [])):
                    raise ValueError('wrong E4 candidates')
                if coverage['route_sets_exact']:
                    pairs=coverage['feasible_route_sets']
            if pairs is not None and any(p not in value['possible_pairs'] for p in pairs):
                raise ValueError('wrong E4 feasible pair')
            d={'complete':checked['package'],'pairs':pairs,'replayed':checked['replayed']}
        complete[row['method']]+=int(d['complete']); replays+=int(d['replayed'])
        details[row['rank'],row['method']]={**d,'facts':facts}
    equal=unavailable=0; strata=[]
    for rank,sample in enumerate(value['smoke_samples']):
        observed=[r for r in rows if r['rank']==rank]
        statuses={r['status'] for r in observed}
        if 'UNSAFE' in statuses and statuses & {'SAFE','POSITIVE'}:
            raise ValueError('whole-model witness/positive conflict')
        pairs=[details[rank,r['method']]['pairs'] for r in observed if details[rank,r['method']]['pairs'] is not None]
        if pairs and any(p!=pairs[0] for p in pairs):
            raise ValueError('complete route inventories disagree')
        a,b=[details.get((rank,arm),{}).get('facts') for arm in ('adaptive','monolithic')]
        if a is not None and b is not None:
            if a!=b:raise ValueError('common facts differ')
            equal+=1
        else:unavailable+=1
        strata.append({'rank':rank,'dataset_index':sample['dataset_index'],
                       'pair_count':len(pairs[0]) if pairs else None,
                       'states':{r['method']:r['status'] for r in observed}})
    return {'status':'PASS','issues':[], 'smoke_gate':'PASS' if gate(rows,complete) and end['state']=='EXECUTION_COMPLETED' else 'FAIL',
        'rows':len(rows),'planned_rows':6,'unattempted':end['unattempted'], 'complete_per_arm':complete,
        'unsafe_replayed':replays,'common_fact_pairs_equal':equal,'common_fact_pairs_unavailable':unavailable,
        'methods':{a:{'states':dict(Counter(r['status'] for r in rows if r['method']==a)),
                    'mean_observed_seconds':mean([r['wall_seconds'] for r in rows if r['method']==a]) if any(r['method']==a for r in rows) else None,
                    'evidence_level':'CROWN_NUMERICAL_FILTER' if a=='crown' else 'HZ_POLICY_ACCEPTED'} for a in ARMS},
        'strata':strata,'runtime_sha256':_sha256(root/'runtime.json'),
        'rows_sha256':_sha256(root/'rows.jsonl') if rows else None,
        'run_terminal_sha256':_sha256(root/'run_terminal.json'), 'full_started':False,
        'scope':'Frozen six-request conformance only. Structural audit and full-model witness replay, not independent SAFE reproof. Missing facts are unavailable, not equal. Full90 requires separate authorization.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=audit(args.root)
    except Exception as exc:
        result={'status':'FAIL','issues':[repr(exc)],'smoke_gate':'FAIL','full_started':False}
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('output escapes project')
    if output.exists():raise FileExistsError('audit output already exists; use a new review path')
    atomic_json(output,result);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
