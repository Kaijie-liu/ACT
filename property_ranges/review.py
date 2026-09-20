"""Terminal identity, costs, all row evidence and fresh relocated mathematical check."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from checked_gate.candidate_run import ACT,execute
from router_source.capture import ROOT,sha
from router_source.build import save
from full_bounds.review import strip

ENVIRONMENT={'scipy':'1.16.3','numpy':'2.3.5','bundled_highs':'1.8.0','method':'highs','threads':1}


def audit(root,required=9,budget=300,recheck=True):
    started=time.monotonic();load=lambda n:json.loads((root/n).read_bytes())
    t,p=load('terminal.json'),load('publication.json')
    if t['schema']!='RANGED_SOURCE_TERMINAL_V1' or sha(root/'terminal.json')!=p['terminal_sha256']:
        raise ValueError('terminal identity')
    if t['budget_seconds']!=budget or t['production_verdict_changed'] or t['deployed_floating_point_proof']:
        raise ValueError('contract drift')
    positive=t['status']=='CHECKED_POSITIVE_DECLARED_REAL_MOE'
    if (t['status'] not in ('ERROR','TIMEOUT','CHECKED_POSITIVE_DECLARED_REAL_MOE',
            'UNKNOWN_MISSING_BOUND_EVIDENCE','UNKNOWN_NONPOSITIVE_BOUNDS') or
            t['complete_declared_real_output_proof']!=positive or
            (positive and (t['error'] is not None or t['check'] is None or len(t['stages'])!=4 or
                any(s['state']!='COMPLETED' for s in t['stages'] if s['phase']!='propose') or
                any(s['state']=='ERROR' for s in t['stages']) or not t['check']['complete_declared_real_output_proof']))):
        raise ValueError('terminal positive claim without complete evidence')
    for name,h in t['inventory'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:raise ValueError('raw evidence changed')
    previous=0;names=['build','propose','seal','check'];caps=[100,120,5,300]
    for i,r in enumerate(t['stages']):
        if (i>=4 or r['phase']!=names[i] or r['cap_seconds']!=caps[i] or r!=load(names[i]+'_stage.json') or
                not previous<=r['start_seconds']<=r['end_seconds']<=t['elapsed_before_publication']):
            raise ValueError('phase accounting')
        previous=r['end_seconds']
    late=p['seconds']>=budget or (root/'publication_timeout.json').exists()
    snapshots={}
    from source_enclosure.format import identity
    for path in sorted(root.glob('selection_expert*.json')):
        snap=json.loads(path.read_bytes());record=snap['record']
        if (snap['record_sha256']!=identity(record) or record['arm'] not in ('prefix','property') or
                record['quota']!=2 or record['query_order']!='ascending_row' or
                not 0<=snap['seconds_since_build_start']<=t['elapsed_before_publication']):
            raise ValueError('selection snapshot identity/cost')
        i=int(path.stem.removeprefix('selection_expert'));chosen=record['selected']
        if (len(chosen)!=min(2,len(record['rows'])) or len(set(chosen))!=len(chosen) or chosen!=sorted(chosen)
                or any(type(j)is not int or not 0<=j<len(record['rows']) for j in chosen)):
            raise ValueError('selection quota/duplicates')
        snapshots[i]=snap
    rows=[]
    for entered in sorted((root/'range_queries').glob('*_entered.json')):
        e=json.loads(entered.read_bytes());result=entered.with_name(entered.name.replace('_entered',''))
        if (not 0<e['native_limit_seconds']<=3 or type(e['row']) is not int or e['context']['layer']!='6'
                or e['side'] not in ('lower','negative_upper')):raise ValueError('native range cap/roster')
        matches=[i for i,snap in snapshots.items() if e['context']==snap['record']['context']
                 and e['source_sha256']==snap['record']['source_sha256'] and e['row'] in snap['record']['selected']]
        if len(matches)!=1 or entered.name!=f"e{matches[0]}_r{e['row']}_{e['side']}_entered.json":
            raise ValueError('call outside prepublished selection')
        r=json.loads(result.read_bytes()) if result.exists() else None
        if r is not None and (r['native_limit_seconds']!=e['native_limit_seconds'] or
                (r['candidate'] is not None and r['candidate']['lp_sha256']!=e['lp_sha256'])):
            raise ValueError('range side identity')
        if r is not None and r.get('environment')!=ENVIRONMENT:raise ValueError('range solver environment drift')
        rows.append({'entry':entered.name,'source_sha256':e['source_sha256'],'row':e['row'],'side':e['side'],
            'complete':r is not None,'candidate_available':r is not None and r['candidate'] is not None,
            'native_seconds':r['native_seconds'] if r else None})
    if len(rows)>8:raise ValueError('range call cap, including partial execution')
    bound_calls=[json.loads(p.read_bytes()) for p in sorted(root.glob('entered_*.json'))]
    if bound_calls:
        obligations=load('relocated/source/obligations.json')
        expected=[r['competitor'] for r in obligations['rows']]
        if (len(bound_calls)>required or [r['competitor'] for r in bound_calls]!=expected[:len(bound_calls)] or
                any(r['native_seconds']!=16 for r in bound_calls)):
            raise ValueError('output call roster/limit, including partial execution')
    out={'status':'PASS','issues':[],'outcome':'TIMEOUT' if late else t['status'],'total_seconds':p['seconds'],
        'stages':t['stages'],'range_calls':rows,'range_calls_entered':len(rows),
        'output_calls_entered':len(bound_calls),
        'selection_snapshots':snapshots,'terminal_sha256':p['terminal_sha256'],'production_verdict_changed':False,'deployed_floating_point_proof':False}
    if late or t['error'] or len(t['stages'])!=4 or t['stages'][-1]['state']!='COMPLETED':
        out.update(complete=False,failed_execution_preserved=True);return out
    if not t['elapsed_before_publication']<=p['seconds']<budget:raise ValueError('publication budget')
    g=load('generation.json');job=load('job.json');src=root/'relocated';sm=json.loads((src/'source/manifest.json').read_bytes())
    if (g['arm']!=job['arm'] or sm['policy']!={'selection':job['arm'],'layer':6,'rows_per_expert':2,'native_seconds':3.} or
            g['new_output_obligations']!=required or g['range_native_calls']!=len(rows) or len(rows)>8):
        raise ValueError('arm/row/call coverage')
    trace=json.loads((src/'source/trace.json').read_bytes());accepted={};expected_range=[];range_context={}
    for step in trace['steps']:
        if step['layer']!=6:continue
        range_context[step['expert']]=(step['proof']['context'],step['proof']['source'])
        selected=json.loads((src/'source'/step['selection_file']).read_bytes())
        snap=snapshots.get(step['expert'])
        if snap is None or snap['record']!=selected or selected['source_sha256']!=step['proof']['source'] or selected['context']!=step['proof']['context']:
            raise ValueError('durable selection/source disagreement')
        expected_range.extend((step['expert'],j) for j in selected['selected'])
        for j,fact in enumerate(step['proof']['facts']):
            if fact is not None:
                from source_enclosure.format import identity
                accepted[(step['expert'],j)]=identity(fact)
    if [(r['expert'],r['row']) for r in g['range_rows']]!=expected_range:
        raise ValueError('complete range row inventory')
    if set(snapshots)!={r['expert'] for r in trace['steps']}:raise ValueError('expert selection completeness')
    selected_build_seconds=sum(r['selection_seconds'] for r in g['steps'])
    if selected_build_seconds<0 or selected_build_seconds>g['build_seconds']:
        raise ValueError('selection build cost')
    for path in (root/'range_queries').glob('*_entered.json'):
        e=json.loads(path.read_bytes())
        matches=[i for i,(ctx,h) in range_context.items() if e['context']==ctx and e['source_sha256']==h]
        if len(matches)!=1 or path.name!=f"e{matches[0]}_r{e['row']}_{e['side']}_entered.json":
            raise ValueError('range call/source binding')
    for r in g['range_rows']:
        if r['fact_sha256']!=accepted.get((r['expert'],r['row'])):raise ValueError('accepted range fact accounting')
        path=root/'range_queries'/f"e{r['expert']}_r{r['row']}_complete.json"
        if json.loads(path.read_bytes())!={k:v for k,v in r.items() if k!='expert'}:
            raise ValueError('range completion ledger')
        count=sum(v['entry'].startswith(f"e{r['expert']}_r{r['row']}_") for v in rows)
        if r['calls']!=count:raise ValueError('per-row range call accounting')
    if [(r['context'],r['source_sha256']) for r in g['range_rows']]!=[range_context[r['expert']] for r in g['range_rows']]:
        raise ValueError('range result context/source')
    bound_calls=[json.loads(p.read_bytes()) for p in sorted(root.glob('entered_*.json'))]
    expected=[r['competitor'] for r in json.loads((src/'source/obligations.json').read_bytes())['rows']]
    if [r['competitor'] for r in bound_calls]!=expected[:len(bound_calls)] or any(r['native_seconds']!=16 for r in bound_calls):
        raise ValueError('output call roster/limit')
    environment=load('solver_environment.json') if (root/'solver_environment.json').exists() else None
    if environment is None:
        if bound_calls:raise ValueError('missing output solver environment')
    elif ({k:environment[k] for k in ENVIRONMENT if k!='threads'}!={k:v for k,v in ENVIRONMENT.items() if k!='threads'} or
            environment['options']!={'time_limit':16.,'threads':1}):raise ValueError('output solver environment drift')
    checked=t['check']
    if recheck:
        moved=root/'review_relocated';shutil.copytree(src,moved)
        check=execute([ACT,'-I','-S',str(moved/'verify_bounds.py'),'--manifest-hash',sha(moved/'manifest.json')],
            root/'fresh_review.log',time.monotonic()+298,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if check['state']!='COMPLETED':raise ValueError('fresh relocated check incomplete')
        checked=load('fresh_review.log')
        if strip(checked)!=strip(t['check']):raise ValueError('fresh check disagreement')
    if (checked['required_obligations']!=required or
            checked['positive_bounds']+checked['nonpositive_bounds']+checked['missing_bounds']!=required):
        raise ValueError('all output obligations required')
    valid_positive=checked['complete_declared_real_output_proof'] and t['status']=='CHECKED_POSITIVE_DECLARED_REAL_MOE'
    if t['complete_declared_real_output_proof']!=valid_positive:raise ValueError('positive endpoint conflict')
    out.update(complete=True,result=strip(checked),generation=g,accepted_range_facts=len(accepted),output_calls=len(bound_calls),selection_seconds=selected_build_seconds,
        solver_environment=environment,sealed=load('sealed.json'),review_seconds=time.monotonic()-started)
    return out


def compare_selections(arms):
    """Compare only available pre-query snapshots; missing is not agreement."""
    maps=[a.get('selection_snapshots',{}) for a in arms]
    if len(maps)!=2:raise ValueError('two required arms')
    common=set(maps[0])&set(maps[1]);pairs=[]
    for expert in sorted(common):
        a,b=[m[expert]['record'] for m in maps]
        left={k:v for k,v in a.items() if k not in ('arm','selected')}
        right={k:v for k,v in b.items() if k not in ('arm','selected')}
        if left!=right:raise ValueError('pre-query source/properties differ between arms')
        pairs.append({'expert':expert,'source_sha256':a['source_sha256'],
                      'prefix_rows':a['selected'],'property_rows':b['selected']})
    return {'compared':pairs,'unavailable_experts':sorted((set(maps[0])|set(maps[1]))-common),
            'complete_two_expert_comparison':len(common)==2,
            'scope':'Snapshot consistency; full mathematical replay only for complete checked packages.'}


def review(root):
    from property_ranges.run import FREEZE
    f=json.loads(FREEZE.read_bytes());e=json.loads((root/'execution.json').read_bytes())
    if e['freeze_sha256']!=sha(FREEZE):raise ValueError('frozen execution identity')
    for group in ('sources','artifacts'):
        for name,h in f[group].items():
            if sha(ROOT/name)!=h:raise ValueError('frozen inputs/code changed')
    batch=json.loads((root/'batch_terminal.json').read_bytes())
    if (batch['required_arms']!=f['arms'] or batch['complete']!=(len(batch['arms'])==2) or
            [r['arm'] for r in batch['arms']]!=f['arms'][:len(batch['arms'])]):raise ValueError('batch denominator/order')
    results=[]
    for name in f['arms']:
        if not (root/name/'terminal.json').exists():results.append({'arm':name,'status':'NOT_COMPLETED'});continue
        result=audit(root/name);save(root/name/'review.json',result);results.append({'arm':name,**result})
        row=next((r for r in batch['arms'] if r['arm']==name),None)
        terminal=json.loads((root/name/'terminal.json').read_bytes())
        if row is None or row['status']!=terminal['status'] or row['seconds']!=result['total_seconds']:
            raise ValueError('batch terminal/cost summary disagreement')
    compared=compare_selections(results)
    result={'status':'PASS','issues':[],'execution_head':e['head'],'required_arms':f['arms'],'arms':results,
        'prequery_comparison':compared,
        'scope':'Finite one-input source-range control; not population, route-changing or speedup evidence.'}
    save(root/'review.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':review(Path(sys.argv[1]).resolve())
