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
    rows=[]
    for entered in sorted((root/'range_queries').glob('*_entered.json')):
        e=json.loads(entered.read_bytes());result=entered.with_name(entered.name.replace('_entered',''))
        if (not 0<e['native_limit_seconds']<=3 or e['row'] not in (0,1) or e['context']['layer']!='6'
                or e['side'] not in ('lower','negative_upper')):raise ValueError('native range cap/roster')
        r=json.loads(result.read_bytes()) if result.exists() else None
        if r is not None and (r['native_limit_seconds']!=e['native_limit_seconds'] or
                (r['candidate'] is not None and r['candidate']['lp_sha256']!=e['lp_sha256'])):
            raise ValueError('range side identity')
        if r is not None and r.get('environment')!=ENVIRONMENT:raise ValueError('range solver environment drift')
        rows.append({'entry':entered.name,'source_sha256':e['source_sha256'],'row':e['row'],'side':e['side'],
            'complete':r is not None,'candidate_available':r is not None and r['candidate'] is not None,
            'native_seconds':r['native_seconds'] if r else None})
    out={'status':'PASS','issues':[],'outcome':'TIMEOUT' if late else t['status'],'total_seconds':p['seconds'],
        'stages':t['stages'],'range_calls':rows,'range_calls_entered':len(rows),
        'terminal_sha256':p['terminal_sha256'],'production_verdict_changed':False,'deployed_floating_point_proof':False}
    if late or t['error'] or len(t['stages'])!=4 or t['stages'][-1]['state']!='COMPLETED':
        out.update(complete=False,failed_execution_preserved=True);return out
    if not t['elapsed_before_publication']<=p['seconds']<budget:raise ValueError('publication budget')
    g=load('generation.json');job=load('job.json');src=root/'relocated';sm=json.loads((src/'source/manifest.json').read_bytes())
    if (g['arm']!=job['arm'] or sm['policy']['enabled']!=(job['arm']=='range_on') or
            g['new_output_obligations']!=required or g['range_native_calls']!=len(rows) or len(rows)>8):
        raise ValueError('arm/row/call coverage')
    trace=json.loads((src/'source/trace.json').read_bytes());accepted={};expected_range=[];range_context={}
    for step in trace['steps']:
        if step['layer']!=6:continue
        range_context[step['expert']]=(step['proof']['context'],step['proof']['source'])
        expected_range.extend((step['expert'],j) for j in range(min(2,len(step['proof']['facts']))))
        for j,fact in enumerate(step['proof']['facts']):
            if fact is not None:
                from source_enclosure.format import identity
                accepted[(step['expert'],j)]=identity(fact)
    if [(r['expert'],r['row']) for r in g['range_rows']]!=expected_range:
        raise ValueError('complete range row inventory')
    for path in (root/'range_queries').glob('*_entered.json'):
        e=json.loads(path.read_bytes())
        matches=[i for i,(ctx,h) in range_context.items() if e['context']==ctx and e['source_sha256']==h]
        if len(matches)!=1 or path.name!=f"e{matches[0]}_r{e['row']}_{e['side']}_entered.json":
            raise ValueError('range call/source binding')
    for r in g['range_rows']:
        if r['fact_sha256']!=accepted.get((r['expert'],r['row'])):raise ValueError('accepted range fact accounting')
        path=root/'range_queries'/f"e{r['expert']}_r{r['row']}_complete.json"
        if job['arm']=='range_on' and json.loads(path.read_bytes())!={k:v for k,v in r.items() if k!='expert'}:
            raise ValueError('range completion ledger')
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
    out.update(complete=True,result=strip(checked),generation=g,accepted_range_facts=len(accepted),output_calls=len(bound_calls),
        solver_environment=environment,sealed=load('sealed.json'),review_seconds=time.monotonic()-started)
    return out


def review(root):
    from range_pipeline.run import FREEZE
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
    result={'status':'PASS','issues':[],'execution_head':e['head'],'required_arms':f['arms'],'arms':results,
        'scope':'Finite one-input source-range control; not population, route-changing or speedup evidence.'}
    save(root/'review.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':review(Path(sys.argv[1]).resolve())
