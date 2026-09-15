"""Fresh terminal/proof review and input-block statistics; distinct evidence grades."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean,median
import subprocess
import time
from portable_proof.runtime import digest
from scripts.optional_evidence_dev_contract import ROOT,ACT,read,save
from moe_evidence.execution import LEVELS,PHASES
from evidence_cohort.contract import OUTPUT,EXECUTION,FREEZE,ARMS,verify_freeze,selection,request_for


def roster(rows,jobs,end):
    if len(rows)>len(jobs) or any(any(r.get(k)!=v for k,v in j.items()) for r,j in zip(rows,jobs)):
        raise ValueError('reordered/replaced/extra request')
    if end['completed_job_ids']!=[r['job_id'] for r in rows]:raise ValueError('completed roster differs')
    aborted=end['aborted_job'];n=len(rows)
    if aborted is not None and (n==len(jobs) or aborted!=jobs[n] or not end['error']):
        raise ValueError('invalid aborted request')
    if end['unattempted']!=jobs[n+(aborted is not None):]:raise ValueError('hidden pending request')
    if end['state']=='EXECUTION_COMPLETED':
        if n!=len(jobs) or aborted or end['error'] or any(r['status']=='ERROR' for r in rows):
            raise ValueError('incomplete run called complete')
    elif end['state']!='EXECUTION_ERROR' or not end['error']:raise ValueError('invalid run terminal')
    if any(r['status']=='ERROR' for r in rows[:-1]):raise ValueError('continued after ERROR')


def envelope(root,row,req):
    t=read(root/'terminal.json')
    if digest((root/'terminal.json').read_bytes())!=row['terminal_sha256'] or any(row[k]!=v for k,v in t.items()):
        raise ValueError('terminal/ledger changed')
    if (read(root/'request.json')!=req or t['request_sha256']!=digest((root/'request.json').read_bytes())
        or t['evidence_level']!=LEVELS[row['arm']] or t['budget_seconds']!=300
        or not math.isfinite(t['wall_seconds']) or t['wall_seconds']<0
        or t['production_gate_changed'] or t['deployed_float_SAFE']):raise ValueError('request/terminal semantics mismatch')
    actual={str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p!=root/'terminal.json'}
    if actual!=t['artifact_sha256']:raise ValueError('artifact inventory drift')
    if (t['wall_seconds']>300 or t['outer_timeout'] or t['whole_request_timeout']) and t['status']!='TIMEOUT':
        raise ValueError('late positive promotion')
    if t['whole_request_timeout'] and t['complete_independent_check']:raise ValueError('deadline failure marked complete')
    if t['whole_request_timeout'] and not t['outer_timeout']:raise ValueError('whole timeout misclassified as internal')
    owner=t['outer_process']
    if owner['killed'] and not t['whole_request_timeout']:raise ValueError('killed driver promoted')
    if t['status'] not in ('ERROR','TIMEOUT') and (owner['return_code']!=0 or owner['killed']):
        raise ValueError('failed driver promoted')
    keys=[k for k in PHASES[row['arm']] if k in t['stages']]
    if set(t['stages'])!=set(PHASES[row['arm']][:len(keys)]):raise ValueError('phase inventory is not a registered prefix')
    previous=0
    for k in keys:
        p=t['stages'][k]
        if (not all(math.isfinite(p[x]) for x in ('start_seconds','elapsed_seconds','allowed_seconds'))
            or p['start_seconds']<previous or p['elapsed_seconds']<0 or p['allowed_seconds']<=0
            or p['start_seconds']+p['allowed_seconds']>298+.01
            or p['start_seconds']+p['elapsed_seconds']>t['wall_seconds']+.01):raise ValueError('invalid phase cost')
        previous=p['start_seconds']+p['elapsed_seconds']
    w=row['resource_wait'];r=w['at_launch'];p=EXECUTION['resource']
    if (not math.isfinite(w['seconds']) or not 0<=w['seconds']<=p['wait_limit_seconds']
        or r['ram_gib']<p['minimum_ram_gib'] or r['disk_gib']<p['minimum_disk_gib']
        or not 0<=r['load_per_core']<=p['maximum_load_per_core']):raise ValueError('resource gate violated')
    return t


def check_one(root,row,req):
    d=root/row['job_id'];t=envelope(d,row,req)
    if t['whole_request_timeout'] or t['status']=='ERROR':
        # No success aggregation of late/partial files. Keep observation only.
        result={'status':'PASS','issues':[],'terminal':t['status'],'pairs':None,'facts':None,
            'replayed':False,'conditional_check':False,'complete_package':False,
            'scope':'retained unsuccessful terminal and hashed partial artifacts, not a proof'}
        if (d/'budget_journal.jsonl').exists():
            from scripts.check_budget_contract_v2 import check
            from moe_evidence.schema import validate_request
            identity={'conditional_evidence_request':validate_request(req['evidence_request'])} if row['arm']=='evidence' else {'request_sha256':t['request_sha256']}
            result['journal']=check(d/'budget_journal.jsonl',identity=identity,killed=True)
    else:
        from moe_evidence.audit import audit_request
        result=audit_request(d,req,row['arm'])
    # Even killed requests can retain an independently bound *observation* of
    # common intervals; never infer proof/equality from a missing observation.
    if result['facts'] is None and (d/'common_facts.json').exists():
        snap=read(d/'common_facts.json')
        if row['arm']=='matched':
            from act.pipeline.moe.common_fact_snapshot import check_snapshot
            check_snapshot(snap,expected_config=read(req['config']['path']))
            p=snap['payload'];identity=p['identity'];r=req['evidence_request']
            if any(identity[k]!=r[k] for k in ('model_state','center','lower','upper')):raise ValueError('partial facts scope differs')
            result['facts']={'pairs':p['feasible_route_sets'],'branches':[{'expert':b['candidate'],'interval':b['proof_output_bounds']} for b in p['branches']]}
        elif row['arm']=='evidence':
            from moe_evidence.schema import validate_request
            r=req['evidence_request']
            if snap['request_id']!=validate_request(r) or snap['identity']!={k:r[k] for k in ('model_state','center','lower','upper')}:
                raise ValueError('partial evidence facts scope differs')
            result['facts']={k:snap[k] for k in ('pairs','branches')}
    if result['pairs'] is None and result['facts'] is not None:result['pairs']=result['facts']['pairs']
    result.update(job_id=row['job_id'],rank=row['rank'],arm=row['arm'],dataset_index=row['dataset_index'])
    result['phase_seconds']={k:v['elapsed_seconds'] for k,v in t['stages'].items()}
    result['censored_phase']=t['censored_phase']
    result['proof_size']=read(d/'packing.json') if (d/'packing.json').exists() else None
    # This is already checked only when the completed request checker ran.
    result['checked_obligation_counts']=None
    if result['conditional_check']:
        r=read(d/'independent.json')
        result['checked_obligation_counts']={k:r.get(k) for k in ('required_obligations','positive_obligations','missing_obligations','nonpositive_obligations')}
    return result


def positive(row):
    return row['status']=={'matched':'SAFE','evidence':'CHECKED_CONDITIONAL','crown':'POSITIVE'}[row['arm']]


def summarize(rows,details,jobs,complete):
    n=len({j['rank'] for j in jobs});by={(r['rank'],r['arm']):r for r in rows}
    ds={(d['rank'],d['arm']):d for d in details};methods={};comparisons={};strata=[]
    for arm in ARMS:
        r=[v for v in rows if v['arm']==arm];times=[v['wall_seconds'] for v in r]
        methods[arm]={'denominator':n,'attempted_terminals':len(r),'without_terminal':n-len(r),
            'states':dict(Counter(v['status'] for v in r)),'positive_evidence_level':LEVELS[arm],
            'positive_count':sum(positive(v) for v in r),'unsafe_replayed':sum(ds[v['rank'],arm]['replayed'] for v in r),
            'outer_timeout':sum(v['outer_timeout'] for v in r),
            'internal_timeout':sum(v['status']=='TIMEOUT' and not v['outer_timeout'] for v in r),
            'sum_seconds':sum(times),'mean_seconds':mean(times) if times else None,'median_seconds':median(times) if times else None,
            'resource_wait_seconds':sum(v['resource_wait']['seconds'] for v in r)}
    equal=0;missing=0
    for i in range(n):
        facts=[ds.get((i,a),{}).get('facts') for a in ('matched','evidence')]
        if all(f is not None for f in facts):
            if facts[0]!=facts[1]:raise ValueError('charged common facts disagree')
            equal+=1
        else:missing+=1
        pairs=[ds[i,a]['pairs'] for a in ARMS if (i,a) in ds and ds[i,a]['pairs'] is not None]
        if pairs and any(p!=pairs[0] for p in pairs):raise ValueError('legal pair observations disagree')
        states=[by[i,a]['status'] for a in ARMS if (i,a) in by]
        if 'UNSAFE' in states and any(positive(by[i,a]) for a in ARMS if (i,a) in by):
            raise ValueError('positive/full-model witness conflict')
        count=len(pairs[0]) if pairs else None
        strata.append({'rank':i,'dataset_index':next(j['dataset_index'] for j in jobs if j['rank']==i),
            'pair_count':count,'stratum':'unavailable' if count is None else 'single' if count==1 else 'multiple',
            'states':{a:by[i,a]['status'] if (i,a) in by else 'WITHOUT_TERMINAL' for a in ARMS}})
    if complete:
        import numpy as np
        draws=np.random.default_rng(20260916).integers(0,n,(10000,n))
        for baseline in ('matched','crown'):
            a={i for i in range(n) if positive(by[i,'evidence'])};b={i for i in range(n) if positive(by[i,baseline])}
            diffs=np.array([int(i in a)-int(i in b) for i in range(n)])
            delta=[by[i,'evidence']['wall_seconds']-by[i,baseline]['wall_seconds'] for i in range(n)]
            comparisons[baseline]={'shared_positive_indices':[strata[i]['dataset_index'] for i in sorted(a&b)],
                'evidence_only_positive_indices':[strata[i]['dataset_index'] for i in sorted(a-b)],
                'baseline_only_positive_indices':[strata[i]['dataset_index'] for i in sorted(b-a)],
                'paired_positive_difference':float(diffs.mean()),
                'descriptive_input_interval95':np.quantile(diffs[draws].mean(1),[.025,.975]).tolist(),
                'degenerate_difference':len(set(diffs.tolist()))==1,
                'mean_paired_seconds':mean(delta),'median_paired_seconds':median(delta),
                'evidence_contracts_equal':False,'formal_SAFE_comparison':False}
    return {'methods':methods,'comparisons':comparisons,'route_strata':strata,
        'common_fact_pairs_equal':equal,'common_fact_pairs_unavailable':missing,'completed_cohort_analysis':complete,
        'statistical_unit':f'{n} input blocks, not {3*n} independent runs',
        'cost_scope':'all observed terminals including incomplete and capped; not uncensored time-to-proof',
        'guarantee':'conditioned on stated upstream trusted components; evidence levels never pooled as formal SAFE'}


def audit(root):
    verify_freeze();v=selection();rt=read(root/'runtime.json');end=read(root/'run_terminal.json')
    if (rt['execution']!=EXECUTION or rt['schema']!=EXECUTION['schema'] or rt['freeze_sha256']!=digest(FREEZE.read_bytes())
        or rt['jobs']!=v['jobs'] or rt['selection_sha256']!=EXECUTION['selection_sha256']
        or rt['remote_before_launch']!=rt['git_head']):raise ValueError('runtime freeze drift')
    rows=[json.loads(l) for l in (root/'rows.jsonl').read_text().splitlines()] if (root/'rows.jsonl').exists() else []
    roster(rows,v['jobs'],end)
    if any(rt[k]!=end[k] for k in ('completed_job_ids','unattempted','aborted_job')) or rt['state']!=end['state']:
        raise ValueError('runtime/terminal roster differs')
    allowed={r['job_id'] for r in rows}|({'control','reviews'})
    if end['aborted_job']:allowed.add(end['aborted_job']['job_id'])
    actual={p.name for p in root.iterdir() if p.is_dir()}
    if not actual<=allowed or not {r['job_id'] for r in rows}<=actual:raise ValueError('unaccounted/missing request directories')
    details=[];start=time.monotonic()
    for row in rows:
        expected=request_for(v,next(j for j in v['jobs'] if j['job_id']==row['job_id']),rt['git_head'])
        if read(root/row['job_id']/'request.json')!=expected:raise ValueError('worker request differs from freeze')
        # Bound even a slow independent checker. Never amend the request terminal.
        out=root/'reviews'/f"{row['job_id']}.json"
        from evidence_cohort.run import wait_owned,environment
        with (root/'reviews'/f"{row['job_id']}.log").open('x') as log:
            p=subprocess.Popen([ACT,'-m','evidence_cohort.audit','--root',str(root),'--one',row['job_id']],
                cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            outcome=wait_owned(p,time.monotonic()+600)
        if outcome['return_code']!=0 or not out.exists():raise ValueError('request audit failed: '+row['job_id'])
        details.append(read(out))
    summary=summarize(rows,details,v['jobs'],end['state']=='EXECUTION_COMPLETED')
    return {'status':'PASS','issues':[],'execution_state':end['state'],'requests_with_terminal':len(rows),
        'unattempted':end['unattempted'],'aborted_job':end['aborted_job'],'summary':summary,'details':details,
        'archive_audit_seconds':time.monotonic()-start,'scope':'independent terminal, source, witness and conditional LP checks; not independent network lowering proof',
        'rows_sha256':digest((root/'rows.jsonl').read_bytes()) if rows else None,'run_terminal_sha256':digest((root/'run_terminal.json').read_bytes())}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--one');a=p.parse_args()
    root=a.root.resolve()
    if root!=OUTPUT:raise ValueError('only frozen execution root accepted')
    out=root/'reviews'/f'{a.one}.json' if a.one else root/'audit.final.json'
    if out.exists():raise FileExistsError('audit immutable; use a new identified follow-up, not overwrite')
    try:
        if a.one:
            verify_freeze();v=selection();rt=read(root/'runtime.json');rows=[json.loads(l) for l in (root/'rows.jsonl').read_text().splitlines()]
            row=next(r for r in rows if r['job_id']==a.one);job=next(j for j in v['jobs'] if j['job_id']==a.one)
            result=check_one(root,row,request_for(v,job,rt['git_head']))
        else:result=audit(root)
    except Exception as exc:result={'status':'FAIL','issues':[repr(exc)],'completed_cohort_analysis':False}
    save(out,result);print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
