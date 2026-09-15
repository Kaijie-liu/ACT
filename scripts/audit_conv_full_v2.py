"""Frozen full-cohort structural/evidence-level audit and descriptive analysis."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median

from scripts.conv_full_v2_contract import ROOT, ARMS, FREEZE, identity, full_selection, request_for
from scripts.conv_three_arm_contract import read
from scripts.conv_budget_smoke_v2 import artifacts, DEFAULT as SMOKE
from scripts.audit_conv_budget_smoke_v2 import journal_check, audit as smoke_audit
from scripts.audit_conv_three_arm import (terminal_contract, external_result, route_inventory,
    audit as old_audit)
from act.pipeline.moe.schedule_confirmation import inspect_row, expected_identity
from act.pipeline.moe.common_fact_snapshot import check_snapshot, fact_view
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256

OLD=ROOT/'data/moe/results/conv_three_arm_smoke_20260915_r1'


def preflight():
    from act.pipeline.moe.freeze_conv_three_arm import audit as clean_audit
    ident=identity();value=full_selection();protocol=ident['protocol']
    if (_sha256(SMOKE/'audit.final.json')!=protocol['act_smoke_audit_sha256']
            or _sha256(OLD/'audit.final.json')!=protocol['old_smoke_audit_sha256']):
        raise ValueError('registered prior audits changed')
    new=smoke_audit(SMOKE);old=old_audit(OLD)
    if (new!=read(SMOKE/'audit.final.json') or new['smoke_gate']!='PASS'
            or new!=read(SMOKE/'audit.independent.json') or old!=read(OLD/'audit.final.json')
            or old['status']!='PASS' or old['complete_per_arm']['crown']!=2
            or old['methods']['crown']['states'].get('ERROR',0)):
        raise ValueError('ACT V2 or preserved CROWN conformance unavailable')
    clean=clean_audit()
    if clean['status']!='PASS' or clean['samples']!=30:raise ValueError('clean reconstruction failed')
    return {'schema':'CONV_FULL_V2_FREEZE','status':'PASS','issues':[],'execution':ident,
        'selection_sha256':protocol['selection_sha256'],'indices':[s['dataset_index'] for s in value['samples']],
        'clean_only_reconstruction':clean,'v2_smoke':new,'old_crown_complete_records':2,
        'old_overall_smoke_gate_preserved':old['smoke_gate'],
        'parent_artifacts':{str(p.relative_to(ROOT)):_sha256(p) for d in (OLD,SMOKE)
                            for p in sorted(d.rglob('*')) if p.is_file()},
        'full_execution_started':False,
        'scope':'New full-run authorization plus verified component gates, not a relabelled R1 PASS. No full endpoint queried by this freeze.'}


def roster(rows,jobs,end):
    if (len(rows)>90 or any(any(r.get(k)!=v for k,v in j.items()) for r,j in zip(rows,jobs))
            or end['completed_job_ids']!=[r['job_id'] for r in rows] or end['unattempted']!=jobs[len(rows):]
            or end['full_started'] is not True):raise ValueError('incomplete/hidden/reordered roster')
    if end['state']=='EXECUTION_COMPLETED':
        if len(rows)!=90 or end['error'] is not None or any(r['status']=='ERROR' for r in rows):
            raise ValueError('unfinished run called complete')
    elif end['state']!='EXECUTION_ERROR' or not end['error']:raise ValueError('invalid failure terminal')
    if any(r['status']=='ERROR' for r in rows[:-1]):raise ValueError('continued after error')


def observed_diagnostics(directory):
    """Read-only stopping observations, never causal attribution or bound repair."""
    path=directory/'budget_journal.jsonl'
    if not path.exists():return None
    events=[json.loads(l) for l in path.read_bytes().splitlines(keepends=True) if l.endswith(b'\n')]
    properties=[e for e in events if e['kind']=='PROPERTY_RESULT']
    ready={e['seq']:e for e in events if e['kind']=='NATIVE_READY'}
    returned=[e for e in events if e['kind'] in ('NATIVE_RETURN','NATIVE_RAISE')]
    terminal_tokens={e['token'] for e in events if e['kind'] in ('NATIVE_RETURN','NATIVE_RAISE','NATIVE_SKIPPED')}
    overruns=[max(0.,e['clock_elapsed']-ready[e['token']]['deadline']) for e in returned]
    return {'property_reasons':dict(Counter(e['result'].get('reason','UNAVAILABLE') for e in properties)),
        'property_solver_statuses':dict(Counter(str(e['result'].get('solver_status','UNAVAILABLE')) for e in properties)),
        'native_unreturned_tokens':sorted(set(ready)-terminal_tokens),
        'maximum_observed_native_return_overrun_seconds':max(overruns) if overruns else None,
        'last_state':next((e.get('record') for e in reversed(events) if e['kind']=='STATE'),None),
        'interpretation':'Stop locations and native-return observations only; limits do not establish true safety or a unique root cause.'}


def summarize(rows,details,complete):
    by={(r['rank'],r['method']):r for r in rows}
    def positive(r):return r['status']==('POSITIVE' if r['method']=='crown' else 'SAFE')
    methods={}
    for arm in ARMS:
        observed=[r for r in rows if r['method']==arm];times=[r['wall_seconds'] for r in observed]
        methods[arm]={'denominator':30,'attempted':len(observed),'unattempted':30-len(observed),
            'states':dict(Counter(r['status'] for r in observed)),
            'complete_unknown':sum(r['status']=='UNKNOWN' for r in observed),
            'internal_timeout':sum(r['status']=='TIMEOUT' and not r['outer_timeout'] for r in observed),
            'outer_timeout':sum(r['outer_timeout'] for r in observed),
            'positive_evidence_level':'CROWN_NUMERICAL_FILTER' if arm=='crown' else 'HZ_POLICY_ACCEPTED',
            'all_observed_seconds':sum(times),'mean_observed_seconds':mean(times) if times else None,
            'median_observed_seconds':median(times) if times else None,
            'resource_wait_seconds':sum(r['resource_wait']['seconds'] for r in observed)}
    comparisons={}
    if complete:
        import numpy as np
        for baseline in ('monolithic','crown'):
            a={i for i in range(30) if positive(by[i,'adaptive'])}
            b={i for i in range(30) if positive(by[i,baseline])}
            differences=[float(positive(by[i,'adaptive']))-float(positive(by[i,baseline])) for i in range(30)]
            delta=[by[i,'adaptive']['wall_seconds']-by[i,baseline]['wall_seconds'] for i in range(30)]
            c={'shared_positive_ranks':sorted(a&b),'adaptive_only_positive_ranks':sorted(a-b),
               'baseline_only_positive_ranks':sorted(b-a),'mean_positive_indicator_difference':mean(differences),
               'mean_paired_seconds_difference':mean(delta),'median_paired_seconds_difference':median(delta),
               'evidence_levels_interchangeable':baseline!='crown'}
            if baseline=='monolithic':
                sa={i for i in range(30) if by[i,'adaptive']['status'] in ('SAFE','UNSAFE')}
                sb={i for i in range(30) if by[i,baseline]['status'] in ('SAFE','UNSAFE')}
                draws=np.random.default_rng(20260915).integers(0,30,(10000,30))
                c.update(shared_solved_ranks=sorted(sa&sb),adaptive_only_solved_ranks=sorted(sa-sb),
                    baseline_only_solved_ranks=sorted(sb-sa),
                    descriptive_SAFE_interval_95=np.quantile(np.asarray(differences)[draws].mean(1),[.025,.975]).tolist(),
                    degenerate_observed_differences=len(set(differences))==1)
            comparisons[baseline]=c
    strata=[]
    for rank in range(30):
        ds=[d for d in details if d['rank']==rank];pairs=[d['pairs'] for d in ds if d['pairs'] is not None]
        if pairs and any(p!=pairs[0] for p in pairs):raise ValueError('route inventories disagree')
        states={r['status'] for (i,_),r in by.items() if i==rank}
        if 'UNSAFE' in states and states&{'SAFE','POSITIVE'}:raise ValueError('positive/witness conflict')
        strata.append({'rank':rank,'pair_count':len(pairs[0]) if pairs else None,
            'route_stratum':'unavailable' if not pairs else 'single' if len(pairs[0])==1 else 'multiple',
            'states':{a:by[rank,a]['status'] if (rank,a) in by else 'UNATTEMPTED' for a in ARMS}})
    return {'methods':methods,'completed_cohort_analysis':complete,'comparisons':comparisons,
        'route_strata':strata,'statistical_unit':'30 input blocks, not90 independent samples',
        'interval_scope':'descriptive unadjusted percentile; zero-width zero interval is not population equivalence',
        'cost_scope':'all observed requests including unresolved/capped runs, not uncensored time-to-proof'}


def audit(root):
    root=root.resolve()
    if not root.is_relative_to(ROOT/'data/moe/results'):raise ValueError('outside results')
    runtime=read(root/'runtime.json');value=full_selection();execution=identity();freeze=read(FREEZE)
    if (runtime['schema']!='conv_three_arm_full_v2' or runtime['smoke'] is not False
            or runtime['selection']!=value or runtime['execution']!=execution
            or runtime['freeze_sha256']!=_sha256(FREEZE) or freeze['execution']!=execution
            or freeze['status']!='PASS' or runtime['jobs']!=value['full_jobs']
            or runtime['config']!={'methods':value['identities']['method_configs']}):
        raise ValueError('full execution identity drift')
    for p,sha in freeze['parent_artifacts'].items():
        if _sha256(ROOT/p)!=sha:raise ValueError('parent artifact changed')
    from act.pipeline.moe.external_pair_worker import load
    for job in value['full_jobs'][::3]:load(request_for(value,job,runtime['git_head'],execution))
    rows=[json.loads(l) for l in (root/'rows.jsonl').read_text().splitlines()] if (root/'rows.jsonl').exists() else []
    end=read(root/'run_terminal.json');roster(rows,value['full_jobs'],end)
    if {p.name for p in root.iterdir() if p.is_dir()}!={r['job_id'] for r in rows}:
        raise ValueError('unaccounted request directory')
    configs={a:read(v['path']) for a,v in value['identities']['method_configs'].items()}
    generic={**value,'models':{'conv':value['subject']},'request':{'epsilon':2/255}}
    details=[];facts={};complete=Counter();replays=0
    for row in rows:
        directory=root/row['job_id'];req=request_for(value,next(j for j in value['full_jobs'] if j['job_id']==row['job_id']),runtime['git_head'],execution)
        if (read(directory/'request.json')!=req or _sha256(directory/'request.json')!=row['request_sha256']
                or read(directory/'terminal.json')!=row or artifacts(directory)!=row['artifacts']):
            raise ValueError('request/terminal/partial inventory changed')
        terminal_contract(row)
        level='CROWN_NUMERICAL_FILTER' if row['method']=='crown' else 'HZ_POLICY_ACCEPTED'
        if row['evidence_level']!=level:raise ValueError('mixed evidence levels')
        wait=row['resource_wait'];s=wait['at_launch']
        if (not math.isfinite(wait['seconds']) or not 0<=wait['seconds']<=86400
                or s['available_ram_gib']<16 or s['free_disk_gib']<5 or not 0<=s['load_per_core']<=.5):
            raise ValueError('resource gate violation')
        for name,key in [('common_facts.json','snapshot_sha256'),('routes.json','routes_sha256'),('external.json','external_sha256')]:
            path=directory/name
            if path.exists()!=(row.get(key) is not None) or path.exists() and _sha256(path)!=row[key]:
                raise ValueError('partial identity omitted')
        f=None;journal=None;reason=None
        if row['status']=='ERROR':
            if row['snapshot_sha256']:
                snap=read(directory/'common_facts.json');cfg=configs[row['method']]
                check_snapshot(snap,expected_identity=expected_identity(generic,{**row,'model':'conv'},cfg,False),expected_config=cfg)
                f=fact_view(snap)
            if row.get('routes_sha256'):
                r=read(directory/'routes.json')
                if r['request']!=req:raise ValueError('partial CROWN identity differs')
                route_inventory(r['routes'])
            result={'complete':False,'pairs':f['pairs'] if f else None,'replayed':False}
        elif row['method']=='crown':
            result=external_result(root,row,req)
            if (directory/'external.json').exists():reason=read(directory/'external.json').get('reason')
        else:
            checked=inspect_row(root,{**row,'model':'conv'},runtime,generic,configs);f=checked['facts']
            pairs=f['pairs'] if f else None
            if row['package']:
                e=read(Path(row['package'])/'evidence.json');coverage=e['route_coverage']
                if coverage['route_sets_exact']:pairs=coverage['feasible_route_sets']
                reason=e['verdict'].get('reason')
            result={'complete':checked['package'],'pairs':pairs,'replayed':checked['replayed']}
        if row['method']!='crown':journal=journal_check(directory,row,req)
        elif (directory/'budget_journal.jsonl').exists():raise ValueError('V2 adapter applied to CROWN')
        if result['pairs'] is not None and any(p not in value['possible_pairs'] for p in result['pairs']):
            raise ValueError('wrong E4 pair')
        facts[row['rank'],row['method']]=f;complete[row['method']]+=int(result['complete']);replays+=int(result['replayed'])
        details.append({'job_id':row['job_id'],'rank':row['rank'],'dataset_index':row['dataset_index'],
            'method':row['method'],'status':row['status'],'outer_timeout':row['outer_timeout'],
            'reason':reason,'complete_record':result['complete'],'pairs':result['pairs'],'journal':journal,
            'stopping_observations':observed_diagnostics(directory) if row['method']!='crown' else None})
    equal=0
    for rank in range(30):
        a,b=[facts.get((rank,arm)) for arm in ('adaptive','monolithic')]
        if a is not None and b is not None:
            if a!=b:raise ValueError('common facts differ')
            equal+=1
    return {'schema':'CONV_FULL_V2_AUDIT','status':'PASS','issues':[],'rows':len(rows),'planned_rows':90,
        'complete_per_arm':{a:complete[a] for a in ARMS},'unsafe_replayed':replays,
        'common_fact_pairs_equal':equal,'common_fact_pairs_unavailable':30-equal,
        'details':details,'analysis':summarize(rows,details,end['state']=='EXECUTION_COMPLETED'),
        'runtime_sha256':_sha256(root/'runtime.json'),'rows_sha256':_sha256(root/'rows.jsonl') if rows else None,
        'run_terminal_sha256':_sha256(root/'run_terminal.json'),'unattempted':end['unattempted'],
        'scope':'Structural, execution-accounting and full-model witness audit; not independent SAFE reproof. CROWN positives remain numerical filters.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or args.output.exists():raise ValueError('new project output required')
    try:result=preflight() if args.preflight else audit(args.root)
    except Exception as exc:result={'status':'FAIL','issues':[repr(exc)]}
    atomic_json(args.output,result);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
