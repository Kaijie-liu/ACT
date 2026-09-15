"""Independent V2 terminal/package/journal audit; not independent SAFE reproof."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

from scripts.conv_budget_smoke_v2 import ROOT, ACT, ARMS, identity, schedule, request, artifacts
from scripts.conv_three_arm_contract import selection, read
from scripts.audit_conv_three_arm import terminal_contract
from scripts.check_budget_contract_v2 import check
from act.pipeline.moe.schedule_confirmation import inspect_row, expected_identity
from act.pipeline.moe.common_fact_snapshot import check_snapshot, fact_view
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256


def roster(rows,jobs,end):
    if (len(rows)>len(jobs) or any(any(r.get(k)!=v for k,v in j.items()) for r,j in zip(rows,jobs))
            or end['completed_job_ids']!=[r['job_id'] for r in rows]
            or end['unattempted']!=jobs[len(rows):] or end['full_started'] is not False):
        raise ValueError('missing/reordered/hidden terminal')
    if end['state']=='EXECUTION_COMPLETED':
        if len(rows)!=4 or end['error'] is not None or any(r['status']=='ERROR' for r in rows):
            raise ValueError('incomplete execution called complete')
    elif end['state']!='EXECUTION_ERROR' or not end['error']:
        raise ValueError('invalid fail-stop terminal')
    if any(r['status']=='ERROR' for r in rows[:-1]):raise ValueError('continued after ERROR')


def journal_check(directory,row,req):
    path=directory/'budget_journal.jsonl'
    if not path.exists():
        if row['package']:raise ValueError('complete package lacks journal')
        return None
    killed=row['outer_timeout'] or row['status']=='ERROR'
    expected={'request_sha256':row['request_sha256'],'execution_budget_contract':req['execution_budget_contract']}
    result=check(path,identity=expected,killed=killed)
    cmd=[ACT,'-S','-m','scripts.check_budget_contract_v2',str(path)]+(['--killed'] if killed else [])
    external=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
    # JSON round trip normalizes property-token dict keys.
    if json.loads(external.stdout)!=json.loads(json.dumps(result)):
        raise ValueError('separate stdlib journal check differs')
    events=[json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
    if any(e.get('clock_elapsed',0)>row['wall_seconds'] for e in events):
        raise ValueError('event after outer terminal')
    if row['package']:
        e=read(Path(row['package'])/'evidence.json'); contract=e['execution_budget_contract']
        expected_contract={'version':2,'total_seconds':300,'terminal_reserve_seconds':5,
            'local_construction_charged':True,'native_entry_rechecked':True,
            'journal_identity':expected,'partial_journal_can_establish_SAFE':False,
            'journal_sha256':_sha256(path)}
        if contract!=expected_contract:raise ValueError('package execution contract differs')
        if not result['work_complete'] or events[-1]['kind']!='WORK_COMPLETE' or events[-1]['status']!=row['status']:
            raise ValueError('journal and complete terminal disagree')
    return {'events':result['events'],'work_complete':result['work_complete'],
        'property_results':sum(p['result'] is not None for p in result['properties'].values()),
        'property_replays':sum(p['replay'] is not None for p in result['properties'].values()),
        'native_calls':result['native_calls'],'native_unreturned':result['native_unreturned'],
        'native_skipped':result['native_skipped'],'partial_tail_bytes':result['partial_tail_bytes'],
        'journal_can_establish_SAFE':False,'last_kind':events[-1]['kind']}


def audit(root):
    root=root.resolve()
    if not root.is_relative_to(ROOT/'data/moe/results'):raise ValueError('outside results')
    rt=read(root/'runtime.json');value=selection();jobs=schedule(value)
    if (rt['schema']!='conv_budget_smoke_v2' or rt['smoke'] is not True or rt['full_started'] is not False
            or rt['execution']!=identity() or rt['selection']!=value or rt['jobs']!=jobs
            or rt['config']!={'methods':value['identities']['method_configs']}):
        raise ValueError('frozen runtime differs')
    for p,sha in rt['parents'].items():
        if _sha256(ROOT/p)!=sha:raise ValueError('historical artifact changed')
    from act.pipeline.moe.external_pair_worker import load
    for job in jobs[::2]:load(request(value,job,rt['git_head'],rt['execution']))
    rows=[json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()] if (root/'rows.jsonl').exists() else []
    end=read(root/'run_terminal.json');roster(rows,jobs,end)
    allowed={j['job_id'] for j in jobs[:len(rows)]}
    if {p.name for p in root.iterdir() if p.is_dir()}!=allowed:
        raise ValueError('unaccounted request directory')
    configs={a:read(value['identities']['method_configs'][a]['path']) for a in ARMS}
    generic={**value,'models':{'conv':value['subject']},'request':{'epsilon':2/255}}
    details=[];facts={};complete=Counter();replayed=0
    for row in rows:
        directory=root/row['job_id'];req=request(value,row,rt['git_head'],rt['execution'])
        if (read(directory/'terminal.json')!=row or read(directory/'request.json')!=req
                or _sha256(directory/'request.json')!=row['request_sha256']
                or artifacts(directory)!=row['artifacts'] or row['evidence_level']!='HZ_POLICY_ACCEPTED'):
            raise ValueError('terminal/request/partial artifact changed')
        terminal_contract(row)
        wait=row['resource_wait']; state=wait['at_launch']
        if (not math.isfinite(wait['seconds']) or not 0<=wait['seconds']<=86400
                or state['available_ram_gib']<16 or state['free_disk_gib']<5 or not 0<=state['load_per_core']<=.5):
            raise ValueError('resource gate violated')
        augmented={**row,'model':'conv'}
        if row['status']=='ERROR':
            f=None
            if row['snapshot_sha256']:
                snap=read(directory/'common_facts.json')
                check_snapshot(snap,expected_identity=expected_identity(generic,augmented,configs[row['method']],True),
                               expected_config=configs[row['method']])
                if snap['payload']['completion_elapsed_seconds']>row['wall_seconds']:raise ValueError('late snapshot')
                f=fact_view(snap)
            checked={'facts':f,'package':False,'replayed':False,'pair_count':len(f['pairs']) if f else None}
        else:checked=inspect_row(root,augmented,rt,generic,configs)
        facts[row['rank'],row['method']]=checked['facts']
        complete[row['method']]+=int(checked['package']);replayed+=int(checked['replayed'])
        journal=journal_check(directory,row,req)
        details.append({'job_id':row['job_id'],'status':row['status'],'outer_timeout':row['outer_timeout'],
            'wall_seconds':row['wall_seconds'],'complete_package':checked['package'],
            'pair_count':checked['pair_count'],'journal':journal})
    equal=0
    for rank in range(2):
        states={r['status'] for r in rows if r['rank']==rank}
        if {'SAFE','UNSAFE'}<=states:raise ValueError('positive/witness conflict')
        a,b=[facts.get((rank,arm)) for arm in ARMS]
        if a is not None and b is not None:
            if a!=b:raise ValueError('common facts disagree')
            equal+=1
    gate=(end['state']=='EXECUTION_COMPLETED' and len(rows)==4 and all(complete[a]>=1 for a in ARMS)
          and equal>=1 and all(d['journal'] is not None for d in details))
    return {'status':'PASS','issues':[],'smoke_gate':'PASS' if gate else 'FAIL','rows':len(rows),'planned_rows':4,
        'complete_per_arm':{a:complete[a] for a in ARMS},'unsafe_replayed':replayed,
        'common_fact_pairs_equal':equal,'common_fact_pairs_unavailable':2-equal,'details':details,
        'methods':{a:{'states':dict(Counter(r['status'] for r in rows if r['method']==a)),
            'mean_observed_seconds':mean([r['wall_seconds'] for r in rows if r['method']==a])
            if any(r['method']==a for r in rows) else None} for a in ARMS},
        'runtime_sha256':_sha256(root/'runtime.json'), 'rows_sha256':_sha256(root/'rows.jsonl') if rows else None,
        'run_terminal_sha256':_sha256(root/'run_terminal.json'),'preserved_parent_artifacts':len(rt['parents']),
        'full_started':False,'unattempted':end['unattempted'],
        'scope':'Old-input V2 conformance, not independent SAFE reproof or performance confirmation; no full90 authorization.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'data/moe/results') or args.output.exists():
        raise ValueError('new local audit output required')
    try:result=audit(args.root)
    except Exception as exc:result={'status':'FAIL','issues':[repr(exc)],'smoke_gate':'FAIL','full_started':False}
    atomic_json(args.output,result);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
