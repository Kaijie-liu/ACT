"""Frozen two-arm saved-only archive: no models, solvers, retries or repairs."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def forbid(event,args):
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy','act'):
        raise ImportError('saved-only audit cannot import model/solver')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec'):
        raise PermissionError('saved-only audit cannot launch work')


sys.addaudithook(forbid)
from scoped_proof.io import load, save, sha
from scoped_parse_proof.batch_audit import review
from scoped_parse_proof.audit import audit

CONFIG=ROOT/'configs/backend_controls/scoped_parse_proof_compare_r1.json'
FREEZE=ROOT/'docs/scoped_parse_proof_freeze_review_20260924_r1.json'
AUDIT=ROOT/'docs/scoped_parse_proof_execution_audit_20260924_r1.json'
ARCHIVE=ROOT/'docs/scoped_parse_proof_execution_archive_20260924_r1.json'


def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in ('separate_audit_seconds','archive_seconds')}
    if isinstance(value,list):return [stable(v) for v in value]
    return value


def trace(events,stage):
    """Nested complete events, censored open stack; never invent missing exits."""
    stack=[];totals={};completed=[];previous=stage['start_seconds']
    for e in events:
        elapsed=e['elapsed'];name=e['operation'];kind=e['event']
        if not math.isfinite(elapsed) or not previous<=elapsed<=stage['end_seconds']:
            raise ValueError('event outside ordered charged stage')
        previous=elapsed
        if kind=='ENTER':stack.append({'event':e,'nested_seconds':0.})
        elif kind in ('EXIT','EXIT_ERROR'):
            if not stack or stack[-1]['event']['operation']!=name:
                raise ValueError('unmatched nested event')
            frame=stack.pop();duration=e['seconds']
            if not math.isfinite(duration) or duration<0:raise ValueError('invalid event cost')
            for key in ('tag','pair'):
                if e.get(key)!=frame['event'].get(key):raise ValueError('event identity changed')
            if stack:stack[-1]['nested_seconds']+=duration
            row=totals.setdefault(name,{'completed':0,'errors':0,'inclusive_seconds':0.,'exclusive_seconds':0.})
            row['completed']+=1;row['errors']+=int(kind=='EXIT_ERROR')
            row['inclusive_seconds']+=duration
            row['exclusive_seconds']+=max(0.,duration-frame['nested_seconds'])
            if name in ('affine_lift','relu_graph','pair_guard','output_lp'):
                completed.append({'operation':name,'tag':e.get('tag'),'pair':e.get('pair'),
                    'seconds':duration,'finished_seconds':elapsed,'error':kind=='EXIT_ERROR'})
        else:raise ValueError('unexpected event type')
    return {'phase':stage['phase'],'closed_operation_totals':totals,
        'completed_structural_operations':completed,
        'open_stack_at_stop':[{'operation':f['event']['operation'],
            'tag':f['event'].get('tag'),'pair':f['event'].get('pair'),
            'entered_seconds':f['event']['elapsed'],
            'observed_seconds_to_phase_end':stage['end_seconds']-f['event']['elapsed']} for f in stack],
        'event_count':len(events),'note':'nested inclusive times overlap; open operations are censored, not complete timings'}


def collect():
    start=time.monotonic();freeze=load(FREEZE);cfg=load(CONFIG,freeze['config_sha256'])
    root=Path(cfg['output']);launch=load(root/'launch.json')
    if launch['config']!=cfg or launch['config_sha256']!=sha(CONFIG):raise ValueError('launch is not frozen config')
    for name,digest in cfg['common']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('frozen execution code changed')
    fresh=review(root);details=[]
    for row in fresh['rows']:
        folder=root/row['id'];observations=[];native=0
        if not folder.exists():
            details.append({'id':row['id'],'status':row['status'],'evidence':None});continue
        terminal=load(folder/'terminal.json');cost=load(folder/'cost.json')
        for stage in terminal['stages']:
            path=folder/(stage['phase']+'_events.jsonl')
            raw=path.read_bytes() if path.exists() else b''
            # A killed write may leave only an unterminated final fragment. Keep/hash it,
            # but do not parse it as a completed event or discard a malformed full line.
            lines=raw.splitlines(keepends=True);partial=0
            if lines and not lines[-1].endswith(b'\n'):partial=len(lines.pop())
            events=[json.loads(line) for line in lines]
            item=trace(events,stage);item['trailing_partial_bytes']=partial;observations.append(item)
            native+=sum(e['event']=='ENTER' and e['operation'].startswith('native_lp_') for e in events)
        candidate_names=sorted(p.name for p in (folder/'candidates').glob('*.json'))
        proof=load(folder/'evidence_check.json') if (folder/'evidence_check.json').exists() else None
        checked=0 if proof is None else proof['checked_bounds'];positive=0 if proof is None else proof['positive_bounds']
        details.append({'id':row['id'],'status':row['status'],'cost':cost,'return_seconds':row['cost_seconds'],
            'stages':terminal['stages'],'operation_observations':observations,
            'evidence':{'source_captured':(folder/'source.json').exists(),
                'source_sha256':sha(folder/'source.json') if (folder/'source.json').exists() else None,
                'construction_published':(folder/'construction.json').exists(),
                'construction_receipt':(folder/'construction_receipt.json').exists(),
                'source_independently_checked':(folder/'source_check.json').exists(),
                'native_lp_calls_started':native,'candidate_files':candidate_names,'exact_bounds_checked':checked,
                'positive_bounds':positive,'obligations_without_checked_bound':cfg['required_per_call']-checked,
                'complete_output_positive_proof':row['audit']['complete_output_positive_proof'],
                'native_float_proof':False,'route_changing_established':False}})
    files={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('raw symlink')
        if path.is_file():files[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':sha(path)}
    result={'schema':'SCOPED_PARSE_PROOF_COMPARISON_ARCHIVE_V1','audit':'PASS','issues':0,
        'config_sha256':sha(CONFIG),'launch_head':launch['head'],'implementation_commit':cfg['implementation_commit'],
        'dataset_index':cfg['sample']['dataset_index'],'required_per_call':cfg['required_per_call'],
        'frozen_source_files_checked':len(cfg['common']['sources']),'calls':details,'frozen_batch_audit':stable(fresh),
        'raw_root':str(root),'files':files,'total_raw_bytes':sum(r['bytes'] for r in files.values()),
        'archive_source_sha256':sha(__file__),'archive_controls':controls(),
        'archive_seconds':time.monotonic()-start,'new_solver_calls':0,
        'scope':'saved identity/cost/events and available exact evidence only; no missing proof recovery',
        'decision':'SEAL_TWO_FROZEN_CALLS; no retries, extra time, new samples or old positive bound reuse'}
    return result,fresh


def controls():
    stage={'phase':'construct','start_seconds':0.,'end_seconds':5.}
    events=[{'event':'ENTER','operation':'outer','elapsed':1.},
        {'event':'ENTER','operation':'inner','elapsed':2.},
        {'event':'EXIT','operation':'inner','elapsed':3.,'seconds':1.}]
    got=trace(events,stage)
    assert got['open_stack_at_stop'][0]['observed_seconds_to_phase_end']==4.
    assert got['closed_operation_totals']['inner']['inclusive_seconds']==1.
    rejected=0
    for bad in ([{**events[0],'elapsed':-1.}], [events[0],{**events[2],'operation':'wrong'}],
                [events[0],{**events[2],'operation':'outer','seconds':float('nan')}],
                [events[0],{**events[1],'elapsed':.5}]):
        try:trace(bad,stage)
        except ValueError:rejected+=1
        else:raise AssertionError('corrupt trace accepted')
    cfg=load(CONFIG);count=0;unchanged=0
    for call in cfg['calls']:
        root=Path(cfg['output'])/call['id']
        if not (root/'cost.json').exists():continue
        terminal=load(root/'terminal.json')
        # Small receipt-prefix checks are sufficient for incomplete construction.
        if any(s['phase']=='source_check' for s in terminal['stages']):continue
        names=['invocation.json','spec.json','cost.json','terminal.json','receipt.json']+[p.name for p in root.glob('*_stage.json')]
        source={n:load(root/n) for n in names}
        mutations=[None,('budget_seconds',301),('invocation','wrong'),('stage_seconds',-1),
            ('terminal_sha256','0'*64),('receipt_sha256','0'*64),('complete_output_positive_proof',True)]
        for mutation in mutations:
            with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE',prefix='parse_receipt_') as tmp:
                data=copy.deepcopy(source)
                if mutation:data['cost.json'][mutation[0]]=mutation[1]
                for name,value in data.items():save(Path(tmp)/name,value)
                try:audit(tmp,recheck=False)
                except ValueError:
                    if mutation is None:raise
                    count+=1
                else:
                    if mutation is not None:raise AssertionError('receipt corruption accepted')
                    unchanged+=1
    return {'valid_nested_trace_accepted':1,'trace_corruptions_rejected':rejected,
        'unchanged_receipt_prefixes_accepted':unchanged,'receipt_corruptions_rejected':count,'new_solves':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--controls',action='store_true');a=p.parse_args()
    if not sys.flags.no_site:p.error('use python -S')
    if a.controls:print(json.dumps(controls()))
    else:
        value,fresh=collect()
        if a.check:
            if stable(value)!=stable(load(ARCHIVE)) or stable(fresh)!=stable(load(AUDIT)):raise ValueError('saved archive drift')
            print('PASS: frozen two-arm archive reproduces; zero new solves')
        else:
            if ARCHIVE.exists() or AUDIT.exists():raise FileExistsError('archive exists; use --check')
            save(AUDIT,fresh);save(ARCHIVE,value)
            print(json.dumps({'calls':[{k:r[k] for k in ('id','status','evidence')} for r in value['calls']],
                'total_raw_bytes':value['total_raw_bytes'],'audit':'PASS'}))
