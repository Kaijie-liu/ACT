"""Saved-only frozen frontier comparison archive. Never opens a checkpoint/solver."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def forbid(event,args):
    if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','highspy','act','gurobipy') or
            args[0] in ('checked_route_frontier.build','scoped_source.build','full_source.obligations')):
        raise ImportError('archive cannot import model/solver/producer')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.fork','os.exec'):
        raise PermissionError('archive cannot launch work')


sys.addaudithook(forbid)
from frontier_proof.batch_audit import review
from scoped_proof.evidence import POSITIVE
from scoped_proof.io import load, save, sha
from source_enclosure.format import identity

CONFIG=ROOT/'configs/backend_controls/frontier_proof_compare_r1.json'
FREEZE=ROOT/'docs/frontier_proof_freeze_review_20260924_r1.json'
AUDIT=ROOT/'docs/frontier_proof_execution_audit_20260924_r1.json'
ARCHIVE=ROOT/'docs/frontier_proof_execution_archive_20260924_r1.json'


def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in ('separate_audit_seconds','archive_seconds')}
    if isinstance(value,list):return [stable(v) for v in value]
    return value


def operations(raw,stage):
    lines=raw.splitlines(keepends=True);partial=0
    if lines and not lines[-1].endswith(b'\n'):partial=len(lines.pop())
    events=[json.loads(line) for line in lines];pending=None;closed=[];previous=stage['start_seconds']
    for e in events:
        elapsed=e['elapsed']
        if not math.isfinite(elapsed) or not previous<=elapsed<=stage['end_seconds']:
            raise ValueError('event outside charged ordered phase')
        previous=elapsed
        if e['event']=='ENTER':
            if pending is not None:raise ValueError('unexpected overlapping worker operation')
            pending=e
        elif e['event'] in ('EXIT','EXIT_ERROR'):
            if pending is None or pending['operation']!=e['operation']:
                raise ValueError('event identity/order')
            if not math.isfinite(e['seconds']) or not 0<=e['seconds']<=stage['end_seconds']-stage['start_seconds']:
                raise ValueError('invalid operation time')
            closed.append({'operation':e['operation'],'status':e['event'],'seconds':e['seconds'],
                           'entered_seconds':pending['elapsed'],'finished_seconds':elapsed})
            pending=None
        else:raise ValueError('unknown event')
    return {'phase':stage['phase'],'completed_operations':closed,
        'open_operation_at_stop':None if pending is None else {'operation':pending['operation'],
            'entered_seconds':pending['elapsed'],'observed_seconds_to_stop':stage['end_seconds']-pending['elapsed']},
        'event_count':len(events),'trailing_partial_bytes':partial,
        'native_lp_calls_started':sum(e['event']=='ENTER' and e['operation'].startswith('native_lp_') for e in events)}


def cost_check(inv,terminal,receipt,cost,returned):
    """A second small receipt check independent of the frozen audit implementation."""
    if (any(v['invocation']!=inv['invocation'] for v in (terminal,receipt,cost)) or
            terminal['request_sha256']!=inv['request_sha256'] or cost['request_sha256']!=inv['request_sha256'] or
            cost['terminal_sha256']!=identity(terminal) or receipt['terminal_sha256']!=identity(terminal) or
            cost['receipt_sha256']!=identity(receipt) or
            any(v['budget_seconds']!=300 for v in (inv,terminal,cost)) or
            terminal['required']!=252 or terminal['acceptance_requires_cost_receipt'] is not True or
            cost['complete_output_positive_proof']!=(cost['status']==POSITIVE)):
        raise ValueError('terminal identity/coverage/budget')
    if (abs(inv['deadline_monotonic']-inv['started_monotonic']-300)>1e-8 or
            abs(inv['deadline_monotonic']-inv['work_deadline_monotonic']-2)>1e-8):
        raise ValueError('absolute deadline')
    vals=[cost[k] for k in ('stage_seconds','overhead_seconds','end_to_end_seconds')]+[returned]
    if (not all(math.isfinite(v) and v>=0 for v in vals) or
            abs(vals[0]-sum(s['seconds'] for s in terminal['stages']))>1e-8 or
            abs(vals[0]+vals[1]-vals[2])>1e-8 or returned<vals[2] or
            vals[2]<receipt['seconds_before_receipt'] or vals[2]<terminal['seconds_before_publication'] or
            receipt['status']!=terminal['status_before_publication']):
        raise ValueError('cost closure')
    if cost['status']==POSITIVE and (vals[2]>=300 or returned>=300 or any(s['status']!='COMPLETED' for s in terminal['stages'])):
        raise ValueError('late/failed phase cannot close')


def controls(records):
    rejected=0
    for inv,terminal,receipt,cost,seconds in records:
        cost_check(inv,terminal,receipt,cost,seconds)
        for key,value in (('budget_seconds',301),('invocation','wrong'),('stage_seconds',-1),
                          ('terminal_sha256','0'*64),('receipt_sha256','0'*64),
                          ('complete_output_positive_proof',not cost['complete_output_positive_proof'])):
            bad=copy.deepcopy(cost);bad[key]=value
            try:cost_check(inv,terminal,receipt,bad,seconds)
            except ValueError:rejected+=1
            else:raise AssertionError('cost corruption accepted')
    stage={'phase':'control','start_seconds':0.,'end_seconds':3.}
    start={'event':'ENTER','operation':'x','elapsed':1.}
    end={'event':'EXIT','operation':'x','elapsed':2.,'seconds':1.}
    encode=lambda rows:b''.join((json.dumps(r)+'\n').encode() for r in rows)
    assert operations(encode([start]),stage)['open_operation_at_stop']['observed_seconds_to_stop']==2.
    assert operations(encode([start,end])+b'{partial',stage)['trailing_partial_bytes']==8
    traces=0
    for rows in ([{**start,'elapsed':4.}], [start,{**end,'operation':'wrong'}],
                 [start,{**end,'seconds':-1}], [start,{**end,'elapsed':.5}]):
        try:operations(encode(rows),stage)
        except ValueError:traces+=1
        else:raise AssertionError('trace corruption accepted')
    return {'valid_actual_cost_records':len(records),'cost_corruptions_rejected':rejected,
            'trace_corruptions_rejected':traces,'open_and_partial_trace_controls_passed':2,'new_solves':0}


def collect():
    start=time.monotonic();freeze=load(FREEZE);cfg=load(CONFIG,freeze['config_sha256']);root=Path(cfg['output'])
    launch=load(root/'launch.json')
    if launch['config']!=cfg or launch['config_sha256']!=sha(CONFIG):raise ValueError('not the frozen launch')
    for name,digest in cfg['common']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('execution source drift')
    fresh=review(root);details=[];records=[]
    for row in fresh['rows']:
        folder=root/row['id'];observations=[]
        if row['status'] in ('NOT_STARTED_RESOURCE','SUPERVISOR_ERROR'):
            details.append({'id':row['id'],'status':row['status'],'evidence':None});continue
        inv=load(folder/'invocation.json');terminal=load(folder/'terminal.json');receipt=load(folder/'receipt.json');cost=load(folder/'cost.json')
        records.append((inv,terminal,receipt,cost,row['seconds']))
        for stage in terminal['stages']:
            path=folder/(stage['phase']+'_events.jsonl')
            observations.append(operations(path.read_bytes() if path.exists() else b'',stage))
        proof=load(folder/'evidence_check.json') if (folder/'evidence_check.json').exists() else None
        route=load(folder/'route_check.json') if (folder/'route_check.json').exists() else None
        candidates=sorted(p.name for p in (folder/'candidates').glob('*.json'))
        route_candidates=sorted(p.name for p in (folder/'route_candidates').glob('*.json'))
        output_bounds=0 if proof is None else proof['checked_bounds']
        positive_bounds=0 if proof is None else proof['positive_bounds']
        # A saved, freshly rechecked route receipt supplies logical exclusions
        # even when complete output construction/aggregation was not reached.
        excluded_pairs=0 if route is None else route['frontier']['excluded_pairs']
        excluded_duties=excluded_pairs*(cfg['common']['scope']['classes']-1)
        completed_phases={s['phase'] for s in terminal['stages'] if s['status']=='COMPLETED'}
        evidence={'source_captured':(folder/'source.json').exists(),
            'router_prefix_published':(folder/'router_prefix.json').exists(),
            'route_candidate_files':route_candidates,'route_completion_published':(folder/'route_complete.json').exists(),
            'route_receipt_published':route is not None,'checked_excluded_pairs':excluded_pairs,
            'route_check_phase_completed': 'route_check' in completed_phases,
            'strict_router_bounds':None if route is None else sum(b['status']=='STRICT_DOMINANCE' for b in route['frontier']['bounds']),
            'route_discharged_duties':excluded_duties,
            'retained_pairs_after_checked_route_receipt':None if route is None else route['frontier']['retained_pairs'],
            'retained_pair_list':None if route is None else [p['pair'] for p in route['frontier']['pairs'] if p['status']=='RETAINED'],
            'needed_experts_after_checked_route_receipt':None if route is None else route['frontier']['needed_experts'],
            'construction_published':(folder/'construction.json').exists(),
            'source_check_receipt_published':(folder/'source_check.json').exists(),
            'native_lp_calls_started':sum(o['native_lp_calls_started'] for o in observations),
            'output_candidate_files':candidates,'independently_checked_output_bounds':output_bounds,
            'positive_output_bounds':positive_bounds,
            'duties_without_positive_checked_evidence':cfg['required_per_call']-excluded_duties-positive_bounds,
            'final_aggregation_published':proof is not None,'complete_output_positive_proof':row['positive'],
            'native_float_proof':False,'route_changing_established':False}
        details.append({'id':row['id'],'status':row['status'],'return_seconds':row['seconds'],'cost':cost,
                        'stages':terminal['stages'],'operations':observations,'evidence':evidence})
    files={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('raw evidence symlink')
        if path.is_file():files[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':sha(path)}
    result={'schema':'FRONTIER_PROOF_EXECUTION_ARCHIVE_R1','audit':fresh['audit'],'issues':fresh['issues'],
        'config_sha256':sha(CONFIG),'launch_head':launch['head'],'implementation_commit':cfg['implementation_commit'],
        'dataset_index':cfg['sample']['dataset_index'],'required_per_call':cfg['required_per_call'],
        'frozen_source_files_checked':len(cfg['common']['sources']),'calls':details,'frozen_batch_audit':stable(fresh),
        'raw_root':str(root),'files':files,'total_raw_bytes':sum(v['bytes'] for v in files.values()),
        'archive_source_sha256':sha(__file__),'archive_controls':controls(records),
        'archive_seconds':time.monotonic()-start,'new_solver_calls':0,
        'scope':'saved identities, all terminal costs, available exact evidence; no missing obligation recovery',
        'decision':'SEAL_TWO_FROZEN_CALLS; no retries, samples, extra time or old proof reuse'}
    return result,fresh


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    if not sys.flags.no_site:p.error('use python -S')
    result,fresh=collect()
    if a.check:
        if stable(result)!=stable(load(ARCHIVE)) or stable(fresh)!=stable(load(AUDIT)):raise ValueError('archive drift')
        print('PASS: frozen two-call archive reproduces, no new solves')
    else:
        if ARCHIVE.exists() or AUDIT.exists():raise FileExistsError('append-only archive exists; use --check')
        save(AUDIT,fresh);save(ARCHIVE,result)
        print(json.dumps({'audit':result['audit'],'issues':result['issues'],'calls':[
            {'id':r['id'],'status':r['status'],'evidence':r['evidence']} for r in result['calls']],
            'raw_files':len(result['files']),'raw_bytes':result['total_raw_bytes'],'archive_controls':result['archive_controls']}))
