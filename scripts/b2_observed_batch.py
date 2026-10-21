"""B2 finite failure-selected diagnostic queue; reuse owned-session supervision.

Ten guarded-pair diagnostics,41 all-pair requests,then at most19 legacy CROWN
requests,all fixed-order and charged to one six-hour stage. This is not a fair
population win-rate experiment. Missing/late obligations are never positive.
"""
import time
ENTRY=time.monotonic()
import argparse
import inspect
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha,tick,PYTHON
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
from b1_observed_batch import sizes as directory_sizes,resolve_request_folder,session_ok,require
from b1_diagnostic_queue import last_inventory
from act.pipeline.moe.genbab_spec import digest
import genbab_cpu_ready as backend

CONTROL_FILES=('scripts/b2_observed_worker.py','scripts/genbab_cpu_ready.py',
    'scripts/b1_observed_batch.py','act/back_end/moe/genbab_pair.py','act/back_end/moe/static_pair.py',
    'act/pipeline/moe/genbab_spec.py','scoped_proof/io.py','scoped_proof/owned_session.py',
    'scoped_proof/owned_bounded.py','scoped_proof/pidfd_compat.py','scoped_source/development_work.py',
    'configs/genbab_cpu_20261021_requirements.txt')


def sizes(root):
    return root.stat().st_size if root.is_file() else directory_sizes(root)


def local_selection(spec,online):
    selected=[]
    for item in spec['requests']:
        inventory=last_inventory(resolve_request_folder(online,item))
        if inventory is None:continue
        by_pair=defaultdict(set)
        for row in inventory['obligations']:
            if row['state']=='REUSED' or row.get('result')=='SAFE':continue
            by_pair[tuple(row['pair'])].add(row['property_index'])
        for pair,indices in sorted(by_pair.items()):
            selected.append(dict(kind='guarded_local',item=item,pair=list(pair),
                                 property_indices=sorted(indices),full_slot=False,
                                 selection='ACT_UNRESOLVED_DIAGNOSIS_ONLY'))
            if len(selected)==10:return selected
    return selected


def query_catalog(spec,online):
    queries=local_selection(spec,online)
    queries.extend(dict(kind='genbab_aggregate',item=item,full_slot=True) for item in spec['requests'])
    queries.extend(dict(kind='plain_crown',item=item,full_slot=True) for item in spec['requests'][:19])
    for i,row in enumerate(queries):row.update(id=f'b2_{i:03d}',seconds=300)
    require(sum(q['full_slot'] for q in queries)==60 and len(queries)<=70,'stage slots and diagnostics')
    return queries


def freeze(path,online,offline,b1_freeze):
    spec=load(b1_freeze);started=load(online/'start.json');done=load(online/'online_complete.json')
    require(started['freeze_sha256']==sha(b1_freeze) and done['completed_calls']==41
            and done['status']=='COMPLETE',
            'registered B1 queue complete')
    diagnostic=load(offline/'complete.json')
    require(diagnostic['status']=='COMPLETE','B1 registered diagnostic catalog finished, no positive gate')
    queries=query_catalog(spec,online)
    save(path,dict(schema='B2_FAILURE_SELECTED_DIAGNOSIS_V1',policy=POLICY,
        b1_freeze_sha256=sha(b1_freeze),b1_online_complete_sha256=sha(online/'online_complete.json'),
        b1_diagnostics_complete_sha256=sha(offline/'complete.json'),queries=queries,
        stage_seconds=21600,full_slots_max=60,local_slots_max=10,artifact_bytes=128*2**20,
        deadline_seconds=300,worker_seconds=294,run_seconds=296,cleanup_seconds=298,
        all_pair_policy='all unordered pairs, no borrowed route feasibility exclusions',
        genbab_configuration=backend.CONFIG,
        source_binding=backend.source_identity(),
        crown_configuration='legacy ACT route frontend plus plain CROWN whole-box variable-weight pair',
        missing='UNEXECUTED fixed remainder; no new queue, sample or permission request',
        statistical_claim='FAILURE_SELECTED_DEVELOPMENT_NOT_INDEPENDENT_WIN_RATE'))


def invoke(folder,query,commit,deadline,*,all_artifacts):
    """One existing hard supervisor call; common reception/cleanup accounting."""
    start=deadline-300
    query=dict(query,code_commit=commit,started_at=start)
    folder.mkdir();(folder/'tmp').mkdir();(folder/'cache').mkdir()
    identity=digest(query)
    remaining=128*2**20-sum(sizes(p) for p in all_artifacts)-2**20
    require(remaining>0,'remaining stage artifacts including terminal reserve')
    save(folder/'launch.json',dict(query=query,query_sha256=identity,worker_deadline=deadline-6,
                                 started_at=start,deadline=deadline,commit=commit,
                                 remaining_file_limit=remaining))
    interpreter=PYTHON if query['kind']=='plain_crown' else str(backend.ENVIRONMENT/'bin/python')
    env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'))
    row=dict(query_id=query['id'],kind=query['kind'],full_slot=query.get('full_slot',False),
             accepted=False,status='ERROR',outcome=None)
    try:
        receipt=execute([interpreter,'-I','-B',str(ROOT/'scripts/b2_observed_worker.py'),'--worker',str(folder)],
            run_deadline=deadline-4,cleanup_deadline=deadline-2,env=env,rss_limit=None,development_policy=POLICY)
        row.update(session=receipt,status=receipt['status'])
        if receipt.get('returncode')==124 and (folder/'native.log').exists():
            with (folder/'native.log').open('rb') as stream:
                stream.seek(max(0,(folder/'native.log').stat().st_size-256))
                if b'B2_AUTHORIZED_CUTOFF' in stream.read():row['status']='TIMEOUT'
        if (folder/'outcome.json').exists():
            value=load(folder/'outcome.json',limit=8*2**20)
            require(value['query_sha256']==identity and value['worker_deadline']==deadline-6,
                    'same request/property/code/deadline receipt')
            row.update(outcome_sha256=sha(folder/'outcome.json'),outcome=value['result'])
            row['accepted']=session_ok(receipt) and value['clock']<deadline-6
            if row['accepted']:row['status']=value['result']['status']
        elif row['status']=='COMPLETED':
            raise ValueError('completed worker missing bound outcome')
        tick(deadline)
        require(sum(sizes(p) for p in all_artifacts)+2**16<=128*2**20,'cumulative stage artifact limit')
    except BaseException as exc:
        row.update(accepted=False,reception_error=repr(exc),status='ERROR')
    row['seconds']=time.monotonic()-start
    save(folder/'terminal.json',row)
    within=time.monotonic()<deadline and sum(sizes(p) for p in all_artifacts)+2**16<=128*2**20
    row['accepted']=row['accepted'] and within
    if not within:row['status']='RECEPTION_LIMIT'
    row['seconds']=time.monotonic()-start
    save(folder/'reception.json',dict(accepted=row['accepted'],status=row['status'],
        seconds_through_terminal=row['seconds'],deadline=deadline,
        measurement_tail='this compact write belongs to the cumulative stage clock'))
    return row


def validate_control(check,commit,binding=None):
    require(check['result']['kind']=='synthetic_vector_control' and check['result']['accepted'],
            'ordinary multi-output API compatibility, no positivity gate')
    if check['commit']==commit:return
    require(binding is not None and check['commit']==binding['control_commit']
            and digest(check)==binding['control_identity'], 'explicit previous control identity')
    require(sha(__file__)==binding['repaired_batch_sha256']
            and digest(inspect.getsource(invoke))==binding['invoke_source_identity']
            and set(binding['unchanged_files'])==set(CONTROL_FILES)
            and all(sha(ROOT/path)==value for path,value in binding['unchanged_files'].items()),
            'unchanged controlled worker, graph, backend and supervision')


def validate_backend_binding(spec, configuration, source):
    # Frozen JSON arrays and in-memory tuples denote the same patch text.
    # Keep every source hash and configuration value in the comparison.
    require(digest(spec['genbab_configuration'])==digest(configuration)
            and digest(spec['source_binding'])==digest(source),
            'frozen external source and configuration')


def should_stop(row):
    return (row['status'] in ('ERROR','RECEPTION_LIMIT') or bool(row.get('session',{}).get('cleanup_error'))
            or row.get('session',{}).get('cleanup_status')!='LEADER_REAPED_NO_LIVE_SESSION')


def retained_prefix(spec, parent):
    """Retain all old attempts, including ERROR; execute only the untouched suffix."""
    require(parent['status']=='PREFIX_TOOLING_OR_IDENTITY_STOP', 'stopped parent catalog')
    rows=parent['rows'];queries=spec['queries']
    require(len(rows)==len(queries) and all(r['query_id']==q['id'] and r['kind']==q['kind']
            and r['full_slot']==q['full_slot'] for r,q in zip(rows,queries)), 'unchanged parent catalog')
    count=next((i for i,r in enumerate(rows) if r['status']=='UNEXECUTED'),len(rows))
    require(count>0 and all('session' in r for r in rows[:count])
            and all(r['status']=='UNEXECUTED' and 'session' not in r for r in rows[count:]),
            'contiguous attempted prefix and untouched suffix')
    require(parent['full_attempts']==sum(r['full_slot'] for r in rows[:count])
            and parent['local_attempts']==sum(not r['full_slot'] for r in rows[:count]),
            'retained attempt accounting')
    return list(rows[:count])


def validate_suffix(spec, binding, check):
    parent=load(binding['parent_complete'],binding['parent_complete_sha256'])
    rows=retained_prefix(spec,parent)
    require(binding['charged_seconds_upper_bound']>=parent['elapsed'], 'carry full failed-stage cost')
    require(digest(backend.CONFIG)==digest(spec['genbab_configuration'])
            and digest(backend.source_identity())==binding['repaired_backend_identity'],
            'same configuration, explicitly versioned compatibility source')
    expected=set(CONTROL_FILES)-{'scripts/genbab_cpu_ready.py'}
    repair=binding.get('initial_identity_read_repair')
    if repair is not None:
        path='scoped_proof/owned_session.py'
        require(repair['path']==path and sha(repair['previous_path'])==repair['previous_sha256']
                and sha(ROOT/path)==repair['current_sha256'],
                'explicit initial development identity-read compatibility version')
        expected.remove(path)
    require(check['commit']==binding['original_control_commit'] and check['result']['accepted']
            and check['result']['kind']=='synthetic_vector_control'
            and set(binding['unchanged_controlled_files'])==expected
            and all(sha(ROOT/p)==h for p,h in binding['unchanged_controlled_files'].items()),
            'reuse API control for unchanged graph/worker/solver; explicit observer repair tested separately')
    for p,h in binding['retained_files'].items():require(sha(p)==h,'retained terminal and control identity')
    parent_root=Path(binding['parent_complete']).parent
    require(all(not (parent_root/q['id']).exists() for q in spec['queries'][len(rows):]),
            'suffix has no previous execution directory')
    return rows


def frozen_commit():
    require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').exists(),'independent code copy')
    return (ROOT/'.frozen_commit').read_text().strip()


def control(root):
    """One vector-OR API control; after B1, no trained source or model selection."""
    commit=frozen_commit();require(not root.exists(),'new control root');root.mkdir()
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    query=dict(id='vector_or_control',kind='synthetic_vector_control',full_slot=False)
    row=invoke(root/'query',query,commit,ENTRY+300,all_artifacts=[root])
    save(root/'complete.json',dict(commit=commit,result=row,real_calls=0,seconds=time.monotonic()-ENTRY))
    return 0 if row['accepted'] else 1


def run(root,manifest,control_root,records=None):
    commit=frozen_commit();spec=load(manifest)
    require(spec['schema']=='B2_FAILURE_SELECTED_DIAGNOSIS_V1' and not root.exists(),'new frozen B2 stage')
    check=load(control_root/'complete.json')
    document={} if records is None else load(records)
    continuation=document.get('execution_continuation',{})
    suffix=document.get('suffix_continuation')
    if continuation:
        require(sha(manifest)==continuation['freeze_sha256']
                and sha(control_root/'complete.json')==continuation['control_complete_sha256'],
                'unchanged queue and original control receipt')
        for path,value in continuation['retained_artifacts'].items():
            require(sha(path)==value, 'retained failed-attempt artifact')
    if suffix:
        rows=validate_suffix(spec,suffix,check)
        carried=suffix['charged_seconds_upper_bound']
    else:
        validate_control(check,commit,continuation.get('control_reuse'))
        validate_backend_binding(spec,backend.CONFIG,backend.source_identity())
        carried=continuation.get('charged_seconds_upper_bound',0);rows=[]
    require(0<=carried<spec['stage_seconds'],'retained cost within original stage budget')
    deadline=ENTRY+spec['stage_seconds']-carried
    artifacts=[root,*map(Path,continuation.get('retained_artifacts',{}))]
    if continuation:artifacts.append(Path(continuation['controller_log']))
    if suffix:artifacts.extend(map(Path,suffix['additional_artifacts']))
    require(len(set(map(str,artifacts)))==len(artifacts),'no duplicate cost roots')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    root.mkdir();save(root/'start.json',dict(commit=commit,freeze_sha256=sha(manifest),entry=ENTRY,
        control_sha256=sha(control_root/'complete.json'),deadline=deadline,
        carried_seconds_upper_bound=carried,continuation_identity=digest(continuation),
        suffix_identity=None if suffix is None else digest(suffix),retained_count=len(rows),
        artifacts=[str(p) for p in artifacts]))
    stopped=False
    for query in spec['queries'][len(rows):]:
        if stopped or time.monotonic()+300>=deadline or sum(sizes(p) for p in artifacts)+2**20>=128*2**20:
            rows.append(dict(query_id=query['id'],kind=query['kind'],full_slot=query['full_slot'],
                             status='UNEXECUTED',accepted=False));continue
        row=invoke(root/query['id'],query,commit,time.monotonic()+300,all_artifacts=artifacts)
        rows.append(row)
        save(root/f'ledger_{len(rows):03d}.json',dict(completed=len(rows),last=row,
            full_attempts=sum(r['full_slot'] for r in rows if 'session' in r),
            local_attempts=sum(not r['full_slot'] for r in rows if 'session' in r),
            seconds=carried+time.monotonic()-ENTRY,bytes=sum(sizes(p) for p in artifacts)))
        print(json.dumps(dict(completed=len(rows),kind=query['kind'],status=row['status'],accepted=row['accepted'])),flush=True)
        if should_stop(row):
            stopped=True
    save(root/'complete.json',dict(status='PREFIX_TOOLING_OR_IDENTITY_STOP' if stopped else 'CATALOG_FINISHED',
        rows=rows,elapsed=carried+time.monotonic()-ENTRY,bytes=sum(sizes(p) for p in artifacts),
        carried_seconds_upper_bound=carried,
        full_attempts=sum(r['full_slot'] for r in rows if 'session' in r),
        local_attempts=sum(not r['full_slot'] for r in rows if 'session' in r),
        guarantee='NUMERICAL_FILTERS_WITH_ALL_MISSING_OBLIGATIONS_RETAINED'))
    return 1 if stopped else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',type=Path);parser.add_argument('--online',type=Path)
    parser.add_argument('--offline',type=Path);parser.add_argument('--b1-freeze',type=Path)
    parser.add_argument('--control',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--control-root',type=Path)
    parser.add_argument('--records',type=Path);args=parser.parse_args()
    if args.freeze:freeze(args.freeze,args.online,args.offline,args.b1_freeze)
    elif args.control:raise SystemExit(control(args.control))
    elif args.run:raise SystemExit(run(args.run,args.manifest,args.control_root,args.records))
    else:parser.error('explicit operation required')
