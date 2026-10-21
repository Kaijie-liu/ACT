"""Finite historical-success diagnosis; reuse B2 workers and hard supervision.

ACT historical acceptance is a reference, not a fresh/source-complete proof.
A negative CROWN lower bound is an unresolved sufficient condition, not a
feasible violation or proof that a guard is necessary.
"""
import time
ENTRY=time.monotonic()
import argparse
import ast
import json
import math
import resource
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from b1_observed_batch import ROWS,SELECTION,SEALED,require
import b2_observed_batch as batch
import b2_observed_worker as worker
from act.pipeline.moe.genbab_spec import digest

SELECTION_LABEL='ACT_HISTORICAL_SUCCESS_SELECTED_DIAGNOSIS_NOT_FAIR_POPULATION_COMPARISON'
B2_COPY=Path('/data1/Kane/MOE/executions/b2_observed_20261021_7af2c5ff2')


def unchanged_numerical_worker():
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.FunctionDef)}
    before=functions(B2_COPY/'scripts/b2_observed_worker.py');after=functions(ROOT/'scripts/b2_observed_worker.py')
    names=('properties','obligation_status','load_model','full_replay','verify_graph')
    require(all(before[n]==after[n] for n in names),'unchanged B2 numerical API and replay functions')
    return dict(prior_copy=str(B2_COPY),functions={n:digest(after[n]) for n in names})


def select(rows,coverage):
    safe=[(i,r) for i,r in enumerate(rows) if r['method']=='adaptive'
          and r['status']=='SAFE' and r['dataset_index'] not in SEALED]
    order=lambda x:(x[1]['rank'],int(x[1]['model'][4:]))
    multi=sorted(((i,r) for i,r in safe if len(coverage[r['job_id']])>=2),key=order)
    single=sorted(((i,r) for i,r in safe if len(coverage[r['job_id']])==1),key=order)[:9]
    require(len(multi)==51 and len(single)==9,'authorized 51 multi-pair plus nine single-pair controls')
    return [(i,r,'multi_pair') for i,r in multi]+[(i,r,'single_pair_control') for i,r in single]


def make_freeze():
    rows=[json.loads(s) for s in ROWS.read_text().splitlines()]
    coverage={};evidence={}
    for r in rows:
        if r['method']!='adaptive' or r['status']!='SAFE' or r['dataset_index'] in SEALED:continue
        p=Path(r['package'])/'evidence.json';e=load(p)
        require(e['route_coverage']['route_sets_exact'],'historical numerical roster complete')
        coverage[r['job_id']]=e['route_coverage']['feasible_route_sets'];evidence[r['job_id']]=(p,e)
    selection=load(SELECTION);requests=[];sources={str(ROWS):sha(ROWS),str(SELECTION):sha(SELECTION)}
    for ordinal,(row_index,r,group) in enumerate(select(rows,coverage)):
        p,e=evidence[r['job_id']];folder=p.parent.parent;sample=selection['samples'][r['rank']]
        require(sample['dataset_index']==r['dataset_index'],'sample identity')
        subject=selection['models'][r['model']];require(sha(subject['checkpoint'])==subject['checkpoint_sha256'],'checkpoint identity')
        require(e['execution']['git_head']=='bc0791976b00879c28c268692aecc5854b3bc091','historical verifier version')
        for f in (p,p.parent/'manifest.json',folder/'terminal.json'):sources[str(f)]=sha(f)
        item=dict(ordinal=ordinal,job_id=f"{ordinal:02d}_{r['model']}_{r['dataset_index']}",model=r['model'],
            dataset_index=r['dataset_index'],previous_status='SAFE',historical_job=r['job_id'],
            subject=subject,sample=sample,tensors=dict(path=str(p.parent/'request.pt'),sha256=sha(p.parent/'request.pt')),
            epsilon=selection['request']['epsilon'])
        requests.append(dict(id=f'adv_{ordinal:03d}',kind='genbab_aggregate',full_slot=True,seconds=300,
            selection=SELECTION_LABEL,item=item,historical=dict(group=group,row_index=row_index,
            pairs=coverage[r['job_id']],wall_seconds=r['wall_seconds'],evidence_sha256=sha(p),
            code_commit='bc0791976b00879c28c268692aecc5854b3bc091')))
    b2=load(ROOT/'docs/b2_observed_freeze_20261021.json')
    # Bind the actual already disclosed B2 r2 backend, not the original pre-repair snapshot.
    backend_identity=batch.backend.source_identity()
    require(digest(backend_identity)==load(ROOT/'docs/b2_observed_records_20261021.json')['suffix_continuation']['repaired_backend_identity']
            and digest(batch.backend.CONFIG)==digest(b2['genbab_configuration']),'unchanged final B2 backend/config')
    return dict(schema='ACT_ADVANTAGE_REGION_DIAGNOSIS_V1',selection=SELECTION_LABEL,queries=requests,
        order='all multi-pair first, then nine single-pair; each by rank then numeric seed',sources=sources,
        code_files={p:sha(ROOT/p) for p in (*batch.CONTROL_FILES,'act/pipeline/moe/external_pair_worker.py',
            'scripts/advantage_region.py','scripts/b2_observed_batch.py')},
        genbab_configuration=batch.backend.CONFIG,backend_identity=digest(backend_identity),
        numerical_control_reuse=unchanged_numerical_worker(),
        b2_complete_sha256=load(ROOT/'docs/b2_observed_records_20261021.json')['final_stage_result']['complete_sha256'],
        stage_seconds=21600,full_slots_max=60,local_slots_max=51,request_seconds=300,artifact_bytes=128*2**20,
        local_rule='after all full calls; one first returned unresolved pair per accepted multi-pair UNKNOWN, original request order',
        decision=dict(continue_at_least=3,stop_at_most=2,
            pattern='historical multi-pair SAFE, GenBaB accepted UNKNOWN, selected whole-box pair has a finite negative CROWN lower bound',
            interpretation='observed numerical nonclosure pattern, not guard necessity, exact feasible violation, or source-complete ACT advantage',
            missing='retain missing/error/unexecuted; cannot treat these as negative evidence of a mechanism'),
        history='ACT references only, no rerun; input/source gaps and old timings/versions retained')


def local_queries(queries,rows):
    result=[]
    for query,row in zip(queries,rows):
        if query['historical']['group']!='multi_pair' or not row.get('accepted') or row['status']!='UNKNOWN':continue
        unresolved=[p for p in row['outcome']['pairs'] if p['outcome']['status']!='verified']
        if not unresolved:continue
        pair=unresolved[0]['pair']
        result.append(dict(id=f'adv_diag_{len(result):03d}',kind='plain_crown',full_slot=False,seconds=300,
            item=query['item'],diagnostic_pair=pair,parent_query=query['id'],selection=SELECTION_LABEL,
            in_historical_roster=pair in query['historical']['pairs']))
    return result


def decision(queries,rows):
    hits=[];missing=[];violations=[]
    for q,r in zip(queries,rows):
        if r.get('accepted') and r.get('status')=='UNSAFE_REPLAYED':
            violations.append(q['parent_query']);continue
        if not r.get('accepted') or not r.get('outcome',{}).get('pairs'):
            missing.append(q['parent_query']);continue
        pairs=r['outcome']['pairs'];require(len(pairs)==1 and pairs[0]['pair']==q['diagnostic_pair'],'same diagnostic pair')
        lows=pairs[0]['lower'];require(len(lows)==9 and all(math.isfinite(v) for v in lows),'all finite output properties')
        if min(lows)<0:hits.append(dict(parent=q['parent_query'],pair=q['diagnostic_pair'],minimum=min(lows),
            in_historical_roster=q['in_historical_roster']))
    return dict(pattern_requests=len(hits),hits=hits,missing=missing,replayed_violation_conflicts=violations,
        expected_local_requests=len(queries),observed_local_requests=len(queries)-len(missing),
        roster_inside_hits=sum(h['in_historical_roster'] for h in hits),
        roster_outside_hits=sum(not h['in_historical_roster'] for h in hits),
        reason=('OBSERVED_PATTERN_THRESHOLD' if len(hits)>=3 else
                'INSUFFICIENT_DIAGNOSTIC_COVERAGE' if missing else 'OBSERVED_PATTERN_AT_MOST_TWO'),
        next='GUARD_CONDITIONED_PROPAGATION_RESEARCH_CANDIDATE' if len(hits)>=3 else 'STOP_B3_MECHANISM_SELECTION_REPORT',
        guarantee='NEGATIVE_LOWER_IS_NOT_GUARD_NECESSITY_OR_MODEL_UNSAFETY')


def compact(row,root):
    out={k:row[k] for k in ('query_id','kind','full_slot','accepted','status','seconds') if k in row}
    folder=root/row['query_id']
    if (folder/'terminal.json').is_file():
        out.update(terminal=str(folder/'terminal.json'),terminal_sha256=sha(folder/'terminal.json'),
                   reception_sha256=sha(folder/'reception.json'))
    return out


def run(root,manifest,controller_log):
    commit=batch.frozen_commit();spec=load(manifest)
    require(spec['schema']=='ACT_ADVANTAGE_REGION_DIAGNOSIS_V1' and not root.exists(),'new frozen stage')
    require(all(sha(p)==h for p,h in spec['sources'].items())
            and all(sha(ROOT/p)==h for p,h in spec['code_files'].items()),'frozen source/code binding')
    require(digest(batch.backend.CONFIG)==digest(spec['genbab_configuration'])
            and digest(batch.backend.source_identity())==spec['backend_identity'],'unchanged external method')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30));root.mkdir();deadline=ENTRY+spec['stage_seconds']
    artifacts=[root,controller_log];require(controller_log.is_file(),'charged controller log')
    save(root/'start.json',dict(commit=commit,freeze_sha256=sha(manifest),entry=ENTRY,deadline=deadline,
                              artifacts=list(map(str,artifacts)),full_slots=60))
    rows=[];stopped=False
    def call(q):
        nonlocal stopped
        if stopped or time.monotonic()+300>=deadline or sum(batch.sizes(p) for p in artifacts)+2**20>=spec['artifact_bytes']:
            return dict(query_id=q['id'],kind=q['kind'],full_slot=q['full_slot'],status='UNEXECUTED',accepted=False)
        r=batch.invoke(root/q['id'],q,commit,time.monotonic()+300,all_artifacts=artifacts)
        stopped=batch.should_stop(r)
        return r
    for q in spec['queries']:
        rows.append(call(q));save(root/f'ledger_{len(rows):03d}.json',dict(last=rows[-1],completed=len(rows),
            seconds=time.monotonic()-ENTRY,bytes=sum(batch.sizes(p) for p in artifacts)))
        print(json.dumps(dict(query=q['id'],status=rows[-1]['status'])),flush=True)
    localspec=local_queries(spec['queries'],rows);require(len(localspec)<=spec['local_slots_max'],'bounded local catalog')
    save(root/'local_catalog.json',dict(queries=localspec,rule=spec['local_rule'],full_rows_sha256=digest(rows)))
    localrows=[]
    for q in localspec:
        localrows.append(call(q));save(root/f'local_ledger_{len(localrows):03d}.json',dict(last=localrows[-1],
            seconds=time.monotonic()-ENTRY,bytes=sum(batch.sizes(p) for p in artifacts)))
        print(json.dumps(dict(query=q['id'],status=localrows[-1]['status'])),flush=True)
    save(root/'complete.json',dict(status='PREFIX_STOP' if stopped else 'CATALOG_FINISHED',
        rows=[compact(r,root) for r in rows],local_queries=localspec,
        local_rows=[compact(r,root) for r in localrows],decision=decision(localspec,localrows),
        full_attempts=sum('session'in r for r in rows),local_attempts=sum('session'in r for r in localrows),
        seconds_before_final_record=time.monotonic()-ENTRY,formal_SAFE=False))
    require(sum(batch.sizes(p) for p in artifacts)+2**16<=spec['artifact_bytes'] and time.monotonic()<deadline,
            'complete record within stage budget')
    save(root/'stage_receipt.json',dict(seconds_through_complete=time.monotonic()-ENTRY,
        bytes_through_complete=sum(batch.sizes(p) for p in artifacts),
        complete_sha256=sha(root/'complete.json'),measurement_tail='this compact receipt and interpreter exit'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-stdout',action='store_true')
    parser.add_argument('--run',type=Path);parser.add_argument('--manifest',type=Path);parser.add_argument('--controller-log',type=Path)
    args=parser.parse_args()
    if args.freeze_stdout:print(json.dumps(make_freeze(),sort_keys=True))
    elif args.run:run(args.run,args.manifest,args.controller_log)
    else:parser.error('explicit freeze or run')
