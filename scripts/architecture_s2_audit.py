"""S2a recorded-evidence audit and complete primary/secondary analysis.

No solver, checkpoint load, bound recomputation or witness replay occurs here.
Online accepted evidence is recounted, never manufactured by this audit.
"""
from collections import Counter
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,sha
from architecture_s1_execution import Artifacts,read_record,require,digest
from architecture_s1_audit import audit_request
from architecture_s2_study import all_requests,ARMS
from act.pipeline.moe.architecture_s2_evaluation import unpack_arrays,check_inputs,check_primary_arrays
from act.pipeline.moe.architecture_s2_statistics import primary_summary,verification_summary
from act.pipeline.moe.architecture_statistics import model_summary


def check_secondary(value,query,model,spec,label):
    require(value['backend_sources_sha256']==spec['backend_sources_sha256']
        and value['family']==model['family'],'same frozen secondary engine/family')
    require(value['properties']==9 and value['label']==label,'all original-label properties')
    require(value['result']['evaluation']['family']==model['family'],'graph family')
    if query['mode']=='complete':
        require(value['result']['evaluation'].get('explicit_complete_configuration')==spec['complete_configuration'],
                'same common complete-verifier configuration')
        require(all(h['outcome'].get('branching')==spec['complete_configuration']['bab.branching.method']
                    for h in value['result']['histories']),'recorded common branch heuristic')
    else:
        require(query['mode']=='incomplete','registered secondary mode')
        require(all(h['outcome'].get('bound_propagation')=='CROWN-IBP'
                    and h['outcome'].get('complete_verifier') is None
                    for h in value['result']['histories']),'incomplete bound method only')
    require(type(value['clean_correct']) is bool
            and value['clean_correct']==(value['clean_prediction']==label),'secondary clean flag')
    require(value['result']['status']!='POSITIVE' or value['clean_correct'],
            'clean mistakes cannot be numerical positives')


def check_primary(root,query,value,model,spec,labels):
    require(value['family']==model['family'] and value['checkpoint_sha256']==model['checkpoint_sha256']
        and value['input_sha256']==spec['inputs']['sha256'] and value['count']==10000,
        'complete primary source and denominator')
    require(value['torch']==spec['primary_torch_version']
        and value['bound_source_sha256']==spec['source_files']['act/pipeline/moe/route_envelope_ibp.py'],
        'same frozen training/primary bound function')
    arrays=unpack_arrays(read_record(root/query['id']/'arrays.json.gz'))
    check_primary_arrays(arrays,family=model['family'],count=10000)
    require(np.array_equal(arrays['label'],labels),'same physical input labels and order')
    require(value['clean']==int(arrays['clean'].sum())
        and value['certified']==int(arrays['certified'].sum()),'same online primary counts')
    batches=value['batches'];size=spec['primary_batch_size']
    require([(b['start'],b['stop']) for b in batches]==[(s,min(10000,s+size)) for s in range(0,10000,size)],
            'complete ordered primary batches')
    require(all(np.isfinite(b['seconds']) and b['seconds']>=0 for b in batches),'finite batch costs')
    for b in batches:
        region=slice(b['start'],b['stop'])
        require(b['count']==b['stop']-b['start'] and b['clean']==int(arrays['clean'][region].sum())
                and b['certified']==int(arrays['certified'][region].sum()),'batch statistics match raw arrays')
    if model['family']=='moe':
        require(value['single_route']==int((arrays['route_count']==1).sum()),'route stability count')
    return arrays


def analyze(root):
    began=time.monotonic();root=Path(root);manifest=root/'manifest.json';spec=load(manifest);mh=sha(manifest)
    require(spec['study']=='S2A','registered S2a study')
    require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'frozen analysis source')
    requests=all_requests(spec['models'])
    require(digest(requests)==spec['request_catalog_sha256'],'full fixed request roster')
    start=read_record(root/'start.json.gz');require(start['manifest_sha256']==mh,'execution freeze')
    # A live queue may still be publishing ledgers; only settled prefixes get
    # this final audit. The separate primary-report entry remains available.
    queue=read_record(root/'queue_complete.json.gz')
    require(queue['source_manifest_sha256']==mh and queue['request_catalog_sha256']==digest(requests),
            'final queue identity')
    require(sha(spec['inputs']['path'])==spec['inputs']['sha256'],'shared physical input file')
    inputs=unpack_arrays(read_record(spec['inputs']['path']));check_inputs(inputs)
    models={m['id']:m for m in spec['models']};records={};primary={};primary_details={}
    states=Counter();by_mode={};missing_cost=0;identity=[];bounds_seconds=0.
    for q in requests:
        row,value=audit_request(root,q,mh)
        states[row['status']]+=1;mode=q['mode'];stats=by_mode.setdefault(mode,dict(states=Counter(),seconds=0.))
        stats['states'][row['status']]+=1
        if row['seconds'] is not None:
            require(np.isfinite(row['seconds']) and row['seconds']>=0,'finite charged cost')
            stats['seconds']+=row['seconds']
        else:missing_cost+=row['status']!='NOT_EXECUTED'
        lp=root/(q['id']+'.ledger.json.gz')
        if lp.exists():identity.append((q['id'],sha(lp)))
        model=models[q['model']]
        if mode=='primary':
            if value is not None:
                arrays=check_primary(root,q,value,model,spec,inputs['labels'])
                primary[q['model']]=arrays
                batch_cost=sum(b['seconds'] for b in value['batches']);bounds_seconds+=batch_cost
                primary_details[q['model']]=dict(request_seconds=row['seconds'],
                    batch_compute_seconds=batch_cost,amortized_batch_seconds_per_input=batch_cost/10000,
                    mean_margin_width=float((arrays['upper_margin'].astype(np.float64)
                                             -arrays['lower_margin'].astype(np.float64)).mean()),
                    raw_margin_sha256=value['array_sha256'])
        else:
            if value is not None:check_secondary(value,q,model,spec,int(inputs['labels'][q['index']]))
            row['index']=q['index'];records[(mode,q['model'],q['index'])]=row
    charged=sum(m['seconds'] for m in by_mode.values());not_executed=states['NOT_EXECUTED']
    require(queue['settled']==len(requests)-not_executed and queue['not_executed']==not_executed,
            'all terminal and unexecuted denominators')
    require(queue['states']=={s:states[s] for s in queue['states']},'all online terminal counts')
    require(abs(queue['summed_request_seconds']-charged)<1e-5 and queue['missing_request_cost']==missing_cost,
            'all startup/worker/receive/cleanup costs included')
    require(missing_cost==0,'complete executed-cost record')
    complete=queue['status']=='COMPLETE' and not_executed==0
    options={k:spec['bootstrap'][k] for k in ('resamples','seed')}
    primary_result=None
    if len(primary)==20:
        shape=(4,5,10000);cert=np.zeros(shape,dtype=bool);clean=np.zeros(shape,dtype=bool)
        routes=np.zeros(shape,dtype=np.uint8)
        for m in models.values():
            a,s=ARMS.index(m['arm']),m['seed'];v=primary[m['id']]
            cert[a,s]=v['certified'];clean[a,s]=v['clean'];routes[a,s]=v['route_count']
        primary_result=primary_summary(cert,clean,routes,**options)
        old=root/'primary_summary.json.gz'
        if old.exists():
            prior=read_record(old)
            require(all(prior[k]==v for k,v in primary_result.items()),'primary report differential')
    comparisons={};tables=[];clean_disagreements=[];numeric_conflicts=[]
    for mode,count in [('complete',200),('incomplete',1000)]:
        ss=np.full((4,5,count),'NOT_EXECUTED',dtype='<U20');cost=np.full(ss.shape,np.nan)
        for m in models.values():
            rows=[records[(mode,m['id'],i)] for i in range(count)]
            a,s=ARMS.index(m['arm']),m['seed']
            for i,r in enumerate(rows):
                ss[a,s,i]=r['status'];cost[a,s,i]=r['seconds'] if r['seconds'] is not None else np.nan
                if m['id'] in primary:
                    p=primary[m['id']]
                    if r['clean_correct'] is not None and bool(p['clean'][i])!=r['clean_correct']:
                        clean_disagreements.append(dict(mode=mode,model=m['id'],index=i,
                            float32_primary=bool(p['clean'][i]),float64_secondary=r['clean_correct']))
                    if p['certified'][i] and r['status']=='UNSAFE_REPLAYED':
                        numeric_conflicts.append(dict(mode=mode,model=m['id'],index=i,
                            kind='float32_IBP_positive_vs_float64_replay'))
            tables.append(dict(mode=mode,model=m['id'],arm=m['arm'],seed=m['seed'],**model_summary(rows,count)))
        comparisons[mode]=verification_summary(ss,cost,expected_inputs=count,**options)
    for m in models.values():
        for i in range(200):
            statuses={records[(mode,m['id'],i)]['status'] for mode in ('complete','incomplete')}
            if {'POSITIVE','UNSAFE_REPLAYED'}<=statuses:
                numeric_conflicts.append(dict(model=m['id'],index=i,kind='same_float64_positive_vs_replay'))
    account=load(root/'artifact_account.json')
    physical=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    require(account['cap']==spec['artifact_bytes'] and
            physical+account['initial_bytes']<=account['used']<=account['cap'],'shared evidence account')
    decision=(primary_result['decision'] if complete and primary_result is not None
              else 'PENDING_COMPLETE_S2A_EVIDENCE')
    return dict(status='COMPLETE_AUDIT' if complete else 'PREFIX_AUDIT',
        source_manifest_sha256=mh,record_identity_sha256=digest(identity),
        online_states=dict(states),planned_requests=len(requests),primary_measurements=len(primary),
        charged_request_seconds=charged,queue_wall_seconds=queue['queue_wall_seconds'],
        by_mode=by_mode,missing_executed_cost=missing_cost,
        primary=primary_result,primary_cost_and_width=primary_details,
        secondary=comparisons,model_tables=tables,
        clean_precision_disagreements=clean_disagreements,numerical_conflicts=numeric_conflicts,
        registered_decision=decision,requires_evidence_investigation=bool(numeric_conflicts),
        artifact_physical_bytes=physical,artifact_charged_bytes=account['used'],
        preparation_seconds=spec['preparation_seconds'],primary_batch_compute_seconds=bounds_seconds,
        controller_startup_and_admission_seconds=start.get('startup_and_admission_seconds'),
        cost_scope='queue wall and summed full requests separately; shared prep/startup/train/incomplete separately; prep starts at Python entry (pre-entry and final-write tail excluded); batch time not cold input latency',
        audit_seconds=time.monotonic()-began,auditor_sha256=sha(__file__),
        scientific_scope='numerical-policy comparison; independent record recount, not a rerun of bounds, witnesses or source proof')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',default='evaluation_summary.json.gz');args=parser.parse_args()
    require(Path(args.output).name==args.output,'new local summary filename')
    value=analyze(args.root);Artifacts(args.root).write(args.root/args.output,value,terminal=True)
    print(value['status'],value['registered_decision'],value['online_states'])
