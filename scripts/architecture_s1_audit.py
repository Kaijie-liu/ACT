"""S1 terminal/coverage audit and registered paired-input analysis.

Read-only on original execution evidence. This is structural/numerical-policy
analysis, not an independent rerun of CROWN bounds or model witnesses.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,sha,save
from architecture_s1_execution import read_record,digest,audit_value,audit_pgd,require,Artifacts
from architecture_s1_study import all_requests
from act.pipeline.moe.architecture_statistics import model_summary,paired,s1_decision


def audit_request(root,query,manifest_sha256):
    ledger_path=root/(query['id']+'.ledger.json.gz')
    if not ledger_path.exists():
        return dict(status='NOT_EXECUTED',seconds=None,clean_correct=None),None
    ledger=read_record(ledger_path)
    require(ledger['id']==query['id'],'ledger request identity')
    if 'terminal_sha256' not in ledger:
        require(ledger['status']=='ERROR' and not ledger['accepted'] and 'controller_error' in ledger,
            'explicit controller failure')
        return dict(status='ERROR',seconds=ledger.get('seconds'),clean_correct=None,
            cost_missing='seconds' not in ledger),None
    folder=root/query['id'];terminal_path=folder/'terminal.json.gz'
    require(sha(terminal_path)==ledger['terminal_sha256'],'terminal hash')
    terminal=read_record(terminal_path)
    require(terminal['id']==query['id'],'terminal identity')
    for name,h in terminal['files'].items():
        require(Path(name).name==name and sha(folder/name)==h,'terminal evidence binding '+name)
    request=read_record(folder/'request.json.gz')
    require(request['manifest_sha256']==manifest_sha256 and all(request[k]==v for k,v in query.items()),
        'frozen request/model/input/radius/mode')
    require(request['hard_deadline']==request['started_at']+300,'300s full-request clock')
    seconds=ledger['seconds']
    require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>=0,
        'finite complete request cost')
    require(terminal['started_at']==request['started_at'] and terminal['hard_deadline']==request['hard_deadline']
        and terminal['received_at']-request['started_at']<=seconds,'reception included in cost')
    if ledger.get('terminal_overrun'):
        require(ledger['status']=='TIMEOUT' and ledger['accepted'] is False and seconds>=300,
            'late terminal cannot be accepted offline')
    else:
        require(ledger['status']==terminal['status'] and ledger['accepted']==terminal['accepted'],
            'ledger/terminal agreement')
    row=dict(status=ledger['status'],seconds=seconds,clean_correct=terminal['clean_correct'],
        accepted=ledger['accepted'],cost_missing=False,route_candidate_upper=None,
        coverage=terminal['coverage'])
    if not ledger['accepted']:
        require(row['status'] in ('TIMEOUT','ERROR'),'no positive from missing online receipt')
        return row,None
    require(seconds<300 and terminal['received_at']<request['hard_deadline'],'online full budget')
    session=read_record(folder/'session.json.gz')
    require(session['status']=='COMPLETED' and session['returncode']==0 and session['execution_error'] is None
        and session['cleanup_error'] is None and session['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION',
        'accepted session identity/exit/cleanup')
    publication=read_record(folder/'published.json.gz')
    require(publication['sha256']==sha(folder/'outcome.json.gz'),'outcome publication')
    outcome=read_record(folder/'outcome.json.gz')
    require(outcome['request_sha256']==sha(folder/'request.json.gz')
        and outcome['clock']<=publication['clock']<request['worker_deadline'],'on-time candidate identity/publication')
    value=outcome['value'];result=value['result']
    require(result['status']==row['status'] and result['evaluation']['mode']==query['mode'],
        'accepted result semantics')
    if query['mode']=='pgd':
        audit_pgd(value,folder,request)
        return row,value
    cov=audit_value(result,family=value['family'],experts=value['experts'],properties=value['properties'])
    require(cov==terminal['coverage']==value['coverage'],'independent coverage recount')
    require(result['formal_SAFE'] is False and result['evaluation']['source_complete_proof'] is False,
        'numerical result is not an independent strict proof')
    if result['status']=='UNSAFE_REPLAYED':
        require(value['replayed'] is True and 'witness' in result,'recorded full-model replay')
    if value['family']=='moe':
        row['route_candidate_upper']=cov['expected']-cov['pruned']
    row['clean_correct']=value['clean_correct']
    return row,value


def analyze(root,manifest):
    began=time.monotonic();spec=load(manifest);manifest_hash=sha(manifest)
    requests=all_requests(spec['models'])
    segments=[(root,manifest_hash,spec)]
    if 'continuation' in spec:
        from architecture_s1_continuation import validate,locate_requests
        old,full,remaining=validate(spec)
        require(requests==full,'same complete study denominator')
        c=spec['continuation']
        segments.insert(0,(Path(c['root']),c['manifest_sha256'],old))
        account=load(root/'artifact_account.json')
        require(account['cap']==spec['artifact_bytes'] and
            account['initial_bytes']==c['prior_charged_bytes']+Path(manifest).stat().st_size,
            'aggregate capacity carries old usage, not a fresh allowance')
    else:
        require(digest(requests)==spec['request_catalog_sha256'],'frozen complete roster')
        from architecture_s1_continuation import locate_requests
    owners=locate_requests(requests,[s[0] for s in segments])
    for folder,h,_ in segments:
        start=read_record(folder/'start.json.gz')
        require(start['manifest_sha256']==h,'execution freeze')
    model_map={m['id']:m for m in spec['models']}
    catalog=load(spec['inputs']['catalog'],spec['inputs']['catalog_sha256'])
    require(sha(spec['inputs']['path'])==spec['inputs']['sha256'],'shared input source')
    records={};attacks={};all_states=Counter();total_seconds=0.;missing_cost=0
    segment_seconds=[0.]*len(segments);segment_counts=[0]*len(segments)
    for query in requests:
        owner=owners.get(query['id'],len(segments)-1)
        folder,h,_=segments[owner]
        row,value=audit_request(folder,query,h)
        all_states[row['status']]+=1
        total_seconds+=row['seconds'] or 0.
        segment_seconds[owner]+=row['seconds'] or 0.
        segment_counts[owner]+=row['status']!='NOT_EXECUTED'
        missing_cost+=row['seconds'] is None and row['status']!='NOT_EXECUTED'
        if value is not None:
            require(value['backend_sources_sha256']==spec['backend_sources_sha256']
                and value['family']==model_map[query['model']]['family'],'same model family/engine')
        if query['mode']=='pgd':
            domain_map={} if value is None else {d['index']:d for d in value['domains']}
            for index in query['indices']:
                key=(query['model'],query['radius'],index)
                if index not in domain_map:
                    attacks[key]=dict(status='NOT_EXECUTED' if row['status']=='NOT_EXECUTED' else 'BATCH_UNACCEPTED',
                        clean_correct=None)
                    continue
                record=read_record(folder/query['id']/domain_map[index]['file'])
                require(record['label']==catalog['samples'][index]['label'],'public attack true label')
                attacks[key]=dict(status=record['result']['status'],clean_correct=record['clean_prediction']==record['label'])
        else:
            key=(query['mode'],query['model'],query['radius'],query['index'])
            row['index']=query['index'];records[key]=row
            if value is not None:
                require(value['label']==catalog['samples'][query['index']]['label'],'verifier true label')
    # A later auxiliary result never changes an online verifier terminal. It
    # supplies explanatory clean-correct strata and exposes disagreements.
    for (mode,model,radius,index),row in records.items():
        attack=attacks[(model,radius,index)]
        if row['clean_correct'] is not None and attack['clean_correct'] is not None:
            require(row['clean_correct']==attack['clean_correct'],'same-object clean prediction')
        if row['clean_correct'] is None:
            row['clean_correct']=attack['clean_correct']
        require(not (row['status']=='POSITIVE' and attack['status']=='UNSAFE_REPLAYED'),
            'numerical positive conflicts with accepted same-object full-model attack')
    tables=[]
    for mode,count in (('complete',200),('incomplete',1000)):
        for radius in (2,8):
            for model in spec['models']:
                rows=[records[(mode,model['id'],radius,i)] for i in range(count)]
                summary=model_summary(rows,count)
                relevant=[attacks[(model['id'],radius,i)] for i in range(count)]
                summary['public_attack_states']=dict(Counter(a['status'] for a in relevant))
                tables.append(dict(mode=mode,radius=radius,model=model['id'],**summary))
    comparisons=[]
    for mode,count in (('complete',200),('incomplete',1000)):
        for radius in (2,8):
            moe=[[records[(mode,f'L1E8_seed{s}',radius,i)] for i in range(count)] for s in range(3)]
            for group in ('total','active'):
                dense=[[records[(mode,f'Dense_{group}_seed{s}',radius,i)] for i in range(count)] for s in range(3)]
                comparisons.append(dict(mode=mode,radius=radius,match=group,
                    **paired(moe,dense,count=count,resamples=spec['bootstrap']['resamples'],seed=spec['bootstrap']['seed'])))
    primaries={r['match']:r for r in comparisons if r['mode']=='complete' and r['radius']==2}
    settled=all_states['NOT_EXECUTED']==0
    queues=[];physical=0
    for i,(folder,h,part_spec) in enumerate(segments):
        queue_path=folder/'queue_complete.json.gz'
        queue=read_record(queue_path) if queue_path.exists() else None
        require(not settled or queue is not None,'completed queue final receipt')
        if queue is not None:
            require(queue['source_manifest_sha256']==h and
                queue['request_catalog_sha256']==part_spec['request_catalog_sha256'],'final queue identity')
            require(queue['settled']==segment_counts[i],'full terminal denominator')
            require(abs(queue['summed_request_seconds']-segment_seconds[i])<1e-5,'total charged request time')
        queues.append(queue)
        account=load(folder/'artifact_account.json')
        part_physical=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        require(part_physical+account['initial_bytes']<=account['used']<=account['cap'],'shared artifact accounting')
        physical+=part_physical
    return dict(status='COMPLETE_AUDIT' if settled else 'PREFIX_AUDIT',source_manifest_sha256=manifest_hash,
        online_states=dict(all_states),planned_native_requests=len(requests),charged_request_seconds=total_seconds,
        missing_executed_cost=missing_cost,
        queue_wall_seconds=None if any(q is None for q in queues) else sum(q['queue_wall_seconds'] for q in queues),
        segment_roots=[str(s[0]) for s in segments],segment_charged_seconds=segment_seconds,
        artifact_physical_bytes=physical,artifact_charged_bytes=account['used'],
        model_tables=tables,paired_comparisons=comparisons,
        s1_decision=s1_decision(primaries['total'],primaries['active'],all_queue_items_settled=settled),
        scientific_scope='numerical-policy architecture comparison; structural audit is not independent bound/model reproof',
        zero_certificate_primary=all(p['both_certified_rates_zero'] for p in primaries.values()),
        audit_seconds=time.monotonic()-began,auditor_sha256=sha(__file__))


def public_summary(root,manifest):
    """Summarize the completed public attack queue while verifiers continue.

    Only immutable settled PGD records are read; no live verifier account/queue
    snapshot is required. Unaccepted batch prefixes remain missing, not repaired.
    """
    from architecture_s1_continuation import validate,locate_requests
    began=time.monotonic();spec=load(manifest)
    segments=[(root,sha(manifest))]
    if 'continuation' in spec:
        _,full,_=validate(spec);c=spec['continuation']
        segments.insert(0,(Path(c['root']),c['manifest_sha256']))
    else:
        full=all_requests(spec['models'])
        require(digest(full)==spec['request_catalog_sha256'],'full public source roster')
    owners=locate_requests(full,[s[0] for s in segments])
    queries=[q for q in full if q['mode']=='pgd']
    require(all(q['id'] in owners for q in queries),'public queue not yet settled')
    catalog=load(spec['inputs']['catalog'],spec['inputs']['catalog_sha256'])
    require(sha(spec['inputs']['path'])==spec['inputs']['sha256'],'public physical inputs')
    models={m['id']:m for m in spec['models']}
    tables={(m,r):dict(model=m,radius=r,denominator=1000,clean_correct=0,
        clean_unknown=0,states=Counter()) for m in models for r in (2,8)}
    states=Counter();seconds=0.;bindings=[]
    for q in queries:
        folder,h=segments[owners[q['id']]]
        row,value=audit_request(folder,q,h)
        require(row['seconds'] is not None,'public request cost present')
        states[row['status']]+=1;seconds+=row['seconds']
        ledger=folder/(q['id']+'.ledger.json.gz');bindings.append((str(ledger),sha(ledger)))
        table=tables[(q['model'],q['radius'])]
        if value is None:
            table['states']['BATCH_UNACCEPTED']+=len(q['indices'])
            table['clean_unknown']+=len(q['indices'])
            continue
        require(value['family']==models[q['model']]['family'] and
            value['backend_sources_sha256']==spec['backend_sources_sha256'],'public model/engine binding')
        for d in value['domains']:
            record=read_record(folder/q['id']/d['file'])
            require(record['label']==catalog['samples'][d['index']]['label'],'public true label')
            table['clean_correct']+=record['clean_prediction']==record['label']
            table['states'][record['result']['status']]+=1
    groups=[]
    for radius in (2,8):
        for group in ('L1','L2','total','active'):
            items=[tables[(m,radius)] for m in models if models[m]['group']==group]
            require(len(items)==3,'three fixed seeds per architecture group')
            counts=sum((t['states'] for t in items),Counter())
            groups.append(dict(group=group,radius=radius,denominator=3000,states=dict(counts),
                clean_correct=sum(t['clean_correct'] for t in items),
                clean_unknown=sum(t['clean_unknown'] for t in items),
                not_disproved_count=3000-counts['UNSAFE_REPLAYED']))
    require(len(queries)==2400 and all(sum(t['states'].values())==1000 for t in tables.values()),
        '2400 batches, all24000 domains, missing retained')
    return dict(status='PUBLIC_PGD_RECORD_AUDIT',source_manifest_sha256=sha(manifest),
        online_states=dict(states),planned_native_requests=2400,planned_domains=24000,
        charged_request_seconds=seconds,record_identity_digest=digest(bindings),
        model_tables=list(tables.values()),groups=groups,s1_decision='PENDING_VERIFICATION',
        scientific_scope='fixed attack results only; no-witness and unaccepted are not certificates; record audit does not independently replay witnesses',
        audit_seconds=time.monotonic()-began,auditor_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--public-only',action='store_true')
    args=parser.parse_args()
    result=(public_summary if args.public_only else analyze)(args.root,args.manifest)
    if args.output.resolve().parent==args.root.resolve():
        Artifacts(args.root).write(args.output,result,terminal=True)
    else:
        save(args.output,result)
    print(result['status'],result['s1_decision'],result['online_states'])
