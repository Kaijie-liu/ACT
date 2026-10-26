"""B4 catalog and evidence reception using the existing request worker.

No mechanism is admitted here from a running prefix. No model/solver import,
new supervisor, or permission to start an unfrozen real queue is introduced.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from act.pipeline.moe import moe_aware_study as S
from act.pipeline.moe.multilayer_study import coverage
from scoped_proof.io import load,sha,tick
import multilayer_stage1 as Q


def source_plan():
    """Saved18-model identities and full named exclusion union; no forward."""
    from b1_candidate_history import audit as history_audit,ranges
    history=history_audit();excluded={i for a,b in history['excluded_inclusive_ranges'] for i in range(a,b+1)}
    models=[];bindings=[];cohorts=[];data=None
    for manifest,root in (
        ('multilayer_stage1_freeze_20261022_r2.json','multilayer_stage1_20261022_r1'),
        ('multilayer_stage1b_pruned_freeze_20261025_r1.json','multilayer_stage1b_pruned_20261025_r1'),
    ):
        path=ROOT/'docs'/manifest;spec=load(path)
        key=(spec['data_root'],spec['test_batch_sha256'])
        S.require(data is None or data==key,'one physical CIFAR test source');data=key
        bindings.append(dict(path=str(path),sha256=sha(path)))
        models.extend(spec['models'])
        excluded.update(i for a,b in spec['excluded_ranges'] for i in range(a,b+1))
        # ROOT may be an immutable executions/<run> checkout at execution time;
        # the original source inventory remains in the explicitly named root.
        path=Path('/data1/Kane/MOE/results')/root/'selection/outcome.json.gz'
        selection=Q.read_gzip(path)['value'];S.require(selection['status']=='COMPLETE','completed historical selection')
        ids=sorted({r['index'] for r in selection['samples']+selection['scanned']})
        excluded.update(ids);cohorts.append(dict(path=str(path),sha256=sha(path),indices=ids,
            scope='all named selected and explicitly scanned identities; not whole-test aggregate telemetry'))
    S.require(len(models)==18 and {m['id'] for m in models}==set(S.MULTILAYER+S.SINGLELAYER),'fixed18 checkpoints')
    S.require({98,4088,4096,4098,4099}<=excluded,'sealed identities preserved')
    for item in models:
        S.require(sha(item['checkpoint'])==item['checkpoint_sha256'],'saved trained weight identity')
    return dict(models=models,data_root=data[0],test_batch_sha256=data[1],excluded_ranges=ranges(excluded),
        history_union_sha256=hashlib.sha256(json.dumps(sorted(excluded),separators=(',',':')).encode()).hexdigest(),
        source_manifests=bindings,historical_cohorts=cohorts,history_sources=history['sources'],
        excluded_count=len(excluded),full_test_telemetry_disclosed=True,
        selection=dict(count=20,verification_count=20,additional_domains={'4/255':4/255},
            criterion='common clean-correct across18 fixed models only; no route/property-bound selection',
            order='ascending outside full named exclusion union',
            preprocessing='ToTensor float32 THEN float64; no normalization'),real_model_calls=0)


def admitted(summary):
    """Recompute the frozen local admission; caller binds the completed audit."""
    S.require(summary['status']=='AUDITED_DIAGNOSTIC_TERMINALS'
        and summary['primary_decision']=='COMPLETE_FIXED_MECHANISM_GATE'
        and summary['completed_calls']==summary['planned_calls']==103
        and not summary['unexecuted'] and summary['started_without_terminal']==0,
        'complete frozen5a, never prefix admission')
    choices=[]
    for key,name in (('mb','M-B'),('mc2','M-C-2'),('mc4','M-C-4')):
        row=summary[key];median=row['paired_median_bound_improvement']
        passed=len(row['gained'])>=5 and median is not None and median>0
        S.require(row['passes'] is passed,'unchanged registered local gate')
        if passed:choices.append(name)
    expected=([m for m in choices if m=='M-B']+
        (['M-C-2'] if 'M-C-2' in choices else ['M-C-4'] if 'M-C-4' in choices else []))
    S.require(summary['admitted_mechanisms']==expected,'fixed2then4 global choice')
    pieces=2 if 'M-C-2' in expected else 4 if 'M-C-4' in expected else None
    return dict(components=['M-A']+(['M-B'] if 'M-B' in expected else [])+(['M-C'] if pieces else []),
                margin_pieces=pieces)


def catalog(spec,samples,selection_root):
    """Same input tensors for every arm; four nontransferable study parts."""
    models={m['id']:m for m in spec['models']}
    S.require(len(models)==len(spec['models'])==18
        and set(models)==set(S.MULTILAYER+S.SINGLELAYER),'all fixed18 models')
    indices=[s['index'] for s in samples]
    S.require(len(indices)==20 and indices==sorted(set(indices)),'fixed20 common input blocks')
    excluded={i for a,b in spec['excluded_ranges'] for i in range(a,b+1)}
    S.require(not excluded.intersection(indices),'new inputs outside bound historical union')
    by={s['index']:s for s in samples};chosen=admitted(spec['stage5a_summary'])
    rows=[]
    for entry in S.catalog(indices,chosen['components']):
        base=by[entry['index']]
        if entry['part']=='radius':
            sample=base['additional_domains']['4/255']
            S.require(all(sample[k]==base[k] for k in ('index','label','rank'))
                and all(sample['tensors'][k]==base['tensors'][k] for k in ('center','rows','thresholds')),
                'radius changes the box only, not input or property')
        else:sample={k:v for k,v in base.items() if k!='additional_domains'}
        options=S.execution_options(entry['components'],margin_pieces=chosen['margin_pieces'])
        query=dict(kind='verification',model=models[entry['model']],sample=sample,
            selection_root=str(selection_root),arm='genbab_pruned',**options)
        rows.append(dict(**entry,query=query))
    return rows


def auxiliary(spec,samples,selection_root,part):
    """Public attack/census costs belong to their cohort, never either arm.

    Ablations share the already charged MAIN public records, not verifier
    answers; their full-arm comparator is also the registered MAIN on run.
    """
    models={m['id']:m for m in spec['models']}
    names={'main':S.MULTILAYER,'ablation':(),'radius':S.SUBSET,'boundary':S.SINGLELAYER}[part]
    chosen=([s['additional_domains']['4/255'] for s in samples] if part=='radius' else
        [{k:v for k,v in s.items() if k!='additional_domains'} for s in samples])
    eps=4/255 if part=='radius' else 2/255
    return [dict(name=f'{part}_{kind}_{name}',part=part,seconds=spec['budgets'][kind+'_model'],
        query=dict(kind=kind,model=models[name],samples=chosen,selection_root=str(selection_root),epsilon=eps))
        for kind in ('pgd','census') for name in names]


def audit_call(folder,entry,ledger,manifest,backend_identity,deadline):
    """Structural/source/coverage checks, never independent CROWN reproof."""
    from b1_observed_batch import session_ok
    tick(deadline);terminal=load(folder/'terminal.json',ledger['terminal_sha256'])
    S.require(all(ledger[k]==v for k,v in terminal.items()),'unaltered terminal including failures')
    S.require(0<=terminal['through_reception_seconds']<=ledger['seconds']<=300
        and terminal['hard_budget']==300,'complete request cost and unchanged cap')
    for name,h in terminal['files'].items():
        tick(deadline)
        S.require(Path(name).name==name and sha(folder/name)==h,'owned bound request artifacts')
    request=load(folder/'request.json')
    S.require(request['manifest_sha256']==sha(manifest)
        and all(request.get(k)==v for k,v in entry['query'].items()),'same model/input/domain/property/arm')
    # No unregistered optional flag may hide in an otherwise matching request.
    for name in ('gate_cascade','guard_dual','margin_branch_pieces'):
        S.require(request.get(name)==entry['query'].get(name),'exact optional mechanism settings')
    result=None
    if ledger['accepted']:
        S.require(session_ok(load(folder/'session.json')),'owned completed native session')
        publication=load(folder/'published.json');raw=Q.read_gzip(folder/'outcome.json.gz')
        S.require(publication['sha256']==sha(folder/'outcome.json.gz')
            and raw['request_sha256']==sha(folder/'request.json')
            and raw['clock']<request['worker_deadline'] and publication['clock']<request['worker_deadline'],
            'original on-time outcome; offline audit cannot upgrade timeout')
        value=raw['value'];result=value['result'];cfg=request['model']['architecture']
        environment=load(folder/'environment.json')
        S.require(environment['status']=='PASS' and environment['sources_canonical_sha256']==backend_identity,
            'same actual imported author engine')
        S.require(value['status']==ledger['status']==result['status'],'same accepted terminal')
        Q.external_audit(result,[cfg['experts']]*cfg['layers'],cfg['classes']-1)
        S.require(value['coverage']==coverage(result,'genbab_pruned',result['expected_histories']),
            'complete same-history accounting')
        if request.get('guard_dual'):
            from moe_aware_frontend import audit_result
            audit_result(result,request,backend_identity,deadline)
        else:
            S.require(all('guard_support' not in r for r in result['prefix_bounds'])
                and all('guard_support' not in r['outcome'] for r in result['histories']),
                'off arm cannot inherit M-B facts')
        if request.get('gate_cascade') or request.get('margin_branch_pieces'):
            from gate_cascade import audit_files
            audit_files(folder,result,request,deadline)
        else:
            S.require(not result.get('cascade_files')
                and all('cascade' not in r['outcome'] for r in result['histories']),'off arm cannot inherit cascade')
        if value['status']=='UNSAFE_REPLAYED':
            S.require(value['replayed'] and result.get('witness') is not None,'saved complete-model replay')
    else:
        S.require(ledger['status'] not in ('POSITIVE','UNSAFE_REPLAYED'),'no partial positive/witness')
    tick(deadline)
    return dict(part=entry['part'],model=entry['model'],index=entry['index'],rank=entry['rank'],
        arm=entry['arm'],status=ledger['status'],accepted=ledger['accepted'],seconds=ledger['seconds'],
        coverage=None if result is None else value['coverage'],
        guarantee=None if result is None else value['guarantee'])


def contrasts(rows,indices,components):
    """Keep main/radius/boundary separate; full ablation references are MAIN."""
    result={part:S.contrast([r for r in rows if r['part']==part],models,indices)
        for part,models in (('main',S.MULTILAYER),('radius',S.SUBSET),('boundary',S.SINGLELAYER))}
    result['ablation']={}
    for component in components:
        off='minus_'+component
        paired=[r for r in rows if r['model'] in S.SUBSET and
            ((r['part']=='main' and r['arm']=='on') or (r['part']=='ablation' and r['arm']==off))]
        result['ablation'][component]=S.contrast(paired,S.SUBSET,indices,off=off,on='on')
    return result


def execute_part(folder,manifest,entries,*,part,deadline,invoke=Q.invoke,clock=time.monotonic):
    """Use the existing300s worker, keeping a contiguous stopped prefix.

    The enclosing stage sets the already charged part deadline and shared
    artifact accounting; unused time never transfers between parts.
    """
    began=clock();S.require(part in S.CAPS and deadline-began<=S.CAPS[part][1],'own remaining part time')
    S.require(len(entries)<=S.CAPS[part][0] and all(e['part']==part for e in entries),'own part calls')
    ledgers=[];reason='ALL_REGISTERED_CALLS_EXECUTED'
    for i,entry in enumerate(entries):
        if clock()+300>deadline:
            reason='PART_TIME_CAP';break
        name=f'{part}_verify_{i:04d}'
        Q.save(folder/(name+'_started.json'),dict(index=i,part=part,entry=entry,clock=clock()))
        try:
            terminal,_=invoke(folder/name,entry['query'],300,manifest,deadline)
        except BaseException as error:
            Q.save(folder/(name+'_failed_start.json'),dict(kind=type(error).__name__,message=str(error),clock=clock()))
            raise
        row=dict(**terminal,id=name,part=part,model=entry['model'],index=entry['index'],rank=entry['rank'],arm=entry['arm'])
        Q.save(folder/(name+'_ledger.json'),row);ledgers.append(row)
        if terminal['status'] in ('ERROR','UNSUPPORTED'):
            reason='TOOL_OR_RECEPTION_ERROR';break
    return dict(part=part,rows=ledgers,unexecuted=entries[len(ledgers):],stop_reason=reason,
                seconds=clock()-began)
