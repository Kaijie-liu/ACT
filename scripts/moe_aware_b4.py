"""B4 paired study using the existing request worker and owned supervisor.

No mechanism is admitted here from a running prefix. No model/solver import,
new supervisor, or permission to start an unfrozen real queue is introduced.
"""
import time
ENTRY=time.monotonic()
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from act.pipeline.moe import moe_aware_study as S
from act.pipeline.moe.multilayer_study import coverage
from scoped_proof.io import load,sha,tick
import multilayer_stage1 as Q
from scoped_source.development_work import POLICY

SCHEMA='moe-aware-b4-20261026-v1'
RESUME_TIMING=(12,10,5)  # worker, supervisor, cleanup tails inside the same300s


def result_chain(folder,manifest,seen=None):
    """Append-only execution roots; a repair never overwrites an old attempt."""
    folder=Path(folder).resolve();seen=set() if seen is None else set(seen)
    S.require(folder not in seen,'acyclic execution chain');seen.add(folder)
    path=folder/'continuation.json'
    if not path.exists():return [folder]
    binding=load(path);parent=Path(binding['parent_root'])
    S.require(binding['manifest_sha256']==sha(manifest),'same experiment after repair')
    start=load(folder/'start.json');pointer=start['execution_binding']
    S.require(pointer['sha256']==sha(path) and load(pointer['path'],pointer['sha256'])==binding,
        'same execution and continuation identity')
    S.require(all(sha(Path(start['execution_root'])/p)==h for p,h in binding['source_files'].items()),
        'retained repaired execution source')
    for row in binding['external_records']:
        S.require(sha(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes'],
            'retained repair evidence and console')
    for name,h in binding['parent_files'].items():
        S.require(not Path(name).is_absolute() and '..' not in Path(name).parts
            and sha(parent/name)==h,'unchanged failed prefix')
    S.require(set(binding['parent_files'])=={str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()},
        'no unbound additions to failed prefix')
    return result_chain(parent,manifest,seen)+[folder]


def owned_file(roots,name):
    found=[r/name for r in roots if (r/name).exists()]
    S.require(len(found)<=1,'no repeated call or overwritten evidence')
    return found[0] if found else None


def failed_attempt(owner,name,entry,manifest):
    """Account for a stopped tooling attempt, never manufacture online acceptance."""
    event=load(owner/(name+'_started.json'));failed=load(owner/(name+'_failed_start.json'))
    stopped=load(owner/'stopped.json');start=load(owner/'start.json')
    terminal=load(owner/name/'terminal.json');request=load(owner/name/'request.json')
    S.require(event['entry']==entry and request['manifest_sha256']==sha(manifest)
        and all(request.get(k)==v for k,v in entry['query'].items()),'failed object binding')
    S.require(not terminal['accepted'] and terminal['status'] not in ('POSITIVE','UNSAFE_REPLAYED'),
        'failed attempt cannot supply a positive')
    for p,h in terminal['files'].items():
        S.require(Path(p).name==p and sha(owner/name/p)==h,'failed original terminal files')
    finish=start['entry']+stopped['seconds']
    S.require(event['clock']<=failed['clock']<=finish,'measured stopped attempt interval')
    return dict(part=entry['part'],model=entry['model'],index=entry['index'],rank=entry['rank'],
        arm=entry['arm'],status='ERROR',accepted=False,seconds=finish-event['clock'],coverage=None,guarantee=None,
        failure_kind=failed['kind'],failure_message=failed['message'],observed_terminal_status=terminal['status'],
        accounting='OFFLINE_TOOL_FAILURE_UPPER_COST_NOT_ONLINE_ACCEPTANCE',
        original_terminal_sha256=sha(owner/name/'terminal.json'),failed_start_sha256=sha(owner/(name+'_failed_start.json')))


def sources():
    from gate_decoupling_control import CODE
    return tuple(dict.fromkeys((*Q.CODE,*CODE,'scripts/moe_aware_b4.py','scripts/gate_cascade.py',
        'scripts/margin_cascade.py','scripts/moe_aware_frontend.py','scripts/moe_aware_diagnostics.py',
        'act/back_end/moe/moe_aware_bab.py','act/pipeline/moe/moe_aware_study.py',
        'scoped_proof/owned_bounded.py','tests/test_moe_aware_b4.py')))


def freeze(path,diagnostic_root,diagnostic_manifest,control):
    """Called only after final5a audit; freezes before any B4 source loading."""
    from gate_decoupling_worker import digest
    summary=load(diagnostic_root/'audit.json');choice=admitted(summary)
    start=load(diagnostic_root/'start.json');final=load(diagnostic_root/'final_accounting.json')
    S.require(start['manifest_sha256']==sha(diagnostic_manifest)
        and final['audit_sha256']==sha(diagnostic_root/'audit.json'),'completed5a provenance')
    tested=load(control);S.require(tested['status']=='PASS' and tested['trained_calls']==0,'ordinary controls')
    S.require(all(sha(ROOT/p)==h for p,h in tested['source_files'].items()),'same controlled execution code')
    original=load(ROOT/'docs/multilayer_stage1_freeze_20261022_r2.json')
    plan=source_plan()
    Q.save(path,dict(schema=SCHEMA,policy=POLICY,**plan,stage5a_summary=summary,admission=choice,
        stage5a=dict(root=str(diagnostic_root.resolve()),manifest=str(diagnostic_manifest.resolve()),
            manifest_sha256=sha(diagnostic_manifest),start_sha256=sha(diagnostic_root/'start.json'),
            audit_sha256=sha(diagnostic_root/'audit.json'),final_sha256=sha(diagnostic_root/'final_accounting.json')),
        control=dict(path=str(control.resolve()),sha256=sha(control)),
        source_files={p:sha(ROOT/p) for p in sources()},genbab_config=Q.backend.CONFIG,
        backend_identity=digest(Q.backend.source_identity()),epsilon=2/255,
        part_caps={p:dict(slots=n,seconds=t) for p,(n,t) in S.CAPS.items()},
        budgets=dict(request=300,stage=110*3600,slots=1320,selection=300,
            pgd_model=100,census_model=300,artifacts=Q.CAP),
        public_pgd=original['public_pgd'],census=original['census'],
        census_multipliers=original['census_multipliers'],
        decision=dict(seed=20261026,bootstrap=10000,net_strict_min=0,rate_min=.05,cost_ratio_max=2),
        execution=dict(cpu_only=True,threads=1,as_bytes=8*2**30,shared_artifact_cap=Q.CAP,
            parts=list(S.CAPS),no_time_or_slot_transfer=True,cold_worker_per_request=True),
        guarantee='NUMERICAL_FILTER_NOT_SOURCE_COMPLETE_SAFE'))


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


def execute_part(folder,manifest,entries,*,part,deadline,invoke=Q.invoke,clock=time.monotonic,start_index=0):
    """Use the existing300s worker, keeping a contiguous stopped prefix.

    The enclosing stage sets the already charged part deadline and shared
    artifact accounting; unused time never transfers between parts.
    """
    began=clock();S.require(part in S.CAPS and deadline-began<=S.CAPS[part][1],'own remaining part time')
    S.require(len(entries)<=S.CAPS[part][0] and all(e['part']==part for e in entries),'own part calls')
    ledgers=[];reason='ALL_REGISTERED_CALLS_EXECUTED'
    S.require(0<=start_index<=len(entries),'registered continuation offset')
    for i in range(start_index,len(entries)):
        entry=entries[i]
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
    return dict(part=part,rows=ledgers,unexecuted=entries[start_index+len(ledgers):],stop_reason=reason,
                seconds=clock()-began)


def audit_aux(folder,expected,ledger,manifest,deadline):
    """Retain individually received public domains even if their batch timed out."""
    tick(deadline);query=expected['query'];request=load(folder/'request.json')
    terminal=load(folder/'terminal.json',ledger['terminal_sha256'])
    S.require(all(ledger[k]==v for k,v in terminal.items()) and
        0<=terminal['through_reception_seconds']<=ledger['seconds']<=expected['seconds']<=300
        and terminal['hard_budget']==expected['seconds'],
        'public terminal and complete cost')
    S.require(request['manifest_sha256']==sha(manifest) and all(request.get(k)==v for k,v in query.items()),
        'public model/input/radius identity')
    for name,h in terminal['files'].items():
        tick(deadline);S.require(Path(name).name==name and sha(folder/name)==h,'public source-bound files')
    rows=[];seen=set();indices={s['index'] for s in query.get('samples',[])}
    multipliers=load(manifest)['census_multipliers'] if query['kind']=='census' else [1.]
    for name in sorted(n for n in terminal['files'] if n.startswith('domain_') and n.endswith('.json')):
        path=folder/name
        tick(deadline);row=load(path);key=(row['index'],row['multiplier'])
        S.require(key not in seen and key[0] in indices and key[1] in multipliers
            and row['request_sha256']==sha(folder/'request.json')
            and row['accepted']==(row['clock']<row['deadline']) and row['deadline']<request['worker_deadline'],
            'one on-time public domain, never accepted by later audit')
        seen.add(key)
        if row['accepted']:
            result=row['result']
            if query['kind']=='pgd' and result['status']=='UNSAFE_REPLAYED':
                S.require(result.get('witness') is not None and result['guarantee']=='FULL_MODEL_REPLAY','public replay')
            rows.append(dict(model=query['model']['id'],index=row['index'],multiplier=row['multiplier'],
                result=result,seconds=row['seconds']))
    return dict(kind=query['kind'],model=query.get('model',{}).get('id'),seconds=ledger['seconds'],
                status=ledger['status'],domains=rows,
                unreceived_domains=[[i,m] for i in sorted(indices) for m in multipliers
                    if not any(r['index']==i and r['multiplier']==m for r in rows)])


def audit_part(folder,manifest,entries,samples,*,part,deadline,roots=None):
    roots=[folder] if roots is None else roots
    spec=load(manifest);selected=roots[0]/'selection';rows=[];aux=[];failures=[]
    S.require(all(e['part']==part for e in entries),'registered part catalog')
    starts=sorted((p for r in roots for p in r.glob(part+'_verify_*_started.json')),key=lambda p:p.name)
    S.require(len(starts)<=len(entries),'started prefix count')
    S.require(all((p.parent/(p.name.removesuffix('_ledger.json')+'_started.json')).exists()
        for r in roots for p in r.glob(part+'_verify_*_ledger.json')),'no orphan terminal ledger')
    for i,path in enumerate(starts):
        event=load(path);name=f'{part}_verify_{i:04d}'
        S.require(path.name==name+'_started.json' and event['index']==i and event['entry']==entries[i],
                  'fixed full-request attempt order')
        ledger=owned_file(roots,name+'_ledger.json')
        if ledger is not None:
            row=load(ledger);S.require(row['id']==name and ledger.parent==path.parent,'same attempt terminal')
            request=load(ledger.parent/name/'request.json')
            if (ledger.parent/'continuation.json').exists():
                b=load(ledger.parent/'continuation.json')
                S.require(request.get('execution_binding')==load(ledger.parent/'start.json')['execution_binding']
                    and request.get('timing_reserve')==b['timing_reserve'],'same repaired execution policy')
            rows.append(audit_call(ledger.parent/name,entries[i],row,manifest,spec['backend_identity'],deadline))
        elif (path.parent/(name+'_failed_start.json')).exists():
            rows.append(failed_attempt(path.parent,name,entries[i],manifest));failures.append(name)
        else:
            S.require(i==len(starts)-1,'only the final attempt may be in flight')
    auxiliary_missing=[];auxiliary_started=[]
    for item in auxiliary(spec,samples,selected,part):
        path=owned_file(roots,item['name']+'_ledger.json')
        begin=owned_file(roots,item['name']+'_started.json')
        if begin is not None:
            S.require(load(begin)['item']==item,'fixed public query')
        if path is not None:
            S.require(begin is not None and begin.parent==path.parent,'public attempt before terminal')
            aux.append(audit_aux(path.parent/item['name'],item,load(path),manifest,deadline))
        elif begin is not None:auxiliary_started.append(item['name'])
        else:auxiliary_missing.append(item['name'])
    result=dict(rows=rows,auxiliary=aux,completed=len(rows),planned=len(entries),
        started=len(starts),started_without_terminal=entries[len(rows):len(starts)],
        unexecuted=entries[len(starts):],auxiliary_unexecuted=auxiliary_missing,
        auxiliary_started_without_terminal=auxiliary_started,mathematical_bound_reproof=False)
    if failures:result['retained_failed_attempts']=failures
    return result


def explain_rows(rows,auxiliary_rows):
    """Observed >=2 is evidence; one sample history is not route stability."""
    census={};unsafe=set()
    for batch in auxiliary_rows:
        for row in batch['domains']:
            key=(row['model'],row['index'])
            if batch['kind']=='pgd' and row['result']['status']=='UNSAFE_REPLAYED':unsafe.add(key)
            elif batch['kind']=='census' and row['multiplier']==1:
                census[key]=row['result']
    by={}
    for row in rows:
        if row['arm'] in ('off','on'):by.setdefault((row['model'],row['index']),{})[row['arm']]=row
    strata={};conflicts=[]
    for key,pair in by.items():
        c=census.get(key);group=('OBSERVED_MULTIPLE' if c and c['observed_lower']>=2 else
            'ONE_OBSERVED_NOT_PROVEN_STABLE' if c and c['observed_lower']==1 else 'NO_USABLE_WITNESS_COUNT')
        tally=strata.setdefault(group,dict(pairs=0,complete_pairs=0,gained=0,lost=0,observed=[]))
        tally['pairs']+=1;tally['observed'].append(dict(model=key[0],index=key[1],
            observed_lower=None if c is None else c['observed_lower'],
            candidate_upper=None if c is None else c['candidate_upper'],exact_legal_count=None))
        if len(pair)==2:
            tally['complete_pairs']+=1
            a,b=pair['off']['status']=='POSITIVE',pair['on']['status']=='POSITIVE'
            tally['gained']+=int(b and not a);tally['lost']+=int(a and not b)
        if key in unsafe and any(r['status']=='POSITIVE' for r in pair.values()):conflicts.append(list(key))
    return dict(strata=strata,public_replayed_unsafe=[list(k) for k in sorted(unsafe)],
                public_positive_conflicts=conflicts,interpretation='descriptive sampled routing, not exact reachable count or causality')


def part_summary(part,rows,auxiliary_rows,indices,components,main_rows):
    if part=='ablation':
        result={}
        for component in components:
            off='minus_'+component
            paired=[r for r in main_rows if r['model'] in S.SUBSET and r['arm']=='on']
            paired += [r for r in rows if r['arm']==off]
            result[component]=S.contrast(paired,S.SUBSET,indices,off=off)
        return dict(contrasts=result,reference='same paid MAIN on runs',primary_decision=False)
    models={'main':S.MULTILAYER,'radius':S.SUBSET,'boundary':S.SINGLELAYER}[part]
    result=S.contrast(rows,models,indices);explanation=explain_rows(rows,auxiliary_rows)
    if explanation['public_positive_conflicts']:result['decision']='EVIDENCE_CONFLICT_REPORT'
    return dict(contrast=result,routing=explanation,primary_decision=part=='main')


def audit(folder,manifest,*,deadline=None,source_root=None):
    """Independent record pass; all old numerical verdicts remain unchanged."""
    from b1_observed_batch import session_ok
    deadline=time.monotonic()+300 if deadline is None else deadline
    roots=result_chain(folder,manifest);origin=roots[0]
    binding=load(folder/'continuation.json') if (folder/'continuation.json').exists() else None
    if binding:source_root=Path(binding['original_execution'])
    spec=load(manifest);check_frozen(spec,source_root=source_root);start=load(folder/'start.json')
    S.require(start['manifest_sha256']==sha(manifest),'same B4 experiment')
    selection=origin/'selection';ledger=load(origin/'selection_ledger.json')
    terminal=load(selection/'terminal.json',ledger['terminal_sha256'])
    S.require(ledger['accepted'] and ledger['status']=='COMPLETE'
        and all(ledger[k]==v for k,v in terminal.items()) and
        0<=terminal['through_reception_seconds']<=ledger['seconds']<=300,'complete original selection')
    request=load(selection/'request.json');raw=Q.read_gzip(selection/'outcome.json.gz')
    publication=load(selection/'published.json')
    S.require(session_ok(load(selection/'session.json')) and request['manifest_sha256']==sha(manifest)
        and raw['request_sha256']==sha(selection/'request.json')
        and publication['sha256']==sha(selection/'outcome.json.gz')
        and raw['clock']<request['worker_deadline'] and publication['clock']<request['worker_deadline'],
        'original selection acceptance and timing')
    for name,h in terminal['files'].items():S.require(Path(name).name==name and sha(selection/name)==h,'selection file binding')
    samples=raw['value']['samples'];table=catalog(spec,samples,selection)
    saved=load(origin/'catalog.json')
    S.require(saved['manifest_sha256']==sha(manifest) and saved['entries']==table
        and saved['selection_outcome_sha256']==sha(selection/'outcome.json.gz'),'exact paired call catalog')
    excluded={i for a,b in spec['excluded_ranges'] for i in range(a,b+1)}
    S.require(hashlib.sha256(json.dumps(sorted(excluded),separators=(',',':')).encode()).hexdigest()
        ==spec['history_union_sha256'],'frozen exclusion union')
    scanned=raw['value']['scanned'];chosen=[r['index'] for r in scanned if all(p==r['label'] for p in r['predictions'])]
    first=samples[-1]['index'];expected=[i for i in range(first+1) if i not in excluded]
    S.require([r['index'] for r in scanned]==expected and all(len(r['predictions'])==18 for r in scanned)
        and chosen==[s['index'] for s in samples],'no skipped/reselected common clean-correct input')
    for sample in samples:
        for record in (sample,*sample['additional_domains'].values()):
            tick(deadline);S.require(Path(record['file']).name==record['file']
                and sha(selection/record['file'])==record['sha256'],'unaltered materialized paired input')
    reports={};main=[];costs={}
    for part in S.CAPS:
        entries=[r for r in table if r['part']==part]
        checked=audit_part(folder,manifest,entries,samples,part=part,deadline=deadline,roots=roots)
        summary=part_summary(part,checked['rows'],checked['auxiliary'],[s['index'] for s in samples],
            spec['admission']['components'],main)
        if part=='main':main=checked['rows']
        reports[part]=dict(checked=checked,summary=summary)
        accounting=owned_file(roots,part+'_accounting.json')
        if accounting is not None:
            record=load(accounting);costs[part]=record
            owner=accounting.parent
            S.require(record['completed']==len(checked['rows']) and record['planned']==len(entries)
                and 0<=record['seconds']<=S.CAPS[part][1]
                and record['audit_sha256']==sha(owner/(part+'_audit.json'))
                and record['result_sha256']==sha(owner/(part+'_result.json')),
                'each part cost and original online audit')
            S.require(load(owner/(part+'_audit.json'))==checked
                and load(owner/(part+'_result.json'))==json.loads(json.dumps(summary)),
                'independent recount agrees with original result')
    full=folder/'final_accounting.json'
    if full.exists():
        final=load(full);complete=load(folder/'complete.json')
        S.require(set(costs)==set(S.CAPS) and complete['full_calls']==sum(r['checked']['completed'] for r in reports.values())
            and all(complete['parts'][p]['completed']==reports[p]['checked']['completed'] for p in S.CAPS),
            'full receipt counts include every part')
        S.require(final['complete_sha256']==sha(folder/'complete.json')
            and final['seconds']<=110*3600 and all(0<=final['charged_part_seconds'][p]<=S.CAPS[p][1] for p in S.CAPS),
            'full cost and four nontransferable caps')
        S.require(abs(sum(final['charged_part_seconds'].values())-final['seconds'])<1e-6,
            'all measured stage time allocated, no uncharged common cost')
    tick(deadline)
    return dict(status='PASS_COMPLETE' if full.exists() and all(r['checked']['completed']==r['checked']['planned'] for r in reports.values())
        else 'PASS_RETAINED_PREFIX',reports=reports,cost=costs,
        numerical_bound_reproof=False,new_verdicts=0,full_calls=sum(r['checked']['completed'] for r in reports.values()),
        bytes=sum(Q.size(r) for r in roots)+(0 if binding is None else
            binding['external_retained_bytes']+Path(start['execution_binding']['path']).stat().st_size))


def check_frozen(spec,*,source_root=None):
    from gate_decoupling_worker import digest
    S.require(spec['schema']==SCHEMA and spec['policy']==POLICY,'B4 registered numerical policy')
    S.require(spec['part_caps']=={p:dict(slots=n,seconds=t) for p,(n,t) in S.CAPS.items()}
        and spec['budgets']==dict(request=300,stage=110*3600,slots=1320,selection=300,
            pgd_model=100,census_model=300,artifacts=Q.CAP),'user quotas, no transfer')
    S.require(spec['admission']==admitted(spec['stage5a_summary']),'unchanged complete admission')
    binding=spec['stage5a'];root=Path(binding['root'])
    S.require(sha(binding['manifest'])==binding['manifest_sha256']
        and sha(root/'start.json')==binding['start_sha256']
        and sha(root/'audit.json')==binding['audit_sha256']
        and sha(root/'final_accounting.json')==binding['final_sha256']
        and load(root/'audit.json')==spec['stage5a_summary'],'frozen diagnostic result')
    source_root=ROOT if source_root is None else Path(source_root)
    S.require(set(spec['source_files'])==set(sources())
        and all(sha(source_root/p)==h for p,h in spec['source_files'].items()),'complete frozen numerical/acceptance source')
    S.require(spec['genbab_config']==Q.backend.CONFIG and digest(Q.backend.source_identity())==spec['backend_identity'],
        'same strong coupled engine')
    for row in spec['source_manifests']+spec['historical_cohorts']:
        S.require(sha(row['path'])==row['sha256'],'same named source/exclusion metadata')
    S.require(load(spec['control']['path'],spec['control']['sha256'])['status']=='PASS','ordinary control binding')


def freeze_resume(path,manifest,parent,original_execution,control):
    """One compatibility continuation: same catalog, no failed-call retry."""
    began=time.monotonic();parent=Path(parent).resolve();original_execution=Path(original_execution).resolve()
    spec=load(manifest);check_frozen(spec,source_root=original_execution)
    prefix=audit(parent,manifest,source_root=original_execution)
    tested=load(control);S.require(tested['status']=='PASS' and tested['trained_calls']==0,'repair controls')
    S.require(all(sha(ROOT/p)==h for p,h in tested['source_files'].items()),'controlled repair source')
    roots=result_chain(parent,manifest);stop=load(parent/'stopped.json')
    S.require(all(not p['checked']['started_without_terminal'] and
        not p['checked']['auxiliary_started_without_terminal'] for p in prefix['reports'].values()),
        'every original attempt retained before continuing')
    carried=dict(stop.get('charged_part_seconds',dict(main=stop['seconds'])))
    pending=[p for p in S.CAPS if prefix['reports'][p]['checked']['completed']<prefix['reports'][p]['checked']['planned']]
    S.require(bool(pending),'unfinished registered queue');active=pending[0]
    control_files={str(p.relative_to(control.parent)):sha(p) for p in control.parent.rglob('*') if p.is_file()}
    prior_binding=load(parent/'continuation.json') if (parent/'continuation.json').exists() else None
    external={r['path']:r for r in ([] if prior_binding is None else prior_binding['external_records'])}
    paths=[control.parent/p for p in control_files]
    if prior_binding:paths.append(Path(load(parent/'start.json')['execution_binding']['path']))
    prior_control=tested.get('retained_control');seen_controls={str(control.resolve())}
    while prior_control:
        receipt=Path(prior_control['path']).resolve()
        S.require(str(receipt) not in seen_controls,'acyclic retained control costs');seen_controls.add(str(receipt))
        previous=load(receipt,prior_control['sha256'])
        paths.extend(p for p in receipt.parent.rglob('*') if p.is_file())
        prior_control=previous.get('retained_control')
    paths += [parent.with_name(parent.name+suffix) for suffix in ('_controller.log','_launch.json','_preparation.json')]
    for p in paths:
        if p.is_file():external[str(p.resolve())]=dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
    external_bytes=sum(r['bytes'] for r in external.values())
    parent_files={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file()}
    duration=time.monotonic()-began
    # Control and this source/audit preparation consume the same unfinished part.
    carried[active]=carried.get(active,0)+tested['seconds']+duration
    S.require(all(0<=v<S.CAPS[p][1] for p,v in carried.items()),'remaining part time, no borrowing')
    prior_bytes=sum(Q.size(r) for r in roots)+external_bytes
    S.require(prior_bytes<Q.CAP,'remaining shared artifact budget')
    Q.atomic_save(path,dict(schema='moe-aware-b4-continuation-v1',parent_root=str(parent),
        parent_files=parent_files,manifest_sha256=sha(manifest),original_execution=str(original_execution),
        source_files={p:sha(ROOT/p) for p in sources()},backend_identity=spec['backend_identity'],
        timing_reserve=list(RESUME_TIMING),prior_bytes=prior_bytes,external_retained_bytes=external_bytes,
        external_records=list(external.values()),
        control=dict(path=str(control.resolve()),sha256=sha(control),files=control_files),
        carried_part_seconds=carried,preparation_seconds=duration,repair_downtime_not_execution=True,
        next_indices={p:r['checked']['started'] for p,r in prefix['reports'].items()},
        prior_settled_calls=prefix['full_calls'],failed_attempts_remain_in_denominator=True))


def batch(folder,manifest,*,entry=ENTRY,continuation=None):
    """Four study parts; all native execution remains Q.invoke/owned_session."""
    S.require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),'immutable execution copy')
    S.require(not folder.exists(),'new B4 result root')
    spec=load(manifest);binding=None;roots=[];carried={}
    if continuation is not None:
        binding=load(continuation)
        S.require(binding['manifest_sha256']==sha(manifest) and binding['timing_reserve']==list(RESUME_TIMING),
            'fixed compatibility continuation')
        check_frozen(spec,source_root=binding['original_execution'])
        S.require(all(sha(ROOT/p)==h for p,h in binding['source_files'].items()),'new immutable execution identity')
        S.require(load(binding['control']['path'],binding['control']['sha256'])['status']=='PASS','repair passed')
        parent=Path(binding['parent_root']);roots=result_chain(parent,manifest)
        S.require(set(binding['parent_files'])=={str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}
            and all(sha(parent/p)==h for p,h in binding['parent_files'].items()),'unchanged parent before execution')
        for row in binding['external_records']:
            S.require(sha(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes'],
                'same repair cost and evidence')
        carried=dict(binding['carried_part_seconds'])
    else:check_frozen(spec)
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30));os.nice(10)
    folder.mkdir();Q.ARTIFACT_ROOT=folder
    Q.PRIOR_BYTES=0 if binding is None else binding['prior_bytes']+continuation.stat().st_size
    Q.ARTIFACT_LIMIT=Q.CAP-Q.PRIOR_BYTES
    Q.EXECUTION_BINDING=continuation
    if binding:Q.save(folder/'continuation.json',binding)
    roots=roots+[folder]
    Q.save(folder/'start.json',dict(entry=entry,pid=os.getpid(),started_at=time.time(),
        commit=(ROOT/'.frozen_commit').read_text().strip(),manifest_sha256=sha(manifest),execution_root=str(ROOT),
        **({} if binding is None else dict(execution_binding=dict(path=str(continuation),sha256=sha(continuation)),
            carried_part_seconds=carried))))
    parts={};samples=None;table=None;main_rows=[]
    timing=(6,4,2) if binding is None else RESUME_TIMING
    def invoke(*args):return Q.invoke(*args,timing_reserve=timing)
    if binding:
        samples=Q.read_gzip(roots[0]/'selection/outcome.json.gz')['value']['samples']
        table=load(roots[0]/'catalog.json')['entries']
    def call(item,deadline):
        name=item['name'];Q.save(folder/(name+'_started.json'),dict(item=item,clock=time.monotonic()))
        row,value=invoke(folder/name,item['query'],item['seconds'],manifest,deadline)
        Q.save(folder/(name+'_ledger.json'),dict(id=name,part=item['part'],**row))
        if row['status'] in ('ERROR','UNSUPPORTED'):raise RuntimeError('retained tooling error '+name)
        return row,value
    try:
        for part,(_,seconds) in S.CAPS.items():
            previous=owned_file(roots[:-1],part+'_accounting.json')
            if previous is not None:
                parts[part]=load(previous)
                if part=='main':main_rows=load(previous.parent/'main_audit.json')['rows']
                continue
            started=entry if not parts else time.monotonic();deadline=started+seconds-carried.get(part,0)
            Q.save(folder/(part+'_start.json'),dict(entry=started,deadline=deadline))
            selection_seconds=0.
            if part=='main' and binding is None:
                selected,value=call(dict(name='selection',part=part,seconds=300,query=dict(kind='selection')),deadline)
                S.require(selected['accepted'] and value is not None and value['status']=='COMPLETE','complete registered input selection')
                selection_seconds=selected['seconds'];samples=value['samples'];table=catalog(spec,samples,folder/'selection')
                Q.save(folder/'catalog.json',dict(manifest_sha256=sha(manifest),entries=table,
                    selection_outcome_sha256=sha(folder/'selection/outcome.json.gz')))
            for item in auxiliary(spec,samples,roots[0]/'selection',part):
                if owned_file(roots[:-1],item['name']+'_started.json') is not None:continue
                if time.monotonic()+item['seconds']>deadline:break
                call(item,deadline)
            entries=[r for r in table if r['part']==part]
            offset=0 if binding is None else binding['next_indices'][part]
            executed=execute_part(folder,manifest,entries,part=part,deadline=deadline,invoke=invoke,start_index=offset)
            if executed['stop_reason']=='TOOL_OR_RECEPTION_ERROR':raise RuntimeError('retained verification tooling error '+part)
            checked=audit_part(folder,manifest,entries,samples,part=part,deadline=deadline,roots=roots)
            summary=part_summary(part,checked['rows'],checked['auxiliary'],[s['index'] for s in samples],
                spec['admission']['components'],main_rows)
            if part=='main':main_rows=checked['rows']
            Q.save(folder/(part+'_audit.json'),checked);Q.save(folder/(part+'_result.json'),summary)
            parts[part]=dict(completed=checked['completed'],planned=len(entries),started=checked['started'],
                stop_reason=executed['stop_reason'],seconds=carried.get(part,0)+time.monotonic()-started,
                request_seconds=sum(r['seconds'] for r in checked['rows']),
                auxiliary_seconds=(load(roots[0]/'selection_ledger.json')['seconds'] if part=='main' else 0)
                    +sum(r['seconds'] for r in checked['auxiliary']),
                audit_sha256=sha(folder/(part+'_audit.json')),result_sha256=sha(folder/(part+'_result.json')))
            Q.save(folder/(part+'_accounting.json'),parts[part]);tick(deadline)
        # Final common publication is charged to the last part, not a new pool.
        parts['boundary']['seconds']=carried.get('boundary',0)+time.monotonic()-started
        Q.save(folder/'complete.json',dict(parts=parts,seconds=sum(carried.values())+time.monotonic()-entry,
            full_calls=sum(p['completed'] for p in parts.values()),
            status='COMPLETE' if all(p['completed']==p['planned'] for p in parts.values()) else 'REPORT_UNEXECUTED'))
        tick(deadline)
        through=time.monotonic();charged={p:r['seconds'] for p,r in parts.items()}
        charged['boundary']=carried.get('boundary',0)+through-started
        overhead=sum(carried.values())+through-entry-sum(charged.values());charged['main']+=overhead
        S.require(overhead>=0 and all(charged[p]<=S.CAPS[p][1] for p in S.CAPS),'common publication cost charged, no borrowing')
        Q.save(folder/'final_accounting.json',dict(seconds=sum(carried.values())+through-entry,charged_part_seconds=charged,
            shared_boundary_bookkeeping_charged_main=overhead,
            complete_sha256=sha(folder/'complete.json'),bytes_before_receipt=Q.PRIOR_BYTES+Q.size(folder)))
    except BaseException as error:
        cost=dict(carried)
        cost.update({p:v['seconds'] for p,v in parts.items()})
        if 'started' in locals():cost[part]=carried.get(part,0)+time.monotonic()-started
        Q.save(folder/'stopped.json',dict(kind=type(error).__name__,message=str(error),seconds=time.monotonic()-entry,
            charged_part_seconds=cost,
            completed_verification_ledgers=len(list(folder.glob('*_verify_*_ledger.json'))),
            started_verification_calls=len(list(folder.glob('*_verify_*_started.json')))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','batch','audit','freeze-resume','resume'))
    parser.add_argument('folder',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--diagnostic-root',type=Path);parser.add_argument('--diagnostic-manifest',type=Path)
    parser.add_argument('--control',type=Path);parser.add_argument('--parent',type=Path)
    parser.add_argument('--original-execution',type=Path);parser.add_argument('--continuation',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze(args.folder,args.diagnostic_root,args.diagnostic_manifest,args.control)
    elif args.mode=='batch':batch(args.folder,args.manifest)
    elif args.mode=='resume':batch(args.folder,args.manifest,continuation=args.continuation)
    elif args.mode=='freeze-resume':freeze_resume(args.folder,args.manifest,args.parent,args.original_execution,args.control)
    else:print(json.dumps(audit(args.folder,args.manifest,source_root=args.original_execution),sort_keys=True))
