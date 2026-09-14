"""Clean-only freeze and independent reconstruction; never runs verification."""
import argparse
import itertools
import json
from pathlib import Path

import torch

from act.pipeline.moe.conv_training import atomic_json,sha
from act.pipeline.moe.external_compatibility import ROOT,COMMITS,ENV,git
from act.pipeline.moe.schedule_confirmation_selection import inventory,index_fields,source_indices
from act.pipeline.moe.staged_verifier import _tensor_identity,_model_state_identity

BASE=ROOT/'act/pipeline/moe'
PROTOCOL=BASE/'configs/conv_three_arm_protocol_r1.json'
SELECTION=BASE/'configs/conv_three_arm_selection_r1.json'
REVIEW=BASE/'results/conv_three_arm_freeze_review_20260915_r1.json'
RAW=ROOT/'data/moe/results/conv_three_arm_freeze_20260915_r1'
ARMS=('adaptive','monolithic','crown')


def choose_indices(scanned,excluded,count,smoke_count):
    """Predictions and labels only; route/bound fields cannot influence choice."""
    smoke=[];full=[]
    for row in scanned:
        if row['prediction']!=row['label']:continue
        target=smoke if row['dataset_index'] in excluded else full
        limit=smoke_count if target is smoke else count
        if len(target)<limit:target.append(row['dataset_index'])
    if len(full)!=count or len(smoke)!=smoke_count:raise ValueError('insufficient clean-only selection')
    return smoke,full


def jobs(samples):
    return [dict(rank=rank,dataset_index=sample['dataset_index'],method=ARMS[(rank+position)%3],
                 position=position,job_id=f'rank{rank}_{ARMS[(rank+position)%3]}')
            for rank,sample in enumerate(samples) for position in range(3)]


def exclusions():
    records,_=inventory(ignore_selection_paths={SELECTION})
    by_path={r['path']:r for r in records}
    # Include newer external terminals/requests and proof manifests as well as
    # old HZ ledgers. Skip archived source copies, not experimental failures.
    for name in ('terminal.json','request.json','manifest.json'):
        for path in sorted((ROOT/'data/moe/results').rglob(name)):
            if 'source' in path.relative_to(ROOT/'data/moe/results').parts:continue
            ids=index_fields(json.loads(path.read_text()))
            if ids:
                if min(ids)<0 or max(ids)>=10000:raise ValueError(f'non-CIFAR index: {path}')
                by_path[str(path)]={'path':str(path),'sha256':sha(path),'indices':sorted(ids)}
    return [by_path[p] for p in sorted(by_path)]


def verify_records(records):
    for record in records:
        p=Path(record['path'])
        if not p.resolve().is_relative_to(ROOT):raise ValueError('out-of-project exclusion source')
        if sha(p)!=record['sha256'] or sorted(source_indices(p))!=record['indices']:
            raise ValueError(f'exclusion source drift: {p}')
    return {i for r in records for i in r['indices']}


def provenance():
    cfg=json.loads(PROTOCOL.read_text())
    review_path=ROOT/cfg['training_review'];review=json.loads(review_path.read_text())
    chosen=ROOT/cfg['checkpoint']
    if (review['status']!='PASS' or review['landed']['status']!='LANDED_AUDITED'
        or review['landed']['best_epoch']!=89 or review['landed']['checkpoint_sha256']!=cfg['checkpoint_sha256']
        or sha(chosen)!=cfg['checkpoint_sha256']):raise ValueError('selected checkpoint or training review drift')
    for path,commit in COMMITS.items():
        if git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain'):
            raise ValueError('external source drift')
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'act').rglob('*.py')) if '__pycache__' not in p.parts}
    methods={a:{'path':str(ROOT/cfg['arms'][a]['config']),'sha256':sha(ROOT/cfg['arms'][a]['config'])}
             for a in ('adaptive','monolithic')}
    adaptive=json.loads(Path(methods['adaptive']['path']).read_text())
    mono=json.loads(Path(methods['monolithic']['path']).read_text())
    if {k:v for k,v in adaptive.items() if k!='comparison_method'}!={k:v for k,v in mono.items() if k!='comparison_method'}:
        raise ValueError('matched methods do not obtain identical fact/scheduling policies')
    if adaptive['route_complexity_schedule']['multi_pair_tier1_fraction']!=.25:
        raise ValueError('registered schedule changed')
    return cfg,dict(protocol_sha256=sha(PROTOCOL),training_review_sha256=sha(review_path),
        checkpoint_sha256=sha(chosen),method_configs=methods,algorithm_sources=sources,
        external_commits=COMMITS,external_python=ENV)


@torch.no_grad()
def clean_selection(cfg,excluded):
    from act.util.device_manager import initialize_device
    from act.back_end.moe.factory import load_output_moe_checkpoint
    from act.pipeline.moe.train import _load_dataset
    torch.set_num_threads(1);initialize_device('cpu','float64')
    model,payload=load_output_moe_checkpoint(ROOT/cfg['checkpoint'],map_location='cpu')
    model.double().eval()
    if (payload['format']!='act-output-conv-moe-v1' or len(model.experts)!=4
        or model.spec.top_k!=2 or payload['factory_config']['num_classes']!=10):
        raise ValueError('not the frozen E4/C10 family')
    data=_load_dataset('CIFAR10',False,download=False)
    scanned=[];smoke=[];full=[];sample_rows={};tensors={}
    for i in range(cfg['selection']['start_index'],len(data)):
        if i in excluded and len(smoke)>=cfg['selection']['smoke_count']:continue
        if i not in excluded and len(full)>=cfg['selection']['sample_count']:continue
        x,label=data[i];x=x.unsqueeze(0)
        prediction=int(model(x).argmax())
        scanned.append(dict(dataset_index=i,label=int(label),prediction=prediction))
        if prediction!=label:continue
        target=smoke if i in excluded else full
        target.append(i)
        values={'center':x,'lower':(x-2/255).clamp(0,1),'upper':(x+2/255).clamp(0,1)}
        sample_rows[i]=dict(dataset_index=i,label=int(label),clean_prediction=prediction,
            **{k:_tensor_identity(v) for k,v in values.items()})
        tensors[i]=values
        if len(smoke)==cfg['selection']['smoke_count'] and len(full)==cfg['selection']['sample_count']:break
    if (smoke,full)!=choose_indices(scanned,excluded,cfg['selection']['sample_count'],cfg['selection']['smoke_count']):
        raise ValueError('clean-only ordering mismatch')
    samples=[dict(sample_rank=j,**sample_rows[i]) for j,i in enumerate(full)]
    controls=[dict(sample_rank=j,**sample_rows[i]) for j,i in enumerate(smoke)]
    return dict(subject={'checkpoint':str(ROOT/cfg['checkpoint']),'checkpoint_sha256':cfg['checkpoint_sha256'],
            'model_state':_model_state_identity(model),'dataset':'CIFAR10'},
        samples=samples,smoke_samples=controls,scanned_clean_only=scanned),tensors


def freeze():
    if RAW.exists() or SELECTION.exists() or REVIEW.exists():raise FileExistsError('no overwrite/reselection')
    cfg,identity=provenance();records=exclusions();excluded=verify_records(records)
    clean,tensors=clean_selection(cfg,excluded)
    RAW.mkdir(parents=True,exist_ok=False);(RAW/'inputs').mkdir()
    atomic_json(RAW/'excluded_artifacts.json',records)
    files={}
    for i,values in tensors.items():
        path=RAW/'inputs'/f'{i}.pt';torch.save(values,path)
        files[str(i)]={'path':str(path),'sha256':sha(path)}
    raw_test=ROOT/'data/torchvision/CIFAR10/raw/cifar-10-batches-py/test_batch'
    result=dict(schema='conv_three_arm_freeze_v1',status='PROTOCOL_AND_SELECTION_FROZEN_NOT_EXECUTED',
        protocol=cfg,identities=identity,**clean,excluded_indices=sorted(excluded),
        exclusion_inventory={'path':str(RAW/'excluded_artifacts.json'),'sha256':sha(RAW/'excluded_artifacts.json'),'sources':len(records)},
        dataset={'raw_test_batch':str(raw_test),'sha256':sha(raw_test),'size':10000},
        materialized_inputs=files,smoke_jobs=jobs(clean['smoke_samples']),full_jobs=jobs(clean['samples']),
        possible_pairs=[list(p) for p in itertools.combinations(range(4),2)],
        execution={'started':False,'orchestrator_and_final_auditor':'NOT_YET_INTEGRATED',
                   'next_gate':'implement/test frozen three-arm orchestration; freeze execution source; six smoke calls before any full query'})
    atomic_json(SELECTION,result)
    return {'selection':str(SELECTION),'count':len(clean['samples']),'smoke_count':len(clean['smoke_samples']),
            'excluded_count':len(excluded),'sources':len(records),'indices':[s['dataset_index'] for s in clean['samples']]}


def audit():
    value=json.loads(SELECTION.read_text());cfg,identity=provenance()
    if value['protocol']!=cfg or value['identities']!=identity:raise ValueError('frozen policy or implementation drift')
    record=value['exclusion_inventory'];path=Path(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('inventory changed')
    records=json.loads(path.read_text());excluded=verify_records(records)
    if sorted(excluded)!=value['excluded_indices']:raise ValueError('exclusion union mismatch')
    if sha(Path(value['dataset']['raw_test_batch']))!=value['dataset']['sha256']:raise ValueError('test data drift')
    rebuilt,tensors=clean_selection(cfg,excluded)
    for key in rebuilt:
        if value[key]!=rebuilt[key]:raise ValueError(f'clean reconstruction mismatch: {key}')
    if (jobs(value['samples'])!=value['full_jobs'] or jobs(value['smoke_samples'])!=value['smoke_jobs']
            or len(value['full_jobs'])!=90 or len(value['smoke_jobs'])!=6):raise ValueError('job schedule mismatch')
    for i,expected in tensors.items():
        record=value['materialized_inputs'][str(i)];path=Path(record['path'])
        if sha(path)!=record['sha256']:raise ValueError('materialized tensor file changed')
        actual=torch.load(path,weights_only=True,map_location='cpu')
        if actual.keys()!=expected.keys() or any(not torch.equal(actual[k],expected[k]) for k in expected):
            raise ValueError('materialized inputs do not reconstruct')
    if set(s['dataset_index'] for s in value['samples'])&excluded:raise ValueError('prior endpoint overlap')
    return dict(status='PASS',issues=[],selection_sha256=sha(SELECTION),protocol_sha256=sha(PROTOCOL),
        samples=30,smoke_samples=2,full_jobs=90,smoke_jobs=6,excluded_count=len(excluded),
        source_count=len(records),indices=[s['dataset_index'] for s in value['samples']],
        smoke_indices=[s['dataset_index'] for s in value['smoke_samples']],
        scope='separate-process clean-only reconstruction, identities and order; no route/bound query or SAFE audit',
        execution_started=False,orchestrator_and_final_auditor='NOT_YET_INTEGRATED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','audit','register'])
    args=p.parse_args()
    if args.mode=='freeze':result=freeze()
    else:
        result=audit()
        if args.mode=='register':
            if REVIEW.exists():raise FileExistsError(REVIEW)
            atomic_json(REVIEW,result)
    print(json.dumps(result,indent=2))
