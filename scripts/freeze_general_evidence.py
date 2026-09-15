"""Freeze/reconstruct NEW clean-only inputs. This module never verifies them."""
import argparse
import copy
import json
from pathlib import Path
import time
from scripts.optional_evidence_dev_contract import ROOT,read,save
from portable_proof.runtime import digest

PROTOCOL=ROOT/'docs/general_evidence_v1_protocol.json'
SELECTION=ROOT/'docs/general_evidence_v1_selection.json'
REVIEW=ROOT/'docs/general_evidence_v1_freeze_review.json'
RAW=ROOT/'data/moe/results/general_evidence_freeze_20260916_v1'


def policy():
    p=read(PROTOCOL)
    if (p['sample_count']!=20 or p['arms']!=['matched','evidence','crown'] or p['epsilon']!=2/255
            or p['total_seconds']!=300 or p['proposal_cap_seconds']!=60 or p['proposal_check_reserve_seconds']!=80):
        raise ValueError('registered policy differs')
    return p


def sources():
    paths=[p for folder in ('act','scripts','moe_evidence','portable_proof') for p in (ROOT/folder).rglob('*.py')]
    paths.extend((PROTOCOL,ROOT/'docs/general_evidence_v1.md'))
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(paths)}


def jobs(samples):
    arms=policy()['arms']
    return [{'rank':i,'dataset_index':s['dataset_index'],'arm':arms[(i+j)%3],
             'job_id':f'rank{i}_{arms[(i+j)%3]}','position':j} for i,s in enumerate(samples) for j in range(3)]


def provenance():
    from act.pipeline.moe.freeze_conv_three_arm import provenance as original
    cfg,ident=original();p=policy();cfg=copy.deepcopy(cfg)
    cfg['selection'].update(sample_count=p['sample_count'],smoke_count=0,start_index=p['start_index'])
    return cfg,ident


def request_for(value,job,head):
    from moe_evidence.schema import classification_properties
    if job not in value['jobs']:raise ValueError('unregistered job')
    sample=value['samples'][job['rank']]
    if sample['dataset_index']!=job['dataset_index']:raise ValueError('rank/index changed')
    dims=value['dimensions'];subject=value['subject']
    r={'schema':'WEIGHTED_TOP2_REQUEST_V1','top_k':2,**dims,'mode':'eval','tie_policy':'ANY_LEGAL_TOPK',
       'gate':'selected_softmax','epsilon':2/255,'model_state':subject['model_state'],'clean_prediction':sample['label'],
       **{k:sample[k] for k in ('center','lower','upper')},'properties':classification_properties(dims['classes'],sample['label'])}
    return {'method':job['arm'],'protocol':policy()['protocol'],'epsilon':2/255,'subject':subject,'sample':sample,
        'tensors':value['materialized_inputs'][str(sample['dataset_index'])],
        'config':value['parent_identity']['method_configs']['monolithic'],'head':head,'evidence_request':r}


def freeze():
    if any(p.exists() for p in (SELECTION,REVIEW,RAW)):raise FileExistsError('no overwrite/reselection')
    controls=read(ROOT/'docs/general_evidence_v1_controls.json')
    if controls['status']!='PASS' or controls['errors'] or controls['failures']:
        raise ValueError('control gate not passed')
    for name,sha in controls['sources'].items():
        if digest((ROOT/name).read_bytes())!=sha:raise ValueError('control source drift')
    from act.pipeline.moe.freeze_conv_three_arm import exclusions,verify_records,clean_selection
    from scripts.run_conv_sign_lp import resources
    import torch
    start=time.monotonic();resource=resources();cfg,ident=provenance()
    records=exclusions();excluded=verify_records(records)
    # Also bind the complete previous convolutional selection, including any unattempted rows.
    from act.pipeline.moe.freeze_conv_three_arm import SELECTION as previous
    from act.pipeline.moe.schedule_confirmation_selection import source_indices
    by={r['path']:r for r in records}
    by[str(previous)]={'path':str(previous),'sha256':digest(previous.read_bytes()),'indices':sorted(source_indices(previous))}
    records=[by[k] for k in sorted(by)];excluded=verify_records(records)
    clean,tensors=clean_selection(cfg,excluded)
    RAW.mkdir(exist_ok=False);(RAW/'inputs').mkdir();save(RAW/'excluded_artifacts.json',records)
    materialized={}
    for i,values in tensors.items():
        path=RAW/'inputs'/f'{i}.pt';torch.save(values,path)
        materialized[str(i)]={'path':str(path),'sha256':digest(path.read_bytes())}
    from act.back_end.moe.factory import load_output_moe_checkpoint
    net,payload=load_output_moe_checkpoint(clean['subject']['checkpoint'],map_location='cpu')
    dimensions={'experts':len(net.experts),'classes':payload['factory_config']['num_classes']}
    rawtest=ROOT/'data/torchvision/CIFAR10/raw/cifar-10-batches-py/test_batch'
    value={'schema':'GENERAL_EVIDENCE_SELECTION_V1','status':'FROZEN_NOT_EXECUTED','protocol':policy(),
        'controls_sha256':digest((ROOT/'docs/general_evidence_v1_controls.json').read_bytes()),
        'source_sha256':sources(),'parent_identity':ident,'dimensions':dimensions,**clean,
        'excluded_indices':sorted(excluded),'exclusion_inventory':{'path':str(RAW/'excluded_artifacts.json'),
            'sha256':digest((RAW/'excluded_artifacts.json').read_bytes()),'sources':len(records)},
        'materialized_inputs':materialized,'dataset':{'path':str(rawtest),'sha256':digest(rawtest.read_bytes()),'size':10000},
        'jobs':jobs(clean['samples']),'execution_started':False,'resource':resource,
        'freeze_seconds':time.monotonic()-start,'scope':'clean-only selection; no verification query'}
    save(SELECTION,value)
    return {'status':value['status'],'indices':[s['dataset_index'] for s in value['samples']],
        'requests':len(value['jobs']),'exclusions':len(excluded),'sources':len(records),'seconds':value['freeze_seconds']}


def audit():
    from act.pipeline.moe.freeze_conv_three_arm import verify_records,clean_selection
    import torch
    started=time.monotonic();v=read(SELECTION);cfg,ident=provenance()
    if v['protocol']!=policy() or v['source_sha256']!=sources() or v['parent_identity']!=ident or v['execution_started']:
        raise ValueError('source/protocol/parent drift')
    if digest((ROOT/'docs/general_evidence_v1_controls.json').read_bytes())!=v['controls_sha256']:
        raise ValueError('control record changed')
    f=v['exclusion_inventory'];path=Path(f['path'])
    if digest(path.read_bytes())!=f['sha256']:raise ValueError('exclusion inventory changed')
    records=read(path);excluded=verify_records(records)
    if sorted(excluded)!=v['excluded_indices'] or len(records)!=f['sources']:raise ValueError('exclusion union mismatch')
    if digest(Path(v['dataset']['path']).read_bytes())!=v['dataset']['sha256']:raise ValueError('dataset changed')
    rebuilt,tensors=clean_selection(cfg,excluded)
    if any(v[k]!=value for k,value in rebuilt.items()) or v['jobs']!=jobs(rebuilt['samples']) or len(v['jobs'])!=60:
        raise ValueError('clean-only order/selection mismatch')
    if set(tensors)!=set(int(k) for k in v['materialized_inputs']) or set(tensors)&excluded:raise ValueError('overlap/tensor inventory')
    from act.back_end.moe.factory import load_output_moe_checkpoint
    net,payload=load_output_moe_checkpoint(v['subject']['checkpoint'],map_location='cpu')
    if v['dimensions']!={'experts':len(net.experts),'classes':payload['factory_config']['num_classes']}:
        raise ValueError('checkpoint dimension drift')
    for i,expected in tensors.items():
        record=v['materialized_inputs'][str(i)];p=Path(record['path'])
        if digest(p.read_bytes())!=record['sha256']:raise ValueError('materialized file drift')
        actual=torch.load(p,map_location='cpu',weights_only=True)
        if actual.keys()!=expected.keys() or any(not torch.equal(actual[k],expected[k]) for k in expected):
            raise ValueError('materialized box differs')
    for job in v['jobs']:
        from moe_evidence.worker import validate_transport
        validate_transport(request_for(v,job,'freeze-audit'))
    return {'status':'PASS','issues':[],'selection_sha256':digest(SELECTION.read_bytes()),
        'samples':20,'requests':60,'indices':[s['dataset_index'] for s in v['samples']],
        'excluded_count':len(excluded),'source_count':len(records),'seconds':time.monotonic()-started,
        'execution_started':False,'scope':'fresh clean reconstruction, identity/exclusion audit; no route or proof queries'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','audit','register'));a=p.parse_args()
    result=freeze() if a.mode=='freeze' else audit()
    if a.mode=='register':
        if REVIEW.exists():raise FileExistsError(REVIEW)
        save(REVIEW,result)
    print(json.dumps(result,indent=2))
