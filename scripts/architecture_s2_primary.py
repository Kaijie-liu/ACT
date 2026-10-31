"""S2 primary worker/receiver plugged into the existing S1 request supervisor."""
import time
from pathlib import Path
from scoped_proof.io import sha,tick


def worker(folder,query,spec,store):
    from multilayer_train import setup
    torch=setup(1)
    import numpy as np
    from architecture_s1_execution import read_record,require
    from act.pipeline.moe.architecture_s2_evaluation import (
        primary_batch,check_primary_arrays,pack_arrays,unpack_arrays,check_inputs,
    )
    deadline=query['worker_deadline'];began=time.monotonic()
    require(spec.get('study')=='S2A','S2a primary registration')
    if spec.get('synthetic_control',False):
        from torch import nn
        from act.back_end.moe.multilayer import RoutedLayer
        from act.back_end.moe.multilayer_external import SequentialTop2
        family=query['family'];count=8
        def constant(values):
            layer=nn.Linear(3072,len(values))
            with torch.no_grad():layer.weight.zero_();layer.bias.copy_(torch.tensor(values))
            return nn.Sequential(nn.Flatten(),layer)
        if family=='dense':model=constant([2.]+[0.]*9)
        else:model=SequentialTop2([RoutedLayer(constant([0.]*8),[constant([2.]+[0.]*9) for _ in range(8)],k=2,mode='selected_softmax')])
        inputs=dict(pixels=np.zeros((count,3,32,32),dtype=np.uint8),labels=np.arange(count,dtype=np.int64))
        checkpoint_digest=input_digest='fixed synthetic source'
    else:
        item=next(m for m in spec['models'] if m['id']==query['model'])
        family=item['family'];count=10000
        require(sha(item['checkpoint'])==item['checkpoint_sha256'],'selected S2 weights')
        if family=='dense':from act.pipeline.moe.architecture_dense import load_checkpoint
        else:from act.pipeline.moe.multilayer_training import load_checkpoint
        model,payload=load_checkpoint(item['checkpoint'])
        require(payload['architecture']==item['architecture'] and payload['seed']==item['seed'],
                'S2 primary model identity')
        require(str(torch.__version__)==spec['primary_torch_version'],'frozen primary numerical environment')
        source=spec['inputs'];require(sha(source['path'])==source['sha256'],'same physical S2 inputs')
        inputs=unpack_arrays(read_record(source['path']));check_inputs(inputs)
        checkpoint_digest=item['checkpoint_sha256'];input_digest=source['sha256']
    load_seconds=time.monotonic()-began;tick(deadline)
    model=model.float().eval();chunks=[];batches=[]
    for start in range(0,count,spec['primary_batch_size']):
        tick(deadline);end=min(count,start+spec['primary_batch_size'])
        arrays,metrics=primary_batch(model,torch.from_numpy(inputs['pixels'][start:end]),
                                    torch.from_numpy(inputs['labels'][start:end]),
                                    layer_diagnostics=spec.get('layer_diagnostics')=='IBP_LAYER_WIDTHS_V1')
        chunks.append(arrays);batches.append(dict(start=start,stop=end,**metrics))
        store.write(folder/'progress.json.gz',dict(clock=time.monotonic(),phase='primary_IBP',
            observed=end,expected=count,complete=False),replace=True)
    arrays={k:np.concatenate([part[k] for part in chunks],axis=0) for k in chunks[0]}
    check_primary_arrays(arrays,family=family,count=count)
    tick(deadline)
    packet=pack_arrays(arrays)
    # Compare exact stored array bytes; no floating-point recomputation here.
    restored=unpack_arrays(packet)
    require(all(restored[k].dtype==v.dtype and restored[k].shape==v.shape and
                restored[k].tobytes()==v.tobytes() for k,v in arrays.items()),'lossless primary archive')
    array_digest=store.write(folder/'arrays.json.gz',packet)
    value=dict(result=dict(status='AUXILIARY_COMPLETE',evidence_grade='IBP_NUMERICAL_POLICY',
        formal_SAFE=False,evaluation=dict(family=family,mode='primary',source_complete_proof=False)),
        coverage=None,clean_correct=None,family=family,count=count,properties=9,
        array_sha256=array_digest,checkpoint_sha256=checkpoint_digest,input_sha256=input_digest,
        clean=int(arrays['clean'].sum()),certified=int(arrays['certified'].sum()),
        single_route=int((arrays['route_count']==1).sum()) if family=='moe' else None,
        batches=batches,load_seconds=load_seconds,worker_seconds=time.monotonic()-began,
        torch=str(torch.__version__),bound_source_sha256=sha(Path(__file__).resolve().parents[1]/
            'act/pipeline/moe/route_envelope_ibp.py'))
    tick(deadline)
    digest=store.write(folder/'outcome.json.gz',dict(request_sha256=sha(folder/'request.json.gz'),
                      clock=time.monotonic(),value=value))
    tick(deadline);store.write(folder/'published.json.gz',dict(sha256=digest,clock=time.monotonic()))
    tick(deadline);return 0


def receive(folder,query,spec,value,deadline):
    # Called AFTER existing common source/request/publication/deadline/session checks.
    from architecture_s1_execution import read_record,require
    from act.pipeline.moe.architecture_s2_evaluation import unpack_arrays,check_primary_arrays,summarize_layer_diagnostics
    require(spec.get('study')=='S2A' and value['result']['status']=='AUXILIARY_COMPLETE',
            'primary measurement, not a request certificate')
    require(value['result']['evaluation']['mode']=='primary' and value['properties']==9,'all primary properties')
    count=8 if spec.get('synthetic_control',False) else 10000
    require(value['count']==count,'all primary inputs')
    if not spec.get('synthetic_control',False):
        item=next(m for m in spec['models'] if m['id']==query['model'])
        require(value['family']==item['family'] and value['checkpoint_sha256']==item['checkpoint_sha256']
                and value['input_sha256']==spec['inputs']['sha256'],'received primary source')
        require(value['torch']==spec['primary_torch_version'],'received primary environment')
    require(sha(folder/'arrays.json.gz')==value['array_sha256'],'primary array binding')
    arrays=unpack_arrays(read_record(folder/'arrays.json.gz'))
    check_primary_arrays(arrays,family=value['family'],count=count)
    require(value['clean']==int(arrays['clean'].sum()) and value['certified']==int(arrays['certified'].sum()),
            'received primary flags')
    if value['family']=='moe':require(value['single_route']==int((arrays['route_count']==1).sum()),'route diagnostic')
    intervals=[(b['start'],b['stop']) for b in value['batches']]
    require(intervals==[(s,min(count,s+spec['primary_batch_size'])) for s in range(0,count,spec['primary_batch_size'])],
            'complete ordered primary batches')
    if spec.get('layer_diagnostics')=='IBP_LAYER_WIDTHS_V1':
        require(summarize_layer_diagnostics(value['batches'],family=value['family'])['count']==count,
                'complete diagnostic cohort and cost')
    require(value['bound_source_sha256']==spec['source_files']['act/pipeline/moe/route_envelope_ibp.py'],
            'same training/evaluation bound function')
    tick(deadline)
    return 'AUXILIARY_COMPLETE',value
