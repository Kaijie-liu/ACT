"""Freeze/control the explicitly authorized six-model validity study."""
import time
ENTRY=time.monotonic()
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
import genbab_cpu_ready as backend

CODE=('scripts/multilayer_train.py','scripts/multilayer_stage1b.py',
      'act/pipeline/moe/multilayer_training.py','act/pipeline/moe/train.py',
      'act/back_end/moe/multilayer.py','act/back_end/moe/multilayer_external.py',
      'act/pipeline/moe/multilayer_study.py','act/back_end/moe/factory.py')


def control(folder):
    assert not folder.exists();folder.mkdir()
    save(folder/'freeze.json',dict(sources={p:sha(ROOT/p) for p in CODE},seconds=300,
        cases=['L2E8 residual_stem','L3E8 residual_stem'],real_training=0,
        acceptance='forward, sum-balance, serialized optimizer continuation, static prefix/head/residual differential'))
    env=dict(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    receipt=execute([backend.ACT,'-I','-B',str(ROOT/'scripts/multilayer_train.py'),
        '--control',str(folder/'fixed'),'--residual-control'],
        run_deadline=ENTRY+294,cleanup_deadline=ENTRY+298,env=env,rss_limit=None,development_policy=POLICY)
    save(folder/'session.json',receipt)
    from b1_observed_batch import session_ok
    good=session_ok(receipt) and (folder/'fixed/complete.json').exists() and time.monotonic()<ENTRY+300
    save(folder/'complete.json',dict(status='PASS' if good else 'FAIL',seconds=time.monotonic()-ENTRY,
        session_sha256=sha(folder/'session.json'),freeze_sha256=sha(folder/'freeze.json'),
        result_sha256=sha(folder/'fixed/complete.json') if good else None))
    print(json.dumps(load(folder/'complete.json')),flush=True)


def freeze(path,control_root):
    original=load(ROOT/'docs/multilayer_training_freeze_20261022.json')
    result=load(control_root/'complete.json');assert result['status']=='PASS'
    receipt=load(control_root/'freeze.json')
    assert receipt['sources']=={p:sha(ROOT/p) for p in CODE}
    models=[]
    for layers in (2,3):
        for seed in range(3):
            models.append(dict(id=f'R{layers}E8_seed{seed}',seed=seed,architecture=dict(
                family='residual_stem_v1',input_dim=3072,hidden_output=128,classes=10,
                router_hidden=[128],expert_hidden=[256,128],layers=layers,experts=8)))
    save(path,dict(schema='multilayer-stage1b-training-20261022-v1',models=models,
        epochs=50,split_seed=0,threads=2,dataset_root=original['dataset_root'],data_files=original['data_files'],
        source_files={p:sha(ROOT/p) for p in CODE},
        original_training_manifest_sha256=sha(ROOT/'docs/multilayer_training_freeze_20261022.json'),
        original_split_sha256=sha('/data1/Kane/MOE/results/multilayer_training_20261022_r1/split.json'),
        control_result=str(control_root/'fixed/complete.json'),control_sha256=sha(control_root/'fixed/complete.json'),
        control_receipt=str(control_root/'complete.json'),control_receipt_sha256=sha(control_root/'complete.json'),
        recipe=dict(stem='Linear3072->128, no extra activation',residual='x+MoE(x), each128->128',
            head='Linear128->10 after every routed block',balance='0.1*sum(layer_switch_balance)',
            optimizer='AdamW',lr=.001,weight_decay=.0001,batch=256,scheduler=None,
            augmentation=None,normalization=None,checkpoint='earliest validation maximum',
            test='report every seed, never select by verifiability'),
        hardware=dict(cpu_only=True,numerical_threads=2,as_bytes=8*2**30,
            reason='bounded concurrent CPU opportunity; training cost separate; no GPU'),
        verification=dict(not_launched=True,models=6,inputs=20,arms=['act','genbab_pruned'],
            slots=240,execution_seconds=20*3600,request_seconds=300,
            selection='fixed ascending common clean-correct outside history and stage1a cohort',
            census='30 inputs x 3 radii; candidate/observed counts separate',
            public_pgd='same fixed recipe, no cross-arm answer sharing',
            decision='same L3 primary rule; compare1a/1b, disagreement gray/report')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['control','freeze']);p.add_argument('path',type=Path)
    p.add_argument('--control-root',type=Path);a=p.parse_args()
    if a.mode=='control':control(a.path)
    else:freeze(a.path,a.control_root)
