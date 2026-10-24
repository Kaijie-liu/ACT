"""Freeze/control the explicitly authorized six-model validity study."""
import time
ENTRY=time.monotonic()
import argparse
import json
import hashlib
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
    # Training/propagation implementations must be exactly those controlled.
    # This wrapper's freeze-only metadata is not part of the tested operator.
    assert all(receipt['sources'][p]==sha(ROOT/p) for p in CODE if p!='scripts/multilayer_stage1b.py')
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
            head='Linear128->10 once after all routed blocks',balance='0.1*sum(layer_switch_balance)',
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


def freeze_verification(path,control_root,control_manifest=None,prior_control=None):
    """Freeze existing six weights and selection rule; no model/data execution."""
    path=Path(path).resolve()
    from multilayer_training_audit import audit as training_audit
    from multilayer_stage1 import CODE as common_code,read_gzip
    from b1_candidate_history import audit as history_audit,ranges
    run=Path('/data1/Kane/MOE/results/multilayer_stage1b_training_20261023_r1')
    train_path=ROOT/'docs/multilayer_stage1b_training_freeze_20261022.json'
    checked=training_audit(train_path,run);assert checked['status']=='PASS'
    train=load(train_path);complete=load(run/'complete.json')
    history=history_audit()
    old=ROOT/'docs/multilayer_stage1_freeze_20261022_r2.json'
    spec=load(old)
    parent_selection=Path('/data1/Kane/MOE/results/multilayer_stage1_20261022_r1/selection/outcome.json.gz')
    cohort=read_gzip(parent_selection)['value']['samples']
    assert len(cohort)==30
    excluded={i for a,b in history['excluded_inclusive_ranges'] for i in range(a,b+1)}
    excluded.update(s['index'] for s in cohort)
    codes=tuple(dict.fromkeys((*common_code,'scripts/multilayer_stage1b.py','scripts/multilayer_training_audit.py',
        'scripts/b1_candidate_history.py','tests/test_multilayer_stage1b.py',
        'act/pipeline/verification/torch2act.py','act/pipeline/verification/act2torch.py',
        'act/back_end/transfer_functions.py','act/pipeline/moe/experiment1.py',
        'act/back_end/hybridz_tf/hybridz_tf.py','act/back_end/moe/route_a.py')))
    models=[]
    for config,row in zip(train['models'],complete['models']):
        assert config['id']==row['id']
        models.append(dict(**config,checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256']))
    spec.update(schema='multilayer-stage1b-verification-20261024-v1',models=models,
        study=dict(name='residual_stem_validity_followup',arms=['act','genbab_pruned'],
            endpoint_configs=['R3E8'],collapsed_models=[],cost_anchor=None,
            cross_study_decision='same endpoint/mechanism gates;1a/1b disagreement gray/report'),
        source_files={p:sha(ROOT/p) for p in codes},
        training_complete=dict(path=str(run/'complete.json'),sha256=sha(run/'complete.json')),
        training_audit=dict(path=str(ROOT/'docs/multilayer_stage1b_training_records_20261024.json'),
            sha256=sha(ROOT/'docs/multilayer_stage1b_training_records_20261024.json')),
        inherited_protocol=dict(path=str(old),sha256=sha(old)),
        backend_identity_sha256=hashlib.sha256(json.dumps(backend.source_identity(),sort_keys=True).encode()).hexdigest(),
        excluded_ranges=ranges(excluded),
        history_union_sha256=hashlib.sha256(json.dumps(sorted(excluded),separators=(',',':')).encode()).hexdigest(),
        extra_exclusion=dict(path=str(parent_selection),sha256=sha(parent_selection),
            indices=[s['index'] for s in cohort],scope='all30 registered1a inputs, not only completed verifier calls'),
        full_test_telemetry_disclosed=True)
    spec['selection']['criterion']='clean correct under all6 frozen float64 models; no route or verification selection'
    spec['selection']['order']='ascending outside named historical union and all30 stage1a selected identities'
    spec['budgets'].update(stage=20*3600,slots=240)
    spec['endpoint']['primary']='R3E8 incomplete traversal among non-replay-UNSAFE union'
    spec['endpoint']['mechanism_supplement']='registered post189 rule; B/n-prime Wilson lower>0.1 in addition to original bottleneck gate'
    spec['execution'].update(full_order='rank ascending; six model catalog; two arms rotated by (rank+model_position)%2',
        history_caps={'R2E8':784,'R3E8':21952})
    control_manifest=(Path(control_manifest).resolve() if control_manifest else
                      ROOT/'docs/multilayer_stage1b_verification_freeze_20261024_r1.json')
    controlled=('tests/test_multilayer_stage1b.py','scripts/multilayer_stage1.py',
        'scripts/multilayer_stage1_audit.py','act/pipeline/moe/multilayer_study.py','act/back_end/moe/multilayer_external.py')
    target=('test_multilayer_stage1b.ResidualQueueControls.test_two_arm_240_full_ledger_and_unsafe_union'
            if prior_control else 'test_multilayer_stage1b')
    spec['controls']=dict(module=target,batch_seconds=300,trained_models=False,
        result=str(control_root/'complete.json'),controlled_sources=controlled,
        manifest_sha256=sha(control_manifest) if control_manifest.exists() else None)
    if prior_control:
        spec['controls']['prior_result']=dict(path=str(Path(prior_control).resolve()),sha256=sha(prior_control),
            meaning='catalog control PASS retained; only corrected ledger fixture reruns; prior failures preserved')
    if path!=control_manifest:
        result=load(control_root/'complete.json');assert result['status']=='PASS'
        prior=load(control_manifest)
        assert all(prior['source_files'][name]==spec['source_files'][name] for name in controlled)
    save(path,spec)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['control','freeze','freeze-verification']);p.add_argument('path',type=Path)
    p.add_argument('--control-root',type=Path);p.add_argument('--control-manifest',type=Path)
    p.add_argument('--prior-control',type=Path);a=p.parse_args()
    if a.mode=='control':control(a.path)
    elif a.mode=='freeze':freeze(a.path,a.control_root)
    else:freeze_verification(a.path,a.control_root,a.control_manifest,a.prior_control)
