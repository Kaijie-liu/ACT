"""Saved-only independent full-state sanity audit of the frozen GPU control."""
import json
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__=='__main__':
    started=time.monotonic()
    path=Path('configs/recent_moe/robust_experts_gpu_step_r3.json')
    cfg=json.loads(path.read_text());root=Path(cfg['output_root'])
    for p,h in cfg['files'].items():
        if sha256(p)!=h:raise ValueError('source/input binding')
    summary=json.loads((root/'summary.json').read_text())
    if summary['config_sha256']!=sha256(path) or not summary['accepted']:
        raise ValueError('not a completed two-arm control')
    sys.path.insert(0,cfg['repo'])
    import torch
    torch.set_num_threads(2)
    rows=[]
    for arm in ['dense','convmoe']:
        row=next(r for r in summary['records'] if r['arm']==arm)
        folder=root/arm;result=json.loads((folder/'result.json').read_text())
        receipt=json.loads((folder/'receipt.json').read_text())
        if result!=row['result'] or receipt!=row['receipt'] or receipt['status']!='COMPLETED':
            raise ValueError('terminal drift')
        for name in ['stdout','stderr']:
            if sha256(folder/(name+'.txt'))!=receipt[name+'_sha256']:raise ValueError('log binding')
        cp=folder/'full_step.ckpt'
        if sha256(cp)!=result['checkpoint_sha256']:raise ValueError('checkpoint changed')
        # Locally generated native checkpoint, exact hash checked before load.
        state=torch.load(cp,map_location='cpu',weights_only=False)
        if state['global_step']!=1 or not state['lr_schedulers'] or len(state['state_dict'])!=result['state_tensors']:
            raise ValueError('incomplete state')
        if not all(torch.isfinite(v).all() for v in state['state_dict'].values()):
            raise ValueError('nonfinite saved model')
        momentum=[]
        for opt in state['optimizer_states']:
            for value in opt['state'].values():
                m=value.get('momentum_buffer')
                if m is not None:
                    if not torch.isfinite(m).all():raise ValueError('nonfinite SGD state')
                    momentum.append(m.numel())
        if not momentum:raise ValueError('missing optimizer continuation state')
        rows.append({'arm':arm,'result':result,'finite_momentum_buffers':len(momentum),
            'execution_seconds':receipt['execution_including_preflight_seconds'],
            'with_postflight_seconds':receipt['total_with_postflight_seconds'],
            'saved_hashes':{str(p):sha256(p) for p in [cp,folder/'result.json',folder/'receipt.json',folder/'prepared.json']}})
    write('docs/robust_experts_gpu_step_archive_20260922_r3.json',{
        'audit':'INDEPENDENT_SAVED_GPU_STEP_REVIEW_PASS','rows':rows,
        'config_sha256':sha256(path),'summary_sha256':sha256(root/'summary.json'),
        'separate_audit_seconds':time.monotonic()-started,
        'full_GPU_batch_update_save':True,'full_training':False,'exact_GPU_resume':False,
        'formal_SAFE':False,'trust':'checks stored state and frozen native execution records; does not rerun update'})
