"""Frozen zero-solver, same-point layer correspondence diagnostic.

Uses the already frozen replay worker in a NEW output root, with explicit
dependency validation before and after. No transfer function is changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
import metamoe_assignment_replay as replay
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise

ROOT = replay.ROOT
CONFIG = ROOT/'configs/recent_moe/metamoe_assignment_layers_r1.json'
OUTPUT = Path('/data1/Kane/MOE/baseline_runs/metamoe_assignment_layers_20260923_r1')


def concrete_layer(layer, previous, seed):
    """Independent torch point execution of the diagnostic IR subset only."""
    k, p = layer.kind.upper(), layer.params
    if k in ('INPUT','INPUT_SPEC'):
        return seed.reshape(-1)
    if previous is None:
        raise ValueError('missing concrete predecessor')
    x = previous
    if k == 'CONV2D':
        return F.conv2d(x.reshape(p['input_shape']), p['weight'], p.get('bias'),
            p.get('stride',1), p.get('padding',0), p.get('dilation',1), p.get('groups',1)).reshape(-1)
    if k == 'AVGPOOL2D':
        return F.avg_pool2d(x.reshape(p['input_shape']), p['kernel_size'], p.get('stride'),
            p.get('padding',0), p.get('ceil_mode',False), p.get('count_include_pad',True),
            p.get('divisor_override')).reshape(-1)
    if k == 'DENSE':
        return F.linear(x.reshape(1,-1),p['weight'],p.get('bias')).reshape(-1)
    if k in ('SCALE','BIAS'):
        a=p['a' if k=='SCALE' else 'c'].reshape(-1)
        if a.numel() not in (1,x.numel()):
            raise ValueError('ambiguous flattened broadcast')
        return x*a if k=='SCALE' else x+a
    if k == 'RELU':
        return torch.relu(x)
    if k in ('RESHAPE','FLATTEN'):
        return x.reshape(-1)
    raise ValueError('unsupported concrete diagnostic operator '+k)


def evaluate_factors(hz, point, final_nc):
    if hz.n_cont > final_nc or final_nc+hz.n_bin > point.size:
        raise ValueError('factor dimensions outside final model')
    return hz.c + hz.Gc @ point[:hz.n_cont] + hz.Gb @ (2*point[final_nc:final_nc+hz.n_bin]-1)


def freeze():
    require_clean()
    if OUTPUT.exists() or CONFIG.exists(): raise FileExistsError('new diagnostic identity required')
    prior=json.loads(replay.CONFIG.read_text()); replay.validate(prior)
    paths=[replay.CONFIG, ROOT/'docs/metamoe_current_assignment_archive_20260923_r1.json',
        replay.OUTPUT/'terminal.json', replay.OUTPUT/'worker/replay.npz', Path(__file__),
        ROOT/'tests/test_metamoe_assignment_layers.py', ROOT/'docs/metamoe_assignment_layers_protocol_20260923_r1.md']
    cfg={'protocol':'metamoe_same_point_layers_r1','seconds':30.,'group_rss_limit_bytes':prior['group_rss_limit_bytes'],
         'output_root':str(OUTPUT),'python':prior['python'],'files':{str(p):sha256(p) for p in paths},
         'replay_config_sha256':sha256(replay.CONFIG),'new_native_queries':0,'new_proposals':0,
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write(CONFIG,cfg)


def validate(cfg):
    if (cfg['protocol']!='metamoe_same_point_layers_r1' or cfg['seconds']!=30. or cfg['output_root']!=str(OUTPUT)
            or cfg['new_native_queries']!=0 or cfg['new_proposals']!=0):
        raise ValueError('bounded layer diagnostic contract')
    for p,h in cfg['files'].items():
        if sha256(p)!=h: raise ValueError('layer diagnostic identity drift: '+p)
    prior=json.loads(replay.CONFIG.read_text()); parent=replay.validate(prior)
    if cfg['python']!=prior['python'] or cfg['group_rss_limit_bytes']!=prior['group_rss_limit_bytes']:
        raise ValueError('resource/environment drift')
    return prior,parent


def worker(cfg):
    prior,parent=validate(cfg)
    from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
    from act.back_end.solver.isolated_feasibility import save_npz
    with np.load(replay.DIAG/'proposal.npz',allow_pickle=False) as z: point=z['point'].copy()
    with np.load(replay.OUTPUT/'worker/replay.npz',allow_pickle=False) as z:
        seed=torch.from_numpy(z['point'].copy())
    archive=json.loads((ROOT/'docs/metamoe_current_assignment_archive_20260923_r1.json').read_text())
    final_nc=archive['base_check']['variables']-json.loads((replay.DIAG/'assignment.json').read_text())['relu_rows']
    orig_apply,orig_entry=HybridzTF.apply,HybridzTF.set_entry_hz
    phase={'expert':False}; values={}; observations=[]; stored={}
    def entry(tf,hz):
        phase['expert']=hz is not None
        return orig_entry(tf,hz)
    def apply(tf,layer,bounds,net,before,after):
        result=orig_apply(tf,layer,bounds,net,before,after)
        if not phase['expert'] or layer.kind.upper()=='ASSERT': return result
        preds=net.preds.get(layer.id,[])
        if len(preds)>1: raise ValueError('unexpected branching IR in fixed diagnostic')
        x=values.get(preds[0]) if preds else None
        with torch.no_grad(): actual=concrete_layer(layer,x,seed)
        values[layer.id]=actual
        hz=tf.get_sparse_hz(layer.id)
        if hz is None: raise ValueError('missing sparse layer object')
        represented=evaluate_factors(hz,point,final_nc)
        if actual.numel()!=hz.n_out: raise ValueError('layer dimension disagreement')
        delta=float(np.max(np.abs(actual.numpy()-represented),initial=0.))
        info={'layer':int(layer.id),'kind':layer.kind,'n_out':hz.n_out,'n_cont':hz.n_cont,'n_bin':hz.n_bin,
              'max_abs_difference':delta,'point_disagrees_at_1e_9':delta>1e-9,'hz_exact_flag':bool(hz.exact)}
        observations.append(info)
        stored[f'{layer.id}_ir']=actual.numpy().copy(); stored[f'{layer.id}_hz']=represented.copy()
        # Progress survives a later deadline/error; it never becomes a positive certificate.
        write(OUTPUT/'worker/layer_progress.json',{'layers':observations,'identity_validated_at_end':False})
        return result
    # All old dependencies were validated above. Only destination and the
    # already-validated parent lookup are rebound; no old artifact is written.
    with patch.object(replay,'OUTPUT',OUTPUT), patch.object(replay,'validate',return_value=parent), \
         patch.object(HybridzTF,'apply',apply), patch.object(HybridzTF,'set_entry_hz',entry):
        replay.worker(prior)
    validate(cfg)
    result=json.loads((OUTPUT/'worker/result.json').read_text())
    if result.get('model_sha256')!=result.get('saved_model_sha256') or not result.get('fresh_full_matrix_check',{}).get('accepted'):
        raise ValueError('matrix or assignment changed during observation')
    with np.load(OUTPUT/'worker/replay.npz',allow_pickle=False) as z:
        source=z['source'].copy(); represented=z['represented'].copy()
    final=values[observations[-1]['layer']].numpy()
    stored['source']=source;stored['final_represented']=represented
    save_npz(OUTPUT/'worker/layers.npz',**stored)
    write(OUTPUT/'worker/layers.json',{'layers':observations,'arrays_sha256':sha256(OUTPUT/'worker/layers.npz'),
        'first_disagreeing_layer':next((r for r in observations if r['point_disagrees_at_1e_9']),None),
        'final_ir_vs_source':float(np.max(np.abs(final-source))),
        'final_ir_vs_hz':float(np.max(np.abs(final-represented))),
        'identity_validated_at_end':True,'new_native_queries':0,'new_proposals':0,
        'not_source_complete_or_robustness_certificate':True})


def run(cfg):
    import time
    start=time.monotonic();validate(cfg);require_clean();OUTPUT.mkdir(parents=True,exist_ok=False)
    write(OUTPUT/'launch.json',{'config_sha256':sha256(CONFIG),'queries':0,'proposals':0})
    receipt=supervise([cfg['python'],str(Path(__file__).resolve()),'--worker'],str(ROOT),OUTPUT/'worker',
                       cfg['seconds'],cfg['group_rss_limit_bytes'])
    validate(cfg)
    path=OUTPUT/'worker/layers.json'
    result=json.loads(path.read_text()) if path.exists() else None
    write(OUTPUT/'terminal.json',{'outer_status':receipt['status'],'receipt':receipt,'result':result,
        'config_sha256':sha256(CONFIG),'total_seconds':time.monotonic()-start,
        'historical_result_relabelled':False,'opens_formal_cohort':False})
    print(receipt['status'],result['first_disagreeing_layer'] if result else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for f in ('freeze','run','worker'):g.add_argument('--'+f,action='store_true')
    a=p.parse_args()
    if a.freeze:freeze()
    else:
        cfg=json.loads(CONFIG.read_text());worker(cfg) if a.worker else run(cfg)
