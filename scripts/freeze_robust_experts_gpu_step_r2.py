"""Preserve R1 pre-training serialization error; freeze metadata-only repair."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__=='__main__':
    parent=Path('configs/recent_moe/robust_experts_gpu_step_r1.json')
    cfg=json.loads(parent.read_text())
    old=Path(cfg['output_root'])
    raw=json.loads((old/'summary.json').read_text())
    if raw['accepted'] or len(raw['records'])!=1 or raw['records'][0]['receipt']['status']!='ERROR':
        raise ValueError('unexpected old outcome')
    error=(old/'dense/stderr.txt').read_text()
    if 'Object of type function is not JSON serializable' not in error or (old/'dense/result.json').exists():
        raise ValueError('not the expected pre-training metadata failure')
    write('docs/robust_experts_gpu_step_archive_20260922_r1.json',{
        'audit':'SAVED_PRE_TRAIN_FAILURE_REVIEW','summary':raw,
        'error':'our resolved OmegaConf metadata contains author get_method callables; JSON cannot serialize',
        'files':{str(p):sha256(p) for p in [old/'summary.json',old/'dense/receipt.json',old/'dense/stderr.txt',old/'dense/prepared.json']},
        'trained':False,'retry':'separate R2 keeps symbolic native resolver strings, unchanged recipe/budget'})
    cfg.update(protocol='robust_experts_native_full_batch_gpu_step_r2',
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_gpu_step_20260922_r2')
    for p in [parent,Path(__file__),Path('scripts/robust_experts_gpu_step_control_r2.py'),
        Path('tests/test_robust_experts_gpu_serialization.py'),Path('docs/robust_experts_gpu_step_archive_20260922_r1.json')]:
        cfg['files'][str(p)]=sha256(p)
    write('configs/recent_moe/robust_experts_gpu_step_r2.json',cfg)
