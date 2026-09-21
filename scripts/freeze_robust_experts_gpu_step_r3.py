"""Preserve failed IPC control; freeze location/AF_UNIX repair only."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__=='__main__':
    parent=Path('configs/recent_moe/robust_experts_gpu_step_r2.json')
    cfg=json.loads(parent.read_text());old=Path(cfg['output_root'])
    raw=json.loads((old/'summary.json').read_text())
    if raw['accepted'] or 'AF_UNIX path too long' not in (old/'dense/stderr.txt').read_text():
        raise ValueError('unexpected prior outcome')
    abort=json.loads((old/'operator_abort.json').read_text())
    if (old/'dense/result.json').exists():raise ValueError('unexpected completed update')
    write('docs/robust_experts_gpu_step_archive_20260922_r2.json',{
        'audit':'SAVED_IPC_FAILURE_AND_MANUAL_STOP_REVIEW','summary':raw,'operator_abort':abort,
        'files':{str(p):sha256(p) for p in [old/'summary.json',old/'operator_abort.json',
            old/'dense/receipt.json',old/'dense/stderr.txt',old/'dense/prepared.json']},
        'training_success':False,'not_a_natural_budget_timeout':True,
        'repair':'short private IPC temp path; permit AF_UNIX only; keep all model/PGD/batch/budget settings'})
    cfg.update(protocol='robust_experts_native_full_batch_gpu_step_r3',
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_gpu_step_20260922_r3')
    for p in [parent,Path(__file__),Path('scripts/robust_experts_gpu_step_control_r3.py'),
        Path('scripts/author_local_ipc.py'),Path('tests/test_author_local_ipc.py'),
        Path('docs/robust_experts_gpu_step_archive_20260922_r2.json')]:cfg['files'][str(p)]=sha256(p)
    write('configs/recent_moe/robust_experts_gpu_step_r3.json',cfg)
