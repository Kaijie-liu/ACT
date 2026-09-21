"""Compact saved-only intake diagnostic archive; no propagation or solving."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__=='__main__':
    root=Path('/data1/Kane/MOE/baseline_runs/metamoe_hz_intake_diagnostic_20260922_r1')
    raw=json.loads((root/'diagnostic.json').read_text())
    receipt=json.loads((root/'receipt.json').read_text())
    cfg=Path('configs/recent_moe/metamoe_paired_smoke_r2.json')
    if raw['config_sha256']!=sha256(cfg) or receipt['status']!='COMPLETED':
        raise ValueError('diagnostic identity/termination')
    drops=[r for r in raw['events'] if r['drop']]
    if (raw['new_solver_queries']!=0 or len(drops)!=1 or drops[0]['id']!=3 or
            drops[0]['drop']!='sparse_relu_size_limit' or raw['sparse_affine_cell_cap']!=64000000):
        raise ValueError('unexpected diagnostic; review before claim')
    write('docs/metamoe_hz_intake_diagnostic_20260922_r1.json',{
        'audit':'SAVED_DIAGNOSTIC_RECORD_REVIEW_PASS','source':str(root),
        'source_hashes':{str(p):sha256(p) for p in sorted(root.glob('*')) if p.is_file()},
        'result':raw,'total_with_postflight_seconds':receipt['total_with_postflight_seconds'],
        'conclusion':'current ACT representation cap drops shared HZ at first router ReLU; formal paired execution blocked',
        'NOT_concluded':['negative property bound','model unsafe','author comparison won','capacity repair completed']})
