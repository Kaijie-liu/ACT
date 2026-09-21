"""Same bounded native workflow, isolate stage logs after R1 collision."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__=='__main__':
    parent=Path('configs/recent_moe/robust_experts_pipeline_control_r1.json')
    cfg=json.loads(parent.read_text())
    old=Path('docs/robust_experts_pipeline_archive_20260922_r1.json')
    audit=json.loads(old.read_text())
    if audit['audit']!='INDEPENDENT_SAVED_WORKFLOW_REVIEW_PASS' or audit['logging_preservation_pass']:
        raise ValueError('unexpected parent outcome')
    cfg.update(protocol='robust_experts_supervised_full_workflow_control_r2',
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_pipeline_20260922_r2_control',
        repair='stage-isolated CSV paths only; no old logs fabricated')
    for p in [parent,old,Path(__file__),Path('scripts/robust_experts_pipeline_r2.py'),
        Path('scripts/archive_robust_experts_pipeline.py'),Path('tests/test_robust_experts_pipeline_r2.py')]:
        cfg['files'][str(p)]=sha256(p)
    write('configs/recent_moe/robust_experts_pipeline_control_r2.json',cfg)
