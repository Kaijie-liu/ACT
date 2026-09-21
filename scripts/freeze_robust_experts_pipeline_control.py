"""Freeze two full640 training updates plus final native evaluation per model."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__ == '__main__':
    parent=Path('configs/recent_moe/robust_experts_gpu_step_r3.json')
    cfg=json.loads(parent.read_text())
    passed=Path('docs/robust_experts_gpu_step_archive_20260922_r3.json')
    if json.loads(passed.read_text())['audit']!='INDEPENDENT_SAVED_GPU_STEP_REVIEW_PASS':
        raise ValueError('fullbatch prerequisite')
    cfg.update(protocol='robust_experts_supervised_full_workflow_control_r1',mode='control',
        seconds_per_arm=900,execution_freeze_approved=False,
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_pipeline_20260922_r1_control',
        control_scope='2epochs x1 full640 train/val batch; final640 clean +256PGD20 +256APGD20; saved audit',
        long_training=False,automatic_resume=False)
    for p in [parent,passed,Path(__file__),Path('scripts/robust_experts_supervised_pipeline.py'),
        Path('scripts/robust_experts_resume_control.py'),Path('scripts/audit_dual_rs_training_control.py'),
        Path('tests/test_robust_experts_pipeline.py')]:
        cfg['files'][str(p)]=sha256(p)
    write('configs/recent_moe/robust_experts_pipeline_control_r1.json',cfg)
