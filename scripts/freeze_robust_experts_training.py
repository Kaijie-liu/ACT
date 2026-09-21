"""Freeze tested native200epoch scientific recipe, one bounded attempt per arm."""
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__=='__main__':
    parent=Path('configs/recent_moe/robust_experts_pipeline_control_r2.json')
    cfg=json.loads(parent.read_text())
    evidence=Path('docs/robust_experts_pipeline_archive_20260922_r2.json')
    audit=json.loads(evidence.read_text())
    if (audit['audit']!='INDEPENDENT_SAVED_WORKFLOW_REVIEW_PASS' or
            not audit['logging_preservation_pass'] or not audit['summary']['accepted']):
        raise ValueError('native pipeline/logging controls not accepted')
    for key in ['control_scope','seconds_each','repair']:
        cfg.pop(key,None)
    cfg.update(protocol='robust_experts_paper_recipe_training_r1',mode='training',
        seconds_per_arm=86400,execution_freeze_approved=True,long_training=True,
        output_root='/data1/Kane/MOE/baseline_runs/robust_experts_paper_training_20260922_r1',
        launch_root='/data1/Kane/MOE/baseline_runs/robust_experts_paper_launch_20260922_r1',
        execution_scope='native200epochs, full train/val, final checkpoint, full10k clean/PGD20/APGD20 and saved audit',
        models_in_order=['dense','convmoe'],stop_after_failed_arm=True,
        checkpoint_selection='completed epoch200 only; no selection by robustness or verification',
        predeclared_numerical_identity='explicit R3 eval-mode compatibility variant; native CUDA float32',
        max_active_training_jobs=1,automatic_resume=False,automatic_retry=False,
        automatic_git_write=False,ACT_whole_domain_certification='NOT_PART_OF_TRAINING_RESULT')
    for p in [parent,evidence,Path(__file__),Path('scripts/launch_robust_experts_training.py'),
        Path('tests/test_robust_experts_training_recipe.py')]:cfg['files'][str(p)]=sha256(p)
    write('configs/recent_moe/robust_experts_paper_training_execution_r1.json',cfg)
