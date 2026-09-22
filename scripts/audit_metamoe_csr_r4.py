"""R4 saved-only resource/planner audit; no repropagation or bound proof."""
import argparse
import json
from pathlib import Path
import time
from audit_metamoe_csr import diagnostic_review
from metamoe_csr_r4 import validate
from robust_experts_workflow_control import write


def audit(config):
    start = time.monotonic()
    cfg = json.loads(config.read_text())
    validate(cfg)
    review = diagnostic_review(config, Path(cfg['output_root']))
    rows = []
    result = review['result']
    if result:
        for event in result['events']:
            admissions = result['admissions'][event['admission_start']:event['admission_end']]
            if event['kind'] != 'CONV2D':
                continue
            before = [a for a in admissions if a['stage'] == 'conv_planner_pre']
            plans = [a for a in admissions if a['stage'] == 'operator_pre']
            if not plans:  # pre-counter rejection/exception cannot be success
                if result['status'] == 'COMPLETE':
                    raise ValueError('missing convolution plan')
                continue
            if len(plans) != 1 or len(before) != 1 or not before[0]['accepted']:
                raise ValueError('planner admission missing')
            plan = plans[0]['details']
            if (plan['planner'] != 'conv_support_union_v1' or
                    plan['nnz_upper_bound'] > plan['coarse_nnz_upper_bound'] or
                    plan['workspace_reserve_bytes'] > plan['coarse_workspace_reserve_bytes'] or
                    before[0]['requested_reserve_bytes'] != plan['planner_scratch_reserve_bytes']):
                raise ValueError('plan/budget consistency')
            if event['sparse'] and event['sparse']['bytes'] > plan['estimated_retained_bytes']:
                raise ValueError('retained upper bound violated')
            rows.append({'phase': event['phase'], 'layer': event['id'], 'accepted': plans[0]['accepted'],
                'coarse_nnz_upper_bound': plan['coarse_nnz_upper_bound'], 'nnz_upper_bound': plan['nnz_upper_bound'],
                'coarse_workspace_reserve_bytes': plan['coarse_workspace_reserve_bytes'],
                'workspace_reserve_bytes': plan['workspace_reserve_bytes'], 'planner_seconds': plan['planner_seconds'],
                'actual': event['sparse'], 'total_accounted_bytes': plans[0]['total_accounted_bytes']})
    return {**review, 'convolution_plans': rows, 'separate_review_seconds': time.monotonic()-start,
            'trust': 'source/receipt/planner accounting review, not independent HZ or output-bound proof',
            'opens_formal_cohort': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = audit(args.config)
    write(args.output, value)
    print('diagnostic_pass', value['passed'])
