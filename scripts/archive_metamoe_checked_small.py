"""Saved-only freeze review: selection/replay, full costs and no execution."""
import argparse
import json
from pathlib import Path
from audit_metamoe_current_assignment import require, inventory
from audit_metamoe_csr_paired_r4 import check_receipt
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
import freeze_metamoe_checked_small as freeze
import metamoe_checked_paired as paired


def audit():
    cfg=json.loads(freeze.CONFIG.read_text());paired.validate(cfg)
    plan=json.loads(freeze.PLAN.read_text());root=freeze.SELECTION
    selection=json.loads((root/'selection.json').read_text())
    reviewed=json.loads((root/'review/result.json').read_text())
    receipts=[]
    for folder,command in [(root,[plan['python'],str(freeze.ROOT/'scripts/freeze_metamoe_checked_small.py'),'--worker']),
            (root/'review',[plan['python'],str(freeze.ROOT/'scripts/review_metamoe_checked_selection.py'),str(freeze.PLAN),str(root)])]:
        receipt=json.loads((folder/'receipt.json').read_text())
        check_receipt(receipt,{'seconds':receipt['execution_including_preflight_seconds']},plan,command)
        require(receipt['status']=='COMPLETED','incomplete selection/review')
        for stream in ('stdout','stderr'):
            require(receipt[stream+'_sha256']==sha256(folder/f'{stream}.txt'),'selection/replay stream changed')
        receipts.append(receipt)
    require(0<=selection['worker_seconds']<=receipts[0]['execution_including_preflight_seconds'] and
            0<=reviewed['seconds']<=receipts[1]['execution_including_preflight_seconds'],'worker costs')
    require(not Path(cfg['output_root']).exists(),'comparison already executed; not a freeze-only archive')
    saved=json.loads((root/'selection_review.json').read_text())
    require(saved['independent_review_receipt']==receipts[1] and
            saved['independent_replay_sha256']==sha256(root/'review/result.json') and
            saved['through_independent_review_seconds']>=sum(r['total_with_postflight_seconds'] for r in receipts),'complete preparation cost')
    return {'audit':'PASS','issues':0,'state':'FROZEN_NOT_EXECUTED','config_sha256':sha256(freeze.CONFIG),
        'plan_sha256':sha256(freeze.PLAN),'requests':cfg['requests'],'roster':cfg['roster'],
        'per_dataset':plan['per_dataset'],'excluded':plan['excluded'],
        'scanned_prefix_lengths':{k:len(v) for k,v in selection['scans'].items()},
        'independent_source_review':reviewed,'selection_receipt':receipts[0],'review_receipt':receipts[1],
        'preparation_through_review_seconds':saved['through_independent_review_seconds'],
        'cost_excludes':'final manifest publication and this saved-only archival audit',
        'verification_calls':0,'files':inventory(root),'guarantees':'selection/source identity only, not network SAFE'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=audit();write(a.output,v);print(v['audit'],v['state'],len(v['requests']))
