"""Freeze-only same-cohort/method/budget review, including saved control re-audit."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
from robust_experts_workflow_control import write
from recent_moe_deployment import sha256
import metamoe_las_paired as control
import metamoe_las_repair as repair
from audit_metamoe_las_paired import audit as recheck


def audit():
    start=time.monotonic();path=control.CONFIGS['followup']
    cfg=json.loads(path.read_text());parent=json.loads(control.PARENT.read_text())
    control.validate(cfg)
    fields=('requests','roster','arms','python','environment','repositories','checkpoint',
            'repo','backend_repo','seconds','epsilon','margin','hybridz','act_options','group_rss_limit_bytes')
    if any(cfg[k]!=parent[k] for k in fields) or len(cfg['requests'])!=10 or len(cfg['roster'])!=20:
        raise ValueError('same-cohort method/resource drift')
    if any(cfg['files'].get(k)!=v for k,v in parent['files'].items()):
        raise ValueError('old identity overwritten')
    if Path(cfg['output_root']).exists() or cfg['automatic_followup'] is not False:
        raise ValueError('not an unexecuted freeze')
    gpath=control.CONFIGS['control'];gcfg=json.loads(gpath.read_text())
    replay=Path(gcfg['output_root'])/'original_replay.json'
    saved=json.loads(control.GATE.read_text())
    with patch.object(repair,'install',side_effect=AssertionError('no backend')),patch.object(
            repair,'active_ancestors',side_effect=AssertionError('independent check')):
        fresh=recheck(gpath,replay)
    for v in (saved,fresh):v.pop('audit_seconds')
    if saved!=fresh or not fresh['repair_control_gate']:raise ValueError('saved compatibility gate mismatch')
    return {'audit':'PASS','issues':0,'config_sha256':sha256(path),
        'parent_sha256':sha256(control.PARENT),'control_archive_sha256':sha256(control.GATE),
        'analysis_source_sha256':sha256(__file__),'roster':cfg['roster'],'preserved_fields':list(fields),
        'registered_requests':20,'distinct_inputs':10,'new_executions':0,'output_root_exists':False,
        'control_reaudit_identical':True,'production_repair_helpers_disabled_during_review':True,
        'automatic_execution':False,'audit_seconds':time.monotonic()-start,
        'scope':'Observed-cohort full rerun frozen only; not a new holdout or old-result splice.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();v=audit();write(a.output,v);print(v['audit'],v['config_sha256'],'new_executions',v['new_executions'])
