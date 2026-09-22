"""Freeze one old-input R4 diagnostic, never a new cohort or paired run."""
import json
from pathlib import Path
import subprocess
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_r4 import contract

CHANGED = ('act/config/config.py', 'act/back_end/hybridz_tf/hybridz_tf.py',
           'tests/test_metamoe_csr_execution.py')
ADDED = ('act/back_end/hybridz_tf/sparse_conv_plan.py', 'tests/test_sparse_conv_plan.py',
         'scripts/metamoe_csr_r4.py', 'scripts/freeze_metamoe_csr_r4.py',
         'scripts/audit_metamoe_csr_r4.py', 'tests/test_metamoe_csr_r4.py',
         'docs/metamoe_csr_protocol_20260922_r4.md')


def rebind(parent):
    cfg = json.loads(json.dumps(parent))
    changes = {}
    for file, old in parent['files'].items():
        new = sha256(file)
        if old != new:
            if file not in CHANGED:
                raise ValueError('unapproved source drift: '+file)
            changes[file] = {'before': old, 'after': new}
    cfg['files'].update({file: sha256(file) for file in CHANGED+ADDED})
    cfg['explicit_source_rebinding'] = changes
    cfg['protocol'] = 'metamoe_csr_spatial_r4_old_input'
    cfg['seconds'] = 90
    cfg['requests'] = [next(r for r in parent['requests'] if r['id'] == 'mnist_0')]
    cfg['hybridz']['sparse_resource_policy'] = 'csr_spatial_v2'
    cfg['output_root'] = '/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_diagnostic'
    cfg['scope'] = 'old-input zero-solver representation control ONLY; no formal cohort authorization'
    cfg['selection'] = 'existing MNIST0 physical tensor; no reselection/rematerialization'
    cfg['arms'] = ['ACT_ConvSupportUnion_Representation_Diagnostic']
    cfg['cost'] = '90s includes imports/identity/translation/planning/propagation/recording/cleanup; independent audit separate'
    contract(cfg)
    return cfg


if __name__ == '__main__':
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('commit implementation before freeze')
    parent = Path('configs/recent_moe/metamoe_csr_smoke_r3.json')
    cfg = rebind(json.loads(parent.read_text()))
    cfg['parent_config_sha256'] = sha256(parent)
    for file in (parent, Path('docs/metamoe_csr_diagnostic_20260922_r3.json')):
        cfg['files'][str(file)] = sha256(file)
    cfg['execution_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    out = Path('configs/recent_moe/metamoe_csr_diagnostic_r4.json')
    write(out, cfg)
    print('FROZEN_NOT_EXECUTED', sha256(out))
