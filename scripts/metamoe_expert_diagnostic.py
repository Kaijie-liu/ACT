"""ONE old MNIST0 ACT request, unchanged queries/options, durable observation.

Freeze and execution are separate; no resume, retry, comparison or cohort gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_execution import supervise
from metamoe_csr_paired_r4 import collect_terminal
from metamoe_paired_execution_r2 import validate as validate_parent, worker

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'configs/recent_moe/metamoe_csr_paired_smoke_r4.json'
PRIOR = ROOT/'docs/metamoe_csr_smoke_archive_20260922_r4.json'
DEFAULT = ROOT/'configs/recent_moe/metamoe_expert_trace_r1.json'
OUTPUT = '/data1/Kane/MOE/baseline_runs/metamoe_expert_trace_20260922_r1'
PROTOCOL = 'metamoe_expert_observation_r1_one_old_mnist0'
ADDED = ('scripts/metamoe_expert_trace.py', 'scripts/metamoe_expert_diagnostic.py',
    'scripts/audit_metamoe_expert_diagnostic.py', 'scripts/f0_timing_trace.py',
    'scripts/audit_conv_f0_timing.py', 'tests/test_metamoe_expert_diagnostic.py',
    'docs/metamoe_expert_diagnostic_protocol_20260922_r1.md')


def contract(cfg):
    parent = json.loads(PARENT.read_text())
    request = next(r for r in parent['requests'] if r['id'] == 'mnist_0')
    if (cfg['protocol'] != PROTOCOL or cfg['output_root'] != OUTPUT or
            cfg['requests'] != [request] or cfg['roster'] != [['mnist_0', 'act']] or
            cfg['arms'] != ['act'] or cfg['automatic_followup'] is not False or
            cfg['parent_config_sha256'] != sha256(PARENT)):
        raise ValueError('one old-request observation contract changed')
    for key in ('seconds', 'epsilon', 'margin', 'hybridz', 'group_rss_limit_bytes',
                'python', 'environment', 'repositories', 'checkpoint', 'repo', 'backend_repo'):
        if cfg[key] != parent[key]:
            raise ValueError('frozen parent setting changed: '+key)
    expected = {**parent['files'], **{str(p.relative_to(ROOT)): sha256(p) for p in (PARENT, PRIOR)},
                **{p: sha256(ROOT/p) for p in ADDED}}
    if cfg['files'] != expected or cfg['explicit_source_rebinding'] != {}:
        raise ValueError('source binding differs; no production edits allowed')


def validate(cfg, launch=False):
    contract(cfg)
    validate_parent(cfg)
    if launch:
        require_clean()
        subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['execution_commit'], 'HEAD'],
                       cwd=ROOT, check=True)


def require_clean():
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
        raise ValueError('wrong branch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('dirty worktree; commit before freeze/launch')


def build_config():
    cfg = json.loads(PARENT.read_text())
    validate_parent(cfg)
    cfg.update(protocol=PROTOCOL, output_root=OUTPUT,
        requests=[r for r in cfg['requests'] if r['id'] == 'mnist_0'],
        roster=[['mnist_0', 'act']], arms=['act'], automatic_followup=False,
        parent_config_sha256=sha256(PARENT), explicit_source_rebinding={},
        scope='one observation only; unchanged production calls; no efficacy or strict proof claim',
        cost='same300s outer; imports/installation/trace fsync charged; postflight+review separate',
        failure='preserve deadline/error/partial trace; no retry/resume or formal cohort',
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    cfg['files'].update({str(p.relative_to(ROOT)): sha256(p) for p in (PARENT, PRIOR)})
    cfg['files'].update({p: sha256(ROOT/p) for p in ADDED})
    contract(cfg)
    return cfg


def command_for(cfg, path):
    return [cfg['python']['act'], str(Path(__file__).resolve()), '--config', str(path.resolve()), '--worker']


def identity(path):
    cfg = json.loads(path.read_text())
    return {'config_sha256': sha256(path), 'request_id': 'mnist_0', 'arm': 'act',
            'tensor_sha256': cfg['files'][cfg['requests'][0]['tensor_file']]}


def observed_worker(path):
    started = time.monotonic()
    cfg = json.loads(path.read_text())
    contract(cfg)
    from metamoe_expert_trace import ExpertRecorder, install, restore
    recorder = ExpertRecorder(Path(OUTPUT)/'mnist_0_act/trace.jsonl', started, identity(path))
    patches = []
    try:
        recorder.emit('INSTALL_BEGIN')
        patches = install(recorder)
        from act.config.config import HybridZConfig
        recorder.wrap(worker, 'unchanged_metamoe_worker')(
            cfg, path, 'mnist_0', 'act', hybridz_config=HybridZConfig(**cfg['hybridz']))
        recorder.emit('WORKER_COMPLETE')
    finally:
        restore(patches)
        recorder.close()


def run(path):
    started = time.monotonic()
    cfg = json.loads(path.read_text())
    validate(cfg, launch=True)
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'launch.json', {'config_sha256': sha256(path), 'protocol': PROTOCOL,
        'pid': os.getpid(), 'automatic_followup': False,
        'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    folder = root/'mnist_0_act'
    receipt = supervise(command_for(cfg, path), str(ROOT), folder, cfg['seconds'], cfg['group_rss_limit_bytes'])
    row = {'id': 'mnist_0', 'arm': 'act', **collect_terminal(folder, receipt),
        'seconds': receipt['execution_including_preflight_seconds'],
        'receipt_sha256': sha256(folder/'receipt.json'),
        'trace_sha256': sha256(folder/'trace.jsonl') if (folder/'trace.jsonl').exists() else None}
    write(folder/'terminal.json', row)
    write(root/'summary.json', {'config_sha256': sha256(path), 'rows': [row]})
    write(root/'batch_cost.json', {'config_sha256': sha256(path),
        'batch_wall_through_summary_seconds': time.monotonic()-started,
        'charged_request_seconds': row['seconds'],
        'includes': 'validation+launch+charged request+cleanup+postflight+terminal+summary',
        'excludes': 'this write and separate saved-only audit; trace/imports ARE charged'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if a.freeze:
        require_clean()
        if a.config != DEFAULT or Path(OUTPUT).exists():
            raise ValueError('fixed new identity; preserve existing artifacts')
        write(a.config, build_config())
        print('FROZEN_NOT_EXECUTED', sha256(a.config))
    elif a.worker:
        observed_worker(a.config)
    else:
        run(a.config)
