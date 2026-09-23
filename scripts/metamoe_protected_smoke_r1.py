"""Freeze/run ONE old-input protected-expert control; no automatic follow-up."""
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
from metamoe_expert_diagnostic import require_clean

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'configs/recent_moe/metamoe_expert_trace_r1.json'
PRIOR = ROOT/'docs/metamoe_expert_diagnostic_archive_20260923_r1.json'
DEFAULT = ROOT/'configs/recent_moe/metamoe_protected_smoke_r1.json'
OUTPUT = '/data1/Kane/MOE/baseline_runs/metamoe_protected_20260923_r1'
PROTOCOL = 'metamoe_protected_expert_old_mnist0_r1'
ADDED = ('act/back_end/solver/native_feasibility_worker.py',
    'act/back_end/solver/isolated_feasibility.py', 'act/back_end/solver/protected_hz_solver.py',
    'scripts/metamoe_protected_smoke_r1.py', 'scripts/audit_metamoe_protected_smoke_r1.py',
    'tests/test_protected_hz_solver.py', 'tests/test_metamoe_protected_smoke.py',
    'docs/metamoe_protected_protocol_20260923_r1.md')


def contract(cfg):
    parent = json.loads(PARENT.read_text())
    if (cfg['protocol'] != PROTOCOL or cfg['output_root'] != OUTPUT or
            cfg['parent_config_sha256'] != sha256(PARENT) or cfg['base_fraction'] != .1 or
            cfg['protected_scope'] != 'ONE_LANE_LINEAR_LE' or cfg['automatic_followup'] is not False):
        raise ValueError('protected control identity changed')
    for k in ('seconds', 'epsilon', 'margin', 'hybridz', 'group_rss_limit_bytes',
              'python', 'environment', 'repositories', 'checkpoint', 'repo', 'backend_repo', 'requests', 'roster', 'arms'):
        if cfg[k] != parent[k]:
            raise ValueError('parent setting changed: '+k)
    expected = {**parent['files'], **{str(p.relative_to(ROOT)): sha256(p) for p in (PARENT, PRIOR)},
                **{n: sha256(ROOT/n) for n in ADDED}}
    if cfg['files'] != expected or cfg['explicit_source_rebinding'] != {}:
        raise ValueError('old production source changed or new source binding missing')


def validate(cfg, launch=False):
    contract(cfg)
    validate_parent(cfg)
    if launch:
        require_clean()
        subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['execution_commit'], 'HEAD'], cwd=ROOT, check=True)


def build_config():
    cfg = json.loads(PARENT.read_text())
    validate_parent(cfg)
    cfg.update(protocol=PROTOCOL, output_root=OUTPUT, parent_config_sha256=sha256(PARENT),
        base_fraction=.1, protected_scope='ONE_LANE_LINEAR_LE', automatic_followup=False,
        scope='ONE old-input opt-in schedule control; not paired efficacy or formal-cohort permission',
        cost='same300s/request and <=30s/expert; 10% base cap; all native startup, transfer, validation and cleanup charged',
        failure='no retry/resume; unknown base never certifies; retain all properties and failed/partial queries',
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    cfg['files'].update({str(p.relative_to(ROOT)): sha256(p) for p in (PARENT, PRIOR)})
    cfg['files'].update({n: sha256(ROOT/n) for n in ADDED})
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
    from act.back_end.solver.protected_hz_solver import protected_experts
    folder = Path(cfg['output_root'])/'mnist_0_act'
    recorder = ExpertRecorder(folder/'trace.jsonl', started, identity(path))
    patches = []
    try:
        recorder.emit('INSTALL_BEGIN')
        patches = install(recorder)
        from act.config.config import HybridZConfig
        with protected_experts(folder/'protected') as cls:
            cls.evaluate_spec = recorder.wrap(cls.evaluate_spec, 'ProtectedHZSolver.evaluate_spec')
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
    inventory_start = time.monotonic()
    artifacts = {str(p.relative_to(folder)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted((folder/'protected').rglob('*')) if p.is_file()}
    inventory_seconds = time.monotonic()-inventory_start
    row = {'id': 'mnist_0', 'arm': 'act', **collect_terminal(folder, receipt),
        'seconds': receipt['execution_including_preflight_seconds'], 'receipt_sha256': sha256(folder/'receipt.json'),
        'trace_sha256': sha256(folder/'trace.jsonl') if (folder/'trace.jsonl').exists() else None,
        'protected_artifacts': artifacts, 'postflight_inventory_seconds': inventory_seconds}
    write(folder/'terminal.json', row)
    write(root/'summary.json', {'config_sha256': sha256(path), 'rows': [row]})
    write(root/'batch_cost.json', {'config_sha256': sha256(path),
        'batch_wall_through_summary_seconds': time.monotonic()-started, 'charged_request_seconds': row['seconds'],
        'includes': 'validation+launch+request+native children+cleanup+postflight+terminal+summary',
        'excludes': 'this write and separate audit; no free cache or history inputs'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT)
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ('freeze', 'run', 'worker'):
        mode.add_argument('--'+flag, action='store_true')
    a = p.parse_args()
    if a.freeze:
        require_clean()
        if a.config != DEFAULT or Path(OUTPUT).exists():
            raise ValueError('fixed new identity only')
        write(a.config, build_config())
        print('FROZEN_NOT_EXECUTED', sha256(a.config))
    elif a.worker:
        observed_worker(a.config)
    else:
        run(a.config)
