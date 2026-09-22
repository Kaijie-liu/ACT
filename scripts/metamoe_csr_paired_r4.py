"""Frozen FOUR-call old-input smoke; never selects or launches a new cohort."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_execution import supervise, terminal
from metamoe_paired_execution_r2 import validate as validate_parent, worker

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'configs/recent_moe/metamoe_csr_diagnostic_r4.json'
OLD = ROOT/'configs/recent_moe/metamoe_csr_smoke_r3.json'
GATE = ROOT/'docs/metamoe_csr_diagnostic_20260922_r4.json'
OUTPUT = '/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke'
PROTOCOL = 'metamoe_csr_spatial_r4_complete_old_input_smoke'
ROSTER = [('cifar10_0', 'act'), ('cifar10_0', 'author'),
          ('mnist_0', 'author'), ('mnist_0', 'act')]
ARTIFACTS = ('prepared.json', 'functional_rewrite.json', 'obligation_identity.json',
             'request.vnnlib', 'instances.csv', 'backend.yaml', 'backend.stdout',
             'backend.stderr', 'backend_results.pkl')
ADDED = ('scripts/metamoe_csr_paired_r4.py', 'scripts/freeze_metamoe_csr_paired_r4.py',
         'scripts/audit_metamoe_csr_paired_r4.py', 'scripts/replay_metamoe_paired.py',
         'tests/test_metamoe_csr_paired_r4.py', 'docs/metamoe_csr_paired_protocol_20260922_r4.md')


def contract(cfg):
    parent, old = json.loads(PARENT.read_text()), json.loads(OLD.read_text())
    if (cfg['protocol'] != PROTOCOL or cfg['output_root'] != OUTPUT or
            cfg['seconds'] != 300 or cfg['epsilon'] != 2/255 or cfg['margin'] != 1e-7 or
            cfg['hybridz'] != {'sparse_resource_policy': 'csr_spatial_v2',
                              'sparse_representation_bytes': 2**31} or
            cfg['group_rss_limit_bytes'] != 8*2**30 or
            cfg['requests'] != old['requests'] or cfg['roster'] != [list(x) for x in ROSTER] or
            cfg['arms'] != ['act', 'author'] or cfg['automatic_followup'] is not False):
        raise ValueError('fixed old-input smoke contract changed')
    for field in ('python', 'environment', 'repositories', 'checkpoint', 'repo', 'backend_repo'):
        if cfg[field] != parent[field]:
            raise ValueError('inherited model/environment changed: '+field)
    for name, digest in parent['files'].items():
        if cfg['files'].get(name) != digest:
            raise ValueError('parent source binding changed: '+name)
    expected = {str(PARENT.relative_to(ROOT)): sha256(PARENT),
                str(OLD.relative_to(ROOT)): sha256(OLD),
                str(GATE.relative_to(ROOT)): sha256(GATE)}
    if cfg['prerequisites'] != expected:
        raise ValueError('prerequisite binding changed')
    if cfg['parent_config_sha256'] != sha256(PARENT):
        raise ValueError('parent diagnostic identity changed')
    for name, digest in {**expected, **{n: sha256(ROOT/n) for n in ADDED}}.items():
        if cfg['files'].get(name) != digest:
            raise ValueError('missing prerequisite/execution source: '+name)
    gate = json.loads(GATE.read_text())
    if (gate['passed'] is not True or gate['config_sha256'] != sha256(PARENT) or
            gate['opens_formal_cohort'] is not False):
        raise ValueError('old-input representation gate closed')
    for name, digest in gate['files'].items():
        if sha256(name) != digest:
            raise ValueError('representation evidence changed: '+name)


def validate(cfg, *, launch=False):
    contract(cfg)
    validate_parent(cfg)
    if launch:
        if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'feat/moe-route-verification':
            raise ValueError('wrong branch')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise ValueError('dirty worktree; no launch')
        subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['execution_commit'], 'HEAD'],
                       cwd=ROOT, check=True)


def command_for(cfg, path, request_id, arm):
    return [cfg['python'][arm], str(Path(__file__).resolve()), '--config', str(path.resolve()),
            '--worker', request_id, '--arm', arm]


def collect_terminal(folder, receipt):
    value = terminal(folder, receipt)
    if value['result_sha256'] and not value['result_parse_error']:
        result = json.loads((Path(folder)/'result.json').read_text())
        required = {'config_sha256', 'request_id', 'arm', 'label', 'tensor_file_sha256', 'worker_seconds'}
        if not required <= result.keys():
            value['result_parse_error'] = 'incomplete candidate identity/cost schema'
            if receipt['status'] == 'COMPLETED':
                value['status'] = 'ERROR'
    return value


def run(path):
    start = time.monotonic()
    cfg = json.loads(path.read_text())
    validate(cfg, launch=True)
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'launch.json', {'config_sha256': sha256(path), 'pid': os.getpid(),
        'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol': PROTOCOL, 'automatic_followup': False})
    rows, blocked = [], False
    for request_id, arm in ROSTER:
        if blocked:
            rows.append({'id': request_id, 'arm': arm, 'status': 'NOT_STARTED_AFTER_ERROR'})
            continue
        folder = root/f'{request_id}_{arm}'
        receipt = supervise(command_for(cfg, path, request_id, arm), str(ROOT), folder,
                            cfg['seconds'], cfg['group_rss_limit_bytes'])
        row = {'id': request_id, 'arm': arm, **collect_terminal(folder, receipt),
               'seconds': receipt['execution_including_preflight_seconds'],
               'receipt_sha256': sha256(folder/'receipt.json'),
               'artifacts': {name: sha256(folder/name) for name in ARTIFACTS if (folder/name).is_file()}}
        write(folder/'terminal.json', row)
        rows.append(row)
        blocked = row['status'] in ('ERROR', 'SOURCE_CHANGED')
        write(root/f'progress_{len(rows):03d}.json', {'config_sha256': sha256(path), 'rows': rows})
    write(root/'summary.json', {'config_sha256': sha256(path), 'rows': rows})
    write(root/'batch_cost.json', {'config_sha256': sha256(path),
        'batch_wall_through_summary_seconds': time.monotonic()-start,
        'charged_request_seconds': sum(r.get('seconds', 0.) for r in rows),
        'includes': 'validation, launch, requests, cleanup, log hashes, receipts, terminals, progress, summary',
        'excludes': 'this cost-file write and separate independent audit/original-model replay'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--arm', choices=['act', 'author'])
    args = parser.parse_args()
    if args.worker:
        cfg = json.loads(args.config.read_text())
        contract(cfg)  # Full source/environment validation occurs inside the unchanged worker.
        if (args.worker, args.arm) not in ROSTER:
            raise ValueError('unregistered worker')
        from act.config.config import HybridZConfig
        worker(cfg, args.config, args.worker, args.arm, hybridz_config=HybridZConfig(**cfg['hybridz']))
    elif args.arm:
        parser.error('--arm requires --worker')
    else:
        run(args.config)
