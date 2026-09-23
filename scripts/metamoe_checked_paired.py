"""Combined checked ACT vs unchanged author-backend sufficient adapter.

Smoke and later selection are separate immutable protocols. No automatic
followup, retry or resume. Both arms use independently loaded identical
objects, materialized boxes and 300-second process-group supervision.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise
from metamoe_csr_paired_r4 import collect_terminal, ARTIFACTS
from metamoe_paired_execution_r2 import validate as validate_environment, worker

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'configs/recent_moe/metamoe_nonzero_precheck_control_r1.json'
GATE = ROOT/'docs/metamoe_nonzero_precheck_archive_20260923_r1.json'
OLD = ROOT/'configs/recent_moe/metamoe_paired_smoke_r2.json'
SMOKE = ROOT/'configs/recent_moe/metamoe_checked_paired_smoke_r1.json'
OUTPUT = Path('/data1/Kane/MOE/baseline_runs/metamoe_checked_paired_20260923_r1_smoke')
OPTIONS = {'expert_base': True, 'route_feasibility': True, 'score_nonzero': True}
ADDED = ('act/back_end/moe/checked_execution.py', 'scripts/metamoe_checked_paired.py',
         'scripts/audit_metamoe_checked_paired.py', 'tests/test_metamoe_checked_paired.py',
         'docs/metamoe_checked_paired_protocol_20260923_r1.md')


def roster(requests):
    return [[r['id'], arm] for n, r in enumerate(requests)
            for arm in (('act', 'author') if n % 2 == 0 else ('author', 'act'))]


def build_smoke():
    cfg = json.loads(PARENT.read_text())
    validate_environment(cfg)
    gate = json.loads(GATE.read_text())
    if gate['audit'] != 'PASS' or gate['issues'] != 0 or gate['config_sha256'] != sha256(PARENT):
        raise ValueError('precheck control gate')
    requests = json.loads(OLD.read_text())['requests']
    cfg.update(protocol='metamoe_checked_paired_smoke_r1', output_root=str(OUTPUT),
               requests=requests, roster=roster(requests), arms=['act', 'author'],
               act_options=OPTIONS, automatic_followup=False, new_source_rebinding={},
               parent_sha256=sha256(PARENT), gate_sha256=sha256(GATE),
               scope='two old inputs; combined opt-in ACT vs unchanged author-backend sufficient adapter',
               cost='300s complete child including imports, identity checks, loading, lowering, checking, solving and result; postflight/audit separate',
               selection='original CIFAR10/0 and MNIST/0; no new inputs',
               author_scope='same full checkpoint/global property; strict route-invariance sufficient adapter, not raw dynamic graph',
               numerical_guarantees_equated=False)
    cfg['historical_source_rebinding'] = cfg.pop('explicit_source_rebinding', {})
    for key in ('variants', 'expert_checked_base_both', 'router_checked_both',
                'support_unchanged', 'preparation_seconds', 'parent_config_sha256'):
        cfg.pop(key, None)
    cfg['files'].update({str(p): sha256(p) for p in (PARENT, GATE, OLD)})
    cfg['files'].update({p: sha256(ROOT/p) for p in ADDED})
    return cfg


def validate(cfg):
    if cfg['protocol'] == 'metamoe_checked_paired_smoke_r1':
        expected = build_smoke()
        if {k:v for k,v in cfg.items() if k != 'execution_commit'} != {
                k:v for k,v in expected.items() if k != 'execution_commit'}:
            raise ValueError('new smoke contract drift')
    elif cfg['protocol'] == 'metamoe_checked_paired_small_r1':
        from freeze_metamoe_checked_small import contract
        contract(cfg)
    else:
        raise ValueError('unregistered protocol')
    validate_environment(cfg)


def command(cfg, path, rid, arm):
    return [cfg['python'][arm], str(Path(__file__).resolve()), '--config', str(path.resolve()),
            '--worker', rid, '--arm', arm]


def identity(cfg, path, rid, arm):
    req = next(r for r in cfg['requests'] if r['id'] == rid)
    return {'config_sha256': sha256(path), 'request_id': rid, 'arm': arm,
            'tensor_sha256': cfg['files'][req['tensor_file']]}


def observed_worker(cfg, path, rid, arm):
    began = time.monotonic()
    validate(cfg)
    if [rid, arm] not in cfg['roster']:
        raise ValueError('unregistered worker')
    folder = Path(cfg['output_root'])/f'{rid}_{arm}'
    scope = identity(cfg, path, rid, arm)
    write(folder/'runtime.json', {'identity': scope, 'worker_started_monotonic': began,
          'act_options': cfg['act_options'] if arm == 'act' else None})
    if arm == 'author':
        # Deliberately no ACT hook/import or changed backend option here.
        worker(cfg, path, rid, arm)
        return
    from metamoe_expert_trace import ExpertRecorder, install, restore
    from act.back_end.moe.checked_execution import CheckedExecutionOptions, checked_execution
    from act.config.config import HybridZConfig
    recorder = ExpertRecorder(folder/'trace.jsonl', began, scope)
    patches = []
    try:
        recorder.emit('INSTALL_BEGIN')
        patches = install(recorder)
        with checked_execution(folder, request_sha256=scope['config_sha256'],
                input_sha256=scope['tensor_sha256'], options=CheckedExecutionOptions(**cfg['act_options'])) as cls:
            if cls is not None:
                cls.evaluate_spec = recorder.wrap(cls.evaluate_spec, 'ProtectedHZSolver.evaluate_spec')
            recorder.wrap(worker, 'unchanged_metamoe_worker')(cfg, path, rid, arm,
                hybridz_config=HybridZConfig(**cfg['hybridz']))
        recorder.emit('WORKER_COMPLETE')
    finally:
        restore(patches)
        recorder.close()


def evidence_inventory(folder):
    return {str(p.relative_to(folder)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for area in ('protected', 'routing', 'nonzero')
            for p in sorted((folder/area).rglob('*')) if p.is_file()}


def run(path):
    began = time.monotonic()
    cfg = json.loads(path.read_text())
    validate(cfg)
    require_clean()
    subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['execution_commit'], 'HEAD'], cwd=ROOT, check=True)
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'launch.json', {'config_sha256': sha256(path), 'protocol': cfg['protocol'],
          'roster': cfg['roster'], 'automatic_followup': False,
          'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    rows, blocked = [], False
    for rid, arm in cfg['roster']:
        if blocked:
            rows.append({'id': rid, 'arm': arm, 'status': 'NOT_STARTED_AFTER_ERROR'})
            continue
        folder = root/f'{rid}_{arm}'
        receipt = supervise(command(cfg, path, rid, arm), str(ROOT), folder,
                            cfg['seconds'], cfg['group_rss_limit_bytes'])
        post = time.monotonic()
        row = {'id': rid, 'arm': arm, **collect_terminal(folder, receipt),
               'seconds': receipt['execution_including_preflight_seconds'],
               'receipt_sha256': sha256(folder/'receipt.json'),
               'artifacts': {n: sha256(folder/n) for n in (*ARTIFACTS, 'runtime.json', 'trace.jsonl') if (folder/n).is_file()},
               'evidence_artifacts': evidence_inventory(folder),
               'postflight_inventory_seconds': time.monotonic()-post}
        write(folder/'terminal.json', row)
        rows.append(row)
        blocked = row['status'] in ('ERROR', 'SOURCE_CHANGED')
        write(root/f'progress_{len(rows):03d}.json', {'config_sha256': sha256(path), 'rows': rows})
        print(rid, arm, row['status'], row['seconds'], flush=True)
    validate(cfg)
    write(root/'summary.json', {'config_sha256': sha256(path), 'rows': rows})
    write(root/'batch_cost.json', {'config_sha256': sha256(path),
          'batch_wall_through_summary_seconds': time.monotonic()-began,
          'charged_request_seconds': sum(r.get('seconds', 0) for r in rows),
          'excludes': 'cost-file publication and independent audit/replay only; postflight included in batch wall'})


def freeze():
    require_clean()
    if SMOKE.exists() or OUTPUT.exists():
        raise FileExistsError('new execution identity required')
    cfg = build_smoke()
    cfg['execution_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    validate(cfg)
    write(SMOKE, cfg)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-smoke', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--worker')
    p.add_argument('--arm', choices=['act', 'author'])
    a = p.parse_args()
    if a.freeze_smoke:
        if a.config or a.worker or a.arm: p.error('freeze is separate from execution')
        freeze()
    elif a.config:
        if bool(a.worker) != bool(a.arm): p.error('worker and arm required together')
        observed_worker(json.loads(a.config.read_text()), a.config, a.worker, a.arm) if a.worker else run(a.config)
    else:
        p.error('config or freeze required')
