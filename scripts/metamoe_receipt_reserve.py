"""Single-factor ACT execution control: native soft cap 1.0 vs 0.8.

All parent/query deadlines and numerical obligations remain unchanged. Freeze
and execution are separate operations; no resume, retry or automatic launch.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metamoe_checked_paired as paired
from metamoe_paired_execution_r2 import validate as environment, worker
from metamoe_csr_execution import supervise
from metamoe_csr_paired_r4 import collect_terminal, ARTIFACTS
from metamoe_expert_diagnostic import require_clean
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

ROOT = paired.ROOT
PARENT = ROOT/'configs/backend_controls/metamoe_las_followup_r1.json'
GATE = ROOT/'docs/metamoe_receipt_reserve_controls_20260923_r1.json'
CONFIG = ROOT/'configs/backend_controls/metamoe_receipt_reserve_r1.json'
OUTPUT = Path('/data1/Kane/MOE/baseline_runs/metamoe_receipt_reserve_20260923_r1')
VARIANTS = {'full_native': 1.0, 'receipt_reserve': 0.8}
NATIVE = 'act/back_end/solver/native_feasibility_worker.py'
SOURCES = (NATIVE, 'act/back_end/solver/receipt_reserve.py',
    'scripts/metamoe_receipt_reserve.py', 'scripts/audit_metamoe_receipt_reserve.py',
    'tests/test_receipt_reserve.py', 'tests/test_receipt_reserve_protocol.py',
    'scripts/review_receipt_reserve_controls.py',
    'docs/metamoe_receipt_reserve_protocol_20260923_r1.md')


def selected_requests(parent):
    # Fixed prefix in each original raw-order dataset, NOT selected by outcomes.
    return [r for dataset in ('CIFAR10', 'MNIST')
            for r in [q for q in parent['requests'] if q['dataset'] == dataset][:2]]


def roster(requests):
    return [[r['id'], v] for i, r in enumerate(requests)
            for v in (tuple(VARIANTS) if i % 2 == 0 else tuple(reversed(VARIANTS))) ]


def build():
    cfg = json.loads(PARENT.read_text())
    gate = json.loads(GATE.read_text())
    if (gate['status'] != 'PASS' or not gate['controls_passed'] or gate['real_requests_executed'] != 0
            or set(gate['source_sha256']) != set(SOURCES)):
        raise ValueError('controls gate closed')
    for name, digest in gate['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('tested source changed: '+name)
    old = cfg['files'][NATIVE]
    # Explicitly rebind ONLY the instrumented private worker; never mutate old cfg.
    cfg['files'][NATIVE] = sha256(ROOT/NATIVE)
    cfg['requests'] = selected_requests(cfg)
    cfg.update(protocol='metamoe_receipt_reserve_r1', output_root=str(OUTPUT),
        variants=VARIANTS, roster=roster(cfg['requests']), arms=['act'],
        parent_sha256=sha256(PARENT), controls_sha256=sha256(GATE),
        automatic_followup=False, numerical_guarantees_equated=False,
        explicit_worker_rebinding={'path':NATIVE, 'old_sha256':old, 'new_sha256':cfg['files'][NATIVE]},
        scope='four observed prefix inputs, two ACT executions; no author arm or new holdout',
        single_factor='output native soft-limit fraction only; same 300s outer/30s expert/equal-share query deadlines',
        cost='both arms charge imports, validation, model loading, lowering, checking, solving and publication; parent postflight/audit separately disclosed')
    for k in ('backend_las_repair', 'repair_control_sha256', 'author_scope'):
        cfg.pop(k, None)
    cfg['files'].update({str(PARENT):sha256(PARENT), str(GATE):sha256(GATE),
                         **{n:sha256(ROOT/n) for n in SOURCES}})
    return cfg


def validate(cfg):
    expected = build()
    if {k:v for k,v in cfg.items() if k!='execution_commit'} != {
            k:v for k,v in expected.items() if k!='execution_commit'}:
        raise ValueError('receipt-reserve protocol drift')
    environment(cfg)


def view(cfg, variant):
    if variant not in VARIANTS:
        raise ValueError('unregistered variant')
    return {**cfg, 'output_root':str(Path(cfg['output_root'])/variant), '_variant':variant}


def identity(cfg, path, rid, arm='act'):
    req = next(r for r in cfg['requests'] if r['id']==rid)
    return dict(config_sha256=sha256(path), request_id=rid, arm=arm,
                tensor_sha256=cfg['files'][req['tensor_file']], variant=cfg['_variant'])


def command(cfg, path, rid, variant):
    return [cfg['python']['act'], str(Path(__file__).resolve()), '--config',str(path.resolve()),
            '--worker',rid,'--variant',variant]


def observed_worker(cfg, path, rid, variant):
    started = time.monotonic()
    validate(cfg)
    if [rid,variant] not in cfg['roster']:
        raise ValueError('unregistered worker')
    cfg = view(cfg,variant)
    folder = Path(cfg['output_root'])/f'{rid}_act'
    scope = identity(cfg,path,rid)
    write(folder/'runtime.json', {'identity':scope, 'worker_started_monotonic':started,
        'act_options':cfg['act_options'], 'output_budget_fraction':VARIANTS[variant]})
    from metamoe_expert_trace import ExpertRecorder, install, restore
    from act.back_end.moe.checked_execution import CheckedExecutionOptions, checked_execution
    from act.back_end.solver.receipt_reserve import reserve_on_checked_experts
    from act.config.config import HybridZConfig
    recorder = ExpertRecorder(folder/'trace.jsonl',started,scope)
    patches = []
    try:
        recorder.emit('INSTALL_BEGIN')
        patches = install(recorder)
        with checked_execution(folder,request_sha256=scope['config_sha256'],input_sha256=scope['tensor_sha256'],
                options=CheckedExecutionOptions(**cfg['act_options'])) as cls:
            with reserve_on_checked_experts(cls,VARIANTS[variant]):
                cls.evaluate_spec = recorder.wrap(cls.evaluate_spec,'ProtectedHZSolver.evaluate_spec')
                recorder.wrap(worker,'unchanged_metamoe_worker')(
                    cfg,path,rid,'act',hybridz_config=HybridZConfig(**cfg['hybridz']))
        recorder.emit('WORKER_COMPLETE')
    finally:
        restore(patches)
        recorder.close()


def run(path):
    began = time.monotonic()
    cfg = json.loads(path.read_text())
    validate(cfg)
    require_clean()
    subprocess.run(['git','merge-base','--is-ancestor',cfg['execution_commit'],'HEAD'],cwd=ROOT,check=True)
    root = Path(cfg['output_root'])
    root.mkdir(parents=True,exist_ok=False)
    write(root/'launch.json',{'config_sha256':sha256(path),'protocol':cfg['protocol'],
        'roster':cfg['roster'],'automatic_followup':False,
        'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    rows,blocked = [],False
    for rid,variant in cfg['roster']:
        if blocked:
            rows.append({'id':rid,'variant':variant,'arm':'act','status':'NOT_STARTED_AFTER_ERROR'})
            continue
        folder = root/variant/f'{rid}_act'
        receipt = supervise(command(cfg,path,rid,variant),str(ROOT),folder,cfg['seconds'],cfg['group_rss_limit_bytes'])
        post = time.monotonic()
        row = {'id':rid,'variant':variant,'arm':'act',**collect_terminal(folder,receipt),
            'seconds':receipt['execution_including_preflight_seconds'],
            'receipt_sha256':sha256(folder/'receipt.json'),
            'artifacts':{n:sha256(folder/n) for n in (*ARTIFACTS,'runtime.json','trace.jsonl') if (folder/n).is_file()},
            'evidence_artifacts':paired.evidence_inventory(folder),
            'postflight_inventory_seconds':time.monotonic()-post}
        write(folder/'terminal.json',row)
        rows.append(row)
        blocked = row['status'] in ('ERROR','SOURCE_CHANGED')
        write(root/f'progress_{len(rows):03d}.json',{'config_sha256':sha256(path),'rows':rows})
        print(rid,variant,row['status'],row['seconds'],flush=True)
    validate(cfg)
    write(root/'summary.json',{'config_sha256':sha256(path),'rows':rows})
    write(root/'batch_cost.json',{'config_sha256':sha256(path),
        'batch_wall_through_summary_seconds':time.monotonic()-began,
        'charged_request_seconds':sum(r.get('seconds',0.) for r in rows),
        'excludes':'cost-file publication and independent replay/audit; parent postflight included in batch'})


def freeze():
    require_clean()
    if CONFIG.exists() or OUTPUT.exists():
        raise FileExistsError('new identity required')
    cfg = build()
    cfg['execution_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    validate(cfg)
    write(CONFIG,cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',action='store_true');p.add_argument('--config',type=Path)
    p.add_argument('--worker');p.add_argument('--variant',choices=list(VARIANTS))
    a=p.parse_args()
    if a.freeze:
        if a.config or a.worker or a.variant:p.error('freeze is separate')
        freeze()
    elif a.config:
        if bool(a.worker)!=bool(a.variant):p.error('worker and variant together')
        observed_worker(json.loads(a.config.read_text()),a.config,a.worker,a.variant) if a.worker else run(a.config)
    else:p.error('freeze or config required')
