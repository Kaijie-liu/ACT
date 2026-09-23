"""One old physical request twice, changing routing feasibility proposals only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise
from metamoe_csr_paired_r4 import collect_terminal
from metamoe_paired_execution_r2 import validate as validate_environment, worker
from metamoe_checked_base_control import validate as validate_previous

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'configs/recent_moe/metamoe_checked_base_control_r1.json'
GATE=ROOT/'docs/metamoe_checked_base_archive_20260923_r1.json'
CONFIG=ROOT/'configs/recent_moe/metamoe_checked_routing_control_r1.json'
OUTPUT=Path('/data1/Kane/MOE/baseline_runs/metamoe_checked_routing_control_20260923_r1')
VARIANTS=('router_native','router_checked')
ADDED=('act/back_end/solver/checked_route_feasibility.py','tests/test_checked_route_feasibility.py',
       'scripts/metamoe_checked_routing_control.py','scripts/audit_metamoe_checked_routing_control.py',
       'docs/metamoe_checked_routing_protocol_20260923_r1.md')


def build():
    cfg=json.loads(PARENT.read_text());validate_previous(cfg)
    gate=json.loads(GATE.read_text())
    if gate['audit']!='PASS' or not gate['same_expert_matrix']:raise ValueError('prior audit gate closed')
    cfg.update(protocol='metamoe_checked_guarded_routing_r1',output_root=str(OUTPUT),variants=list(VARIANTS),
        parent_sha256=sha256(PARENT),gate_sha256=sha256(GATE),automatic_followup=False,
        scope='old MNIST0 twice; expert checked-base ON in both; only route-module feasibility differs',
        cost='300s outer; unchanged query/native/support allocations; proposal capped3s INSIDE original query; native local fallback remains soft',
        proposal_seconds=3.,support_unchanged=True,expert_checked_base_both=True)
    cfg['files'].update({str(PARENT):sha256(PARENT),str(GATE):sha256(GATE),**{p:sha256(ROOT/p) for p in ADDED}})
    return cfg


def validate(cfg):
    expected=build()
    if {k:v for k,v in cfg.items() if k!='execution_commit'}!={k:v for k,v in expected.items() if k!='execution_commit'}:
        raise ValueError('routing control binding drift')
    validate_environment(cfg)


def command(cfg,variant):return [cfg['python']['act'],str(Path(__file__).resolve()),'--worker',variant]


def identity(cfg,variant):
    return {'config_sha256':sha256(CONFIG),'request_id':'mnist_0','arm':'act','variant':variant,
            'tensor_sha256':cfg['files'][cfg['requests'][0]['tensor_file']]}


def observed_worker(cfg,variant):
    began=time.monotonic();validate(cfg)
    from metamoe_expert_trace import ExpertRecorder,install,restore
    from act.back_end.solver.checked_base_session import checked_base_experts
    from act.back_end.solver.checked_route_feasibility import checked_route_feasibility
    from act.config.config import HybridZConfig
    folder=OUTPUT/variant/'mnist_0_act';recorder=ExpertRecorder(folder/'trace.jsonl',began,identity(cfg,variant));patches=[]
    write(folder/'runtime.json',{'identity':identity(cfg,variant),'worker_started_monotonic':began})
    try:
        recorder.emit('INSTALL_BEGIN');patches=install(recorder)
        args={'request_sha256':sha256(CONFIG),'input_sha256':sha256(cfg['requests'][0]['tensor_file'])}
        with checked_base_experts(folder/'protected',**args) as cls,checked_route_feasibility(
                folder/'routing',**args,enabled=variant=='router_checked'):
            cls.evaluate_spec=recorder.wrap(cls.evaluate_spec,'ProtectedHZSolver.evaluate_spec')
            execution={**cfg,'output_root':str(OUTPUT/variant)}
            recorder.wrap(worker,'unchanged_metamoe_worker')(
                execution,CONFIG,'mnist_0','act',hybridz_config=HybridZConfig(**cfg['hybridz']))
        recorder.emit('WORKER_COMPLETE')
    finally:restore(patches);recorder.close()


def run(cfg):
    began=time.monotonic();validate(cfg);require_clean()
    subprocess.run(['git','merge-base','--is-ancestor',cfg['execution_commit'],'HEAD'],cwd=ROOT,check=True)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    write(OUTPUT/'launch.json',{'config_sha256':sha256(CONFIG),'variants':list(VARIANTS),'automatic_followup':False})
    rows=[];blocked=False
    for variant in VARIANTS:
        if blocked:rows.append({'variant':variant,'status':'NOT_STARTED_AFTER_ERROR'});continue
        folder=OUTPUT/variant/'mnist_0_act'
        receipt=supervise(command(cfg,variant),str(ROOT),folder,cfg['seconds'],cfg['group_rss_limit_bytes'])
        post=time.monotonic()
        artifacts={str(p.relative_to(folder)):{'sha256':sha256(p),'bytes':p.stat().st_size}
                   for area in ('protected','routing') for p in sorted((folder/area).rglob('*')) if p.is_file()}
        row={'variant':variant,'id':'mnist_0','arm':'act',**collect_terminal(folder,receipt),
            'seconds':receipt['execution_including_preflight_seconds'],'receipt_sha256':sha256(folder/'receipt.json'),
            'trace_sha256':sha256(folder/'trace.jsonl') if (folder/'trace.jsonl').exists() else None,
            'runtime_sha256':sha256(folder/'runtime.json') if (folder/'runtime.json').exists() else None,
            'evidence_artifacts':artifacts,'postflight_inventory_seconds':time.monotonic()-post}
        write(folder/'terminal.json',row);rows.append(row)
        write(OUTPUT/f'progress_{len(rows):03d}.json',{'rows':rows,'config_sha256':sha256(CONFIG)})
        blocked=row['status'] in ('ERROR','SOURCE_CHANGED');print(variant,row['status'],row['seconds'],flush=True)
    validate(cfg);write(OUTPUT/'summary.json',{'rows':rows,'config_sha256':sha256(CONFIG)})
    write(OUTPUT/'batch_cost.json',{'config_sha256':sha256(CONFIG),'batch_wall_through_summary_seconds':time.monotonic()-began,
        'charged_request_seconds':sum(r.get('seconds',0) for r in rows),
        'excludes':'this cost-file publication and separate audit; postflight separately included in batch wall'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',choices=VARIANTS)
    a=p.parse_args()
    if a.freeze:
        require_clean()
        if CONFIG.exists() or OUTPUT.exists():raise FileExistsError('new control identity required')
        cfg=build();cfg['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        validate(cfg);write(CONFIG,cfg)
    else:
        cfg=json.loads(CONFIG.read_text());observed_worker(cfg,a.worker) if a.worker else run(cfg)
