"""Frozen old-input pair: unchanged support vs selected-score nonzero precheck."""
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
from metamoe_paired_execution_r2 import validate as validate_environment,worker

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'configs/recent_moe/metamoe_checked_routing_control_r1.json'
GATE=ROOT/'docs/metamoe_checked_routing_archive_20260923_r1.json'
CONFIG=ROOT/'configs/recent_moe/metamoe_nonzero_precheck_control_r1.json'
OUTPUT=Path('/data1/Kane/MOE/baseline_runs/metamoe_nonzero_precheck_control_20260923_r1')
VARIANTS=('support_native','support_precheck')
SOURCE='act/back_end/moe/class_separated_top1.py'
OLD_SOURCE='19d9aac1f76d20b8fc9089b6f297bbcf41dc9aed3ebeff992ce99831af738307'
ADDED=('act/back_end/solver/checked_nonzero_support.py','tests/test_checked_nonzero_support.py',
    'scripts/metamoe_nonzero_precheck_control.py','scripts/audit_metamoe_nonzero_precheck_control.py',
    'docs/metamoe_nonzero_precheck_protocol_20260923_r1.md')


def build():
    cfg=json.loads(PARENT.read_text());gate=json.loads(GATE.read_text())
    if cfg['files'][SOURCE]!=OLD_SOURCE:raise ValueError('wrong historical entry')
    for name,h in cfg['files'].items():
        if name!=SOURCE and sha256(name)!=h:raise ValueError('sealed dependency changed: '+name)
    if (gate['audit']!='PASS' or gate['config_sha256']!=sha256(PARENT) or not gate['same_guarded_router_matrices']
            or not gate['same_expert_matrix']):raise ValueError('previous gate closed')
    cfg.update(protocol='selected_score_nonzero_precheck_r1',output_root=str(OUTPUT),variants=list(VARIANTS),
        parent_sha256=sha256(PARENT),gate_sha256=sha256(GATE),automatic_followup=False,
        scope='old MNIST0 twice; route and expert checked points ON both; definedness callsite only',
        cost='same300s request/30s support;3s precheck cap inside original support allocation; no reset',
        support_unchanged=False,native_support_policy_unchanged=True,router_checked_both=True,
        nonzero_precheck_seconds=3.,
        new_source_rebinding={SOURCE:{'old':OLD_SOURCE,'new':sha256(ROOT/SOURCE),
            'reason':'same default support through definedness-only hook; nonfinite unresolved observations serialize as null'}})
    cfg['files'][SOURCE]=sha256(ROOT/SOURCE)
    cfg['files'].update({str(PARENT):sha256(PARENT),str(GATE):sha256(GATE),**{p:sha256(ROOT/p) for p in ADDED}})
    return cfg


def validate(cfg):
    expected=build()
    if {k:v for k,v in cfg.items() if k!='execution_commit'}!={k:v for k,v in expected.items() if k!='execution_commit'}:
        raise ValueError('nonzero control binding drift')
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
    from act.back_end.solver.checked_nonzero_support import checked_nonzero_support
    from act.config.config import HybridZConfig
    folder=OUTPUT/variant/'mnist_0_act';recorder=ExpertRecorder(folder/'trace.jsonl',began,identity(cfg,variant));patches=[]
    write(folder/'runtime.json',{'identity':identity(cfg,variant),'worker_started_monotonic':began})
    try:
        recorder.emit('INSTALL_BEGIN');patches=install(recorder)
        args={'request_sha256':sha256(CONFIG),'input_sha256':sha256(cfg['requests'][0]['tensor_file'])}
        with checked_base_experts(folder/'protected',**args) as cls,checked_route_feasibility(
                folder/'routing',**args,enabled=True),checked_nonzero_support(
                folder/'nonzero',**args,enabled=variant=='support_precheck'):
            cls.evaluate_spec=recorder.wrap(cls.evaluate_spec,'ProtectedHZSolver.evaluate_spec')
            recorder.wrap(worker,'unchanged_metamoe_worker')({**cfg,'output_root':str(OUTPUT/variant)},CONFIG,
                'mnist_0','act',hybridz_config=HybridZConfig(**cfg['hybridz']))
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
            for area in ('protected','routing','nonzero') for p in sorted((folder/area).rglob('*')) if p.is_file()}
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
        'excludes':'this cost-file publication and separate audit; postflight included in batch wall'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',choices=VARIANTS)
    a=p.parse_args()
    if a.freeze:
        require_clean()
        if CONFIG.exists() or OUTPUT.exists():raise FileExistsError('new execution identity required')
        cfg=build();cfg['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        validate(cfg);write(CONFIG,cfg)
    else:
        cfg=json.loads(CONFIG.read_text());observed_worker(cfg,a.worker) if a.worker else run(cfg)
