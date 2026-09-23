"""Two old-input requests: repaired BN, protected native vs checked base.

This is a bounded execution diagnostic, not a cohort or timing significance test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_bn_corrected_control import changed_parent, PARENT
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise
from metamoe_csr_paired_r4 import collect_terminal
from metamoe_paired_execution_r2 import validate as validate_environment, worker

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/recent_moe/metamoe_checked_base_control_r1.json'
OUTPUT=Path('/data1/Kane/MOE/baseline_runs/metamoe_checked_base_control_20260923_r1')
GATE=ROOT/'docs/metamoe_bn_corrected_archive_20260923_r1.json'
VARIANTS=('native','checked')
ADDED=('act/back_end/solver/checked_base_session.py','tests/test_checked_base_session.py',
    'scripts/metamoe_checked_base_control.py','scripts/audit_metamoe_checked_base_control.py',
    'scripts/metamoe_bn_corrected_control.py','docs/metamoe_checked_base_protocol_20260923_r1.md')


def build():
    cfg=changed_parent(json.loads(PARENT.read_text()))
    gate=json.loads(GATE.read_text())
    if gate['audit']!='PASS' or gate['status']!='POINT_CONFORMANCE_PASS':raise ValueError('conformance gate closed')
    cfg.update(protocol='repaired_bn_checked_base_two_old_requests_r1',output_root=str(OUTPUT),
        variants=list(VARIANTS),automatic_followup=False,parent_sha256=sha256(PARENT),gate_sha256=sha256(GATE),
        scope='MNIST0 twice; only base proposal changes; not a new cohort or significance test',
        cost='300s full request, at most30s/expert, 10% base cap includes candidate check+publication; no reset',
        explicit_source_rebinding={'act/pipeline/verification/torch2act.py':{
            'old':json.loads(PARENT.read_text())['files']['act/pipeline/verification/torch2act.py'],
            'new':cfg['files']['act/pipeline/verification/torch2act.py']}})
    cfg['files'].update({str(PARENT):sha256(PARENT),str(GATE):sha256(GATE),**{p:sha256(ROOT/p) for p in ADDED}})
    return cfg


def validate(cfg):
    expected=build()
    if {k:v for k,v in cfg.items() if k!='execution_commit'}!={k:v for k,v in expected.items() if k!='execution_commit'}:
        raise ValueError('frozen control drift')
    validate_environment(cfg)


def command(cfg,variant):
    return [cfg['python']['act'],str(Path(__file__).resolve()),'--worker',variant]


def identity(cfg,variant):
    return {'config_sha256':sha256(CONFIG),'request_id':'mnist_0','arm':'act','variant':variant,
            'tensor_sha256':cfg['files'][cfg['requests'][0]['tensor_file']]}


def observed_worker(cfg,variant):
    began=time.monotonic();validate(cfg)
    from metamoe_expert_trace import ExpertRecorder, install, restore
    from act.back_end.solver.protected_hz_solver import protected_experts
    from act.back_end.solver.checked_base_session import checked_base_experts
    from act.config.config import HybridZConfig
    folder=OUTPUT/variant/'mnist_0_act';recorder=ExpertRecorder(folder/'trace.jsonl',began,identity(cfg,variant));patches=[]
    try:
        recorder.emit('INSTALL_BEGIN');patches=install(recorder)
        ctx=protected_experts(folder/'protected') if variant=='native' else checked_base_experts(
            folder/'protected',request_sha256=sha256(CONFIG),input_sha256=sha256(cfg['requests'][0]['tensor_file']))
        with ctx as cls:
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
        if blocked:
            rows.append({'variant':variant,'status':'NOT_STARTED_AFTER_ERROR'});continue
        folder=OUTPUT/variant/'mnist_0_act'
        receipt=supervise(command(cfg,variant),str(ROOT),folder,cfg['seconds'],cfg['group_rss_limit_bytes'])
        post=time.monotonic()
        artifacts={str(p.relative_to(folder)):{'sha256':sha256(p),'bytes':p.stat().st_size}
                   for p in sorted((folder/'protected').rglob('*')) if p.is_file()}
        row={'variant':variant,'id':'mnist_0','arm':'act',**collect_terminal(folder,receipt),
             'seconds':receipt['execution_including_preflight_seconds'],'receipt_sha256':sha256(folder/'receipt.json'),
             'trace_sha256':sha256(folder/'trace.jsonl') if (folder/'trace.jsonl').exists() else None,
             'protected_artifacts':artifacts,'postflight_inventory_seconds':time.monotonic()-post}
        write(folder/'terminal.json',row);rows.append(row)
        write(OUTPUT/f'progress_{len(rows):03d}.json',{'rows':rows,'config_sha256':sha256(CONFIG)})
        blocked=row['status'] in ('ERROR','SOURCE_CHANGED')
        print(variant,row['status'],row['seconds'],flush=True)
    validate(cfg)
    write(OUTPUT/'summary.json',{'rows':rows,'config_sha256':sha256(CONFIG)})
    write(OUTPUT/'batch_cost.json',{'config_sha256':sha256(CONFIG),'batch_wall_through_summary_seconds':time.monotonic()-began,
        'charged_request_seconds':sum(r.get('seconds',0.) for r in rows),
        'excludes':'cost-file publication and independent audit only; postflight separately included in batch wall'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',choices=VARIANTS)
    a=p.parse_args()
    if a.freeze:
        require_clean()
        if CONFIG.exists() or OUTPUT.exists():raise FileExistsError('existing control identity')
        cfg=build();cfg['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        validate(cfg);write(CONFIG,cfg)
    else:
        cfg=json.loads(CONFIG.read_text());observed_worker(cfg,a.worker) if a.worker else run(cfg)
