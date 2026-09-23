"""Versioned repaired author adapter; original ACT and R1 artifacts unchanged.

Reuse the registered supervisor/ledger and ACT worker under a new identity.
Only the exact author backend subprocess is redirected to the explicit
compatibility wrapper. Freeze-only full followup requires the control audit.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import metamoe_checked_paired as paired
from metamoe_paired_execution_r2 import worker, validate as environment
from robust_experts_workflow_control import write
from recent_moe_deployment import sha256
from metamoe_expert_diagnostic import require_clean

ROOT=paired.ROOT
PARENT=ROOT/'configs/recent_moe/metamoe_checked_paired_small_r1.json'
GATE=ROOT/'docs/metamoe_las_repair_control_archive_20260923_r1.json'
OBSERVED=ROOT/'docs/metamoe_las_observe_archive_20260923_r1.json'
CONFIGS={k:ROOT/f'configs/backend_controls/metamoe_las_{k}_r1.json' for k in ('control','followup')}
SOURCES=('scripts/metamoe_las_repair.py','scripts/metamoe_las_backend.py','scripts/metamoe_las_paired.py',
         'scripts/audit_metamoe_las_paired.py','tests/test_metamoe_las_repair.py',
         'tests/test_metamoe_las_supervision.py','docs/metamoe_las_repair_protocol_20260923_r1.md')


def build(kind):
    cfg=json.loads(PARENT.read_text())
    if kind not in CONFIGS:raise ValueError('unregistered kind')
    observation=json.loads(OBSERVED.read_text())
    if observation['audit']!='PASS' or observation['issues']!=0 or not observation['missing_initially_zero_keys']:
        raise ValueError('schema diagnosis gate closed')
    cfg.update(protocol=f'metamoe_las_{kind}_r1',
        output_root=f'/data1/Kane/MOE/baseline_runs/metamoe_las_{kind}_20260923_r1',
        backend_las_repair='disconnected_zero_branching_metadata_v1',
        author_scope='author backend plus disclosed lAs compatibility repair and strict route-invariance sufficient adapter',
        automatic_followup=False,
        scope='old-input compatibility controls only' if kind=='control' else
              'full20 rerun, same previously observed cohort; NOT new holdout, not a continuation of R1')
    if kind=='control':
        cfg['requests']=[r for r in cfg['requests'] if r['id'] in ('mnist_1','mnist_7')]
        cfg['roster']=[[r['id'],'author'] for r in cfg['requests']]
    else:
        gate=json.loads(GATE.read_text())
        if gate['audit']!='PASS' or not gate['repair_control_gate'] or gate['config_sha256']!=sha256(CONFIGS['control']):
            raise ValueError('repair control gate closed')
        cfg['files'].update({str(GATE):sha256(GATE),str(CONFIGS['control']):sha256(CONFIGS['control'])})
        cfg['repair_control_sha256']=sha256(GATE)
    cfg['files'].update({str(PARENT):sha256(PARENT),str(OBSERVED):sha256(OBSERVED),
                        **{p:sha256(ROOT/p) for p in SOURCES}})
    return cfg


def validate(cfg):
    kind=next((k for k in CONFIGS if cfg.get('protocol')==f'metamoe_las_{k}_r1'),None)
    expected=build(kind)
    if {k:v for k,v in cfg.items() if k!='execution_commit'}!={k:v for k,v in expected.items() if k!='execution_commit'}:
        raise ValueError('repair protocol drift')
    environment(cfg)


def command(cfg,path,rid,arm):
    return [cfg['python'][arm],str(Path(__file__).resolve()),'--config',str(path.resolve()),'--worker',rid,'--arm',arm]


@contextmanager
def protocol_context():
    with patch.object(paired,'validate',validate),patch.object(paired,'command',command):
        yield


def observed_worker(cfg,path,rid,arm):
    validate(cfg)
    if [rid,arm] not in cfg['roster']:raise ValueError('unregistered worker')
    if arm=='act':
        with protocol_context():paired.observed_worker(cfg,path,rid,arm)
        return
    start=time.monotonic();folder=Path(cfg['output_root'])/f'{rid}_{arm}'
    write(folder/'runtime.json',{'identity':paired.identity(cfg,path,rid,arm),
          'worker_started_monotonic':start,'act_options':None})
    target=str(Path(cfg['backend_repo'])/'complete_verifier/abcrown.py')
    native=subprocess.run
    def run(cmd,*args,**kwargs):
        if isinstance(cmd,list) and len(cmd)>1 and cmd[1]==target:
            if len(cmd)!=4 or cmd[2]!='--config':raise ValueError('unknown backend invocation')
            redirected=[cmd[0],str(ROOT/'scripts/metamoe_las_backend.py'),'--backend',cfg['backend_repo'],
                '--events',str(folder/'las_repair_events.jsonl'),'--request-config',str(path.resolve()),'--config',cmd[3]]
            write(folder/'backend_launch.json',{'original':cmd,'redirected':redirected,
                'config_sha256':sha256(path),'repair':cfg['backend_las_repair']})
            return native(redirected,*args,**kwargs)
        return native(cmd,*args,**kwargs)
    with patch.object(subprocess,'run',run):worker(cfg,path,rid,arm)


def freeze(kind):
    require_clean();path=CONFIGS[kind];cfg=build(kind)
    if path.exists() or Path(cfg['output_root']).exists():raise FileExistsError('new identity required')
    cfg['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    validate(cfg);write(path,cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',choices=list(CONFIGS));p.add_argument('--config',type=Path)
    p.add_argument('--worker');p.add_argument('--arm',choices=['act','author'])
    a=p.parse_args()
    if a.freeze:
        if a.config or a.worker or a.arm:p.error('freeze is separate from run')
        freeze(a.freeze)
    elif a.config:
        if bool(a.worker)!=bool(a.arm):p.error('worker and arm required together')
        if a.worker:observed_worker(json.loads(a.config.read_text()),a.config,a.worker,a.arm)
        else:
            with protocol_context():paired.run(a.config)
    else:p.error('freeze or config required')
