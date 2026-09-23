"""One old-input observation, same300s, no retries/repair/search in this version."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_paired_execution_r2 import validate as environment, worker
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise
from metamoe_csr_paired_r4 import collect_terminal

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'configs/recent_moe/metamoe_checked_paired_small_r1.json'
CONFIG = ROOT/'configs/backend_controls/metamoe_las_observe_r1.json'
OUTPUT = Path('/data1/Kane/MOE/baseline_runs/metamoe_las_observe_20260923_r1')
SOURCES = ('scripts/metamoe_las_observe.py', 'scripts/metamoe_las_diagnostic.py',
           'docs/metamoe_las_observe_protocol_20260923_r1.md', 'tests/test_metamoe_las_observe.py')


def build():
    cfg = json.loads(PARENT.read_text())
    cfg.update(protocol='metamoe_las_observe_r1', output_root=str(OUTPUT),
               requests=[r for r in cfg['requests'] if r['id']=='mnist_7'],
               roster=[['mnist_7','author']], automatic_followup=False,
               scope='same failed input only; instrumentation, no math or option repair')
    cfg['files'].update({str(PARENT):sha256(PARENT), **{s:sha256(ROOT/s) for s in SOURCES}})
    return cfg


def validate(cfg):
    if {k:v for k,v in cfg.items() if k!='execution_commit'} != {k:v for k,v in build().items() if k!='execution_commit'}:
        raise ValueError('observation contract changed')
    environment(cfg)


def observed(cfg, path):
    validate(cfg)
    native = subprocess.run
    target = str(Path(cfg['backend_repo'])/'complete_verifier/abcrown.py')
    folder = Path(cfg['output_root'])/'mnist_7_author'
    def run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd)>1 and cmd[1] == target:
            if len(cmd)!=4 or cmd[2]!='--config':
                raise ValueError('unknown backend command')
            redirected = [cmd[0], str(ROOT/'scripts/metamoe_las_observe.py'),
                '--backend', cfg['backend_repo'], '--events', str(folder/'las_events.jsonl'),
                '--config', cmd[3]]
            write(folder/'backend_launch.json', {'original':cmd, 'redirected':redirected,
                  'config_sha256':sha256(path), 'observation_only':True})
            return native(redirected, *args, **kwargs)
        return native(cmd, *args, **kwargs)
    with patch.object(subprocess, 'run', run):
        worker(cfg, path, 'mnist_7', 'author')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--worker', action='store_true')
    a=p.parse_args()
    if a.freeze:
        require_clean()
        if CONFIG.exists() or OUTPUT.exists(): raise FileExistsError('sealed identity')
        cfg=build()
        cfg['execution_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        validate(cfg);write(CONFIG,cfg)
        return
    cfg=json.loads(CONFIG.read_text());validate(cfg)
    if a.worker:
        observed(cfg,CONFIG);return
    require_clean()
    subprocess.run(['git','merge-base','--is-ancestor',cfg['execution_commit'],'HEAD'],check=True)
    began=time.monotonic();OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT/'launch.json',{'config_sha256':sha256(CONFIG),'roster':cfg['roster'],
          'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    folder=OUTPUT/'mnist_7_author'
    receipt=supervise([cfg['python']['author'],str(Path(__file__).resolve()),'--worker'],
                      str(ROOT),folder,300,8*2**30)
    result=collect_terminal(folder,receipt)
    validate(cfg)
    write(OUTPUT/'summary.json',{'config_sha256':sha256(CONFIG),'terminal':result,
          'receipt':receipt,'batch_seconds':time.monotonic()-began,'automatic_followup':False,
          'excludes':'summary publication and independent saved review'})
    print(result)


if __name__=='__main__': main()
