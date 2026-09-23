"""R2 recorder-only repair: append-only per-layer progress, R1 retained."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import metamoe_assignment_layers as r1
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_expert_diagnostic import require_clean
from metamoe_csr_execution import supervise

ROOT=r1.ROOT
CONFIG=ROOT/'configs/recent_moe/metamoe_assignment_layers_r2.json'
OUTPUT=Path('/data1/Kane/MOE/baseline_runs/metamoe_assignment_layers_20260923_r2')


def append_progress(path,value):
    path=Path(path)
    if path.name=='layer_progress.json':
        layer=value['layers'][-1]['layer']
        if type(layer) is not int or layer<0:raise ValueError('progress layer identity')
        path=path.with_name(f'layer_progress_{layer:03d}.json')
    return write(path,value)


def validate(cfg):
    if (cfg['protocol']!='metamoe_same_point_layers_r2' or cfg['seconds']!=30. or cfg['output_root']!=str(OUTPUT)
            or cfg['new_native_queries']!=0 or cfg['new_proposals']!=0):raise ValueError('R2 contract')
    for p,h in cfg['files'].items():
        if sha256(p)!=h:raise ValueError('R2 identity drift: '+p)
    c1=json.loads(r1.CONFIG.read_text());prior,parent=r1.validate(c1)
    if cfg['python']!=c1['python'] or cfg['group_rss_limit_bytes']!=c1['group_rss_limit_bytes']:
        raise ValueError('environment/resource change')
    return c1,prior,parent


def freeze():
    require_clean()
    if OUTPUT.exists() or CONFIG.exists():raise FileExistsError('new root required')
    c1=json.loads(r1.CONFIG.read_text());r1.validate(c1)
    terminal=json.loads((r1.OUTPUT/'terminal.json').read_text())
    if terminal['outer_status']!='ERROR' or terminal['result'] is not None:raise ValueError('R1 failure identity')
    paths=[r1.CONFIG,Path(__file__),ROOT/'tests/test_metamoe_assignment_layers_r2.py',
        ROOT/'docs/metamoe_assignment_layers_protocol_20260923_r2.md',r1.OUTPUT/'terminal.json',
        r1.OUTPUT/'worker/stderr.txt',r1.OUTPUT/'worker/layer_progress.json']
    write(CONFIG,{'protocol':'metamoe_same_point_layers_r2','seconds':30.,'output_root':str(OUTPUT),
        'python':c1['python'],'group_rss_limit_bytes':c1['group_rss_limit_bytes'],
        'new_native_queries':0,'new_proposals':0,'files':{str(p):sha256(p) for p in paths},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})


def worker(cfg):
    c1,prior,parent=validate(cfg)
    # R1 source/config are untouched. Its only defective operation is repeated
    # exclusive creation of one progress path. Each layer now has its own file.
    with patch.object(r1,'OUTPUT',OUTPUT), patch.object(r1,'validate',return_value=(prior,parent)), \
         patch.object(r1,'write',side_effect=append_progress):
        r1.worker(c1)
    validate(cfg)


def run(cfg):
    start=time.monotonic();validate(cfg);require_clean();OUTPUT.mkdir(parents=True,exist_ok=False)
    write(OUTPUT/'launch.json',{'config_sha256':sha256(CONFIG),'queries':0,'proposals':0})
    receipt=supervise([cfg['python'],str(Path(__file__).resolve()),'--worker'],str(ROOT),OUTPUT/'worker',
                       cfg['seconds'],cfg['group_rss_limit_bytes'])
    validate(cfg)
    path=OUTPUT/'worker/layers.json';result=json.loads(path.read_text()) if path.exists() else None
    write(OUTPUT/'terminal.json',{'outer_status':receipt['status'],'receipt':receipt,'result':result,
        'config_sha256':sha256(CONFIG),'total_seconds':time.monotonic()-start,
        'historical_result_relabelled':False,'opens_formal_cohort':False})
    print(receipt['status'],result['first_disagreeing_layer'] if result else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for f in ('freeze','run','worker'):g.add_argument('--'+f,action='store_true')
    a=p.parse_args()
    if a.freeze:freeze()
    else:
        cfg=json.loads(CONFIG.read_text());worker(cfg) if a.worker else run(cfg)
