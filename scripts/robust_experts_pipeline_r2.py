"""R2 storage-only adapter: distinct train/evaluate CSV logs; R1 unchanged."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import robust_experts_supervised_pipeline as core
from recent_moe_deployment import sha256, supervise
from robust_experts_workflow_control import write

_configuration = core.configuration


def configuration(recipe, root, data, mode, stage):
    if stage not in core.STAGES:
        raise ValueError('unknown stage')
    cfg = _configuration(recipe,root,data,mode)
    cfg.logger.csv.version = stage
    return cfg


def manage(cfg,path,arm):
    root=Path(cfg['output_root'])/arm
    for stage in core.STAGES:
        began=time.monotonic()
        write(root/(stage+'_started.json'),{'stage':stage,'unix':time.time(),
            'config_sha256':sha256(path),'arm':arm})
        with (root/(stage+'.stdout')).open('x') as out,(root/(stage+'.stderr')).open('x') as err:
            code=subprocess.run([cfg['python'],str(Path(__file__).resolve()),'--config',str(path.resolve()),
                '--arm',arm,'--stage',stage],stdout=out,stderr=err).returncode
        write(root/(stage+'_finished.json'),{'stage':stage,'returncode':code,'seconds':time.monotonic()-began})
        if code:raise SystemExit(code)
    logs=[root/'local'/stage/'metrics.csv' for stage in ('train','evaluate')]
    if any(not file.exists() for file in logs):raise ValueError('missing isolated metrics')
    write(root/'logging.json',{'separate_stages':True,'files':{str(p):sha256(p) for p in logs},
        'claim':'training metrics retained separately; original R1 files not reconstructed'})
    write(root/'inner_terminal.json',{'status':'PIPELINE_COMPLETED','arm':arm,
        'config_sha256':sha256(path),'stage_hashes':{s:sha256(root/(s+'.json')) for s in core.STAGES}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--arm',choices=['dense','convmoe'])
    p.add_argument('--stage',choices=['manager',*core.STAGES])
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.stage:
        if a.arm is None:raise ValueError('arm required')
        if a.stage=='manager':manage(cfg,a.config,a.arm)
        else:
            core.configuration=lambda recipe,root,data,mode:configuration(recipe,root,data,mode,a.stage)
            {'train':core.train,'evaluate':core.evaluate,'audit':core.audit}[a.stage](cfg,a.config,a.arm)
    else:
        started=time.monotonic()
        core.check_binding(cfg,a.config)
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('unfrozen checkout')
        if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='feat/moe-route-verification':
            raise ValueError('wrong branch')
        root=Path(cfg['output_root']);root.mkdir(exist_ok=False)
        guard_seconds=time.monotonic()-started
        records=[]
        for arm in ['dense','convmoe']:
            receipt=supervise([cfg['python'],str(Path(__file__).resolve()),'--config',str(a.config.resolve()),
                '--arm',arm,'--stage','manager'],str(Path(__file__).resolve().parents[1]),root/arm,
                cfg['seconds_per_arm'],'NATIVE_FULL_WORKFLOW',cpu_only=False)
            row=core.terminal(root/arm,receipt,sha256(a.config),arm,cfg['mode'])
            write(root/arm/'terminal.json',row);records.append(row)
            if not row['accepted']:break
        write(root/'summary.json',{'config_sha256':sha256(a.config),'records':records,
            'accepted':len(records)==2 and all(r['accepted'] for r in records),
            'launch_guard_seconds_outside_arm_budgets':guard_seconds,
            'whole_launch_with_postflight_seconds':time.monotonic()-started})
