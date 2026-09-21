"""Start ONE frozen serial training supervisor; no git writes or retry daemon."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from recent_moe_deployment import sha256
from robust_experts_supervised_pipeline import check_binding
from robust_experts_workflow_control import write


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if cfg['mode']!='training' or cfg['seconds_per_arm']!=86400 or cfg['models_in_order']!=['dense','convmoe']:
        raise ValueError('not the registered long execution')
    check_binding(cfg,a.config)
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='feat/moe-route-verification':
        raise ValueError('wrong branch')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('unfrozen checkout')
    if Path(cfg['output_root']).exists() or Path(cfg['launch_root']).exists():raise ValueError('no repeat or implicit resume')
    free_mib=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free_mib<48*1024 or shutil.disk_usage('/data1/Kane/MOE').free<128*1024**3:
        raise ValueError('resource gate closed; do not evict or fallback')
    root=Path(cfg['launch_root']);root.mkdir(exist_ok=False)
    command=[sys.executable,str(Path(__file__).with_name('robust_experts_pipeline_r2.py').resolve()),
             '--config',str(a.config.resolve())]
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        WANDB_MODE='disabled',CLEARML_OFFLINE_MODE='1')
    with (root/'stdout.txt').open('x') as out,(root/'stderr.txt').open('x') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
    record={'pid':process.pid,'command':command,'config_sha256':sha256(a.config),
        'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'launch_unix':time.time(),'initial_free_gpu_mib':free_mib,'output_root':cfg['output_root'],
        'terminal':cfg['output_root']+'/summary.json','seconds_per_arm':cfg['seconds_per_arm'],
        'order':['dense','convmoe'],'automatic_git_write':False,'automatic_retry':False,
        'automatic_resume':False,'completion_claim':False}
    write(root/'launch.json',record);print(json.dumps(record,indent=2))
