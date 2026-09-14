"""Fail-closed, source-isolated launcher for the frozen convolutional recipe."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from act.pipeline.moe.conv_training import CONFIG, atomic_json, sha

ROOT=Path(__file__).resolve().parents[3]
PYTHON='/data1/Kane/miniconda3/envs/act-py312/bin/python'


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def read_json_retry(path, attempts=3, delay=.1):
    for attempt in range(attempts):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):
            if attempt+1==attempts:raise
            time.sleep(delay)


def check_gate(path):
    result=read_json_retry(path)
    if result['status']!='PASS':raise ValueError('compatibility gate is not PASS')
    root=Path(path).parent
    if result['launch']['config_sha256']!=sha(CONFIG):raise ValueError('compatibility recipe drift')
    for name,digest in result['launch']['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('compatibility source drift')
    for name,digest in result['hashes'].items():
        if sha(root/name)!=digest:raise ValueError('compatibility artifact drift')
    a,b=(result['results'][k] for k in ('act','crown'))
    if (a['model_state']!=b['model_state'] or a['inputs']!=b['inputs'] or
            len(a['components'])!=5 or len(a['pair_errors'])!=6):
        raise ValueError('incomplete or mismatched compatibility identities')
    import math
    if (a['model_state']['parameter_count']!=155052 or
            any(not row['exact'] or not row['sparse'] for row in a['components']) or
            any(row['maximum']>1e-10 for row in a['pair_errors']) or b['maximum_probe_error']>1e-10 or
            len(b['lower'][0])!=9 or len(b['upper'][0])!=9 or
            any(not math.isfinite(lo) or not math.isfinite(hi) or lo>hi for lo,hi in zip(b['lower'][0],b['upper'][0]))):
        raise ValueError('compatibility acceptance mismatch')
    return result


def resource_wait(root, phase, limit=86400):
    start=time.monotonic()
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
        disk=shutil.disk_usage(root).free/2**30
        if free>=8192 and disk>=10:return dict(free_gpu_mib=free,free_disk_gib=disk)
        atomic_json(root/'supervisor.json',dict(status='WAITING_FOR_RESOURCES',phase=phase,free_gpu_mib=free,
                    free_disk_gib=disk,waited_seconds=time.monotonic()-start,updated=time.time()))
        if time.monotonic()-start>=limit:raise TimeoutError('resource wait exhausted; no training retry')
        time.sleep(30)


def watch(process,root,phase,timeout):
    start=time.monotonic()
    while process.poll() is None:
        age=0.;heartbeat=None;read_error=None
        path=root/'heartbeat.json'
        if phase=='training':
            if path.exists():
                try:heartbeat=read_json_retry(path);age=time.time()-path.stat().st_mtime
                except (OSError,ValueError) as exc:read_error=str(exc)
            else:age=time.monotonic()-start
        # Slow and dead are different. A live but stale worker is not labelled FAILED.
        status='STALLED_SUSPECTED' if age>1800 else 'RUNNING'
        atomic_json(root/'supervisor.json',dict(status=status,phase=phase,pid=process.pid,
            updated=time.time(),heartbeat_age_seconds=age,heartbeat=heartbeat,read_error=read_error,
            elapsed_seconds=time.monotonic()-start))
        if time.monotonic()-start>timeout:raise TimeoutError('owned worker wall-clock limit')
        time.sleep(10)
    if process.returncode!=0:raise RuntimeError(f'{phase} worker exited {process.returncode}; no automatic restart')


def launch(root,compatibility,smoke):
    if sys.executable!=PYTHON:raise ValueError('use act-py312')
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean feature branch required')
    head=git('rev-parse','HEAD')
    if head!=git('rev-parse','origin/feat/moe-route-verification'):raise ValueError('push source before launch')
    if not root.is_relative_to(ROOT/'data/moe/results'):raise ValueError('invalid run root')
    gate=check_gate(compatibility)
    control=read_json_retry(smoke)
    if (control['status']!='PASS' or control['config_sha256']!=sha(CONFIG)
            or control['worker_sha256']!=sha(ROOT/'act/pipeline/moe/conv_training.py')
            or control['device']!='cuda'):
        raise ValueError('CUDA training smoke identity mismatch')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoints','epochs','source'): (root/name).mkdir()
    shutil.copyfile(CONFIG,root/'config.json')
    from act.util.path_config import get_torchvision_data_root
    data_root=Path(get_torchvision_data_root()).resolve()/'CIFAR10/raw'
    files=[data_root/'cifar-10-batches-py'/name for name in
           ['batches.meta','test_batch',*[f'data_batch_{n}' for n in range(1,6)]]]
    # Immutable source export decouples a running job from later checkout edits.
    subprocess.run(['git','archive','--format=tar','-o',str(root/'source.tar'),head],cwd=ROOT,check=True)
    with tarfile.open(root/'source.tar') as archive:archive.extractall(root/'source',filter='data')
    environment={**os.environ,'PYTHONPATH':str(root/'source'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
        'OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'PYTHONDONTWRITEBYTECODE':'1'}
    versions=subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True)
    atomic_json(root/'launch.json',dict(protocol='SECOND_FAMILY_CONV_TOP2_TRAIN_R1',execution_head=head,
        config_sha256=sha(CONFIG),compatibility_sha256=sha(compatibility),compatibility=gate['launch'],
        smoke_sha256=sha(smoke),source_tar_sha256=sha(root/'source.tar'),python=PYTHON,
        environment=versions.splitlines(),dataset_root=str(data_root),dataset_hashes={str(p):sha(p) for p in files},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,uuid','--format=csv,noheader','-i','0'],text=True).strip(),
        started=time.time(),automatic_retry=False,automatic_resume=False,
        precision='float32; no AMP, TF32 off; deterministic algorithms; cuDNN benchmark off',
        seed_policy='factory/split17; train shuffle and workers17+epoch; validation10000+epoch; test20000'))
    process=None
    try:
        for phase,extra,timeout in [('training',[],72*3600),('audit',['--audit'],3600)]:
            resource_wait(root,phase)
            with (root/(phase+'.log')).open('w') as log:
                process=subprocess.Popen(['nice','-n','10',PYTHON,'-u','-m','act.pipeline.moe.conv_training',
                    '--root',str(root),'--device','cuda',*extra],cwd=root/'source',env=environment,
                    start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
                watch(process,root,phase,timeout)
        landed=read_json_retry(root/'CONV_LANDED_summary.json')
        if landed['status']!='LANDED_AUDITED':raise ValueError('landing audit did not pass')
        atomic_json(root/'supervisor.json',dict(status='LANDED_AUDITED',ended=time.time(),
                    landed_sha256=sha(root/'CONV_LANDED_summary.json'),best_epoch=landed['best_epoch'],test=landed['test']))
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        atomic_json(root/'supervisor.json',dict(status='FAILED',error=str(exc),traceback=traceback.format_exc(),
                    ended=time.time(),automatic_retry=False,checkpoints_preserved=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compatibility',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True)
    args=p.parse_args()
    # Single training supervisor for this family, even if two output roots are used.
    with (ROOT/'data/moe/conv_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def terminate(*_):raise KeyboardInterrupt('supervisor signal')
        signal.signal(signal.SIGTERM,terminate)
        launch(args.output.resolve(),args.compatibility.resolve(),args.smoke.resolve())
