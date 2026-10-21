"""Once-only isolated B2 CPU deployment, using the existing owned supervisor.

Installation is charged separately from verifier requests. Every child is <=300s;
no model, solve, GPU, act-py312 modification or dependency repair loop occurs.
"""
import time
ENTRY=time.monotonic()
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys

ROOT=Path('/data1/Kane/MOE/ACT')
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute

FREEZE=ROOT/'docs/genbab_cpu_deployment_freeze_20261021.json'
RUN=Path('/data1/Kane/MOE/results/genbab_cpu_deployment_20261021_r1')
NEW=Path('/data1/Kane/MOE/envs/genbab-cpu-20261021-v1')
BUILD=Path('/data1/Kane/MOE/baselines/deployments/genbab_cpu_20261021_r1')
TOOL=Path('/data1/Kane/MOE/baselines/alpha-beta-CROWN')
ACT='/data1/Kane/miniconda3/envs/act-py312/bin/python'
BASE='/data1/Kane/MOE/envs/alpha-beta-crown/bin/python'
POLICY='DEV_MARGIN_20261021_V1'


def check(value,message):
    if not value: raise ValueError(message)


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True,timeout=10).strip()


def size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def binding(anchor):
    frozen=load(FREEZE,anchor,2**20)
    check(frozen['run']==str(RUN) and frozen['environment']==str(NEW)
          and frozen['build_directory']==str(BUILD) and frozen['attempts']==1,'frozen deployment targets')
    check(frozen['phase_seconds']==[60,300,300,120] and frozen['total_seconds']==900
          and frozen['deployment_bytes']==8*2**30,'deployment limits')
    for name,digest in frozen['files'].items():
        check(sha(ROOT/name)==digest,'deployment file changed: '+name)
    for name,digest in frozen['interpreters'].items():
        check(sha(name)==digest,'base interpreter changed')
    for path,revision in frozen['sources'].items():
        check(git(path,'rev-parse','HEAD')==revision and not git(path,'status','--porcelain'),
              'pinned external source identity')
    return frozen


def commands():
    # Quiet console remains bounded by the existing16KiB supervisor limit.
    # Complete resolved dependency/download identities are kept in pip reports.
    pip=[str(NEW/'bin/python'),'-I','-m','pip','--disable-pip-version-check','--quiet']
    common=['--no-cache-dir','--retries','0','--timeout','30','--progress-bar','off',
            '--index-url','https://pypi.org/simple']
    return [
        [BASE,'-I','-m','venv','--copies',str(NEW)],
        pip+['install',*common,'--only-binary=:all:','--extra-index-url','https://download.pytorch.org/whl/cpu',
             '--report',str(RUN/'packages.json'),'-r',str(ROOT/'configs/genbab_cpu_20261021_requirements.txt')],
        pip+['install',*common,'--no-deps','--no-build-isolation','--report',str(RUN/'converter.json'),
             'git+https://github.com/Verified-Intelligence/onnx2pytorch@0826948e0d50cd09d909b7f26b2ee9ca2d3d71b3'],
        [str(NEW/'bin/python'),'-I','-B',__file__,'--probe'],
    ]


def limit(deadline):
    check(time.monotonic()<deadline,'deployment deadline')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    # Installations are not request evidence: CPU shared objects can exceed128MiB.
    resource.setrlimit(resource.RLIMIT_FSIZE,(2**30,2**30))
    def stop(*_): os._exit(124)
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,deadline-time.monotonic())


def phase(anchor,index,deadline):
    limit(deadline)
    frozen=binding(anchor)
    check(0<=index<len(frozen['commands']) and commands()==frozen['commands'],'fixed command roster')
    return subprocess.run(frozen['commands'][index],check=False).returncode


def probe():
    check(sys.prefix==str(NEW),'new isolated prefix')
    check(sys.version_info[:2]==(3,11),'author Python family')
    check(not any('site-packages' in p and ('/act-py312/' in p or '/envs/alpha-beta-crown/' in p)
                  for p in sys.path),
          'no prior environment site packages')
    sys.path[:0]=[str(TOOL/'auto_LiRPA'),str(TOOL/'complete_verifier')]
    import torch
    import auto_LiRPA
    import api
    from api import ABCrownSolver,VerificationSpec
    check(torch.__version__=='2.11.0+cpu' and torch.version.cuda is None,'CPU-only wheel')
    check(Path(auto_LiRPA.__file__).resolve().is_relative_to(TOOL/'auto_LiRPA'),'pinned LiRPA source')
    check(Path(api.__file__).resolve()==TOOL/'complete_verifier/api.py','pinned external API')
    check(callable(ABCrownSolver.verify) and callable(VerificationSpec.build_from_input_bounds),'raw-clause API')
    value=dict(status='PASS',python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,
               stdlib_path=os.__file__,api_file=api.__file__,torch=torch.__version__,
               cuda_build=torch.version.cuda,lirpa_file=auto_LiRPA.__file__,
               packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
               native_verifier_calls=0,trained_models=0,gpu_calls=0)
    save(RUN/'probe.json',value)
    print(json.dumps(dict(status='PASS',prefix=sys.prefix,torch=torch.__version__)),flush=True)
    return 0


def main(anchor):
    limit(ENTRY+900)
    frozen=binding(anchor)
    check(sys.executable==ACT,'ACT controller interpreter unchanged')
    check(git(ROOT,'branch','--show-current')=='feat/moe-route-verification'
          and not git(ROOT,'status','--porcelain'),'clean frozen branch')
    check(commands()==frozen['commands'],'frozen command identity')
    check(not RUN.exists() and not NEW.exists() and not BUILD.exists(),'single deployment attempt')
    check(shutil.disk_usage(NEW.parent).free>=16*2**30,'deployment free storage admission')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                   if s.startswith('MemAvailable:'))
    check(available>=8*2**30 and time.monotonic()-ENTRY<60,'one fresh resource admission')
    RUN.mkdir(); BUILD.mkdir(parents=True)
    (BUILD/'tmp').mkdir(); (BUILD/'cache').mkdir()
    env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
             CUDA_VISIBLE_DEVICES='',TMPDIR=str(BUILD/'tmp'),XDG_CACHE_HOME=str(BUILD/'cache'),
             TORCH_HOME=str(BUILD/'cache'),PIP_CONFIG_FILE='/dev/null',GIT_TERMINAL_PROMPT='0')
    result=dict(status='FAIL',freeze_sha256=anchor,execution_head=git(ROOT,'rev-parse','HEAD'),
                phase_receipts=[],native_verifier_calls=0,trained_model_calls=0,
                evidence_grade='DEPLOYMENT_ONLY',full_cold_start_seconds=None)
    try:
        for index,seconds in enumerate(frozen['phase_seconds']):
            phase_end=min(ENTRY+890,time.monotonic()+seconds)
            run_end=phase_end-10  # Cleanup belongs to the same <=300s phase.
            receipt=execute([ACT,'-I','-S','-B',__file__,'--phase',anchor,str(index),str(run_end)],
                            run_deadline=run_end,cleanup_deadline=phase_end,
                            env=env,rss_limit=None,development_policy=POLICY)
            result['phase_receipts'].append(receipt)
            save(RUN/('phase_%d.json'%index),receipt)
            check(receipt['status']=='COMPLETED' and receipt['returncode']==0
                  and receipt['execution_error'] is None and receipt['cleanup_error'] is None
                  and receipt['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION','deployment phase failed')
            check(size(NEW)+size(BUILD)<=frozen['deployment_bytes'],'deployment storage stop')
            check(size(RUN)<=128*2**20,'deployment evidence cap')
        check(load(RUN/'probe.json')['status']=='PASS','import conformance')
        result['status']='PASS'
    except BaseException as exc:
        result['error']=repr(exc)
    result.update(seconds_before_terminal_export=time.monotonic()-ENTRY,
                  environment_bytes=size(NEW),build_bytes=size(BUILD),record_bytes=size(RUN),
                  deployment_storage_gate='PHASE_BOUNDARY_ONLY_NOT_PEAK_HARD_CAP',
                  scope='CPU deployment cost charged separately to B2; no verifier request executed')
    result['deployment_storage_exceeded']=result['environment_bytes']+result['build_bytes']>frozen['deployment_bytes']
    if result['deployment_storage_exceeded']:
        result.update(status='FAIL',error=result.get('error','deployment storage exceeded'))
    # Include the prospective final record, not just the earlier phase prefix.
    terminal_bytes=len(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    if result['record_bytes']+terminal_bytes>128*2**20:
        result.update(status='FAIL',error=result.get('error','deployment evidence exceeded'))
    save(RUN/'result.json',result)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds_before_terminal_export'],
                          error=result.get('error'),environment_bytes=result['environment_bytes'])),flush=True)
    return 0 if result['status']=='PASS' else 1


def audit_saved(anchor):
    """No import or verification rerun; inspect frozen receipts and pip reports."""
    frozen=binding(anchor)
    result=load(RUN/'result.json',limit=128*2**20)
    check(result['freeze_sha256']==anchor and result['native_verifier_calls']==0
          and result['trained_model_calls']==0,'deployment-only identity')
    phases=result['phase_receipts']
    check(0<len(phases)<=len(frozen['commands']),'preserved phase prefix')
    for index,receipt in enumerate(phases):
        check(receipt==load(RUN/('phase_%d.json'%index)),'phase receipt identity')
        check(receipt['started_at']<=receipt['finished_at']
              and receipt['cleanup_deadline']-receipt['started_at']<=frozen['phase_seconds'][index],
              'phase budget/cost identity')
        if receipt['status']=='COMPLETED':
            check(receipt['finished_at']<=receipt['cleanup_deadline'],'completed within cleanup deadline')
    check(result['seconds_before_terminal_export']<900,'finite deployment cost')
    if result['status']=='PASS':
        check(len(phases)==len(frozen['commands']),'complete deployment phases')
        for receipt in phases:
            check(receipt['status']=='COMPLETED' and receipt['returncode']==0
                  and receipt['execution_error'] is None and receipt['cleanup_error'] is None
                  and receipt['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION','accepted exit/cleanup')
        probe_record=load(RUN/'probe.json')
        check(probe_record['status']=='PASS' and probe_record['prefix']==str(NEW)
              and probe_record['torch']=='2.11.0+cpu' and probe_record['cuda_build'] is None,
              'saved CPU import conformance')
        installed={name.lower().replace('_','-'):version for name,version in probe_record['packages'].items()}
        check(not any(name.startswith('nvidia-') for name in installed),'no installed CUDA runtime packages')
        packages={}
        for name in ('packages.json','converter.json'):
            report=load(RUN/name,limit=64*2**20)
            check(report['environment']['python_version']=='3.11','Python family')
            for item in report['install']:
                meta=item['metadata']; key=meta['name'].lower().replace('_','-')
                check(not key.startswith('nvidia-'),'no CUDA runtime packages')
                check(key not in packages,'non-replacing dependency install')
                packages[key]=meta['version']
                check(bool(item['download_info']),'recorded dependency source')
        for line in (ROOT/'configs/genbab_cpu_20261021_requirements.txt').read_text().splitlines():
            if not line or line.startswith('#'): continue
            key,version=line.split('==')
            check(installed.get(key.lower().replace('_','-'))==version,'fixed direct dependency: '+key)
        converter=load(RUN/'converter.json')['install']
        check(len(converter)==1 and converter[0]['download_info']['vcs_info']['commit_id']==
              '0826948e0d50cd09d909b7f26b2ee9ca2d3d71b3','converter source')
        check(result['environment_bytes']+result['build_bytes']<=frozen['deployment_bytes'],
              'saved deployment storage')
    else:
        check(result['status']=='FAIL' and bool(result.get('error')),'explicit failed deployment')
    check(size(RUN)<=128*2**20,'complete evidence storage')
    print(json.dumps(dict(audit='PASS',execution_status=result['status'],phase_count=len(phases),
                          result_sha256=sha(RUN/'result.json'),native_verifier_calls=0)),flush=True)
    return 0


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--audit-saved': raise SystemExit(audit_saved(sys.argv[2]))
    if len(sys.argv)==2 and sys.argv[1]=='--probe': raise SystemExit(probe())
    if len(sys.argv)==5 and sys.argv[1]=='--phase':
        raise SystemExit(phase(sys.argv[2],int(sys.argv[3]),float(sys.argv[4])))
    if len(sys.argv)==3 and sys.argv[1]=='--execute-approved-once': raise SystemExit(main(sys.argv[2]))
    raise SystemExit('explicit once-only CPU deployment required')
