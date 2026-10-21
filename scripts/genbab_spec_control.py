"""One frozen stdlib B2 specification batch on the existing owned supervisor."""
import time
ENTRY = time.monotonic()
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

ROOT=Path('/data1/Kane/MOE/ACT')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from margin_native_control import limits, successful, ENV, PYTHON, POLICY, CAPS
from scoped_proof.io import load, save, sha
from scoped_source.hz_request_store import require

FREEZE=ROOT/'docs/genbab_spec_freeze_20261021.json'
RUN=Path('/data1/Kane/MOE/results/genbab_spec_20261021_r1')


def binding(anchor):
    value=load(FREEZE,anchor,2**20)
    require(value['policy']==POLICY and value['caps']==CAPS and value['attempts']==1
            and value['run']==str(RUN) and value['native_calls']==0,'frozen stdlib control')
    for path,digest in value['files'].items():
        require(sha(ROOT/path)==digest,'frozen file: '+path)
    import genbab_spec_cases
    require(value['tests']==list(genbab_spec_cases.TESTS),'fixed test roster')
    return value


def bounded_save(path,value):
    size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    raw=json.dumps(value,allow_nan=False).encode()
    require(len(raw)<=2**20 and 2*(size+2*len(raw)+2**20)<=CAPS['artifact_bytes'],
            'cumulative artifacts plus archive reserve')
    return save(path,value)


def worker(anchor,deadline):
    limits(deadline)
    binding(anchor)
    import genbab_spec_cases
    def prefix(records):
        bounded_save(RUN/('prefix_%02d.json'%len(records)),records)
    result=genbab_spec_cases.run(deadline,prefix)
    result.update(deadline=deadline,freeze_sha256=anchor,
                  rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))
    bounded_save(RUN/'worker.json',result)
    return 0 if result['status']=='PASS' else 1


def main(anchor):
    deadline=ENTRY+300
    limits(deadline)
    from scoped_proof.owned_session import execute
    result=dict(status='FAIL',schema='GENBAB_SPEC_CONTROL_RESULT_V1',freeze_sha256=anchor,
                policy=POLICY,caps=CAPS,entry=ENTRY,deadline=deadline,native_calls=0,
                real_stage_calls=0,full_cold_start_seconds=None)
    created=False
    try:
        require(sys.executable==PYTHON,'existing ACT environment')
        frozen=binding(anchor)
        def git(*args):
            return subprocess.check_output(['git',*args],cwd=ROOT,text=True,timeout=5).strip()
        require(git('branch','--show-current')=='feat/moe-route-verification'
                and not git('status','--porcelain'),'clean frozen branch')
        result['execution_head']=git('rev-parse','HEAD')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                       if s.startswith('MemAvailable:'))
        disk=shutil.disk_usage(RUN.parent).free
        require(time.monotonic()<ENTRY+60 and available>=4*2**30 and disk>=CAPS['artifact_bytes'],
                'one bounded admission')
        result['admission']=dict(available_bytes=available,disk_bytes=disk,seconds=time.monotonic()-ENTRY)
        RUN.mkdir(); created=True
        before=len(os.listdir('/proc/self/fd'))
        receipt=execute([PYTHON,'-I','-S','-B',__file__,'--worker',anchor,str(ENTRY+260)],
                        run_deadline=ENTRY+265,cleanup_deadline=ENTRY+280,env=ENV,
                        rss_limit=None,development_policy=POLICY)
        result['session']=receipt
        bounded_save(RUN/'session.json',receipt)
        require(successful(receipt),'stdlib specification session failed')
        worker_hash=sha(RUN/'worker.json')
        value=load(RUN/'worker.json',worker_hash,2**20)
        require(value['status']=='PASS' and value['freeze_sha256']==anchor
                and value['deadline']==ENTRY+260
                and value['rlimit_as']==[CAPS['as_bytes']]*2,'worker identity/deadline/AS')
        require([row['name'] for row in value['records']]==frozen['tests']
                and all(row['passed'] and not row['errors'] for row in value['records']),
                'complete fixed test reception')
        require(before==len(os.listdir('/proc/self/fd')),'controller FD balance')
        result.update(status='PASS',worker_sha256=worker_hash,tests=len(value['records']),
                      fd_before=before,fd_after=len(os.listdir('/proc/self/fd')))
    except BaseException as exc:
        result['error']=repr(exc)
    if created:
        result['seconds_before_export']=time.monotonic()-ENTRY
        bounded_save(RUN/'result.json',result)
        require(time.monotonic()<deadline,'late control result rejected')
        print(json.dumps(dict(status=result['status'],through_export_seconds=time.monotonic()-ENTRY,
                              result_sha256=sha(RUN/'result.json'))),flush=True)
    else:
        print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        raise SystemExit(worker(sys.argv[2],float(sys.argv[3])))
    if len(sys.argv)==3 and sys.argv[1]=='--execute-approved-once':
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit('explicit frozen once-only stdlib control required')
