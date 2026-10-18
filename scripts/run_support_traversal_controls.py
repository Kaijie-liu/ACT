"""One frozen stdlib control batch, <=300s including setup and archival.

No native bridge, model, GPU, dependency installation or policy change.
Run with the existing act-py312 interpreter using -I -S -B.
"""
import time
START=time.monotonic()
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
PYTHON='/data1/Kane/miniconda3/envs/act-py312/bin/python'
PROTOCOL='docs/hz_support_traversal_controls_20261018.md'
REVISION='docs/hz_support_traversal_controls_20261018_r2.md'
FREEZE='docs/hz_support_traversal_freeze_20261018_r2.json'
TEST='scripts/test_support_traversal.py'


def hashed(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    names={str(p.relative_to(REPO)) for p in (REPO/'scoped_source').glob('*.py')}
    names.update((PROTOCOL,REVISION,TEST,'scripts/run_support_traversal_controls.py',
        'scripts/audit_support_traversal_controls.py',
        'scripts/test_bounded_runtime.py','scripts/run_store_device_controls.py',
        'scripts/hz_store_supervised.py'))
    c=next(n for n in ast.parse((REPO/TEST).read_bytes()).body
           if isinstance(n,ast.ClassDef) and n.name=='Controls')
    tests=sorted(n.name for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    return dict(schema='SCOPED_TRAVERSAL_FINITE_FREEZE_V1',base_head='b7d4187f03b7f564c6ed015ebc340871128dd826',
        revision='R2_DIRECTORY_PREPARATION_ONLY',
        parent_freeze_sha256='b20c6777193916ad78326d82abc9b9d10fa073ff47a916d48d2ddd3f05a901c5',
        files={n:hashed(REPO/n) for n in sorted(names)},tests=tests,
        sources=['deep','half','guarded','eight_private','negative_guard'],
        paired_sources=['deep','half','guarded','eight_private'],profiles=[0,1],
        arms=[False,True],paired_executions=16,oracle_references=8,
        limits=dict(batch_seconds=300,request_seconds_max=300,work_per_request=8000000,
                    arena_bytes=134217728,artifact_bytes=134217728,work_bits=8192,stored_bits=4096),
        attempt_policy='one batch; retain failure; no automatic retry or added cases',
        models=0,gpu=0,optimizer=0,acceptance_changes=False)


def save(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()
    if len(raw)>8*2**20: raise ValueError('bounded control report')
    with path.open('xb') as out: out.write(raw)


def worker(root,deadline):
    if deadline-time.monotonic()>300: raise ValueError('one original deadline')
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('required isolated stdlib interpreter')
    def expired(*_): raise TimeoutError('control batch deadline')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    sys.path.insert(0,str(REPO))
    from scripts import test_support_traversal as controls
    controls.ROOT=root/'sources'; controls.ROOT.mkdir()
    controls.DEADLINE=deadline
    frozen=json.loads((root/'freeze.json').read_bytes())
    suite=unittest.TestSuite(controls.Controls(name) for name in frozen['tests'])
    log=io.StringIO(); result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    bad=sorted({'torch','numpy','scipy','highspy'}&set(sys.modules))
    report=dict(schema='SCOPED_TRAVERSAL_FINITE_RESULT_V1',
        status='PASS' if result.wasSuccessful() and not bad else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        log=log.getvalue(),records=controls.RECORDS,unexpected_modules=bad,
        python=sys.version,elapsed_seconds=time.monotonic()-START,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        full_request_performance_claim=False,real_models=0,gpu=0)
    save(root/'worker.json',report)
    if time.monotonic()>=deadline: raise TimeoutError('report after deadline')
    return int(report['status']!='PASS')


def run(root):
    if sys.executable!=PYTHON: raise ValueError('existing act-py312 only')
    if root.parent!=Path('/data1/Kane/MOE/results') or root.exists():
        raise ValueError('fresh isolated results directory')
    frozen=json.loads((REPO/FREEZE).read_bytes())
    if frozen!=manifest(): raise ValueError('frozen code/protocol drift')
    status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
    if status: raise ValueError('clean frozen checkout required')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!='feat/moe-route-verification': raise ValueError('branch')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    changed=set(subprocess.check_output(['git','diff','--name-only',frozen['base_head'],head],
                                       cwd=REPO,text=True).splitlines())
    if not changed <= set(frozen['files'])|{FREEZE,'docs/CODEX_HANDOFF.md'}:
        raise ValueError('unbound change since frozen base (including transitive dependencies)')
    root.mkdir(parents=True); deadline=START+300; save(root/'freeze.json',frozen)
    logpath=root/'process.log'; timed_out=False; returncode=None
    with logpath.open('xb') as log:
        process=subprocess.Popen([PYTHON,'-I','-S','-B',str(Path(__file__).resolve()),
            '--worker',str(root),repr(deadline)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
        try: returncode=process.wait(timeout=max(.001,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True; os.killpg(process.pid,signal.SIGKILL); process.wait()
    resultpath=root/'worker.json'
    worker_result=json.loads(resultpath.read_bytes()) if resultpath.exists() else None
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashed(p))
           for p in sorted(root.rglob('*')) if p.is_file()}
    bytecount=sum(v['bytes'] for v in files.values())
    complete=(not timed_out and returncode==0 and worker_result is not None
              and worker_result['status']=='PASS' and time.monotonic()<deadline
              and bytecount<134217728-8*2**20)
    terminal=dict(schema='SCOPED_TRAVERSAL_CONTROL_TERMINAL_V1',execution_head=head,
        status='PASS' if complete else 'TIMEOUT' if timed_out else 'FAIL',
        returncode=returncode,timed_out=timed_out,wall_seconds=time.monotonic()-START,
        files=files,artifact_bytes_before_terminal=bytecount,
        classification='component controls; fixture/oracle/fault work included in batch, not an end-to-end speed trial')
    save(root/'terminal.json',terminal)
    # Late publication is never a completed success. Record it without relabelling.
    late=time.monotonic()>=deadline
    if late: save(root/'late.json',dict(status='LATE_TERMINAL',elapsed=time.monotonic()-START))
    print(json.dumps(dict(status=terminal['status'],late=late,path=str(root),head=head,
        wall_seconds=time.monotonic()-START,artifact_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))))
    return int(not complete or late)


if __name__=='__main__':
    if sys.argv[1:]==['--manifest']: print(json.dumps(manifest(),sort_keys=True,indent=2))
    elif len(sys.argv)==3 and sys.argv[1]=='--run': sys.exit(run(Path(sys.argv[2])))
    elif len(sys.argv)==4 and sys.argv[1]=='--worker': sys.exit(worker(Path(sys.argv[2]),float(sys.argv[3])))
    else: raise SystemExit('use --manifest or --run FRESH_DIRECTORY')
