"""Two approved factor faults, <=300s including setup and archival.

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
PROTOCOL='docs/hz_pair_factor_faults_20261019.md'
FREEZE='docs/hz_pair_factor_faults_freeze_20261019.json'
TEST='scripts/test_pair_factor_faults.py'


def hashed(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    old_freeze='docs/hz_pair_scope_lifetime_freeze_20261018.json'
    old=json.loads((REPO/old_freeze).read_bytes())
    names=set(old['files'])
    names.update((old_freeze,PROTOCOL,TEST,'scripts/run_pair_factor_faults.py',
                  'scripts/audit_pair_factor_faults.py',
                  'docs/hz_pair_scope_lifetime_results_20261018.json'))
    c=next(n for n in ast.parse((REPO/TEST).read_bytes()).body
           if isinstance(n,ast.ClassDef) and n.name=='Controls')
    tests=sorted(n.name for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    if tests!=['test_01_factor_missing','test_02_factor_permuted']:
        raise ValueError('exactly two authorized tests')
    if any(hashed(REPO/n)!=h for n,h in old['files'].items()):
        raise ValueError('R1 implementation/protocol unchanged')
    return dict(schema='PAIR_FACTOR_FAULT_FREEZE_V1',
        base_head='bed3d30f4a9d7b6a2c9326c1f2ba261e498441f4',
        revision='FACTOR_FAULT_FOLLOWUP_R1',
        files={n:hashed(REPO/n) for n in sorted(names)},tests=tests,
        source='3a9df8651ea6781098377244b410a1cfa821bc4f431fbf1d067fa5c4a76cb360',
        forest='cfd36bba9a2702c13a4d80e2f093cc4a764f77e85d66a19e12ac51c37d3ef157',
        source_name='eight_private_six_outputs',pair=[0,1],
        expected_errors={'missing':'scope membership binding',
                         'permuted':'canonical traversal rank/order'},
        paired_executions=0,candidate_calls=0,oracle_calls=0,replay_executions=0,
        limits=old['limits'],
        attempt_policy='one batch; retain failure; no automatic retry or added cases',
        models=0,gpu=0,optimizer=0,acceptance_changes=False,
        result_directory='/data1/Kane/MOE/results/pair_factor_faults_20261019_r1')


def save(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()
    if len(raw)>8*2**20: raise ValueError('bounded control report')
    with path.open('xb') as out: out.write(raw)


def decode_worker_report(raw):
    if type(raw) is not bytes or len(raw)>8*2**20: raise ValueError('bounded worker report')
    value=json.loads(raw)
    if type(value) is not dict or value.get('status') not in ('PASS','FAIL'):
        raise ValueError('worker report status')
    return value


def worker(root,deadline):
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(128*2**20,128*2**20))
    if deadline-time.monotonic()>300: raise ValueError('one original deadline')
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('required isolated stdlib interpreter')
    def expired(*_): raise TimeoutError('control batch deadline')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    sys.path.insert(0,str(REPO))
    from scripts import test_pair_factor_faults as controls
    controls.ROOT=root/'sources'; controls.ROOT.mkdir()
    controls.DEADLINE=deadline
    frozen=json.loads((root/'freeze.json').read_bytes())
    suite=unittest.TestSuite(controls.Controls(name) for name in frozen['tests'])
    log=io.StringIO(); result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    bad=sorted({'torch','numpy','scipy','highspy'}&set(sys.modules))
    report=dict(schema='PAIR_FACTOR_FAULT_RESULT_V1',
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
    if root!=Path('/data1/Kane/MOE/results/pair_factor_faults_20261019_r1') or root.exists():
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
    logpath=root/'process.log'; timed_out=False; returncode=None; resource_stop=False; peak_group_rss=0
    with logpath.open('xb') as log:
        process=subprocess.Popen([PYTHON,'-I','-S','-B',str(Path(__file__).resolve()),
            '--worker',str(root),repr(deadline)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
        while process.poll() is None:
            rss=0
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit(): continue
                try:
                    fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
                    if int(fields[2])==process.pid:
                        rss+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
                except (FileNotFoundError,ProcessLookupError,PermissionError): continue
            peak_group_rss=max(peak_group_rss,rss)
            size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            resource_stop=rss>2*2**30 or size>128*2**20-8*2**20
            timed_out=time.monotonic()>=deadline
            if resource_stop or timed_out:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass  # Natural exit raced with this owned-group stop.
                process.wait(); break
            try: process.wait(timeout=min(.1,max(.001,deadline-time.monotonic())))
            except subprocess.TimeoutExpired: pass
        returncode=process.returncode
    resultpath=root/'worker.json'
    worker_result=None; report_error=None
    if resultpath.exists():
        try: worker_result=decode_worker_report(resultpath.read_bytes())
        except (OSError,ValueError,UnicodeError) as exc:
            report_error=dict(type=type(exc).__name__,message=str(exc)[:256])
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashed(p))
           for p in sorted(root.rglob('*')) if p.is_file()}
    bytecount=sum(v['bytes'] for v in files.values())
    complete=(not resource_stop and not timed_out and returncode==0 and worker_result is not None
              and worker_result['status']=='PASS' and time.monotonic()<deadline
              and bytecount<134217728-8*2**20)
    terminal=dict(schema='PAIR_FACTOR_FAULT_TERMINAL_V1',execution_head=head,
        status='PASS' if complete else 'RESOURCE_LIMIT' if resource_stop else 'TIMEOUT' if timed_out else 'FAIL',
        returncode=returncode,timed_out=timed_out,resource_stop=resource_stop,
        report_error=report_error,
        peak_group_rss_bytes=peak_group_rss,wall_seconds=time.monotonic()-START,
        files=files,artifact_bytes_before_terminal=bytecount,
        classification='two factor fault controls; capture/build/check/cleanup/archive included; no performance or proof claim')
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
