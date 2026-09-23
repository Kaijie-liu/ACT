"""Finite regression gate; does not run/freeze real model requests."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import metamoe_receipt_reserve as control
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

PATTERNS=('test_receipt_reserve.py','test_receipt_reserve_protocol.py',
          'test_protected_hz_solver.py','test_metamoe_las_supervision.py',
          'test_metamoe_checked_paired.py','test_checked_base_session.py')


def main():
    if control.GATE.exists():raise FileExistsError(control.GATE)
    if control.OUTPUT.exists():raise FileExistsError('real output must remain absent')
    began=time.monotonic();rows=[]
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
        'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
    for pattern in PATTERNS:
        if not (control.ROOT/'tests'/pattern).is_file():raise FileNotFoundError(pattern)
        cmd=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v']
        if pattern=='test_metamoe_checked_paired.py':
            # The old frozen-smoke hash test intentionally binds the OLD native
            # worker. Re-run its composition controls, not its old live freeze.
            cmd=[sys.executable,'-m','unittest','tests.test_metamoe_checked_paired.CompositionControls','-v']
        start=time.monotonic()
        r=subprocess.run(cmd,cwd=control.ROOT,env=env,capture_output=True,text=True,timeout=120)
        output=r.stdout+r.stderr
        count=re.search(r'Ran (\d+) tests?',output)
        rows.append({'pattern':pattern,'command':cmd,'returncode':r.returncode,
            'tests':int(count.group(1)) if count else 0,'seconds':time.monotonic()-start,'output':output})
        print(pattern,r.returncode,rows[-1]['tests'],flush=True)
    passed=all(r['returncode']==0 and r['tests']>0 for r in rows)
    result={'status':'PASS' if passed else 'FAIL','controls_passed':passed,
        'real_requests_executed':0,'tests':sum(r['tests'] for r in rows),'rows':rows,
        'source_sha256':{n:sha256(control.ROOT/n) for n in control.SOURCES},
        'regression_sha256':{str(Path('tests')/p):sha256(control.ROOT/'tests'/p) for p in PATTERNS},
        'seconds':time.monotonic()-began,'trust':'control gate, not empirical performance or source proof'}
    write(control.GATE,result)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
