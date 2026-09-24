"""Archive synthetic integration and unchanged math/watchdog regressions."""
import argparse
import json
import os
import re
import subprocess
import time
from scoped_proof.io import ROOT, PYTHON, load, save
from frontier_proof.run import GATE, CONFIG, DEST, sources

MODULES=['frontier_proof.tests','checked_route_frontier.tests','checked_route_frontier.supervision_tests',
         'scoped_source.tests','source_enclosure.tests','source_enclosure.portable_tests',
         'full_source.tests','scoped_proof.tests']
EXPECTED_TESTS=94


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        gate=load(GATE)
        if gate['status']!='PASS' or gate['tests_passed']!=EXPECTED_TESTS or gate['source_hashes']!=sources():
            raise ValueError('tested source/control drift')
        print(f'PASS: {EXPECTED_TESTS} controls bound to unchanged sources; zero real requests');return
    if GATE.exists() or CONFIG.exists() or DEST.exists():raise FileExistsError('new control stage required')
    cmd=[PYTHON,'-m','unittest',*MODULES,'-v'];start=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    output=result.stdout+result.stderr;match=re.search(r'Ran (\d+) tests',output)
    passed=int(match.group(1)) if result.returncode==0 and match else 0
    row={'status':'PASS' if passed==EXPECTED_TESTS else 'FAIL','tests_passed':passed,'new_controls':21,
        'unchanged_regressions':73,'command':cmd,'output':output,'returncode':result.returncode,
        'seconds':time.monotonic()-start,'source_hashes':sources(),'real_requests_executed':0,
        'real_model_or_data_loaded':False,'synthetic_native_LPs':True,'real_positive_certificates':0,
        'scope':'real-intake synthetic control; source/route/output/receipt full integration, not real performance evidence'}
    save(GATE,row);print(json.dumps({k:row[k] for k in ('status','tests_passed','seconds','real_requests_executed')}))
    if row['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
