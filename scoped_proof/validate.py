"""Capture control results and tested identities before a separate freeze."""
import argparse
import json
import re
import subprocess
import sys
import time
from scoped_proof.io import ROOT, PYTHON, save, load
from scoped_proof.run import GATE, sources


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check',action='store_true'); a = p.parse_args()
    if a.check:
        gate = load(GATE)
        if gate['status'] != 'PASS' or gate['source_hashes'] != sources() or gate['tests_passed'] != 53:
            raise ValueError('control gate drift')
        print('PASS: tested control/source identities; no real request'); return
    if GATE.exists(): raise FileExistsError(GATE)
    command = [PYTHON,'-m','unittest','scoped_proof.tests','scoped_source.tests',
        'source_enclosure.tests','source_enclosure.portable_tests','full_source.tests','-v']
    start = time.monotonic(); result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
    output = result.stdout+result.stderr; count = re.search(r'Ran (\d+) tests',output)
    passed = int(count.group(1)) if count and result.returncode == 0 else 0
    gate = {'status':'PASS' if passed == 53 else 'FAIL','tests_passed':passed,
        'new_proof_supervision_controls':18,'source_regressions':35,'returncode':result.returncode,
        'command':command,'output':output,'seconds':time.monotonic()-start,'source_hashes':sources(),
        'real_requests_executed':0,'real_checkpoint_or_input_loaded':False,
        'scope':'synthetic checkpoints/LPs only; one-budget supervised exact evidence checks',
        'real_positive_certificates':0,'real_execution_authorized':False}
    save(GATE,gate)
    print(json.dumps({k:gate[k] for k in ('status','tests_passed','seconds','real_requests_executed')}))
    if gate['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
