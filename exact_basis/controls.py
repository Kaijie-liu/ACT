"""Numbered analytic basis controls; never reopens a real diagnostic."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT,save_new,read
from portable_proof.runtime import digest
from lp_diagnostic.study import verify
from exact_primal.controls import sources as primal_sources

def sources():return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}

def old():
    verify()
    if read(ROOT/'docs/exact_primal_controls_attempt003.json')['sources']!=primal_sources():
        raise ValueError('old primal module drift')

def run():
    old();before=sources();n=1
    while (ROOT/f'docs/exact_basis_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['exact_basis.tests','exact_primal.tests','lp_sandwich.tests','lp_diagnostic_archive.tests'])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);old()
    if sources()!=before:raise ValueError('source drift')
    from exact_basis.tests import OBSERVATIONS
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    record={'status':status,'sources':before,'tests_run':result.testsRun,'log':log.getvalue(),
        'seconds':time.monotonic()-begin,'analytic_observations':OBSERVATIONS,
        'old_freezes_unchanged':True,'real_LP_reconstructions':0,'real_solver_calls':0,
        'scope':'original-coordinate declarative basis controls; no native basis capture or production integration'}
    dest=ROOT/f'docs/exact_basis_controls_attempt{n:03}.json';save_new(dest,record)
    print(log.getvalue());print(dest,status)
    if status!='PASS':raise SystemExit(1)

if __name__=='__main__':run()
