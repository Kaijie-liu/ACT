"""Numbered analytic controls with old source freeze validation."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT,save_new
from portable_proof.runtime import digest
from lp_diagnostic.study import verify

def sources():
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}

def run():
    verify();before=sources();n=1
    while (ROOT/f'docs/exact_primal_controls_attempt{n:03}.json').exists():n+=1
    begin=time.monotonic();log=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['exact_primal.tests','lp_sandwich.tests','lp_diagnostic_archive.tests'])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);verify()
    if sources()!=before:raise ValueError('source drift')
    from exact_primal.tests import OBSERVATIONS
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    receipt={'status':status,'sources':before,'log':log.getvalue(),'tests_run':result.testsRun,
        'seconds':time.monotonic()-begin,'analytic_observations':OBSERVATIONS,
        'real_LP_reconstructions':0,'real_solver_calls':0,'old_freezes_unchanged':True,
        'scope':'bounded analytic candidate construction, not native-basis recovery or production integration'}
    dest=ROOT/f'docs/exact_primal_controls_attempt{n:03}.json';save_new(dest,receipt)
    print(log.getvalue());print(dest,status)
    if status!='PASS':raise SystemExit(1)

if __name__=='__main__':run()
