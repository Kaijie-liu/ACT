"""Native analytic captures plus prior regression suite; no real LP queries."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from exact_basis.controls import old as prior_old,sources as basis_sources

def sources():return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}
def old():
    prior_old()
    if read(ROOT/'docs/exact_basis_controls_attempt001.json')['sources']!=basis_sources():raise ValueError('basis source drift')

def run():
    old();before=sources();n=1
    while (ROOT/f'docs/native_basis_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'native_basis.tests','exact_basis.tests','exact_primal.tests','lp_sandwich.tests','lp_diagnostic_archive.tests']))
    old()
    if before!=sources():raise ValueError('source drift')
    from native_basis.adapter import CAPTURES
    from native_basis.tests import OBSERVATIONS
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    receipt={'status':status,'sources':before,'tests_run':result.testsRun,'log':log.getvalue(),
        'seconds':time.monotonic()-begin,'native_analytic_captures':CAPTURES,'analytic_observations':OBSERVATIONS,
        'real_LP_reconstructions':0,'real_solver_calls':0,'old_freezes_unchanged':True,
        'scope':'small native-basis API controls only; no real-request supervision or timing claim'}
    dest=ROOT/f'docs/native_basis_controls_attempt{n:03}.json';save_new(dest,receipt)
    print(log.getvalue());print(dest,status)
    if status!='PASS':raise SystemExit(1)

if __name__=='__main__':run()
