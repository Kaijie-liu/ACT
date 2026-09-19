"""Numbered supervised analytic receipts; no real LP diagnostic calls."""
import io
import sys
import time
import unittest
from single_check_portable.execution import ROOT,save_new
from lp_diagnostic.study import verify_old,hashes

def run():
    verify_old();before=hashes();n=1
    while (ROOT/f'docs/lp_diagnostic_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'lp_diagnostic.tests','lp_sandwich.tests','nonpositive_analysis.tests','source_cache_archive.tests'])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    verify_old()
    if hashes()!=before:raise ValueError('source drift during controls')
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    receipt={'status':status,'sources':before,'tests_run':result.testsRun,'log':log.getvalue(),
        'seconds':time.monotonic()-begin,'real_request_solver_calls':0,'old_freezes_unchanged':True,
        'analytic_observations':getattr(sys.modules.get('lp_diagnostic.tests'),'OBSERVATIONS',[]),
        'scope':'analytic supervision/control tests only; not real request results or performance evidence'}
    path=ROOT/f'docs/lp_diagnostic_controls_attempt{n:03}.json';save_new(path,receipt)
    print(log.getvalue());print(path,status)
    if status!='PASS':raise SystemExit(1)

if __name__=='__main__':run()
