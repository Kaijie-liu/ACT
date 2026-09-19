"""Numbered analytic controls; old freezes remain untouched."""
import io
from pathlib import Path
import sys
import time
import unittest
from portable_proof.runtime import digest
from single_check_portable.execution import ROOT,save_new
from source_cache_ablation.study import verify


def sources():
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}


def run():
    verify();before=sources();n=1
    while (ROOT/f'docs/lp_sandwich_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['lp_sandwich.tests','nonpositive_analysis.tests','source_cache_archive.tests'])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    verify()
    if sources()!=before:raise ValueError('source drift during controls')
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    receipt={'status':status,'sources':before,'tests_run':result.testsRun,'log':log.getvalue(),
        'seconds':time.monotonic()-begin,'real_request_solver_calls':0,'old_freezes_unchanged':True,
        'analytic_observations':getattr(sys.modules.get('lp_sandwich.tests'),'OBSERVATIONS',[]),
        'scope':'analytic component controls only; no real request, performance or complete-network proof claim'}
    path=ROOT/f'docs/lp_sandwich_controls_attempt{n:03}.json';save_new(path,receipt)
    print(log.getvalue());print(path,status)
    if status!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
