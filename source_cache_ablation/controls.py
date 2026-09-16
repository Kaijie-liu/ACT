"""Numbered controls for full flow; do not overwrite earlier receipts."""
import io
import time
import unittest
import sys
from source_cache_ablation.study import ROOT,hashes,verify_old
from single_check_portable.execution import save_new


def run():
    verify_old();before=hashes();number=1
    while (ROOT/f'docs/source_cache_ablation_controls_attempt{number:03}.json').exists():number+=1
    modules=['source_cache_ablation.tests','nonpositive_analysis.tests','reuse_supervised.tests','upstream_reuse.tests','upstream_portable.tests',
             'single_check_portable.tests','cached_portable.tests','exact_matrix_cache.tests',
             'scripts.test_general_evidence','evidence_handoff.tests','evidence_cohort.tests','cohort_analysis.tests']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules);log=io.StringIO();begin=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    verify_old()
    if before!=hashes():raise ValueError('sources changed during controls')
    module=sys.modules.get('source_cache_ablation.tests')
    observations=getattr(module,'OBSERVATIONS',[])
    value={'status':'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
           'tests_run':result.testsRun,'sources':before,'seconds':time.monotonic()-begin,
           'log':log.getvalue(),'analytic_full_flow':observations,'real_verification_calls':0,
           'old_freezes_unchanged':True,'performance_claim':False}
    path=ROOT/f'docs/source_cache_ablation_controls_attempt{number:03}.json'
    save_new(path,value);print(log.getvalue());print(path,value['status'])
    if value['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
