"""Numbered immutable control receipts; no real verification calls."""
import io
import json
import time
import unittest
from upstream_portable.study import ROOT,hashes,verify_old
from single_check_portable.execution import save_new


def run():
    verify_old();before=hashes();number=1
    while (ROOT/f'docs/upstream_portable_v1_controls_attempt{number:03}.json').exists():number+=1
    out=ROOT/f'docs/upstream_portable_v1_controls_attempt{number:03}.json'
    modules=['upstream_portable.tests','single_check_portable.tests','cached_portable.tests',
             'exact_matrix_cache.tests','scripts.test_general_evidence','evidence_handoff.tests',
             'evidence_cohort.tests','cohort_analysis.tests']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    from upstream_portable.tests import OBSERVATIONS
    OBSERVATIONS.clear()
    count=suite.countTestCases();log=io.StringIO();begin=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    verify_old()
    if before!=hashes():raise ValueError('source drift during controls')
    ok=result.wasSuccessful() and not result.skipped and result.testsRun==count
    receipt={'status':'PASS' if ok else 'FAIL','sources':before,'tests_run':result.testsRun,
             'errors':[(t.id(),s) for t,s in result.errors],'failures':[(t.id(),s) for t,s in result.failures],
             'skipped':len(result.skipped),'seconds':time.monotonic()-begin,'log':log.getvalue(),
             'analytic_full_flow':OBSERVATIONS,'real_verification_calls':0,'old_freezes_unchanged':True}
    save_new(out,receipt);print(log.getvalue());print(json.dumps({'receipt':str(out),'status':receipt['status'],'tests':count}))
    if not ok:raise SystemExit(1)


if __name__=='__main__':run()
