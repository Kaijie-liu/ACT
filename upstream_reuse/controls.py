"""Numbered controls, immutable old-freeze checks; no real-request experiments."""
import io
import time
import unittest
from pathlib import Path
from upstream_portable.study import ROOT, hashes as old_hashes, verify_old
from single_check_portable.execution import save_new
from portable_proof.runtime import digest


def hashes():
    return {**old_hashes(), **{str(p.relative_to(ROOT)):digest(p.read_bytes())
                              for p in (ROOT/'upstream_reuse').glob('*.py')}}


def run():
    verify_old(); before=hashes(); index=1
    while (ROOT/f'docs/upstream_reuse_controls_attempt{index:03}.json').exists():index+=1
    modules=['upstream_reuse.tests','upstream_portable.tests','single_check_portable.tests',
             'cached_portable.tests','exact_matrix_cache.tests','scripts.test_general_evidence',
             'evidence_handoff.tests','evidence_cohort.tests','cohort_analysis.tests']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules);log=io.StringIO();start=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    verify_old()
    if before!=hashes():raise ValueError('sources changed during controls')
    ok=result.wasSuccessful() and not result.skipped
    record={'status':'PASS' if ok else 'FAIL','tests_run':result.testsRun,'sources':before,
            'seconds':time.monotonic()-start,'log':log.getvalue(),'real_requests':0,
            'old_freezes_unchanged':True,'performance_claim':False}
    path=ROOT/f'docs/upstream_reuse_controls_attempt{index:03}.json'
    save_new(path,record);print(log.getvalue());print(path,record['status'])
    if not ok:raise SystemExit(1)


if __name__=='__main__':run()
