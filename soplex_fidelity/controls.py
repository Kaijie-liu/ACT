"""Retain large-import controls and original checker regressions, no real solve."""
import argparse
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from soplex_fidelity import tests
from soplex_fidelity.io import save,sha
from soplex_fidelity.run import ROOT,READER


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
    if a.receipt.exists():raise FileExistsError(a.receipt)
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
        unittest.defaultTestLoader.loadTestsFromName('soplex_fidelity.review_tests'),
        unittest.defaultTestLoader.loadTestsFromName('lp_sandwich.tests')])
    from scipy.optimize import linprog
    start=time.monotonic();log=io.StringIO()
    with patch('scipy.optimize.linprog',wraps=linprog) as calls:
        r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        legacy_calls=calls.call_count
    (tests.RAW/'controls.log').write_text(log.getvalue())
    v=dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),
        errors=len(r.errors),skips=len(r.skipped),seconds=time.monotonic()-start,raw_root=str(tests.RAW),
        reader_sha256=sha(READER),soplex_optimization_calls=0,real_optimization_calls=0,
        legacy_analytic_solver_calls=legacy_calls,
        sources={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*') if f.is_file()},
        artifacts={str(f.relative_to(tests.RAW)):sha(f) for f in tests.RAW.rglob('*') if f.is_file()})
    save(a.receipt,v);print(log.getvalue());print(json.dumps({k:v[k] for k in ('status','tests','errors','failures')}))
    return 0 if r.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
