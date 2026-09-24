"""Stage gate, including existing unchanged source/route regression tests."""
import time
import unittest
from scoped_proof.io import ROOT, save


if __name__=='__main__':
    suites=['shared_route_residual.tests','shared_route_residual.supervision_tests',
            'checked_route_frontier.tests','checked_route_frontier.supervision_tests']
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],
        'seconds':time.monotonic()-start,'suites':suites,'real_requests':0,'native_solver_calls':0}
    save(ROOT/'docs/shared_route_residual_controls_20260924_r1.json',report)
    if not result.wasSuccessful():raise SystemExit(1)
