"""Version-bound analytic/regression receipt for the optional handoff revision."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def run():
    from evidence_cohort.contract import verify_freeze
    verify_freeze()
    target = ROOT/'docs/evidence_handoff_v1_controls.json'
    if target.exists():
        raise FileExistsError('controls receipt immutable')
    sources = sorted(Path(__file__).parent.glob('*.py'))+[ROOT/'docs/evidence_handoff_v1.md']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'evidence_handoff.tests', 'scripts.test_general_evidence', 'cohort_analysis.tests', 'evidence_cohort.tests'])
    def names(s):
        for t in s:
            if isinstance(t, unittest.TestSuite): yield from names(t)
            else: yield t.id()
    tests = list(names(suite)); log = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    verify_freeze()
    ok = result.wasSuccessful() and not result.skipped and len(tests) == result.testsRun
    record = {'schema': 'EVIDENCE_HANDOFF_CONTROLS_V1', 'status': 'PASS' if ok else 'FAIL',
              'sources': hashes, 'tests': tests, 'tests_run': result.testsRun,
              'failures': [(t.id(), s) for t, s in result.failures],
              'errors': [(t.id(), s) for t, s in result.errors], 'skipped': len(result.skipped),
              'seconds': time.monotonic()-started, 'log': log.getvalue(),
              'frozen_v1_sources_verified_before_and_after': True,
              'new_real_model_requests': 0, 'cohort_launch_registered': False,
              'production_acceptance_changed': False}
    with target.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')
    print(log.getvalue()); print(json.dumps({k: record[k] for k in ('status', 'tests_run', 'seconds')}))
    if not ok: raise SystemExit(1)


if __name__ == '__main__':
    run()
