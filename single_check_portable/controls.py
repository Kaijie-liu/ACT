"""V3 gate: preserve attempts and all frozen V2/earlier source identities."""
import hashlib
import io
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'docs/cached_portable_v2_controls_attempt001.json'


def verify_old():
    from cached_portable.controls import verify_old as earlier
    earlier()
    for name, sha in json.loads(OLD.read_bytes())['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
            raise ValueError('V2 source changed: '+name)


def hashes():
    names = list(json.loads(OLD.read_bytes())['sources'])
    names += [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
    names += ['docs/single_check_portable_v3.md', str(OLD.relative_to(ROOT))]
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(set(names))}


def run():
    verify_old(); before = hashes(); number = 1
    while (ROOT/f'docs/single_check_v3_controls_attempt{number:03}.json').exists(): number += 1
    out = ROOT/f'docs/single_check_v3_controls_attempt{number:03}.json'
    modules = ['single_check_portable.tests', 'cached_portable.tests', 'exact_matrix_cache.tests',
               'scripts.test_general_evidence', 'evidence_handoff.tests', 'evidence_cohort.tests', 'cohort_analysis.tests']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    from single_check_portable.tests import OBSERVATIONS
    OBSERVATIONS.clear()
    def names(s):
        for t in s:
            if isinstance(t, unittest.TestSuite): yield from names(t)
            else: yield t.id()
    tests = list(names(suite)); log = io.StringIO(); start = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    verify_old()
    if before != hashes(): raise ValueError('source drift during tests')
    ok = result.wasSuccessful() and not result.skipped and result.testsRun == len(tests)
    record = {'schema': 'SINGLE_CHECK_V3_CONTROLS', 'status': 'PASS' if ok else 'FAIL',
              'sources': before, 'tests_run': result.testsRun, 'tests': tests,
              'failures': [(t.id(), s) for t, s in result.failures], 'errors': [(t.id(), s) for t, s in result.errors],
              'skipped': len(result.skipped), 'seconds': time.monotonic()-start, 'log': log.getvalue(),
              'analytic_observations': OBSERVATIONS, 'old_freezes_verified_before_and_after': True,
              'new_real_requests': 0, 'original_results_changed': False}
    with out.open('x') as f: json.dump(record, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(log.getvalue()); print(json.dumps({k: record[k] for k in ('status', 'tests_run', 'seconds')}))
    if not ok: raise SystemExit(1)


if __name__ == '__main__': run()
