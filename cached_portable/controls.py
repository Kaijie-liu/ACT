"""Source-bound analytic portability controls; preserve every attempt."""
import hashlib
import io
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def verify_old():
    from evidence_cohort.contract import verify_freeze
    verify_freeze()
    old = json.loads((ROOT/'docs/exact_matrix_cache_v1_controls.json').read_bytes())
    for name, sha in old['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
            raise ValueError('old controlled source changed: '+name)


def hashes():
    names = [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
    names += ['docs/cached_portable_v2.md', 'moe_evidence/bundle.py', 'moe_evidence/bundle_runtime.py',
              'scripts/build_portable_conv_proof.py', 'scripts/optional_evidence_budget.py',
              'evidence_cohort/run.py', 'evidence_cohort/ownership.py',
              'docs/exact_matrix_cache_v1_controls.json']
    old = json.loads((ROOT/'docs/exact_matrix_cache_v1_controls.json').read_bytes())
    names += list(old['sources'])
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(set(names))}


def run():
    verify_old(); before = hashes()
    sequence = 1
    while (ROOT/f'docs/cached_portable_v2_controls_attempt{sequence:03}.json').exists(): sequence += 1
    target = ROOT/f'docs/cached_portable_v2_controls_attempt{sequence:03}.json'
    modules = ['cached_portable.tests', 'exact_matrix_cache.tests', 'scripts.test_general_evidence',
               'evidence_handoff.tests', 'evidence_cohort.tests', 'cohort_analysis.tests']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    from cached_portable.tests import OBSERVATIONS
    OBSERVATIONS.clear()
    def names(s):
        for item in s:
            if isinstance(item, unittest.TestSuite): yield from names(item)
            else: yield item.id()
    tests = list(names(suite)); log = io.StringIO(); start = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    verify_old()
    if before != hashes(): raise ValueError('source drift during tests')
    ok = result.wasSuccessful() and not result.skipped and result.testsRun == len(tests)
    record = {'schema': 'CACHED_PORTABLE_V2_CONTROLS', 'status': 'PASS' if ok else 'FAIL',
              'sources': before, 'tests_run': result.testsRun, 'tests': tests,
              'failures': [(t.id(), s) for t, s in result.failures], 'errors': [(t.id(), s) for t, s in result.errors],
              'skipped': len(result.skipped), 'seconds': time.monotonic()-start, 'log': log.getvalue(),
              'analytic_observations': OBSERVATIONS, 'old_freezes_verified_before_and_after': True,
              'new_real_requests': 0, 'original_results_changed': False}
    with target.open('x') as f: json.dump(record, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(log.getvalue()); print(json.dumps({k: record[k] for k in ('status', 'tests_run', 'seconds')}))
    if not ok: raise SystemExit(1)


if __name__ == '__main__': run()
