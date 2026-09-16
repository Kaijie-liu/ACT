"""Analytic correctness gate, frozen before a bounded saved-checker timing."""
import hashlib
import io
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ('act/back_end/solver/lp_certificate.py', 'act/back_end/solver/sparse_lp_certificate.py',
                'act/back_end/solver/check_hz_lp_export.py', 'act/back_end/solver/check_rational_mccormick.py',
                'moe_evidence/checker.py', 'moe_evidence/schema.py', 'moe_evidence/storage.py',
                'portable_proof/runtime.py')


def run():
    from evidence_cohort.contract import verify_freeze
    verify_freeze(); target = ROOT/'docs/exact_matrix_cache_v1_controls.json'
    if target.exists(): raise FileExistsError('immutable control receipt already exists')
    paths = sorted(Path(__file__).parent.glob('*.py'))+[ROOT/'docs/exact_matrix_cache_v1.md']+[ROOT/n for n in DEPENDENCIES]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    suite = unittest.defaultTestLoader.loadTestsFromNames(['exact_matrix_cache.tests', 'scripts.test_general_evidence',
                                                        'evidence_handoff.tests', 'cohort_analysis.tests'])
    def names(s):
        for t in s:
            if isinstance(t, unittest.TestSuite): yield from names(t)
            else: yield t.id()
    tests = list(names(suite)); log = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    verify_freeze()
    ok = result.wasSuccessful() and not result.skipped and result.testsRun == len(tests)
    record = {'schema': 'EXACT_MATRIX_CACHE_CONTROLS_V1', 'status': 'PASS' if ok else 'FAIL',
              'sources': sources, 'tests_run': result.testsRun, 'tests': tests,
              'failures': [(t.id(), s) for t, s in result.failures], 'errors': [(t.id(), s) for t, s in result.errors],
              'skipped': len(result.skipped), 'seconds': time.monotonic()-started, 'log': log.getvalue(),
              'original_freezes_verified_before_and_after': True, 'new_real_requests': 0,
              'timing_executed': False}
    with target.open('x') as f: json.dump(record, f, sort_keys=True, indent=2); f.write('\n')
    print(log.getvalue()); print(json.dumps({k: record[k] for k in ('status', 'tests_run', 'seconds')}))
    if not ok: raise SystemExit(1)


if __name__ == '__main__': run()
