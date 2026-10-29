"""One <=300s control of S1 request execution; no trained model calls."""
import time
ENTRY = time.monotonic()
import io
import json
import os
from pathlib import Path
import signal
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from scoped_proof.io import save, load, sha
from architecture_s1_execution import Artifacts, digest, FILES, invoke, finite_queue, read_record
import genbab_cpu_ready as backend


def run(root):
    root = Path(root); root.mkdir()
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    # Every native case below is <=40s; enclosing batch alarm remains a backstop.
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('control batch 300s')))
    signal.setitimer(signal.ITIMER_REAL, max(.01, 299-(time.monotonic()-ENTRY)))
    files = FILES+('tests/test_architecture_s1_execution.py',
        'scripts/architecture_s1_execution_control.py', 'scripts/multilayer_stage0.py')
    spec = dict(schema='S1_EXECUTION_CONTROLS_V1', synthetic_control=True,
        source_files={p:sha(ROOT/p) for p in files},
        backend_sources_sha256=digest(backend.source_identity()), batch_seconds=300,
        cases=['shared concurrent accounting', 'capacity and immutable evidence',
            'finite full denominators', 'partial/missing properties', 'identity/deadline/replay/cleanup',
            'two concurrent native workers, four analytic requests',
            'native exception', 'native cutoff', 'candidate without publication'], trained_calls=0)
    save(root/'freeze.json', spec)
    import test_architecture_s1_execution as module
    module.CONTROL_ROOT = root
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2, failfast=True).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    save(root/'standard.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, text=output.getvalue()))
    if not result.wasSuccessful():
        save(root/'complete.json', dict(status='FAIL', phase='standard', seconds=time.monotonic()-ENTRY))
        return 1
    store = Artifacts.create(root/'native')
    requests = [dict(id=f'normal_{i}', case=case, mode=mode) for i,(case,mode) in enumerate([
        ('positive','complete'), ('tie','incomplete'), ('wrong','complete'), ('zero','incomplete')])]
    summary = finite_queue(store.root, requests, root/'freeze.json', workers=2,
        seconds=40, timing_reserve=(5, 3, 1))
    expected = ['POSITIVE', 'POSITIVE', 'UNSAFE_REPLAYED', 'UNKNOWN']
    rows = [read_record(store.root/(r['id']+'.ledger.json.gz')) for r in requests]
    if summary['status'] != 'COMPLETE' or [r['status'] for r in rows] != expected:
        save(root/'complete.json', dict(status='FAIL', phase='native_normal', rows=rows,
            seconds=time.monotonic()-ENTRY))
        return 1
    faults = []
    for case, duration, timing, status in [('exception', 10, (2,1,.5), 'ERROR'),
            ('deadline', 3, (1,.6,.2), 'TIMEOUT'), ('partial', 40, (5,3,1), 'ERROR')]:
        row = invoke(store.root, dict(id=case, case=case, mode='incomplete'), root/'freeze.json',
            seconds=duration, timing_reserve=timing)
        faults.append(row)
        if row['status'] != status or row['accepted'] or row['seconds'] >= duration:
            save(root/'complete.json', dict(status='FAIL', phase=case, faults=faults,
                seconds=time.monotonic()-ENTRY))
            return 1
    physical = sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file())
    account = load(store.account)
    if physical > account['used'] or account['used'] > account['cap']:
        raise ValueError('whole-account physical byte reconciliation')
    elapsed = time.monotonic()-ENTRY
    if elapsed >= 300:
        raise TimeoutError('batch complete accounting')
    save(root/'complete.json', dict(status='PASS', standard_tests=result.testsRun,
        native_calls=7, normal=summary, faults=faults, seconds=elapsed,
        physical_bytes=physical, conservative_charged_bytes=account['used'],
        trained_calls=0, source_files=spec['source_files'],
        artifacts={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(status='PASS', standard_tests=result.testsRun, native_calls=7, seconds=elapsed)))
    return 0


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
