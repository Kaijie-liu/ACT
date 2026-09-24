"""Record synthetic controls and freeze finite timing; never executes timing."""
import io
import json
from pathlib import Path
import time
import unittest
from scoped_proof.io import ROOT, load, save, sha
from source_construction_lab.study import CONFIG, OUTPUT

RECORD = ROOT/'docs/source_construction_parse_controls_20260924_r1.json'
PROTOCOL = ROOT/'docs/source_construction_parse_protocol_20260924_r1.md'


def sources():
    old = load(ROOT/'configs/backend_controls/scoped_proof_execution_r1.json')['sources']
    for name, digest in old.items():
        if sha(ROOT/name) != digest: raise ValueError('sealed implementation changed: '+name)
    paths = [PROTOCOL, *sorted((ROOT/'source_construction_lab').glob('*.py'))]
    return {**old, **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


if __name__ == '__main__':
    if RECORD.exists() or CONFIG.exists() or OUTPUT.exists():
        raise FileExistsError('new control/freeze artifacts required')
    stream = io.StringIO(); started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromName('source_construction_lab.tests')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = {'status': 'PASS' if result.wasSuccessful() and result.testsRun == 17 else 'FAIL',
        'tests_run': result.testsRun, 'output': stream.getvalue(), 'seconds': time.monotonic()-started,
        'sources': sources(), 'real_requests': 0, 'solver_calls': 0,
        'scope': 'synthetic exact construction/CSR/isolation/supervision; no positive network claim'}
    save(RECORD, record)
    if record['status'] != 'PASS': raise SystemExit('controls failed; no freeze')
    fixtures = {f'w{w}': dict(experts=4, classes=4, width=w, depth=2, seed=724) for w in (16, 32)}
    order = [('reference','uncached','cached'), ('uncached','cached','reference'), ('cached','reference','uncached')]
    calls = [{'id': f'{key}_r{r}_{mode}', 'fixture': key, 'round': r, 'mode': mode}
        for key in fixtures for r, modes in enumerate(order) for mode in modes]
    cfg = {'schema': 'SYNTHETIC_SOURCE_PARSE_TIMING_V1', 'output': str(OUTPUT),
        'sources': record['sources'], 'control_sha256': sha(RECORD), 'fixtures': fixtures, 'calls': calls,
        'budget_seconds': 30, 'cpu_threads': 2, 'sampled_rss_bytes': 8*2**30,
        'publication_reserve_seconds': 2, 'real_requests': 0, 'solver_calls': 0,
        'acceptance': 'identical full source/construction hashes and original independent checker result; NOT SAFE',
        'no_retry_tuning_or_expansion': True, 'checker_cache': False,
        'cost': 'all generation/construction/serialization/independent check/import/cleanup; final ledger and audit disclosed separately'}
    save(CONFIG, cfg)
    print(json.dumps({'controls': record['tests_run'], 'seconds': record['seconds'], 'timing_calls_frozen': len(calls), 'timing_executed': 0}))
