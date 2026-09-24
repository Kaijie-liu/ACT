"""Record controls and freeze ONLY synthetic comparison before measurement."""
import argparse
import io
from pathlib import Path
import time
import unittest

from scoped_proof.io import ROOT, save, sha, load
from checked_route_frontier.study import OUTPUT, CONFIG
from checked_route_frontier.fixtures import timing
from source_enclosure.format import identity

REPORT = ROOT/'docs/checked_route_frontier_controls_20260924_r1.json'
MODULES = ('checked_route_frontier.tests', 'checked_route_frontier.supervision_tests',
           'scoped_source.tests', 'source_enclosure.tests', 'source_enclosure.portable_tests',
           'full_source.tests', 'scoped_proof.tests')
DIRECTORIES = ('checked_route_frontier', 'scoped_source', 'scoped_proof', 'source_enclosure',
               'full_source', 'upstream_source', 'router_source', 'source_construction_lab')


def inventory():
    paths = [p for name in DIRECTORIES for p in (ROOT/name).glob('*.py')]
    paths += [ROOT/'docs/checked_route_frontier_protocol_20260924_r1.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def run():
    if REPORT.exists() or CONFIG.exists():
        raise FileExistsError('immutable recorded control/freeze already exists')
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    log = io.StringIO()
    start = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print(log.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise RuntimeError('control failures; do not freeze')
    report = {'schema': 'CHECKED_ROUTE_FRONTIER_CONTROLS_V1', 'status': 'PASS',
              'tests_run': result.testsRun, 'seconds': time.monotonic()-start,
              'modules': list(MODULES), 'sources': inventory(), 'log': log.getvalue(),
              'real_requests': 0, 'native_solver_calls_in_new_controls': 0,
              'note': 'Synthetic positives only; unchanged regression controls may use synthetic native LPs.'}
    saved = save(REPORT, report)
    calls = []
    for fixture in ('prunable', 'tied'):
        for repeat in range(3):
            order = ('exhaustive', 'frontier') if repeat % 2 == 0 else ('frontier', 'exhaustive')
            for mode in order:
                calls.append({'id': f'{fixture}_{repeat}_{mode}', 'fixture': fixture,
                              'repeat': repeat, 'mode': mode})
    save(CONFIG, {'schema': 'CHECKED_ROUTE_FRONTIER_SYNTHETIC_V1', 'sources': inventory(),
        'controls': {'path': str(REPORT.relative_to(ROOT)), 'sha256': saved['sha256']},
        'output': str(OUTPUT), 'fixture_sha256': {key: identity(timing(key)) for key in ('prunable', 'tied')},
        'budget_seconds': 30, 'cpu_threads': 2,
        'sampled_rss_limit': 8*2**30, 'calls': calls, 'real_requests': 0,
        'success': 'complete retained-proof equivalence plus all-pair accounting; timing may be worse',
        'scope': 'fixed analytic fixtures only, not real request execution authorization'})


def verify():
    report = load(REPORT)
    config = load(CONFIG)
    if report['status'] != 'PASS' or config['controls']['sha256'] != sha(REPORT):
        raise ValueError('control gate')
    if report['sources'] != inventory() or config['sources'] != inventory():
        raise ValueError('implementation changed')
    print({'status': 'PASS', 'tests': report['tests_run'], 'real_requests': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    verify() if args.check else run()
