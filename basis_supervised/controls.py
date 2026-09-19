"""Numbered controls, retained failures, frozen-source verification; no real selection."""
import io
import time
import unittest
from single_check_portable.execution import ROOT, read, save_new
from native_basis.controls import old as prior_old, sources as native_sources
from basis_supervised.flow import sources


def old():
    prior_old()
    receipt = read(ROOT / 'docs/native_basis_controls_attempt001.json')
    if receipt['status'] != 'PASS' or receipt['sources'] != native_sources():
        raise ValueError('native adapter frozen source drift')


def run():
    old()
    before, n = sources(), 1
    while (ROOT / f'docs/basis_supervised_controls_attempt{n:03}.json').exists():
        n += 1
    log = io.StringIO()
    begin = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'basis_supervised.tests', 'native_basis.tests', 'exact_basis.tests', 'exact_primal.tests',
        'lp_sandwich.tests', 'lp_diagnostic_archive.tests']))
    old()
    if before != sources():
        raise ValueError('sources changed during controls')
    from basis_supervised.tests import OBSERVATIONS, ARTIFACT_ROOT
    from native_basis.adapter import CAPTURES
    status = 'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    receipt = {'status': status, 'sources': before, 'tests_run': result.testsRun, 'log': log.getvalue(),
               'seconds': time.monotonic() - begin, 'observations': OBSERVATIONS,
               'control_artifact_root': str(ARTIFACT_ROOT),
               'supervised_completed_native_captures': sum(
                   o.get('costs', {}).get('native_calls') or 0 for o in OBSERVATIONS),
               'prior_regression_native_captures': CAPTURES,
               'real_network_LP_calls': 0, 'real_LP_reconstructions': 0,
               'sealed_sources_unchanged': True,
               'scope': 'analytic runtime controls; synthetic stalls are fault injection, not measured solver difficulty'}
    dest = ROOT / f'docs/basis_supervised_controls_attempt{n:03}.json'
    save_new(dest, receipt)
    print(log.getvalue())
    print(dest, status, flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    run()
