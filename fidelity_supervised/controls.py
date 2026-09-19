"""Retained analytic controls and sealed-source checks; no real LP solve."""
import io
import time
import unittest
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from fidelity_supervised.flow import sources, audit, costs


def old():
    from sparse_supervised.study import verify
    verify()
    a=read(ROOT/'docs/sparse_supervised_real_v1_execution_results.json')
    for path,h in a['artifact_sha256'].items():
        if digest((ROOT/path).read_bytes())!=h:
            raise ValueError('sealed sparse execution drift')


def run():
    old()
    before, n = sources(), 1
    while (ROOT / f'docs/fidelity_supervised_controls_attempt{n:03}.json').exists():
        n += 1
    log, begin = io.StringIO(), time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'fidelity_supervised.tests', 'fidelity_supervised.native_tests', 'sparse_supervised.tests', 'sparse_basis.tests', 'basis_compatibility.tests',
        'basis_supervised.tests', 'native_basis.tests', 'exact_basis.tests', 'exact_primal.tests',
        'lp_sandwich.tests', 'lp_diagnostic_archive.tests']))
    old()
    if before != sources():
        raise ValueError('sources changed during controls')
    from fidelity_supervised.tests import OBSERVATIONS, ARTIFACT_ROOT
    status = 'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    records = []
    if status == 'PASS':
        for folder in sorted(ARTIFACT_ROOT.iterdir()):
            if folder.name in ('good', 'redundant', 'large', 'inexact', 'error', 'expired', 'reserve',
                               'outer_cutoff', 'batch_timeout', 'size_limit', 'tiny_full') or folder.name.startswith('full_'):
                records.append({'root': str(folder), 'terminal': audit(folder), 'costs': costs(folder)})
    dest = ROOT / f'docs/fidelity_supervised_controls_attempt{n:03}.json'
    save_new(dest, {'status': status, 'sources': before, 'tests_run': result.testsRun,
                   'log': log.getvalue(), 'seconds': time.monotonic() - begin,
                   'observations': OBSERVATIONS, 'terminals': records,
                   'control_artifact_root': str(ARTIFACT_ROOT),
                   'artifact_sha256': {str(p.relative_to(ARTIFACT_ROOT)): digest(p.read_bytes())
                                      for p in sorted(ARTIFACT_ROOT.rglob('*')) if p.is_file()},
                   'real_network_LP_calls': 0, 'real_LP_reconstructions': 0,
                   'sealed_sources_unchanged': True,
                   'scope': 'analytic outer controls, synthetic fault stalls, not real LP efficacy'})
    print(log.getvalue()); print(dest, status, flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    run()
