"""Record tiny V2 reception controls; never repeats frozen profiling objects."""
import argparse
import io
from pathlib import Path
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha


def run():
    report = ROOT / 'docs/source_cost_interface_controls_20260925_r1.json'
    if report.exists(): raise FileExistsError(report)
    cfg = load(ROOT / 'configs/backend_controls/batched_evidence_study_r1.json')
    for name, digest in cfg['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('frozen source changed')
    from source_cost_controls.tests import Controls
    start = time.monotonic(); output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    obj = {'status': 'PASS' if result.wasSuccessful() and result.testsRun == 4 else 'FAIL',
           'tests': result.testsRun, 'seconds': time.monotonic() - start, 'log': output.getvalue(),
           'new_sources': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'source_cost_controls').glob('*.py'))},
           'frozen_sources_intact': len(cfg['sources']), 'real_requests': 0, 'native_solver_queries': 0,
           'new_real_certificates': 0, 'original_two_profiles_rerun': False,
           'fixture': {'experts': 2, 'classes': 2, 'width': 1, 'depth': 0, 'seed': 91},
           'scope': 'interface repair and tiny correctness controls only; no measured frozen profile repetition'}
    save(report, obj)
    print({k: obj[k] for k in ('status', 'tests', 'seconds', 'original_two_profiles_rerun', 'real_requests')})
    if obj['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': run()
