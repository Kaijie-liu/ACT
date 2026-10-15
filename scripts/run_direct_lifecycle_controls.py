"""Frozen fixed-stub lifecycle controls. Prints a compact bound report."""
import io
import json
from pathlib import Path
import time
import unittest

from scripts.test_direct_lifecycle import Controls
from scoped_source.direct_supervised import bindings
from scoped_source.hz_request_store import digest


def main():
    start = time.monotonic(); stream = io.StringIO()
    modules = ('scripts.test_direct_lifecycle', 'scripts.test_direct_join', 'scripts.test_pair_scoped_consumer',
               'scripts.test_bounded_runtime', 'scripts.test_wire_evidence',
               'scripts.test_project_layout', 'scripts.test_review_handoff')
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    root = Path(__file__).resolve().parents[1]
    code = bindings()
    for name in ('scripts/test_direct_lifecycle.py', 'scripts/run_direct_lifecycle_controls.py'):
        code[name] = digest((root/name).read_bytes())
    report = dict(schema='DIRECT_LIFECYCLE_CONTROLS_V1', protocol_commit='c4e8c6253',
                  status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  errors=len(result.errors), failures=len(result.failures), seconds=time.monotonic()-start,
                  log=stream.getvalue(), bindings=code, differentials=Controls.differentials,
                  supervised=Controls.supervised, real_model_runs=0, native_optimizer_runs=0, gpu_runs=0,
                  full_capacity_admitted=False)
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
