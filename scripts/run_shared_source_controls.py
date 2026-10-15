"""Print frozen joint-source control report; no model/native/GPU dispatch."""
import io
import json
from pathlib import Path
import time
import unittest

from scripts.test_shared_source import Controls
from scoped_source.direct_supervised import bindings
from scoped_source.hz_request_store import digest


def main():
    start=time.monotonic(); stream=io.StringIO()
    modules=('scripts.test_shared_source','scripts.test_direct_lifecycle','scripts.test_direct_join',
             'scripts.test_pair_scoped_consumer','scripts.test_bounded_runtime','scripts.test_wire_evidence',
             'scripts.test_project_layout','scripts.test_review_handoff')
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    code=bindings(); root=Path(__file__).resolve().parents[1]
    for name in ('scripts/test_shared_source.py','scripts/run_shared_source_controls.py'):
        code[name]=digest((root/name).read_bytes())
    print(json.dumps(dict(schema='SHARED_SOURCE_CONTROLS_V1',protocol_commit='93a230630',
        status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,errors=len(result.errors),
        failures=len(result.failures),seconds=time.monotonic()-start,log=stream.getvalue(),bindings=code,
        differentials=Controls.differentials,supervised=Controls.supervised,equations=Controls.equations,
        native_optimizer_runs=0,real_model_runs=0,gpu_runs=0,full_capacity_admitted=False),
        sort_keys=True,separators=(',',':')))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
