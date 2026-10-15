"""Retain finite stdlib control evidence in a new directory; no real dispatch."""
import argparse
import io
from pathlib import Path
import time
import unittest

from scripts.test_paged_results import Controls
from scripts.assess_paged_result_implementation import assess
from scoped_source.direct_supervised import bindings
from scoped_source.hz_request_store import compact, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    a=parser.parse_args()
    root=Path(a.directory)
    if not root.is_absolute() or root.resolve()!=root or not root.is_relative_to('/data1/Kane/MOE'):
        raise ValueError('new project-local control directory')
    root.mkdir(parents=False,exist_ok=False)
    Controls.retained_root=root/'requests'
    start=time.monotonic(); log=io.StringIO()
    modules=('scripts.test_paged_results','scripts.test_shared_source','scripts.test_direct_lifecycle',
             'scripts.test_direct_join','scripts.test_pair_scoped_consumer','scripts.test_bounded_runtime',
             'scripts.test_wire_evidence','scripts.test_project_layout','scripts.test_review_handoff')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    source=Path(__file__).resolve().parents[1]
    code=bindings()
    for name in ('scripts/test_paged_results.py','scripts/run_paged_result_controls.py',
                 'scripts/assess_paged_result_implementation.py'):
        code[name]=digest((source/name).read_bytes())
    report=dict(schema='PAGED_RESULT_CONTROLS_V1',protocol_commit='2ea468ed6',
        status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),seconds=time.monotonic()-start,
        log=log.getvalue(),bindings=code,observations=Controls.observations,mutations=Controls.mutations,
        relocations=Controls.relocations,success_layout=assess(),real_model_runs=0,gpu_runs=0,
        native_optimizer_runs=0,full_capacity_admitted=False)
    raw=compact(report)
    with (root/'report.json').open('xb') as f: f.write(raw)
    with (root/'tests.log').open('x') as f: f.write(log.getvalue())
    print(compact(dict(status=report['status'],tests=result.testsRun,errors=report['errors'],failures=report['failures'],
                       seconds=report['seconds'],report=str(root/'report.json'),sha256=digest(raw))).decode())
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
