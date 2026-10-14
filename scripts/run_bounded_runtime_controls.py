"""Print one finite stdlib control report; never run a trained model or solver."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

from scripts import test_bounded_runtime,test_wire_evidence,test_review_handoff,test_project_layout

ROOT=Path(__file__).resolve().parents[1]
FILES=('scoped_source/bounded_runtime.py','scoped_source/direct_candidates.py','scoped_source/scaled_endpoints.py',
       'scoped_source/hz_request_store.py','scoped_source/check_scaled_references.py','scoped_source/wire_forest.py',
       'scoped_source/forest_endpoints.py','scripts/test_bounded_runtime.py','scripts/test_wire_evidence.py',
       'scripts/run_bounded_runtime_controls.py','docs/hz_bounded_runtime_protocol_20261014.md')


def run():
    start=time.monotonic(); log=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (
        test_bounded_runtime,test_wire_evidence,test_review_handoff,test_project_layout))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report=dict(schema='BOUNDED_RUNTIME_FINITE_CONTROLS_V1',
        protocol_commit='36c0e8b91f59f495252f449aaa75b97240454ff3',
        status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),log=log.getvalue(),
        code={p:dict(bytes=(ROOT/p).stat().st_size,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in FILES},
        cases=test_bounded_runtime.Controls.differentials,
        scope='fixed synthetic interface controls, not a performance experiment',
        runtime=dict(python=sys.version,executable=sys.executable,flags=dict(isolated=sys.flags.isolated,no_site=sys.flags.no_site),
                     wall_seconds=time.monotonic()-start),
        limits=dict(seconds=300,work=8000000,stored_bits=4096,working_bits=8192,
                    tile_bytes=4194304,candidate_bytes=16777216,metadata_reserve=8192,arena_bytes=134217728),
        full_cost_closed=False,new_quota=None,real_model_runs=0,gpu_runs=0,optimizer_runs=0,
        goal_status={f'G{i}':'OPEN' for i in range(1,7)},
        notes=['Owned bytes are not RSS; decoded Fraction and Python/native scratch remain open.',
               'RootTable is an added index on the existing checked forest, not direct source replacement.',
               'Publication acknowledged bytes require the separate publisher partial-write ledger.',
               'Historical static source-anchor reports remain bound to their original commits.'])
    if any(m in sys.modules for m in ('torch','numpy','scipy')):
        report['status']='FAIL'; report['dependency_error']='unexpected numerical library import'
    return report


if __name__=='__main__':
    record=run(); print(json.dumps(record,sort_keys=True,indent=2)); sys.exit(record['status']!='PASS')
