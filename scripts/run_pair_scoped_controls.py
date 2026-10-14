"""Finite synthetic report only. No real models, GPU or native solver."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

from scripts import (test_pair_scoped_consumer, test_bounded_runtime,
                     test_wire_evidence, test_review_handoff, test_project_layout)

ROOT=Path(__file__).resolve().parents[1]
FILES=('scoped_source/pair_workspace.py','scoped_source/bounded_runtime.py',
       'scoped_source/direct_candidates.py','scoped_source/scaled_endpoints.py',
       'scoped_source/check_scaled_references.py','scoped_source/wire_forest.py',
       'scoped_source/hz_request_store.py','scoped_source/forest_endpoints.py',
       'scripts/test_pair_scoped_consumer.py','scripts/test_bounded_runtime.py',
       'scripts/test_wire_evidence.py','scripts/run_pair_scoped_controls.py',
       'docs/hz_pair_scoped_consumer_protocol_20261014.md')


def run():
    start=time.monotonic(); log=io.StringIO()
    modules=(test_pair_scoped_consumer,test_bounded_runtime,test_wire_evidence,
             test_review_handoff,test_project_layout)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    record=dict(schema='PAIR_SCOPED_CONSUMER_FINITE_CONTROLS_V1',
        protocol_commit='6ee055f630fff745a5cc4a98d34b21d9fff2b955',status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=log.getvalue(),
        code={p:dict(bytes=(ROOT/p).stat().st_size,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in FILES},
        cases=test_pair_scoped_consumer.Controls.differentials,
        runtime=dict(python=sys.version,executable=sys.executable,isolated=sys.flags.isolated,
                     no_site=sys.flags.no_site,wall_seconds=time.monotonic()-start),
        execution=dict(default_on=False,candidate_width=8,checker_block_width=2),
        limits=dict(seconds=300,work=8000000,stored_bits=4096,working_bits=8192,
                    frame_bytes=4194304,candidate_bytes=16777216-8192,arena_bytes=134217728),
        real_model_runs=0,gpu_runs=0,optimizer_runs=0,full_cost_closed=False,quota_change=False,
        scope='finite synthetic differential and fault controls, not performance or capacity admission',
        unclosed=['source/planning/native/Python temporary memory','Python norm groups and scalar banks',
                  'full-model source, cumulative Work and candidate capacity'],
        history=['First old-runtime regression: portable module inventory omitted pair_workspace; corrected.',
                 'Draft full differential passed; read-only audit then hardened scope lifetime and ownership.',
                 'Added closed-scope test initially had a NameError in the test itself; corrected without changing gates.'])
    if any(m in sys.modules for m in ('torch','numpy','scipy')):
        record['status']='FAIL'; record['unexpected_numerical_dependency']=True
    return record


if __name__=='__main__':
    report=run(); print(json.dumps(report,sort_keys=True,indent=2)); sys.exit(report['status']!='PASS')
