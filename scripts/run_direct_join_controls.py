"""Report the frozen synthetic direct join and unchanged-path regressions."""
import hashlib
import io
import json
from pathlib import Path
import time
import unittest

from scripts import test_direct_join as direct

MODULES=('scripts.test_direct_join','scripts.test_pair_scoped_consumer',
         'scripts.test_bounded_runtime','scripts.test_wire_evidence',
         'scripts.test_project_layout','scripts.test_review_handoff')
FILES=('docs/hz_direct_join_protocol_20261015.md','scoped_source/source_dag.py',
       'scoped_source/direct_source_format.py','scoped_source/direct_source_build.py',
       'scoped_source/check_direct_source.py','scoped_source/check_scaled_references.py',
       'scoped_source/scaled_endpoints.py','scoped_source/direct_candidates.py',
       'scoped_source/bounded_runtime.py','scoped_source/pair_workspace.py',
       'scoped_source/direct_package.py','scoped_source/direct_execution.py',
       'scoped_source/wire_package.py','scoped_source/wire_archive.py',
       'scripts/test_direct_join.py','scripts/run_direct_join_controls.py')


def result_summary(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    keys=('status','required','positive','required_endpoints','checked_endpoints','missing_endpoints',
          'native_float_proof','checkpoint_correspondence_checked')
    return {**{k:value[k] for k in keys},'exact_result_sha256':hashlib.sha256(raw).hexdigest()}


def main():
    start=time.monotonic(); stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    root=Path(__file__).resolve().parents[1]
    bindings=[dict(path=p,sha256=hashlib.sha256((root/p).read_bytes()).hexdigest(),bytes=(root/p).stat().st_size) for p in FILES]
    differentials=[dict(x,result=result_summary(x['result'])) for x in direct.Controls.differentials]
    executions=[]
    for entry in direct.Controls.executions:
        x=dict(entry)
        if 'result' in x: x['result']=result_summary(x['result'])
        if 'replay' in x: x['replay']=dict(x['replay'],result=result_summary(x['replay']['result']))
        executions.append(x)
    report=dict(schema='DIRECT_JOIN_CONTROLS_V1',protocol_commit='485307a93',
                status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                errors=len(result.errors),failures=len(result.failures),seconds=time.monotonic()-start,
                log=stream.getvalue(),bindings=bindings,differentials=differentials,
                executions=executions,real_model_runs=0,native_optimizer_runs=0,gpu_runs=0,
                whole_runtime_admitted=False,hard_outer_supervisor_integrated=False)
    print(json.dumps(report,sort_keys=True,separators=(',',':'))); return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
