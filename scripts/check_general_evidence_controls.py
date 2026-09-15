"""Archive control tests once. No CIFAR/model verification endpoints are queried."""
import io
import json
import time
import unittest
from pathlib import Path
from scripts.optional_evidence_dev_contract import ROOT,save
from portable_proof.runtime import digest

MODULES=['scripts.test_general_evidence','scripts.test_optional_evidence_budget',
         'scripts.test_conv_pre_f0_r2','act.pipeline.moe.test_route_complexity_schedule']
OUTPUT=ROOT/'docs/general_evidence_v1_controls.json'


if __name__=='__main__':
    if OUTPUT.exists():raise FileExistsError('controls report immutable')
    start=time.monotonic();stream=io.StringIO()
    suite=unittest.TestLoader().loadTestsFromNames(MODULES)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'modules':MODULES,'seconds':time.monotonic()-start,'skipped':len(result.skipped),
        'failures':[(str(t),trace) for t,trace in result.failures],
        'errors':[(str(t),trace) for t,trace in result.errors],'log':stream.getvalue(),
        'sources':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted((ROOT/'moe_evidence').glob('*.py'))},
        'scope':'analytic/control/legacy regression only; no new real verification query or model training',
        'development_failure_note':'One initial test wrongly expected terminal_status to raise without an independent check; fixed assertion to the existing UNKNOWN_INCOMPLETE_EVIDENCE contract. Production policy was unchanged.'}
    save(OUTPUT,report);print(json.dumps({k:v for k,v in report.items() if k not in ('log','sources')},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
