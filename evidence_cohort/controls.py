"""Archive execution controls before freeze, without querying new inputs."""
import io
import json
import time
import unittest
from evidence_cohort.contract import CONTROLS,hashes,save


if __name__=='__main__':
    if CONTROLS.exists():raise FileExistsError('control archive immutable')
    stream=io.StringIO();started=time.monotonic()
    modules=['evidence_cohort.tests','scripts.test_general_evidence','scripts.test_optional_evidence_budget']
    tests=unittest.TestLoader().loadTestsFromNames(modules)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    value={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'errors':[(str(t),e) for t,e in result.errors],'failures':[(str(t),e) for t,e in result.failures],
        'skipped':len(result.skipped),'seconds':time.monotonic()-started,'log':stream.getvalue(),
        'sources':hashes(),'modules':modules,'new_selected_endpoints_queried':0}
    save(CONTROLS,value);print(json.dumps({k:v for k,v in value.items() if k not in ('sources','log')},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
