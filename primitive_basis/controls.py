"""Versioned analytic receipts; original real LPs are only hash-checked."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest


def old():
    from fidelity_supervised.study import verify
    verify()
    v=read(ROOT/'docs/fidelity_supervised_real_v2_execution_results.json')
    for path,h in v['artifact_sha256'].items():
        if digest((ROOT/path).read_bytes())!=h:raise ValueError('sealed V2 artifact drift')


def sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in
       ('sparse_basis/engine.py','sparse_basis/tests.py','exact_basis/propose.py','exact_basis/tests.py',
        'exact_primal/propose.py','lp_sandwich/check.py','lp_sandwich/tests.py',
        'single_check_portable/execution.py','portable_proof/runtime.py',
        'docs/fidelity_supervised_real_v2_execution_results.json')]
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(paths)}


def run():
    old();before=sources();n=1
    while (ROOT/f'docs/primitive_basis_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['primitive_basis.tests','exact_basis.tests','exact_primal.tests','lp_sandwich.tests']))
    old()
    if before!=sources():raise ValueError('control source drift')
    from primitive_basis.tests import OBSERVATIONS,ARTIFACT_ROOT
    receipt={'status':'PASS' if tests.wasSuccessful() and not tests.skipped else 'FAIL',
       'tests_run':tests.testsRun,'log':log.getvalue(),'sources':before,'seconds':time.monotonic()-begin,
       'observations':OBSERVATIONS,'control_artifact_root':str(ARTIFACT_ROOT),
       'artifact_sha256':{str(p.relative_to(ARTIFACT_ROOT)):digest(p.read_bytes())
                           for p in sorted(ARTIFACT_ROOT.rglob('*')) if p.is_file()},
       'real_LP_reconstructions':0,'real_native_calls':0,'old_freezes_unchanged':True,
       'production_integration':False,'scope':'analytic arithmetic controls only; no actual-LP efficacy claim'}
    path=ROOT/f'docs/primitive_basis_controls_attempt{n:03}.json';save_new(path,receipt)
    print(log.getvalue());print(path,receipt['status'],flush=True)
    if receipt['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
