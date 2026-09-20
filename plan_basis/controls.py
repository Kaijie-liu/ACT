"""Versioned, fail-preserving controls; no real LP reconstruction or native call."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest


def sealed():
    from modular_diagnostic.contract import verify
    verify()
    v=read(ROOT/'docs/modular_diagnostic_v1_execution_results.json')
    for path,h in v['artifact_sha256'].items():
        if digest((ROOT/path).read_bytes())!=h:raise ValueError('frozen artifact drift')


def sources():
    paths=list((ROOT/'plan_basis').glob('*.py'))
    for folder in ('modular_basis','sparse_basis','exact_basis','exact_primal','lp_sandwich'):
        paths+=list((ROOT/folder).glob('*.py'))
    paths += [ROOT/'docs/plan_basis_v1.md',ROOT/'single_check_portable/execution.py',ROOT/'portable_proof/runtime.py']
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(paths)}


def run():
    sealed();before=sources();n=1
    while (ROOT/f'docs/plan_basis_controls_attempt{n:03}.json').exists():n+=1
    begin=time.monotonic();log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['plan_basis.tests','modular_basis.tests','exact_basis.tests','exact_primal.tests','lp_sandwich.tests']))
    sealed()
    if sources()!=before:raise ValueError('control source drift')
    from .tests import ARTIFACT_ROOT,OBSERVATIONS
    import modular_basis.tests as old
    roots=[p for p in (ARTIFACT_ROOT,old.ARTIFACT_ROOT) if p is not None]
    receipt={'status':'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
             'tests_run':result.testsRun,'log':log.getvalue(),'seconds':time.monotonic()-begin,
             'sources':before,'observations':OBSERVATIONS,'artifact_roots':list(map(str,roots)),
             'artifact_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for root in roots
                                for p in sorted(root.rglob('*')) if p.is_file()},
             'real_LP_calls':0,'native_calls':0,'production_integration':False,
             'scope':'analytic mechanism and regressions, not real efficacy or matched timings'}
    path=ROOT/f'docs/plan_basis_controls_attempt{n:03}.json';save_new(path,receipt)
    print(log.getvalue());print(path,receipt['status'],flush=True)
    if receipt['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
