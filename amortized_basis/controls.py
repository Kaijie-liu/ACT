"""Retained control receipts; all previous files checked, no real queries."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest


def sealed():
    from plan_basis.controls import sealed as old
    old()
    r=read(ROOT/'docs/plan_basis_controls_attempt004.json')
    for path,h in {**r['sources'],**r['artifact_sha256']}.items():
        if digest((ROOT/path).read_bytes())!=h:raise ValueError('sealed V1 drift')


def sources():
    paths=list((ROOT/'amortized_basis').glob('*.py'))
    for folder in ('plan_basis','modular_basis','sparse_basis','exact_basis','exact_primal','lp_sandwich'):
        paths+=list((ROOT/folder).glob('*.py'))
    paths += [ROOT/'docs/amortized_basis_v1.md',ROOT/'single_check_portable/execution.py',ROOT/'portable_proof/runtime.py']
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(paths)}


def run():
    sealed();before=sources();n=1
    while (ROOT/f'docs/amortized_basis_controls_attempt{n:03}.json').exists():n+=1
    begin=time.monotonic();log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['amortized_basis.tests','plan_basis.tests','modular_basis.tests','exact_basis.tests','exact_primal.tests','lp_sandwich.tests']))
    sealed()
    if sources()!=before:raise ValueError('control source drift')
    from .tests import ARTIFACT_ROOT,OBSERVATIONS
    import plan_basis.tests as oldplan
    import modular_basis.tests as oldmodular
    roots=[p for p in (ARTIFACT_ROOT,oldplan.ARTIFACT_ROOT,oldmodular.ARTIFACT_ROOT) if p is not None]
    receipt={'status':'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
             'tests_run':result.testsRun,'log':log.getvalue(),'seconds':time.monotonic()-begin,
             'sources':before,'observations':OBSERVATIONS,'artifact_roots':list(map(str,roots)),
             'artifact_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for root in roots
                                for p in sorted(root.rglob('*')) if p.is_file()},
             'real_LP_calls':0,'native_calls':0,'production_integration':False,
             'scope':'fixed four-mode analytic ablation and regressions; not matched real timing'}
    path=ROOT/f'docs/amortized_basis_controls_attempt{n:03}.json';save_new(path,receipt)
    print(log.getvalue());print(path,receipt['status'],flush=True)
    if receipt['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
