"""Versioned control receipts; no real LP invocation or protocol replacement."""
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest


def old():
    from basis_supervised.controls import old as prior
    from basis_supervised.flow import sources as previous_sources
    prior()
    receipt=read(ROOT/'docs/basis_supervised_controls_attempt002.json')
    if receipt['sources']!=previous_sources():raise ValueError('supervision source drift')


def sources():
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}


def run():
    old();before=sources();n=1
    while (ROOT/f'docs/sparse_basis_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'sparse_basis.tests','basis_compatibility.tests','basis_supervised.tests','native_basis.tests',
        'exact_basis.tests','exact_primal.tests','lp_sandwich.tests','lp_diagnostic_archive.tests']))
    old()
    if before!=sources():raise ValueError('control source drift')
    from sparse_basis.tests import OBSERVATIONS,ARTIFACT_ROOT
    from sparse_basis.native import CAPTURES
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    dest=ROOT/f'docs/sparse_basis_controls_attempt{n:03}.json'
    save_new(dest,{'status':status,'tests_run':result.testsRun,'log':log.getvalue(),'sources':before,
                  'seconds':time.monotonic()-begin,'observations':OBSERVATIONS,
                  'native_analytic_captures':CAPTURES,'control_artifact_root':str(ARTIFACT_ROOT),
                  'artifact_sha256':{str(p.relative_to(ARTIFACT_ROOT)):digest(p.read_bytes())
                                     for p in sorted(ARTIFACT_ROOT.rglob('*')) if p.is_file()},
                  'real_network_calls':0,'real_LP_reconstructions':0,'old_freezes_unchanged':True,
                  'production_integration':False,
                  'scope':'new sparse engine and native mapping controls; structured synthetic scale, not actual-network efficacy'})
    print(log.getvalue());print(dest,status,flush=True)
    if status!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
