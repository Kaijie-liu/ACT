"""Analytic integration only: preserve all attempts, never reconstruct real LPs."""
import io
import time
import unittest
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from modular_supervised.flow import sources,audit,costs


def old():
    from modular_basis.controls import old as sealed,sources as component_sources
    sealed()
    c=read(ROOT/'docs/modular_basis_controls_attempt002.json')
    if c['status']!='PASS' or c['sources']!=component_sources():raise ValueError('frozen modular drift')


def run():
    old();before=sources();n=1
    while (ROOT/f'docs/modular_supervised_controls_attempt{n:03}.json').exists():n+=1
    log=io.StringIO();begin=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'modular_supervised.tests','fidelity_supervised.native_tests',
        'modular_basis.tests','exact_basis.tests','exact_primal.tests','lp_sandwich.tests']))
    old()
    if before!=sources():raise ValueError('control sources changed')
    from modular_supervised.tests import OBSERVATIONS,ARTIFACT_ROOT
    status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'
    records=[]
    if status=='PASS':
        for folder in sorted(ARTIFACT_ROOT.iterdir()):
            if folder.name in ('good','redundant','large','inexact','error','expired','reserve',
                    'outer_cutoff','batch_timeout','size_limit','tiny_full','multi_round') or folder.name.startswith('full_'):
                records.append({'root':str(folder),'terminal':audit(folder),'costs':costs(folder)})
    dest=ROOT/f'docs/modular_supervised_controls_attempt{n:03}.json'
    save_new(dest,{'status':status,'sources':before,'tests_run':result.testsRun,'log':log.getvalue(),
        'seconds':time.monotonic()-begin,'observations':OBSERVATIONS,'terminals':records,
        'control_artifact_root':str(ARTIFACT_ROOT),
        'artifact_sha256':{str(p.relative_to(ARTIFACT_ROOT)):digest(p.read_bytes())
                            for p in sorted(ARTIFACT_ROOT.rglob('*')) if p.is_file()},
        'real_network_LP_calls':0,'real_LP_reconstructions':0,'sealed_sources_unchanged':True,
        'scope':'analytic native LPs and injected faults; supplied-LP integration, not efficacy'})
    print(log.getvalue());print(dest,status,flush=True)
    if status!='PASS':raise SystemExit(1)


if __name__=='__main__':run()
