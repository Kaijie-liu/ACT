"""Execute the frozen 12-call CPU/stub roster, or audit without proposals."""
import argparse
import ast
import io
import os
from pathlib import Path
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha
from scripts import hz_device_lifecycle as life
from scripts.hz_device_lifecycle_audit import audit as audit_call
from scripts.hz_propagation_supervised import finite

TEST='scripts/test_hz_device_lifecycle.py'
EXPECTED_NAMES=sorted([
    'test_failed_prefix_removal_rejected','test_failed_producer_still_releases_and_charges',
    'test_fixed_calls_and_fault_reach','test_foreign_tenant_is_not_owned_release_failure',
    'test_missing_release_and_rewritten_checker_rejected','test_nested_cost_and_api_deadline',
    'test_original_exact_bounds_unchanged','test_owned_pid_reappearance_ambiguity_fails_closed',
    'test_rebound_terminal_cost_and_deadline_rejected','test_release_binding_chronology_and_nan',
    'test_source_identity_and_cuda_refusal','test_unconfirmed_release_prevents_later_launch'])


def names():
    cls=next(n for n in ast.parse((ROOT/TEST).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LifecycleControls')
    actual=sorted(n.name for n in cls.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    if actual!=EXPECTED_NAMES: raise ValueError('fixed test roster changed')
    return actual


def run(root):
    os.environ['HZ_LIFECYCLE_ROOT']=str(root)
    from scripts.test_hz_device_lifecycle import LifecycleControls
    start=time.monotonic(); output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleControls))
    (root/'tests.log').write_text(output.getvalue())
    passed=result.wasSuccessful() and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses
    summary={'status':'PASS' if passed else 'FAIL','tests':result.testsRun,'names':names(),
             'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'expected_failures':len(result.expectedFailures),'unexpected_successes':len(result.unexpectedSuccesses),
             'implementation_sha256':sha(root/'implementation.json'),'tests_sha256':sha(root/'tests.log'),
             'calls_sha256':sha(root/'calls.json') if (root/'calls.json').is_file() else None,
             'seconds':time.monotonic()-start,'physical_cuda_calls':0,'real_requests':0}
    save(root/'summary.json',summary); print(output.getvalue()); print(summary)
    return 0 if passed else 1


def audit(root):
    from scripts.test_hz_device_lifecycle import roster, fault_witness, EXTRA
    summary=load(root/'summary.json'); finite(summary['seconds'])
    if (summary['status']!='PASS' or summary['tests']!=len(names()) or summary['names']!=names()
            or summary['failures'] or summary['errors'] or summary['skipped']
            or summary['expected_failures'] or summary['unexpected_successes']
            or summary['physical_cuda_calls']!=0 or summary['real_requests']!=0
            or summary['implementation_sha256']!=sha(root/'implementation.json')
            or summary['tests_sha256']!=sha(root/'tests.log') or summary['calls_sha256']!=sha(root/'calls.json')):
        raise ValueError('controls did not complete')
    bindings=load(root/'implementation.json')
    if set(bindings)!=set((*life.FILES,*EXTRA)): raise ValueError('source inventory')
    for name,digest in bindings.items():
        if sha(ROOT/name)!=digest or sha(root/'implementation'/name)!=digest: raise ValueError('execution source changed: '+name)
    calls=load(root/'calls.json'); required=roster()
    if set(calls)!=set(required): raise ValueError('fixed roster')
    records=[]; reference=load(ROOT/'docs/hz_device_candidates_20261001_r3.json')['checked']
    for name,(spec,budget,status) in required.items():
        path=root/name; obs=calls[name]; inv=load(path/'invocation.json')
        if (obs!=load(root/(name+'_observed.json')) or inv['spec']!=spec or inv['budget']!=budget
                or obs['result']['status']!=status): raise ValueError('request result/identity: '+name)
        checked=audit_call(path,observation=obs,recheck=status==life.DONE); fault_witness(path,spec['control'])
        if status==life.DONE and load(path/'receive.json')['checked']['results']!=reference[spec['case']]['bounds']:
            raise ValueError('original exact bounds changed')
        term=load(path/'terminal.json')
        records.append({'name':name,'observation':obs,'audit':checked,
                        'phase_seconds':{s['phase']:s['seconds'] for s in term['stages']},
                        'parent_seconds':term['parent_seconds'],
                        'publication_and_return_seconds':obs['result']['seconds']-term['seconds']})
    return {'schema':'HZ_DEVICE_LIFECYCLE_ARCHIVE_V1','status':'PASS','root':str(root),
            'protocol_sha256':life.PROTOCOL_SHA,'implementation_sha256':summary['implementation_sha256'],
            'summary_sha256':sha(root/'summary.json'),'tests':summary['tests'],'calls':records,
            'complete_given_hz_calls':sum(r['observation']['result']['status']==life.DONE for r in records),
            'normal_bounds':8,'physical_cuda_calls':0,'real_requests':0,'native_solves':0,
            'complete_moe_proofs':0,'seconds':summary['seconds']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    p.add_argument('--check',action='store_true'); p.add_argument('--report',type=Path); a=p.parse_args()
    root=a.root.resolve()
    if not root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('project archive only')
    if a.check:
        result=audit(root)
        if a.report:
            if a.report.resolve().parent!=ROOT/'docs': raise ValueError('compact repo report only')
            save(a.report,result)
        print({k:v for k,v in result.items() if k!='calls'})
    else:
        if a.report: raise ValueError('report requires audit')
        raise SystemExit(run(root))
