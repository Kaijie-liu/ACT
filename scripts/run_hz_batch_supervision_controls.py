"""Run/archive the fixed supervisor controls, or audit without new proposals."""
import argparse
import ast
import io
import os
from pathlib import Path
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha
from scripts import hz_batch_support_supervised as flow

TEST = 'scripts/test_hz_batch_support_supervised.py'
SELF = 'scripts/run_hz_batch_supervision_controls.py'


def names():
    cls=next(n for n in ast.parse((ROOT/TEST).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SupervisionControls')
    return sorted('scripts.test_hz_batch_support_supervised.SupervisionControls.'+n.name
                  for n in cls.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))


def run(root):
    os.environ['HZ_BATCH_SUPERVISION_ROOT']=str(root)
    from scripts.test_hz_batch_support_supervised import SupervisionControls
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SupervisionControls)
    start=time.monotonic(); stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    # Preserve first failures as well as successes; later attempts use new roots.
    (root/'tests.log').write_text(stream.getvalue())
    summary={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
             'test_names':names(),'failures':len(result.failures),'errors':len(result.errors),
             'implementation_sha256':sha(root/'implementation.json'),
             'calls_sha256':sha(root/'calls.json') if (root/'calls.json').exists() else None,
             'test_log_sha256':sha(root/'tests.log'),'control_seconds':time.monotonic()-start,
             'no_gpu_or_real_requests':True}
    save(root/'summary.json',summary)
    print(stream.getvalue())
    print(summary)
    return 0 if result.wasSuccessful() else 1


def audit(root):
    from scripts.test_hz_batch_support_supervised import roster, fault_witness
    summary=load(root/'summary.json')
    if (summary['status']!='PASS' or summary['failures'] or summary['errors']
            or summary['test_names']!=names() or summary['tests']!=len(names())
            or summary['implementation_sha256']!=sha(root/'implementation.json')
            or summary['calls_sha256']!=sha(root/'calls.json')
            or summary['test_log_sha256']!=sha(root/'tests.log')):
        raise ValueError('complete controls not established')
    sources=load(root/'implementation.json')
    if set(sources)!=set(flow.FILES)|{TEST,SELF}: raise ValueError('implementation inventory')
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest or sha(root/'implementation'/name)!=digest:
            raise ValueError('executed source binding: '+name)
    calls=load(root/'calls.json'); required=roster()
    if set(calls)!=set(required): raise ValueError('complete frozen call inventory')
    records=[]
    for name,(spec,budget,rss,status) in required.items():
        path=root/name; call=calls[name]
        if call!=load(root/(name+'_observed.json')) or call['status']!=status or load(path/'spec.json')!=spec:
            raise ValueError('call status/spec/observation changed: '+name)
        inv=load(path/'invocation.json')
        if inv['budget']!=budget or inv['rss_limit']!=rss: raise ValueError('frozen call resource policy')
        checked=flow.audit(path,call,recheck=status==flow.DONE)
        fault_witness(path,spec['control'])
        records.append({'name':name,'call':call,'audit':checked})
    injected=[]
    for stem in ('publication','hash'):
        info=load(root/(stem+'_clock_observed.json'))
        if info['synthetic_clock_injection'] is not True or info['call']['status']!='TIMEOUT':
            raise ValueError('clock fault evidence')
        injected.append({'name':stem,'audit':flow.audit(root/(stem+'_clock_control'),info['call']),
                         'not_wallclock_measurement':True})
    return {'schema':'HZ_BATCH_SUPERVISION_ARCHIVE_V1','status':'PASS','root':str(root),
            'protocol_sha256':flow.PROTOCOL_SHA,'implementation_sha256':summary['implementation_sha256'],
            'summary_sha256':sha(root/'summary.json'),'tests':summary['tests'],'calls':records,
            'clock_injection_controls':injected,'complete_given_hz_calls':sum(r['call']['status']==flow.DONE for r in records),
            'complete_moe_proofs':0,'native_solves':0,'gpu_executions':0,'real_requests':0,
            'portable_solver_free_claim':False,'control_seconds':summary['control_seconds']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path); p.add_argument('--check',action='store_true'); p.add_argument('--report',type=Path)
    a=p.parse_args(); root=a.root.resolve()
    if not root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('project archive root')
    if a.check:
        result=audit(root)
        if a.report:
            path=a.report.resolve()
            if path.parent!=ROOT/'docs': raise ValueError('compact repo report only')
            save(path,result)
        print({k:v for k,v in result.items() if k not in ('calls','clock_injection_controls')})
    else:
        if a.report: raise ValueError('report requires audit')
        raise SystemExit(run(root))
