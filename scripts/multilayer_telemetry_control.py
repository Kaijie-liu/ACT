"""One <=300s stdlib-only telemetry/decomposition control, existing supervisor."""
import time
ENTRY=time.monotonic()
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
from genbab_cpu_ready import ACT

CODE=('scripts/multilayer_telemetry_control.py','tests/test_multilayer_telemetry.py',
      'scripts/multilayer_stage1.py','scripts/multilayer_stage1_audit.py',
      'act/pipeline/moe/multilayer_study.py')


def main(folder,worker):
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    if worker:
        spec=load(folder.parent/'freeze.json')
        assert all(sha(ROOT/p)==h for p,h in spec['sources'].items())
        spec=importlib.util.spec_from_file_location('telemetry_controls',ROOT/'tests/test_multilayer_telemetry.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        save(folder/'outcome.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
            real_queries=0,solver_calls=0,as_limit=resource.getrlimit(resource.RLIMIT_AS)))
        return
    folder.mkdir();child=folder/'fixed';child.mkdir()
    save(folder/'freeze.json',dict(sources={p:sha(ROOT/p) for p in CODE},seconds=300,
        artifacts=128*2**20,as_bytes=8*2**30,policy=POLICY,real_models=0,solver_calls=0,
        cases=['mandatory start/identity/return and heartbeat throttle','exception/timeout retained start',
               '150/50/5 boundaries, missing, aggregate not individual',
               'same UNSAFE denominator, original endpoint unchanged','reserved boundary pre-worker stop']))
    env=dict(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    receipt=execute([ACT,'-I','-S','-B',__file__,str(child),'--worker'],run_deadline=ENTRY+294,
        cleanup_deadline=ENTRY+298,env=env,rss_limit=None,development_policy=POLICY)
    save(folder/'session.json',receipt)
    from b1_observed_batch import session_ok
    good=(session_ok(receipt) and (child/'outcome.json').exists()
          and load(child/'outcome.json')['status']=='PASS' and time.monotonic()<ENTRY+300)
    assert sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())<128*2**20
    save(folder/'complete.json',dict(status='PASS' if good else 'FAIL',seconds=time.monotonic()-ENTRY,
        freeze_sha256=sha(folder/'freeze.json'),session_sha256=sha(folder/'session.json'),
        outcome_sha256=sha(child/'outcome.json') if (child/'outcome.json').exists() else None))
    print(json.dumps(load(folder/'complete.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--worker',action='store_true')
    a=p.parse_args();main(a.folder,a.worker)
