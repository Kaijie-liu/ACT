"""One bounded repair control; diagnostic cost carried, no real replay."""
import time
ENTRY=time.monotonic()
import argparse
import io
import json
import resource
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
from genbab_cpu_ready import ACT

CODE=('scripts/multilayer_hz_lifetime_control.py','tests/test_multilayer_hz_lifetime.py',
      'act/back_end/transfer_functions.py','act/pipeline/moe/experiment1.py',
      'act/back_end/hybridz_tf/hybridz_tf.py','act/back_end/moe/route_a.py',
      'act/back_end/moe/multilayer.py')
PRIOR=Path('/data1/Kane/MOE/results/multilayer_hz_loss_diagnosis_20261024_r1/complete.json')


def main(folder,worker,prior_path=None):
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    if worker:
        spec=load(folder.parent/'freeze.json')
        assert all(sha(ROOT/p)==h for p,h in spec['sources'].items())
        output=io.StringIO()
        suite=unittest.defaultTestLoader.loadTestsFromName(spec['test_target'])
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
        save(folder/'outcome.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests=result.testsRun,details=output.getvalue(),real_requests=0,solver_calls=0,
            as_limit=resource.getrlimit(resource.RLIMIT_AS)))
        return
    assert not folder.exists();folder.mkdir();child=folder/'fixed';child.mkdir()
    prior_path=prior_path or PRIOR
    prior=load(prior_path);seconds=prior['seconds']
    target=('test_multilayer_hz_lifetime.LifetimeControls.test_previous_history_released_before_next_propagation'
            if prior_path!=PRIOR else 'test_multilayer_hz_lifetime')
    retained=dict(path=str(prior_path),sha256=sha(prior_path),seconds=seconds,
                  meaning='retained diagnostic/control prefix; prior successful cases not repeated')
    save(folder/'freeze.json',dict(sources={p:sha(ROOT/p) for p in CODE},seconds=300,
        retained_prior=retained,test_target=target,artifacts=128*2**20,as_bytes=8*2**30,policy=POLICY,
        cases=['unset/existing TF success/error and weakref release','old/new L2/L3 exact HZ coefficients',
               'property/excluded next-history release','MemoryError/ValueError missing HZ diagnostics'],
        real_requests=0,gpu=False))
    env=dict(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    receipt=execute([ACT,'-I','-B',__file__,str(child),'--worker'],run_deadline=ENTRY+294-seconds,
        cleanup_deadline=ENTRY+298-seconds,env=env,rss_limit=None,development_policy=POLICY)
    save(folder/'session.json',receipt)
    from b1_observed_batch import session_ok
    good=(session_ok(receipt) and (child/'outcome.json').exists()
          and load(child/'outcome.json')['status']=='PASS' and time.monotonic()<ENTRY+300-seconds)
    roots={folder,PRIOR.parent,prior_path.parent}
    assert sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())<128*2**20
    save(folder/'complete.json',dict(status='PASS' if good else 'FAIL',seconds=seconds+time.monotonic()-ENTRY,
        retained_prior=retained,current_seconds=time.monotonic()-ENTRY,
        freeze_sha256=sha(folder/'freeze.json'),session_sha256=sha(folder/'session.json'),
        outcome_sha256=sha(child/'outcome.json') if (child/'outcome.json').exists() else None))
    print(json.dumps(load(folder/'complete.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--worker',action='store_true')
    p.add_argument('--prior',type=Path)
    a=p.parse_args();main(a.folder,a.worker,a.prior)
