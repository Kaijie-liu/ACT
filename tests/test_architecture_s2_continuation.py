"""No model calls: preserved errors, source identity and cumulative accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from architecture_s1_execution import Artifacts,digest,read_record
from architecture_s1_continuation import remaining_requests,locate_requests
from architecture_s2_continuation import freeze,validate,ROOT,CHANGED_SOURCES
from architecture_s2_audit import analyze
from architecture_s2_study import ARMS,all_requests
from act.pipeline.moe.architecture_s2_evaluation import pack_arrays
from scoped_proof.io import load,save,sha

CONTROL_ROOT=None


class ContinuationControls(unittest.TestCase):
    def setUp(self):
        self.base=Path(tempfile.mkdtemp(dir=CONTROL_ROOT))
        self.prior=self.base/'prior';self.store=Artifacts.create(self.prior)
        inputs=self.prior/'inputs.json.gz'
        ih=self.store.write(inputs,pack_arrays(dict(
            pixels=np.zeros((10000,3,32,32),dtype=np.uint8),labels=np.zeros(10000,dtype=np.int64))))
        models=[dict(id=f'{a}_seed{s}',arm=a,seed=s,family='moe' if a in ('M','M-reg') else 'dense')
                for a in ARMS for s in range(5)]
        self.full=all_requests(models);self.original=self.base/'original';sources={}
        for p in CHANGED_SOURCES:
            f=self.original/p;f.parent.mkdir(parents=True,exist_ok=True)
            f.write_text('old synthetic source, never executed\n');sources[p]=sha(f)
        spec=dict(study='S2A',models=models,source_files=sources,
            inputs=dict(path=str(inputs),sha256=ih),request_catalog_sha256=digest(self.full),
            bootstrap=dict(resamples=10000,seed=20261031),artifact_bytes=128*2**20,
            preparation_seconds=1.,seconds=300,workers=8)
        mh=self.store.write(self.prior/'manifest.json',spec)
        self.store.write(self.prior/'start.json.gz',dict(manifest_sha256=mh,startup_and_admission_seconds=.5))
        self.failed=self.full[20]
        self.store.write(self.prior/(self.failed['id']+'.ledger.json.gz'),dict(
            id=self.failed['id'],status='ERROR',accepted=False,controller_error='fixture',seconds=3.))
        self.store.write(self.prior/'queue_complete.json.gz',dict(status='PREFIX_STOP',
            source_manifest_sha256=mh,request_catalog_sha256=digest(self.full),settled=1,
            not_executed=len(self.full)-1,states={'ERROR':1},summed_request_seconds=3.,
            missing_request_cost=0,queue_wall_seconds=4.))
        control=self.base/'control.json';save(control,dict(status='PASS',synthetic=True))
        self.manifest=self.base/'continuation.json'
        freeze(self.manifest,self.prior,self.original,control)
        self.spec=load(self.manifest)

    def test_error_is_spent_and_order_unchanged(self):
        old,full,remaining=validate(self.spec)
        self.assertEqual(full,self.full)
        self.assertEqual(remaining,self.full[:20]+self.full[21:])
        self.assertNotIn(self.failed,remaining)

    def test_methods_budget_and_source_changes_reject(self):
        for field,value in [('seconds',301),('artifact_bytes',512*2**20),('workers',9)]:
            s=copy.deepcopy(self.spec);s[field]=value
            with self.assertRaises(ValueError):validate(s)
        s=copy.deepcopy(self.spec);s['models'][0]['seed']=9
        with self.assertRaises(ValueError):validate(s)
        s=copy.deepcopy(self.spec);s['source_files']['extra.py']='unbound'
        with self.assertRaises(ValueError):validate(s)

    def test_prior_mutation_and_orphan_reject(self):
        (self.prior/'unbound.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'prior evidence'):validate(self.spec)
        orphan=self.full[21];(self.prior/orphan['id']).mkdir()
        with self.assertRaisesRegex(ValueError,'unsettled attempt'):
            remaining_requests(self.full,self.prior)

    def test_duplicate_execution_is_not_a_replacement(self):
        other=self.base/'duplicate';store=Artifacts.create(other)
        store.write(other/(self.failed['id']+'.ledger.json.gz'),dict(id=self.failed['id'],status='POSITIVE'))
        with self.assertRaisesRegex(ValueError,'duplicate request'):
            locate_requests(self.full,[self.prior,other])

    def test_aggregate_keeps_failure_missing_denominators_and_all_costs(self):
        _,_,remaining=validate(self.spec)
        root=self.base/'continued';store=Artifacts.create(root,
            initial_bytes=self.spec['continuation']['prior_charged_bytes'])
        mh=store.write(root/'manifest.json',self.spec)
        store.write(root/'start.json.gz',dict(manifest_sha256=mh,startup_and_admission_seconds=.25))
        q=remaining[0]
        store.write(root/(q['id']+'.ledger.json.gz'),dict(id=q['id'],status='ERROR',
            accepted=False,controller_error='fixture continued',seconds=2.))
        store.write(root/'queue_complete.json.gz',dict(status='PREFIX_STOP',
            source_manifest_sha256=mh,request_catalog_sha256=digest(remaining),settled=1,
            not_executed=len(remaining)-1,states={'ERROR':1},summed_request_seconds=2.,
            missing_request_cost=0,queue_wall_seconds=3.))
        v=analyze(root)
        self.assertEqual(v['online_states'],{'ERROR':2,'NOT_EXECUTED':24018})
        self.assertEqual(v['planned_requests'],24020)
        self.assertEqual(v['charged_request_seconds'],5.)
        self.assertEqual(v['queue_wall_seconds'],7.)
        self.assertEqual(v['controller_startup_and_admission_seconds'],.75)
        self.assertEqual(v['registered_decision'],'PENDING_COMPLETE_S2A_EVIDENCE')
        self.assertEqual(v['artifact_charged_bytes'],load(root/'artifact_account.json')['used'])
        account=load(root/'artifact_account.json');account['initial_bytes']=0
        # Deliberately corrupt only this disposable negative-control fixture.
        (root/'artifact_account.json').write_text(json.dumps(account))
        with self.assertRaisesRegex(ValueError,'all old charges'):analyze(root)


if __name__=='__main__':unittest.main()
