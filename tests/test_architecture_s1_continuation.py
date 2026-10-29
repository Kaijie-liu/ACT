"""Finite continuation controls: no model, attack or bound queries."""
import copy
from pathlib import Path
import unittest

from architecture_s1_continuation import (CAP, ORIGINAL, remaining_requests,
    locate_requests, tree_identity, validate)
from architecture_s1_execution import Artifacts, digest
from architecture_s1_study import all_requests
from scoped_proof.io import load,save,sha

CONTROL_ROOT=None


class Controls(unittest.TestCase):
    def setup_case(self,name):
        root=CONTROL_ROOT/name;root.mkdir()
        prior=root/'old';store=Artifacts.create(prior)
        spec=dict(models=[dict(id='m')],artifact_bytes=128*2**20,request_seconds=300,
            timing_reserve=[12,10,5],workers=8,numerical_method='frozen',inputs={'identity':'fixed'})
        queries=all_requests(spec['models']);spec['request_catalog_sha256']=digest(queries)
        manifest=root/'old_manifest.json';save(manifest,spec)
        for q in queries[:2]:
            store.write(prior/(q['id']+'.ledger.json.gz'),dict(id=q['id'],status='ERROR',
                accepted=False,seconds=.5,controller_error={'kind':'BufferError'}),terminal=True)
        store.write(prior/'queue_complete.json.gz',dict(status='PREFIX_STOP',settled=2,
            not_executed=len(queries)-2,missing_request_cost=0,summed_request_seconds=1.))
        new=copy.deepcopy(spec);new['artifact_bytes']=CAP
        new['request_catalog_sha256']=digest(queries[2:])
        new['continuation']=dict(root=str(prior),manifest=str(manifest),manifest_sha256=sha(manifest),
            tree=tree_identity(prior),prior_charged_bytes=load(store.account)['used'],
            remaining=len(queries)-2,prior_request_seconds=1.,numerical_execution=str(ORIGINAL))
        return root,prior,new,queries

    def test_errors_spent_remaining_order_preserved(self):
        _,prior,spec,queries=self.setup_case('order')
        old,full,remaining=validate(spec)
        self.assertEqual(full,queries)
        self.assertEqual(remaining,queries[2:])
        self.assertNotIn(queries[0],remaining)
        self.assertEqual(old['request_seconds'],300)

    def test_orphan_and_duplicate_refused(self):
        root,prior,spec,queries=self.setup_case('orphans')
        (prior/queries[2]['id']).mkdir()
        with self.assertRaisesRegex(ValueError,'unsettled'):
            remaining_requests(queries,prior)
        other=Artifacts.create(root/'other')
        other.write(other.root/(queries[0]['id']+'.ledger.json.gz'),{'id':queries[0]['id']})
        with self.assertRaisesRegex(ValueError,'duplicate'):
            locate_requests(queries,[prior,other.root])

    def test_deadline_method_source_and_quota_changes_refused(self):
        _,_,spec,_=self.setup_case('binding')
        for key,value in [('request_seconds',301),('numerical_method','new'),
                          ('inputs',{'identity':'other'}),('artifact_bytes',CAP+1)]:
            changed=copy.deepcopy(spec);changed[key]=value
            with self.assertRaises(ValueError):validate(changed)
        changed=copy.deepcopy(spec);changed['request_catalog_sha256']='wrong'
        with self.assertRaises(ValueError):validate(changed)

    def test_old_evidence_tamper_refused(self):
        _,prior,spec,_=self.setup_case('tamper')
        save(prior/'unexpected.json',{'new':'old prefix must be immutable'})
        with self.assertRaisesRegex(ValueError,'prior evidence'):
            validate(spec)

    def test_global_capacity_carries_failed_reservations(self):
        root,prior,spec,_=self.setup_case('capacity')
        # Small proportional cap exercises the exact same Artifacts path.
        charged=spec['continuation']['prior_charged_bytes']
        cap=charged+16384
        store=Artifacts.create(root/'continued',initial_bytes=charged,cap=cap,terminal_reserve=4096)
        self.assertEqual(load(store.account)['used'],charged+4096)
        with self.assertRaises(BufferError):
            store.write(store.root/'oversize.json',{'data':'x'*16384})
        store.write(store.root/'terminal.json.gz',{'status':'ERROR'},terminal=True)
        self.assertGreater(load(store.account)['used'],charged)
        self.assertEqual(tree_identity(prior),spec['continuation']['tree'])

    def test_completed_and_missing_locations(self):
        root,prior,_,queries=self.setup_case('locations')
        store=Artifacts.create(root/'continued')
        store.write(store.root/(queries[2]['id']+'.ledger.json.gz'),{'id':queries[2]['id']})
        found=locate_requests(queries,[prior,store.root])
        self.assertEqual(found,{queries[0]['id']:0,queries[1]['id']:0,queries[2]['id']:1})
        self.assertNotIn(queries[3]['id'],found)
