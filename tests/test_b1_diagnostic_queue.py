"""Ordinary queue, identity and numerical mapping tests; no real models/solves."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('b1_queue',ROOT/'scripts/b1_diagnostic_queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def source_scope(ordinal=0):
    return dict(model='model',domain='domain',request='request',config='config',
                roster='roster',ordinal=ordinal,property_index=0,pairs=[[0,1]],path='pair')


class QueueControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE');self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def save(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return q.sha(path)

    def online_fixture(self):
        online=self.root/'online';items=[];scope=source_scope()
        binding={k:scope[k] for k in ('model','domain','request','config')}
        for n in range(41):
            item=dict(ordinal=n,job_id=f'{n:02d}',subject={'model_state':{'sha256':'model'}})
            items.append(item);folder=online/item['job_id']
            self.save(folder/'terminal.json',{'status':'UNKNOWN'})
            self.save(folder/'reception.json',{'accepted':True})
            inventory=dict(binding=binding,full_capture_events=[])
            if n==0:
                for j in range(10):
                    scoped=source_scope(j)
                    payload={'scope':scoped};raw=json.dumps(payload).encode()
                    digest=hashlib.sha256(raw).hexdigest()
                    if j!=1:self.save(folder/f'diagnostics/000_native_{digest}.json',payload)
                    inventory['full_capture_events'].append(dict(query=j,property_index=0,pair=[0,1],
                        sha256=digest,status='CAPTURED' if j!=1 else 'MISSING_CAPTURE_CAPACITY',
                        receipt={'scope':scoped}))
            self.save(folder/'diagnostics/000_inventory_last.json',inventory)
        freeze=self.root/'freeze.json';digest=self.save(freeze,dict(requests=items))
        for item in items:self.save(online/item['job_id']/'launch.json',dict(item=item,freeze_sha256=digest))
        self.save(online/'start.json',dict(prior_seconds=7.,freeze_sha256=digest))
        self.save(online/'online_complete.json',dict(completed_calls=41,status='COMPLETE',
                  stage_elapsed=100.,online_seconds=95.,cumulative_online_seconds=102.))
        return online,freeze

    def test_fixed_catalog_missing_second_slot_and_prefix_cost(self):
        online,freeze=self.online_fixture();plan=q.make_plan(online,freeze)
        self.assertEqual(len(plan['queries']),145)
        self.assertEqual(sum(r['seconds'] for r in plan['queries']),7380)
        self.assertEqual(plan['local_seconds_cap'],21600-107)
        self.assertIsNone(plan['sources'][1]['path'])
        slices=[r for r in plan['queries'] if r['kind']=='slice_endpoint']
        self.assertEqual(len(slices),64)
        self.assertEqual({r['source']['slot'] for r in slices},{0,1})
        self.assertEqual(plan['extended'],[])

    def test_wrong_freeze_is_not_tooling_missingness(self):
        online,freeze=self.online_fixture()
        value=json.loads(freeze.read_text());value['changed']=True;self.save(freeze,value)
        with self.assertRaises(ValueError):q.make_plan(online,freeze)

    def test_extended_eligibility_not_unshifted_objective(self):
        value=dict(decision={'minimum':None},mip_dual_bound_with_offset=-.2,
                   raw_candidate_objective=.1,native_result={'mip_dual_bound':5.})
        self.assertTrue(q.extension_eligibility(value)['eligible'])
        for lower,candidate in ((None,.1),(1.,.1),(-1.,-.1),(-1.,float('nan'))):
            value.update(mip_dual_bound_with_offset=lower,raw_candidate_objective=candidate)
            self.assertFalse(q.extension_eligibility(value)['eligible'])

    def test_same_request_property_binding(self):
        scope=source_scope();binding={k:scope[k] for k in ('model','domain','request','config')}
        source=dict(binding=binding,event={'receipt':{'scope':scope},'query':0,'property_index':0,'pair':[0,1]})
        job=dict(binding=binding,item={'subject':{'model_state':{'sha256':'model'}}})
        q.bind_source(NS(value={'scope':scope}),source,job)
        bad=dict(scope,domain='another')
        with self.assertRaises(ValueError):q.bind_source(NS(value={'scope':bad}),source,job)
        wrong=dict(job,binding=dict(binding,request='another'))
        with self.assertRaises(ValueError):q.bind_source(NS(value={'scope':scope}),source,wrong)

    def problem(self,point):
        hz=NS(n_out=1,n_cont=1,n_bin=1,frame_id=17,c=np.array([2.]),
              Gc=sp.csr_matrix([[3.]]),Gb=sp.csr_matrix([[5.]]))
        return NS(value={'candidate':'RAW_UNCHECKED','scope':{'path':'pair'}},
                  array=lambda _:np.array(point),hz=lambda _:hz,
                  native=dict(variables=4,n_cont=3,n_bin=1))

    def test_binary_uses_native_not_input_offset(self):
        value,missing=q.mapped_candidate(self.problem([.5,0.,0.,1.]),(1,1))
        self.assertIsNone(missing);self.assertEqual(value.tolist(),[[8.5]])
        _,missing=q.mapped_candidate(self.problem([.5,0.,1.]),(1,1))
        self.assertEqual(missing,'NATIVE_CANDIDATE_SHAPE')

    def test_fractional_selector_is_not_rounded(self):
        p=NS(value={'candidate':'RAW_UNCHECKED','scope':{'path':'monolithic'}},
             array=lambda _:np.array([.1,.5,.5]),
             native=dict(variables=3,selector_offset=1,branches=[{},{}]))
        self.assertEqual(q.mapped_candidate(p,(1,1))[1],'NONUNIT_SELECTOR_NOT_PROJECTED')

    def test_authorized_cutoff_not_generic_error(self):
        folder=self.root/'cutoff';folder.mkdir()
        self.assertFalse(q.cutoff_terminal({'status':'ERROR','returncode':124},folder))
        (folder/'native.log').write_text('B1_DIAGNOSTIC_AUTHORIZED_CUTOFF\n')
        self.assertTrue(q.cutoff_terminal({'status':'ERROR','returncode':124},folder))
        self.assertFalse(q.cutoff_terminal({'status':'ERROR','returncode':1},folder))

    def test_descriptive_tags_do_not_claim_unsafe_or_unique_relaxation_cause(self):
        value=dict(native_result={'status':1},decision={'minimum':None},
                   mip_dual_bound_with_offset=1.,raw_candidate_objective=2.,
                   point={'gate_consistency':1e-16,'product_gap':-.01},replay={'valid':False})
        tags=q.observation_tags(value)
        self.assertIn('SOLVER_LIMIT',tags)
        self.assertIn('RAW_POSITIVE_DUAL_NOT_ACCEPTED_BY_STATUS_POLICY',tags)
        self.assertNotIn('NEGATIVE_CANDIDATE_NOT_CHECKED_FEASIBLE',tags)
        self.assertFalse(any('UNSAFE' in t or 'UNIQUE' in t for t in tags))


if __name__=='__main__':unittest.main()
