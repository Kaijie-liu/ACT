"""Finite stdlib contract/dual/tiling controls; no optimizer or native solver."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import time
import unittest

from scoped_source.hz_request_store import Store, Work, identity, evaluate
from scoped_source.hz_store_control_build import build
from scoped_source.check_store_endpoints import (
    SCHEMA, EndpointWork, MAX_VISITS, plan, evaluate_tile, check_tile, aggregate,
)


class Endpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='endpoint-unit-',dir='/data1/Kane/MOE/baseline_runs')
        path=Path(cls.temp.name)/'store'; w=Work(time.monotonic()+300)
        binding=build(path,large=False,work=w)
        cls.store=Store(path,expected_manifest=binding['manifest'],expected_request=binding['request'],work=w)
        cls.doc=dict(schema=SCHEMA,store=cls.store.identity,request=cls.store.request,mode='given_hz',
                     classes=2,label=0,margin='0',semantics='selected-softmax-top2',ties='ANY_LEGAL_TOPK')

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.work=EndpointWork(time.monotonic()+300)
        self.plan=plan(self.store,self.doc,expected_contract=identity(self.doc),declaration=None,work=self.work)
        self.tasks=next(self.plan.tiles(8)); self.records=self.candidates(self.tasks)

    def candidates(self, tasks, nonzero=False):
        candidates=[]
        for t in tasks:
            candidates.append(dict(query=t.query_id,
                equality=['1/8' if nonzero else '0']*sum(1 for _ in self.store.constraints(t.pair,'eq')),
                inequality=['-1/16' if nonzero else '0']*sum(1 for _ in self.store.constraints(t.pair,'ine')),lower='0'))
        bounds,_=evaluate_tile(self.store,tasks,candidates,self.work)
        for c,b in zip(candidates,bounds): c['lower']=str(b)
        return [dict(slot=t.slot,candidate=c) for t,c in zip(tasks,candidates)]

    def test_all_pairs_properties_and_tile_tail(self):
        self.assertEqual((len(self.plan.duties),len(self.plan.tasks)),(3,6))
        self.assertEqual([t.slot for ts in self.plan.tiles(1) for t in ts],
                         [t.slot for ts in self.plan.tiles(8) for t in ts])
        self.assertEqual([len(ts) for ts in self.plan.tiles(1)],[1]*6)

    def test_exact_single_row_differential(self):
        for tasks in self.plan.tiles(8):
            records=self.candidates(tasks,True); candidates=[r['candidate'] for r in records]
            bounds,residual=evaluate_tile(self.store,tasks,candidates,self.work)
            for t,c,b,r in zip(tasks,candidates,bounds,residual):
                old,oldr=evaluate(self.store,t.pair,left=list(t.left),right=list(t.right),offset=t.offset,
                    equality=c['equality'],inequality=c['inequality'],work=self.work)
                self.assertEqual(old,b)
                self.assertEqual({j:v for j,v in oldr if v},{j:v for j,v in r.items() if v})

    def test_explicit_partial_is_unknown(self):
        values={t.slot:None for t in self.plan.tasks}
        result=aggregate(self.plan,values,self.work)
        self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['missing_endpoints'],6)

    def test_missing_inventory_rejected(self):
        with self.assertRaisesRegex(ValueError,'inventory'): aggregate(self.plan,{},self.work)
        with self.assertRaisesRegex(ValueError,'slots'): check_tile(self.store,self.tasks,self.records[:-1],self.work)

    def test_duplicate_and_reversed_endpoint_rejected(self):
        for records in ([self.records[0],self.records[0]],list(reversed(self.records))):
            with self.assertRaisesRegex(ValueError,'slot'): check_tile(self.store,self.tasks,records,self.work)

    def test_wrong_pair_or_request_query_rejected(self):
        for q in (self.plan.tasks[-1].query_id,'f'*64):
            records=deepcopy(self.records); records[0]['candidate']['query']=q
            with self.assertRaisesRegex(ValueError,'binding'): check_tile(self.store,self.tasks,records,self.work)

    def test_dual_sign_and_inventory(self):
        for values in (['1']*len(self.records[0]['candidate']['inequality']),[]):
            records=deepcopy(self.records); records[0]['candidate']['inequality']=values
            with self.assertRaises(ValueError): check_tile(self.store,self.tasks,records,self.work)

    def test_overstated_bound_and_nonfinite(self):
        for value in ('999','nan','1.0'):
            records=deepcopy(self.records); records[0]['candidate']['lower']=value
            with self.assertRaises(ValueError): check_tile(self.store,self.tasks,records,self.work)

    def test_external_label_margin_and_source_binding(self):
        for key,value in (('label',1),('margin','1/10'),('store','f'*64),('request','e'*64),('ties','FIRST_ONLY')):
            doc=dict(self.doc,**{key:value})
            with self.assertRaisesRegex(ValueError,'contract'):
                plan(self.store,doc,expected_contract=identity(self.doc),declaration=None,work=self.work)
        changed=dict(self.doc,label=1)
        p=plan(self.store,changed,expected_contract=identity(changed),declaration=None,work=self.work)
        with self.assertRaisesRegex(ValueError,'slot'): check_tile(self.store,next(p.tiles(8)),self.records,self.work)

    def test_dimensions_and_extra_fields(self):
        for changes in ({'classes':3},{'label':True},{'margin':'-1'},{'gate':['1/2','1/2']}):
            doc=dict(self.doc,**changes)
            with self.assertRaises(ValueError): plan(self.store,doc,expected_contract=identity(doc),declaration=None,work=self.work)

    def test_given_cannot_promote_to_source(self):
        doc=dict(self.doc,mode='checked_source')
        with self.assertRaises(ValueError): plan(self.store,doc,expected_contract=identity(doc),declaration=None,work=self.work)

    def test_tile_bounds_and_deadline(self):
        for width in (0,9,True):
            with self.assertRaises(ValueError): list(self.plan.tiles(width))
        self.work.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): check_tile(self.store,self.tasks,self.records,self.work)

    def test_no_per_tile_work_reset(self):
        self.work.counts['arithmetic']=MAX_VISITS
        with self.assertRaisesRegex(ValueError,'work capacity'): check_tile(self.store,self.tasks,self.records,self.work)

    def test_caller_lists_are_captured_before_arithmetic(self):
        records=deepcopy(self.records); expected=check_tile(self.store,self.tasks,self.records,self.work)
        original=self.work.tick; triggered=[]
        def tick():
            original()
            if not triggered:
                triggered.append(True); records[0]['candidate']['lower']='999'
        self.work.tick=tick
        self.assertEqual(check_tile(self.store,self.tasks,records,self.work),expected)
        self.assertTrue(triggered)

    def test_offset_once(self):
        doc=dict(self.doc,margin='1/4'); p=plan(self.store,doc,expected_contract=identity(doc),declaration=None,work=self.work)
        tasks=next(p.tiles(8)); result=check_tile(self.store,tasks,self.candidates(tasks),self.work)
        old=check_tile(self.store,self.tasks,self.records,self.work)
        self.assertEqual(list(result.values()),[v-F(1,4) for v in old.values()])


if __name__=='__main__': unittest.main()
