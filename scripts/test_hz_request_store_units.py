"""Small regression controls; no solver, numerical library or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source import hz_request_store as c
from scoped_source.hz_store_control_build import build
from scoped_source.check_hz_store_control import records, oracle
from scripts.test_hz_request_store import Suite

ROOT=Path(__file__).resolve().parents[1]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='hz-store-unit-',dir=ROOT)
        self.root=Path(self.temp.name); self.addCleanup(self.temp.cleanup)
        self.ids=build(self.root/'store',large=False,work=self.work())
        self.store=c.Store(self.root/'store',expected_manifest=self.ids['manifest'],expected_request=self.ids['request'],work=self.work())
        self.base=records('small',**self.ids)[0]

    def work(self): return c.Work(time.monotonic()+300)

    def receive(self,item=None,work=None):
        item=self.base if item is None else item
        return c.check_candidate(self.store,item['query'],item['candidate'],expected_query=item['expected_query'],work=work or self.work())

    def test_complete_formula_and_shared_residual(self):
        for i,item in enumerate(records('small',**self.ids)):
            self.assertEqual(self.receive(item),oracle('small',item['query']['pair'],bool(i%2)))
        lower,residual=self.receive(); self.assertEqual(str(lower),'-4')
        self.assertTrue(all(v==0 for j,v in residual if j<4))

    def test_rows_retain_identity(self):
        common=list(self.store.common.eq)
        for pair in self.store.entries:
            rows=list(self.store.constraints(pair,'eq'))
            self.assertIs(rows[0],common[0]); self.assertEqual(len(rows),3)
            self.assertEqual(len(self.store.pair(pair)[3]),8)

    def test_seal_cannot_be_deleted(self):
        for name in ('_sealed','common','templates'):
            with self.assertRaises(AttributeError): delattr(self.store,name)
        with self.assertRaises(AttributeError): self.store.templates=()
        self.receive()

    def test_candidate_is_privately_captured(self):
        item=copy.deepcopy(self.base)
        class Mutate(c.Work):
            def visit(self,kind,amount=1):
                item['query']['left'][0]='100'; item['candidate']['lower']='1000'
                super().visit(kind,amount)
        self.assertEqual(self.receive(item,Mutate(time.monotonic()+300)),oracle('small',(0,1),False))

    def test_external_property_and_claim_binding(self):
        for section,key,value,reason in [('query','offset','1','external binding'),
            ('candidate','lower','1','claimed bound'),('candidate','query','0'*64,'candidate query binding')]:
            item=copy.deepcopy(self.base); item[section][key]=value
            with self.assertRaisesRegex(ValueError,reason): self.receive(item)

    def test_malformed_vectors_and_complete_duals(self):
        for section,key,value,reason in [('candidate','equality',[],'dual row coverage'),
            ('candidate','inequality',[{}],'capacity/type'),('query','pair',[True,1],'pair type')]:
            item=copy.deepcopy(self.base); item[section][key]=value
            with self.assertRaisesRegex(ValueError,reason): self.receive(item)

    def test_limits_and_deadline_remain(self):
        with self.assertRaises(TimeoutError): c.Work(time.monotonic()-1)
        with self.assertRaisesRegex(ValueError,'intermediate rational capacity'): self.work().exact(c.F(1<<8192))
        item=copy.deepcopy(self.base); item['candidate']['inequality']=['0']*(3*c.POLICY['view_rows']+1)
        with self.assertRaisesRegex(ValueError,'vector capacity'): self.receive(item)

    def test_failure_receipt_does_not_repeat_failed_admission(self):
        suite=Suite(self.root/'hz_request_store_20261012_r_fault')
        suite.started=time.monotonic()-1801; suite.error='intentional deadline control'
        with patch.object(suite,'envelope',side_effect=RuntimeError('failed gate must not rerun')):
            receipt=suite.finish()
        self.assertEqual(receipt['status'],'INCOMPLETE')
        self.assertEqual(json.loads((suite.root/'report.json').read_text())['error'],suite.error)

    def test_stage_bytes_cover_other_attempts(self):
        suite=Suite(self.root/'hz_request_store_20261012_r_new')
        sibling=self.root/'hz_request_store_20261012_r_old'; sibling.mkdir()
        (sibling/'retained.bin').write_bytes(b'x'*17)
        self.assertEqual(suite.stage_bytes(),17)


if __name__=='__main__': unittest.main()
