"""Small source controls. Never starts the fixed dense run or a solver."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Work, compact, digest
from scoped_source.tensor_row_controls import fixture
from scoped_source.source_dag import build
from scoped_source.check_source_dag import check
from scoped_source.source_dag_controls import mixed, old_differential


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE'); cls.base=Path(cls.tmp.name)
        cls.root=cls.base/'source'; cls.source=fixture(cls.root,chunk_bytes=40)
        cls.raw,_=build(cls.root,cls.source,work=Work(time.monotonic()+300))
        cls.doc=json.loads(cls.raw)
        cls.mix,cls.ms=mixed(cls.base/'builder')
        cls.mraw,_=build(cls.mix,cls.ms,work=Work(time.monotonic()+300)); cls.md=json.loads(cls.mraw)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def verify(self,doc=None,*,mixed=False,cut=None,work=None):
        raw=compact(doc) if doc is not None else self.mraw if mixed else self.raw
        return check(raw,self.mix if mixed else self.root,self.ms if mixed else self.source,
                     expected=digest(raw),work=work or Work(time.monotonic()+300),cut=cut)

    def test_complete_and_old_compensation_differential(self):
        self.assertEqual(self.verify()['status'],'CHECKED_SOURCE_FOREST')
        result=old_differential(self.root,self.source,self.raw,self.base/'oldsmall',Work(time.monotonic()+300))
        self.assertGreater(result['checked_expanded_roots'],20)
        names=[f[0] for f in self.doc['factors']]
        self.assertTrue(any('/error/' in n for n in names)); self.assertTrue(any('/lift/output/' in n for n in names))

    def test_mixed_ties_private_and_old_differential(self):
        self.verify(mixed=True)
        self.assertEqual([p[:2] for p in self.md['roles']['pairs']],[[0,1],[0,2],[1,2]])
        self.assertEqual({f[2] for f in self.md['factors']},{-1,0,1,2})
        self.assertTrue(any('/lift/eq/' in f[0] for f in self.md['factors']))
        result=old_differential(self.mix,self.ms,self.mraw,self.base/'oldmixed',Work(time.monotonic()+300))
        self.assertEqual(result['checked_expanded_roots'],91)

    def test_expert_first_relu_is_private(self):
        root,source=mixed(self.base/'firstrelu',first_relu=True)
        w=Work(time.monotonic()+300); raw,_=build(root,source,work=w)
        check(raw,root,source,expected=digest(raw),work=w)
        d=json.loads(raw)
        for e in range(3):
            vi=next(i for i,v in enumerate(d['views']) if v['name']==f'expert{e}/0')
            self.assertEqual(d['views'][vi]['scope'],e)
            self.assertTrue(any(f[3]==vi for f in d['factors']))
            self.assertTrue(all(f[2]==e for f in d['factors'] if f[3]==vi))

    def test_prefix_cannot_claim_full(self):
        cut=['expert0',2]; raw,_=build(self.root,self.source,work=Work(time.monotonic()+300),cut=cut)
        d=json.loads(raw); r=self.verify(d,cut=cut)
        self.assertEqual(r['status'],'CHECKED_SOURCE_PREFIX')
        self.assertEqual(r['source_receipt'],self.doc['source_receipt'])
        with self.assertRaises(ValueError): self.verify(d)
        d['cut']=None
        with self.assertRaises((ValueError,IndexError)): self.verify(d)

    def test_identity_and_frame(self):
        with self.assertRaises(ValueError): check(self.raw,self.root,'0'*64,expected=digest(self.raw),work=Work(time.monotonic()+300))
        for key,value in [('frame',2),('source','0'*64),('request','0'*64)]:
            d=copy.deepcopy(self.doc); d[key]=value
            with self.assertRaises(ValueError): self.verify(d)

    def test_omissions_extra_roles_and_roots(self):
        for change in (lambda d:d['views'].pop(),lambda d:d['nodes'].pop(),lambda d:d['factors'].pop(),
                       lambda d:d['roles']['pairs'].clear(),lambda d:d['views'][2]['output'].pop(),
                       lambda d:d['views'].append(copy.deepcopy(d['views'][0]))):
            d=copy.deepcopy(self.doc); change(d)
            with self.assertRaises((ValueError,IndexError)): self.verify(d)

    def test_inactive_roots_and_ancestral_inventory(self):
        relu=next(i for i,v in enumerate(self.md['views']) if v['name']=='expert0/2')
        for vi,ri in ((relu,1),(relu,3),(relu-1,1)):
            d=copy.deepcopy(self.md); d['views'][vi]['output'].pop(ri)
            with self.assertRaises(ValueError): self.verify(d,mixed=True)

    def test_relu_rhs_ref_sign_and_compensation(self):
        relu=next(i for i,v in enumerate(self.md['views']) if v['name']=='expert0/2')
        row=self.md['views'][relu]['eq'][0]; ni=row[1]
        for change in (lambda d:d['views'][relu]['eq'][0].__setitem__(0,'99'),
                       lambda d:d['nodes'][ni][2][0].__setitem__(1,'1'),
                       lambda d:d['nodes'][ni][1].pop(),
                       lambda d:d['nodes'][ni][2][0].__setitem__(0,ni)):
            d=copy.deepcopy(self.md); change(d)
            with self.assertRaises(ValueError): self.verify(d,mixed=True)

    def test_affine_enclosure_missing_or_changed(self):
        ai=next(i for i,v in enumerate(self.doc['views']) if v['op']=='Linear'); ni=self.doc['views'][ai]['output'][0][1]
        for change in (lambda d:d['nodes'][ni][1].clear(),lambda d:d['views'][ai]['output'][0].__setitem__(0,'1'),
                       lambda d:d['factors'].reverse()):
            d=copy.deepcopy(self.doc); change(d)
            with self.assertRaises(ValueError): self.verify(d)

    def test_cross_expert_scope_and_node_binding(self):
        for change in (lambda d:d['factors'][-1].__setitem__(2,0),
                       lambda d:d['views'][-2].__setitem__('parent',1),
                       lambda d:d['nodes'][-1].__setitem__(0,0)):
            d=copy.deepcopy(self.md); change(d)
            with self.assertRaises(ValueError): self.verify(d,mixed=True)

    def test_guard_sign_and_missing_pair(self):
        d=copy.deepcopy(self.md); pair=d['roles']['pairs'][0][2]; n=d['views'][pair]['ine'][0][1]
        d['views'][pair]['ine'][0][0]='-1'
        with self.assertRaises(ValueError): self.verify(d,mixed=True)
        d=copy.deepcopy(self.md)
        self.assertTrue(d['nodes'][n][1])
        d['nodes'][n][1][0][1]=str(-F(d['nodes'][n][1][0][1]))
        with self.assertRaises(ValueError): self.verify(d,mixed=True)
        d=copy.deepcopy(self.md); d['roles']['pairs'].pop()
        with self.assertRaises(ValueError): self.verify(d,mixed=True)

    def test_suffix_receipt_and_source_bytes(self):
        d=copy.deepcopy(self.doc); d['source_receipt']['tensor_hashes'].popitem()
        with self.assertRaises(ValueError): self.verify(d)
        p=self.root/'t/00000-00000.bin'; original=p.read_bytes()
        try:
            p.write_bytes(bytes([original[0]^1])+original[1:])
            with self.assertRaises(ValueError): self.verify()
        finally: p.write_bytes(original)

    def test_prefix_checks_unlowered_suffix_bytes(self):
        cut=['expert0',2]; raw,_=build(self.root,self.source,work=Work(time.monotonic()+300),cut=cut)
        manifest=json.loads((self.root/'manifest.json').read_bytes())
        t=next(t for t in manifest['tensors'] if t['name']=='experts.1.1.weight')
        p=self.root/t['chunks'][0]['file']; original=p.read_bytes()
        try:
            p.write_bytes(bytes([original[0]^1])+original[1:])
            with self.assertRaises(ValueError): self.verify(json.loads(raw),cut=cut)
        finally: p.write_bytes(original)

    def test_deadline_and_work(self):
        w=Work(time.monotonic()+300); w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): self.verify(work=w)
        w=Work(time.monotonic()+300); w.counts['arithmetic']=8000000
        with self.assertRaisesRegex(ValueError,'work capacity'): self.verify(work=w)

    def test_checker_does_not_call_producer_or_native(self):
        with patch('scoped_source.source_dag.Builder.finish',side_effect=AssertionError('producer')):
            self.verify()

    def test_prefix_alias_captured_before_later_ticks(self):
        cut=['expert0',2]; raw,_=build(self.root,self.source,work=Work(time.monotonic()+300),cut=cut)
        class Mutate(Work):
            def visit(s,kind,amount=1):
                super().visit(kind,amount)
                if cut[1]==2: cut[1]=1
        result=check(raw,self.root,self.source,expected=digest(raw),work=Mutate(time.monotonic()+300),cut=cut)
        self.assertEqual(result['cut'],['expert0',2]); self.assertEqual(cut,['expert0',1])

    def test_stored_bits_and_zero_entries(self):
        for value in ('0','1/'+str(2**4096)):
            d=copy.deepcopy(self.doc); d['nodes'][0][1][0][1]=value
            with self.assertRaises(ValueError): self.verify(d)


if __name__=='__main__': unittest.main()
