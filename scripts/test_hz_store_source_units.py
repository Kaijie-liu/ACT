"""Small exact source-contract regressions, no solver or numerical package."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Row,Rows,View,Work,identity,compact
from scoped_source import check_hz_store_source as c


def sequence(rows=(),parent=None):
    rows=tuple(rows)
    return Rows(parent,(rows,) if rows else (),len(rows)+(parent.count if parent else 0))


class SourceViews(unittest.TestCase):
    def work(self): return Work(time.monotonic()+300)

    def relu(self):
        empty=sequence(); source=View('input','common',None,None,1,frozenset({0}),
            sequence([Row('input/output/0',F(1,2**54),((0,F(1)),))]),empty,empty,1)
        factors=[('input/c/0','continuous','common','input')]
        factors += [(f'r/{name}',kind,'common','r') for name,kind in
                    [('negative/0','continuous'),('positive/0','continuous'),('sign/0','binary'),
                     ('lift/output/0','continuous'),('lift/eq/0','continuous')]]
        target=View('r','common',None,source,1,frozenset(range(6)),
            sequence([Row('r/output/0',F(1,2),((2,F(-1,2)),(4,F(1,2**54))))]),
            sequence([Row('r/eq/0',F(-1,2),((0,F(-1)),(1,F(-1,2)),(2,F(-1,2)),(3,F(-1,2)),(5,F(1,2**53))))],empty),
            sequence([Row('r/ine/0',F(0),((1,F(-1)),(3,F(-1)))),
                      Row('r/ine/1',F(0),((2,F(-1)),(3,F(1))))],empty),2)
        return SimpleNamespace(factors=tuple(factors)),source,target

    def test_nonbinary_relu_reference_is_enclosed(self):
        store,source,target=self.relu()
        self.assertEqual(c.check_relu(store,source,target,self.work()),dict(active=0,inactive=0,unstable=1))

    def test_output_compensation_is_not_original_basis(self):
        store,source,target=self.relu(); row=next(iter(target.output))
        bad=replace(target,output=sequence([replace(row,coefficients=((2,F(-1,2)),))]))
        with self.assertRaisesRegex(ValueError,'missing/inward source enclosure'):
            c.check_relu(store,source,bad,self.work())

    def test_output_compensation_cannot_contaminate_equalities(self):
        store,source,target=self.relu(); row=next(iter(target.eq))
        coeff=tuple(sorted((*row.coefficients,(4,F(1)))))
        bad=replace(target,eq=sequence([replace(row,coefficients=coeff)],source.eq))
        with self.assertRaisesRegex(ValueError,'missing/inward source enclosure'):
            c.check_relu(store,source,bad,self.work())

    def test_old_constraints_are_not_replaced(self):
        store,source,target=self.relu(); bad=replace(target,eq=sequence(list(target.eq),sequence()))
        with self.assertRaisesRegex(ValueError,'inherited constraints'): c.check_relu(store,source,bad,self.work())

    def test_factor_owner_and_kind_bound(self):
        store,source,target=self.relu()
        for index,field,value in ((3,1,'continuous'),(1,3,'foreign'),(4,2,'expert/0')):
            factors=[list(f) for f in store.factors]; factors[index][field]=value
            with self.assertRaisesRegex(ValueError,'factor allocation'):
                c.check_relu(SimpleNamespace(factors=tuple(map(tuple,factors))),source,target,self.work())

    def test_all_rows_even_zero_duals_are_required(self):
        store,source,target=self.relu(); bad=replace(target,ine=sequence([],source.ine))
        with self.assertRaisesRegex(ValueError,'row inventory'): c.check_relu(store,source,bad,self.work())

    def test_binary_target_required(self):
        store,source,target=self.relu(); row=next(iter(target.output))
        bad=replace(target,output=sequence([replace(row,rhs=F(1,3))]))
        with self.assertRaisesRegex(ValueError,'not exact binary64'): c.check_relu(store,source,bad,self.work())

    def test_input_contains_declared_rationals(self):
        empty=sequence(); source=View('input','common',None,None,1,frozenset({0}),
            sequence([Row('input/output/0',F(1,2),((0,F(1,4)),))]),empty,empty,1)
        store=SimpleNamespace(views={'input':source},factors=(('input/c/0','continuous','common','input'),))
        self.assertIs(c.check_input(store,['1/3'],['2/3'],self.work()),source)
        with self.assertRaisesRegex(ValueError,'input enclosure inward'): c.check_input(store,['0'],['1'],self.work())

    def test_capture_binds_actual_bytes_not_second_serialization(self):
        doc={'a':1}; intended={'a':2}
        def changed(value):
            raw=compact(value); doc['a']=2; return raw
        with patch.object(c,'compact',side_effect=changed),self.assertRaisesRegex(ValueError,'source declaration identity'):
            c.capture(doc,identity(intended),self.work())

    def test_captured_data_not_caller_alias(self):
        doc={'a':[1]}; captured=c.capture(doc,identity(doc),self.work());doc['a'][0]=2
        self.assertEqual(captured,{'a':[1]})

    def test_deadline_and_bits_not_dropped(self):
        store,source,target=self.relu(); work=self.work();work.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):c.check_relu(store,source,target,work)
        with self.assertRaisesRegex(ValueError,'intermediate rational capacity'):self.work().exact(F(1<<8192))


if __name__=='__main__': unittest.main()
