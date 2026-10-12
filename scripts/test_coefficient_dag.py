"""Finite exact controls, no native solver or GPU."""
import copy
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import sys
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scoped_source.coefficient_dag import Graph, query, evaluate, check_candidate
from scoped_source.coefficient_dag_controls import small, expanded, REQUEST
from scoped_source.hz_request_store import Work, compact, digest, identity


class Controls(unittest.TestCase):
    def setUp(self):
        self.work = Work(time.monotonic()+300)
        self.doc, self.rows = small()

    def graph(self, doc=None, **kwargs):
        doc = self.doc if doc is None else doc; raw = compact(doc)
        args = dict(expected_graph=digest(raw),expected_request=REQUEST,expected_source=doc['source'],work=self.work)
        args.update(kwargs); return Graph(raw,**args)

    def bound(self, graph=None):
        g = graph or self.graph(); q = query(g,['1','-1','2','-3'],'1/8')
        duals = dict(eq=['1/4'],ine=['-1/2','-1/4'])
        return evaluate(g,q,duals,expected_query=identity(q),work=self.work),q,duals

    def test_all_three_pairs_expanded_differential(self):
        for pair in combinations(range(3),2):
            doc,rows=small(pair); g=self.graph(doc); bound,q,duals=self.bound(g)
            lower,constant,residual=expanded(rows,q['weights'],duals)
            self.assertEqual((bound.lower,bound.constant,dict(bound.residual)),(lower,constant,residual))

    def test_alias_and_input_mutation(self):
        g=self.graph(); expected=self.bound(g)[0]
        self.doc['nodes'][0][2][0][1]='999'
        self.assertEqual(self.bound(g)[0],expected)
        with self.assertRaises(AttributeError): g.frame=99
        with self.assertRaises(TypeError): g.stats['nodes']=0

    def test_rhs_independent_of_reference(self):
        result,q,duals=self.bound()
        lower,const,residual=expanded(self.rows,q['weights'],duals)
        self.assertEqual(result.constant,const)
        self.assertEqual(result.lower,lower)
        bad=copy.deepcopy(self.rows); bad[4]=('eq','1/4',bad[4][2])
        self.assertNotEqual(result.constant,expanded(bad,q['weights'],duals)[1])

    def test_shared_cancellation_private_no_cancellation(self):
        g=self.graph(); q=query(g,['1','-1','0','0'])
        result=evaluate(g,q,dict(eq=['0'],ine=['0','0']),expected_query=identity(q),work=self.work)
        self.assertTrue(all(v==0 for _,v in result.residual)); self.assertEqual(result.lower,F(-1))
        q=query(g,['0','0','1','-1'])
        result=evaluate(g,q,dict(eq=['0'],ine=['0','0']),expected_query=identity(q),work=self.work)
        self.assertEqual(dict(result.residual)[2],F(-1,2))
        self.assertEqual(dict(result.residual)[3],F(-1,2))

    def test_candidate_binding_and_overclaim(self):
        g=self.graph(); result,q,d=self.bound(g)
        candidate=dict(query=identity(q),**d,lower=str(result.lower))
        self.assertEqual(check_candidate(g,q,candidate,expected_query=identity(q),work=self.work),result)
        candidate['lower']=str(result.lower+1)
        with self.assertRaisesRegex(ValueError,'overclaim'):
            check_candidate(g,q,candidate,expected_query=identity(q),work=self.work)

    def test_graph_and_source_anchors(self):
        for key in ('expected_graph','expected_request','expected_source'):
            with self.assertRaisesRegex(ValueError,'identity'): self.graph(**{key:'0'*64})

    def test_missing_and_duplicate_roots(self):
        g=self.graph(); result,q,d=self.bound(g)
        doc=copy.deepcopy(self.doc);doc['roots'].pop();new=self.graph(doc)
        with self.assertRaisesRegex(ValueError,'binding'):
            evaluate(new,q,d,expected_query=identity(q),work=self.work)
        doc=copy.deepcopy(self.doc);doc['roots'].append(doc['roots'][0])
        with self.assertRaisesRegex(ValueError,'root identity'):self.graph(doc)

    def test_changed_property(self):
        g=self.graph(); _,q,d=self.bound(g); old=identity(q);q['weights'][0]='2'
        with self.assertRaisesRegex(ValueError,'query identity'):
            evaluate(g,q,d,expected_query=old,work=self.work)

    def test_complete_dual_inventory_and_sign(self):
        g=self.graph();_,q,d=self.bound(g)
        for bad in (dict(eq=[],ine=d['ine']),dict(eq=d['eq'],ine=['1','0'])):
            with self.assertRaises(ValueError):evaluate(g,q,bad,expected_query=identity(q),work=self.work)

    def test_cycle_forward_and_zero_references(self):
        for refs in ([[1,'1']],[[0,'0']],[[0,'1'],[0,'1']]):
            doc=copy.deepcopy(self.doc);doc['nodes'][1][3]=refs
            with self.assertRaises(ValueError):self.graph(doc)

    def test_private_factor_and_reference_pollution(self):
        doc=copy.deepcopy(self.doc);doc['nodes'][0][2].append([2,'1'])
        with self.assertRaisesRegex(ValueError,'private factor'):self.graph(doc)
        doc=copy.deepcopy(self.doc);doc['nodes'][3][3]=[[2,'1']]
        with self.assertRaisesRegex(ValueError,'ancestry'):self.graph(doc)

    def test_guard_promotion(self):
        doc=copy.deepcopy(self.doc)
        doc['domains'].append(dict(name='weakened',frame=17,experts=[0,1],guard=None,parents=[3]))
        with self.assertRaisesRegex(ValueError,'guard promotion'):self.graph(doc)

    def test_frame_and_factor_alias(self):
        doc=copy.deepcopy(self.doc);doc['domains'][1]['frame']=18
        with self.assertRaisesRegex(ValueError,'frame'):self.graph(doc)
        doc=copy.deepcopy(self.doc);doc['factors'][1][0]=doc['factors'][0][0]
        with self.assertRaisesRegex(ValueError,'factor identity'):self.graph(doc)

    def test_noncanonical_numbers_zero_nodes_checked(self):
        for value in ('2/2','0','1e999999','1/'+str(2**4097)):
            doc=copy.deepcopy(self.doc);doc['nodes'][0][2][0][1]=value
            with self.assertRaises(ValueError):self.graph(doc)

    def test_deadline_and_work_capacity(self):
        with self.assertRaises(TimeoutError):self.graph(work=Work(time.monotonic()-1))
        self.work.counts['arithmetic']=8000000
        with self.assertRaisesRegex(ValueError,'work capacity'):self.graph()

    def test_intermediate_bit_limit(self):
        with self.assertRaisesRegex(ValueError,'rational capacity'):self.work.exact(F(2**8193))

    def test_caller_query_and_dual_pollution_after_capture(self):
        g=self.graph();expected,q,d=self.bound(g);qid=identity(q)
        original_visit=self.work.visit;mutated=[]
        def visit(kind, amount=1):
            if not mutated:
                q['offset']='999';q['weights'][0]='100';d['eq'][0]='999';mutated.append(True)
            original_visit(kind,amount)
        self.work.visit=visit
        got=evaluate(g,q,d,expected_query=qid,work=self.work)
        self.assertEqual(got,expected);self.assertTrue(mutated)

    def test_caller_candidate_pollution_cannot_hide_overclaim(self):
        g=self.graph();b,q,d=self.bound(g);candidate=dict(query=identity(q),**d,lower=str(b.lower+1))
        original_visit=self.work.visit
        def visit(kind, amount=1):
            candidate['lower']=str(b.lower-100);original_visit(kind,amount)
        self.work.visit=visit
        with self.assertRaisesRegex(ValueError,'overclaim'):
            check_candidate(g,q,candidate,expected_query=identity(q),work=self.work)

    def test_exact_growth_during_adjoint(self):
        doc=copy.deepcopy(self.doc);large=str(2**4095)
        doc['nodes']=[['n0',0,[[0,large]],[]],['n1',0,[],[[0,large]]],['n2',0,[],[[1,large]]]]
        doc['roots']=[['o',0,'output','0',2]]
        g=self.graph(doc);q=query(g,['1'])
        with self.assertRaisesRegex(ValueError,'intermediate rational capacity'):
            evaluate(g,q,dict(eq=[],ine=[]),expected_query=identity(q),work=self.work)

    def test_deadline_and_work_exhaustion_inside_accumulation(self):
        for limit in ('time','work'):
            g=self.graph();_,q,d=self.bound(g);w=Work(time.monotonic()+300);visit=w.visit;seen=[]
            def counted(kind, amount=1):
                if kind=='coefficient_visits' and not seen:
                    seen.append(True)
                    if limit=='time':w.deadline=time.monotonic()-1
                    else:w.counts['arithmetic']=8000000
                visit(kind,amount)
            w.visit=counted
            with self.assertRaises((ValueError,TimeoutError)):
                evaluate(g,q,d,expected_query=identity(q),work=w)
            self.assertTrue(seen)


if __name__=='__main__':unittest.main()
