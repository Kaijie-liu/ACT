"""No real checkpoint, historical input, native solver or GPU queries."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from checked_route_frontier import build as old
from checked_route_frontier.check import check_frontier, reconstruct_margin
from checked_route_frontier.fixtures import analytic
from scoped_proof.evidence import lower_bound
from scoped_proof.io import PYTHON, ROOT, save
from source_construction_lab.fixtures import document
from source_enclosure.format import empty, identity, pack, sparse, unpack
from shared_route_residual import check as verifier, propose as producer
from shared_route_residual.format import binding


def end():
    return time.monotonic()+60


def make(doc):
    pre=old.prefix(doc,expected_source_sha256=identity(doc),deadline=end())
    cert=producer.propose(doc,pre,invocation='control',deadline=end())
    return pre,cert


def inspect(doc,pre,cert,**kw):
    return verifier.check(doc,pre,cert,expected_source_sha256=identity(doc),
                          invocation=kw.get('invocation','control'),deadline=kw.get('deadline',end()))


def comparable(result):
    keys=('higher','lower','checked_lower_bound','residual_box_term','nonzero_residual_coordinates','status')
    return {'bounds':[{k:r[k] for k in keys} for r in result['bounds']],
            'pairs':[{k:v for k,v in r.items() if k not in ('lp_sha256','evidence_sha256')} for r in result['pairs']],
            'needed_experts':result['needed_experts']}


class ResidualControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=analytic();cls.pre,cls.cert=make(cls.doc)

    def test_exact_differential_analytic(self):
        for kind in ('prunable','tied','crossing'):
            doc=analytic(kind);pre,cert=make(doc)
            reference=check_frontier(doc,pre,old.propose_final_affine(doc,pre,deadline=end()),
                                    expected_source_sha256=identity(doc),deadline=end())
            self.assertEqual(comparable(inspect(doc,pre,cert)),comparable(reference))

    def test_random_depth_dimension_and_zero_radius_differential(self):
        for e,c,w,d,seed,r in ((2,2,1,0,7,'0'),(3,5,3,1,11,'1/8'),
                               (4,3,4,2,17,'1/4'),(5,2,2,3,21,'1/8')):
            doc=document(experts=e,classes=c,width=w,depth=d,seed=seed,radius=r)
            pre,cert=make(doc)
            reference=check_frontier(doc,pre,old.propose_final_affine(doc,pre,deadline=end()),
                                    expected_source_sha256=identity(doc),deadline=end())
            self.assertEqual(comparable(inspect(doc,pre,cert)),comparable(reference))

    def test_constant_router(self):
        doc=document(experts=3,classes=4,width=2,depth=1,constant=True)
        pre,cert=make(doc);r=inspect(doc,pre,cert)
        self.assertEqual(r['retained_pairs'],3)
        self.assertTrue(all(F(b['checked_lower_bound'])==0 for b in r['bounds']))

    def test_general_equality_potentials_binary_residuals(self):
        h=empty(3);h.update(c=[F(1,3),F(-2,7),F(5,11)],
            Gc=[{0:F(1),1:F(3)},{1:F(4)},{0:F(-2)}],Gb=[{0:F(2)},{0:F(-3)},{}],
            Ac=[{0:F(1),1:F(2)},{1:F(-1)}],Ab=[{0:F(3)},{0:F(2)}],b=[F(1),F(0)],
            Auc=[{0:F(1)}],Aub=[{0:F(-1)}],ub=[F(4)])
        state=pack(h,['x','y'],['binary']);duals=[{0:F(2,3)},{1:F(-1,7)},{0:F(-2),1:F(5)}]
        result=verifier.evaluate(state,sparse(duals,2),deadline=end())
        for row in result['bounds']:
            j,i=row['higher'],row['lower'];lp=reconstruct_margin(state,j,i)
            z=[duals[j].get(k,F(0))-duals[i].get(k,F(0)) for k in range(2)]
            ref=lower_bound(lp,{'lp_sha256':identity(lp),'inequality_dual':['0'],
                                'equality_dual':list(map(str,z))})
            for k in ('checked_lower_bound','residual_box_term','nonzero_residual_coordinates'):
                self.assertEqual(row[k],ref[k])

    def test_shared_cancellation_not_independent_interval_subtraction(self):
        h=empty(2);h.update(c=[F(1),F(0)],Gc=[{0:F(100)},{0:F(100)}])
        result=verifier.evaluate(pack(h,['x'],[]),sparse([{},{}],0),deadline=end())
        self.assertEqual(F(result['bounds'][0]['checked_lower_bound']),1)
        self.assertEqual(F(1)-100-100,-199)  # separate boxes cannot prove it

    def test_ties_and_crossing_keep_all_legal_pairs(self):
        from itertools import combinations
        doc=analytic('crossing');pre,cert=make(doc);r=inspect(doc,pre,cert)
        kept={tuple(p['pair']) for p in r['pairs'] if p['status']=='RETAINED'}
        for x in (F(-1),F(-1,7),F(0),F(1,7),F(1)):
            scores=[x,-x,F(0),F(-3)]
            legal={p for p in combinations(range(4),2)
                   if all(scores[i]>=scores[j] for i in p for j in range(4) if j not in p)}
            self.assertTrue(legal<=kept)
        d=analytic('tied');p,c=make(d);self.assertEqual(inspect(d,p,c)['retained_pairs'],6)

    def test_no_output_or_literal_route_claim(self):
        r=inspect(self.doc,self.pre,self.cert)
        self.assertEqual((r['total_pairs'],r['retained_pairs'],r['original_output_obligations']), (6,1,12))
        for k in ('native_float_proof','route_changing_established','complete_output_positive_proof'):
            self.assertIs(r[k],False)
        self.assertEqual(r['algebra_counts']['algebra_final_state_parses'],1)

    def test_identity_source_request_prefix_factor_and_run(self):
        for key in self.cert['binding']:
            c=copy.deepcopy(self.cert);c['binding'][key]='wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):inspect(self.doc,self.pre,c)
        with self.assertRaises(ValueError):inspect(self.doc,self.pre,self.cert,invocation='other')
        doc=copy.deepcopy(self.doc);doc['request']['margin']='2/100'
        with self.assertRaises(ValueError):inspect(doc,self.pre,self.cert)

    def test_no_inequality_potentials_or_trusted_claim(self):
        for key in ('score_inequalities','claimed_lower_bound','residuals','positive'):
            c=copy.deepcopy(self.cert);c[key]=[]
            with self.subTest(key=key),self.assertRaises(ValueError):inspect(self.doc,self.pre,c)

    def test_bad_csr_missing_score_nonfinite_and_shape(self):
        for mode in ('missing','shape','duplicate','nonfinite','bool'):
            c=copy.deepcopy(self.cert);m=c['score_equalities']
            if mode=='missing':m['indptr'].pop()
            if mode=='shape':m['shape'][0]+=1
            if mode=='duplicate':m['indices'][1]=m['indices'][0];m['indptr'][1]=2
            if mode=='nonfinite':m['data'][0]='NaN'
            if mode=='bool':m['data'][0]=True
            with self.subTest(mode=mode),self.assertRaises((ValueError,TypeError)):
                inspect(self.doc,self.pre,c)

    def test_input_layer_and_factor_mutations_even_with_rebinding(self):
        for mode in ('input','layer','factor'):
            p=copy.deepcopy(self.pre);c=copy.deepcopy(self.cert)
            if mode=='input':p['input']['hz']['Gc']['data'][0]='1/2'
            if mode=='layer':p['router']['steps'][-1]['state']['hz']['c'][0]='999'
            if mode=='factor':p['router']['steps'][-1]['state']['continuous_ids'][0]='foreign'
            c['binding']=binding(self.doc,p,'control')
            with self.subTest(mode=mode),self.assertRaises(ValueError):inspect(self.doc,p,c)

    def test_unused_inequality_matrix_is_validated(self):
        p=copy.deepcopy(self.pre);s=p['router']['steps'][-1]['state']
        s['hz']['Auc']['shape'][1]+=1
        with self.assertRaises(ValueError):verifier.evaluate(s,self.cert['score_equalities'],deadline=end())

    def test_no_cache_mutation_or_pollution(self):
        expected=inspect(self.doc,self.pre,self.cert)
        c=copy.deepcopy(self.cert);c['score_equalities']['data'][0]='999'
        changed=inspect(self.doc,self.pre,c)  # arbitrary equality potential is legal, not trusted
        self.assertNotEqual(expected['bounds'],changed['bounds'])
        self.assertEqual(expected,inspect(self.doc,self.pre,self.cert))

    def test_empty_invalid_invocation(self):
        for inv in ('',False,'x'*129):
            with self.assertRaises(ValueError):binding(self.doc,self.pre,inv)

    def test_deadline_before_and_after_exact_arithmetic(self):
        for stop in (time.monotonic()-1,time.monotonic()+301,float('inf')):
            with self.assertRaises((ValueError,TimeoutError)):
                inspect(self.doc,self.pre,self.cert,deadline=stop)
        state=self.pre['router']['steps'][-1]['state'];real_unpack=verifier.unpack
        with patch('scoped_source.graph.time.monotonic',return_value=10) as now:
            def late(value):
                result=real_unpack(value);now.return_value=12;return result
            with patch.object(verifier,'unpack',side_effect=late),self.assertRaises(TimeoutError):
                verifier.evaluate(state,self.cert['score_equalities'],deadline=11)

    def test_late_source_check_cannot_be_accepted(self):
        original=verifier.check_network
        with patch('scoped_source.graph.time.monotonic',return_value=10) as now:
            def late(*args):
                result=original(*args);now.return_value=12;return result
            with patch.object(verifier,'check_network',side_effect=late),self.assertRaises(TimeoutError):
                inspect(self.doc,self.pre,self.cert,deadline=11)

    def test_independent_checker_in_fresh_python_no_site(self):
        with (patch.object(producer,'propose',side_effect=AssertionError('producer')),
              patch.object(old,'prefix',side_effect=AssertionError('producer'))):
            inspect(self.doc,self.pre,self.cert)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as folder:
            path=Path(folder)/'packet.json';save(path,[self.doc,self.pre,self.cert])
            code='''import json,sys,time
from shared_route_residual.check import check
from source_enclosure.format import identity
d,p,c=json.load(open(sys.argv[1]))
r=check(d,p,c,expected_source_sha256=identity(d),invocation='control',deadline=time.monotonic()+30)
assert r['retained_pairs']==1 and not r['complete_output_positive_proof']
assert not any(n.split('.')[0] in ('numpy','scipy','torch','act','highspy') for n in sys.modules)
assert 'shared_route_residual.propose' not in sys.modules
print('PASS')
'''
            run=subprocess.run([PYTHON,'-S','-c',code,str(path)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(run.stdout.strip(),'PASS')


if __name__=='__main__':unittest.main()
