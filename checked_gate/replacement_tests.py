"""Analytic proof fixtures; no numerical solver or model needed."""
import copy
from fractions import Fraction as F
import unittest

from checked_gate.bootstrap import setup
setup()
from act.back_end.solver.lp_certificate import identity
from act.back_end.solver.rational_mccormick import build, csr
from act.back_end.solver.sparse_lp_certificate import evaluate
from act.pipeline.moe.check_request_lp import RATIONAL_TRUSTED
from checked_gate.propose import propose
from checked_gate.replacement import check


def zero_dual(lp):
    c = {'lp_sha256': identity(lp), 'inequality_dual': [0]*len(lp['b']),
         'equality_dual': [0]*len(lp['h'])}
    c['claimed_lower_bound'] = str(evaluate(lp, c)[0])
    return c


def source(centers):
    return {'c': centers, 'Gc': csr([{} for _ in centers], 1),
        'Gb': csr([{} for _ in centers], 0), 'Ac': csr([], 1), 'Ab': csr([], 0),
        'Auc': csr([], 1), 'Aub': csr([], 0), 'b': [], 'ub': []}


def fixture(margin='2'):
    request = {'top_k': 2, 'tie_policy': 'ANY_LEGAL_TOPK', 'experts': 2,
               'classes': 2, 'clean_prediction': 0}
    rid = identity(request)
    m = {'schema': 'request_lp_rational_v3', 'request': request, 'request_id': rid,
         'trusted_base': RATIONAL_TRUSTED, 'positive_threshold': 1e-7,
         'routes': {'feasible': [[0,1]], 'infeasible': [], 'unresolved': [], 'exact': True},
         'proofs': {}, 'obligations': []}
    store = {}
    def add(key, src, q, kind, prop):
        center = sum(F(v)*F(w) for v,w in zip(src['c'], q))
        lp = {'matrix_format': 'csr_v1', 'c': ['0'], 'offset': str(center),
              'lower': [-1], 'upper': [1], 'A': csr([],1), 'b': [], 'E': csr([],1), 'h': []}
        store[key+'e'] = {'source': src, 'source_sha256': identity(src), 'lp': lp,
            'q': q, 'offset': 0, 'relaxation': 'BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX',
            'n_relaxed_binaries': 0}
        store[key+'c'] = zero_dual(lp)
        m['proofs'][key] = {'request_id': rid, 'kind': kind, 'scope': {'pair':[0,1]},
            'property_index': prop, 'status': 'CHECKED', 'export': key+'e',
            'certificate': key+'c', 'hz_sha256': identity(src)}
    joint = source(['1/5', '0', '-4/5', '0'])
    router = source([margin, '0'])
    add('dl', joint, [1,-1,-1,1], 'difference', 0)
    add('du', joint, [-1,1,1,-1], 'difference', 0)
    add('gl', router, [1,-1], 'router_order', None)
    add('gu', router, [-1,1], 'router_order', None)
    coarse = [0.5, 0.5 if F(margin) == 0 else 1]
    old = build(joint, [1,-1], 0, coarse, [1,1])
    store.update(oute=old, outc=zero_dual(old['lp']))
    m['proofs']['out'] = {'request_id':rid,'kind':'rational_weighted','scope':{'pair':[0,1]},
        'property_index':0,'status':'CHECKED','export':'oute','certificate':'outc',
        'hz_sha256':identity(joint)}
    m['obligations'] = [{'pair':[0,1],'property_index':0,'kind':'residual',
        'difference_lower':'dl','difference_upper':'du','difference_bounds':[1,1],
        'gate_lower':'gl','gate_upper':'gu','lambda_bounds':coarse,'source':'out'}]
    replacement = {'schema':'SINGLE_GATE_REPLACEMENT_V1','request_id':rid,
                   'manifest_identity':identity(m),'pair':[0,1],'property_index':0}
    context = {'request_id':rid,'ordered_pair':[0,1],
        'margin_lower_proof':identity(store['glc']),
        'margin_negative_upper_proof':identity(store['guc'])}
    gate = propose(context,[margin,margin])
    record = build(joint,[1,-1],0,gate['gate'],[1,1])
    return m,store,replacement,gate,record,zero_dual(record['lp'])


def run(parts):
    m,s,r,g,e,c = parts
    return check(m,s.__getitem__,r,g,e,c,expected_request_id=m['request_id'],
                 expected_manifest_identity=r['manifest_identity'])


class Replacements(unittest.TestCase):
    def test_complete_analytic_request(self):
        result = run(fixture())
        self.assertEqual(result['counts'], {'reused':0,'residual':1,'unknown':0})
        self.assertEqual(result['status'], 'CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING')
        self.assertFalse(result['complete_strict_network_certificate'])

    def test_tie_nonpositive_not_unsafe(self):
        result = run(fixture('0'))
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['counts']['unknown'],1)

    def test_multi_pair_partial_reuse_inventory(self):
        # A supplied-HZ composition fixture, not a claimed network execution.
        p=fixture();m,s,r,g,e,c=p
        m['request']['experts']=3;rid=identity(m['request']);m['request_id']=rid
        m['routes'].update(feasible=[[0,1],[0,2]],infeasible=[[1,2]])
        for item in m['proofs'].values():item['request_id']=rid
        for key,q in [('gl',[1,-1,0]),('gu',[-1,1,0])]:
            src=source(['2','0','0']);s[key+'e'].update(source=src,source_sha256=identity(src),q=q)
            m['proofs'][key]['hz_sha256']=identity(src)
        for expert in (0,2):
            key='expert'+str(expert);src=source(['1','0'])
            record=copy.deepcopy(s['gle'])
            record.update(source=src,source_sha256=identity(src),q=[1,-1])
            record['lp']['offset']='1';s[key+'e']=record;s[key+'c']=zero_dual(record['lp'])
            m['proofs'][key]={'request_id':rid,'kind':'expert','scope':{'membership':expert},
                'property_index':0,'status':'CHECKED','export':key+'e','certificate':key+'c',
                'hz_sha256':identity(src)}
        m['obligations'].append({'pair':[0,2],'property_index':0,'kind':'reused',
                                 'sources':['expert0','expert2']})
        r.update(request_id=rid,manifest_identity=identity(m));g['context']['request_id']=rid
        result=run(p)
        self.assertEqual(result['required'],2)
        self.assertEqual(result['counts'],{'reused':1,'residual':1,'unknown':0})
        m['obligations'][1]['kind']='unknown';r['manifest_identity']=identity(m)
        with self.assertRaisesRegex(ValueError,'exactly one'):run(p)

    def test_missing_inventory_source_scope_gate_and_dual(self):
        mutations = [lambda p:p[0]['obligations'].clear(),
            lambda p:p[0]['obligations'].append(copy.deepcopy(p[0]['obligations'][0])),
            lambda p:p[0]['routes'].update(exact=False),
            lambda p:p[0]['proofs']['dl']['scope'].update(pair=[1,0]),
            lambda p:p[2].update(request_id='wrong'),
            lambda p:p[2].update(property_index=1),
            lambda p:p[3]['context'].update(ordered_pair=[1,0]),
            lambda p:p[3]['context'].update(margin_lower_proof='f'*64),
            lambda p:p[4]['source']['c'].__setitem__(0,'9'),
            lambda p:p[4]['q'].__setitem__(0,'2'),
            lambda p:p[4]['gate'].__setitem__(0,'1'),
            lambda p:p[5].update(claimed_lower_bound='100'),
            lambda p:p[5].update(lp_sha256='wrong')]
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i):
                p=copy.deepcopy(fixture()); mutate(p)
                # Even a freshly hash-bound malformed request must fail semantic
                # checks, not only the outer immutable-manifest hash comparison.
                p[2]['manifest_identity']=identity(p[0])
                with self.assertRaises((ValueError,KeyError)): run(p)

    def test_no_solver_or_builder_in_checker(self):
        from unittest.mock import patch
        p=fixture()
        with patch('act.back_end.solver.rational_mccormick.build', side_effect=AssertionError), \
             patch('act.back_end.solver.sparse_lp_certificate.propose', side_effect=AssertionError), \
             patch('checked_gate.propose.propose', side_effect=AssertionError):
            self.assertEqual(run(p)['counts']['unknown'],0)


if __name__ == '__main__':
    unittest.main()
