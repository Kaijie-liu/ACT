import base64
import copy
from fractions import Fraction as F
import hashlib
import itertools
import json
import struct
import unittest
from unittest.mock import patch

from router_source.checker import check,compact,digest,tensor
from router_source.propose import propose


def encode(values,shape):
    return {'dtype':'torch.float64','shape':shape,'byte_order':'little',
            'bytes':base64.b64encode(struct.pack('<'+'d'*len(values),*values)).decode()}


def fixture(weights=None,bias=None,pool=2):
    weights=weights or [[1.],[-1.],[0.]];bias=bias or [2.,0.,0.]
    e=len(weights);features=4//(pool*pool)
    parameters={'router.2.weight':encode(sum(weights,[]),[e,features]),
                'router.2.bias':encode(bias,[e])}
    inv=[{'name':name,**tensor(t)[0]} for name,t in sorted(parameters.items())]
    h=hashlib.sha256()
    for t in inv:h.update(t['name'].encode());h.update(t['sha256'].encode())
    image={key:encode([v]*4,[1,1,2,2]) for key,v in [('center',.5),('lower',0.),('upper',1.)]}
    request={'experts':e,'classes':2,'top_k':2,'tie_policy':'ANY_LEGAL_TOPK',
        'model_state':{'sha256':h.hexdigest(),'tensor_count':len(inv),
                       'parameter_count':e*(features+1)},
        **{k:tensor(t)[0] for k,t in image.items()}}
    doc={'schema':'AFFINE_ROUTER_SOURCE_V1','request':request,'parameters':parameters,
        'state_inventory':inv,'input':image,
        'graph':{'schema':'REAL_NONOVERLAP_AVGPOOL_FLATTEN_LINEAR_V1','pool':pool,
          'pool_stride':pool,'pool_padding':0,'ceil_mode':False,'count_include_pad':True,
          'divisor_override':None,'flatten':[1,-1],'training':False,
          'weight':'router.2.weight','bias':'router.2.bias'}}
    return doc


def checked(doc,proof=None):
    return check(doc,proof or propose(doc),expected_request=doc['request'],expected_source_sha256=digest(compact(doc)))


class Routing(unittest.TestCase):
    def test_ties_keep_every_route(self):
        d=fixture([[0.],[0.],[0.]],[0.,0.,0.]);r=checked(d)
        self.assertEqual(r['covered_pairs'],[[0,1],[0,2],[1,2]])
        self.assertEqual(r['excluded_pairs'],[])
        p=propose(d);p['routes'][0]={'pair':[0,1],'kind':'excluded','witness':[2,0],'strict_lower_bound':'0'}
        with self.assertRaises(ValueError):checked(d,p)

    def test_different_dimensions_and_exact_corner_differential(self):
        for pool,weights,bias in [(2,[[1.],[-1.],[0.]],[2.,0.,0.]),
                (1,[[1.,-1.,2.,-2.],[2.,1.,-1.,1.],[0.,0.,0.,0.]],[.25,-.5,1.])]:
            d=fixture(weights,bias,pool);r=checked(d)
            for row in r['margins']:
                a,b=row['pair'];values=[]
                for pixels in itertools.product([F(0),F(1)],repeat=4):
                    features=[sum(pixels,F(0))/4] if pool==2 else pixels
                    values.append(F(bias[a])-F(bias[b])+sum((
                        (F(weights[a][j])-F(weights[b][j]))*v for j,v in enumerate(features)),F(0)))
                self.assertEqual(F(row['lower']),min(values));self.assertEqual(F(row['upper']),max(values))
            # Every legal tie-aware pair at all corners belongs to the cover.
            for pixels in itertools.product([F(0),F(1)],repeat=4):
                features=[sum(pixels,F(0))/4] if pool==2 else pixels
                scores=[F(b)+sum((F(w)*v for w,v in zip(ws,features)),F(0)) for ws,b in zip(weights,bias)]
                for pair in itertools.combinations(range(len(scores)),2):
                    if all(scores[i]>=scores[j] for i in pair for j in range(len(scores)) if j not in pair):
                        self.assertIn(list(pair),r['covered_pairs'])

    def test_semantic_and_identity_mutations(self):
        d=fixture();original=digest(compact(d));p=propose(d)
        mutations=[lambda d:d['graph'].update(pool_stride=1),
            lambda d:d['graph'].update(pool_padding=1),
            lambda d:d['graph'].update(divisor_override=3),
            lambda d:d['graph'].update(training=True),
            lambda d:d['graph'].update(flatten=[0,-1]),
            lambda d:d['parameters']['router.2.bias'].update(bytes=encode([9.,0.,0.],[3])['bytes']),
            lambda d:d['input']['lower'].update(bytes=d['input']['upper']['bytes']),
            lambda d:d['state_inventory'].pop(),
            lambda d:d['request'].update(top_k=1)]
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i):
                bad=copy.deepcopy(d);mutate(bad)
                with self.assertRaises(ValueError):check(bad,p,expected_request=d['request'],expected_source_sha256=original)
                # Recomputed source hashes cannot bypass structural/state checks.
                p2=copy.deepcopy(p);p2['source_sha256']=digest(compact(bad))
                with self.assertRaises(ValueError):check(bad,p2,expected_request=d['request'],expected_source_sha256=digest(compact(bad)))

    def test_coverage_and_bound_mutations(self):
        d=fixture()
        for mutate in [lambda p:p['routes'].pop(),lambda p:p['routes'].append(p['routes'][0]),
                       lambda p:p['margins'][0].update(lower='10'),
                       lambda p:p['routes'][-1].update(witness=[1,0]),
                       lambda p:p['routes'][-1].update(strict_lower_bound='999')]:
            p=propose(d);mutate(p)
            with self.assertRaises(ValueError):checked(d,p)

    def test_checker_does_not_use_producer(self):
        d=fixture();p=propose(d)
        with patch('router_source.propose.propose',side_effect=AssertionError('producer used')):
            self.assertEqual(checked(d,p)['status'],'CHECKED_ROUTER_COVER_FOR_DECLARED_REAL_GRAPH')


if __name__=='__main__':unittest.main()
