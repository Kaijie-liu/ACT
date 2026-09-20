"""Algebra, source identity, exact witnesses and fail-closed mutation controls."""
import copy
from fractions import Fraction as F
import itertools
import math
import unittest

from source_enclosure.format import empty,pack,unpack,identity
from source_enclosure.produce import box,redundant_guards,affine,relu,join
from source_enclosure.check import check_box,check_guards,check_affine,check_relu,check_join


def value(h,xi,beta):
    return [c+sum((v*xi[k] for k,v in a.items()),F(0))+sum((v*beta[k] for k,v in b.items()),F(0))
            for c,a,b in zip(h['c'],h['Gc'],h['Gb'])]


def feasible(h,xi,beta):
    if any(not -1<=v<=1 for v in xi) or any(v not in (-1,1) for v in beta):return False
    for ck,bk,r,eq in (('Ac','Ab','b',True),('Auc','Aub','ub',False)):
        for a,b,rhs in zip(h[ck],h[bk],h[r]):
            v=sum((w*xi[k] for k,w in a.items()),F(0))+sum((w*beta[k] for k,w in b.items()),F(0))
            if (v!=rhs if eq else v>rhs):return False
    return True


class SourceControls(unittest.TestCase):
    def test_box_no_small_radius_loss_and_points(self):
        lower=[0.,.1,-2.,0.,1.];upper=[1e-13,.2,3.,math.ulp(0.),1.]
        state=box(lower,upper)
        self.assertEqual(check_box(lower,upper,state)['coordinates'],5)
        h,c,b=unpack(state)
        self.assertGreater(h['Gc'][0][0],0)
        self.assertGreater(h['Gc'][3][3],0)
        self.assertEqual(h['Gc'][4],{})
        mutated=copy.deepcopy(state);mutated['hz']['Gc']['data'][0]='0'
        with self.assertRaises(ValueError):check_box(lower,upper,mutated)

    def test_box_coordinate_alias_and_order(self):
        state=box([0,0],[1,1])
        x=copy.deepcopy(state);x['continuous_ids'][1]=x['continuous_ids'][0]
        with self.assertRaises(ValueError):check_box([0,0],[1,1],x)
        x=copy.deepcopy(state);x['hz']['Gc']['indices']=[1,0]
        with self.assertRaises(ValueError):check_box([0,0],[1,1],x)
        with self.assertRaises(ValueError):box([1],[0])

    def test_redundant_guard_tie_and_bad_rows(self):
        base=box([-1],[1]);ac=[{0:F(1)}];ab=[{}];rhs=[F(1)]
        guarded=redundant_guards(base,ac,ab,rhs)
        self.assertEqual(check_guards(base,guarded,ac,ab,rhs)['slacks'],['0'])
        with self.assertRaises(ValueError):check_guards(base,guarded,ac,ab,[F(0)])
        x=copy.deepcopy(guarded);x['hz']['ub'][0]='2'
        with self.assertRaises(ValueError):check_guards(base,x,ac,ab,rhs)

    def fixture(self):
        base=box([.1,-1],[.3,1]);s,c,b=unpack(base)
        # Include one inherited binary factor: its error must also be enclosed.
        s['Gb'][0]={0:F(1,8)};base=pack(s,c,['base/sign'])
        base=redundant_guards(base,[{0:F(1)}],[{0:F(1)}],[F(2)])
        s,c,b=unpack(base);n=empty(2)
        n['c']=[F(.031),F(-.07)];n['Gc']=[{0:F(.013)},{1:F(.01)}];n['Gb']=[{},{}]
        nominal=pack(n,c,b)['hz'];op=[{0:F(.1),1:F(.2)},{0:F(-.3)}];bias=[F(.01),F(.1)]
        return base,nominal,op,bias

    def test_affine_compensation_exact_witnesses(self):
        base,nominal,op,bias=self.fixture();out,cert=affine(base,op,bias,nominal,'a')
        result=check_affine(base,out,op,bias,nominal,cert,'a');self.assertEqual(result['error_factors'],2)
        s,c,b=unpack(base);t,ct,bt=unpack(out)
        for xi in itertools.product(map(F,[-1,0,1]),repeat=2):
            for beta in ((F(-1),),(F(1),)):
                self.assertTrue(feasible(s,xi,beta))
                actual=value(s,xi,beta);target=[bias[i]+sum(w*actual[j] for j,w in row.items()) for i,row in enumerate(op)]
                # Existential new coordinates reconstruct exact real affine values.
                candidate=value(t,list(xi)+[F(0)]*2,beta)
                extra=[(a-b)/F(r) for a,b,r in zip(target,candidate,cert['error_bounds'])]
                extended=list(xi)+extra
                self.assertTrue(feasible(t,extended,beta));self.assertEqual(value(t,extended,beta),target)

    def test_affine_missing_error_wrong_source_and_factors(self):
        base,nominal,op,bias=self.fixture();out,cert=affine(base,op,bias,nominal,'a')
        for mode in ('radius','source','generator','private_id','constraint','operator'):
            x=copy.deepcopy(out);p=copy.deepcopy(cert);w=copy.deepcopy(op)
            if mode=='radius':p['error_bounds'][0]='0'
            if mode=='source':p['source']='0'*64
            if mode=='generator':x['hz']['Gc']['data'][-1]='0'
            if mode=='private_id':x['continuous_ids'][-1]=x['continuous_ids'][0]
            if mode=='constraint':x['hz']['ub'][0]='1'
            if mode=='operator':w[0][0]+=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):check_affine(base,x,w,bias,nominal,p,'a')

    def test_exact_affine_needs_no_error_factor(self):
        base=box([-1],[1]);out,cert=affine(base,[{0:F(1)}],[F(0)],base['hz'],'a')
        self.assertEqual(check_affine(base,out,[{0:F(1)}],[F(0)],base['hz'],cert,'a')['error_factors'],0)

    def relu_fixture(self):
        h=empty(4);h['c']=[F(2),F(-2),F(0),F(0)]
        h['Gc']=[{0:F(1)},{0:F(1)},{1:F(1)},{}];h['Gb'][2]={0:F(1,4)}
        h['Auc']=[{0:F(1)}];h['Aub']=[{}];h['ub']=[F(1)]
        return pack(h,['input/c/0','input/c/1'],['base/sign'])

    def test_relu_active_inactive_unstable_zero_and_witnesses(self):
        base=self.relu_fixture();out,cert=relu(base,'r');r=check_relu(base,out,cert,'r')
        self.assertEqual((r['active'],r['inactive'],r['unstable']),(2,1,1))
        s,c,b=unpack(base);t,ct,bt=unpack(out)
        for xi in itertools.product(map(F,[-1,0,1]),repeat=2):
            for beta in ((F(-1),),(F(1),)):
                actual=value(s,xi,beta);extra=[];sign=[]
                for a,(lo,hi),kind in zip(actual,cert['ranges'],cert['branches']):
                    if kind!='unstable':continue
                    lo,hi=F(lo),F(hi)
                    if a<=0:extra.extend([2*a/lo-1,F(1)]);sign.append(F(1))
                    else:extra.extend([F(1),1-2*a/hi]);sign.append(F(-1))
                self.assertTrue(feasible(t,list(xi)+extra,list(beta)+sign))
                self.assertEqual(value(t,list(xi)+extra,list(beta)+sign),[max(F(0),v) for v in actual])

    def test_relu_range_sign_slot_and_missing_rows(self):
        base=self.relu_fixture();out,cert=relu(base,'r')
        for mode in ('range','branch','equation','inequality','slot','source','missing_range'):
            x=copy.deepcopy(out);p=copy.deepcopy(cert)
            if mode=='range':p['ranges'][2]=['0','1']
            if mode=='branch':p['branches'][2]='active'
            if mode=='equation':x['hz']['b'][-1]='1'
            if mode=='inequality':x['hz']['Aub']['data'][-1]='-1'
            if mode=='slot':x['binary_ids'][-1]=x['continuous_ids'][-1]
            if mode=='source':p['source']='0'*64
            if mode=='missing_range':p['ranges'].pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):check_relu(base,x,p,'r')

    def test_shared_frame_join_and_complete_constraint_map(self):
        base=self.relu_fixture();a,pa=relu(base,'a');b,pb=relu(base,'b');out,p=join(base,a,b)
        r=check_join(base,a,b,out,p)
        self.assertEqual((r['shared_continuous'],r['shared_binary'],r['binary']),(2,1,3))
        for mode in ('map','alias','source','rhs','output'):
            x=copy.deepcopy(out);cert=copy.deepcopy(p)
            if mode=='map':cert['maps']['right_c'][-1]=cert['maps']['left_c'][-1]
            if mode=='alias':x['binary_ids'][-1]=x['binary_ids'][-2]
            if mode=='source':cert['base']='0'*64
            if mode=='rhs':x['hz']['ub'][-1]='1'
            if mode=='output':x['hz']['c'][0]='0'
            with self.subTest(mode=mode),self.assertRaises(ValueError):check_join(base,a,b,x,cert)

    def test_join_rejects_consistently_aliased_private_sources(self):
        base=self.relu_fixture();a,_=relu(base,'same');b,_=relu(base,'same')
        out,p=join(base,a,b)
        with self.assertRaises(ValueError):check_join(base,a,b,out,p)


if __name__=='__main__':unittest.main()
