"""Analytic controls only. No native solve, real checkpoint, or effect search."""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from source_enclosure.format import unpack,pack,identity
from source_enclosure.produce import box,join,relu as old_relu
from source_enclosure.check import check_join
from full_source.lift import affine as old_affine
from full_source.obligations import build,materialize
from full_source.check_obligations import check as check_output
from full_bounds.check import check_one
from source_ranges.produce import range_fact,affine,relu
from source_ranges.check import projection,check_range,check_affine,check_relu

CONTEXT={'request_id':'analytic-only-v1','scope':'same-source-domain','layer':'control'}


def constrained(lo=F(1,4),hi=F(3,4),many=False):
    s=box([-1],[1]);h,c,b=unpack(s)
    h['Auc']=[{0:F(-1)},{0:F(1)}];h['Aub']=[{},{}];h['ub']=[-lo,hi]
    if many:
        h['c']=[F(0)]*4;h['Gc']=[{0:F(1)},{0:F(-1)},{0:F(1)},{}];h['Gb']=[{} for _ in range(4)]
    return pack(h,c,b)


def interval_fact(s,row,index,lo,hi,negative=False,context=CONTEXT):
    ylo,yhi=([0,-1],[-1,0]) if negative else ([-1,0],[0,-1])
    return range_fact(s,row,0,context,index,[lo,hi],(ylo,[]),(yhi,[]))


def mixed_control():
    s=constrained(many=True)
    facts=[interval_fact(s,{0:1},0,F(1,4),F(3,4)),
           interval_fact(s,{1:1},1,F(-3,4),F(-1,4),negative=True),None,None]
    t,p=relu(s,facts,CONTEXT,'mixed')
    return s,t,p


class RangeControls(unittest.TestCase):
    def test_projection_and_both_sides(self):
        s=constrained();f=interval_fact(s,{0:1},0,F(1,4),F(3,4))
        lp,natural=projection(s,{0:1},0)
        self.assertEqual(natural,(F(-1),F(1)));self.assertEqual(lp['c'],['1'])
        self.assertEqual(lp['A']['data'],['-1','1']);self.assertEqual(lp['b'],['-1/4','3/4'])
        self.assertEqual(identity(lp),f['lower']['lp_sha256'])
        bounds,r=check_range(s,{0:1},0,f,CONTEXT,0)
        self.assertEqual(bounds,(F(1,4),F(3,4)));self.assertEqual(r['negative_upper_checked'],'-3/4')

    def test_affine_recentering_preserves_assignments(self):
        s=constrained();f=interval_fact(s,{0:1},0,F(1,4),F(3,4))
        t,p=affine(s,[{0:1}],[0],[f],CONTEXT,'shift')
        result=check_affine(s,t,[{0:1}],[0],p,CONTEXT,'shift');h,ci,_=unpack(t)
        self.assertEqual(h['c'],[F(1,2)]);self.assertEqual(h['Gc'],[{1:F(1,4)}])
        self.assertEqual(h['b'],[F(1,2)]);self.assertEqual(result['new_continuous_factors'],1)
        for x in (F(1,4),F(1,2),F(3,4)):
            factors=[x,4*x-2]
            self.assertEqual(sum(v*factors[j] for j,v in h['Ac'][-1].items()),h['b'][-1])
            self.assertEqual(h['c'][0]+sum(v*factors[j] for j,v in h['Gc'][0].items()),x)

    def test_mixed_relu_partial_fallback_and_zero_tie(self):
        s,t,p=mixed_control();old,_=old_relu(s,'mixed')
        r=check_relu(s,t,p,CONTEXT,'mixed')
        self.assertEqual((r['active'],r['inactive'],r['unstable']),(2,1,1))
        self.assertEqual((r['checked_range_rows'],r['fallback_rows']),(2,2))
        self.assertEqual(len(unpack(old)[2]),3);self.assertEqual(len(unpack(t)[2]),1)
        h,_,_=unpack(t);self.assertEqual(h['Auc'][:2],unpack(s)[0]['Auc'])

    def test_unstable_tight_graph_pointwise_extension(self):
        lo,hi=F(-1,4),F(3,4);s=constrained(lo,hi)
        fact=interval_fact(s,{0:1},0,lo,hi);t,p=relu(s,[fact],CONTEXT,'unstable')
        self.assertEqual(check_relu(s,t,p,CONTEXT,'unstable')['unstable'],1)
        h,_,_=unpack(t)
        for x in (lo,F(-1,8),F(0),F(1,4),hi):
            z=F(1) if x<0 else F(-1)
            factors=[x,2*x/lo-1 if x<0 else F(1),F(1) if x<0 else 1-2*x/hi]
            self.assertTrue(all(-1<=v<=1 for v in factors))
            for a,b,rhs in zip(h['Ac'],h['Ab'],h['b']):
                self.assertEqual(sum(v*factors[j] for j,v in a.items())+sum(v*z for v in b.values()),rhs)
            for a,b,rhs in zip(h['Auc'],h['Aub'],h['ub']):
                self.assertLessEqual(sum(v*factors[j] for j,v in a.items())+sum(v*z for v in b.values()),rhs)
            self.assertEqual(h['c'][0]+sum(v*factors[j] for j,v in h['Gc'][0].items()),max(F(0),x))

    def test_no_fact_differential_and_binary_source(self):
        s,_=old_relu(box([-1,0],[1,2]),'base')
        t,p=affine(s,[{0:-2,1:1}],[F(1,3)],[None],CONTEXT,'lift')
        expected,_=old_affine(s,[{0:F(-2),1:F(1)}],[F(1,3)],'lift')
        self.assertEqual(t,expected);check_affine(s,t,[{0:-2,1:1}],[F(1,3)],p,CONTEXT,'lift')
        u,q=relu(s,[None,None],CONTEXT,'r');v,_=old_relu(s,'r')
        self.assertEqual(u,v);check_relu(s,u,q,CONTEXT,'r')
        lp,_=projection(s,{0:1},0)
        self.assertEqual(len(lp['c']),len(unpack(s)[1])+len(unpack(s)[2]))

    def test_zero_width_correlation_and_complete_weighted_properties(self):
        s=box([-1,-1],[1,1]);h,c,b=unpack(s)
        h['Ac']=[{0:F(1),1:F(1)}];h['Ab']=[{}];h['b']=[F(0)];s=pack(h,c,b)
        ends=[]
        for i,value in enumerate((F(1,5),F(3,10))):
            ctx={**CONTEXT,'scope':f'expert{i}'};op=[{0:1,1:1},{},{}];bias=[value,0,F(1,20)]
            fact=range_fact(s,op[0],value,ctx,0,[value,value],([],[1]),([],[-1]))
            t,p=affine(s,op,bias,[fact,None,None],ctx,f'e{i}')
            r=check_affine(s,t,op,bias,p,ctx,f'e{i}')
            self.assertEqual(r['new_continuous_factors'],0)
            self.assertEqual(len(unpack(t)[0]['b']),2) # retained redundant defining equality
            ends.append(t)
        joint,p=join(s,*ends);check_join(s,*ends,joint,p)
        base,obs=build(joint,[0,1],3,0);check_output(joint,base,obs,[0,1],3,0)
        self.assertEqual(len(obs['rows']),2)
        values=[]
        for row in obs['rows']:
            lp=materialize(base,row)
            result=check_one(lp,{'lp_sha256':identity(lp),'inequality_dual':[0]*len(lp['b']),'equality_dual':[0]*len(lp['h'])})
            values.append(F(result['checked_lower_bound']))
        self.assertEqual(values,[F(1,5),F(3,20)])

    def test_bad_bindings_ranges_and_duals_reject(self):
        s,t,p=mixed_control()
        actions=[lambda p:p['facts'][0]['query'].__setitem__('source_sha256','old-source'),
            lambda p:p['facts'][0]['query']['context'].__setitem__('request_id','other'),
            lambda p:p['facts'][0]['query']['context'].__setitem__('scope','other-domain'),
            lambda p:p['facts'][0]['query']['context'].__setitem__('layer','other-layer'),
            lambda p:p['facts'][0]['query'].__setitem__('row',1),
            lambda p:p['facts'][0]['query'].__setitem__('expression',[[0,'2']]),
            lambda p:p['facts'][0].__setitem__('range',['1/2','3/4']),
            lambda p:p['facts'][0].__setitem__('range',['3/4','1/4']),
            lambda p:p['facts'][0]['negative_upper'].__setitem__('lp_sha256','wrong-sign-LP'),
            lambda p:p['facts'][0]['lower'].__setitem__('inequality_dual',[1,0]),
            lambda p:p['facts'][0]['lower'].__setitem__('inequality_dual',[0,0]),
            lambda p:p['facts'][0]['negative_upper'].__setitem__('inequality_dual',[0,0]),
            lambda p:p['facts'][0].pop('negative_upper'),
            lambda p:p['facts'].pop(),
            lambda p:p['facts'][0]['lower'].__setitem__('claimed_lower_bound','999')]
        for action in actions:
            bad=copy.deepcopy(p);action(bad)
            with self.assertRaises((ValueError,KeyError)):check_relu(s,t,bad,CONTEXT,'mixed')
        with self.assertRaises(ValueError):check_relu(s,t,p,{**CONTEXT,'request_id':'other'},'mixed')

    def test_graph_tamper_factor_alias_and_checker_independence(self):
        s=constrained();f=interval_fact(s,{0:1},0,F(1,4),F(3,4));t,p=affine(s,[{0:1}],[0],[f],CONTEXT,'a')
        for change in (lambda t:t['hz']['b'].__setitem__(0,'0'),
                       lambda t:t['hz']['ub'].__setitem__(0,'100'),
                       lambda t:t['continuous_ids'].__setitem__(-1,t['continuous_ids'][0])):
            bad=copy.deepcopy(t);change(bad)
            with self.assertRaises(ValueError):check_affine(s,bad,[{0:1}],[0],p,CONTEXT,'a')
        with patch('source_ranges.produce.affine',side_effect=AssertionError('producer imported')):
            check_affine(s,t,[{0:1}],[0],p,CONTEXT,'a')


if __name__=='__main__':unittest.main()
