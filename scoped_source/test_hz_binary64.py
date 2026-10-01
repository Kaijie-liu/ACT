"""Frozen nine-pattern enclosure controls; no optimization or model loading."""
import ast
from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from source_enclosure.format import empty, pack, unpack, identity
from scoped_source import hz_binary64 as producer
from scoped_source import check_hz_binary64 as checker

OBS={}
REJECTIONS={}
OVERFLOW=[]


def safe(value, restore=False):
    if type(value) is float and not math.isfinite(value):
        return {'__nonfinite_float__':str(value)}
    if type(value) is dict:
        if restore and set(value)=={'__nonfinite_float__'}:
            return {'inf':math.inf,'-inf':-math.inf,'nan':math.nan}[value['__nonfinite_float__']]
        return {k:safe(v,restore) for k,v in value.items()}
    if type(value) is list:return [safe(v,restore) for v in value]
    return value


def owned(h,ci,bi,co=None,bo=None):
    return {'schema':checker.REFERENCE,'state':pack(h,ci,bi),
            'ownership':{'continuous':co if co is not None else ['input']*len(ci),
                         'binary':bo if bo is not None else ['router']*len(bi)}}


def patterns():
    w=F(1)+F(1,2**52);h=empty(1);h['c']=[F(1,4)]
    h['Gc']=[{0:F(1)}];h['Gb']=[{0:F(1,2)}]
    h['Ac']=[{0:F(1)}];h['Ab']=[{0:F(1)}];h['b']=[F(0)]
    cases={'identity':(owned(h,['x'],['b']),[(['-1'],['1']),(['1'],['-1'])])}
    h=empty(1);h['Gc']=[{0:w*w,1:w*w/F(2**96)}]
    cases['affine_sum']=(owned(h,['x','y'],[]),[(['1','1'],[]),(['-1','-1'],[])])
    h=empty(1);h['c']=[F(2)+F(1,2**54)];h['Gc']=[{0:F(-1)}]
    cases['relu_endpoint']=(owned(h,['x'],[]),[(['1'],[]),(['-1'],[])])
    rhs=F(1,2**56)-F(1,2);h=empty(1);h['Gc']=[{0:F(1)}]
    h['Ac']=[{0:F(1)}];h['Ab']=[{}];h['b']=[rhs]
    cases['equality_rhs']=(owned(h,['x'],[]),[([str(rhs)],[])])
    bound=F(1)-F(1,2**54);h=empty(1);h['Gc']=[{0:F(1)}]
    h['Auc']=[{0:F(1)},{0:F(1)}];h['Aub']=[{},{}];h['ub']=[bound,-bound]
    cases['guard_rhs']=(owned(h,['x'],[]),[(['-1'],[]),([str(-bound)],[])])
    tiny=F(1,2**1075);h=empty(2);h['c']=[tiny,-tiny]
    h['Gc']=[{0:tiny},{0:-tiny}];h['Auc']=[{0:tiny},{0:-tiny}];h['Aub']=[{},{}];h['ub']=[tiny,tiny]
    h['Ac']=[{0:tiny}];h['Ab']=[{}];h['b']=[F(0)]
    cases['subnormal']=(owned(h,['x'],[]),[(['0'],[])])
    h=empty(2);h['c']=[F(1)+F(1,2**54),F(1)+F(1,2**54)]
    h['Gc']=[{0:F(1),1:F(1)},{0:F(1),2:F(1)}];h['Gb']=[{0:F(1)},{1:F(1)}]
    cases['shared_private']=(owned(h,['input/x','expert0/a','expert1/a'],['expert0/z','expert1/z'],
                                  ['shared','expert0','expert1'],['expert0','expert1']),
                             [(['0','1','-1'],['1','-1']),(['1','-1','1'],['-1','1'])])
    h=empty(1);h['Ac']=[{}];h['Ab']=[{}];h['b']=[F(0)]
    cases['zero_empty']=(owned(h,[],[]),[([],[])])
    return cases


def overflow_patterns():
    h=empty(1);h['c']=[F(2**1024)];first=owned(h,[],[])
    h=empty(1);h['Auc']=[{}];h['Aub']=[{}];h['ub']=[F(sys.float_info.max)+1]
    return [first,owned(h,[],[])]


def binary_reference():
    r=deepcopy(patterns()['identity'][0]);h,c,b=unpack(r['state'])
    h['Auc']=[{}];h['Aub']=[{0:F(1)}];h['ub']=[F(1)]
    r['state']=pack(h,c,b);return r


def extend(reference,proof,values):
    s,ci,bi=unpack(reference['state']);t,ct,bt=unpack(proof['target']['state'])
    x,z=(list(map(F,values[k])) for k in ('continuous','binary'));new=list(x)
    for kind,vec,ck,bk in (('output','c','Gc','Gb'),('equality','b','Ac','Ab')):
        for i,claim in enumerate(proof[kind+'_errors']):
            radius=F(claim['radius'])
            if not radius:continue
            exact=sum((v*x[j] for j,v in s[ck][i].items()),s[vec][i] if kind=='output' else F(0))
            exact+=sum((v*z[j] for j,v in s[bk][i].items()),F(0))
            approximate=sum((t[ck][i].get(j,F(0))*v for j,v in enumerate(x)),F(0))
            approximate+=sum((t[bk][i].get(j,F(0))*v for j,v in enumerate(z)),F(0))
            residual=exact-(t[vec][i]+approximate) if kind=='output' else t['b'][i]-approximate
            new.append(residual/radius)
    return {'continuous':list(map(str,new)),'binary':list(map(str,z))}


class EnclosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name,(ref,points) in patterns().items():
            start=time.monotonic();deadline=start+300;owner='pair/0-1'
            record={'reference':ref,'owner':owner,'start':start,'deadline':deadline};OBS[name]=record
            before=time.monotonic();proof=producer.produce(ref,expected_reference_sha256=identity(ref),owner=owner,deadline=deadline)
            made=time.monotonic();live,snap=producer.instantiate(proof['target'],expected_target_sha256=proof['target_sha256'],deadline=deadline)
            actual=time.monotonic();encoded=json.dumps(proof,sort_keys=True,allow_nan=False);proof=json.loads(encoded)
            serialized=time.monotonic();result=checker.check(ref,proof,expected_reference_sha256=identity(ref),expected_owner=owner,deadline=deadline)
            embeddings=[]
            for a,b in points:
                source={'continuous':a,'binary':b};target=extend(ref,proof,source)
                checked=checker.check_embedding(ref,proof,source,target,expected_reference_sha256=identity(ref),expected_owner=owner,deadline=deadline)
                embeddings.append({'source':source,'target':target,'checked':checked})
            end=time.monotonic()
            record.update(proof=proof,checked=result,snapshot=snap,embeddings=embeddings,end=end,
                          cost={'produce':made-before,'instantiate':actual-made,'serialize':serialized-actual,
                                'check_including_points':end-serialized,'total':end-start})

    def reject(self,name,modify,reference=False,reason=None,record=None):
        original=OBS[name] if record is None else record
        r=deepcopy(original);obj=r['reference'] if reference else r['proof'];modify(obj)
        expected=identity(original['reference']);phase='target_hash'
        try:
            if not reference:r['proof']['target_sha256']=identity(r['proof']['target'])
            phase='check'
            checker.check(r['reference'],r['proof'],expected_reference_sha256=expected,
                          expected_owner=r['owner'],deadline=time.monotonic()+30)
        except (ValueError,TypeError,KeyError) as exc:
            if record is not None and 'reference binding' in str(exc):self.fail('auxiliary binding masked mutation')
            if reason is not None:self.assertIn(reason,str(exc))
            key=self._testMethodName+':'+str(sum(k.startswith(self._testMethodName+':') for k in REJECTIONS))
            REJECTIONS[key]={'pattern':name,'auxiliary':record is not None,'reference':safe(r['reference']),
                'proof':safe(r['proof']),'expected_reference_sha256':expected,'owner':r['owner'],
                'phase':phase,'error_type':type(exc).__name__,'error':str(exc)}
        else:self.fail('mutation accepted')

    def test_all_eight_normal_patterns(self):
        self.assertEqual(set(OBS),set(patterns()))
        for r in OBS.values():
            self.assertEqual(r['checked']['status'],'CHECKED_BINARY64_OUTER_ENCLOSURE_GIVEN_REFERENCE')
            self.assertTrue(r['end']<r['deadline']);self.assertFalse(r['checked']['network_proof'])
            self.assertTrue(r['embeddings']);self.assertLessEqual(sum(v for k,v in r['cost'].items() if k!='total'),r['cost']['total'])

    def test_affine_radius_rounding(self):
        r=OBS['affine_sum']['checked']['compensation'][0]
        self.assertEqual(F(r['required']),F(1,2**104)+F(1,2**200))
        self.assertGreater(F(r['stored']),F(r['required']))

    def test_exact_identity(self):
        r=OBS['identity'];self.assertEqual(r['proof']['target'],r['reference'])
        self.assertEqual(r['checked']['added_continuous'],0)

    def test_equality_slack(self):
        r=OBS['equality_rhs'];self.assertEqual(r['checked']['added_continuous'],1)
        self.assertEqual(r['embeddings'][0]['target']['continuous'][-1],'-1')

    def test_guard_signs(self):
        r=OBS['guard_rhs']['checked']['inequalities'];self.assertTrue(all(F(x['stored'])>=F(x['required']) for x in r))
        self.reject('guard_rhs',lambda p:p['target']['state']['hz']['ub'].__setitem__(1,'-1'))

    def test_subnormal(self):
        r=OBS['subnormal'];self.assertEqual(r['checked']['added_continuous'],3)
        self.assertTrue(all(F(x['stored'])>0 for x in r['checked']['compensation']))

    def test_owner_and_binary_offset(self):
        p=OBS['shared_private']['proof'];self.assertEqual(p['maps']['flat'],[0,1,2,5,6])
        self.assertEqual(p['target']['ownership']['binary'],['expert0','expert1'])
        self.reject('shared_private',lambda p:p['maps'].__setitem__('flat',[0,1,2,3,4]))
        self.reject('shared_private',lambda p:p['target']['ownership']['continuous'].__setitem__(1,'expert1'))
        self.reject('shared_private',lambda p:p['target']['state']['binary_ids'].__setitem__(1,'expert0/z'))

    def test_zero_and_empty(self):
        r=OBS['zero_empty'];self.assertEqual(r['checked']['added_continuous'],0)
        self.assertEqual(r['snapshot']['Ac']['shape'],[1,0])
        self.reject('zero_empty',lambda p:p['target']['state']['hz']['b'].clear())

    def test_overflow(self):
        for ref in overflow_patterns():
            try:producer.produce(ref,expected_reference_sha256=identity(ref),owner='overflow',deadline=time.monotonic()+30)
            except ValueError as exc:
                self.assertIn('finite',str(exc));OVERFLOW.append({'reference':ref,'error':str(exc),'accepted':False})
            else:self.fail('overflow accepted')

    def test_layout_mutations(self):
        def altered(p,key,coef):
            h,c,b=unpack(p['target']['state']);h[key][0][len(c)-1]=F(coef);p['target']['state']=pack(h,c,b)
        self.reject('equality_rhs',lambda p:altered(p,'Gc',1))
        self.reject('equality_rhs',lambda p:altered(p,'Ac',-1))
        def constrained_error(p):
            h,c,b=unpack(p['target']['state']);h['Auc'][0][1]=F(1)
            p['target']['state']=pack(h,c,b)
        self.reject('subnormal',constrained_error,reason='zero layout')

    def test_compensation_mutations(self):
        self.reject('affine_sum',lambda p:p['output_errors'][0].__setitem__('radius','0'))
        self.reject('equality_rhs',lambda p:p['equality_errors'][0].__setitem__('radius','0'))
        self.reject('relu_endpoint',lambda p:p['target']['state']['hz']['c'].__setitem__(0,'-2'))
        self.reject('equality_rhs',lambda p:p['target']['state']['hz']['b'].__setitem__(0,'1/2'))
        ref=binary_reference();p=producer.produce(ref,expected_reference_sha256=identity(ref),owner='pair/0-1',deadline=time.monotonic()+30)
        extra={'reference':ref,'proof':p,'owner':'pair/0-1'}
        for key in ('Gb','Ab','Aub'):
            def change(p,key=key):
                h,c,b=unpack(p['target']['state']);h[key][0][0]=F(5);p['target']['state']=pack(h,c,b)
            self.reject('identity',change,record=extra,reason='inward')

    def test_external_binding(self):
        self.reject('identity',lambda p:p.__setitem__('reference_sha256','0'*64))
        self.reject('identity',lambda p:p.__setitem__('owner','other'))
        self.reject('identity',lambda r:r['ownership']['binary'].__setitem__(0,'other'),reference=True)
        self.reject('identity',lambda r:r['state']['hz']['c'].__setitem__(0,'1'),reference=True)

    def test_canonical_schema(self):
        self.reject('identity',lambda p:p['target']['state']['hz']['Gc']['shape'].__setitem__(1,0))
        self.reject('identity',lambda p:p['target']['state']['hz']['c'].__setitem__(0,float('inf')))
        self.reject('identity',lambda p:p['target']['state']['hz']['c'].__setitem__(0,'1/3'))
        self.reject('identity',lambda p:p['target']['state']['hz'].__setitem__('exact',True))
        self.reject('identity',lambda p:p['maps'].__setitem__('continuous',[False]))

    def test_deadline(self):
        r=OBS['identity'];past=time.monotonic()-1
        for fn,kw in [(producer.produce,dict(expected_reference_sha256=identity(r['reference']),owner=r['owner'])),
                      (checker.check,dict(proof=r['proof'],expected_reference_sha256=identity(r['reference']),expected_owner=r['owner']))]:
            with self.assertRaises(TimeoutError):fn(r['reference'],deadline=past,**kw)
        with self.assertRaises(TimeoutError):producer.instantiate(r['proof']['target'],expected_target_sha256=r['proof']['target_sha256'],deadline=past)
        # A last identity calculation that consumes the remaining time must fail.
        for module,fn,obj,kw in (
            (producer,producer.produce,r['reference'],{'expected_reference_sha256':identity(r['reference']),'owner':r['owner']}),
            (checker,checker.check,r['reference'],{'proof':r['proof'],'expected_reference_sha256':identity(r['reference']),'expected_owner':r['owner']}),
            (producer,producer.instantiate,r['proof']['target'],{'expected_target_sha256':r['proof']['target_sha256']})):
            now=time.monotonic();clockvalue=[now];seen=[0];original=module.identity
            def delayed(value):
                ans=original(value)
                if value is obj:
                    seen[0]+=1
                    if seen[0]==2:clockvalue[0]=now+31
                return ans
            with patch('scoped_source.graph.time.monotonic',side_effect=lambda:clockvalue[0]),patch.object(module,'identity',delayed),self.assertRaises(TimeoutError):
                fn(obj,deadline=now+30,**kw)

    def test_mutation_during_execution(self):
        r=deepcopy(OBS['identity']);saved=producer.nearest
        def polluted(q):
            r['reference']['state']['hz']['c'][0]='2';return saved(q)
        with patch.object(producer,'nearest',polluted),self.assertRaisesRegex(ValueError,'changed'):
            producer.produce(r['reference'],expected_reference_sha256=identity(r['reference']),owner=r['owner'],deadline=time.monotonic()+30)
        r=deepcopy(OBS['identity']);original=checker.clock;calls=0
        def timer(deadline):
            tick=original(deadline)
            def wrapped():
                nonlocal calls
                calls+=1
                if calls==1:r['reference']['state']['hz']['c'][0]='2'
                tick()
            return wrapped
        with patch.object(checker,'clock',timer),self.assertRaisesRegex(ValueError,'changed'):
            checker.check(r['reference'],r['proof'],expected_reference_sha256=identity(r['reference']),expected_owner=r['owner'],deadline=time.monotonic()+30)

    def test_checker_independence(self):
        r=OBS['affine_sum']
        with patch.object(producer,'produce',side_effect=AssertionError),patch.object(producer,'nearest',side_effect=AssertionError):
            checker.check(r['reference'],r['proof'],expected_reference_sha256=identity(r['reference']),expected_owner=r['owner'],deadline=time.monotonic()+30)
        tree=ast.parse(Path(checker.__file__).read_text())
        imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        self.assertEqual(imports,{'fractions','source_enclosure.format','scoped_source.graph'})
        # Reusing a real rejected input cannot stand in for another mutation.
        from scripts.run_hz_binary64 import NEGATIVE_SHA256,check_negative_binding
        original_key='test_canonical_schema:4'
        receipt=REJECTIONS[original_key]
        check_negative_binding(original_key,receipt)
        for key in NEGATIVE_SHA256:
            if key!=original_key:
                with self.assertRaisesRegex(ValueError,'fixed mutation receipt'):
                    check_negative_binding(key,receipt)

    def test_original_capacity(self):
        ref=owned(empty(1),[f'x{i}' for i in range(9)],[])
        with self.assertRaisesRegex(ValueError,'capacity'):
            producer.produce(ref,expected_reference_sha256=identity(ref),owner='capacity',deadline=time.monotonic()+30)


if __name__=='__main__':unittest.main()
