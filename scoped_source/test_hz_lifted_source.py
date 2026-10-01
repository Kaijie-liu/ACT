"""Frozen row composition, source bridges and four unchanged declarations."""
from copy import deepcopy
from fractions import Fraction as F
import inspect
import json
import time
import unittest
from unittest.mock import patch

from source_enclosure.format import empty,pack,unpack,identity
from source_enclosure import produce
from source_enclosure.check import check_affine
from scoped_source.test_hz_binary64 import patterns,owned
from scoped_source.check_hz_binary64 import check as kernel_check
from scoped_source.hz_binary64 import produce as kernel_produce
from scoped_source import hz_row_enclosure as rows
from scoped_source import check_hz_row_enclosure as checked_rows
from scoped_source import hz_lifted_source as api
from scoped_source import check_hz_lifted_source as checked
from scoped_source.endpoint_source_controls import cases

OBS={}


def sparse_reference(dense=False):
    h=empty(1);h['c']=[F(2)+F(1,2**54)]
    h['Gc']=[{j:F(1) for j in range(9)} if dense else {0:F(1)}]
    h['Auc']=[{i:F(1)} for i in range(6)];h['Aub']=[{} for _ in range(6)];h['ub']=[F(1)]*6
    return owned(h,[f'x{i}' for i in range(9)],[])


def bridge(name,deadline):
    from act.back_end.solver.solver_hz import sparse_hz_linear
    from act.back_end.solver.hz_lp_export import snapshot
    import numpy as np
    h=empty(1);tag='bridge/'+name
    if name=='affine':
        w=F(1)+F(1,2**52);h['Gc']=[{0:w,1:w/F(2**96)}]
        source=owned(h,['x','y'],[]);op=[{0:w}];bias=[F(0)]
        nominal=snapshot(sparse_hz_linear(api.live(source['state']),np.array([[float(w)]]),np.array([0.])))
        ref,cert=produce.affine(source['state'],op,bias,nominal,tag)
        record={'source':source,'operator':[{str(k):str(v) for k,v in r.items()} for r in op],
                'bias':list(map(str,bias)),'nominal':nominal,'certificate':cert}
    elif name=='relu':
        h['c']=[F(1,2)];h['Gc']=[{0:F(1),1:F(1,2**55)}]
        source=owned(h,['x','y'],[]);ref,cert=api.relu(source['state'],tag)
        record={'source':source,'certificate':cert}
    else:
        h['Gc']=[{0:F(1)}];source=owned(h,['x'],[])
        r=empty(3);r['c']=[F(1),F(1,2**54),F(0)];router=owned(r,['x'],[])
        ref=api.entry_for(source['state'],router['state'],(0,2),3)
        record={'source':source,'router':router}
    parent=record.get('router',source)
    _,lift=api.enclosure(api.owned(ref,parent,tag),tag,deadline);record['lift']=lift
    return record


def bridge_sources(name):
    """Fixed input declarations only; no propagation/lift/candidate operation."""
    h=empty(1)
    if name=='affine':
        w=F(1)+F(1,2**52);h['Gc']=[{0:w,1:w/F(2**96)}]
        return {'source':owned(h,['x','y'],[]),'operator':[{'0':str(w)}],'bias':['0']}
    if name=='relu':
        h['c']=[F(1,2)];h['Gc']=[{0:F(1),1:F(1,2**55)}]
        return {'source':owned(h,['x','y'],[])}
    h['Gc']=[{0:F(1)}];r=empty(3);r['c']=[F(1),F(1,2**54),F(0)]
    return {'source':owned(h,['x'],[]),'router':owned(r,['x'],[])}


def check_bridge(name,item,deadline):
    tag='bridge/'+name;source=item['source'];ref=item['lift']['reference']
    if name=='affine':
        checked.owners(source,ref,tag)
        op=[{int(k):F(v) for k,v in r.items()} for r in item['operator']]
        check_affine(source['state'],ref['state'],op,list(map(F,item['bias'])),item['nominal'],item['certificate'],tag)
    elif name=='relu':
        checked.owners(source,ref,tag);checked.relu_reference(source,ref,item['certificate'],tag)
    else:checked.route_reference(source,item['router'],ref,(0,2),3)
    _,result=checked.lift(item['lift'],tag,deadline);return result


def fixed_refusals(obs):
    """Exact fixed corruptions, derived from already independently bound cases.

    Sharing this test-input constructor shares no proof-acceptance computation.
    Per-key hashes bind each concrete query; receipts alone are not trusted.
    """
    out={}
    def source(key,modify,name='weighted_sign'):
        item=deepcopy(obs['sources'][name]);p=item['package'];modify(p)
        out[key]={'op':'source','source':item['source'],'package':p,'expired':False}
    for index in (0,1):
        source('reference:'+str(index),lambda p,i=index:p['pairs'][0]['a'][i]['lift']['reference']['state']['hz']['c'].__setitem__(0,'999'))
    source('layer_tag',lambda p:p['pairs'][0]['a'][0]['certificate'].__setitem__('tag','foreign'))
    source('guard',lambda p:p['pairs'][0]['entry']['reference']['state']['hz']['ub'].__setitem__(-1,'999'))
    source('owner',lambda p:p['pairs'][0]['a'][1]['lift']['reference']['ownership']['binary'].__setitem__(-1,'other'))
    source('layer_omission',lambda p:p['pairs'][0]['a'].pop())
    source('gate',lambda p:p['pairs'][0]['gate_evidence'].__setitem__('bounds',['1','1']))
    source('property',lambda p:p['endpoint_request']['properties'][0].__setitem__('offset','1'))
    source('pair_omission',lambda p:(p['pairs'].pop(),p['endpoint_request']['pairs'].pop()),'tied_partial_reuse')
    source('stale',lambda p:p.__setitem__('proof',deepcopy(obs['sources']['unresolved_sign']['package']['proof'])))
    source('expired',lambda p:None);out['expired']['expired']=True
    for change in ('map','owner','alias','global','omission'):
        item=obs['rows']['shared_private'];p=deepcopy(item['proof'])
        if change=='map':p['rows'][0]['continuous_map'][0]=1
        elif change=='owner':p['rows'][0]['local']['owner']='foreign'
        elif change=='alias':p['target']['state']['continuous_ids'][-1]=p['target']['state']['continuous_ids'][-2]
        elif change=='global':p['target']['state']['hz']['c'][0]='0'
        else:p['rows'].pop()
        p['target_sha256']=identity(p['target'])
        out['row:'+change]={'op':'row','reference':item['reference'],'proof':p}
    item=obs['rows']['equality_rhs'];p=deepcopy(item['proof']);h,c,b=unpack(p['target']['state'])
    h['Gc'][0][len(c)-1]=F(1);p['target']['state']=pack(h,c,b);p['target_sha256']=identity(p['target'])
    out['row:slack']={'op':'row','reference':item['reference'],'proof':p}
    for name in ('affine','relu','guard'):
        item=deepcopy(obs['bridges'][name])
        if name=='affine':item['certificate']['error_bounds'][0]='0'
        elif name=='relu':item['certificate']['ranges'][0]=['0','1']
        else:item['lift']['reference']['state']['hz']['ub'][0]='0'
        out['bridge:'+name]={'op':'bridge','name':name,'item':item}
    ref,_,_=checked_rows.projected(sparse_reference(True),'output',0)
    out['capacity:local']={'op':'local_capacity','reference':ref}
    out['capacity:global']={'op':'global_capacity','reference':owned(empty(1),[f'x{i}' for i in range(129)],[])}
    return out


def recheck_refusal(spec):
    anchor=identity(spec);deadline=time.monotonic()+300
    try:
        if spec['op']=='source':
            checked.check(spec['source'],spec['package'],expected_source_sha256=identity(spec['source']),
                          deadline=time.monotonic()-1 if spec['expired'] else deadline)
        elif spec['op']=='row':
            checked_rows.check(spec['reference'],spec['proof'],expected_reference_sha256=identity(spec['reference']),
                               expected_owner='pair/0-1',deadline=deadline)
        elif spec['op']=='bridge':check_bridge(spec['name'],spec['item'],deadline)
        elif spec['op']=='global_capacity':checked_rows.parse(spec['reference'])
        elif spec['op']=='local_capacity':
            from scoped_source.check_hz_binary64 import parse
            parse(spec['reference'])
        else:raise AssertionError('unknown fixed refusal')
    except (ValueError,TypeError,KeyError,TimeoutError) as exc:
        if isinstance(exc,TimeoutError) and not (spec['op']=='source' and spec['expired']):
            raise
        if identity(spec)!=anchor:raise AssertionError('negative query mutated')
        return {'input_sha256':anchor,'error_type':type(exc).__name__,'error':str(exc)}
    raise AssertionError('fixed negative input accepted')


class LiftedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OBS.clear();OBS.update(sources={},rows={},bridges={})
        from scripts.run_hz_lifted_source import SOURCES
        for name in SOURCES:
            start=time.monotonic();deadline=start+300
            doc=next(d for n,d,_ in cases() if n==name)
            cost={'source_creation':time.monotonic()-start};record={'source':doc,'start':start,'deadline':deadline}
            OBS['sources'][name]=record
            stage='construction'
            def observed(k,v):
                nonlocal stage
                cost[k]=v;stage='proposals' if k=='construction' else 'serialization'
            try:
                p=api.build(doc,expected_source_sha256=identity(doc),deadline=deadline,
                            observe=observed)
                record['package']=p;t=time.monotonic();raw=json.dumps(p,sort_keys=True,allow_nan=False)
                p=json.loads(raw);cost['serialization']=time.monotonic()-t;t=time.monotonic();stage='checking'
                record['checked']=checked.check(doc,p,expected_source_sha256=identity(doc),deadline=deadline)
                cost['checking']=time.monotonic()-t;record['end']=time.monotonic();cost['total']=record['end']-start
                record['cost']=cost;record['serialized_bytes']=len(raw.encode())
            except Exception as exc:
                record['error']=repr(exc);record['end']=time.monotonic();cost['total']=record['end']-start
                record['cost']=cost;record['unfinished_stage']=stage
                record['uncompleted_stage_seconds']=max(0,cost['total']-sum(v for k,v in cost.items() if k!='total'))
                raise
        for name,(ref,_) in patterns().items():
            deadline=time.monotonic()+300;p=rows.produce(ref,expected_reference_sha256=identity(ref),owner='pair/0-1',deadline=deadline)
            OBS['rows'][name]={'reference':ref,'proof':p,'checked':checked_rows.check(ref,p,
                expected_reference_sha256=identity(ref),expected_owner='pair/0-1',deadline=deadline)}
        for name in ('affine','relu','guard'):
            deadline=time.monotonic()+300;item=bridge(name,deadline)
            item['checked']=check_bridge(name,item,deadline);OBS['bridges'][name]=item

    def item(self,name='weighted_sign'):
        v=OBS['sources'][name];return v['source'],deepcopy(v['package'])

    def accept(self,doc,p):
        return checked.check(doc,p,expected_source_sha256=identity(doc),deadline=time.monotonic()+300)

    def reject(self,doc,p):
        with self.assertRaises((ValueError,KeyError,TypeError)):self.accept(doc,p)

    def test_row_kernel_differential(self):
        for item in OBS['rows'].values():
            ref=item['reference'];p=kernel_produce(ref,expected_reference_sha256=identity(ref),owner='pair/0-1',deadline=time.monotonic()+300)
            kernel_check(ref,p,expected_reference_sha256=identity(ref),expected_owner='pair/0-1',deadline=time.monotonic()+300)
            self.assertEqual(p['target'],item['proof']['target'])
            item['kernel_proof']=p

    def test_shared_global_identity(self):
        ref=sparse_reference();p=rows.produce(ref,expected_reference_sha256=identity(ref),owner='global',deadline=time.monotonic()+300)
        result=checked_rows.check(ref,p,expected_reference_sha256=identity(ref),expected_owner='global',deadline=time.monotonic()+300)
        self.assertEqual(result['rows'],7);self.assertEqual(result['added_continuous'],1)
        self.assertEqual(p['target']['state']['continuous_ids'][:9],ref['state']['continuous_ids'])
        OBS['sparse_global']={'reference':ref,'proof':p,'checked':result}

    def test_row_binding_mutations(self):
        item=OBS['rows']['shared_private'];ref=item['reference']
        for change in ('map','owner','alias','global','omission'):
            p=deepcopy(item['proof'])
            if change=='map':p['rows'][0]['continuous_map'][0]=1
            elif change=='owner':p['rows'][0]['local']['owner']='foreign'
            elif change=='alias':p['target']['state']['continuous_ids'][-1]=p['target']['state']['continuous_ids'][-2]
            elif change=='global':p['target']['state']['hz']['c'][0]='0'
            else:p['rows'].pop()
            p['target_sha256']=identity(p['target'])
            with self.assertRaises(ValueError):checked_rows.check(ref,p,expected_reference_sha256=identity(ref),expected_owner='pair/0-1',deadline=time.monotonic()+300)
        item=OBS['rows']['equality_rhs'];p=deepcopy(item['proof']);h,c,b=unpack(p['target']['state'])
        h['Gc'][0][len(c)-1]=F(1);p['target']['state']=pack(h,c,b);p['target_sha256']=identity(p['target'])
        with self.assertRaisesRegex(ValueError,'assembly'):
            checked_rows.check(item['reference'],p,expected_reference_sha256=identity(item['reference']),expected_owner='pair/0-1',deadline=time.monotonic()+300)

    def test_row_capacity_refusal(self):
        ref=sparse_reference(True)
        with self.assertRaisesRegex(ValueError,'capacity'):
            rows.produce(ref,expected_reference_sha256=identity(ref),owner='dense',deadline=time.monotonic()+300)
        ref=owned(empty(1),[f'x{i}' for i in range(129)],[])
        with self.assertRaisesRegex(ValueError,'capacity'):checked_rows.parse(ref)

    def test_complete_four_sources(self):
        from scripts.run_hz_lifted_source import SOURCES
        self.assertEqual(set(OBS['sources']),set(SOURCES))
        for name,spec in SOURCES.items():
            item=OBS['sources'][name];result=item['checked']
            self.assertEqual(identity(item['source']),spec[0]);self.assertEqual(result['required'],spec[1])
            self.assertEqual(result['checked_endpoints'],spec[2]);self.assertEqual(result['missing_endpoints'],0)
        self.assertEqual(OBS['sources']['unsafe_tied']['checked']['positive'],0)

    def test_source_reference_mutations(self):
        for index in (0,1):
            d,p=self.item();p['pairs'][0]['a'][index]['lift']['reference']['state']['hz']['c'][0]='999'
            self.reject(d,p)
        d,p=self.item();p['pairs'][0]['a'][0]['certificate']['tag']='foreign';self.reject(d,p)

    def test_guard_ownership(self):
        d,p=self.item();p['pairs'][0]['entry']['reference']['state']['hz']['ub'][-1]='999';self.reject(d,p)
        d,p=self.item();p['pairs'][0]['a'][1]['lift']['reference']['ownership']['binary'][-1]='other';self.reject(d,p)

    def test_layer_inventory(self):
        d,p=self.item();p['pairs'][0]['a'].pop();self.reject(d,p)

    def test_gate_property(self):
        d,p=self.item();p['pairs'][0]['gate_evidence']['bounds']=['1','1'];self.reject(d,p)
        d,p=self.item();p['endpoint_request']['properties'][0]['offset']='1';self.reject(d,p)

    def test_missing_pair(self):
        d,p=self.item('tied_partial_reuse');p['pairs'].pop();p['endpoint_request']['pairs'].pop();self.reject(d,p)

    def test_partial_evidence(self):
        d,p=self.item('tied_partial_reuse');p['proof']['pairs'][-1]['candidates']=None;r=self.accept(d,p)
        self.assertEqual(r['status'],'UNKNOWN_MISSING_EVIDENCE');self.assertGreater(r['missing_endpoints'],0)
        OBS['partial']={'source':d,'package':p,'checked':r}

    def test_stale_evidence(self):
        d,p=self.item();p['proof']=deepcopy(OBS['sources']['unresolved_sign']['package']['proof']);self.reject(d,p)

    def test_rounding_affine(self):
        r=OBS['bridges']['affine'];self.assertGreater(r['checked']['added_continuous'],0)
        self.assertEqual(F(r['certificate']['error_bounds'][0]),F(1,2**104)+F(1,2**200))
        changed=deepcopy(r);changed['certificate']['error_bounds'][0]='0'
        with self.assertRaises(ValueError):check_bridge('affine',changed,time.monotonic()+300)

    def test_rounding_relu(self):
        r=OBS['bridges']['relu'];self.assertGreater(r['checked']['added_continuous'],0)
        changed=deepcopy(r);changed['certificate']['ranges'][0]=['0','1']
        with self.assertRaises(ValueError):check_bridge('relu',changed,time.monotonic()+300)

    def test_rounding_guard(self):
        r=OBS['bridges']['guard'];a=r['lift']['reference']['state']['hz']['ub'];b=r['lift']['enclosure']['target']['state']['hz']['ub']
        self.assertNotEqual(a,b)
        changed=deepcopy(r);changed['lift']['reference']['state']['hz']['ub'][0]='0'
        with self.assertRaises(ValueError):check_bridge('guard',changed,time.monotonic()+300)

    def test_expiry_mutation(self):
        d,p=self.item()
        with self.assertRaises(TimeoutError):checked.check(d,p,expected_source_sha256=identity(d),deadline=time.monotonic()-1)
        with self.assertRaises(TimeoutError):api.build(d,expected_source_sha256=identity(d),deadline=time.monotonic()-1)
        changed=deepcopy(d);changed['request']['margin']='1'
        with self.assertRaises(ValueError):checked.check(changed,p,expected_source_sha256=identity(d),deadline=time.monotonic()+300)
        original=checked.check_request
        def polluted(*args,**kwargs):
            result=original(*args,**kwargs);p['source_sha256']='0'*64;return result
        with patch.object(checked,'check_request',side_effect=polluted),self.assertRaisesRegex(ValueError,'changed'):
            self.accept(d,p)
        # The final actual snapshot must fit inside the same cooperative clock.
        ref=OBS['rows']['identity']['reference'];now=time.monotonic();at=[now];original=api.snapshot
        def delayed(value):
            result=original(value);at[0]=now+31;return result
        with patch('scoped_source.graph.time.monotonic',side_effect=lambda:at[0]), \
                patch.object(api,'snapshot',side_effect=delayed),self.assertRaises(TimeoutError):
            api.enclosure(ref,'deadline',now+30)

    def test_checker_independence(self):
        d,p=self.item()
        with patch.object(api,'propagate',side_effect=AssertionError),patch.object(rows,'produce',side_effect=AssertionError), \
             patch('act.back_end.moe.hz_endpoints.propose_request',side_effect=AssertionError):
            self.assertEqual(self.accept(d,p),OBS['sources']['weighted_sign']['checked'])
        for module in (checked,checked_rows):
            text=inspect.getsource(module)
            for bad in ('import torch','import scipy','import numpy','from scoped_source.hz_lifted_source import','from scoped_source.hz_row_enclosure import'):
                self.assertNotIn(bad,text)

    def test_cost_inventory(self):
        from scripts.run_hz_lifted_source import check_cost
        for item in OBS['sources'].values():
            check_cost(item);bad=deepcopy(item);bad['cost'].pop('construction')
            with self.assertRaises(ValueError):check_cost(bad)
        OBS['refusals']={key:recheck_refusal(spec) for key,spec in fixed_refusals(OBS).items()}
        self.assertEqual(len(OBS['refusals']),22)
        self.assertEqual(len({r['input_sha256'] for r in OBS['refusals'].values()}),22)
        spec=fixed_refusals(OBS)['reference:0']
        with patch.object(checked,'check',side_effect=TimeoutError('before intended refusal')),self.assertRaises(TimeoutError):
            recheck_refusal(spec)


if __name__=='__main__':unittest.main()
