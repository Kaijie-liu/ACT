"""Frozen block support/source controls. Same four declarations, no real runs."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction as F
import json
import time
import unittest
from unittest.mock import patch

import torch

from scoped_source.endpoint_source_controls import cases
from scoped_source import hz_templates as old, check_hz_templates as old_check
from scoped_source import hz_block_source as api, check_hz_block_source as receive
from act.back_end.moe import block_support as kernel, check_block_support as exact
from act.back_end.moe import hz_endpoints, batched_support, weighted_top2
from act.back_end.solver import hz_lp_export
from act.back_end.moe.test_hz_endpoints import hz, relation, CONTEXT
from scoped_source.block_differential import compare_sources, compare_pair
from scoped_source.rowwise_bound import identity, clock
from scoped_source.rowwise_bound import check_bound, rational

OBS={}


def boundaries(stack):
    # Observe all old construction entry sites; normal block paths call none.
    return {name:stack.enter_context(patch.object(mod,attr,wraps=getattr(mod,attr))) for name,mod,attr in
            [('joint',hz_endpoints,'shared_input_pair_hz'),('remap',weighted_top2,'_remap_columns'),
             ('export',batched_support,'export'),('direct_export',hz_lp_export,'export'),
             ('old_source_check',old_check,'check')]}


def run_arm(name,arm):
    start=time.monotonic();deadline=start+300;mod=old if arm=='materialized' else api
    checker=old_check if arm=='materialized' else receive
    doc=next(d for n,d,_ in cases() if n==name);cost={'source_creation':time.monotonic()-start}
    item={'source':doc,'start':start,'deadline':deadline,'cost':cost};OBS['cases'].setdefault(name,{})[arm]=item
    stage='construction'
    def observe(k,v):
        nonlocal stage
        cost[k]=v;stage='proposals' if k=='construction' else 'serialization'
    try:
        with ExitStack() as stack:
            calls=stack.enter_context(patch.object(mod,'propagate',wraps=mod.propagate));counts=boundaries(stack)
            p=mod.build(doc,expected_source_sha256=identity(doc),deadline=deadline,observe=observe)
            item['dispatch']=[c.args[1] for c in calls.call_args_list];item['package']=p
            t=time.monotonic();raw=json.dumps(p,sort_keys=True,allow_nan=False);p=json.loads(raw)
            cost['serialization']=time.monotonic()-t;stage='checking';t=time.monotonic()
            item['checked']=checker.check(doc,p,expected_source_sha256=identity(doc),deadline=deadline)
            cost['checking']=time.monotonic()-t
            item['boundaries']={k:v.call_count for k,v in counts.items()}
        item['end']=time.monotonic();cost['total']=item['end']-start;item['serialized_bytes']=len(raw.encode())
    except Exception as exc:
        item['error']=repr(exc);item['end']=time.monotonic();cost['total']=item['end']-start
        item['unfinished_stage']=stage
        item['uncompleted_stage_seconds']=max(0,cost['total']-sum(v for k,v in cost.items() if k!='total'));raise
    return item


def given(name):
    if name=='private':
        entry=hz([0],[[1]],[[1]],ac=[[1]],ab=[[1]],b=[0])
        a=hz([.25,0],[[1,1],[0,0]],[[1,1],[0,0]],ac=[[1,0]],ab=[[1,0]],b=[0]);b=deepcopy(a)
    else:
        pairs,_,_,_=relation();p=pairs[0];entry,a,b=p['entry'],p['a'],p['b']
    queries=[{'id':'lower','q':['1/2','-1/2','1/2','-1/2'],'offset':'0','side':'min'},
             {'id':'upper_offset','q':['1/2','-1/2','1/2','-1/2'],'offset':'1/7','side':'max'}]
    batch=kernel.prepare_batch(entry,entry,a,b,queries,context=CONTEXT,deadline=time.monotonic()+300)
    parts,targets=exact.validate_batch(batch,expected_batch_sha256=identity(batch),deadline=time.monotonic()+300)
    entries=[]
    for q,target in zip(batch['queries'],targets):
        dual=[{'source':p['source'],'y':[0]*len(p['b']),'t':[0]*len(p['h'])} for p in parts]
        val,_=exact.evaluate(parts,target,dual,clock(time.monotonic()+300))
        entries.append({'id':q['id'],'duals':dual,'claimed_lower_bound':str(val),'zero_candidate_lower_bound':str(val)})
    proof={'batch_sha256':identity(batch),'entries':entries,'algorithm':exact.ALGORITHM,'iterations':128,'dtype':'float64','device':'cpu'}
    joint=weighted_top2.shared_input_pair_hz(entry,a,b).output_hz
    ob=batched_support.prepare_batch(joint,queries,context=CONTEXT,deadline=time.monotonic()+300)
    # Corresponding zero duals for this given-HZ fixture, not extra optimizer calls.
    flat=[]
    from act.back_end.moe.check_batched_support import validated_records
    for (q,lp),be in zip(validated_records(ob,expected_batch_sha256=identity(ob),deadline=time.monotonic()+300),entries):
        flat.append({'id':q['id'],'certificate':{'lp_sha256':identity(lp),'inequality_dual':[0]*len(lp['b']),
                    'equality_dual':[0]*len(lp['h']),'claimed_lower_bound':be['claimed_lower_bound']}})
    left={'pair':[0,1],'gate':{'bounds':['1/2','1/2']},'batch':ob}
    right={'pair':[0,1],'gate':{'bounds':['1/2','1/2']},'batch':batch}
    flat_candidate={'batch_sha256':identity(ob),'entries':flat}
    comparison=compare_pair(left,right,flat_candidate,proof,time.monotonic()+300)
    checked=exact.check_batch(batch,proof,expected_batch_sha256=identity(batch),deadline=time.monotonic()+300)
    extra={}
    if name=='relation':
        independent=weighted_top2.independent_input_pair_hz(entry,a,b).output_hz
        ib=batched_support.prepare_batch(independent,queries,context=CONTEXT,deadline=time.monotonic()+300)
        certs=[];results=[]
        for q,lp in validated_records(ib,expected_batch_sha256=identity(ib),deadline=time.monotonic()+300):
            val=rational(lp['offset'])-sum(map(lambda v:abs(rational(v)),lp['c']),F(0))
            cert={'lp_sha256':identity(lp),'inequality_dual':[0]*len(lp['b']),
                  'equality_dual':[0]*len(lp['h']),'claimed_lower_bound':str(val)}
            certs.append(cert);results.append(check_bound(lp,cert,deadline=time.monotonic()+300))
        extra={'independent_batch':ib,'certificates':certs,'results':results}
    else:
        nonzero=deepcopy(proof)
        parts,targets=exact.validate_batch(batch,expected_batch_sha256=identity(batch),deadline=time.monotonic()+300)
        for item,target in zip(nonzero['entries'],targets):
            item['duals'][0]['t']=[1]
            val,_=exact.evaluate(parts,target,item['duals'],clock(time.monotonic()+300));item['claimed_lower_bound']=str(val)
        extra={'nonzero_candidate':nonzero,'differential':compare_pair(left,right,flat_candidate,nonzero,time.monotonic()+300)}
    return {'batch':batch,'proof':proof,'checked':checked,'flat':left,'flat_candidate':flat_candidate,'comparison':comparison,'extra':extra}


def negative_queries(obs):
    out={};parent=obs['cases']['weighted_sign']['block']
    def source(key,change):
        p=deepcopy(parent['package']);change(p);out[key]={'kind':'source','source':parent['source'],'package':p,'expired':False}
    def batch(key,change,given_name=None):
        if given_name:item=obs['given'][given_name];b,p=deepcopy(item['batch']),deepcopy(item['proof'])
        else:b=deepcopy(parent['package']['endpoint_request']['pairs'][0]['batch']);p=deepcopy(parent['package']['proof']['pairs'][0]['candidates'])
        change(b,p);p['batch_sha256']=identity(b)
        out[key]={'kind':'batch','batch':b,'candidate':p,'expired':False}
    source('template_alias',lambda p:p['templates'][1]['trace'][-1]['lift']['enclosure']['target']['state']['continuous_ids'].__setitem__(-1,'foreign'))
    source('source_binding',lambda p:p.__setitem__('source_sha256','0'*64))
    source('gate',lambda p:p['pairs'][0]['gate_evidence'].__setitem__('bounds',['0','1']))
    source('property',lambda p:p['endpoint_request']['properties'][0].__setitem__('offset','1'))
    source('missing_pair',lambda p:(p['pairs'].pop(),p['endpoint_request']['pairs'].pop()))
    source('missing_target',lambda p:p['proof']['pairs'][0]['candidates']['entries'].pop())
    batch('span',lambda b,p:b['layout']['blocks'][1].__setitem__('inequality_start',999))
    batch('private_alias',lambda b,p:b['layout']['blocks'][2]['columns'].__setitem__(-1,0),'private')
    batch('binary_offset',lambda b,p:b['layout']['blocks'][0]['columns'].__setitem__(-1,1),'private')
    batch('common_output',lambda b,p:b['sources']['entry']['c'].__setitem__(0,99))
    batch('common_prefix',lambda b,p:b['sources']['a']['b'].__setitem__(0,1),'private')
    batch('zero_dual_row',lambda b,p:b['sources']['entry']['Ac']['indices'].__setitem__(0,99),'private')
    batch('objective',lambda b,p:b['queries'][0]['c'].__setitem__(0,'99'))
    batch('offset',lambda b,p:b['queries'][0].__setitem__('constant','99'))
    batch('dual_sign',lambda b,p:p['entries'][0]['duals'][0]['y'].__setitem__(0,1))
    batch('nonfinite',lambda b,p:p['entries'][0]['duals'][0]['y'].__setitem__(0,'NaN'))
    batch('claim',lambda b,p:p['entries'][0].__setitem__('claimed_lower_bound','99999'))
    batch('iteration',lambda b,p:p.__setitem__('iterations',127))
    batch('device',lambda b,p:p.__setitem__('device','cuda'))
    source('expired',lambda p:None);out['expired']['expired']=True
    return out


def check_given(name,item):
    wanted=given(name)
    if name=='relation':
        # Independent builder owns a process-local frame. Verify it is disjoint,
        # then rename only that namespace when comparing fixed coefficients.
        batch=item['extra']['independent_batch'];frame=batch['source']['frame_id']
        if type(frame) is not int or frame==item['batch']['sources']['entry']['frame_id']:
            raise ValueError('independent source frame alias')
        other=wanted['extra']['independent_batch'];other['source']['frame_id']=frame
        other['source_sha256']=identity(other['source'])
    if wanted!=item:raise ValueError('fixed given-HZ fixture/differential')


def recheck_negative(spec):
    anchor=identity(spec);end=time.monotonic()+(-1 if spec['expired'] else 300)
    try:
        if spec['kind']=='source':receive.check(spec['source'],spec['package'],expected_source_sha256=identity(spec['source']),deadline=end)
        else:exact.check_batch(spec['batch'],spec['candidate'],expected_batch_sha256=identity(spec['batch']),deadline=end)
    except (ValueError,TypeError,KeyError,TimeoutError) as exc:
        if isinstance(exc,TimeoutError) and not spec['expired']:raise
        if identity(spec)!=anchor:raise AssertionError('mutation during rejection')
        return {'input_sha256':anchor,'error_type':type(exc).__name__,'error':str(exc)}
    raise AssertionError('negative query accepted')


class BlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OBS.clear();OBS.update(cases={},given={},negatives={},auxiliary={},diagnostics={})
        for i,(name,_,_) in enumerate(cases()):
            for arm in (('materialized','block') if i%2==0 else ('block','materialized')):run_arm(name,arm)
        start=time.monotonic()
        for name in ('private','relation'):OBS['given'][name]=given(name)
        OBS['diagnostics']['given']={'start':start,'end':time.monotonic()}
        item=OBS['cases']['tied_partial_reuse']['block'];p=deepcopy(item['package']);p['proof']['pairs'][-1]['candidates']=None
        OBS['partial']={'source':item['source'],'package':p,'checked':receive.check(item['source'],p,expected_source_sha256=identity(item['source']),deadline=time.monotonic()+300)}

    def reject(self,*keys):
        for k in keys:OBS['negatives'][k]=recheck_negative(negative_queries(OBS)[k])

    def test_complete_sources(self):
        for arms in OBS['cases'].values():
            for item in arms.values():
                self.assertEqual(item['checked']['missing_endpoints'],0)
                self.assertEqual(item['dispatch'],['router']+[f'expert{i}' for i in range(item['source']['request']['experts'])])
        self.assertEqual(OBS['cases']['unsafe_tied']['block']['checked']['positive'],0)

    def test_numeric_differential(self):
        start=time.monotonic();OBS['differentials']={name:compare_sources(a['materialized']['package'],a['block']['package'],start+300) for name,a in OBS['cases'].items()}
        OBS['diagnostics']['differential']={'start':start,'end':time.monotonic()}
        self.assertEqual(sum(map(len,OBS['differentials'].values())),15)

    def test_candidate_crosscheck(self):
        for arms in OBS['cases'].values():
            for arm in arms:
                for p in arms[arm]['package']['proof']['pairs']:
                    self.assertEqual(p['candidates']['iterations'],128)
        # Old checker accepts a roster in any order. Crosscheck by ID, not position.
        x=OBS['cases']['weighted_sign'];oldp=x['materialized']['package'];newp=x['block']['package']
        args=[oldp['endpoint_request']['pairs'][0],newp['endpoint_request']['pairs'][0],
              deepcopy(oldp['proof']['pairs'][0]['candidates']),newp['proof']['pairs'][0]['candidates']]
        expected=compare_pair(*args,time.monotonic()+300);args[2]['entries'].reverse()
        self.assertEqual(compare_pair(*args,time.monotonic()+300),expected)
        args[2]['entries'][0]['certificate']['lp_sha256']='0'*64
        with self.assertRaises(ValueError):compare_pair(*args,time.monotonic()+300)

    def test_binary_mapping(self):
        b=OBS['given']['private']['batch'];self.assertEqual((b['layout']['n_cont'],b['layout']['n_bin']),(3,3))
        self.assertEqual(b['layout']['blocks'][2]['columns'],[0,2,3,5]);self.reject('binary_offset')

    def test_shared_relation(self):
        values=OBS['given']['relation']['checked']['results']
        self.assertEqual(F(values[0]['bound']),F(1,4));self.assertEqual(F(values[1]['bound']),F(1,4)+F(1,7))
        independent=OBS['given']['relation']['extra']
        self.assertEqual(F(independent['results'][0]['checked_lower_bound']),F(-3,4))
        self.assertEqual(independent['independent_batch']['source']['Gc']['shape'],[4,2])

    def test_common_prefix(self):self.reject('common_output','common_prefix')
    def test_private_spans(self):self.reject('span','private_alias','template_alias')
    def test_zero_dual_rows(self):self.reject('zero_dual_row')
    def test_property_offset(self):self.reject('objective','offset','property')
    def test_missing_evidence(self):
        self.reject('missing_pair','missing_target');self.assertEqual(OBS['partial']['checked']['status'],'UNKNOWN_MISSING_EVIDENCE')
    def test_candidate_mutation(self):self.reject('dual_sign','nonfinite','claim')
    def test_source_gate(self):self.reject('source_binding','gate')

    def test_no_joint(self):
        for arms in OBS['cases'].values():
            self.assertTrue(all(v==0 for v in arms['block']['boundaries'].values()))
            self.assertGreater(arms['materialized']['boundaries']['joint'],0)

    def test_fixed_algorithm(self):
        self.reject('iteration','device');b=OBS['given']['private']['batch']
        with self.assertRaises(ValueError):kernel.propose_batch(b,expected_batch_sha256=identity(b),deadline=time.monotonic()+300,device='cuda')

    def test_producer_faults(self):
        for name in ('expired_producer','mutated_source'):
            start=time.monotonic();deadline=start+(-1 if name=='expired_producer' else 300)
            doc=deepcopy(OBS['cases']['weighted_sign']['block']['source']);anchor=identity(doc)
            item={'start':start,'deadline':deadline,'source_sha256':anchor,'cost':{'source_creation':time.monotonic()-start},
                  'injection_reached':False,'actual_candidate_calls':0};OBS['auxiliary'][name]=item
            def inject(*args,**kwargs):item['injection_reached']=True;doc['request']['radius']='1/2';return None
            error=None
            with patch.object(api,'propose_request',side_effect=inject),patch.object(api,'propagate',wraps=api.propagate) as calls:
                try:api.build(doc,expected_source_sha256=anchor,deadline=deadline,observe=lambda k,v:item['cost'].__setitem__(k,v))
                except (ValueError,TimeoutError) as exc:error=exc
                finally:
                    item['end']=time.monotonic();item['cost']['total']=item['end']-start;item['error_type']=type(error).__name__;item['error']=str(error)
                    item['dispatch']=[c.args[1] for c in calls.call_args_list];item['observed_source_sha256']=identity(doc)
            self.assertIsInstance(error,TimeoutError if name=='expired_producer' else ValueError)
            self.assertEqual(item['injection_reached'],name=='mutated_source')
            if name=='mutated_source':self.assertEqual(str(error),'source changed during generation')

    def test_checker_expiry(self):
        self.reject('expired')
        with patch.object(receive,'check',side_effect=TimeoutError('unexpected')),self.assertRaises(TimeoutError):
            recheck_negative(negative_queries(OBS)['gate'])

    def test_checker_independence(self):
        with patch.object(api,'build',side_effect=AssertionError),patch.object(api,'propagate',side_effect=AssertionError),\
                patch.object(kernel,'prepare_batch',side_effect=AssertionError),patch.object(kernel,'propose_batch',side_effect=AssertionError),\
                patch.object(old_check,'check',side_effect=AssertionError):
            for arms in OBS['cases'].values():
                item=arms['block'];self.assertEqual(receive.check(item['source'],item['package'],expected_source_sha256=identity(item['source']),deadline=time.monotonic()+300),item['checked'])

    def test_cost_inventory(self):
        from scripts.run_hz_lifted_source import check_cost
        from scripts.run_hz_blocks import check_inventory,finite_times,SOURCES
        for arms in OBS['cases'].values():
            for item in arms.values():check_cost(item)
        item=deepcopy(OBS['cases']['weighted_sign']['block']);item['cost'].pop('serialization')
        with self.assertRaises(ValueError):check_cost(item)
        inventory={'cases':dict.fromkeys(SOURCES),'given':dict.fromkeys(('private','relation')),
                   'differentials':dict.fromkeys(SOURCES),'diagnostics':dict.fromkeys(('given','differential')),
                   'auxiliary':dict.fromkeys(('expired_producer','mutated_source')),'partial':{},'negatives':{}}
        check_inventory(inventory)
        for group in ('cases','given','differentials','diagnostics','auxiliary'):
            missing=deepcopy(inventory);missing[group].pop(next(iter(missing[group])))
            with self.assertRaises(ValueError):check_inventory(missing)
        for bad in ({'start':float('nan'),'end':2},{'start':0,'end':3},{'start':2,'end':1}):
            with self.assertRaises(ValueError):finite_times(bad,0,2)
