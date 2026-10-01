"""Frozen same-source template/per-pair propagation controls, not benchmarks."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import json
import time
import unittest
from unittest.mock import patch

from source_enclosure.format import pack, unpack, identity
from scoped_source.endpoint_source_controls import cases
from scoped_source import hz_lifted_source as old
from scoped_source import check_hz_lifted_source as old_check
from scoped_source import hz_templates as api
from scoped_source import check_hz_templates as receive
from scoped_source.test_hz_lifted_source import bridge, bridge_sources, check_bridge

OBS = {}


def numeric_comparison(a, b):
    """Ignore identities only; compare every numerical source/base/objective."""
    x, y = a['endpoint_request'], b['endpoint_request']
    if (x['experts'], x['classes'], x['properties']) != (y['experts'], y['classes'], y['properties']):
        raise ValueError('differential source property roster')
    if [p['pair'] for p in x['pairs']] != [p['pair'] for p in y['pairs']]:
        raise ValueError('differential pair roster')
    records = []
    for p,q in zip(x['pairs'], y['pairs']):
        # These snapshots are numeric with canonical source factor ordering.
        for k in ('entry','a','b'):
            if p['sources'][k] != q['sources'][k]: raise ValueError('numeric source differential')
        for k in ('source','base','queries','relaxation','n_relaxed_binaries'):
            if p['batch'][k] != q['batch'][k]: raise ValueError('numeric support differential')
        if p['gate']['bounds'] != q['gate']['bounds']: raise ValueError('gate differential')
        records.append({'pair':p['pair'], 'source_sha256':identity(p['sources']),
                        'base_sha256':identity(p['batch']['base']),
                        'queries_sha256':identity(p['batch']['queries'])})
    return records


def dispatch_roster(e, arm):
    if arm == 'template': return ['router']+[f'expert{i}' for i in range(e)]
    return ['router']+[f'expert{i}' for pair in combinations(range(e),2) for i in pair]


def run_arm(name, arm):
    start=time.monotonic(); deadline=start+300; mod=api if arm=='template' else old
    checker=receive if arm=='template' else old_check
    doc=next(d for n,d,_ in cases() if n==name)
    cost={'source_creation':time.monotonic()-start}; stage='construction'
    item={'source':doc,'start':start,'deadline':deadline,'cost':cost}
    OBS['cases'].setdefault(name,{})[arm]=item
    def observe(k,v):
        nonlocal stage
        cost[k]=v;stage='proposals' if k=='construction' else 'serialization'
    try:
        with patch.object(mod,'propagate',wraps=mod.propagate) as calls:
            p=mod.build(doc,expected_source_sha256=identity(doc),deadline=deadline,observe=observe)
            item['dispatch']=[c.args[1] for c in calls.call_args_list]
        item['package']=p;t=time.monotonic();raw=json.dumps(p,sort_keys=True,allow_nan=False)
        p=json.loads(raw);cost['serialization']=time.monotonic()-t;t=time.monotonic();stage='checking'
        item['checked']=checker.check(doc,p,expected_source_sha256=identity(doc),deadline=deadline)
        cost['checking']=time.monotonic()-t;item['end']=time.monotonic();cost['total']=item['end']-start
        item['serialized_bytes']=len(raw.encode())
    except Exception as exc:
        item['error']=repr(exc);item['end']=time.monotonic();cost['total']=item['end']-start
        item['unfinished_stage']=stage
        item['uncompleted_stage_seconds']=max(0,cost['total']-sum(v for k,v in cost.items() if k!='total'))
        raise
    return item


def view_bridge(name, deadline):
    item=bridge(name,deadline); item['checked']=check_bridge(name,item,deadline)
    base=item['source']; terminal=item['lift']['enclosure']['target']
    if name=='guard': entry=terminal; template=base
    else:
        template=terminal; h,c,b=unpack(base['state']);h['Auc'].append({0:F(1)})
        h['Aub'].append({});h['ub'].append(F(1,2))
        entry=deepcopy(base);entry['state']=pack(h,c,b)
    view=api.specialize(base,template,entry,pair=[0,1],expert=0,deadline=deadline)
    checked=receive.check_view(base,template,entry,view,pair=[0,1],expert=0,deadline=deadline)
    return {'operator':item,'common':base,'template':template,'entry':entry,'view':view,'checked':checked}


def check_view_bridge(name, item, deadline):
    op=item['operator']
    if any(op.get(k)!=v for k,v in bridge_sources(name).items()): raise ValueError('fixed bridge input')
    if check_bridge(name,op,deadline)!=op['checked']: raise ValueError('bridge lowering')
    base=op['source']; terminal=op['lift']['enclosure']['target']
    if name=='guard': entry=terminal; template=base
    else:
        template=terminal; h,c,b=unpack(base['state']);h['Auc'].append({0:F(1)})
        h['Aub'].append({});h['ub'].append(F(1,2));entry=deepcopy(base);entry['state']=pack(h,c,b)
    if (base,template,entry)!=(item['common'],item['template'],item['entry']): raise ValueError('bridge view input')
    return receive.check_view(base,template,entry,item['view'],pair=[0,1],expert=0,deadline=deadline)


def negative_queries(obs):
    out={}
    def put(key, mutate, name='weighted_sign'):
        item=obs['cases'][name]['template'];p=deepcopy(item['package']);mutate(p)
        out[key]={'source':item['source'],'package':p,'expired':False}
    put('common_output',lambda p:p['common']['state']['hz']['c'].__setitem__(0,'1'))
    put('common_owner',lambda p:p['common']['ownership']['continuous'].__setitem__(0,'foreign'))
    put('template_missing',lambda p:p['templates'].pop())
    put('layer_missing',lambda p:p['templates'][0]['trace'].pop())
    put('template_foreign_source',lambda p:p['templates'][0]['trace'][0]['certificate'].__setitem__('source','0'*64))
    put('wrong_expert',lambda p:p['pairs'][0]['views'][0].__setitem__('expert',1))
    put('wrong_template',lambda p:p['pairs'][0]['views'][0].__setitem__('template_sha256',p['pairs'][0]['views'][1]['template_sha256']))
    put('guard_omission',lambda p:p['pairs'][0]['entry']['reference']['state']['hz']['ub'].pop())
    put('guard_inward',lambda p:p['pairs'][0]['entry']['reference']['state']['hz']['ub'].__setitem__(-1,'-999'))
    put('view_guard',lambda p:p['pairs'][0]['views'][0]['target']['state']['hz']['ub'].__setitem__(0,'999'))
    put('private_alias',lambda p:p['pairs'][0]['views'][1]['target']['state']['continuous_ids'].__setitem__(-1,p['pairs'][0]['views'][0]['target']['state']['continuous_ids'][-1]))
    put('private_column',lambda p:p['pairs'][0]['views'][0]['target']['state']['hz']['Gc']['data'].__setitem__(-1,'999'))
    put('view_owner',lambda p:p['pairs'][0]['views'][0]['target']['ownership']['continuous'].__setitem__(-1,'foreign'))
    put('property',lambda p:p['endpoint_request']['properties'][0].__setitem__('offset','1'))
    put('gate',lambda p:p['pairs'][0]['gate_evidence'].__setitem__('bounds',['0','1']))
    put('pair_omission',lambda p:(p['pairs'].pop(),p['endpoint_request']['pairs'].pop()))
    put('stale_proof',lambda p:p.__setitem__('proof',obs['cases']['weighted_sign']['per_pair']['package']['proof']))
    put('inward_relu',lambda p:p['templates'][0]['trace'][1]['certificate']['ranges'].__setitem__(0,['0','0']))
    put('expired',lambda p:None);out['expired']['expired']=True
    return out


def recheck_negative(spec):
    anchor=identity(spec)
    try:
        receive.check(spec['source'],spec['package'],expected_source_sha256=identity(spec['source']),
                      deadline=time.monotonic()-1 if spec['expired'] else time.monotonic()+300)
    except (ValueError,TypeError,KeyError,TimeoutError) as exc:
        if isinstance(exc,TimeoutError) and not spec['expired']: raise
        if identity(spec)!=anchor: raise AssertionError('negative query mutated')
        return {'input_sha256':anchor,'error_type':type(exc).__name__,'error':str(exc)}
    raise AssertionError('fixed negative query accepted')


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OBS.clear();OBS.update(cases={},bridges={},negatives={},auxiliary={})
        for i,(name,_,_) in enumerate(cases()):
            for arm in (('per_pair','template') if i%2==0 else ('template','per_pair')): run_arm(name,arm)
        for name in ('affine','relu','guard'): OBS['bridges'][name]=view_bridge(name,time.monotonic()+300)
        item=OBS['cases']['tied_partial_reuse']['template'];p=deepcopy(item['package'])
        p['proof']['pairs'][-1]['candidates']=None
        OBS['partial']={'source':item['source'],'package':p,
                        'checked':receive.check(item['source'],p,expected_source_sha256=identity(item['source']),deadline=time.monotonic()+300)}

    def test_complete_sources(self):
        for arms in OBS['cases'].values():
            for item in arms.values():
                self.assertEqual(item['checked']['missing_endpoints'],0)
                self.assertFalse(item['checked']['real_model_claim'])
        self.assertEqual(OBS['cases']['unsafe_tied']['template']['checked']['positive'],0)

    def test_numeric_differential(self):
        OBS['differentials']={name:numeric_comparison(arms['per_pair']['package'],arms['template']['package'])
                              for name,arms in OBS['cases'].items()}
        # Full mathematical agreement is required even if outcomes coincide.
        a=OBS['cases']['weighted_sign']['per_pair']['package'];b=deepcopy(OBS['cases']['weighted_sign']['template']['package'])
        b['endpoint_request']['pairs'][0]['batch']['base']['b'][0]+=1
        with self.assertRaises(ValueError):numeric_comparison(a,b)

    def test_propagation_counts(self):
        for arms in OBS['cases'].values():
            for arm,item in arms.items():self.assertEqual(item['dispatch'],dispatch_roster(item['source']['request']['experts'],arm))

    def test_private_identity(self):
        for arms in OBS['cases'].values():
            p=arms['template']['package'];_,c,b=unpack(p['common']['state']);all_private=[]
            for t in p['templates']:
                s=t['trace'][-1]['lift']['enclosure']['target'];_,tc,tb=unpack(s['state'])
                all_private+=tc[len(c):]+tb[len(b):]
            self.assertEqual(len(set(all_private)),len(all_private));self.assertFalse(set(all_private)&set(c+b))

    def rejected(self,*keys):
        specs=negative_queries(OBS)
        for key in keys:OBS['negatives'][key]=recheck_negative(specs[key])

    def test_common_entry(self):self.rejected('common_output','common_owner')
    def test_template_inventory(self):self.rejected('template_missing','layer_missing','template_foreign_source')
    def test_template_selection(self):self.rejected('wrong_expert','wrong_template')
    def test_guard(self):self.rejected('guard_omission','guard_inward','view_guard')
    def test_private_pollution(self):self.rejected('private_alias','private_column','view_owner')
    def test_gate_property(self):self.rejected('property','gate')
    def test_missing_pair(self):self.rejected('pair_omission')
    def test_partial(self):self.assertEqual(OBS['partial']['checked']['status'],'UNKNOWN_MISSING_EVIDENCE')
    def test_stale(self):self.rejected('stale_proof')
    def test_operator_bridges(self):
        for name,item in OBS['bridges'].items():self.assertEqual(check_view_bridge(name,item,time.monotonic()+300),item['checked'])
    def test_guard_dependent_range(self):self.rejected('inward_relu')

    def test_deadline_mutation(self):
        self.rejected('expired')
        for name in ('expired_producer','mutated_source'):
            start=time.monotonic();deadline=start-1 if name=='expired_producer' else start+300
            doc=deepcopy(OBS['cases']['weighted_sign']['template']['source']);anchor=identity(doc)
            item={'start':start,'deadline':deadline,'source_sha256':anchor,
                  'cost':{'source_creation':time.monotonic()-start},'injection_reached':False,
                  'actual_candidate_calls':0}
            OBS['auxiliary'][name]=item
            def inject(*args,**kwargs):
                # Deliberately no optimizer: this is source-pollution reception,
                # not a hidden ninth complete source/candidate execution.
                item['injection_reached']=True;doc['request']['radius']='1/2';return None
            error=None
            with patch.object(api,'propose_request',side_effect=inject),patch.object(api,'propagate',wraps=api.propagate) as calls:
                try:api.build(doc,expected_source_sha256=anchor,deadline=deadline,observe=lambda k,v:item['cost'].__setitem__(k,v))
                except (ValueError,TimeoutError) as exc:error=exc
                finally:
                    item['end']=time.monotonic();item['cost']['total']=item['end']-start
                    item['dispatch']=[c.args[1] for c in calls.call_args_list]
                    item['error_type']=type(error).__name__;item['error']=str(error)
                    item['observed_source_sha256']=identity(doc)
            self.assertIsInstance(error,TimeoutError if name=='expired_producer' else ValueError)
            self.assertEqual(item['injection_reached'],name=='mutated_source')
            if name=='mutated_source':self.assertEqual(str(error),'source changed during generation')
        spec=negative_queries(OBS)['wrong_expert']
        with patch.object(receive,'check',side_effect=TimeoutError('unregistered')),self.assertRaises(TimeoutError):recheck_negative(spec)

    def test_checker_independence(self):
        with patch.object(api,'build',side_effect=AssertionError),patch.object(api,'propagate',side_effect=AssertionError),\
                patch.object(api,'specialize',side_effect=AssertionError),patch.object(api,'propose_request',side_effect=AssertionError):
            for arms in OBS['cases'].values():
                item=arms['template'];actual=receive.check(item['source'],item['package'],expected_source_sha256=identity(item['source']),deadline=time.monotonic()+300)
                self.assertEqual(actual,item['checked'])

    def test_cost_inventory(self):
        from scripts.run_hz_lifted_source import check_cost
        for arms in OBS['cases'].values():
            for item in arms.values():check_cost(item)
