"""Pure fixed control inputs and checking-only helpers for source integration."""
from copy import deepcopy
from fractions import Fraction as F
import time

from source_enclosure.format import empty, pack, unpack, identity
from source_enclosure.check import check_affine
from scoped_source.check_hz_binary64 import REFERENCE
from scoped_source import check_hz_sparse_source as receive
from scoped_source import check_hz_lifted_source as old_receive
from scripts.run_hz_lifted_source import SOURCES

ARMS = ('old', 'sparse')
BRIDGES = ('affine', 'relu', 'guard')
TESTS = tuple(f'test_{i:02d}_{n}' for i, n in enumerate((
    'complete', 'differential', 'dispatch', 'source_mutations', 'sparse_mutations',
    'bridges', 'dense_capacity', 'partial_stale', 'deadlines', 'identity_faults',
    'independence', 'archive_cost'), 1))
FAULTS = ('expired_producer', 'expired_checker', 'delayed_live', 'source_before_proposals', 'checker_late_package')
ERRORS = {'expired_producer': ('TimeoutError', 'source construction/check deadline'),
          'expired_checker': ('TimeoutError', 'source construction/check deadline'),
          'delayed_live': ('TimeoutError', 'sparse-row deadline: before_final_identity'),
          'source_before_proposals': ('ValueError', 'source changed before proposals'),
          'checker_late_package': ('ValueError', 'source/package changed')}
NEGATIVE_ERRORS = {
    'source_identity':'sparse block source package identity/schema', 'input':'input enclosure inward',
    'affine':'compensation generator placement/value', 'relu':'unchecked/inward ReLU range',
    'relu_order':'ReLU exact graph/constraint construction', 'guard':'exact conditional route reference',
    'owner':'source reference ownership', 'layer':'all source layers required',
    'template':'all expert templates required in order', 'pair':'all source pairs/properties binding',
    'property':'all source pairs/properties binding', 'gate':'checked source gate binding',
    'old_proof':'proof request coverage', 'sparse_map':'continuous/binary/flat factor maps',
    'sparse_frame':'original factor/owner/frame/row inventory', 'sparse_missing_row':'compensation row inventory',
    'sparse_inward':'inward compensation', 'sparse_alias':'factor identity alias',
    'bridge_affine':'missing/incorrect exact affine compensation', 'bridge_relu':'unchecked/inward ReLU range',
    'bridge_guard':'exact conditional route reference'}


def owned(h, ci, bi):
    return {'schema': REFERENCE, 'state': pack(h,ci,bi),
            'ownership': {'continuous': ['input']*len(ci), 'binary': ['router']*len(bi)}}


def bridge_sources(name):
    h = empty(1)
    if name == 'affine':
        w = F(1)+F(1,2**52); h['Gc'] = [{0:w, 1:w/F(2**96)}]
        return {'source': owned(h,['x','y'],[]), 'operator': [{'0':str(w)}], 'bias': ['0']}
    if name == 'relu':
        h['c'] = [F(1,2)]; h['Gc'] = [{0:F(1), 1:F(1,2**55)}]
        return {'source': owned(h,['x','y'],[])}
    if name != 'guard': raise ValueError('fixed bridge name')
    h['Gc'] = [{0:F(1)}]; router = empty(3); router['c'] = [F(1), F(1,2**54), F(0)]
    return {'source': owned(h,['x'],[]), 'router': owned(router,['x'],[])}


def check_bridge(name, item, arm, deadline):
    for key, value in bridge_sources(name).items():
        if item[key] != value: raise ValueError('fixed bridge source')
    tag = 'bridge/'+name; source = item['source']; ref = item['lift']['reference']
    if name == 'affine':
        old_receive.owners(source,ref,tag)
        op = [{int(k):F(v) for k,v in row.items()} for row in item['operator']]
        check_affine(source['state'],ref['state'],op,list(map(F,item['bias'])),item['nominal'],item['certificate'],tag)
    elif name == 'relu':
        old_receive.owners(source,ref,tag); old_receive.relu_reference(source,ref,item['certificate'],tag)
    else:
        old_receive.route_reference(source,item['router'],ref,(0,2),3)
    return (old_receive if arm=='old' else receive).lift(item['lift'],tag,deadline)[1]


def dense_reference():
    h = empty(1); h['c'] = [F(2)+F(1,2**54)]; h['Gc'] = [{j:F(1) for j in range(9)}]
    h['Auc'] = [{i:F(1)} for i in range(6)]; h['Aub'] = [{} for _ in range(6)]; h['ub'] = [F(1)]*6
    return owned(h,[f'x{i}' for i in range(9)],[])


def capacity(name):
    if name == 'global_factor': return owned(empty(1),[f'x{i}' for i in range(129)],[])
    if name == 'row_output': return owned(empty(17),['x'],[])
    raise ValueError('fixed capacity name')


def lift_records(package):
    records = [package['input']]+[t['lift'] for t in package['router']]
    for template in package['templates']: records += [t['lift'] for t in template['trace']]
    return records+[p['entry'] for p in package['pairs']]


def differential(left, right):
    if left['source'] != right['source']: raise ValueError('same declaration required')
    a,b = left['package'],right['package']
    records = list(zip(lift_records(a),lift_records(b)))
    if len(lift_records(a)) != len(lift_records(b)): raise ValueError('source step differential inventory')
    for x,y in records:
        if x['reference'] != y['reference'] or x['snapshot'] != y['snapshot'] or x['enclosure']['target'] != y['enclosure']['target']:
            raise ValueError('exact source target differential')
    if a['common'] != b['common']: raise ValueError('common source differential')
    if a['endpoint_request']['properties'] != b['endpoint_request']['properties']: raise ValueError('properties differential')
    for x,y in zip(a['endpoint_request']['pairs'],b['endpoint_request']['pairs']):
        # Context binds the new execution schema; mathematical block content is unchanged.
        for key in ('sources','layout','queries'):
            if x['batch'][key] != y['batch'][key]: raise ValueError('complete block differential: '+key)
        if x['gate'] != y['gate']: raise ValueError('gate differential')
    if left['checked']['results'] != right['checked']['results']: raise ValueError('all final endpoint bounds differential')
    if a['endpoint_request']['context'] == b['endpoint_request']['context'] or a['proof']['request_sha256'] == b['proof']['request_sha256']:
        raise ValueError('new context and fresh proof identity required')
    return {'source_lifts':len(records), 'pairs':len(a['pairs']),
            'endpoints':left['checked']['checked_endpoints'], 'exact_numeric_match':True}


def negative_queries(obs, only=None):
    if only is not None and only not in NEGATIVE_ERRORS: raise ValueError('fixed negative name')
    out = {}; parent = obs['cases']['weighted_sign']['sparse']
    def source(name, fn):
        if only is not None and name != only: return
        p = deepcopy(parent['package']); fn(p)
        out[name] = {'kind':'source', 'source':parent['source'], 'package':p}
    source('source_identity',lambda p:p.__setitem__('source_sha256','0'*64))
    source('input',lambda p:p['input']['reference']['state']['hz']['c'].__setitem__(0,'999'))
    source('affine',lambda p:p['templates'][0]['trace'][0]['lift']['reference']['state']['hz']['c'].__setitem__(0,'999'))
    source('relu',lambda p:p['templates'][0]['trace'][1]['certificate']['ranges'].__setitem__(0,['0','1']))
    def reorder(p):
        ref = p['templates'][0]['trace'][1]['lift']['reference']
        h, ci, bi = unpack(ref['state'])
        for key in ('Auc','Aub','ub'): h[key][0],h[key][-1] = h[key][-1],h[key][0]
        changed = pack(h,ci,bi)
        if changed == ref['state']: raise AssertionError('ReLU order mutation must change rows')
        ref['state'] = changed
    source('relu_order',reorder)
    source('guard',lambda p:p['pairs'][0]['entry']['reference']['state']['hz']['ub'].__setitem__(-1,'999'))
    source('owner',lambda p:p['templates'][0]['trace'][1]['lift']['reference']['ownership']['binary'].__setitem__(-1,'other'))
    source('layer',lambda p:p['templates'][0]['trace'].pop())
    source('template',lambda p:p['templates'].pop())
    source('pair',lambda p:(p['pairs'].pop(),p['endpoint_request']['pairs'].pop()))
    source('property',lambda p:p['endpoint_request']['properties'][0].__setitem__('offset','1'))
    source('gate',lambda p:p['pairs'][0]['gate_evidence'].__setitem__('bounds',['1','1']))
    source('old_proof',lambda p:p.__setitem__('proof',deepcopy(obs['cases']['weighted_sign']['old']['package']['proof'])))
    for name in ('map','frame','missing_row','inward','alias'):
        if only is not None and 'sparse_'+name != only: continue
        item = deepcopy(obs['bridges']['relu']['sparse']['item']); lift = item['lift']; p = lift['enclosure']
        if name=='map': p['maps']['flat'][0] = 1
        elif name=='frame': p['target']['state']['hz']['frame_id'] += 1
        elif name=='missing_row': p['equality_errors'].pop()
        elif name=='inward': p['output_errors'][0]['radius'] = '0'
        else: p['target']['state']['continuous_ids'][-1] = p['target']['state']['continuous_ids'][0]
        p['target_sha256'] = identity(p['target'])
        out['sparse_'+name] = {'kind':'bridge','name':'relu','item':item}
    for name in BRIDGES:
        if only is not None and 'bridge_'+name != only: continue
        item = deepcopy(obs['bridges'][name]['sparse']['item'])
        if name=='affine': item['certificate']['error_bounds'][0] = '0'
        elif name=='relu': item['certificate']['ranges'][0] = ['0','1']
        else: item['lift']['reference']['state']['hz']['ub'][0] = '0'
        out['bridge_'+name] = {'kind':'bridge','name':name,'item':item}
    if set(out) != ({only} if only is not None else set(NEGATIVE_ERRORS)):
        raise AssertionError('fixed negative/error roster')
    for name,spec in out.items(): spec['expected_error'] = ['ValueError',NEGATIVE_ERRORS[name]]
    return out


def recheck_negative(spec, deadline):
    anchor = identity(spec)
    try:
        if spec['kind']=='source':
            receive.check(spec['source'],spec['package'],expected_source_sha256=identity(spec['source']),deadline=deadline)
        elif spec['kind']=='bridge': check_bridge(spec['name'],spec['item'],'sparse',deadline)
        else: raise AssertionError('unexpected query kind')
    except (ValueError, KeyError, TypeError) as exc:
        if identity(spec) != anchor: raise AssertionError('negative input changed')
        if [type(exc).__name__,str(exc)] != spec['expected_error']:
            raise AssertionError('unrelated exception instead of intended refusal: '+str(exc)) from exc
        return {'input_sha256':anchor,'error_type':type(exc).__name__,'error':str(exc)}
    raise AssertionError('negative input accepted')
