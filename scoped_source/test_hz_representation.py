"""Fixed same-source representation controls; retain either arm's outcomes."""
from copy import deepcopy
from fractions import Fraction as F
import inspect
import json
import time
import unittest
from unittest.mock import patch

from scoped_source import hz_representation as api
from scoped_source import check_hz_representation as audit
from scoped_source.endpoint_source_controls import cases
from scoped_source.rowwise_bound import identity

OBS={}


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.run_hz_representation import protocol
        cfg=protocol(); OBS.clear()
        for i,(name,expected) in enumerate(cfg['cases'].items()):
            for mode in cfg['arms'][::1 if i%2==0 else -1]:
                begin=time.monotonic(); end=begin+cfg['deadline_seconds']; cost={}
                item={'case':name,'mode':mode,'start':begin,'deadline':end,'cost_seconds':cost}
                OBS[name+'/'+mode]=item
                try:
                    doc=next(d for n,d,_ in cases() if n==name)
                    item['source']=doc; cost['source_creation']=time.monotonic()-begin
                    if identity(doc)!=expected['source_sha256']: raise ValueError('frozen declaration identity')
                    package=api.build(doc,mode,end,lambda k,v:cost.update({k:v}))
                    item['package']=package; t=time.monotonic()
                    raw=json.dumps(package,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                    restored=json.loads(raw); cost['serialization']=time.monotonic()-t
                    item['serialized_bytes']=len(raw); t=time.monotonic()
                    item['checked']=audit.check(doc,restored,expected_source_sha256=identity(doc),expected_mode=mode,deadline=end)
                    cost['checking']=time.monotonic()-t
                    item['end']=time.monotonic(); cost['total']=item['end']-begin
                    if item['end']>=end: raise TimeoutError('finite representation deadline')
                except Exception as exc:
                    item['error']=repr(exc); item['end']=time.monotonic(); cost['total']=item['end']-begin
                    raise

    def item(self,mode='mccormick',name='weighted_sign'):
        r=OBS[name+'/'+mode]; return r['source'],deepcopy(r['package'])

    def check(self,doc,p):
        return audit.check(doc,p,expected_source_sha256=identity(doc),expected_mode=p['mode'],deadline=time.monotonic()+300)

    def reject(self,doc,p):
        with self.assertRaises((ValueError,KeyError)): self.check(doc,p)

    def test_complete_roster_same_domain(self):
        from scripts.run_hz_representation import compare,protocol
        report=compare(OBS)
        self.assertEqual(len(report),4)
        for name,spec in protocol()['cases'].items():
            a,b=[OBS[name+'/'+m] for m in ('endpoints','mccormick')]
            self.assertEqual(a['package']['lowering'],b['package']['lowering'])
            for r in (a,b): self.assertEqual(r['checked']['required'],spec['duties'])
            self.assertEqual(a['checked']['checked_targets'],spec['endpoints'])
            self.assertEqual(b['checked']['checked_targets'],spec['duties'])
        bad=deepcopy(OBS); bad['weighted_sign/mccormick']['package']['lowering']['input']['continuous_ids'][0]='foreign'
        with self.assertRaises(ValueError): compare(bad)

    def test_source_binding(self):
        doc,p=self.item(); changed=deepcopy(doc); changed['request']['margin']='1'
        with self.assertRaises(ValueError):
            audit.check(changed,p,expected_source_sha256=identity(doc),expected_mode='mccormick',deadline=time.monotonic()+300)
        p['lowering']['input']['hz']['Gc']['data'][0]='1/2'; self.reject(doc,p)

    def test_mc_projection(self):
        for what in ('q','offset','u','d','objective'):
            doc,p=self.item(); r=p['duties'][0]['construction']
            if what=='q': r['q'][0]='2'
            elif what=='offset': r['offset']='1'
            elif what in ('u','d'): r[what]['constant']='999'
            else: r['lp']['offset']='999'
            self.reject(doc,p)

    def test_mc_planes(self):
        for plane in range(4):
            doc,p=self.item(); lp=p['duties'][0]['construction']['lp']
            row=len(lp['b'])-4+plane; pos=lp['A']['indptr'][row+1]-1
            lp['A']['data'][pos]=str(-F(lp['A']['data'][pos])); self.reject(doc,p)

    def test_difference_range(self):
        doc,p=self.item(); r=p['duties'][0]['construction']
        r['difference']=['0','0']; self.reject(doc,p)
        doc,p=self.item(); r=p['duties'][0]['construction']; r['lp']['lower'][-1]='0'
        self.reject(doc,p)

    def test_gate_binding(self):
        doc,p=self.item(); p['duties'][0]['construction']['gate']=['1/2','1']; self.reject(doc,p)
        doc,p=self.item(); p['lowering']['pairs'][0]['gate_evidence']['weight_expert']=1; self.reject(doc,p)

    def test_binary_factor_binding(self):
        doc,p=self.item(); p['duties'][0]['construction']['n_relaxed_binaries']=0; self.reject(doc,p)
        doc,p=self.item(); p['lowering']['pairs'][0]['b'][1]['target']['binary_ids'][-1]='aliased'; self.reject(doc,p)

    def test_missing_duties(self):
        doc,p=self.item(); p['duties'].pop(); self.reject(doc,p)
        doc,p=self.item(); p['duties'][0]['certificate']=None
        result=self.check(doc,p); self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['missing_targets'],1)
        doc,p=self.item('endpoints'); p['proof']['pairs'][0]['candidates']=None
        result=self.check(doc,p); self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['missing_targets'],2)

    def test_stale_candidates(self):
        doc,p=self.item(); p['duties'][0]['certificate']=deepcopy(p['duties'][1]['certificate']); self.reject(doc,p)
        doc,p=self.item(); p['duties'][0]['certificate']['claimed_lower_bound']='999'; self.reject(doc,p)
        doc,p=self.item(); p['duties'][0]['property']='other'; self.reject(doc,p)

    def test_registered_point(self):
        from scripts.run_hz_representation import diagnostic
        result=diagnostic(OBS,time.monotonic()+300)
        # A failed predeclared candidate is retained, not replaced or searched.
        self.assertIn(result['status'],('EXACT_FEASIBLE','REGISTERED_POINT_NOT_FEASIBLE'))
        doc,p=self.item(); point=api.registered_point(p)['point']; point[0]='999'
        with self.assertRaises(ValueError): audit.check_point(p['duties'][0]['construction']['lp'],point,time.monotonic()+300)

    def test_deadline(self):
        doc,p=self.item()
        with self.assertRaises(TimeoutError): audit.check(doc,p,expected_source_sha256=identity(doc),expected_mode='mccormick',deadline=time.monotonic()-1)
        with self.assertRaises(TimeoutError): api.build(doc,'mccormick',time.monotonic()-1,lambda *_:None)

    def test_checker_independence(self):
        doc,p=self.item()
        with patch.object(api,'build_mc',side_effect=AssertionError('builder forbidden')), \
             patch.object(api,'propagate',side_effect=AssertionError('propagation forbidden')), \
             patch.object(api,'_candidate_columns',side_effect=AssertionError('optimizer forbidden')):
            self.assertEqual(self.check(doc,p),OBS['weighted_sign/mccormick']['checked'])
        source=inspect.getsource(audit)
        for forbidden in ('import torch','import scipy','from scoped_source.hz_representation import'):
            self.assertNotIn(forbidden,source)

    def test_cost_and_archive(self):
        from scripts.run_hz_representation import check_cost,compare
        for item in OBS.values(): check_cost(item)
        changed=deepcopy(next(iter(OBS.values()))); changed['cost_seconds'].pop('source_creation')
        with self.assertRaises(ValueError): check_cost(changed)
        with self.assertRaises(ValueError): compare({k:v for i,(k,v) in enumerate(OBS.items()) if i})
