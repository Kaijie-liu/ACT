"""Correctness controls, not timing evidence and never sealed real requests."""
import copy
from fractions import Fraction as F
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT,PYTHON,save,load
from source_enclosure.format import identity,unpack,pack,empty
from source_construction_lab.fixtures import document
from parsed_source_reuse.cache import SourceParser,scope_key,LIMITS
from parsed_source_reuse.check import check,private_checker


def specimen(frame=1):
    h=empty(2,frame);h['c']=[F(1,3),F(-2,7)]
    h['Gc']=[{0:F(2,5)},{1:F(-3,11)}];h['Gb']=[{0:F(1,13)},{}]
    h['Ac']=[{0:F(2),1:F(-1)}];h['Ab']=[{}];h['b']=[F(0)]
    h['Auc']=[{1:F(5,9)}];h['Aub']=[{0:F(1,2)}];h['ub']=[F(6)]
    return pack(h,['input0','input1'],['relu0'])


def built(fixture,invocation):
    from checked_route_frontier.build import prefix
    from shared_route_residual.propose import propose
    from residual_proof.build import finish
    doc=document(**fixture);sid=identity(doc);end=time.monotonic()+30
    pre=prefix(doc,expected_source_sha256=sid,deadline=end)
    cert=propose(doc,pre,invocation=invocation,deadline=end)
    bundle=finish(doc,pre,cert,mode='shared',invocation=invocation,expected_source_sha256=sid,deadline=end)
    return doc,bundle,invocation


class ParserControls(unittest.TestCase):
    def setUp(self):
        self.state=specimen();self.scope={'invocation':'control','source_sha256':identity({'request':'one'})}
    def parser(self,**kw):
        return SourceParser(self.scope,enabled=kw.pop('enabled',True),tick=kw.pop('tick',lambda:None),**kw)
    def test_exact_differential_and_reuse(self):
        for enabled in (False,True):
            p=self.parser(enabled=enabled)
            for _ in range(3):self.assertEqual(p.unpack(self.state,scope=self.scope),unpack(self.state))
            self.assertEqual(p.stats()['hits'],2 if enabled else 0)
            self.assertEqual(p.stats()['parses'],1 if enabled else 3)
    def test_mutated_input_rebound_not_stale(self):
        p=self.parser();p.unpack(self.state,scope=self.scope)
        self.state['hz']['c'][0]='9/17'
        self.assertEqual(p.unpack(self.state,scope=self.scope),unpack(self.state))
        self.assertEqual(p.stats()['hits'],0)
    def test_container_alias_and_fraction_slots_cannot_poison(self):
        p=self.parser();value=p.unpack(self.state,scope=self.scope)
        value[0]['Gc'][0][0]._numerator=999999
        value[0]['c'][0]._denominator=19
        value[0]['Gc'][0][0]=F(777);value[1].append('bad');value[0]['Auc'].clear()
        self.assertEqual(p.unpack(self.state,scope=self.scope),unpack(self.state))
    def test_full_source_binding_changes(self):
        p=self.parser();p.unpack(self.state,scope=self.scope)
        changed=[]
        for key,value in [('frame_id',2),('ub',['7'])]:
            v=copy.deepcopy(self.state);v['hz'][key]=value;changed.append(v)
        v=copy.deepcopy(self.state);v['continuous_ids'][0]='different_input';changed.append(v)
        for v in changed:self.assertEqual(p.unpack(v,scope=self.scope),unpack(v))
        self.assertEqual(p.stats()['hits'],0)
    def test_csr_factor_and_exact_flag_corruption_rejected(self):
        bads=[]
        for which in ('ptr','index','value','duplicate_factor','exact'):
            v=copy.deepcopy(self.state)
            if which=='ptr':v['hz']['Gc']['indptr'][-1]+=1
            elif which=='index':v['hz']['Gc']['indices'][0]=100
            elif which=='value':v['hz']['Gc']['data'][0]='not a rational'
            elif which=='duplicate_factor':v['binary_ids'][0]=v['continuous_ids'][0]
            else:v['hz']['exact']=True
            bads.append(v)
        for enabled in (False,True):
            p=self.parser(enabled=enabled);p.unpack(self.state,scope=self.scope)
            for bad in bads:
                with self.assertRaises((ValueError,TypeError)):p.unpack(bad,scope=self.scope)
    def test_hash_collision_refused(self):
        p=self.parser()
        with patch('parsed_source_reuse.cache.digest',return_value='same'):
            p.unpack(self.state,scope=self.scope)
            with self.assertRaises(ValueError):p.unpack(specimen(2),scope=self.scope)
    def test_wrong_scope_and_transplanted_entry(self):
        p=self.parser();p.unpack(self.state,scope=self.scope)
        other=dict(self.scope,invocation='other')
        with self.assertRaises(ValueError):p.unpack(self.state,scope=other)
        q=SourceParser(other,enabled=True,tick=lambda:None)
        key,value=next(iter(p._SourceParser__items.items()))
        q._SourceParser__items[(scope_key(other),key[1],key[2])]=value
        with self.assertRaises(ValueError):q.unpack(self.state,scope=other)
    def test_no_persistent_cache_and_close(self):
        p=self.parser();p.unpack(self.state,scope=self.scope);p.close()
        self.assertEqual(p.stats()['live_entries'],0)
        with self.assertRaises(ValueError):p.unpack(self.state,scope=self.scope)
        q=self.parser();q.unpack(self.state,scope=self.scope);self.assertEqual(q.stats()['hits'],0)
    def test_eviction_and_oversized_fallback(self):
        p=self.parser(limits=dict(LIMITS,entries=1))
        for v in (self.state,specimen(2),self.state):self.assertEqual(p.unpack(v,scope=self.scope),unpack(v))
        self.assertEqual(p.stats()['evictions'],2);self.assertEqual(p.stats()['peak_entries'],1)
        for limits in (dict(LIMITS,payload_bytes=1),dict(LIMITS,cells=1)):
            p=self.parser(limits=limits)
            for _ in range(2):self.assertEqual(p.unpack(self.state,scope=self.scope),unpack(self.state))
            self.assertEqual(p.stats()['oversized'],2);self.assertEqual(p.stats()['live_entries'],0)
    def test_deadline_at_hit_and_mid_snapshot(self):
        calls=[0];limit=[100000]
        def tick():
            calls[0]+=1
            if calls[0]>limit[0]:raise TimeoutError('inherited clock')
        p=self.parser(tick=tick);p.unpack(self.state,scope=self.scope)
        limit[0]=calls[0]
        with self.assertRaises(TimeoutError):p.unpack(self.state,scope=self.scope)
        limit[0]=calls[0]+5
        with self.assertRaises(TimeoutError):p.unpack(self.state,scope=self.scope)
        limit[0]=100000
        self.assertEqual(p.unpack(self.state,scope=self.scope),unpack(self.state))
    def test_json_boundary_and_invalid_policy(self):
        for bad in (dict(LIMITS,entries=0),dict(LIMITS,cells=True),dict(LIMITS,payload_bytes=2**40)):
            with self.assertRaises(ValueError):self.parser(limits=bad)
        p=self.parser()
        for value in ([float('nan')],{1:'bad key'},('tuple',),object()):
            with self.assertRaises(ValueError):p.unpack(value,scope=self.scope)
    def test_cost_breakdown_and_no_result_caching(self):
        p=self.parser()
        for _ in range(2):p.unpack(self.state,scope=self.scope)
        stats=p.stats();self.assertFalse(stats['results_or_bounds_cached'])
        self.assertGreaterEqual(stats['other_seconds'],0)
        self.assertAlmostEqual(sum(v for k,v in stats['seconds'].items() if k!='total')+stats['other_seconds'],stats['seconds']['total'])


class CheckerControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent=Path(os.environ.get('PARSED_SOURCE_CONTROL_ROOT','/data1/Kane/MOE'))
        parent.mkdir(parents=True,exist_ok=True)
        cls.root=Path(tempfile.mkdtemp(prefix='parsed_source_controls_',dir=parent))
        cls.cases=[built(dict(experts=2,classes=2,width=1,depth=0,seed=91),'tiny'),
                   built(dict(experts=3,classes=3,width=2,depth=1,seed=91,tied=True),'ties'),
                   built(dict(experts=4,classes=4,width=3,depth=1,seed=91,constant=True),'zero')]
        for doc,bundle,inv in cls.cases:
            (cls.root/inv).mkdir();save(cls.root/inv/'source.json',doc);save(cls.root/inv/'bundle.json',bundle)

    def test_complete_multipair_ties_dimensions_and_zero_differential(self):
        from residual_proof.check import check as baseline
        rows=[]
        for doc,bundle,inv in self.cases:
            expected=baseline(doc,bundle,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
            stats=[]
            for enabled in (False,True):
                result=check(doc,bundle,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30,
                             enabled=enabled,stats_sink=stats.append)
                self.assertEqual(result,expected);self.assertFalse(result['complete_output_positive_proof'])
            self.assertGreater(stats[1]['parser']['hits'],0)
            self.assertLess(stats[1]['parser']['parses'],stats[0]['parser']['parses'])
            self.assertEqual(stats[1]['after_close']['live_entries'],0)
            rows.append({'id':inv,'expected':expected,'modes':stats})
        save(self.root/'differential.json',rows)

    def test_all_guard_projection_output_predicates_still_reject(self):
        doc,bundle,inv=self.cases[1]
        def corrupt(b,kind):
            p=b['pairs'][0]
            if kind=='guard':p['guarded']['hz']['ub'][0]='987'
            elif kind=='projection':p['projected']['hz']['c'][0]='987'
            elif kind=='factor':p['joint']['continuous_ids'][0]='other'
            elif kind=='plane':p['obligations']['rows'][0]['b_extra'][0]='987'
            elif kind=='property':p['obligations']['rows'][0]['competitor']=999
            elif kind=='missing':p['obligations']['rows'].pop()
            elif kind=='pair':p['pair']=[0,2]
            elif kind=='base':p['base']['source_sha256']='0'*64
        from residual_proof.check import check as baseline
        for kind in ('guard','projection','factor','plane','property','missing','pair','base'):
            b=copy.deepcopy(bundle);corrupt(b,kind)
            with self.assertRaises(ValueError):baseline(doc,b,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
            for enabled in (False,True):
                with self.assertRaises(ValueError):check(doc,b,invocation=inv,expected_source_sha256=identity(doc),
                    deadline=time.monotonic()+30,enabled=enabled)

    def test_scope_changed_request_wrong_invocation_and_deadline(self):
        doc,b,inv=self.cases[0]
        for kwargs in ({'expected_source_sha256':'0'*64},{'invocation':'different'}, {'deadline':time.monotonic()-1}):
            args=dict(invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30,enabled=True);args.update(kwargs)
            with self.assertRaises((ValueError,TimeoutError)):check(doc,b,**args)

    def test_private_globals_and_original_bytecode_unchanged(self):
        from residual_proof import check as original
        before={k:original.check.__globals__[k] for k in ('check_join','check_guards','check_projection','check_outputs')}
        doc,b,inv=self.cases[1];scope={'invocation':inv,'source_sha256':identity(doc)}
        p=SourceParser(scope,enabled=True,tick=lambda:None);fn=private_checker(p,scope)
        self.assertIs(fn.__code__,original.check.__code__)
        for k,value in before.items():
            self.assertIs(original.check.__globals__[k],value)
            self.assertIs(fn.__globals__[k].__code__,value.__code__)
            self.assertIsNot(fn.__globals__[k].__globals__,value.__globals__)
        fn(doc,b,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
        for k,value in before.items():self.assertIs(original.check.__globals__[k],value)

    def test_warm_parser_does_not_cache_property_acceptance(self):
        from residual_proof import check as original
        doc,b,inv=self.cases[1];scope={'invocation':inv,'source_sha256':identity(doc)}
        p=SourceParser(scope,enabled=True,tick=lambda:None);fn=private_checker(p,scope)
        fn(doc,b,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
        bad=copy.deepcopy(b);bad['pairs'][0]['obligations']['rows'].pop()
        hits=p.stats()['hits']
        with self.assertRaises(ValueError):fn(doc,bad,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
        self.assertGreater(p.stats()['hits'],hits)

    def test_exception_stats_and_closed_storage(self):
        doc,b,inv=self.cases[1];bad=copy.deepcopy(b);bad['pairs'][0]['base']['source_sha256']='0'*64;stats=[]
        with self.assertRaises(ValueError):check(doc,bad,invocation=inv,expected_source_sha256=identity(doc),
            deadline=time.monotonic()+30,enabled=True,stats_sink=stats.append)
        self.assertEqual(stats[0]['status'],'ERROR');self.assertTrue(stats[0]['after_close']['closed'])
        self.assertEqual(stats[0]['after_close']['live_entries'],0)

    def test_relocated_fresh_python_S_no_solver_or_producer(self):
        import shutil
        target=self.root/'relocated';shutil.copytree(self.root/'ties',target)
        code='''import sys,time
from pathlib import Path
def forbid(event,args):
    if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or args[0] in ('residual_proof.build','checked_route_frontier.build','shared_route_residual.propose','source_construction_lab.fixtures')):raise ImportError(args[0])
sys.addaudithook(forbid)
from scoped_proof.io import load
from source_enclosure.format import identity
from parsed_source_reuse.check import check
p=Path(sys.argv[1]);doc=load(p/'source.json');b=load(p/'bundle.json');results=[]
for mode in (False,True):results.append(check(doc,b,invocation='ties',expected_source_sha256=identity(doc),deadline=time.monotonic()+30,enabled=mode))
assert results[0]==results[1] and not results[1]['complete_output_positive_proof']
'''
        out=subprocess.run([PYTHON,'-S','-c',code,str(target)],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(out.returncode,0,out.stderr)


if __name__=='__main__':unittest.main()
