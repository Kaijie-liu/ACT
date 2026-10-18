"""Frozen finite traversal controls. Standard library; no numerical backend."""
import copy
from fractions import Fraction as F
from pathlib import Path
import time
import unittest

from scoped_source.hz_request_store import Work,POLICY,digest,identity
from scoped_source.scaled_endpoint_controls import fixture,expanded,scalar_bound
from scoped_source.forest_endpoint_controls import capture
from scoped_source.scaled_references import build
from scoped_source.check_scaled_references import CheckedScaledForest
from scoped_source.shared_build import build as shared_build
from scoped_source.check_shared_source import CheckedSharedForest
from scoped_source.scaled_endpoints import contract,pullback
from scoped_source.pair_workspace import PairWorkspace,Scope,U32
from scoped_source.bounded_runtime import Owner
from scoped_source.direct_candidates import Driver
from scripts.test_bounded_runtime import candidates,publish

ROOT=None
DEADLINE=None
RECORDS=[]


class TraceWork(Work):
    def __init__(self):
        self.trace=None; self.remaining=None; self.fault='deadline'
        super().__init__(DEADLINE)

    def exact(self,q):
        if self.trace is not None: self.trace.append(str(q))
        return super().exact(q)

    def visit(self,kind,amount=1):
        super().visit(kind,amount)
        if self.remaining is not None:
            self.remaining-=1
            if self.remaining<=0:
                self.remaining=None
                if self.fault=='exception': raise RuntimeError('fixed traversal fault')
                self.deadline=time.monotonic()-1
                self.tick()


def special(case):
    affine=lambda rows:dict(kind='affine',rows=rows)
    if case=='negative_guard':
        router=[affine([['0',[]],['0',[]],['1/2',[[0,'1']]]])]
        experts=[[affine([['-1/4',[]],['0',[]]])] for _ in range(3)]
    else:
        assert case=='eight_private'
        router=[affine([['0',[[0,str(F(i-4,8))]]] for i in range(8)])]
        experts=[]
        for i in range(8):
            experts.append([affine([['1/8',[[0,'1']]],['-1/4',[[0,'-1/2']]]]),
                dict(kind='relu'),
                affine([['1/4',[[0,'1'],[1,'-1/2']]],['0',[[0,'-1/2'],[1,'1']]]]),
                dict(kind='relu'),affine([['1',[[0,'1']]],['0',[[1,'1/2']]]])])
    return dict(schema='REQUEST_STORE_SOURCE_V1',lower=['-1'],upper=['1'],
                router=router,experts=experts)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={}; start=time.monotonic()
        for name in ('deep','half','guarded','eight_private','negative_guard'):
            dest=ROOT/name
            if name in ('eight_private','negative_guard'):
                source=capture(dest,special(name)); root=dest
            else: root,source=fixture(dest,name)
            w=TraceWork()
            if name in ('deep','eight_private'):
                raw,_=shared_build(root,source,work=w,enabled=True)
            else: raw,_=build(root,source,work=w)
            cls.cases[name]=(root,source,raw)
        RECORDS.append(dict(kind='setup',seconds=time.monotonic()-start,
                            note='fixed fixture construction counted in batch wall, not paired execution time'))

    def forest(self,name='deep'):
        root,source,raw=self.cases[name]; w=TraceWork()
        if name in ('deep','eight_private'):
            f=CheckedSharedForest(raw,root,source,expected=raw.identity,work=w,enabled=True)
        else: f=CheckedScaledForest(raw,root,source,expected=digest(raw),work=w)
        return f,w

    def driver(self,name='deep',enabled=True):
        f,w=self.forest(name); c=contract(f)
        d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True,scoped_traversal=enabled)
        self.addCleanup(d.abort); return d

    def workspace(self,name='deep',enabled=True):
        f,w=self.forest(name); ws=PairWorkspace(f,w,scoped_traversal=enabled)
        self.addCleanup(ws.close); ws.select((0,1)); return ws

    def test_01_default_and_opt_in(self):
        f,w=self.forest(); ws=PairWorkspace(f,w); self.addCleanup(ws.close)
        self.assertFalse(ws.scoped_traversal)
        ws.select((0,1)); self.assertFalse(ws.scope.traversal_ready)
        self.assertTrue(all(b._data is not None for b in ws.scope._blocks))
        for value in (1,None,'yes'):
            with self.assertRaises(ValueError): PairWorkspace(f,w,scoped_traversal=value)
        c=contract(f)
        with self.assertRaises(ValueError): Driver(f,c,expected=identity(c),work=w,scoped_traversal=True)
        with self.assertRaises(AttributeError): ws.scoped_traversal=True
        with self.assertRaises(AttributeError): del ws.scoped_traversal
        self.assertEqual((POLICY['visits'],POLICY['total_bytes'],POLICY['work_bits']),
                         (8000000,134217728,8192))

    def test_02_complete_fixed_candidate_differential(self):
        for name in ('deep','half','guarded','eight_private'):
            for profile in (0,1):
                both=[]; records=[]
                for enabled in (False,True):
                    start=time.monotonic(); d=self.driver(name,enabled); receipts=[]
                    for ordinal in range(d.session.tile_count):
                        tasks,_,token=d.prepare()
                        raw=candidates(d.forest,tasks,general=bool(profile)); saved=copy.deepcopy(raw)
                        got=d.submit(token,raw,publish,profile=profile)
                        self.assertEqual(raw,saved)
                        receipts.extend(dict(r.record) if r is not None else None for r in got)
                    result=d.finish(); elapsed=time.monotonic()-start
                    self.assertEqual(d.owner.stats['live_bytes'],0)
                    self.assertEqual(d.owner.stats['allocated_bytes'],d.owner.stats['released_bytes'])
                    both.append((receipts,result))
                    records.append(dict(kind='paired',source=name,profile=profile,enabled=enabled,
                        work=dict(d.work.counts),seconds=elapsed,owner=dict(d.owner.stats),
                        traversal=dict(d.workspace.traversal_stats),costs=dict(d.costs),result=result,
                        receipts_sha256=identity(receipts),receipts=receipts))
                self.assertEqual(both[0],both[1])
                a,b=records; stats=b['traversal']
                delta=sum(a['work'].values())-sum(b['work'].values())
                self.assertEqual(delta,stats['planned_iteration_saving']-stats['setup_work']-stats['selector_work'])
                self.assertEqual(a['work']['arithmetic'],b['work']['arithmetic'])
                # Independent expanded exact formula, separately counted oracle.
                f,w=self.forest(name); rows=expanded(f,w)
                ref=self.driver(name,False); oracle_rows=0
                for _ in range(ref.session.tile_count):
                    tasks,_,token=ref.prepare(); rr=candidates(ref.forest,tasks,general=bool(profile))
                    got=ref.submit(token,rr,publish,profile=profile)
                    for task,r,receipt in zip(tasks,rr,got,strict=True):
                        for role in ('proposed','zero'):
                            value,_=scalar_bound(f,task,r[role],rows,w)
                            self.assertEqual(value,F(receipt.record[role])); oracle_rows+=1
                ref.finish()
                RECORDS.extend(records)
                RECORDS.append(dict(kind='differential',source=name,profile=profile,
                    measured_work_saving=delta,oracle_candidate_checks=oracle_rows,oracle_work=dict(w.counts)))
        allstats=[r['traversal'] for r in RECORDS if r.get('kind')=='paired' and r['enabled']]
        self.assertGreater(sum(s['scoped_calls'] for s in allstats),0)
        self.assertGreater(sum(s['global_calls'] for s in allstats),0)

    def test_03_residual_and_arithmetic_order(self):
        for name in ('deep','eight_private'):
            for width in (1,2):
                values=[]; traces=[]; outbytes=[]
                for enabled in (False,True):
                    ws=self.workspace(name,enabled); f=ws.owner.forest; w=ws.owner.work
                    guard=next(v for a,b,v in f.roles['pairs'] if (a,b)==(0,1))
                    n=f.views[guard]['ine'][0][1]
                    roots=[{n:F(-1,3)}]+([{n:F(0)}] if width==2 else [])
                    maps=[ws.mapping('nodes',r) for r in roots]; w.trace=[]
                    out=ws.pullback(maps); traces.append(tuple(w.trace)); w.trace=None
                    values.append([dict(m) for m in out])
                    outbytes.append([m._slots.block.get(0,m._slots.block.size) for m in out])
                    for m in maps:
                        with self.assertRaises(ValueError): m.get(n,F(0))
                self.assertEqual(values[0],values[1]); self.assertEqual(traces[0],traces[1])
                self.assertEqual(outbytes[0],outbytes[1])

    def test_04_roster_faults(self):
        for mode in ('missing','duplicate','permuted','foreign_node','size','owner','count','factor'):
            f,w=self.forest(); owner=Owner(f,w); self.addCleanup(owner.close)
            s=Scope(owner,(0,1)); self.addCleanup(s.close)
            kind='factors' if mode=='factor' else 'nodes'
            count,selected,forward,reverse=s.banks[kind]
            ids=[s.global_id(kind,k) for k in range(selected)]
            if mode in ('missing','factor'): forward.put(ids[0]*4,U32.pack(0))
            elif mode=='duplicate': reverse.put(4,U32.pack(ids[0]))
            elif mode=='permuted':
                reverse.put(0,U32.pack(ids[1])); reverse.put(4,U32.pack(ids[0]))
                forward.put(4*ids[0],U32.pack(2)); forward.put(4*ids[1],U32.pack(1))
            elif mode=='foreign_node':
                foreign=next(j for j in range(count) if j not in ids)
                reverse.put(0,U32.pack(foreign)); forward.put(4*foreign,U32.pack(1))
            elif mode=='size': forward.size-=4
            elif mode=='owner':
                other=Owner(f,w); self.addCleanup(other.close); forward.owner=other
            elif mode=='count': forward.put(4*ids[-1],U32.pack(selected+1))
            try:
                with self.assertRaises(ValueError): s.prepare_traversal(owner,(0,1))
            finally:
                if mode=='size': forward.size+=4
                if mode=='owner': forward.owner=owner
            self.assertFalse(s.traversal_ready)

    def test_05_binding_sealed_alias_and_closed(self):
        ws=self.workspace(); s=ws.scope
        self.assertTrue(s.traversal_ready)
        for block in s._blocks:
            with self.assertRaises(ValueError): block.put(0,bytes(4))
        with self.assertRaises(ValueError): s.prepare_traversal(ws.owner,(0,1))
        with self.assertRaises(ValueError): s.bind(ws.owner,(0,2))
        other=self.workspace()
        with self.assertRaises(ValueError): s.bind(other.owner,(0,1))
        f,w=self.forest()
        with self.assertRaises(ValueError): ws.bind(f,w)
        ws._pair=(0,2)
        with self.assertRaises(ValueError): ws._node_ids(1)
        ws._pair=(0,1); ws.close()
        with self.assertRaises(ValueError): s.global_id('nodes',0)

    def test_06_intermediate_bit_rejection(self):
        traces=[]
        for enabled in (False,True):
            ws=self.workspace('deep',enabled); f=ws.owner.forest; w=ws.owner.work
            n=next(n for n,(_,local,refs) in enumerate(f.nodes)
                   if ws.scope.local('nodes',n) is not None
                   for _,q in (*local,*refs) if abs(q.numerator)>1)
            m=ws.mapping('nodes',{n:F((1<<8192)-1)}); w.trace=[]
            with self.assertRaisesRegex(ValueError,'intermediate'): ws.pullback([m])
            traces.append(tuple(w.trace)); w.trace=None
            self.assertFalse(m.active)
        self.assertEqual(traces[0],traces[1])

    def test_07_guard_residual_not_pruned(self):
        for enabled in (False,True):
            d=self.driver('negative_guard',enabled); tasks,_,token=d.prepare()
            self.assertEqual(tasks[0].pair,(0,1)); rr=candidates(d.forest,tasks)
            guard=next(v for a,b,v in d.forest.roles['pairs'] if (a,b)==(0,1))
            rows=d.forest.rows((0,1),'ine'); pos=next(i for i,r in enumerate(rows) if r[0]==guard)
            for r in rr:
                for role in ('proposed','zero'):
                    r[role]['equality']=['0']*len(r[role]['equality'])
                    r[role]['inequality']=['0']*len(r[role]['inequality'])
                r['proposed']['inequality'][pos]='-1'
            got=d.submit(token,rr,publish)
            self.assertTrue(all(F(r.record['proposed'])==F(-3,4) for r in got))
            self.assertTrue(all(F(r.record['zero'])==F(-1,4) for r in got))

    def test_08_partial_and_later_invalid_candidates(self):
        for enabled in (False,True):
            for partial in ('first','block','all'):
                d=self.driver('guarded',enabled); missing=0
                for _ in range(d.session.tile_count):
                    t,_,token=d.prepare(); rr=candidates(d.forest,t)
                    count=1 if partial=='first' else min(2,len(t)) if partial=='block' else len(t)
                    for r in rr[:count]: r['proposed']=None; r['zero']=None
                    missing+=count
                    got=d.submit(token,rr,publish)
                    self.assertTrue(all(x is None for x in got[:count]))
                    self.assertTrue(all(x is not None for x in got[count:]))
                result=d.finish(); self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
                self.assertEqual(result['missing_endpoints'],missing)
            for mode in ('sign','zero','query','half_missing'):
                d=self.driver('guarded',enabled); t,_,token=d.prepare(); rr=candidates(d.forest,t)
                self.assertEqual(len(t),8)
                if mode=='sign': rr[2]['proposed']['inequality'][0]='1'
                elif mode=='zero': rr[2]['zero']['inequality'][0]='-1'
                elif mode=='query': rr[2]['proposed']['query']='0'*64
                else: rr[2]['zero']=None
                with self.assertRaises(ValueError): d.submit(token,rr,publish)
                self.assertTrue(d.closed); self.assertEqual(d.receipts,[])
                self.assertEqual(d.owner.stats['live_bytes'],0)
                self.assertEqual(d.session._Session__issued,{})
                self.assertEqual(d.session._Session__values,{})

    def test_09_deadline_exception_work_and_seal_limits(self):
        for stage in ('roster','pullback','receive'):
            for fault in ('deadline','exception'):
                if stage=='roster':
                    f,w=self.forest(); o=Owner(f,w); self.addCleanup(o.close)
                    s=Scope(o,(0,1)); self.addCleanup(s.close)
                    w.remaining=5; w.fault=fault
                    with self.assertRaises((TimeoutError,RuntimeError)): s.prepare_traversal(o,(0,1))
                    self.assertFalse(s.traversal_ready)
                else:
                    d=self.driver('eight_private' if stage=='pullback' else 'guarded')
                    t,roots,token=d.prepare(); w=d.work; w.fault=fault
                    if stage=='pullback': m=d.workspace.mapping('nodes',roots[0][1])
                    else:
                        from scoped_source.direct_candidates import encode
                        block,raw=encode(d.owner,d.forest,d.contract,t,d.ordinal,candidates(d.forest,t))
                        block.release()
                    before=d.workspace.traversal_stats['scoped_calls']
                    def after_publish(i,raw):
                        w.remaining=5; return publish(i,raw)
                    with self.assertRaises((TimeoutError,RuntimeError)):
                        if stage=='pullback':
                            w.remaining=5; d.workspace.pullback([m])
                        else: d.receive(token,raw,after_publish)
                    if stage=='pullback':
                        self.assertEqual(d.workspace.traversal_stats['scoped_calls'],before+1)
                    d.abort(); self.assertEqual(d.receipts,[])
                    self.assertEqual(d.owner.stats['live_bytes'],0)
        ws=self.workspace(); w=ws.owner.work; m=ws.mapping('nodes',{})
        w.counts['row_visits']=POLICY['visits']
        with self.assertRaisesRegex(ValueError,'capacity'): ws.pullback([m])
        self.assertGreater(sum(w.counts.values()),POLICY['visits'])
        f,w=self.forest(); ref=Owner(f,w); self.addCleanup(ref.close); s=Scope(ref,(0,1))
        # A separate smaller arena control, never a production limit increase.
        size=ref.stats['live_bytes']; s.close(); ref.close()
        o=Owner(f,w,capacity=size); self.addCleanup(o.close); s=Scope(o,(0,1))
        self.addCleanup(s.close)
        with self.assertRaisesRegex(ValueError,'overlap'): s.prepare_traversal(o,(0,1))
        self.assertFalse(s.traversal_ready)

    def test_10_binding_after_prepare_and_failed_publication(self):
        for mode in ('late','wrong_ack','exception','partial_buffer'):
            d=self.driver('guarded'); tasks,_,token=d.prepare(); rr=candidates(d.forest,tasks)
            def sink(i,raw):
                if mode=='late': d.work.deadline=time.monotonic()-1
                if mode=='wrong_ack': return '0'*64,len(raw)
                if mode=='exception': raise OSError('fixed publication fault')
                if mode=='partial_buffer': d.workspace.scope.banks['nodes'][3].release()
                return publish(i,raw)
            with self.assertRaises((ValueError,TimeoutError,OSError)): d.submit(token,rr,sink)
            self.assertTrue(d.closed); self.assertEqual(d.receipts,[])
            self.assertEqual(d.owner.stats['live_bytes'],0)
            with self.assertRaises(ValueError): d.finish()
