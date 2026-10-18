"""Two frozen synthetic sources; no model, numerical backend or optimizer."""
import copy
from fractions import Fraction as F
import hashlib
import shutil
import sys
import time
import unittest

from scoped_source.hz_request_store import Work,POLICY,digest,identity
from scoped_source.forest_endpoint_controls import capture,fixture
from scoped_source.shared_build import build
from scoped_source.check_shared_source import CheckedSharedForest
from scoped_source.scaled_endpoint_controls import expanded,scalar_bound
from scoped_source.scaled_endpoints import contract
from scoped_source.pair_workspace import PairWorkspace,Scope,U32
from scoped_source.bounded_runtime import Owner
from scoped_source.direct_candidates import Driver
from scoped_source.direct_lifecycle import prepare as leased_prepare,revoke
from scripts.test_support_traversal import special
from scripts.test_bounded_runtime import candidates,publish

ROOT=None
DEADLINE=None
RECORDS=[]
ARMS=((False,False),(True,False),(False,True),(True,True))
PROFILES={'guarded':(0,1),'eight_private_six_outputs':(0,)}


class ControlWork(Work):
    def __init__(self):
        self.point=None; self.remaining=None; self.fault='exception'; self.trace=None
        super().__init__(DEADLINE)

    def inject(self):
        self.point=None; self.remaining=None
        if self.fault=='deadline':
            self.deadline=time.monotonic()-1
            super().tick()
        raise RuntimeError('fixed lifetime failure')

    def visit(self,kind,amount=1):
        super().visit(kind,amount)
        if self.point==sys._getframe(1).f_code.co_name:
            self.remaining-=1
            if self.remaining==0: self.inject()

    def tick(self):
        super().tick()
        if self.point=='finish_tile' and sys._getframe(1).f_code.co_name==self.point:
            self.inject()

    def exact(self,q):
        if self.trace is not None: self.trace.update((str(q)+'\n').encode())
        return super().exact(q)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={}; cls.saved={}; started=time.monotonic()
        doc=special('eight_private')
        for expert in doc['experts']:
            expert[-1]['rows'].extend(copy.deepcopy(expert[-1]['rows'][1:2])*4)
        path=ROOT/'eight_private_six_outputs'; source=capture(path,doc)
        items=[('eight_private_six_outputs',path,source)]
        path,source=fixture(ROOT/'guarded','guarded'); items.append(('guarded',path,source))
        for name,path,source in items:
            w=ControlWork(); packet,construction=build(path,source,work=w,enabled=True)
            cls.cases[name]=(path,source,packet)
            RECORDS.append(dict(kind='source',name=name,source=source,forest=packet.identity,
                                work=dict(w.counts),construction=construction))
        RECORDS.append(dict(kind='setup',seconds=time.monotonic()-started))

    def forest(self,name='eight_private_six_outputs',root=None):
        original,source,packet=self.cases[name]; w=ControlWork()
        f=CheckedSharedForest(packet,root or original,source,expected=packet.identity,work=w,enabled=True)
        return f,w

    def driver(self,name='eight_private_six_outputs',arm=(True,True),**kw):
        f,w=self.forest(name,kw.pop('root',None)); c=contract(f)
        d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True,
                 scoped_traversal=arm[0],retain_pair_scope=arm[1],**kw)
        self.addCleanup(d.abort); return d

    def advance(self,d):
        t,r,token=d.prepare(); scope=d.workspace.scope; lease=d.workspace._lease
        d.submit(token,candidates(d.forest,t),publish)
        return scope,lease,t,r,token

    def rejected(self,d,fn):
        counts=dict(d.work.counts); lease=d.workspace._lease
        with self.assertRaises((ValueError,TimeoutError,RuntimeError,OSError)): fn()
        self.assertTrue(d.closed); self.assertTrue(d.owner.closed)
        self.assertEqual(d.receipts,[]); self.assertEqual(d.owner.stats['live_bytes'],0)
        self.assertEqual(d.owner.stats['allocated_bytes'],d.owner.stats['released_bytes'])
        self.assertIsNone(d.workspace._lease); self.assertIsNone(d.workspace._session)
        if lease is not None:
            self.assertTrue(lease.closed); self.assertEqual(lease.records,()); self.assertIsNone(lease.owner)
        self.assertTrue(all(d.work.counts[k]>=v for k,v in counts.items()))
        with self.assertRaises(ValueError): d.finish()

    def test_01_defaults_and_limits(self):
        f,w=self.forest('guarded'); c=contract(f)
        d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True); self.addCleanup(d.abort)
        self.assertFalse(d.workspace.retain_pair_scope)
        for bad in (1,None,'yes'):
            with self.assertRaises(ValueError): PairWorkspace(f,w,retain_pair_scope=bad)
        with self.assertRaises(ValueError): Driver(f,c,expected=identity(c),work=w,retain_pair_scope=True)
        with self.assertRaises(AttributeError): d.workspace.retain_pair_scope=True
        self.assertEqual((POLICY['visits'],POLICY['total_bytes'],POLICY['work_bits']),
                         (8000000,134217728,8192))

    def test_02_full_four_arm_differential(self):
        for name in ('guarded','eight_private_six_outputs'):
            for profile in PROFILES[name]:
                reference=None; oracle_start=time.monotonic()
                f,w=self.forest(name); matrix=expanded(f,w); oracle_checks=0
                oracle_pre_seconds=time.monotonic()-oracle_start
                order=range(4) if profile==0 else reversed(range(4))
                for arm_index in order:
                    arm=ARMS[arm_index]; started=time.monotonic(); d=self.driver(name,arm)
                    receipts=[]; frames=[]; previous=None; held=[]; pair_count=0
                    for ordinal in range(d.session.tile_count):
                        tasks,roots,token=d.prepare(); scope=d.workspace.scope
                        if previous is not None and previous[0]==tasks[0].pair and arm[1]:
                            self.assertIs(scope,previous[1])
                        if previous is None or previous[0]!=tasks[0].pair: pair_count+=1
                        previous=(tasks[0].pair,scope)
                        records=candidates(d.forest,tasks,general=bool(profile)); saved=copy.deepcopy(records)
                        def sink(i,raw):
                            frames.append(raw); return publish(i,raw)
                        lease=d.workspace._lease
                        got=d.submit(token,records,sink,profile=profile)
                        self.assertEqual(records,saved)
                        if scope.closed and lease is not None:
                            self.assertTrue(lease.closed); self.assertEqual(lease.records,())
                        receipts.extend(dict(receipt.record) for receipt in got)
                        with self.assertRaises(ValueError): next(iter(roots[0][1]))
                        self.assertEqual(d.workspace._maps,[]); self.assertEqual(d.workspace._descriptors,[])
                    self.assertIsNone(d.workspace.scope)  # before finish/source seal
                    result=d.finish(); self.assertEqual(d.owner.stats['live_bytes'],0)
                    self.assertEqual(d.owner.stats['allocated_bytes'],d.owner.stats['released_bytes'])
                    value=(receipts,result)
                    if reference is None: reference=value
                    else: self.assertEqual(value,reference)
                    ls=d.workspace.lifetime_stats
                    self.assertEqual(ls['scopes_created'],pair_count if arm[1] else d.session.tile_count)
                    self.assertEqual(ls['scopes_retired'],ls['scopes_created'])
                    expected=(140,280,56) if name=='eight_private_six_outputs' else (12,24,3)
                    self.assertEqual((result['required'],result['required_endpoints'],d.session.tile_count),expected)
                    RECORDS.append(dict(kind='paired',source=name,profile=profile,arm=arm_index,
                        work=dict(d.work.counts),owner=dict(d.owner.stats),lifetime=dict(ls),
                        traversal=dict(d.workspace.traversal_stats),costs=dict(d.costs),
                        seconds=time.monotonic()-started,receipts=receipts,
                        receipts_sha256=identity(receipts),result=result,
                        candidate_sha256=hashlib.sha256(b''.join(frames)).hexdigest()))
                    if arm_index==3 and profile==PROFILES[name][-1]:
                        self.saved[name]=(profile,frames,receipts,result)
                oracle_start=time.monotonic()
                for task,receipt in zip(d.session.tasks,reference[0],strict=True):
                    record=candidates(f,[task],general=bool(profile))[0]
                    for role in ('proposed','zero'):
                        bound,_=scalar_bound(f,task,record[role],matrix,w)
                        self.assertEqual(bound,F(receipt[role])); oracle_checks+=1
                RECORDS.append(dict(kind='oracle',source=name,profile=profile,
                    seconds=oracle_pre_seconds+time.monotonic()-oracle_start,
                    candidate_checks=oracle_checks,work=dict(w.counts)))

    def test_03_missing_complete_denominator(self):
        for arm in ((False,True),(True,True)):
            for count in (1,2,8):
                d=self.driver('eight_private_six_outputs',arm); missing=0
                for _ in range(d.session.tile_count):
                    t,_,token=d.prepare(); rr=candidates(d.forest,t)
                    for r in rr[:count]: r['proposed']=r['zero']=None
                    missing+=min(count,len(t)); d.submit(token,rr,publish)
                result=d.finish()
                self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
                self.assertEqual(result['missing_endpoints'],missing)
                self.assertEqual(result['required'],140)

    def test_04_cross_tile_block_identity(self):
        for arm in ((False,True),(True,True)):
            for mode in ('bytes','owner','size','mutable','released'):
                d=self.driver(arm=arm); _,lease,*_=self.advance(d); b=lease.records[0][0]
                if mode=='bytes': b._sealed=bytes(bytearray(b._sealed))
                elif mode=='owner':
                    other=Owner(d.forest,d.work); self.addCleanup(other.close); b.owner=other
                elif mode=='size': b.size+=4
                elif mode=='mutable': b._data=bytearray(b._sealed)
                else: b.release()
                self.rejected(d,d.prepare)

    def test_05_plan_pair_and_stale_token(self):
        for mode in ('plan','pair','index','token'):
            d=self.driver(); _,_,_,_,oldtoken=self.advance(d)
            if mode=='plan': d.session._Session__tiles=tuple(list(d.session._Session__tiles))
            elif mode=='pair': d.workspace._pair=(0,2)
            elif mode=='index': d.workspace._next+=1
            else:
                t,_,_=d.prepare()
                self.rejected(d,lambda:d.submit(oldtoken,candidates(d.forest,t),publish)); continue
            self.rejected(d,d.prepare)
        d=self.driver(); other=self.driver()
        with self.assertRaises(ValueError): d.workspace.configure(other.session)

    def test_06_old_tile_handles_expire(self):
        d=self.driver(); t,roots,token=leased_prepare(d); scope=d.workspace.scope
        root=roots[0][1]; first=next(iter(root)); m=d.workspace.mapping('nodes',{})
        d.submit(token,candidates(d.forest,t),publish)
        with self.assertRaises(ValueError): root[first]
        with self.assertRaises(ValueError): m.get(0,F(0))
        with self.assertRaises(ValueError): d.workspace.mapping('nodes')
        with self.assertRaises(ValueError): d.workspace.objective(t[0])
        revoke(roots); self.assertIsNone(roots._roots)
        d.prepare(); self.assertIs(d.workspace.scope,scope)

    def test_07_initial_canonical_faults(self):
        for kind in ('nodes','factors'):
            for mode in ('missing','permuted'):
                f,w=self.forest('guarded'); o=Owner(f,w); self.addCleanup(o.close)
                s=Scope(o,(0,1)); self.addCleanup(s.close)
                _,selected,forward,reverse=s.banks[kind]
                self.assertGreater(selected,1)
                a,b=(s.global_id(kind,k) for k in (0,1))
                if mode=='missing': forward.put(4*a,U32.pack(0))
                else:
                    forward.put(4*a,U32.pack(2)); forward.put(4*b,U32.pack(1))
                    reverse.put(0,U32.pack(b)); reverse.put(4,U32.pack(a))
                with self.assertRaises(ValueError): s.prepare_traversal(o,(0,1))
                self.assertFalse(s.traversal_ready)

    def test_08_deadline_exception_and_partial_seal(self):
        for stage in ('seal','check','finish_tile'):
            for fault in ('deadline','exception'):
                d=self.driver(); w=d.work
                if stage=='check': self.advance(d)
                if stage=='finish_tile':
                    self.advance(d)  # second tile is the last tile of this pair
                    t,_,token=d.prepare()
                w.point=stage; w.remaining=2 if stage=='seal' else 1; w.fault=fault
                if stage=='finish_tile':
                    self.rejected(d,lambda:d.submit(token,candidates(d.forest,t),publish))
                else: self.rejected(d,d.prepare)
        # Partial worker reports remain failed artifacts, not lost terminal records.
        from scripts.run_pair_scope_lifetime_controls import decode_worker_report
        for raw in (b'{"status":"PASS"',b'[]'):
            with self.assertRaises(ValueError): decode_worker_report(raw)
        self.assertEqual(decode_worker_report(b'{"status":"FAIL"}'),{'status':'FAIL'})

    def test_09_work_and_arena_rejection(self):
        d=self.driver(); self.advance(d); d.work.counts['row_visits']=POLICY['visits']
        self.rejected(d,d.prepare)
        d=self.driver('guarded'); base=d.owner.stats['live_bytes']
        s=Scope(d.owner,(0,1)); size=d.owner.stats['live_bytes']; s.close(); d.abort()
        limited=self.driver('guarded',capacity=size)
        self.assertGreater(size,base)
        self.rejected(limited,limited.prepare)

    def test_10_publication_and_later_candidate_faults(self):
        for mode in ('ack','late','exception','sign','zero','half_missing'):
            d=self.driver(); self.advance(d); t,_,token=d.prepare(); rr=candidates(d.forest,t)
            if mode=='sign': rr[-1]['proposed']['inequality'][0]='1'
            elif mode=='zero': rr[-1]['zero']['inequality'][0]='-1'
            elif mode=='half_missing': rr[-1]['zero']=None
            def sink(i,raw):
                if mode=='ack': return '0'*64,len(raw)
                if mode=='late': d.work.deadline=time.monotonic()-1
                if mode=='exception': raise OSError('fixed publication error')
                return publish(i,raw)
            self.rejected(d,lambda:d.submit(token,rr,sink))

    def test_11_relocated_raw_receive(self):
        for name in ('guarded','eight_private_six_outputs'):
            started=time.monotonic(); root,source,_=self.cases[name]; dest=ROOT/('relocated_'+name)
            shutil.copytree(root,dest)
            d=self.driver(name,root=dest); profile,frames,receipts,result=self.saved[name]; actual=[]
            for raw in frames:
                _,_,token=d.prepare(); actual.extend(dict(r.record) for r in d.receive(token,raw,publish))
            self.assertEqual(actual,receipts); self.assertEqual(d.finish(),result)
            RECORDS.append(dict(kind='replay',source=name,profile=profile,seconds=time.monotonic()-started,
                receipts_sha256=identity(actual),result=result,work=dict(d.work.counts),owner=dict(d.owner.stats)))

    def test_12_arithmetic_trace_and_residual(self):
        hashes=[]
        for arm in ((False,True),(True,True)):
            d=self.driver(); t,_,token=d.prepare(); d.work.trace=hashlib.sha256()
            got=d.submit(token,candidates(d.forest,t,general=True),publish,profile=1)
            hashes.append((d.work.trace.hexdigest(),[dict(r.record) for r in got]))
            d.work.trace=None
        self.assertEqual(hashes[0],hashes[1])
        RECORDS.append(dict(kind='trace',source='eight_private_six_outputs',profile=1,
                            first_tile_arithmetic_sha256=hashes[0][0]))
