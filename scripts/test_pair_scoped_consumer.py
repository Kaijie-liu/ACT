"""Frozen synthetic pair-scope controls; no native optimization or tensors."""
import copy
from fractions import Fraction as F
from pathlib import Path
import shutil
import struct
import time
from unittest.mock import patch

from scripts import test_bounded_runtime as base
from scripts.test_bounded_runtime import candidates, publish, work
from scoped_source.hz_request_store import POLICY, digest, identity, compact, decode
from scoped_source.direct_candidates import Driver
from scoped_source.bounded_runtime import Owner, Workspace
from scoped_source.pair_workspace import PairWorkspace, Scope, ScopedMap, SparseRoots
from scoped_source.scaled_endpoints import Session, contract, pullback
from scoped_source.scaled_endpoint_controls import expanded, scalar_bound
from scoped_source.check_scaled_references import CheckedScaledForest
from scoped_source.wire_forest import CheckedWireForest, NodeTable, encode as encode_forest


class Controls(base.Controls):
    def driver(self,name='deep',**kw):
        return super().driver(name,pair_scoped=True,**kw)

    def test_full_exact_differential_both_profiles(self):
        for name in self.cases:
            for profile in (0,1):
                d=self.driver(name); f,w=self.forest(name); c=contract(f)
                old=Session(f,c,expected=identity(c),work=w,width=8)
                matrix=expanded(f,w); oldreceipts=[]; count=0
                dense_f,dense_w=self.forest(name)
                dense=Driver(dense_f,c,expected=identity(c),work=dense_w)
                self.addCleanup(dense.abort)
                for i in range(d.session.tile_count):
                    tasks,roots,token=d.prepare(); ot,_=old.prepare(i)
                    dt,_,dtoken=dense.prepare(); self.assertEqual(tasks,ot); self.assertEqual(tasks,dt)
                    self.assertEqual(d.workspace._maps,[])  # No dense objectives before publication.
                    self.assertTrue(all(len(a)<=4 for _,a in roots))
                    r=candidates(f,tasks,general=profile==1); saved=copy.deepcopy(r)
                    got=d.submit(token,r,publish,profile=profile)
                    ref=old.accept(r); oldreceipts.extend(ref)
                    dr=dense.submit(dtoken,r,publish,profile=profile)
                    self.assertEqual(r,saved)
                    for task,rr,a,b,cc in zip(tasks,r,got,ref,dr,strict=True):
                        self.assertEqual(a.record,cc.record)
                        aa=dict(a.record); bb=dict(b.record); aa.pop('raw_tile'); bb.pop('raw_tile')
                        self.assertEqual(aa,bb)
                        for role in ('proposed','zero'):
                            value,_=scalar_bound(f,task,rr[role],matrix,w)
                            self.assertEqual(value,F(a.record[role])); count+=1
                    self.assertFalse(d.workspace._maps); self.assertIsNone(d.workspace.scope)
                result=d.finish(); self.assertEqual(result,old.finish(oldreceipts))
                self.assertEqual(result,dense.finish()); self.assertEqual(d.owner.stats['live_bytes'],0)
                self.assertLessEqual(d.workspace.maximum_block,2)
                self.differentials.append(dict(case=name,profile=profile,candidate_width=8,check_width=2,
                    candidate_checks=count,result=result,scoped_owner=dict(d.owner.stats),
                    dense_owner=dict(dense.owner.stats),scoped_work=dict(d.work.counts),
                    dense_work=dict(dense.work.counts),blocks=d.workspace.blocks_checked,
                    maximum_block=d.workspace.maximum_block,bytes=d.total_bytes))

    def test_profile_fixed_and_default_unchanged(self):
        f,w=self.forest(); c=contract(f)
        for width in (1,2,7,9,True):
            with self.assertRaises(ValueError): Driver(f,c,expected=identity(c),work=w,width=width,pair_scoped=True)
        for mode in (1,None,'yes'):
            with self.assertRaises(ValueError): Driver(f,c,expected=identity(c),work=w,pair_scoped=mode)
        d=Driver(f,c,expected=identity(c),work=w); self.addCleanup(d.abort)
        self.assertIs(type(d.workspace),Workspace)

    def test_scope_private_and_other_guard_exclusion(self):
        for name in ('mixed','deep'):
            d=self.driver(name); d.prepare(); f=d.forest; s=d.workspace.scope
            self.assertEqual(s.pair,(0,1)); common=f.roles['common']
            expected=set(f.chain(common))|set(f.chain(f.roles['experts'][0],common))|set(f.chain(f.roles['experts'][1],common))
            expected.add(next(v for a,b,v in f.roles['pairs'] if (a,b)==(0,1)))
            self.assertEqual(s.views,expected)
            for a,b,v in f.roles['pairs']: self.assertEqual(v in s.views,(a,b)==(0,1))
            for kind,owners in (('nodes',[n[0] for n in f.nodes]),('factors',[r[3] for r in f.factors])):
                present=[]; absent=[]
                for j,o in enumerate(owners):
                    local=s.local(kind,j)
                    self.assertEqual(local is not None,o in expected)
                    if local is not None: present.append(j); self.assertEqual(s.global_id(kind,local),j)
                    else: absent.append(j)
                self.assertTrue(absent); m=d.workspace.mapping(kind)
                for j in present: m[j]=F(j+1)  # Equal scales/parameters never merge factor identities.
                self.assertEqual(list(m),present)
                before=d.owner.stats['allocated_bytes']; visits=sum(d.work.counts.values())
                m[absent[0]]=F(0); self.assertEqual(m.get(absent[0],F(0)),0)
                self.assertEqual(d.owner.stats['allocated_bytes'],before)
                self.assertGreater(sum(d.work.counts.values()),visits)
                with self.assertRaises(ValueError): m[absent[0]]=F(1)
                for bad in (-1,len(owners),True):
                    with self.assertRaises(ValueError): m[bad]=F(0)
                with self.assertRaises(ValueError): m[absent[0]]=0
                with self.assertRaises(ValueError): m[present[0]]=F(1<<8192)

    def test_scope_binding_and_map_pollution(self):
        d=self.driver(); d.prepare(); s=d.workspace.scope; other=self.driver(); other.prepare()
        with self.assertRaises(ValueError): s.bind(other.owner,s.pair)
        with self.assertRaises(ValueError): s.bind(d.owner,(0,2))
        f,w=self.forest()
        with self.assertRaises(ValueError): d.workspace.bind(f,w)
        with self.assertRaises(ValueError): d.workspace.bind(d.forest,work())
        with self.assertRaises(ValueError): s.pair=(0,2)
        with self.assertRaises(AttributeError): del s.pair
        with self.assertRaises(TypeError): s.banks['nodes']=other.workspace.scope.banks['nodes']
        j=next(j for j in range(len(d.forest.nodes)) if s.local('nodes',j) is not None)
        s.banks['nodes'][2].put(j*4,bytes(4))
        with self.assertRaises(ValueError): s.local('nodes',j)

    def test_closed_scope_rejected_before_allocation(self):
        d=self.driver(); d.prepare(); scope=d.workspace.scope
        m=d.workspace.mapping('nodes'); scope.close()
        before=d.owner.stats['allocated_bytes']
        with self.assertRaises(ValueError): scope.bind(d.owner,scope.pair)
        with self.assertRaises(ValueError): ScopedMap(scope,'nodes')
        with self.assertRaises(ValueError): m.take()
        self.assertEqual(d.owner.stats['allocated_bytes'],before)
        d.abort()
        with self.assertRaises(ValueError): scope.local('nodes',0)

    def test_transfer_and_intermediate_bits_not_cancelled(self):
        d=self.driver(); tasks,roots,_=d.prepare(); ws=d.workspace
        alpha=ws.mapping('nodes',roots[0][1]); original=dict(alpha)
        out=pullback(d.forest,[alpha],d.work,ws)[0]
        self.assertIsNone(alpha._slots)
        self.assertFalse(hasattr(alpha,'slots'))
        alpha.close()  # Cannot release the new owner's slots.
        with self.assertRaises(ValueError): alpha.get(next(iter(original)),F(0))
        with self.assertRaises(ValueError): alpha.take()
        self.assertEqual(dict(out),pullback(d.forest,[original],d.work)[0])
        # Deliberately large admitted working seed, not a candidate from a model.
        n,q=next((n,q) for n,(_,local,refs) in enumerate(d.forest.nodes)
                 if ws.scope.local('nodes',n) is not None
                 for _,q in (*local,*refs) if abs(q.numerator)>1)
        seed=F((1<<8192)-1); private=ws.mapping('nodes',{n:seed})
        with self.assertRaisesRegex(ValueError,'intermediate'):
            pullback(d.forest,[{n:seed}],d.work)
        with self.assertRaisesRegex(ValueError,'intermediate'):
            pullback(d.forest,[private],d.work,ws)
        with self.assertRaises(ValueError): private[n]=F(0)

    def test_sparse_root_alias_and_pair_binding(self):
        d=self.driver(); tasks,roots,_=d.prepare()
        for _,r in roots:
            if len(r):
                j=next(iter(r))
                with self.assertRaises(TypeError): r[j]=F(0)
        changed=tasks[0].__class__((0,2),tasks[0].other,tasks[0].weight,tasks[0].left,
                                  tasks[0].right,tasks[0].offset,tasks[0].query_id,tasks[0].slot)
        with self.assertRaises(ValueError): d.workspace.objective(changed)
        for r in d.workspace._descriptors:
            with self.assertRaises(ValueError): r.block.put(0,b'x')
        d.abort()
        with self.assertRaises(ValueError): dict(roots[0][1])

    def test_later_block_fault_poison_and_prefix_accounting(self):
        for mode in ('sign','zero','exception','deadline','release'):
            d,t,token,r=self.prepared(name='guarded'); self.assertEqual(len(t),8)
            if mode=='sign': r[2]['proposed']['inequality'][0]='1'
            if mode=='zero': r[2]['zero']['inequality'][0]='-1'
            call=d.session._consume_block; seen=[]
            def block(*args):
                start=args[4]; seen.append(start)
                if start==2:
                    if mode=='exception': raise RuntimeError('later check block fault')
                    if mode=='deadline': d.work.deadline=time.monotonic()-1
                result=call(*args)
                if mode=='release' and start==0: d.work.deadline=time.monotonic()-1
                return result
            with patch.object(d.session,'_consume_block',side_effect=block):
                self.rejected(d,lambda:d.submit(token,r,publish),(ValueError,TimeoutError,RuntimeError))
            self.assertEqual(seen,[0] if mode=='release' else [0,2])
            self.assertEqual(d.receipts,[]); self.assertGreater(d.total_bytes,0)
            self.assertEqual(d.session._Session__issued,{})
            self.assertEqual(d.session._Session__values,{})
            self.assertEqual(d.owner.stats['live_bytes'],0)

    def test_missing_in_one_block_does_not_drop_remaining_obligations(self):
        d=self.driver('guarded'); count=0
        for i in range(d.session.tile_count):
            t,_,token=d.prepare(); r=candidates(d.forest,t)
            r[0]['proposed']=None; r[0]['zero']=None; count+=1
            out=d.submit(token,r,publish)
            self.assertIsNone(out[0]); self.assertTrue(all(x is not None for x in out[1:]))
        result=d.finish(); self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['missing_endpoints'],count)
        self.assertEqual(result['required_endpoints'],len(d.session.tasks))

    def test_singleton_tail_block_and_odd_tile(self):
        # Consumer partition control on the unchanged full plan, not a new experiment.
        from scoped_source.forest_endpoints import Plan
        orig=Plan.tiles
        def odd(plan,width):
            for t in orig(plan,width):
                for i in range(0,len(t),3): yield t[i:i+3]
        for name,patch_tiles in (('half',False),('guarded',True)):
            ctx=patch.object(Plan,'tiles',new=odd if patch_tiles else orig)
            with ctx: d=self.driver(name)
            seen=[]; consume=d.session._consume_block
            def record(*args): seen.append(len(args[1])); return consume(*args)
            with patch.object(d.session,'_consume_block',side_effect=record):
                for i in range(d.session.tile_count):
                    t,_,token=d.prepare(); d.submit(token,candidates(d.forest,t),publish)
                d.finish()
            self.assertIn(1,seen); self.assertTrue(all(k<=2 for k in seen))

    def test_wire_metadata_without_source_decode(self):
        root,source,raw=self.cases['deep']; w=work()
        wire=encode_forest(raw,root,source,expected=digest(raw),work=w)
        f=CheckedWireForest(wire,root,source,expected=digest(wire),work=w)
        with patch.object(NodeTable,'__getitem__',side_effect=AssertionError('metadata decoded coefficients')):
            ws=PairWorkspace(f,w); self.addCleanup(ws.close); ws.select((0,1))
            vi=f.roles['experts'][0]; _,n=ws.roots.row((vi,'output',0))
            self.assertIsNotNone(ws.scope.local('nodes',n))

    def test_wire_excluded_source_still_resealed(self):
        root,source,raw=self.cases['mixed']
        for tamper in (False,True):
            dest=self.root/f'excluded-source-{tamper}'; shutil.copytree(root,dest); w=work()
            wire=encode_forest(raw,dest,source,expected=digest(raw),work=w)
            f=CheckedWireForest(wire,dest,source,expected=digest(wire),work=w); c=contract(f)
            d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True); self.addCleanup(d.abort)
            for i in range(d.session.tile_count):
                t,_,token=d.prepare(); d.submit(token,candidates(f,t),publish)
            if tamper:
                manifest=decode((dest/'manifest.json').read_bytes())
                tensor=next(t for t in manifest['tensors'] if t['name'].startswith('experts.2.'))
                path=dest/tensor['chunks'][0]['file']; data=bytearray(path.read_bytes()); data[0]^=1
                path.write_bytes(data)
                self.rejected(d,d.finish)
            else: d.finish()

    def test_rehashed_cross_private_source_ref_rejected(self):
        root,source,raw=self.cases['deep']; doc=decode(raw)
        target=next(i for i,n in enumerate(doc['nodes']) if doc['views'][n[0]]['scope']==0)
        victim=next(n for n in doc['nodes'] if doc['views'][n[0]]['scope']==1 and n[2])
        victim[2]=[[target,'1']]; bad=compact(doc)
        with self.assertRaises(ValueError):
            CheckedScaledForest(bad,root,source,expected=digest(bad),work=work())
