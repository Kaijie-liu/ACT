"""Synthetic controls only. Run with act-py312 -B -I -S; no native solver."""
import copy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Work,POLICY,digest,identity,compact
from scoped_source.scaled_endpoint_controls import fixture,expanded,scalar_bound
from scoped_source.scaled_references import build
from scoped_source.check_scaled_references import CheckedScaledForest
from scoped_source.scaled_endpoints import Session,contract
from scoped_source.bounded_runtime import Owner,Workspace,ExactMap,RootTable,ROOT,Block,encode_scalar,decode_scalar,scalar_size
from scoped_source.direct_candidates import Driver,Tile,encode,PREFIX,RECORD,RATIONAL,MAX_TOTAL
from scoped_source.wire_archive import Archive


def work(): return Work(time.monotonic()+300)


def candidates(f,tasks,*,missing=False,general=False):
    result=[]
    for t in tasks:
        if missing: result.append(dict(slot=t.slot,proposed=None,zero=None)); continue
        eq=len(f.rows(t.pair,'eq')); ine=len(f.rows(t.pair,'ine'))
        proposed=dict(query=t.query_id,equality=[str(F((j%3)-1,3 if general else 8)) for j in range(eq)],
                      inequality=[str(F(-1,3) if general else F(-(j%2),16)) for j in range(ine)],lower='1/3')
        zero=dict(query=t.query_id,equality=['0']*eq,inequality=['0']*ine,lower='0')
        result.append(dict(slot=t.slot,proposed=proposed,zero=zero))
    return result


def publish(i,raw): return digest(raw),len(raw)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='hz-owned-controls-',dir='/data1/Kane/MOE')
        cls.root=Path(cls.tmp.name); cls.cases={}; cls.differentials=[]
        for name in ('mixed','deep','half','guarded'):
            root,source=fixture(cls.root/name,name); raw,_=build(root,source,work=work())
            cls.cases[name]=(root,source,raw)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def forest(self,name='deep'):
        root,source,raw=self.cases[name]; w=work()
        return CheckedScaledForest(raw,root,source,expected=digest(raw),work=w),w

    def driver(self,name='deep',**kw):
        f,w=self.forest(name); c=contract(f)
        d=Driver(f,c,expected=identity(c),work=w,**kw)
        self.addCleanup(d.abort); return d

    def prepared(self,**kw):
        d=self.driver(**kw); tasks,_,token=d.prepare()
        return d,tasks,token,candidates(d.forest,tasks)

    def raw(self,d,tasks,records):
        b,raw=encode(d.owner,d.forest,d.contract,tasks,d.ordinal,records)
        b.release(); return raw

    def rejected(self,d,fn,exc=(ValueError,TimeoutError,OSError)):
        before=dict(d.work.counts)
        with self.assertRaises(exc): fn()
        self.assertTrue(d.closed); self.assertTrue(d.owner.closed)
        self.assertEqual(d.owner.stats['live_bytes'],0)
        self.assertEqual(d.owner.stats['allocated_bytes'],d.owner.stats['released_bytes'])
        self.assertTrue(all(d.work.counts[k]>=v for k,v in before.items()))
        with self.assertRaises(ValueError): d.prepare()
        with self.assertRaises(ValueError): d.finish()

    def test_full_exact_differential_both_profiles(self):
        for name in self.cases:
            for width in (1,8):
                for profile in (0,1):
                    d=self.driver(name,width=width); f,w=self.forest(name); c=contract(f)
                    old=Session(f,c,expected=identity(c),work=w,width=width); oracle=expanded(f,w)
                    receipts=[]; comparisons=0
                    for i in range(d.session.tile_count):
                        tasks,_,token=d.prepare(); oldtasks,_=old.prepare(i); self.assertEqual(tasks,oldtasks)
                        records=candidates(f,tasks,general=profile==1)
                        frozen_records=copy.deepcopy(records)
                        with patch('scoped_source.scaled_endpoints.compact',side_effect=AssertionError('no candidate JSON')):
                            got=d.submit(token,records,publish,profile=profile)
                        ref=old.accept(records); receipts.extend(ref)
                        self.assertEqual(records,frozen_records)
                        for task,r,a,b in zip(tasks,records,got,ref,strict=True):
                            aa=dict(a.record); bb=dict(b.record)
                            self.assertNotEqual(aa.pop('raw_tile'),bb.pop('raw_tile')); self.assertEqual(aa,bb)
                            for role in ('proposed','zero'):
                                bound,_=scalar_bound(f,task,r[role],oracle,w)
                                self.assertEqual(bound,F(a.record[role])); comparisons+=1
                    result=d.finish(); self.assertEqual(result,old.finish(receipts))
                    self.assertFalse(result['native_float_proof']); self.assertFalse(result['checkpoint_correspondence_checked'])
                    self.assertEqual(d.owner.stats['live_bytes'],0)
                    self.assertEqual(d.owner.stats['allocated_bytes'],d.owner.stats['released_bytes'])
                    self.differentials.append(dict(case=name,width=width,profile=profile,candidate_checks=comparisons,
                        result=result,owner=dict(d.owner.stats),work=dict(d.work.counts),bytes=d.total_bytes))

    def test_scalar_roundtrip_and_limits(self):
        for bits in (4096,8192):
            values=(F(0),F(-3,8),F(1,3),F((1<<bits)-1),F(1,(1<<bits)-1),F(1,1<<(bits-1)))
            for q in values:
                raw=encode_scalar(q,bits); self.assertEqual(len(raw),scalar_size(bits))
                self.assertEqual(decode_scalar(raw,bits),q)
                if q.denominator & (q.denominator-1)==0:
                    raw=encode_scalar(q,bits,dyadic=True); self.assertEqual(decode_scalar(raw,bits,dyadic=True),q)
            for q in (F(1<<bits),F(1,1<<bits)):
                with self.assertRaises(ValueError): encode_scalar(q,bits)
        with self.assertRaises(ValueError): encode_scalar(F(1,3),8192,dyadic=True)
        raw=encode_scalar(F(1,3),4096)
        for broken in (raw[:-1],raw+b'X',bytes([0])+raw[1:],raw[:1]+bytes([2])+raw[2:]):
            with self.assertRaises(ValueError): decode_scalar(broken,4096)
        raw=bytearray(encode_scalar(F(1,2),4096)); raw[16]=2
        with self.assertRaises(ValueError): decode_scalar(bytes(raw),4096)

    def test_exact_map_and_no_mutable_alias(self):
        f,w=self.forest(); owner=Owner(f,w); self.addCleanup(owner.close)
        m=ExactMap(owner,3); m[0]=F(0); m[2]=F(1,3)
        self.assertEqual(dict(m),{0:F(0),2:F(1,3)}); self.assertNotIn(1,m)
        value=m[2]; m[2]=F(5); self.assertEqual(value,F(1,3))
        del m[0]; self.assertEqual(dict(m),{2:F(5)})
        with self.assertRaises(ValueError): m[3]=F(0)
        b=owner.allocate(4); b.put(0,b'abcd'); raw=b.seal()
        with self.assertRaises(ValueError): b.put(0,b'xxxx')
        self.assertEqual(raw,b'abcd'); b.release(); self.assertEqual(raw,b'abcd')
        with self.assertRaises(ValueError): b.get(0,1)
        m.close(); owner.close(); self.assertEqual(owner.stats['live_bytes'],0)

    def test_allocation_and_seal_preflight(self):
        f,w=self.forest(); o=Owner(f,w,capacity=4); self.addCleanup(o.close)
        with self.assertRaises(ValueError): o.allocate(5)
        self.assertEqual(o.stats['allocated_bytes'],0)
        b=o.allocate(3)
        with self.assertRaises(ValueError): b.seal()
        self.assertEqual(o.stats['allocated_bytes'],3)
        with self.assertRaises(ValueError): Owner(f,w,capacity=POLICY['total_bytes']+1)
        with self.assertRaises(ValueError): Block(o,5)
        with self.assertRaises(AttributeError): o.capacity=POLICY['total_bytes']
        with self.assertRaises(AttributeError): o.work=work()

    def test_root_membership_center_owner_and_identity(self):
        for mode in ('center','owner','member'):
            d=self.driver(); rt=d.workspace.roots; f=d.forest
            vi=f.roles['experts'][0]; row=(vi,'output',0); c,n=rt.row(row)
            if mode=='center': rt.roots.put(n*ROOT.size,ROOT.pack(float(c)+1,f.nodes[n][0],1))
            elif mode=='owner': rt.roots.put(n*ROOT.size,ROOT.pack(float(c),f.nodes[n][0]+1,1))
            else: rt.members[vi,'output'].put(0,struct.pack('<I',(n+1)%len(f.nodes)))
            self.rejected(d,lambda:d.prepare())
        d=self.driver(); f,w=self.forest()
        with self.assertRaises(ValueError): d.workspace.bind(f,w)

    def test_cross_owner_stale_token_and_partial_candidates(self):
        d,t,token,r=self.prepared(); other,_,other_token,_=self.prepared()
        self.rejected(d,lambda:d.submit(other_token,r,publish))
        d,t,token,r=self.prepared(); d.submit(token,r,publish)
        self.rejected(d,lambda:d.submit(token,r,publish))
        d,t,token,r=self.prepared(); r[0]['zero']=None
        self.rejected(d,lambda:d.submit(token,r,publish))

    def test_raw_faults_and_no_mutable_intake(self):
        for mode in ('short','trailing','request','forest','contract','ordinal','dims','roles','slot','query','nan','negative_zero','mutable'):
            d,t,token,r=self.prepared(); raw=self.raw(d,t,r); buf=bytearray(raw)
            if mode=='short': bad=raw[:-1]
            elif mode=='trailing': bad=raw+b'X'
            elif mode=='mutable': bad=buf
            else:
                if mode in ('request','forest','contract'): buf[8+32*('request','forest','contract').index(mode)]^=1
                elif mode in ('ordinal','dims'):
                    fields=list(PREFIX.unpack_from(buf)); fields[4 if mode=='ordinal' else 6]+=1; PREFIX.pack_into(buf,0,*fields)
                elif mode=='roles': buf[PREFIX.size+65]=1
                elif mode=='slot': buf[PREFIX.size]^=1
                elif mode=='query': buf[PREFIX.size+32]^=1
                else:
                    offset=PREFIX.size+RECORD.size+2*RATIONAL
                    struct.pack_into('<d',buf,offset,math.nan if mode=='nan' else -0.)
                bad=bytes(buf)
            self.rejected(d,lambda:d.receive(token,bad,publish))

    def test_sign_zero_dimensions_and_advisory_lower(self):
        for mode in ('sign','zero','dimension','bad_lower'):
            d,t,token,r=self.prepared()
            if mode=='sign': r[0]['proposed']['inequality'][0]='1'
            elif mode=='zero': r[0]['zero']['inequality'][0]='-1'
            elif mode=='dimension': r[0]['proposed']['equality'].append('0')
            else: r[0]['proposed']['lower']='01'
            self.rejected(d,lambda:d.submit(token,r,publish))
        d,t,token,r=self.prepared(); r[0]['proposed']['lower']='999999999'
        got=d.submit(token,r,publish)
        self.assertNotEqual(got[0].record['lower'],'999999999')

    def test_native_does_not_round_general_rational(self):
        d,t,token,r=self.prepared(); r[0]['proposed']['inequality'][0]='-1/3'
        self.rejected(d,lambda:d.submit(token,r,publish))

    def test_accessor_immutable_and_foreign_unissued_rejected(self):
        d,t,token,r=self.prepared(); raw=self.raw(d,t,r)
        tile=Tile(raw,d.owner,d.forest,d.contract,t,0)
        with self.assertRaises(ValueError): tile.raw=b'corrupt'
        with self.assertRaises(AttributeError): del tile.raw
        with self.assertRaises(ValueError): d.session.accept_direct(tile)
        other=self.driver()
        with self.assertRaises(ValueError): Tile(raw,other.owner,d.forest,d.contract,t,0)

    def test_late_terminal_and_early_terminal_poison(self):
        d=self.driver(); self.rejected(d,d.finish)
        d=self.driver('half')
        for i in range(d.session.tile_count):
            tasks,_,token=d.prepare(); d.submit(token,candidates(d.forest,tasks),publish)
        original=d.session.finish
        def late(receipts):
            result=original(receipts); d.work.deadline=time.monotonic()-1; return result
        with patch.object(d.session,'finish',side_effect=late): self.rejected(d,d.finish)

    def test_missing_is_unknown_not_positive(self):
        d=self.driver('half')
        for i in range(d.session.tile_count):
            tasks,_,token=d.prepare(); d.submit(token,candidates(d.forest,tasks,missing=True),publish)
        result=d.finish(); self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['checked_endpoints'],0)
        self.rejected(d,d.finish)

    def test_publication_order_failure_and_no_accept(self):
        for mode in ('failure','wrong_ack','late'):
            d,t,token,r=self.prepared(); called=[]
            def pub(i,raw):
                called.append(raw)
                if mode=='failure': raise OSError('injected publication fault')
                if mode=='late': d.work.deadline=time.monotonic()-1
                return ('0'*64,len(raw)) if mode=='wrong_ack' else publish(i,raw)
            with patch.object(d.session,'accept_direct',side_effect=AssertionError('must not accept')):
                self.rejected(d,lambda:d.submit(token,r,pub))
            self.assertEqual(len(called),1); self.assertGreater(d.total_bytes,0)

    def test_deadline_before_during_and_after_math(self):
        for mode in ('before','during','after'):
            d,t,token,r=self.prepared()
            if mode=='before':
                d.work.deadline=time.monotonic()-1
                self.rejected(d,lambda:d.submit(token,r,publish)); continue
            original=d.session.accept_direct
            def late(tile):
                if mode=='during': d.work.deadline=time.monotonic()-1
                result=original(tile)
                if mode=='after': d.work.deadline=time.monotonic()-1
                return result
            with patch.object(d.session,'accept_direct',side_effect=late):
                self.rejected(d,lambda:d.submit(token,r,publish))
            self.assertEqual(d.receipts,[])

    def test_work_and_candidate_bytes_never_refunded(self):
        d,t,token,r=self.prepared(); d.work.counts['row_visits']=POLICY['visits']
        self.rejected(d,lambda:d.submit(token,r,publish))
        self.assertGreater(sum(d.work.counts.values()),POLICY['visits'])
        d,t,token,r=self.prepared(); d.total_bytes=MAX_TOTAL
        allocations=d.owner.stats['allocated_bytes']
        self.rejected(d,lambda:d.submit(token,r,publish))
        self.assertEqual(d.owner.stats['allocated_bytes'],allocations)
        self.assertEqual(d.total_bytes,MAX_TOTAL)

    def test_legacy_failed_slot_is_still_charged(self):
        f,w=self.forest(); c=contract(f); s=Session(f,c,expected=identity(c),work=w)
        tasks,_=s.prepare(0); r=candidates(f,tasks); r[0]['slot']='0'*64
        raw=compact(r)
        with self.assertRaises(ValueError): s.accept(raw)
        self.assertEqual(s._Session__bytes,len(raw))
        s.abort(); self.assertEqual(s._Session__bytes,len(raw))

    def test_duplicate_members_checked_before_root_accumulation(self):
        # A synthetic consumer-only control: duplicate an already checked row,
        # not a new source artifact or a source-correspondence acceptance test.
        for role,weights in (('proposed',['-2','1']),('zero',['1','-1'])):
            d=self.driver(); original=type(d.forest).rows
            def duplicates(f,pair,kind):
                rows=original(f,pair,kind)
                return (rows[0],rows[0],*rows[1:]) if kind=='ine' and rows else rows
            with patch.object(type(d.forest),'rows',new=duplicates):
                tasks,_,token=d.prepare(); r=candidates(d.forest,tasks)
                r[0][role]['inequality'][:2]=weights
                self.rejected(d,lambda:d.submit(token,r,publish))

    def test_release_cutoff_and_partial_receipt_discard(self):
        d,t,token,r=self.prepared(); original=Block.release
        def late_release(block):
            sealed=block._sealed is not None
            original(block)
            if sealed: d.work.deadline=time.monotonic()-1
        with patch.object(Block,'release',new=late_release):
            self.rejected(d,lambda:d.submit(token,r,publish))
        self.assertEqual(d.receipts,[])
        d,t,token,r=self.prepared(); raw=self.raw(d,t,r)
        original=Tile.weight; calls=0
        def fail(tile,*args):
            nonlocal calls
            calls+=1
            if calls==4: raise RuntimeError('injected mid-consumption failure')
            return original(tile,*args)
        with patch.object(Tile,'weight',new=fail):
            self.rejected(d,lambda:d.receive(token,raw,publish),RuntimeError)
        self.assertEqual(d.receipts,[])

    def test_failed_archive_prefix_has_separate_physical_ledger(self):
        d,t,token,r=self.prepared(); arch=Archive(self.root/'failed-typed-archive',d.work)
        def pub(i,raw):
            arch.put(f'tile-{i}.bin',raw)
            raise OSError('fault after durable write')
        self.rejected(d,lambda:d.submit(token,r,pub))
        self.assertEqual(d.costs['acknowledged_bytes'],0)
        self.assertEqual(d.costs['publication_attempted_bytes'],arch.bytes)
        self.assertGreater(arch.bytes,0)
        self.assertTrue((arch.root/'tile-0.bin').is_file())

    def test_direct_wire_source_terminal_reseal(self):
        from scoped_source.wire_forest import encode as encode_forest,CheckedWireForest
        root,source,raw=self.cases['half']
        for tamper in (False,True):
            destination=self.root/f'wire-source-{tamper}'; shutil.copytree(root,destination)
            w=work(); wire=encode_forest(raw,destination,source,expected=digest(raw),work=w)
            f=CheckedWireForest(wire,destination,source,expected=digest(wire),work=w); c=contract(f)
            d=Driver(f,c,expected=identity(c),work=w); self.addCleanup(d.abort)
            for i in range(d.session.tile_count):
                tasks,_,token=d.prepare(); d.submit(token,candidates(f,tasks),publish)
            if tamper:
                p=destination/'manifest.json'; content=bytearray(p.read_bytes()); content[-1]^=1; p.write_bytes(content)
                self.rejected(d,d.finish)
            else: self.assertEqual(d.finish()['status'],'CHECKED_POSITIVE_CAPTURED_REAL_SOURCE')

    def test_archive_relocation_direct_bytes_and_no_dependencies(self):
        d=self.driver('half'); arch=Archive(self.root/'typed-archive',d.work)
        def pub(i,raw):
            a=arch.put(f'tile-{i}.bin',raw); return a['sha256'],a['bytes']
        for i in range(d.session.tile_count):
            tasks,_,token=d.prepare(); d.submit(token,candidates(d.forest,tasks),pub)
        result=d.finish(); root,source,raw=self.cases['half']
        moved=self.root/'relocated'; moved.mkdir(); shutil.copytree(root,moved/'source')
        shutil.copytree(arch.root,moved/'evidence')
        # Actual control artifacts, not code/config files; test temporary tree.
        (moved/'forest.bin').write_bytes(raw)
        modules=('check_source_dag','check_affine_references','check_scaled_references','forest_endpoints','scaled_endpoints',
                 'bounded_runtime','direct_candidates','wire_forest','tensor_rows','factored_io','factored_source','hz_request_store')
        repo=Path(__file__).resolve().parents[1]
        for path in [*(f'scoped_source/{m}.py' for m in modules),'source_enclosure/format.py','upstream_source/checker.py']:
            dest=moved/path; dest.parent.mkdir(exist_ok=True); shutil.copyfile(repo/path,dest)
        program='''import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scoped_source.hz_request_store import Work,digest,identity
from scoped_source.check_scaled_references import CheckedScaledForest
from scoped_source.scaled_endpoints import contract
from scoped_source.direct_candidates import Driver
raw=Path('forest.bin').read_bytes(); w=Work(time.monotonic()+300)
f=CheckedScaledForest(raw,'source',sys.argv[1],expected=digest(raw),work=w)
c=contract(f); d=Driver(f,c,expected=identity(c),work=w)
for i in range(d.session.tile_count):
    _,_,token=d.prepare(); raw=Path('evidence/tile-%d.bin'%i).read_bytes()
    d.receive(token,raw,lambda i,b:(digest(b),len(b)))
print(json.dumps(d.finish(),sort_keys=True))
assert not any(m in sys.modules for m in ('torch','numpy','scipy'))
'''
        p=subprocess.run([sys.executable,'-B','-I','-S','-c',program,source],cwd=moved,
                         capture_output=True,text=True,timeout=30,check=True)
        self.assertEqual(json.loads(p.stdout),result)
        self.assertEqual(d.total_bytes,arch.bytes); self.assertEqual(d.costs['acknowledged_bytes'],arch.bytes)


if __name__=='__main__': unittest.main()
