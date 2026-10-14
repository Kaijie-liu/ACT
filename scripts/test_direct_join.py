"""Frozen synthetic direct-source controls. No native/real model/GPU calls."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Work,POLICY,compact,decode,digest,identity
from scoped_source.scaled_endpoint_controls import fixture,expanded,scalar_bound
from scoped_source.scaled_references import build as legacy_build
from scoped_source.source_dag import Builder as LegacySerializer
from scoped_source.check_scaled_references import CheckedScaledForest,checked
from scoped_source.direct_source_build import build
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.direct_source_format import PREFIX,NODE,ENTRY,ROOT,intake,RootSequence
from scoped_source.scaled_endpoints import contract,Session
from scoped_source.direct_candidates import Driver
from scoped_source.direct_execution import execute
from scoped_source.direct_package import check_package,check_manifest
from scoped_source.wire_archive import Archive,run_child
from scripts.test_bounded_runtime import candidates

REPO=Path(__file__).resolve().parents[1]


def work(): return Work(time.monotonic()+300)


class Controls(unittest.TestCase):
    differentials=[]
    executions=[]

    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='hz-direct-controls-',dir='/data1/Kane/MOE')
        cls.root=Path(cls.temp.name); cls.cases={}
        for name in ('mixed','deep','half','guarded'):
            root,source=fixture(cls.root/name,name)
            with patch.object(LegacySerializer,'serialize',side_effect=AssertionError('old forest serialization forbidden')):
                raw,stats=build(root,source,work=work(),enabled=True)
            legacy,_=legacy_build(root,source,work=work())
            cls.cases[name]=(root,source,raw,legacy)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def forest(self,name='deep',old=False,w=None):
        root,source,raw,legacy=self.cases[name]; w=w or work()
        if old: return CheckedScaledForest(legacy,root,source,expected=digest(legacy),work=w),w
        return CheckedDirectForest(raw,root,source,expected=digest(raw),work=w,enabled=True),w

    def package(self,name='half',propose=candidates,**kw):
        root,source,_,_=self.cases[name]; target=self.root/('run-'+str(time.monotonic_ns()))
        result=execute(target,root,source,propose,REPO,work=kw.pop('work',work()),enabled=True,**kw)
        return target,result

    def reject(self,raw,name='deep'):
        root,source,_,_=self.cases[name]
        with self.assertRaises((ValueError,TypeError,KeyError,IndexError)):
            CheckedDirectForest(raw,root,source,expected=digest(raw),work=work(),enabled=True)

    def mutate_header(self,fn,name='deep'):
        raw=self.cases[name][2]; fields=list(PREFIX.unpack_from(raw)); h=fields[1]
        m=decode(raw[PREFIX.size:PREFIX.size+h]); fn(m); header=compact(m); fields[1]=len(header)
        return PREFIX.pack(*fields)+header+raw[PREFIX.size+h:]

    def test_same_candidate_exact_full_differential(self):
        for name in self.cases:
            for profile in (0,1):
                f,w=self.forest(name); old,ow=self.forest(name,old=True); dense,dw=self.forest(name)
                self.assertNotEqual(f.identity,old.identity)
                self.assertEqual(f.nodes,old.nodes); self.assertEqual(f.scales,old.scales)
                self.assertEqual(f.factors,old.factors); self.assertEqual(f.roles,old.roles)
                for v,ov in zip(f.views,old.views,strict=True):
                    for k in ('output','eq','ine'): self.assertEqual(tuple(v[k]),ov[k]); self.assertIs(type(v[k]),RootSequence)
                c=contract(f); dc=contract(dense); oc=contract(old)
                d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True); self.addCleanup(d.abort)
                dd=Driver(dense,dc,expected=identity(dc),work=dw); self.addCleanup(dd.abort)
                s=Session(old,oc,expected=identity(oc),work=ow); matrix=expanded(old,ow); receipts=[]; checks=0
                for i in range(d.session.tile_count):
                    t,_,token=d.prepare(); dt,_,dtoken=dd.prepare(); ot,_=s.prepare(i)
                    for a,b in zip(t,ot,strict=True):
                        self.assertEqual((a.pair,a.other,a.weight,a.left,a.right,a.offset),(b.pair,b.other,b.weight,b.left,b.right,b.offset))
                    r=candidates(f,t,general=profile==1); oldr=candidates(old,ot,general=profile==1)
                    pub=lambda i,b:(digest(b),len(b))
                    a=d.submit(token,r,pub,profile=profile); b=dd.submit(dtoken,candidates(dense,dt,general=profile==1),pub,profile=profile)
                    rr=s.accept(oldr); receipts.extend(rr)
                    for task,rec,ar,br,ref in zip(ot,oldr,a,b,rr,strict=True):
                        for role in ('proposed','zero'):
                            expected,_=scalar_bound(old,task,rec[role],matrix,ow)
                            self.assertEqual(F(ar.record[role]),expected); self.assertEqual(ar.record[role],br.record[role]); self.assertEqual(ar.record[role],ref.record[role]); checks+=1
                result=d.finish(); self.assertEqual(result,dd.finish()); self.assertEqual(result,s.finish(receipts))
                self.assertEqual(d.owner.stats['live_bytes'],0)
                self.differentials.append(dict(case=name,profile=profile,candidate_checks=checks,result=result,
                                               scoped_work=dict(w.counts),dense_work=dict(dw.counts),owner=dict(d.owner.stats)))

    def test_opt_in_and_unchanged_cap_preflight(self):
        root,source,raw,_=self.cases['half']
        for fn in (lambda:build(root,source,work=work()),lambda:CheckedDirectForest(raw,root,source,expected=digest(raw),work=work()),
                   lambda:check_package(root,expected=source,work=work()),lambda:execute(root,root,source,candidates,REPO,work=work())):
            with self.assertRaisesRegex(ValueError,'opt-in'): fn()
        for k,v in ((2,POLICY['rows']+1),(3,POLICY['nnz']+1),(4,65537),(6,POLICY['rows']+1),(7,POLICY['factors']+1)):
            fields=list(PREFIX.unpack_from(raw)); fields[k]=v; self.reject(PREFIX.pack(*fields)+raw[PREFIX.size:],'half')
        self.assertEqual(POLICY['visits'],8000000); self.assertEqual(POLICY['rows'],65536)

    def test_rehashed_semantic_header_mutations(self):
        mutations=(lambda m:m['roles']['pairs'].pop(),lambda m:m['factors'][0].__setitem__(0,'wrong'),
                   lambda m:m['factors'][-1].__setitem__(2,-2),lambda m:m['views'][-1].__setitem__('scope',0),
                   lambda m:m.__setitem__('source','0'*64),lambda m:m.__setitem__('request','0'*64),
                   lambda m:m.__setitem__('cut',['expert0',0]),lambda m:m['policy'].__setitem__('visits',8000001),
                   lambda m:m['views'][0]['output'].__setitem__(0,1),lambda m:m['views'][0]['output'].__setitem__(1,1))
        for fn in mutations: self.reject(self.mutate_header(fn))
        def omit_relu(m):
            v=next(v for v in m['views'] if v['op']=='ReLU' and v['eq'][1]); v['eq'][1]-=1
        self.reject(self.mutate_header(omit_relu))

    def test_rehashed_binary_mutations(self):
        raw=self.cases['deep'][2]; _,h,n,l,r,q,u,s=PREFIX.unpack_from(raw)
        node=PREFIX.size+h; local=node+n*NODE.size; refs=local+l*ENTRY.size; roots=refs+r*ENTRY.size; uses=roots+q*ROOT.size; scales=uses+u*4
        changes=((node,'<I',999),(node+4,'<I',1),(local+4,'<d',2.),(refs+4,'<d',-2.),
                 (roots,'<d',1.),(roots+8,'<I',1),(uses,'<I',1),(scales,'<d',.75),
                 (scales,'<d',float('nan')),(scales,'<d',-0.))
        for offset,fmt,v in changes:
            bad=bytearray(raw); struct.pack_into(fmt,bad,offset,v); self.reject(bytes(bad))
        self.reject(raw[:-1]); self.reject(raw+b'X')

    def test_immutable_source_and_work_identity(self):
        f,w=self.forest(); rows=f.views[0]['output']
        for fn in (lambda:setattr(rows,'_ids',(0,)),lambda:setattr(rows._roots,'_values',()),
                   lambda:setattr(f,'nodes',()),lambda:setattr(f,'_root','/tmp')):
            with self.assertRaises(AttributeError): fn()
        with self.assertRaises(TypeError): f.views[0]['output']=()
        with self.assertRaises(TypeError): f.nodes[0][1][0]=(0,F(99))
        with self.assertRaises(ValueError): checked(f,work())
        class Fake(CheckedDirectForest): pass
        with self.assertRaises(ValueError): checked(object.__new__(Fake),w)
        root,source,raw,_=self.cases['deep']
        with self.assertRaises(ValueError): CheckedDirectForest(bytearray(raw),root,source,expected=digest(raw),work=w,enabled=True)

    def test_independent_check_never_calls_builder(self):
        raw=self.cases['half'][2]
        with patch('scoped_source.direct_source_build.Builder.finish',side_effect=AssertionError('producer forbidden')):
            f,w=self.forest('half'); self.assertTrue(f.complete)

    def test_terminal_source_pollution(self):
        root,source,raw,_=self.cases['half']; own=self.root/'polluted'; shutil.copytree(root,own)
        w=work(); f=CheckedDirectForest(raw,own,source,expected=digest(raw),work=w,enabled=True)
        c=contract(f); d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True)
        self.addCleanup(d.abort)
        for i in range(d.session.tile_count):
            t,_,token=d.prepare(); d.submit(token,candidates(f,t),lambda i,b:(digest(b),len(b)))
        p=next((own/'t').iterdir()); b=bytearray(p.read_bytes()); b[0]^=1; p.write_bytes(b)
        with self.assertRaises(ValueError): d.finish()
        self.assertTrue(d.closed); self.assertEqual(d.owner.stats['live_bytes'],0)

    def test_foreign_legacy_frame_rejected(self):
        old,ow=self.forest('half',True); oc=contract(old); od=Driver(old,oc,expected=identity(oc),work=ow)
        f,w=self.forest('half'); c=contract(f); d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True)
        self.addCleanup(od.abort); self.addCleanup(d.abort); captured=[]
        t,_,tok=od.prepare()
        od.submit(tok,candidates(old,t),lambda i,b:(captured.append(b) or digest(b),len(b)))
        _,_,token=d.prepare()
        with self.assertRaises(ValueError): d.receive(token,captured[0],lambda i,b:(digest(b),len(b)))
        self.assertTrue(d.closed)

    def test_full_connection_shared_budget_and_costs(self):
        for name in self.cases:
            root,r=self.package(name); self.assertEqual(r['status'],'COMPLETE',r)
            self.assertEqual(r['result'],r['replay']['result']); self.assertEqual(r['construction']['native_seconds'],0)
            self.assertEqual(r['archive']['logical_bytes'],sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
            self.assertEqual(r['owner']['live_bytes'],0); self.assertTrue(all(p['complete'] for p in r['phases']))
            self.assertLessEqual(sum(p['seconds'] for p in r['phases']),r['elapsed_seconds'])
            for k in r['work']: self.assertEqual(sum(p['work'][k] for p in r['phases']),r['work'][k])
            self.assertGreaterEqual(sum(r['work'].values()),sum(r['replay']['work'].values()))
            self.assertGreater(sum(next(p for p in r['phases'] if p['name']=='portable_recheck')['work'].values()),0)
            self.assertTrue(r['terminal_received'])
            self.executions.append(dict(case=name,**r))

    def test_portable_does_not_reset_work(self):
        from scoped_source import direct_execution as module
        original=module.check_package; w=work(); seen=[]
        def at_limit(*args,**kw):
            self.assertIs(kw['work'],w); seen.append(dict(w.counts))
            w.counts['row_visits']+=POLICY['visits']-sum(w.counts.values())
            return original(*args,**kw)
        with patch.object(module,'check_package',side_effect=at_limit): _,r=self.package(work=w)
        self.assertEqual(r['status'],'ERROR'); self.assertIn('work capacity',r['error']); self.assertTrue(seen)

    def test_preconsumed_work_and_nonbinary_reference_refusal(self):
        w=work(); w.visit('row_visits',123); _,r=self.package(work=w)
        self.assertEqual(r['status'],'COMPLETE',r); self.assertEqual(r['initial_work']['row_visits'],123)
        for k in r['work']: self.assertEqual(r['initial_work'][k]+sum(p['work'][k] for p in r['phases']),r['work'][k])
        from scoped_source.direct_source_format import number
        for q in (F(1,3),F(2**53+1,2**53)):
            with self.assertRaisesRegex(ValueError,'lossless'): number(q)

    def test_missing_candidates_not_missing_obligations(self):
        _,r=self.package('guarded',lambda f,t:candidates(f,t,missing=True))
        self.assertEqual(r['status'],'COMPLETE',r); self.assertEqual(r['result']['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(r['result']['checked_endpoints'],0)

    def test_manifest_mutations_rehashed(self):
        root,r=self.package('guarded'); self.assertEqual(r['status'],'COMPLETE',r)
        pkg=root/'package'; m=decode((pkg/'manifest.json').read_bytes())
        mutations=(lambda x:x['tiles'].pop(),lambda x:x['tiles'].reverse(),lambda x:x['tiles'].__setitem__(1,x['tiles'][0]),
                   lambda x:x.__setitem__('check_width',1),lambda x:x.__setitem__('candidate_width',1),
                   lambda x:x['source'].__setitem__('sha256','0'*64),lambda x:x['contract'].__setitem__('sha256','0'*64),
                   lambda x:x['code'].pop())
        for fn in mutations:
            x=copy.deepcopy(m); fn(x); raw=compact(x)
            with self.assertRaises((ValueError,KeyError,IndexError)):
                check_manifest(pkg,raw,expected=digest(raw),work=work(),enabled=True)
        c=decode((pkg/'contract.json').read_bytes()); c['label']=(c['label']+1)%c['classes']; cr=compact(c)
        (pkg/'contract.json').write_bytes(cr); m['contract'].update(bytes=len(cr),sha256=digest(cr)); raw=compact(m)
        with self.assertRaises(ValueError): check_manifest(pkg,raw,expected=digest(raw),work=work(),enabled=True)

    def test_later_block_exception_preserves_only_prefix(self):
        original=Session._consume_block
        def fail(*args):
            if args[5]==2: raise RuntimeError('controlled second block')
            return original(*args)
        with patch.object(Session,'_consume_block',new=fail): root,r=self.package('guarded')
        self.assertEqual(r['status'],'ERROR'); self.assertIn('second block',r['error'])
        self.assertGreater(r['archive']['logical_bytes'],0); self.assertFalse((root/'terminal.json').exists())
        self.assertEqual(r['owner']['live_bytes'],0)

    def test_deadlines_across_online_phases(self):
        from scoped_source import direct_execution as module
        for target in ('build','propose','recheck','receive'):
            w=work()
            def expire(*a,**kw): w.deadline=time.monotonic()-1; w.tick()
            if target=='propose': root,r=self.package(propose=expire,work=w)
            else:
                obj,name=({'build':('scoped_source.direct_source_build','build'),
                           'recheck':('scoped_source.direct_execution','check_package'),
                           'receive':('scoped_source.direct_execution','receive_file')}[target])
                with patch(obj+'.'+name,side_effect=expire): root,r=self.package(work=w)
            self.assertEqual(r['status'],'TIMEOUT',r); self.assertGreater(r['archive']['logical_bytes'],0)
            self.assertTrue(any(not p['complete'] for p in r['phases']))
            if r['owner']: self.assertEqual(r['owner']['live_bytes'],0)
            self.assertFalse(r['terminal_received'])
            if (root/'terminal.json').exists(): self.assertEqual(decode((root/'terminal.json').read_bytes())['status'],'PENDING_RECEPTION')

    def test_published_exception_not_success(self):
        original=Archive.put
        def fail(a,name,raw,**kw):
            result=original(a,name,raw,**kw)
            if name.startswith('package/tiles/'): raise OSError('controlled after publication')
            return result
        with patch.object(Archive,'put',new=fail): root,r=self.package('half')
        self.assertEqual(r['status'],'ERROR'); self.assertTrue(list((root/'package/tiles').iterdir()))
        self.assertFalse((root/'terminal.json').exists()); self.assertEqual(r['owner']['live_bytes'],0)

    def test_cleanup_exception_keeps_failure_and_cost(self):
        original=Driver.abort
        def cleanup(d):
            original(d); raise RuntimeError('controlled release error')
        def fail(f,t): raise OSError('primary candidate error')
        with patch.object(Driver,'abort',new=cleanup): root,r=self.package(propose=fail)
        self.assertEqual(r['status'],'ERROR'); self.assertIn('release error',r['error'])
        self.assertIn('candidate error',r['primary_error']); self.assertFalse(r['terminal_received'])
        self.assertFalse(r['phases'][-1]['complete']); self.assertEqual(r['owner']['live_bytes'],0)
        self.assertGreater(r['archive']['logical_bytes'],0)

    def test_cooperative_callback_overrun_not_accepted(self):
        w=work()
        def late(f,t):
            r=candidates(f,t); w.deadline=time.monotonic()-1; return r
        root,r=self.package(propose=late,work=w)
        self.assertEqual(r['status'],'TIMEOUT'); self.assertFalse(r['terminal_received'])
        self.assertFalse((root/'terminal.json').exists()); self.assertGreaterEqual(r['cleanup_after_deadline'],0)

    def test_package_extra_symlink_changed_or_missing(self):
        root,r=self.package(); self.assertEqual(r['status'],'COMPLETE',r)
        for mode in ('extra','symlink','changed','missing'):
            dest=self.root/('bad-'+mode); shutil.copytree(root/'package',dest)
            if mode=='extra': (dest/'unlisted').write_bytes(b'x')
            elif mode=='symlink': (dest/'unlisted').symlink_to(root/'package/forest.bin')
            elif mode=='changed': (dest/'tiles/0000.bin').write_bytes(b'x')
            else: (dest/'tiles/0000.bin').unlink()
            with self.assertRaises(ValueError): check_package(dest,expected=r['package']['sha256'],work=work(),enabled=True)

    def test_relocated_stdlib_without_repository_or_producer(self):
        root,r=self.package(); self.assertEqual(r['status'],'COMPLETE',r)
        # Separately budgeted offline audit, including copies, logs and reception.
        offline=Archive(self.root/'offline-relocation',work()); before=time.monotonic()
        for p in sorted((root/'package').rglob('*')):
            if p.is_file(): offline.copy(p,'moved/'+str(p.relative_to(root/'package')))
        copy_seconds=time.monotonic()-before; destination=offline.root/'moved'
        program="""import sys,json,time
from pathlib import Path
root=Path(sys.argv[1]); forbidden=[Path(p) for p in sys.argv[4:]]
sys.path.insert(0,str(root/'code'))
def deny(event,args):
    if event=='open' and isinstance(args[0],str):
        path=Path(args[0]).resolve()
        if any(path.is_relative_to(p) for p in forbidden): raise PermissionError('original unavailable')
sys.addaudithook(deny)
try: (forbidden[0]/'source_dag.py').read_bytes()
except PermissionError: pass
else: raise AssertionError('read isolation missing')
from scoped_source.hz_request_store import Work
from scoped_source.direct_package import check_package
r=check_package(root,expected=sys.argv[2],work=Work(float(sys.argv[3])),enabled=True)
assert not any(n in sys.modules for n in ('torch','numpy','scipy','scoped_source.source_dag','scoped_source.direct_source_build','scoped_source.scaled_references'))
print(json.dumps(r,sort_keys=True))
"""
        child=run_child([sys.executable,'-I','-S','-B','-c',program,str(destination),r['package']['sha256'],str(offline.work.deadline),str(REPO),str(root),str(self.cases['half'][0])],offline,log_prefix='logs',cwd='/')
        self.assertEqual(child['status'],'COMPLETE',child)
        logs=b''.join((offline.root/f'logs/{i:04d}.log').read_bytes() for i in range(child['log_chunks']))
        replay=json.loads(logs); self.assertEqual(replay['result'],r['result']); offline.audit()
        self.executions.append(dict(case='offline-relocation',copy_seconds=copy_seconds,child=child,archive=offline.ledger(),replay=replay))


if __name__=='__main__': unittest.main()
