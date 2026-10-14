"""Frozen finite interface controls. Stdlib only; no optimizer or real model."""
import copy
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Work,POLICY,compact,decode,digest,identity
from scoped_source.scaled_endpoint_controls import fixture,scalar_bound,expanded
from scoped_source.scaled_references import build
from scoped_source.check_scaled_references import CheckedScaledForest,checked
from scoped_source.scaled_endpoints import Session,contract
from scoped_source.wire_forest import encode,CheckedWireForest,intake,NodeTable,PREFIX,NODE,ENTRY,float_exact,unpack_float
from scoped_source.wire_proof import encode_tile,decode_tile,ProofHash,check_manifest,MANIFEST
from scoped_source.wire_archive import Archive,run_child,receive,receive_file
from scoped_source.wire_package import bundle,check_package


def work(): return Work(time.monotonic()+300)


def candidates(f,tasks,missing=False):
    out=[]
    for task in tasks:
        if missing: out.append(dict(slot=task.slot,proposed=None,zero=None)); continue
        eq=len(f.rows(task.pair,'eq')); ine=len(f.rows(task.pair,'ine'))
        proposed=dict(query=task.query_id,equality=[str(F((j%3)-1,8)) for j in range(eq)],
                      inequality=[str(F(-(j%2),16)) for j in range(ine)],lower='0')
        zero=dict(query=task.query_id,equality=['0']*eq,inequality=['0']*ine,lower='0')
        out.append(dict(slot=task.slot,proposed=proposed,zero=zero))
    return out


CODE_FILES=tuple('scoped_source/'+n+'.py' for n in (
    'wire_forest','wire_proof','wire_archive','wire_package','check_source_dag','check_affine_references',
    'check_scaled_references','forest_endpoints','scaled_endpoints','bounded_runtime','pair_workspace','tensor_rows',
    'factored_io','factored_source','hz_request_store'))+('source_enclosure/format.py','upstream_source/checker.py')


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='hz-wire-tests-',dir='/data1/Kane/MOE')
        cls.root=Path(cls.tmp.name); cls.cases={}
        for name in ('mixed','deep','half','guarded'):
            root,source=fixture(cls.root/name,name); w=work(); raw,_=build(root,source,work=w)
            wr=encode(raw,root,source,expected=digest(raw),work=w)
            cls.cases[name]=dict(root=root,source=source,raw=raw,wire=wr)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def forest(self,name='deep',old=False):
        v=self.cases[name]; w=work(); raw=v['raw' if old else 'wire']; cls=CheckedScaledForest if old else CheckedWireForest
        return cls(raw,v['root'],v['source'],expected=digest(raw),work=w),w

    def reject_wire(self,raw,name='deep'):
        v=self.cases[name]
        with self.assertRaises((ValueError,IndexError,TypeError,KeyError)):
            CheckedWireForest(raw,v['root'],v['source'],expected=digest(raw),work=work())

    def altered_old(self,mutate,name='deep'):
        v=self.cases[name]; d=decode(v['raw']); mutate(d); raw=compact(d)
        with self.assertRaises((ValueError,IndexError,TypeError,KeyError)):
            wr=encode(raw,v['root'],v['source'],expected=digest(raw),work=work())
            CheckedWireForest(wr,v['root'],v['source'],expected=digest(wr),work=work())

    def test_complete_same_candidate_exact_differential(self):
        for name in self.cases:
            f,w=self.forest(name); old,ow=self.forest(name,True)
            self.assertEqual(f.identity,old.identity); self.assertEqual(f.scales,old.scales)
            self.assertEqual(tuple(f.nodes),old.nodes); self.assertEqual(f.roles,old.roles)
            c=contract(f); s=Session(f,c,expected=identity(c),work=w,retain_raw=False)
            osess=Session(old,c,expected=identity(c),work=ow); receipts=[]; oldreceipts=[]; ph=ProofHash(identity(c),8,w)
            matrix=expanded(old,ow)
            for i in range(s.tile_count):
                tasks,_=s.prepare(i); oldtasks,_=osess.prepare(i); self.assertEqual(tasks,oldtasks)
                records=candidates(f,tasks); raw=encode_tile(records,f,tasks,w); logical=decode_tile(raw,f,tasks,w)
                self.assertEqual(logical,compact(records)); ph.tile(logical)
                actual=s.accept(logical); ref=osess.accept(records)
                for task,rec,a,b in zip(tasks,records,actual,ref,strict=True):
                    self.assertEqual(a.record,b.record)
                    for role in ('proposed','zero'):
                        bound,_=scalar_bound(old,task,rec[role],matrix,ow)
                        self.assertEqual(bound,F(a.record[role]))
                receipts.extend(actual); oldreceipts.extend(ref)
            self.assertEqual(s.finish(receipts),osess.finish(oldreceipts))
            oldproof=osess.proof(); h,size=ph.finish()
            self.assertEqual((h,size),(identity(oldproof),len(compact(oldproof))))
            self.assertEqual(s._Session__raw,[])
            with self.assertRaisesRegex(ValueError,'streaming'): s.proof()

    def test_semantic_mutations_rehashed(self):
        self.altered_old(lambda d:d['roles']['pairs'].pop())
        self.altered_old(lambda d:d['factors'][0].__setitem__(0,'wrong-factor'))
        self.altered_old(lambda d:d['scales'].__setitem__(0,'1/2'))
        self.altered_old(lambda d:d['nodes'][0][1][0].__setitem__(1,'2'))
        self.altered_old(lambda d:d['views'][0]['output'][0].__setitem__(0,'1'))
        self.altered_old(lambda d:d['nodes'][d['views'][d['roles']['pairs'][0][2]]['ine'][0][1]][2][0].__setitem__(1,'-2'))
        def relu(d):
            v=next(v for v in d['views'] if v['op']=='ReLU' and v['eq'])
            v['eq'].pop()
        self.altered_old(relu)
        self.altered_old(lambda d:d['views'][-1].__setitem__('scope',0))

    def test_offsets_padding_nonfinite_and_source_binding(self):
        v=self.cases['deep']; raw=v['wire']; _,hs,nc,rc,fc=PREFIX.unpack_from(raw)
        base=PREFIX.size+hs+fc*8+rc*16; payload=base+nc*NODE.size
        for field,value in ((8,1),(3,2),(0,999),(1,POLICY['nnz']+1),(2,65537),(7,POLICY['nnz']+1)):
            buf=bytearray(raw); d=list(NODE.unpack_from(buf,base)); d[field]=value
            NODE.pack_into(buf,base,*d); self.reject_wire(bytes(buf))
        for index in (base+NODE.size-1,payload+ENTRY.size-1):
            buf=bytearray(raw); buf[index]=1; self.reject_wire(bytes(buf))
        for value in (math.nan,math.inf,-0.):
            buf=bytearray(raw); struct.pack_into('<d',buf,PREFIX.size+hs,value); self.reject_wire(bytes(buf))
        for ni in range(nc):
            desc=list(NODE.unpack_from(raw,base+ni*NODE.size))
            if desc[3]:
                for k,new in ((4,desc[4]+1),(5,desc[5]+1),(6,desc[6]+1)):
                    buf=bytearray(raw); changed=list(desc); changed[k]=new
                    NODE.pack_into(buf,base+ni*NODE.size,*changed); self.reject_wire(bytes(buf))
                break
        self.reject_wire(raw[:-1]); self.reject_wire(raw+b'X')
        buf=bytearray(raw); PREFIX.pack_into(buf,0,b'HZWF0001',hs,nc,rc,32769); self.reject_wire(bytes(buf))

    def test_scalar_losslessness_and_unsupported_rational(self):
        for value in (0.,-0.,math.nextafter(0.,1.),float.fromhex('0x1.fffffffffffffp+1023'),-1.5):
            self.assertEqual(F(float_exact(F(value))),F(value))
        for value in (F(1,3),F(1,2**1200)):
            with self.assertRaises(ValueError): float_exact(value)
        with self.assertRaises(ValueError): unpack_float(-0.)

    def test_no_full_coefficient_or_proof_bank_alias_and_work(self):
        f,w=self.forest(); self.assertIs(type(f.nodes),NodeTable)
        with self.assertRaises(AttributeError): f.nodes.raw=b'corrupt'
        with self.assertRaises(AttributeError): f.scales=()
        with self.assertRaises(ValueError): checked(f,work())
        class Forged(type(f)): pass
        fake=object.__new__(Forged)
        with self.assertRaises(ValueError): checked(fake,w)
        self.assertFalse(any(type(getattr(f.nodes,n)) is list for n in NodeTable.__slots__))
        a=f.nodes.document_row(0); a[1][0][1]='123'; self.assertNotEqual(a,f.nodes.document_row(0))

    def test_source_pollution_after_check(self):
        v=self.cases['deep']; own=self.root/'mutated-source'; shutil.copytree(v['root'],own)
        f=CheckedWireForest(v['wire'],own,v['source'],expected=digest(v['wire']),work=work())
        ni=next(i for i in range(len(f.nodes)) if f.nodes.descriptor(i)[3])
        t=f.nodes.source.tensors[f.nodes.descriptor(ni)[4]]; path=own/t.chunks[0].file
        raw=bytearray(path.read_bytes()); raw[0]^=1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'chunk identity'): f.nodes[ni]

    def test_terminal_source_mutation_manifest_and_unaccessed_chunk(self):
        v=self.cases['half']
        for mode in ('manifest','tensor'):
            own=self.root/('terminal-pollution-'+mode); shutil.copytree(v['root'],own)
            w=work(); f=CheckedWireForest(v['wire'],own,v['source'],expected=digest(v['wire']),work=w)
            c=contract(f); s=Session(f,c,expected=identity(c),work=w,retain_raw=False); receipts=[]
            for i in range(s.tile_count):
                tasks,_=s.prepare(i); receipts.extend(s.accept(candidates(f,tasks)))
            p=own/'manifest.json' if mode=='manifest' else own/f.source_access.tensors[-1].chunks[-1].file
            raw=bytearray(p.read_bytes()); raw[-1]^=1; p.write_bytes(raw)
            with self.assertRaises(ValueError): s.finish(receipts)

    def test_midread_source_change_rejected_not_mixed_identity(self):
        v=self.cases['deep']; own=self.root/'midread-pollution'; shutil.copytree(v['root'],own)
        from scoped_source import wire_forest
        read=wire_forest.read; changed=False
        def pollute(root,name,*args):
            nonlocal changed
            data=read(root,name,*args)
            if name.startswith('t/') and not changed:
                changed=True; p=Path(root)/name; raw=bytearray(data); raw[0]^=1; p.write_bytes(raw)
            return data
        with patch.object(wire_forest,'read',side_effect=pollute):
            with self.assertRaises(ValueError): CheckedWireForest(v['wire'],own,v['source'],expected=digest(v['wire']),work=work())

    def test_inline_capacity_rejected_before_row_materialization(self):
        v=self.cases['deep']; raw=bytearray(v['wire']); _,hs,nc,rc,fc=PREFIX.unpack_from(raw)
        base=PREFIX.size+hs+fc*8+rc*16; desc=list(NODE.unpack_from(raw,base)); desc[1]=desc[7]=16385
        NODE.pack_into(raw,base,*desc)
        with patch.object(NodeTable,'document_row',side_effect=AssertionError('must reject before allocation')):
            self.reject_wire(bytes(raw))
        # A tiny physical candidate descriptor cannot authorize arbitrary sizes.
        f,w=self.forest('half'); c=contract(f); s=Session(f,c,expected=identity(c),work=w)
        tasks,_=s.prepare(0); raw=encode_tile(candidates(f,tasks),f,tasks,w)
        from scoped_source.wire_proof import PREFIX as TP
        magic,size=TP.unpack_from(raw); h=decode(raw[TP.size:TP.size+size]); h['counts']['eq']=10**9
        hdr=compact(h); mutated=TP.pack(magic,len(hdr))+hdr+raw[TP.size+size:]
        with self.assertRaises(ValueError): decode_tile(mutated,f,tasks,w)

    def test_budget_exhaustion_during_row_and_terminal_seal(self):
        f,w=self.forest('deep'); w.counts['row_visits']=POLICY['visits']
        with self.assertRaisesRegex(ValueError,'work capacity'): f.nodes[0]
        f,w=self.forest('half'); c=contract(f); s=Session(f,c,expected=identity(c),work=w); receipts=[]
        for i in range(s.tile_count):
            tasks,_=s.prepare(i); receipts.extend(s.accept(candidates(f,tasks)))
        with patch.object(f.source_access,'seal',side_effect=TimeoutError('controlled seal cutoff')):
            with self.assertRaises(TimeoutError): s.finish(receipts)

    def test_both_roles_property_binding_partial_and_missing(self):
        f,w=self.forest('guarded'); c=contract(f); s=Session(f,c,expected=identity(c),work=w,retain_raw=False)
        tasks,_=s.prepare(0); records=candidates(f,tasks)
        for mutation in (lambda r:r[0].__setitem__('zero',None),
                         lambda r:r[0].__setitem__('slot','0'*64),
                         lambda r:r[0]['zero']['inequality'].__setitem__(0,'-1')):
            bad=copy.deepcopy(records); mutation(bad)
            with self.assertRaises(ValueError): encode_tile(bad,f,tasks,w)
        raw=encode_tile(records,f,tasks,w)
        for bad in (raw[:-1],raw+b'X'):
            with self.assertRaises(ValueError): decode_tile(bad,f,tasks,w)
        receipts=[]
        receipts.extend(s.accept(decode_tile(encode_tile(candidates(f,tasks,True),f,tasks,w),f,tasks,w)))
        for i in range(1,s.tile_count):
            tasks,_=s.prepare(i); receipts.extend(s.accept(candidates(f,tasks)))
        self.assertEqual(s.finish(receipts)['status'],'UNKNOWN_MISSING_EVIDENCE')
        c['label']=(c['label']+1)%c['classes']
        with self.assertRaises(ValueError): Session(f,c,expected=identity(c),work=w)

    def package(self,name='half'):
        v=self.cases[name]; w=work(); a=Archive(self.root/('package-'+name),w)
        for p in sorted(v['root'].rglob('*')):
            if p.is_file(): a.copy(p,'source/'+str(p.relative_to(v['root'])))
        fref=a.put('forest.bin',v['wire']); f=CheckedWireForest(v['wire'],a.root/'source',v['source'],expected=fref['sha256'],work=w)
        c=contract(f); cref=a.put('contract.json',compact(c)); s=Session(f,c,expected=identity(c),work=w,retain_raw=False)
        ph=ProofHash(identity(c),8,w); refs=[]; receipts=[]
        for i in range(s.tile_count):
            tasks,_=s.prepare(i); raw=encode_tile(candidates(f,tasks),f,tasks,w)
            refs.append(a.put(f'tiles/{i:04d}.bin',raw))  # Immutable before acceptance.
            logical=decode_tile(raw,f,tasks,w); ph.tile(logical); receipts.extend(s.accept(logical))
        result=s.finish(receipts); logical,size=ph.finish()
        m=dict(schema=MANIFEST,source=dict(directory='source',sha256=v['source']),forest=fref,
               contract=cref,width=8,tiles=refs,logical_proof=logical,logical_bytes=size)
        a.put('manifest.json',compact(m)); return a,m,result

    def test_portable_relocation_no_repository_no_producer(self):
        a,m,result=self.package(); repo=Path(__file__).resolve().parents[1]
        for name in CODE_FILES: a.copy(repo/name,'code/'+name)
        br=bundle(a)
        original=tuple(a.entries); a.relocate('relocated',original)
        # Original copy retained and charged; every attempted read of its tree
        # or the repository is denied by a process audit hook, not merely omitted.
        code="""import sys,time,json
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root/'code'))
def deny_original(event,args):
    if event=='open' and isinstance(args[0],str):
        path=Path(args[0]).resolve()
        if (path.is_relative_to(root.parent.parent) and not path.is_relative_to(root)) or path.is_relative_to(Path(sys.argv[3])):
            raise PermissionError('original source/repository unavailable')
sys.addaudithook(deny_original)
try: (root.parent/'manifest.json').read_bytes()
except PermissionError: pass
else: raise AssertionError('original read not blocked')
from scoped_source.hz_request_store import Work,digest,decode
from scoped_source.wire_package import check_package
r=check_package(root,expected=sys.argv[2],work=Work(float(sys.argv[4])))
assert not any(n in sys.modules for n in ('torch','numpy','scipy','scoped_source.scaled_references'))
print(json.dumps(r['result'],sort_keys=True))
"""
        child=run_child([sys.executable,'-I','-S','-B','-c',code,str(a.root/'relocated'),br['sha256'],str(repo),str(a.work.deadline)],a,log_prefix='check-log',cwd='/')
        self.assertEqual(child['status'],'COMPLETE',child)
        logs=b''.join((a.root/f'check-log/{i:04d}.log').read_bytes() for i in range(child['log_chunks']))
        self.assertEqual(json.loads(logs),result)
        self.assertEqual(a.bytes,sum(p.stat().st_size for p in a.root.rglob('*') if p.is_file()))
        a.audit()
        terminal=compact(dict(status='CONTROL_COMPLETE',bytes_before_terminal=a.bytes))
        ref=a.put('terminal.json',terminal,terminal=True)
        self.assertEqual(receive_file(a,ref)['status'],'CONTROL_COMPLETE')

    def test_durable_terminal_changed_after_publication(self):
        for name,payload in (('truncated',b''),('replaced',b'{"status":"OTHER"}')):
            a=Archive(self.root/('terminal-'+name),work()); ref=a.put('terminal.json',compact(dict(status='CONTROL_COMPLETE')),terminal=True)
            (a.root/'terminal.json').write_bytes(payload)
            with self.assertRaises(ValueError): receive_file(a,ref)

    def test_manifest_completeness(self):
        a,m,result=self.package('guarded')
        self.assertEqual(check_manifest(a.root,compact(m),expected=identity(m),work=work()),result)
        mutations=(lambda x:x['tiles'].pop(),lambda x:x['tiles'].append(x['tiles'][0]),
                   lambda x:x['tiles'].__setitem__(1,x['tiles'][0]),
                   lambda x:x.__setitem__('logical_proof','0'*64),
                   lambda x:x['source'].__setitem__('directory','../other'))
        for mutate in mutations:
            bad=copy.deepcopy(m); mutate(bad)
            with self.assertRaises(ValueError): check_manifest(a.root,compact(bad),expected=identity(bad),work=work())

    def test_package_inventory_tamper_extra_missing_and_symlink(self):
        a,m,result=self.package('deep'); ref=bundle(a)
        self.assertEqual(check_package(a.root,expected=ref['sha256'],work=work())['result'],result)
        (a.root/'extra').write_bytes(b'x')
        with self.assertRaises(ValueError): check_package(a.root,expected=ref['sha256'],work=work())
        with self.assertRaises(ValueError): a.audit()
        (a.root/'extra').unlink()
        p=a.root/'contract.json'; old=p.read_bytes(); p.write_bytes(old+b' ')
        with self.assertRaises(ValueError): check_package(a.root,expected=ref['sha256'],work=work())
        with self.assertRaises(ValueError): a.audit()
        p.write_bytes(old); moved=a.root/'outside'; p.rename(moved); p.symlink_to(moved)
        with self.assertRaises(ValueError): check_package(a.root,expected=ref['sha256'],work=work())

    def test_archive_limits_overwrite_alias_and_copy(self):
        a=Archive(self.root/'small',work(),limit=1000,reserve=100); a.put('data',b'x'*451)
        with self.assertRaises(ValueError): a.relocate('again',('data',))
        with self.assertRaises(ValueError): a.put('data',b'x')
        with self.assertRaises(ValueError): a.put('../escape',b'x')
        (a.root/'link').symlink_to(self.root)
        with self.assertRaises(ValueError): a.put('link/escape',b'x')
        with self.assertRaises(ValueError): a.put('large',b'x'*450)
        self.assertEqual(a.bytes,451)
        a.put('terminal',b'ok',terminal=True)
        with self.assertRaises(ValueError): a.put('late',b'x')

    def test_deadline_partial_postwrite_and_exception(self):
        w=work(); a=Archive(self.root/'deadline',w); calls=0
        original=w.tick
        def tick():
            nonlocal calls
            calls+=1
            if calls==4: raise TimeoutError('controlled mid write')
            original()
        with patch.object(w,'tick',side_effect=tick):
            with self.assertRaises(TimeoutError): a.put('partial',b'x'*70000)
        self.assertEqual(a.bytes,65536); self.assertFalse(a.entries); self.assertEqual((a.root/'partial').stat().st_size,65536)
        self.assertFalse(a.ledger()['events'][0]['complete'])
        # Fail after write/fsync: existing bytes do not constitute reception.
        def late_hash(raw): raise TimeoutError('controlled post-write hash')
        with patch('scoped_source.wire_archive.digest',side_effect=late_hash):
            with self.assertRaises(TimeoutError): a.put('late-terminal',b'ok',terminal=True)
        self.assertNotIn('late-terminal',a.entries); self.assertEqual(a.bytes,65538)
        w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): receive(b'{}',digest(b'{}'),w)
        with self.assertRaises(TimeoutError): a.put('never',b'')
        v=self.cases['half']
        with self.assertRaises(TimeoutError): CheckedWireForest(v['wire'],v['root'],v['source'],expected=digest(v['wire']),work=w)

    def test_partial_prefix_hash_audit_and_many_records(self):
        a=Archive(self.root/'record-cap',work(),limit=1000,reserve=100)
        for i in range(9): a.put(str(i),b'x'*100)
        with self.assertRaises(ValueError): a.put('overflow',b'x')
        self.assertEqual(a.audit()['logical_bytes'],900)
        p=a.root/'0'; p.write_bytes(b'y'*100)
        with self.assertRaises(ValueError): a.audit()

    def test_archive_nonregular_and_zero_prefix_removal(self):
        a=Archive(self.root/'nonregular',work()); a.put('empty',b''); p=a.root/'empty'; p.unlink(); os.mkfifo(p)
        with self.assertRaises(ValueError): a.audit()
        b=Archive(self.root/'zero-prefix',work())
        with patch('scoped_source.wire_archive.os.fsync',side_effect=OSError('control fsync error')):
            with self.assertRaises(OSError): b.put('failed-empty',b'')
        self.assertTrue(b.prefixes['failed-empty']['created']); b.audit()
        (b.root/'failed-empty').unlink()
        with self.assertRaises(ValueError): b.audit()

    def test_after_cleanup_deadline_cannot_report_complete(self):
        a=Archive(self.root/'cleanup-timeout',work())
        import scoped_source.wire_archive as module
        original=module.os.killpg
        def cleanup(pid,sig):
            a.work.deadline=time.monotonic()-1
            return original(pid,sig)
        with patch.object(module.os,'killpg',side_effect=cleanup):
            result=run_child([sys.executable,'-I','-S','-c','pass'],a,log_prefix='cleanup')
        self.assertEqual(result['status'],'TIMEOUT')

    def test_owned_child_log_cap_exception_deadline(self):
        a=Archive(self.root/'childlog',work())
        r=run_child([sys.executable,'-I','-S','-c','import os; os.write(1,b"x"*4096)'],a,log_prefix='out',log_limit=512)
        self.assertEqual(r['status'],'ERROR'); self.assertLessEqual(r['log_bytes_received'],513)
        r=run_child([sys.executable,'-I','-S','-c','raise RuntimeError("control")'],a,log_prefix='error')
        self.assertEqual(r['status'],'ERROR'); self.assertGreater(r['log_chunks'],0)
        a.work.deadline=time.monotonic()+.15
        r=run_child([sys.executable,'-I','-S','-c','import time; print("prefix",flush=True); time.sleep(10)'],a,log_prefix='deadline')
        self.assertEqual(r['status'],'TIMEOUT'); self.assertGreater(r['log_chunks'],0)
        self.assertLess(r['seconds'],3)


if __name__=='__main__': unittest.main()
