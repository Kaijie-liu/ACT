"""Frozen joint-source controls, synthetic stdlib only, no native execution."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Work, POLICY, compact, decode, digest, identity
from scoped_source.scaled_endpoint_controls import fixture
from scoped_source.direct_source_build import build as legacy_build
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.shared_build import build
from scoped_source.shared_source import Snapshot, PackedRow, SourceLocal
from scoped_source.shared_format import Packet, PAGE, PREFIX, NODE, ENTRY, ROOT as ROOT_RECORD, Writer, Reader, load
from scoped_source.check_shared_source import CheckedSharedForest
from scoped_source.shared_archive import capture_owned, transfer, OwnedSource
from scoped_source.wire_archive import Archive
from scoped_source.direct_method import method, validate
from scoped_source.direct_candidates import Driver
from scoped_source.direct_lifecycle import Lifecycle, prepare, revoke
from scoped_source.scaled_endpoints import contract
from scoped_source.direct_supervised import supervise, spec, audit, census, ROOT, PYTHON
from scoped_source.direct_lifecycle_package import check_package


def work(): return Work(time.monotonic()+300)


def altered(packet, edit):
    # Only tiny fixed controls join bytes. Production is separately checked paged.
    raw=b''.join(packet.pages); fields=list(PREFIX.unpack_from(raw)); h=fields[1]
    m=decode(raw[PREFIX.size:PREFIX.size+h]); before=compact(m); edit(m)
    assert compact(m)!=before, 'mutation must actually change metadata'
    head=compact(m); fields[1]=len(head)
    raw=PREFIX.pack(*fields)+head+raw[PREFIX.size+h:]
    return Packet(tuple(raw[i:i+PAGE] for i in range(0,len(raw),PAGE)),work())


class Controls(unittest.TestCase):
    differentials=[]
    supervised=[]
    equations=[]

    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='hz-shared-controls-',dir='/data1/Kane/MOE')
        cls.root=Path(cls.temp.name); cls.cases={}
        for case in ('mixed','deep','half','guarded'):
            r,s=fixture(cls.root/case,case)
            old,_=legacy_build(r,s,work=work(),enabled=True)
            new,cost=build(r,s,work=work(),enabled=True)
            cls.cases[case]=(r,s,old,new,cost)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def path(self,label): return self.root/(label+str(time.monotonic_ns()))

    def forest(self,case='half',shared=True,packet=None,w=None):
        r,s,old,new,_=self.cases[case]; w=w or work()
        return (CheckedSharedForest(packet or new,r,s,expected=(packet or new).identity,work=w,enabled=True)
                if shared else CheckedDirectForest(old,r,s,expected=digest(old),work=w,enabled=True))

    def run_control(self,case='half',arm='pair_scoped',profile=0,fault='',seconds=30,initial=None):
        root=self.path('request-')
        result=supervise(root,spec(case,arm,profile,fault,shared=True),enabled=True,seconds=seconds,initial_work=initial)
        checked=audit(root,result)
        self.supervised.append(dict(case=case,arm=arm,profile=profile,fault=fault,status=result['status'],
            seconds=result['seconds'],retained_bytes=result['storage']['bytes'],work=result['work'],audit=checked,
            accepted=result['accepted']))
        return root,result

    def test_default_off_and_binding(self):
        r,s,_,p,_=self.cases['half']
        for fn in (lambda:build(r,s,work=work()),
                   lambda:CheckedSharedForest(p,r,s,expected=p.identity,work=work()),
                   lambda:supervise(self.path('off-'),spec('half','dense',shared=True))):
            with self.assertRaisesRegex(ValueError,'opt-in'): fn()
        for key in ('protocol','schema'):
            m=method('dense',shared=True); m[key]='invalid'
            with self.assertRaises(ValueError): validate(m)
        self.assertEqual(POLICY['visits'],8000000); self.assertEqual(POLICY['rows'],65536)
        self.assertEqual(POLICY['total_bytes'],128*2**20)
        for shared in (False,True):
            f=self.forest(shared=shared); life=Lifecycle(method('dense',shared=not shared))
            try:
                with self.assertRaisesRegex(ValueError,'representation binding'): life.setup(f,f.session)
            finally: life.close()

    def test_equation_only_nonzero_scales_error_and_lift(self):
        # Explicit equation-only control required by the frozen coverage list;
        # NOT another endpoint/supervised source or performance observation.
        from scoped_source.affine_reference_controls import deep
        from scoped_source.shared_build import Builder
        from scoped_source.shared_format import encode
        r,s=deep(self.path('equation-only-'),center=(1.0,float.fromhex('0x1p-1000')),radius='1/4')
        w=work(); snap=Snapshot(r,s,w); b=Builder(snap,w); packet,_=b.finish(None)
        oldraw,_=legacy_build(r,s,work=work(),enabled=True)
        old=CheckedDirectForest(oldraw,r,s,expected=digest(oldraw),work=work(),enabled=True)
        new=CheckedSharedForest(packet,r,s,expected=packet.identity,work=work(),enabled=True)
        self.assertEqual(new.scales,old.scales); self.assertEqual(new.scales[:2],(F(1,8),F(1,4)))
        self.assertEqual(new.factors,old.factors); self.assertEqual(new.norms,old.norms)
        self.assertEqual([(a,tuple(local),refs) for a,local,refs in new.nodes],list(old.nodes))
        target=next(v for v in new.views if v['name']=='router/1')['output'][1][1]
        local=b.nodes[target][1]; self.assertIs(type(local),SourceLocal)
        self.assertEqual(tuple(q for _,q in local.tail),(F(1,2**55),F(1,2**1000)))
        self.assertEqual(tuple(new.factors[j][0] for j,_ in local.tail),
                         ('router/1/error/1','router/1/lift/output/1'))
        for k in (0,1):
            for action in ('delete','negate'):
                tail=list(local.tail)
                if action=='delete': tail.pop(k)
                else: tail[k]=(tail[k][0],-tail[k][1])
                b.nodes[target][1]=SourceLocal(local.row,tuple(tail),w)
                bad=encode(b,dict(new.roles),None)
                with self.assertRaises(ValueError):
                    CheckedSharedForest(bad,r,s,expected=bad.identity,work=work(),enabled=True)
        b.nodes[target][1]=local
        self.equations.append(dict(kind='EQUATION_ONLY_NOT_ENDPOINT_ROSTER',source=s,
            scales=[str(q) for q in new.scales[:2]],tail=[str(q) for _,q in local.tail],
            source_differential=True,tail_mutations_rejected=4))

    def test_reseal_and_non_affine_alias(self):
        r,s,_,p,_=self.cases['half']; target=self.path('source-reseal-'); shutil.copytree(r,target)
        f=CheckedSharedForest(p,target,s,expected=p.identity,work=work(),enabled=True)
        tensor=next((target/'t').iterdir()); raw=tensor.read_bytes()
        tensor.write_bytes(bytes([raw[0]^1])+raw[1:])
        with self.assertRaises(ValueError): f.seal()
        # Closed node-role check rejects even a numerically zero source alias
        # at a non-affine node (direct equation control, no accepted package).
        from scoped_source.check_shared_source import Checker
        c=Checker(p,r,s,p.identity,work())
        zero=next(n[1] for n in c.nodes if type(n[1]) is SourceLocal and not len(n[1]))
        c.nodes=((0,zero,()),)+c.nodes[1:]
        with self.assertRaisesRegex(ValueError,'only at first affine'): c.node(0,0)

    def test_all_source_equations_and_rows_match(self):
        for case in self.cases:
            old=self.forest(case,False); new=self.forest(case)
            self.assertEqual(old.factors,new.factors); self.assertEqual(old.scales,new.scales)
            self.assertEqual(old.norms,new.norms); self.assertEqual(old.roles,new.roles)
            self.assertEqual(old.factor_identity,new.factor_identity)
            self.assertEqual(len(old.nodes),len(new.nodes))
            for a,b in zip(old.nodes,new.nodes,strict=True):
                self.assertEqual(a,(b[0],tuple(b[1]),b[2]))
            for a,b in zip(old.views,new.views,strict=True):
                for kind in ('output','eq','ine'): self.assertEqual(tuple(a[kind]),tuple(b[kind]))
            stats=new.receipt['stats']
            self.assertLess(stats['stored_memberships'],stats['memberships'])
            self.assertEqual(stats['forest_bytes'],self.cases[case][3].size)
            self.assertEqual(stats['explicit_locals']+stats['aliased_locals'],stats['local_nnz'])

    def test_all_same_candidate_endpoints(self):
        for case in self.cases:
            for profile in (0,1):
                reference=None
                for arm in ('dense','pair_scoped'):
                    for shared in (False,True):
                        f=self.forest(case,shared); c=contract(f)
                        d=Driver(f,c,expected=identity(c),work=f.session,pair_scoped=arm=='pair_scoped')
                        life=Lifecycle(method(arm,profile,shared=shared)); values=[]
                        try:
                            life.setup(f,f.session)
                            for _ in range(d.session.tile_count):
                                token,records=life.propose(d)
                                receipts=d.submit(token,records,lambda i,b:(digest(b),len(b)),profile=profile)
                                values.extend((r.record['proposed'],r.record['zero'],r.record['lower']) for r in receipts)
                            result=d.finish()
                            if reference is None: reference=(values,result)
                            self.assertEqual((values,result),reference)
                            self.differentials.append(dict(case=case,profile=profile,arm=arm,shared=shared,
                                endpoints=len(values),result=result,work=dict(f.session.counts),
                                stats=dict(f.receipt['stats']),arena_peak=d.owner.stats['peak_bytes']))
                        finally:
                            if not life.closed: life.close()
                            if not d.closed: d.abort()

    def test_shared_roots_same_proposal_cost(self):
        from scoped_source.direct_lifecycle import FixedProposer
        rows=[]
        for arm in ('dense','pair_scoped'):
            f=self.forest('deep'); c=contract(f); w=f.session
            d=Driver(f,c,expected=identity(c),work=w,pair_scoped=arm=='pair_scoped')
            p=FixedProposer(method(arm,shared=True)); p.setup(f,w)
            tasks,lease,_=prepare(d); before=dict(w.counts)
            p.propose(tasks,lease)
            rows.append({k:w.counts[k]-v for k,v in before.items()})
            revoke(lease); p.close(); d.abort()
        self.assertEqual(*rows)

    def test_row_and_packet_immutable_no_fraction_alias(self):
        f=self.forest(); local=next(n[1] for n in f.nodes if type(n[1]) is SourceLocal and len(n[1]))
        before=tuple(local); q=local.row.coefficient(0); object.__setattr__(q,'_numerator',999)
        self.assertEqual(before,tuple(local))
        for obj,field in ((local,'tail'),(local.row,'raw'),(self.cases['half'][3],'pages')):
            with self.assertRaises(AttributeError): setattr(obj,field,None)
            with self.assertRaises(AttributeError): delattr(obj,field)
        with self.assertRaises(ValueError): PackedRow(('router',1,0),bytearray(16),work())
        with self.assertRaises(ValueError): Packet((bytearray(16),),work())
        snap=Snapshot(*self.cases['half'][:2],work())
        for field in ('_rows','_plan','_inputs','_receipt','work','packed_bytes'):
            with self.assertRaises(ValueError): setattr(snap,field,None)
            with self.assertRaises(AttributeError): delattr(snap,field)

    def test_source_alias_wrong_identity_even_equal_values(self):
        p=self.cases['half'][3]
        for edit in (
            lambda m:m['aliases'][3].__setitem__(1,'expert1'),
            lambda m:m['aliases'][0].__setitem__(2,999),
            lambda m:m['aliases'][0].__setitem__(3,999),
            lambda m:m['aliases'][0].__setitem__(4,2),
            lambda m:m.__setitem__('source','0'*64),
            lambda m:m.__setitem__('request','0'*64),
            lambda m:m['aliases'].pop(),
            lambda m:m['factors'][0].__setitem__(2,0),
            lambda m:m['roles']['pairs'].pop(),
        ):
            with self.subTest(edit=edit):
                with self.assertRaises((ValueError,IndexError,KeyError)): self.forest(packet=altered(p,edit))

    def test_sequence_missing_order_scope_and_expanded_limit(self):
        p=self.cases['half'][3]
        for edit in (lambda m:m['sequences'][0].__setitem__(1,0),
                     lambda m:m['views'][2].__setitem__('output',m['views'][0]['output']),
                     lambda m:m['views'][1].__setitem__('scope',2),
                     lambda m:m['views'][2].__setitem__('parent',0),
                     lambda m:m['views'][-1].__setitem__('ine',m['views'][0]['output'])):
            with self.subTest(edit=edit):
                with self.assertRaises(ValueError): self.forest(packet=altered(p,edit))
        # Metadata-only alias amplification: no quota monkeypatch or real model.
        from scoped_source.shared_build import Builder
        snap=Snapshot(*self.cases['half'][:2],work()); b=Builder(snap,snap.work)
        b.finish(None)
        b.views[0]['output']*=POLICY['rows']+1
        with self.assertRaisesRegex(ValueError,'membership cap'):
            from scoped_source.shared_format import encode
            encode(b,dict(self.forest().roles),None)

    def test_rehashed_wrong_center_rejected(self):
        p=self.cases['half'][3]; raw=bytearray(b''.join(p.pages))
        _,h,n,l,r,q,u,s=PREFIX.unpack_from(raw)
        offset=PREFIX.size+h+n*NODE.size+(l+r)*ENTRY.size
        struct.pack_into('<d',raw,offset,3.0)
        p=Packet(tuple(bytes(raw[i:i+PAGE]) for i in range(0,len(raw),PAGE)),work())
        with self.assertRaises(ValueError): self.forest(packet=p)

    def test_paged_boundary_and_no_complete_join(self):
        w=work(); out=Writer(w)
        values=[b'a'*(PAGE-3),b'12345678',b'z'*(PAGE+7)]
        for raw in values: out.put(raw)
        p=out.finish(); self.assertEqual(len(p.pages),3)
        rd=Reader(p,w)
        for raw in values: self.assertEqual(rd.read(len(raw)),raw)
        with self.assertRaises(ValueError): rd.read(1)
        with self.assertRaises(ValueError): Packet((b'x',b'y'),work())

    def test_page_loading_damage(self):
        p=self.cases['half'][3]; path=self.path('pages-'); path.mkdir()
        for i,raw in enumerate(p.pages): (path/f'{i:04d}.bin').write_bytes(raw)
        (path/'index.json').write_bytes(p.index)
        self.assertEqual(load(path,p.index,p.identity,work()).pages,p.pages)
        index=decode(p.index); index['pages'][0]['file']='0001.bin'; bad=compact(index)
        with self.assertRaises(ValueError): load(path,bad,digest(bad),work())
        (path/'extra.bin').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'inventory'): load(path,p.index,p.identity,work())
        (path/'extra.bin').unlink()
        (path/'0000.bin').write_bytes(p.pages[0][:-1])
        with self.assertRaises(ValueError): load(path,p.index,p.identity,work())

    def test_one_replay_and_source_close(self):
        snap=Snapshot(*self.cases['half'][:2],work())
        with self.assertRaises(ValueError): snap.receipt()
        it=snap.rows(); next(it); it.close()
        with self.assertRaises(ValueError): snap.receipt()
        with self.assertRaises(ValueError): list(snap.rows())
        f=self.forest(); w=f.session; w.counts['row_visits']=POLICY['visits']
        with self.assertRaisesRegex(ValueError,'work capacity'): w.visit('row_visits')

    def owned(self):
        root=self.path('owned-'); root.mkdir(); w=work(); token=capture_owned(root,'half',w)
        return root,w,token,Archive(root/'artifacts',w)

    def test_transfer_once_and_exact_accounting(self):
        root,w,t,a=self.owned(); before=census(t.root)['bytes']; ref=transfer(t,a)
        self.assertFalse(t.root.exists()); self.assertEqual(ref['bytes'],before)
        self.assertEqual(census(root/'artifacts')['bytes'],before); a.audit()
        with self.assertRaises(ValueError): transfer(t,a)
        with self.assertRaises(TypeError): OwnedSource()

    def test_transfer_rejects_foreign_alias_changed_and_overwrite(self):
        for mode in ('foreign','hardlink','symlink','root_symlink','mutate','destination','small'):
            root,w,t,a=self.owned()
            if mode=='foreign':
                other=self.path('other-'); other.mkdir(); a=Archive(other/'artifacts',w)
            elif mode=='hardlink':
                p=next((t.root/'t').iterdir()); os.link(p,root/'alias')
            elif mode=='symlink':
                (t.root/'bad').symlink_to(root/'absent')
            elif mode=='root_symlink':
                hidden=root/'original'; t.root.rename(hidden); t.root.symlink_to(hidden,target_is_directory=True)
            elif mode=='mutate':
                p=next((t.root/'t').iterdir()); raw=p.read_bytes(); p.write_bytes(bytes([raw[0]^1])+raw[1:])
            elif mode=='destination': (a.root/'package/source').mkdir(parents=True)
            else: a.limit=8193
            with self.assertRaises(ValueError): transfer(t,a)
            self.assertTrue(t.root.exists())

    def test_transfer_member_reserve_preflight(self):
        root,w,t,a=self.owned()
        with self.assertRaisesRegex(ValueError,'capacity'): transfer(t,a,member_limit=1)
        self.assertTrue(t.root.exists()); self.assertEqual(a.bytes,0); self.assertFalse(a.attempted)

    def test_transfer_failures_preserve_files(self):
        for fault in ('before_transfer','after_transfer'):
            root,w,t,a=self.owned()
            with self.assertRaises(RuntimeError): transfer(t,a,fault=fault)
            self.assertEqual(t.root.exists(),fault=='before_transfer')
            if fault=='after_transfer':
                a.audit(); self.assertEqual(a.bytes,t.inventory['bytes'])
            self.assertGreater(census(root)['bytes'],0)

    def test_transfer_deadline_refuses(self):
        root,w,t,a=self.owned(); w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): transfer(t,a)
        self.assertTrue(t.root.exists())

    def test_supervised_all_arms_profiles(self):
        for case in ('half','guarded'):
            for arm in ('dense','pair_scoped'):
                for profile in (0,1):
                    root,r=self.run_control(case,arm,profile)
                    self.assertEqual(r['status'],'COMPLETE')
                    self.assertFalse((root/'source').exists())
                    self.assertEqual(r['accepted']['snapshots']['source_transfer']['copies'],1)
                    self.assertEqual(r['accepted']['snapshots']['owner']['live_bytes'],0)

    def test_supervised_partial_exceptions_and_wrong_binding(self):
        for fault in ('before_transfer','after_transfer','setup_error','propose_error','close_error',
                      'partial','after_tile_error','output_partial','wrong_method','wrong_invocation'):
            _,r=self.run_control(fault=fault)
            self.assertEqual(r['status'],'ERROR',fault); self.assertIsNone(r['accepted'])

    def test_supervised_missing_is_not_positive(self):
        _,r=self.run_control(fault='missing')
        self.assertEqual(r['status'],'COMPLETE')
        self.assertEqual(r['accepted']['result']['status'],'UNKNOWN_MISSING_EVIDENCE')

    def test_supervised_deadlines_and_work(self):
        for fault in ('setup_block','propose_block','replay_block','close_block','output_block','late_exit',
                      'receive_expired','terminal_expired'):
            _,r=self.run_control(fault=fault,seconds=1.2)
            self.assertEqual(r['status'],'TIMEOUT',fault); self.assertIsNone(r['accepted'])
            self.assertFalse(r['child']['remaining_group']['live'])
        _,r=self.run_control(initial=dict(row_visits=POLICY['visits'],coefficient_visits=0,arithmetic=0))
        self.assertEqual(r['status'],'ERROR'); self.assertIsNone(r['accepted'])

    def test_relocation_isolated_full_recheck(self):
        root,r=self.run_control(); self.assertEqual(r['status'],'COMPLETE')
        dest=self.path('relocated-'); start=time.monotonic()
        shutil.copytree(root/'artifacts/package',dest)
        # Old location becomes unavailable, without deleting the archived run.
        hidden=root.with_name(root.name+'-hidden'); root.rename(hidden)
        command=[PYTHON,'-I','-S','-B','-c',
                 'import sys;sys.path.insert(0,'+repr(str(dest/'code'))+');from scoped_source.direct_lifecycle_package import main;main()',
                 str(dest),r['accepted']['package_sha256'],r['invocation'],
                 '--arm','pair_scoped','--profile','0','--shared-source','--enable-lifecycle','--seconds','30']
        out=subprocess.run(command,capture_output=True,text=True,timeout=35,check=True)
        checked=json.loads(out.stdout)
        self.assertEqual(checked['result'],r['accepted']['result'])
        self.assertGreater(census(dest)['bytes'],0); self.assertLess(time.monotonic()-start,35)
        # Package identity cannot be silently reinterpreted as another arm/profile.
        with self.assertRaises(ValueError):
            check_package(dest,expected=r['accepted']['package_sha256'],expected_method=method('dense',shared=True),
                          invocation=r['invocation'],work=work(),enabled=True)


if __name__=='__main__': unittest.main()
