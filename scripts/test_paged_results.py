"""Frozen fixed-candidate result-stream controls; stdlib only, no real models."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import POLICY, Work, compact, decode, digest, identity
from scoped_source.direct_method import method, validate
from scoped_source.direct_supervised import supervise, spec, audit, census, ROOT, PYTHON, read_json
from scoped_source.direct_lifecycle_package import check_package, bundle_schema
from scoped_source.direct_lifecycle_execution import execute
from scoped_source.direct_lifecycle import Lifecycle
from scoped_source.direct_candidates import Driver
from scoped_source.scaled_endpoints import contract
from scoped_source.scaled_endpoint_controls import fixture
from scoped_source.shared_build import build
from scoped_source.check_shared_source import CheckedSharedForest
from scoped_source.shared_archive import capture_owned
from scoped_source.result_stream import Writer, lower, row, summary, error_code, PAGE_BYTES, MAX_RATIONAL
from scoped_source.check_result_stream import Reader


def work(): return Work(time.monotonic()+300)


class Controls(unittest.TestCase):
    retained_root = None
    observations = []
    mutations = []
    relocations = []

    @classmethod
    def setUpClass(cls):
        cls.temp = None
        if cls.retained_root is None:
            cls.temp = tempfile.TemporaryDirectory(prefix='hz-paged-controls-', dir='/data1/Kane/MOE')
            cls.root = Path(cls.temp.name)
        else:
            cls.root = Path(cls.retained_root)
            cls.root.mkdir(parents=False, exist_ok=False)
        cls.cases = {}
        for case in ('half', 'guarded'):
            r,s = fixture(cls.root/case, case)
            packet,_ = build(r,s,work=work(),enabled=True)
            cls.cases[case] = (r,s,packet)

    @classmethod
    def tearDownClass(cls):
        if cls.temp: cls.temp.cleanup()

    def path(self, name): return self.root/(name+'-'+str(time.monotonic_ns()))

    def run_control(self, case='half', arm='pair_scoped', profile=0, fault='', seconds=300, initial=None):
        root = self.path('request')
        call = supervise(root,spec(case,arm,profile,fault,shared=True,paged_results=True),
                         enabled=True,seconds=seconds,initial_work=initial)
        checked = audit(root,call)
        self.observations.append(dict(case=case,arm=arm,profile=profile,fault=fault,status=call['status'],
            root=str(root),seconds=call['seconds'],budget=call['budget'],work=call['work'],audit=checked,storage=call['storage'],
            accepted=call['accepted'],child=call['child']))
        return root,call

    def old_result(self,case,arm,profile):
        r,s,p = self.cases[case]; w=work()
        f=CheckedSharedForest(p,r,s,expected=p.identity,work=w,enabled=True)
        c=contract(f); d=Driver(f,c,expected=identity(c),work=w,pair_scoped=arm=='pair_scoped')
        life=Lifecycle(method(arm,profile,shared=True))
        try:
            life.setup(f,w)
            for _ in range(d.session.tile_count):
                token,records=life.propose(d)
                d.submit(token,records,lambda i,b:(digest(b),len(b)),profile=profile)
            return d.finish()
        finally:
            life.close()
            if not d.closed: d.abort()

    def read_result(self,root):
        package=root/'artifacts/package'
        m=decode((package/'manifest.json').read_bytes())
        rd=Reader(package,m['result'],m['tiles'],work())
        return dict(rd.summary,results=list(rd.rows()))

    def replay(self,root,call,expected=None,config=None):
        return check_package(root,expected=expected or call['accepted']['package_sha256'],
            expected_method=config or call['accepted']['method'],invocation=call['invocation'],work=work(),enabled=True)

    @staticmethod
    def rehash(package):
        """Adversary updates all file-reference hashes, not the external method/invocation."""
        def ref(name):
            raw=(package/name).read_bytes()
            return dict(file=name,bytes=len(raw),sha256=digest(raw))
        idx=decode((package/'result/index.json').read_bytes())
        for p in idx['pages']: p.update(ref(p['file']))
        (package/'result/index.json').write_bytes(compact(idx))
        m=decode((package/'manifest.json').read_bytes()); m['result']=ref('result/index.json')
        (package/'manifest.json').write_bytes(compact(m))
        b=decode((package/'bundle.json').read_bytes())
        b['members']=[ref(p['file']) for p in b['members']]
        raw=compact(b); (package/'bundle.json').write_bytes(raw)
        return digest(raw)

    def copy_package(self,root):
        dest=self.path('mutation'); shutil.copytree(root/'artifacts/package',dest)
        return dest

    def test_default_off_and_limits(self):
        with self.assertRaises(ValueError): method('dense',paged_results=True)
        with self.assertRaisesRegex(ValueError,'opt-in'):
            supervise(self.path('off'),spec('half','dense',shared=True,paged_results=True))
        m=method('dense',shared=True,paged_results=True)
        for key in ('schema','protocol','results'):
            bad=copy.deepcopy(m); bad[key]='bad'
            with self.assertRaises(ValueError): validate(bad)
        self.assertEqual(POLICY['visits'],8000000)
        self.assertEqual(POLICY['rows'],65536)
        self.assertEqual(POLICY['total_bytes'],128*2**20)
        self.assertEqual(m['production']['check_width'],8)
        self.assertEqual(method('pair_scoped',shared=True,paged_results=True)['production']['check_width'],2)

    def test_implemented_success_schema_envelope(self):
        from scripts.assess_paged_result_implementation import assess
        result=assess()
        self.assertEqual(result['metadata_upper'],1702743)
        self.assertEqual(result['members_upper']['candidate_report'],165242)
        self.assertEqual(result['members_upper']['parent_receipt'],38312)
        self.assertEqual(result['page_upper'],40057)
        self.assertEqual(result['page_read_passes_online'],10)
        self.assertFalse(result['real_model_admitted'])

    def test_eight_same_candidate_differentials_and_costs(self):
        for case in self.cases:
            for arm in ('dense','pair_scoped'):
                for profile in (0,1):
                    root,call=self.run_control(case,arm,profile)
                    self.assertEqual(call['status'],'COMPLETE',call)
                    self.assertEqual(call['budget'],300)
                    self.assertEqual(self.read_result(root),self.old_result(case,arm,profile))
                    p=decode((root/'candidate.json').read_bytes())
                    self.assertNotIn('results',p['result'])
                    self.assertNotIn('results',p['replay']['result'])
                    self.assertFalse((root/'artifacts/result.json').exists())
                    self.assertEqual(p['result_ref'],p['replay']['result_ref'])
                    self.assertEqual(p['snapshots']['result_stream']['pages'],1 if case=='half' else 2)
                    self.assertEqual(p['replay']['result_pages_read'],1 if case=='half' else 2)
                    self.assertEqual(p['snapshots']['owner']['live_bytes'],0)
                    for key in p['work']:
                        self.assertEqual(p['work'][key],p['initial_work'][key]+sum(r['work'][key] for r in p['phases']))
                    self.assertEqual(p['archive']['logical_bytes'],census(root/'artifacts')['bytes'])
                    # Observations, not a proof that arbitrary valid reports fit these static envelopes.
                    self.assertLess((root/'candidate.json').stat().st_size,165242)
                    self.assertLess((root/'receipt.json').stat().st_size,38312)

    def test_missing_is_complete_unknown(self):
        root,call=self.run_control('guarded',fault='missing')
        self.assertEqual(call['status'],'COMPLETE')
        r=self.read_result(root)
        self.assertEqual(r['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(len(r['results']),12)
        self.assertTrue(any(r['lower'] is None for r in r['results']))

    def test_rehashed_real_lower_fails_math_both_arms(self):
        for arm in ('dense','pair_scoped'):
            root,call=self.run_control('guarded',arm)
            self.assertEqual(call['status'],'COMPLETE')
            dest=self.copy_package(root)
            page=decode((dest/'result/0000.json').read_bytes())
            r=page['rows'][0]
            r['lower']=str(F(r['lower'])+100)
            r['positive']=True
            (dest/'result/0000.json').write_bytes(compact(page))
            expected=self.rehash(dest)
            with self.assertRaisesRegex(ValueError,'independently recomputed result row'):
                self.replay(dest,call,expected)
            self.mutations.append(dict(kind='REHASHED_REAL_LOWER',arm=arm,expected=expected,
                                      rejection='independently recomputed result row'))

    def test_complete_roster_and_bindings_rehashed(self):
        root,call=self.run_control('guarded')
        for action in ('delete','duplicate','swap','extra','pair','property','source','forest','forest_index',
                       'contract','method','invocation','roster','candidates','missing_page','page_order'):
            dest=self.copy_package(root)
            idx=decode((dest/'result/index.json').read_bytes())
            page=decode((dest/'result/0000.json').read_bytes())
            if action=='delete': page['rows'].pop()
            elif action=='duplicate': page['rows'][1]=copy.deepcopy(page['rows'][0])
            elif action=='swap': page['rows'][0],page['rows'][1]=page['rows'][1],page['rows'][0]
            elif action=='extra': page['rows'].append(copy.deepcopy(page['rows'][0]))
            elif action=='pair': page['rows'][0]['pair']=[1,2]
            elif action=='property': page['rows'][0]['other']=9
            elif action=='candidates': idx['candidates']='0'*64
            elif action=='missing_page': idx['pages'].pop()
            elif action=='page_order': idx['pages'].reverse()
            else:
                idx['context'][action]='0'*64
                # Rebind page context too: refusal is at private plan, not old page hash.
                page['context']=identity(idx['context'])
                second=decode((dest/'result/0001.json').read_bytes()); second['context']=identity(idx['context'])
                (dest/'result/0001.json').write_bytes(compact(second))
            (dest/'result/0000.json').write_bytes(compact(page))
            (dest/'result/index.json').write_bytes(compact(idx))
            expected=self.rehash(dest)
            with self.subTest(action=action), self.assertRaises(ValueError): self.replay(dest,call,expected)
            self.mutations.append(dict(kind=action,expected=expected,rejected=True))
        with self.assertRaises(ValueError):
            self.replay(root/'artifacts/package',call,config=method('pair_scoped',shared=True))

    def test_reader_prechecks(self):
        root,call=self.run_control('guarded')
        dest=self.copy_package(root)
        m=decode((dest/'manifest.json').read_bytes())
        idx=decode((dest/'result/index.json').read_bytes())
        raw=b'x'*(PAGE_BYTES+1)
        (dest/'result/0000.json').write_bytes(raw)
        idx['pages'][0].update(bytes=len(raw),sha256=digest(raw))
        raw=compact(idx); (dest/'result/index.json').write_bytes(raw)
        ref=dict(file='result/index.json',bytes=len(raw),sha256=digest(raw))
        with self.assertRaisesRegex(ValueError,'bounded result member'): Reader(dest,ref,m['tiles'],work())
        for value in ('01','-0','1/1','1/0','nan','1e3','1/'+'9'*MAX_RATIONAL,str(2**8192)):
            with self.subTest(value=value[:10]), self.assertRaises(ValueError): lower(value)
        q=F(2**8191-1,2**8191+1)
        self.assertEqual(lower(str(q)),q)  # work bits, not smaller stored-bit gate
        for value in (dict(pair=[0,1],other=1,lower='0',positive=True),
                      dict(pair=[0,1],other=True,lower='0',positive=False)):
            with self.assertRaises(ValueError): row(value)

    def test_writer_alias_bounds_and_source_seal(self):
        r,s,p=self.cases['guarded']; w=work()
        f=CheckedSharedForest(p,r,s,expected=p.identity,work=w,enabled=True)
        c=contract(f); d=Driver(f,c,expected=identity(c),work=w,pair_scoped=True)
        life=Lifecycle(method('pair_scoped',shared=True,paged_results=True)); saved={}; refs=[]
        def put(name,raw):
            if name in saved: raise ValueError('no overwrite')
            saved[name]=raw
            return dict(file=name,bytes=len(raw),sha256=digest(raw))
        def publish(i,raw):
            ref=put(f'tiles/{i:04d}.bin',raw); refs.append(ref); return ref['sha256'],ref['bytes']
        try:
            life.setup(f,w)
            for _ in range(d.session.tile_count):
                token,records=life.propose(d); d.submit(token,records,publish)
            sink=Writer(put,life.method if hasattr(life,'method') else method('pair_scoped',shared=True,paged_results=True),
                        '1'*64,p.identity,refs,w)
            with patch.object(CheckedSharedForest,'seal',side_effect=ValueError('seal rejection')):
                with self.assertRaisesRegex(ValueError,'seal rejection'): d.finish(sink=sink)
            self.assertIn('result/0000.json',saved)
            self.assertNotIn('result/index.json',saved)
            self.assertTrue(d.closed)
        finally:
            life.close()
            if not d.closed: d.abort()
        # Closed-format unit control, no source/endpoint claim: caller aliases cannot mutate published bytes.
        saved={}; w=work()
        sink=Writer(put,method('dense',shared=True,paged_results=True),'1'*64,'2'*64,
                    [dict(file='tiles/0000.bin',bytes=1,sha256='3'*64)],w)
        sink.context=dict(schema='SHARED_RESULT_CONTEXT_V1',**{k:'1'*64 for k in
                          ('source','forest','forest_index','contract','method','invocation','roster')})
        item=dict(pair=[0,1],other=1,lower='0',positive=False)
        sink.emit(item); item['pair'][0]=7
        val=dict(status='UNKNOWN_NONPOSITIVE',required=1,positive=0,required_endpoints=1,checked_endpoints=1,
                 missing_endpoints=0,native_float_proof=False,checkpoint_correspondence_checked=False)
        sink.finish(val); val['positive']=1
        self.assertEqual(decode(saved['result/0000.json'])['rows'][0]['pair'],[0,1])
        self.assertEqual(decode(saved['result/index.json'])['summary']['positive'],0)
        with self.assertRaises(ValueError): sink.finish(val)
        with self.assertRaises(ValueError): sink.emit(item)

    def test_partial_and_exception_controls(self):
        for fault in ('after_result_page_error','partial','after_tile_error','setup_error','propose_error','close_error',
                      'output_partial','wrong_method','wrong_invocation','startup_error'):
            root,call=self.run_control('guarded',fault=fault)
            self.assertEqual(call['status'],'ERROR',fault)
            self.assertIsNone(call['accepted'])
            self.assertGreater(call['storage']['bytes'],0)
            if fault=='after_result_page_error':
                self.assertTrue((root/'artifacts/package/result/0000.json').exists())
                self.assertFalse((root/'artifacts/package/result/index.json').exists())

    def test_hard_deadlines_and_preconsumed_work(self):
        for fault in ('after_result_page_block','index_publication_expired','replay_block',
                      'result_receive_expired','terminal_expired','late_exit'):
            root,call=self.run_control('guarded',fault=fault,seconds=1.5)
            self.assertEqual(call['status'],'TIMEOUT',fault)
            self.assertIsNone(call['accepted'])
            self.assertFalse(call['child']['remaining_group']['live'])
            if fault=='after_result_page_block':
                self.assertTrue((root/'artifacts/package/result/0000.json').exists())
        _,call=self.run_control(initial=dict(row_visits=POLICY['visits'],coefficient_visits=0,arithmetic=0))
        self.assertEqual(call['status'],'ERROR')

    def test_bounded_errors_never_stringify(self):
        class Hostile(Exception):
            def __str__(self): raise AssertionError('str called')
            def __repr__(self): raise AssertionError('repr called')
        self.assertEqual(error_code(Hostile()),'EXECUTION_ERROR')
        root=self.path('hostile'); root.mkdir(); w=work(); owned=capture_owned(root,'half',w)
        with patch.object(Lifecycle,'setup',side_effect=Hostile()):
            r=execute(root/'artifacts',owned.root,owned.source,method('dense',shared=True,paged_results=True),
                      '1'*64,ROOT,work=w,enabled=True,owned_source=owned)
        self.assertEqual(r['error'],'EXECUTION_ERROR')
        from scoped_proof.owned_bounded import execute as run_owned
        with patch('scoped_proof.owned_bounded.subprocess.Popen',side_effect=Hostile()):
            r=run_owned(['not-run'],self.path('hostile-log'),run_deadline=time.monotonic()+1,
                        cleanup_deadline=time.monotonic()+2,env={},rss_limit=2**31,bounded_errors=True)
        self.assertEqual(r['error'],'EXECUTION_ERROR')

    def test_metadata_only_maximum_pages(self):
        # No source acceptance claim: exercise serializer capacity, not a full-size model.
        saved={}; w=work()
        def put(name,raw):
            if name in saved: raise ValueError('overwrite')
            saved[name]=raw; return dict(file=name,bytes=len(raw),sha256=digest(raw))
        sink=Writer(put,method('dense',shared=True,paged_results=True),'1'*64,'2'*64,
                    [dict(file='tiles/0000.bin',bytes=1,sha256='3'*64)],w)
        sink.context=dict(schema='SHARED_RESULT_CONTEXT_V1',**{k:'1'*64 for k in
                          ('source','forest','forest_index','contract','method','invocation','roster')})
        q=str(F(2**8191-1,2**8191+1))
        for _ in range(252): sink.emit(dict(pair=[0,1],other=1,lower=q,positive=True))
        self.assertEqual(len(sink.page),4)
        with self.assertRaises(ValueError): sink.emit(dict(pair=[0,1],other=1,lower=q,positive=True))
        sink.finish(dict(status='CHECKED_POSITIVE_CAPTURED_REAL_SOURCE',required=252,positive=252,
            required_endpoints=504,checked_endpoints=504,missing_endpoints=0,
            native_float_proof=False,checkpoint_correspondence_checked=False))
        self.assertEqual(len(sink.pages),32)
        self.assertTrue(all(r['bytes']<=PAGE_BYTES for r in sink.pages))
        self.assertEqual(len(decode(saved['result/0031.json'])['rows']),4)

    def test_final_publication_failure_and_late_return(self):
        import scoped_source.direct_supervised as module
        original=module.write_json
        for mode in ('partial','late'):
            root=self.path('final')
            def changed(path,value,cap):
                if Path(path).name=='receipt.json':
                    if mode=='partial':
                        Path(path).write_bytes(b'{')
                        raise OSError('fixed partial final write')
                    inv,_=read_json(root/'invocation.json',65536)
                    while time.monotonic()<inv['deadline']:
                        time.sleep(min(.01,max(0.,inv['deadline']-time.monotonic())))
                return original(path,value,cap)
            with patch.object(module,'write_json',changed):
                call=supervise(root,spec('half','dense',shared=True,paged_results=True),enabled=True,seconds=1.5)
            self.assertIsNone(call['accepted'])
            checked=audit(root,call)
            if mode=='partial':
                self.assertIsNone(call['receipt_sha256'])
                self.assertEqual(checked['status'],'INCOMPLETE_TERMINAL')
                self.assertFalse(checked['cost_complete'])
                self.assertTrue(checked['storage_complete'])
            else:
                self.assertEqual(call['status'],'TIMEOUT')
                self.assertEqual(decode((root/'receipt.json').read_bytes())['status'],'COMPLETE')
            self.observations.append(dict(kind='FINAL_PUBLICATION_'+mode.upper(),call=call,audit=checked))

    def test_worker_early_read_fixed_error_and_failed_schema(self):
        root=self.path('bad-invocation'); root.mkdir()
        (root/'invocation.json').write_bytes(b'{')
        script='import sys;sys.path.insert(0,'+repr(str(ROOT))+');from scoped_source.direct_lifecycle_worker import main;main()'
        cp=subprocess.run([PYTHON,'-I','-S','-B','-c',script,str(root),'0'*64,'--paged-errors'],
                          capture_output=True,timeout=5,check=False)
        self.assertEqual(cp.returncode,2)
        self.assertEqual(cp.stderr,b'PAGED_WORKER_REJECTED\n')
        root,call=self.run_control(fault='startup_error')
        inv=decode((root/'invocation.json').read_bytes()); inv['schema']='OTHER'
        raw=compact(inv); (root/'invocation.json').write_bytes(raw)
        receipt=decode((root/'receipt.json').read_bytes()); receipt['invocation_sha256']=digest(raw)
        raw=compact(receipt); (root/'receipt.json').write_bytes(raw)
        wrong=copy.deepcopy(call); wrong['receipt_sha256']=digest(raw); wrong['storage']=census(root)
        with self.assertRaisesRegex(ValueError,'invocation schema'): audit(root,wrong)

    def test_relocated_isolated_both_arms_no_producer(self):
        for arm in ('dense','pair_scoped'):
            root,call=self.run_control('guarded',arm)
            self.assertEqual(call['status'],'COMPLETE')
            dest=self.path('relocated'); start=time.monotonic()
            shutil.copytree(root/'artifacts/package',dest)
            hidden=root.with_name(root.name+'-hidden'); root.rename(hidden)
            script="""import sys,os
sys.path.insert(0,sys.argv.pop(1))
blocked=[os.path.realpath(p) for p in sys.argv.pop(1).split('|')]
def hook(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=os.path.realpath(os.fsdecode(args[0]))
        if any(p==b or p.startswith(b+'/') for b in blocked): raise RuntimeError('forbidden historical path')
sys.addaudithook(hook)
for p in blocked:
    try: open(p+'/invocation.json','rb')
    except RuntimeError: pass
    else: raise AssertionError('old path not blocked')
from scoped_source.direct_lifecycle_package import main
main()
assert not any(n.split('.')[0] in ('numpy','torch','scipy') for n in sys.modules)
assert 'scoped_source.shared_build' not in sys.modules
"""
            command=[PYTHON,'-I','-S','-B','-c',script,str(dest/'code'),str(ROOT)+'|'+str(root)+'|'+str(hidden),
                     str(dest),call['accepted']['package_sha256'],call['invocation'],'--arm',arm,'--profile','0',
                     '--shared-source','--paged-results','--enable-lifecycle','--seconds','300']
            out=subprocess.run(command,capture_output=True,text=True,timeout=305,check=True)
            report=json.loads(out.stdout)
            self.assertEqual(report['result'],call['accepted']['result'])
            self.assertEqual(report['result_ref'],call['accepted']['result_ref'])
            self.relocations.append(dict(arm=arm,seconds=time.monotonic()-start,report=report,
                bytes=census(dest)['bytes'],budget=300,old_paths_denied=True,isolated=True,no_site=True))


if __name__=='__main__': unittest.main()
