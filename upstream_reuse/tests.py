"""Analytic controls only; no real requests, schedule tuning or speed claims."""
import ast
import copy
import inspect
import unittest
from unittest.mock import patch

from evidence_handoff.tests import source, checked
from evidence_handoff.proposal import propose_with_handoff
from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired
from scripts.optional_evidence_dev_contract import read, save
from portable_proof.runtime import digest
from upstream_reuse.storage import SourceCache
from upstream_reuse.operations import Operations
from upstream_reuse.proposal import run
from upstream_reuse.timing import Timers


class Controls(unittest.TestCase):
    def test_original_algorithm_ast(self):
        from moe_evidence.generate import propose_all
        from act.back_end.solver.sparse_lp_certificate import propose
        from upstream_reuse import schedule, native
        for original, derived in ((propose_all, schedule.propose_all), (propose, native.propose)):
            def tree(fn):
                node = ast.parse(inspect.getsource(fn)).body[0]
                node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
                return ast.dump(node, include_attributes=False)
            self.assertEqual(tree(original), tree(derived))

    def test_four_way_exact_differential(self):
        for dims in ((2,3,3,False),(3,3,3,True),(4,4,-2,False)):
            baseline = None; evaluations = None
            for flags in ((False,False),(True,False),(False,True),(True,True)):
                with source(*dims) as (root, req):
                    report = run(root, req, EvidenceBudget(0,clock=lambda:0),
                                 source_enabled=flags[0], matrix_enabled=flags[1])
                    value = (read(root/'manifest.json'), checked(root,req))
                    if baseline is None: baseline = value
                    self.assertEqual(value, baseline)
                    calls = report['timings']['dual_evaluate']['calls']
                    if evaluations is None: evaluations = calls
                    self.assertEqual(calls, evaluations)
                    log = read(root/'query_log.json')
                    self.assertEqual(calls, 3*len(log))
                    self.assertEqual(report['matrix_cache']['live_entries'],0)
                    self.assertEqual(report['source_cache']['live_entries'],0)
                    self.assertEqual(report['matrix_cache']['hits']>0, flags[1])
                    self.assertEqual(report['source_cache']['hits']>0, flags[0])
            with source(*dims) as (root, req):
                propose_with_handoff(root,EvidenceBudget(0,clock=lambda:0))
                self.assertEqual((read(root/'manifest.json'),checked(root,req)),baseline)

    def test_alias_hash_scope_and_pollution(self):
        with source() as (root,req):
            p=root/read(root/'manifest.json')['common_facts']['file']; sha=digest(p.read_bytes())
            cache=SourceCache('r',enabled=True)
            original=cache.load(p,sha,scope='r')
            original['bad']=True
            self.assertNotIn('bad',cache.load(p,sha,scope='r'))
            hit=cache.load(p,sha,scope='r');hit['bad']=True
            self.assertNotIn('bad',cache.load(p,sha,scope='r'))
            with self.assertRaises(ValueError):cache.load(p,sha,scope='other')
            key=next(iter(cache._items));entry=cache._items[key]
            cache._items[key]=('foreign',*entry[1:])
            with self.assertRaises(ValueError):cache.load(p,sha,scope='r')
            cache.clear();save(p,{'tampered':1})
            with self.assertRaises(ValueError):cache.load(p,sha,scope='r')

    def test_eviction_oversize_and_deadline(self):
        with source() as (root,req):
            cache=SourceCache('r',enabled=True,limits={'entries':1,'payload_bytes':100000,'nodes':10000})
            for name in (read(root/'manifest.json')['common_facts']['file'],'manifest.json'):
                p=root/name;cache.load(p,digest(p.read_bytes()),scope='r')
            self.assertEqual(cache.stats()['evictions'],1)
            p=root/'manifest.json'
            tiny=SourceCache('r',enabled=True,limits={'entries':1,'payload_bytes':1,'nodes':1})
            tiny.load(p,digest(p.read_bytes()),scope='r')
            self.assertEqual(tiny.stats()['oversized'],1)
            cache.tick=lambda: (_ for _ in ()).throw(EvidenceBudgetExpired('deadline'))
            with self.assertRaises(EvidenceBudgetExpired):cache.load(p,digest(p.read_bytes()),scope='r')

    def test_bound_source_and_request_mutation(self):
        with source() as (root,req):
            ops=Operations(root,req,EvidenceBudget(0,clock=lambda:0),source_enabled=True,matrix_enabled=True)
            try:
                name=next(iter(ops.refs));ops.read(root/name)
                value=read(root/name);value['tampered']=True;save(root/name,value)
                with self.assertRaises(ValueError):ops.read(root/name)
                with self.assertRaises(ValueError):ops.read(root/'unbound.json')
                m=read(root/'manifest.json');m['request']['classes']+=1;save(root/'manifest.json',m)
                with self.assertRaises(ValueError):ops.read(root/'manifest.json')
            finally:ops.close()

    def test_warm_cache_does_not_cache_validation(self):
        with source() as (root,req):
            ops=Operations(root,req,EvidenceBudget(0,clock=lambda:0),matrix_enabled=True)
            try:
                item=next(iter(read(root/'manifest.json')['supports'].values()))
                record=ops.read(root/item['export']['file'])
                ops.export_check(record,None,expected_source_sha256=item['source_sha256'])
                with self.assertRaises(ValueError):ops.export_check(record,None,expected_source_sha256='bad')
                cert=ops.propose(record['lp'],time_limit=1)
                bad=copy.deepcopy(cert);bad['claimed_lower_bound']='1000000000'
                with self.assertRaises(ValueError):ops.export_check(record,bad,expected_source_sha256=item['source_sha256'])
                from act.back_end.solver.rational_mccormick import csr
                matrix=csr([{0:1}],1);self.assertEqual(list(ops.rows(matrix,1))[0][0][1],1)
                matrix['data'][0]='2';self.assertEqual(list(ops.rows(matrix,1))[0][0][1],2)
                with self.assertRaises(ValueError):list(ops.rows(matrix,2))
            finally:ops.close()

    def test_reserve_and_failure_clear(self):
        with source() as (root,req):
            report=run(root,req,EvidenceBudget(0,clock=lambda:221),source_enabled=True,matrix_enabled=True)
            self.assertFalse((root/'query_log.json').exists())
            self.assertGreater(checked(root,req)['missing_obligations'],0)
            with self.assertRaises(FileExistsError):run(root,req,EvidenceBudget(0,clock=lambda:0))
        with source() as (root,req), patch.object(Operations,'propose',side_effect=ValueError('bad dual')):
            with self.assertRaisesRegex(ValueError,'bad dual'):
                run(root,req,EvidenceBudget(0,clock=lambda:0),source_enabled=True,matrix_enabled=True)
            report=read(root/'upstream_reuse_report.json')
            self.assertEqual(report['reason'],'ERROR')
            self.assertEqual(report['matrix_cache']['live_entries'],0)
            self.assertEqual(report['source_cache']['live_entries'],0)

    def test_nested_timing_and_exception(self):
        now=[0.];timer=Timers(clock=lambda:now[0])
        def child():now[0]+=2
        def parent():
            now[0]+=1;timer.call('child',child);now[0]+=3
        timer.call('parent',parent)
        self.assertEqual(timer.values['parent']['inclusive_seconds'],6)
        self.assertEqual(timer.values['parent']['exclusive_seconds'],4)
        def fail():now[0]+=1;raise ValueError('fail')
        with self.assertRaises(ValueError):timer.call('failure',fail)
        self.assertEqual(timer.stack,[])


if __name__=='__main__':unittest.main()
