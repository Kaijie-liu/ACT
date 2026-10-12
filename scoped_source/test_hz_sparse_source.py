"""Frozen sparse-source connection controls, not a performance experiment."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction as F
import json
import time
import unittest
from unittest.mock import patch

from source_enclosure import produce
from source_enclosure.format import identity
from scoped_source.endpoint_source_controls import cases
from scoped_source import hz_block_source as old, check_hz_block_source as old_check
from scoped_source import hz_sparse_source as api, check_hz_sparse_source as receive
from scoped_source import hz_lifted_source as legacy
from scoped_source import hz_row_enclosure as old_rows, check_hz_row_enclosure as old_row_check
from scoped_source import hz_sparse_rows as rows, graph
from scoped_source import hz_sparse_source_controls as control
from act.back_end.moe import hz_endpoints, weighted_top2, batched_support
from act.back_end.solver import hz_lp_export

OBS = {}


def arm(name, mode):
    start = time.monotonic(); deadline = start+300
    mod, check = (old,old_check) if mode=='old' else (api,receive)
    item = {'start':start,'deadline':deadline,'cost':{}}; OBS['cases'].setdefault(name,{})[mode] = item
    doc = next(d for n,d,_ in cases() if n==name); item['source'] = doc
    cost = item['cost']; cost['source_creation'] = time.monotonic()-start
    stage = 'construction'
    def observe(k,v):
        nonlocal stage
        cost[k] = v; stage = 'proposals' if k=='construction' else 'serialization'
    try:
        with ExitStack() as stack:
            calls = stack.enter_context(patch.object(mod,'propagate',wraps=mod.propagate))
            sites = [('joint',hz_endpoints,'shared_input_pair_hz'), ('remap',weighted_top2,'_remap_columns'),
                     ('export',batched_support,'export'), ('direct_export',hz_lp_export,'export'),
                     ('old_projection',old_row_check,'projected')]
            counts = {k:stack.enter_context(patch.object(m,a,wraps=getattr(m,a))) for k,m,a in sites}
            item['package'] = mod.build(doc,expected_source_sha256=identity(doc),deadline=deadline,observe=observe)
            item['dispatch'] = [c.args[1] for c in calls.call_args_list]
            t = time.monotonic(); raw = json.dumps(item['package'],sort_keys=True,allow_nan=False)
            package = json.loads(raw); cost['serialization'] = time.monotonic()-t
            stage = 'checking'; t = time.monotonic()
            item['checked'] = check.check(doc,package,expected_source_sha256=identity(doc),deadline=deadline)
            cost['checking'] = time.monotonic()-t; item['serialized_bytes'] = len(raw.encode())
            item['boundaries'] = {k:c.call_count for k,c in counts.items()}
        item['end'] = time.monotonic(); cost['total'] = item['end']-start
        if item['end'] >= deadline: raise TimeoutError('complete source control deadline')
    except Exception as exc:
        item['end'] = time.monotonic(); cost['total'] = item['end']-start
        item['error'] = repr(exc); item['unfinished_stage'] = stage
        raise


def bridge(name, mode):
    start = time.monotonic(); deadline = start+30; record = {'start':start,'deadline':deadline}
    OBS['bridges'].setdefault(name,{})[mode] = record
    try:
        _bridge(name,mode,record,deadline)
        record['end'] = time.monotonic()
        if record['end'] >= deadline: raise TimeoutError('complete bridge control deadline')
    except Exception as exc:
        record['end'] = time.monotonic(); record['error'] = repr(exc)
        raise


def _bridge(name, mode, record, deadline):
    item = deepcopy(control.bridge_sources(name)); source = item['source']; tag = 'bridge/'+name
    record['item'] = item
    if name=='affine':
        import numpy as np
        from act.back_end.solver.solver_hz import sparse_hz_linear
        op = [{int(k):F(v) for k,v in row.items()} for row in item['operator']]; bias = list(map(F,item['bias']))
        nominal = hz_lp_export.snapshot(sparse_hz_linear(api.live(source['state']),np.array([[float(op[0][0])]]),np.array([0.])))
        ref, cert = produce.affine(source['state'],op,bias,nominal,tag)
        item.update(nominal=nominal,certificate=cert)
    elif name=='relu':
        ref, cert = api.relu(source['state'],tag); item['certificate'] = cert
    else:
        ref = api.entry_for(source['state'],item['router']['state'],(0,2),3)
    reference = api.owned(ref,item.get('router',source),tag)
    _, item['lift'] = (legacy if mode=='old' else api).enclosure(reference,tag,deadline)
    record['checked'] = control.check_bridge(name,item,mode,deadline)


class SparseSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OBS.clear(); OBS.update(cases={},bridges={},negatives={},faults={},capacity={},partial={})
        for index,name in enumerate(control.SOURCES):
            for mode in control.ARMS[::1 if index%2==0 else -1]: arm(name,mode)
        for name in control.BRIDGES:
            for mode in control.ARMS: bridge(name,mode)

    def test_01_complete(self):
        self.assertEqual(set(OBS['cases']),set(control.SOURCES))
        for name,(sha,duties,endpoints) in control.SOURCES.items():
            for item in OBS['cases'][name].values():
                r = item['checked']; self.assertEqual(identity(item['source']),sha)
                self.assertEqual((r['required'],r['checked_endpoints'],r['missing_endpoints']),(duties,endpoints,0))
                if name=='unsafe_tied': self.assertEqual(r['positive'],0)

    def test_02_differential(self):
        OBS['differentials'] = {n:control.differential(v['old'],v['sparse']) for n,v in OBS['cases'].items()}

    def test_03_dispatch(self):
        for name,arms in OBS['cases'].items():
            for mode,r in arms.items():
                self.assertEqual(r['dispatch'],['router']+[f'expert{i}' for i in range(r['source']['request']['experts'])])
                for k,v in r['boundaries'].items():
                    if k=='old_projection' and mode=='old': self.assertGreater(v,0)
                    else: self.assertEqual(v,0)
            r = arms['sparse']['checked']; self.assertEqual(r['sparse_core_work']['source_parses'],r['checked_sparse_lifts'])
            self.assertEqual(r['sparse_core_work']['target_parses'],r['checked_sparse_lifts'])

    def negatives(self, sparse):
        for name in control.NEGATIVE_ERRORS:
            if name.startswith('sparse_') != sparse: continue
            start = time.monotonic(); deadline = start+30
            spec = control.negative_queries(OBS,only=name)[name]
            got = control.recheck_negative(spec,deadline)
            OBS['negatives'][name] = {'input':spec,'checked':got,'start':start,'end':time.monotonic(),'deadline':deadline}

    def test_04_source_mutations(self): self.negatives(False)
    def test_05_sparse_mutations(self): self.negatives(True)

    def test_06_bridges(self):
        for name,arms in OBS['bridges'].items():
            for key in ('reference','snapshot'):
                self.assertEqual(arms['old']['item']['lift'][key],arms['sparse']['item']['lift'][key])
            self.assertEqual(arms['old']['item']['lift']['enclosure']['target'],arms['sparse']['item']['lift']['enclosure']['target'])
            if name!='guard': self.assertGreater(arms['sparse']['checked']['added_continuous'],0)
            else:
                lift = arms['sparse']['item']['lift']
                self.assertNotEqual(lift['reference']['state']['hz']['ub'],lift['enclosure']['target']['state']['hz']['ub'])

    def test_07_dense_capacity(self):
        for name in ('old_dense','global_factor','row_output'):
            start = time.monotonic(); deadline = start+30
            ref = control.dense_reference() if name=='old_dense' else control.capacity(name)
            fn = legacy.enclosure if name=='old_dense' else api.enclosure
            try: fn(ref,'capacity',deadline)
            except ValueError as exc:
                self.assertIn('capacity',str(exc))
                OBS['capacity'][name] = {'reference':ref,'error':str(exc),'error_type':type(exc).__name__,
                    'start':start,'end':time.monotonic(),'deadline':deadline}
            else: self.fail('capacity refusal missing')
        start = time.monotonic(); deadline = start+30; ref = control.dense_reference()
        _, lift = api.enclosure(ref,'dense',deadline); _, checked = receive.lift(lift,'dense',deadline)
        OBS['dense'] = {'reference':ref,'lift':lift,'checked':checked,'start':start,'end':time.monotonic(),'deadline':deadline}
        self.assertEqual(checked['added_continuous'],1)

    def test_08_partial_stale(self):
        for mode, checker in (('old',old_check),('sparse',receive)):
            start = time.monotonic(); deadline = start+30
            parent = OBS['cases']['tied_partial_reuse'][mode]; package = deepcopy(parent['package'])
            package['proof']['pairs'][-1]['candidates'] = None
            got = checker.check(parent['source'],package,expected_source_sha256=identity(parent['source']),deadline=deadline)
            self.assertEqual((got['status'],got['checked_endpoints'],got['missing_endpoints']),('UNKNOWN_MISSING_EVIDENCE',15,3))
            OBS['partial'][mode] = {'package':package,'checked':got,'start':start,'end':time.monotonic(),'deadline':deadline}
        self.assertIn('old_proof',OBS['negatives'])

    def fault(self, name):
        real = time.monotonic; start = real(); deadline = start+30; now = [start]; reached = []
        parent = OBS['cases']['weighted_sign']['sparse']; doc = deepcopy(parent['source']); p = deepcopy(parent['package'])
        anchors = [identity(doc),identity(p)]; original_snapshot = hz_lp_export.snapshot; original_check = receive.check_request
        def snap(hz):
            result = original_snapshot(hz); reached.append('after_live_snapshot'); now[0] = deadline+1; return result
        def checked(*args,**kwargs):
            result = original_check(*args,**kwargs); reached.append('after_endpoints'); p['source_sha256'] = '0'*64; return result
        def observe(k,v):
            if k=='construction': reached.append('after_construction'); doc['request']['margin'] = '1'
        with patch.object(api,'propose_request',wraps=api.propose_request) as proposed:
            try:
                with patch.object(graph.time,'monotonic',side_effect=lambda:now[0]):
                    if name.startswith('expired'):
                        reached.append('before_'+name.split('_')[1]); now[0] = deadline+1
                    if name in ('expired_producer','source_before_proposals'):
                        api.build(doc,expected_source_sha256=anchors[0],deadline=deadline,observe=observe if name=='source_before_proposals' else None)
                    elif name=='delayed_live':
                        with patch.object(hz_lp_export,'snapshot',side_effect=snap): api.enclosure(control.dense_reference(),'fault',deadline)
                    else:
                        with patch.object(receive,'check_request',side_effect=checked if name=='checker_late_package' else original_check):
                            receive.check(doc,p,expected_source_sha256=anchors[0],deadline=deadline)
            except (TimeoutError, ValueError) as exc:
                self.assertEqual((type(exc).__name__,str(exc)),control.ERRORS[name]); self.assertEqual(len(reached),1)
                OBS['faults'][name] = {'error_type':type(exc).__name__,'error':str(exc),'reached':reached,
                    'start':start,'end':real(),'deadline':deadline,'simulated_clock':now[0],
                    'source_before':anchors[0],'package_before':anchors[1],
                    'source_after':identity(doc),'package_after':identity(p),'proposal_calls':proposed.call_count}
            else: self.fail('fault did not occur')
            self.assertEqual(proposed.call_count,0)

    def test_09_deadlines(self):
        for name in control.FAULTS[:3]: self.fault(name)

    def test_10_identity_faults(self):
        for name in control.FAULTS[3:]: self.fault(name)

    def test_11_independence(self):
        parent = OBS['cases']['weighted_sign']['sparse']; start = time.monotonic(); deadline = start+30
        with patch.object(api,'build',side_effect=AssertionError),patch.object(rows,'produce',side_effect=AssertionError), \
                patch.object(old_rows,'produce',side_effect=AssertionError),patch.object(api,'propose_request',side_effect=AssertionError):
            got = receive.check(parent['source'],parent['package'],expected_source_sha256=identity(parent['source']),deadline=deadline)
        self.assertEqual(got,parent['checked']); OBS['independence'] = {'start':start,'end':time.monotonic(),'deadline':deadline,'checked':got}

    def test_12_archive_cost(self):
        from scripts.run_hz_sparse_source import inventory, times, check_cost
        inventory(OBS)
        for arms in OBS['cases'].values():
            for item in arms.values(): check_cost(item)
        for kind in ('negatives','capacity','faults','partial'):
            for item in OBS[kind].values(): times(item,30)
        for key in ('cases','bridges','negatives','faults','capacity','partial'):
            bad = deepcopy(OBS); bad[key].pop(next(iter(bad[key])))
            with self.assertRaises(ValueError): inventory(bad)
        bad = deepcopy(OBS); name = next(iter(bad['faults'])); bad['faults'][name]['error'] = 'unrelated deadline'
        with self.assertRaises(ValueError): inventory(bad)
        bad = deepcopy(OBS); bad['negatives']['owner']['input'] = deepcopy(bad['negatives']['gate']['input'])
        with self.assertRaises(ValueError): inventory(bad)


if __name__ == '__main__': unittest.main()
