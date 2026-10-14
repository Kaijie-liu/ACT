"""Static arithmetic/source-binding tests; never import or execute a consumer."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts import assess_pair_scoped_capacity as a


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=a.analyze()

    def test_literal_layout_and_source_anchors(self):
        a.verify_files()
        self.assertEqual(self.r['layout'],dict(slot=2065,scalar=2064,stored=1040,
                                              root=16,index=4,prefix=128,record=72))
        with patch.dict(a.ANCHORS,{'scoped_source/pair_workspace.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor drift'): a.verify_files()

    def test_owned_lifetime_exact_integer_balance(self):
        r=self.r['conservative_global_arena']; running=r['roots']
        for event in r['events']:
            running+=event['delta']; self.assertEqual(running,event['owned_upper'])
            self.assertGreaterEqual(running,r['roots'])
        self.assertEqual(running,r['roots'])
        self.assertEqual(r['owned_upper'],110559088)
        self.assertEqual(r['owned_upper'],r['roots']+r['maps']+r['sparse_roots']+r['frame']+r['vector_peak'])
        self.assertEqual(r['block_count_upper'],266)
        self.assertLess(r['block_count_upper'],4096)

    def test_descriptor_and_frame_seal_do_not_overlap_large_vectors(self):
        events=self.r['conservative_global_arena']['events']
        self.assertEqual(next(e['owned_upper'] for e in events if e['event']=='descriptor 7 seal overlap'),1205456)
        self.assertEqual(next(e['owned_upper'] for e in events if e['event']=='frame seal overlap'),1923072)
        self.assertEqual(next(e['owned_upper'] for e in events if e['event']=='objective 1 factor output'),56059608)
        self.assertEqual(next(e['owned_upper'] for e in events if e['event']=='consume dual nodes'),79204128)

    def test_tighter_factor_bound_and_max_admitted_frame(self):
        self.assertEqual(self.r['scope'],dict(pair_nodes=7592,pair_factors=9400,tighter_global_factors=25648))
        self.assertEqual(self.r['tighter_global_arena']['owned_upper'],110533488)
        self.assertEqual(self.r['any_admitted_frame_arena']['owned_upper'],114390448)
        self.assertTrue(self.r['any_admitted_frame_arena']['owned_upper_fits'])
        with self.assertRaises(ValueError): a.arena(self.r['inventory'],self.r['scope'],frame_override=a.FRAME+1)

    def test_candidate_envelope_is_not_an_actual_overflow(self):
        n=self.r['native_stream']; g=self.r['general_stream']
        self.assertEqual(n['per_pair_frame_upper'],[362944,362944,90832])
        self.assertEqual(n['stream_upper'],22868160)
        self.assertEqual(g['largest_frame_upper'],44945344)
        self.assertEqual(g['stream_upper'],2831559360)
        for r in (n,g):
            self.assertFalse(r['stream_upper_fits']); self.assertIsNone(r['actual_bytes'])
            self.assertFalse(r['upper_overflow_proves_actual_overflow'])
        self.assertTrue(n['frame_upper_fits']); self.assertFalse(g['frame_upper_fits'])
        self.assertEqual(n['endpoints'],504); self.assertEqual(n['roles'],1008)

    def test_full_support_floor_and_scope_invariant(self):
        r=self.r['support_work_floor']; scoped=r['pair_scoped']; dense=r['dense_reference']
        self.assertEqual(scoped['merge_floor'],1+3+3+1+(2+1+1))
        self.assertEqual(scoped['one_norm_floor'],3+1+1)
        self.assertEqual(dense['merge_floor'],1+1+1+1+1)
        self.assertEqual(dense['one_norm_floor'],1+1+1)
        self.assertEqual(scoped['work_lower'],17031168)
        self.assertEqual(dense['work_lower'],8515584)
        for k in (scoped,dense):
            self.assertTrue(k['exceeds_work_cap']); self.assertTrue(k['requires_all_endpoint_candidate_pairs_present'])
            self.assertIn('pullbacks',k['omitted'])
        # One lower endpoint count or a missing record cannot substitute for the
        # registered complete plan. The helper is arithmetic, not an acceptor.
        self.assertEqual(self.r['inventory']['endpoints_min'],28*9)
        self.assertEqual(self.r['scope_rebuilds_upper'],84)
        self.assertEqual(self.r['old_pair_once_count'],28)

    def test_invalid_counts_rejected(self):
        for count in (0,-1,True,1.0):
            with self.assertRaises(ValueError): a.minimum_support_work(count,252,True)
            with self.assertRaises(ValueError): a.frame_size(12,count,0)
        for profile in (-1,2,True):
            with self.assertRaises(ValueError): a.frame_size(12,8,profile)
        with self.assertRaises(ValueError): a.frame_size(12,9,0)
        with self.assertRaises(ValueError): a.minimum_support_work(3072,252,1)
        scope=copy.deepcopy(self.r['scope']); scope['pair_nodes']=100000
        with self.assertRaises(ValueError): a.arena(self.r['inventory'],scope)

    def test_static_only_no_admission_or_runtime_import(self):
        self.assertFalse(self.r['execution_admitted']); self.assertFalse(self.r['production_changed'])
        self.assertFalse(self.r['complete_cost_closed']); self.assertIsNone(self.r['new_quota'])
        self.assertIsNone(self.r['complete_work_upper']); self.assertIsNone(self.r['complete_rss_upper'])
        for name in ('source_builds','model_runs','gpu_runs','solver_calls','large_allocations'):
            self.assertEqual(self.r[name],0)
        tree=ast.parse(Path(a.__file__).read_bytes())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [n.name for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names]
        self.assertFalse(any(n and n.startswith(('scoped_source','torch','numpy','scipy','act.')) for n in imports))
        self.assertFalse(any(n.startswith(('scoped_source.','torch','numpy','scipy')) for n in sys.modules))
        self.assertEqual(self.r['unchanged_caps'],dict(arena=134217728,frame=4194304,
                        candidates=16769024,work=8000000,seconds=300))


if __name__=='__main__': unittest.main()
