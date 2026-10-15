"""Static ledger tests only; no source construction, workers or native imports."""
import ast
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest.mock import patch

from scripts import assess_direct_lifecycle_cost as a


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = a.analyze()

    def test_code_binding_and_report_status(self):
        a.verify_bindings(self.r['anchors'])
        self.assertEqual(self.r['based_on'], a.BASE)
        with self.assertRaisesRegex(ValueError, 'implementation drift'):
            a.verify_bindings({'scoped_source/direct_supervised.py': '0'*64})

    def test_independent_full_work_refusal_both_consumers(self):
        f = self.r['conditional_complete_work_floors']
        self.assertEqual(f['dense']['total'], 37085184)
        self.assertEqual(f['pair_scoped']['total'], 54116352)
        for row in f.values():
            self.assertEqual(row['source_first_affine'], 20054016)
            self.assertEqual(row['total'], row['source_first_affine'] + row['two_consumers'])
            self.assertGreater(row['total'], self.r['barriers']['work_cap'])
        self.assertEqual(self.r['source_core_upper']['total'], 140116400)

    def test_historical_capacity_adapters_explicitly_checked(self):
        migration = self.r['historical_capacity_migration']
        self.assertFalse(migration['old_anchor_suite_passes_current'])
        self.assertTrue(migration['old_results_unchanged'])
        self.assertTrue(migration['consumer_loops_and_arena_unchanged'])
        self.assertEqual(len(migration['changed_dependencies']), 2)
        with patch.object(a.subprocess, 'check_output', return_value=b'incorrect frozen source'):
            with self.assertRaisesRegex(ValueError, 'blob identity'):
                a.capacity_migration(a.load(a.CAPACITY))

    def test_fixed_stub_roots_and_dual_roster_counts(self):
        self.assertEqual(a.lease_work(10, [2, 4], [3, 5], 'dense'),
                         dict(row_visits=10, coefficient_visits=80, arithmetic=8))
        self.assertEqual(a.lease_work(10, [2, 4], [3, 5], 'pair_scoped'),
                         dict(row_visits=10, coefficient_visits=47, arithmetic=8))
        counts = self.r['fixed_proposer_only_upper']['arms']
        self.assertEqual(counts['dense']['coefficient_visits'] - 504*2*2700, 19660032)
        self.assertEqual(counts['pair_scoped']['coefficient_visits'] - 504*2*2700, 11088)
        self.assertEqual(self.r['fixed_proposer_only_upper']['totals']['pair_scoped'], 2738232)

    def test_tuple_length_hint_is_not_free_for_dense_mapping(self):
        # Only reproduce the standard-library traversal contract, no HZ or LP.
        class Slots(Mapping):
            def __init__(self): self.reads = 0
            def __iter__(self):
                for i in range(11):
                    self.reads += 1
                    if i in (2, 8): yield i
            def __len__(self): return sum(1 for _ in self)
            def __getitem__(self, key):
                self.reads += 2
                if key not in (2, 8): raise KeyError(key)
                return key
        self.assertEqual(sys.implementation.name, 'cpython')
        self.assertEqual(sys.version_info[:2], (3, 12))
        slots = Slots(); keys = tuple(MappingProxyType(slots))
        self.assertEqual(keys, (2, 8)); self.assertEqual(slots.reads, 22)
        for key in keys: slots[key]
        self.assertEqual(slots.reads, 26)

    def test_archived_increment_complete_and_same_mathematics(self):
        rows = self.r['archived_increment_checks']
        self.assertEqual(len(rows), 16)
        expected = {'mixed': (24,18,924,222), 'deep': (24,18,1248,330),
                    'half': (26,21,454,164), 'guarded': (96,72,1488,312)}
        for row in rows:
            r, m, dense, scoped = expected[row['case']]
            self.assertEqual(row['delta'], dict(row_visits=r, arithmetic=m,
                coefficient_visits=dense if row['arm']=='dense' else scoped))

    def test_archived_omission_relabel_and_bound_mutation_refused(self):
        old = a.load(a.V1); new = a.load(a.V2)
        for fault in ('missing', 'duplicate', 'result', 'count'):
            broken = deepcopy(new)
            if fault == 'missing': broken['differentials'].pop()
            elif fault == 'duplicate': broken['differentials'][1] = broken['differentials'][0]
            elif fault == 'result': broken['differentials'][0]['exact_result_sha256'] = '0'*64
            else: broken['differentials'][0]['work']['row_visits'] += 1
            with self.assertRaises(ValueError): a.archived_delta(old, broken)

    def test_full_online_read_schedule_not_v1(self):
        io = self.r['io']
        self.assertEqual(io['coefficients'], dict(T=18, X=3, S=9, C=9))
        self.assertEqual(len(io['events']), 15)
        parent = [e for e in io['events'] if e['phase'].startswith('parent')]
        self.assertEqual(sum(e['reads'].get('T', 0) for e in parent), 7)
        self.assertEqual(sum(e['reads'].get('S', 0) for e in parent), 4)
        self.assertEqual(io['registered_tensor_terms_only'], 18*55715520+3*24576)
        self.assertIn('offline audit and relocation', io['omitted'])

    def test_whole_storage_counts_originals_and_metadata(self):
        sizes = dict.fromkeys(a.STORAGE, 0)
        sizes.update(T=100, H=10, S=30, C=5, V=7)
        self.assertEqual(a.retained_bytes(sizes), 262)
        self.assertEqual(self.r['storage']['dense_first_weights_subcase_lower'], 191647104)
        self.assertGreater(self.r['storage']['dense_first_weights_subcase_lower'], 128*2**20)
        self.assertEqual(self.r['storage']['outer_metadata_reserve'], 3276800)
        self.assertIsNone(self.r['storage']['full_archive_upper'])

    def test_limits_and_control_only_dispatch(self):
        b = self.r['barriers']
        self.assertEqual(b['memberships_lower'], 126632)
        self.assertEqual(b['memberships_cap'], 65536)
        self.assertEqual(b['worker_single_file_cap'], 1048576)
        self.assertEqual(b['allowed_cases'], ['half', 'guarded'])
        self.assertFalse(b['real_or_native_dispatch'])
        self.assertTrue(b['binary64_source_coefficients_only'])

    def test_lifetime_and_clock_do_not_claim_full_rss(self):
        v = self.r['lifetime']
        self.assertFalse(v['production_graph_retained_during_replay'])
        self.assertTrue(v['production_owner_released_before_replay'])
        self.assertTrue(v['worker_cleaned_before_parent_receive'])
        self.assertFalse(v['python_allocator_rss_release_proven'])
        self.assertIsNone(v['complete_rss_upper'])
        self.assertFalse(self.r['clock']['guaranteed_return_within_300'])
        self.assertEqual(self.r['clock']['budget'], 300)

    def test_invalid_arithmetic_input_refused(self):
        for value in (-1, True, 1.0):
            with self.assertRaises(ValueError): a.lease_work(10, [value], [2], 'dense')
        with self.assertRaises(ValueError): a.lease_work(10, [5], [2], 'dense')
        with self.assertRaises(ValueError): a.lease_work(10, [2], [], 'dense')
        with self.assertRaises(ValueError): a.lease_work(10, [2], [1], 'gpu')
        with self.assertRaises(ValueError): a.retained_bytes({'T': 100})
        bad = dict.fromkeys(a.STORAGE, 0); bad['T'] = -1
        with self.assertRaises(ValueError): a.retained_bytes(bad)

    def test_no_execution_import_or_admission(self):
        tree = ast.parse(Path(a.__file__).read_bytes())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names]
        for prefix in ('scoped_source', 'scoped_proof', 'act', 'torch', 'numpy', 'scipy'):
            self.assertFalse(any(n and n.startswith(prefix) for n in imports))
            self.assertFalse(any(n == prefix or n.startswith(prefix+'.') for n in sys.modules))
        for name in ('source_builds', 'candidate_generations', 'model_runs', 'gpu_runs', 'solver_calls'):
            self.assertEqual(self.r[name], 0)
        self.assertFalse(self.r['runtime_changes']); self.assertFalse(self.r['execution_admitted'])
        self.assertFalse(self.r['complete_cost_closed']); self.assertIsNone(self.r['complete_work_upper'])
        self.assertIsNone(self.r['new_quota'])


if __name__ == '__main__':
    unittest.main()
