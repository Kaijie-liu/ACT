"""Static controls only; no source producer, numeric library or device call."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import audit_hz_combined_capacity as a


class CombinedCapacity(unittest.TestCase):
    def test_explicit_inventory_only(self):
        seen = []; original = Path.read_bytes
        def read(path):
            name = path.relative_to(a.ROOT).as_posix()
            self.assertIn(name, a.FILES); seen.append(name)
            return original(path)
        with patch.object(Path, 'read_bytes', read): a.derive()
        self.assertEqual(set(seen), set(a.FILES))

    def test_producer_counts_include_enclosure(self):
        report = a.derive(); p = report['producer_bound']; d = p['objects']
        self.assertEqual(report['registered_recipe']['router_widths'], [3072, 128, 8])
        self.assertEqual(report['registered_recipe']['expert_widths'], [3072, 256, 128, 10])
        self.assertEqual([d[k]['total_factor_upper_bound'] for k in d], [3984, 6692, 9400, 25648])
        self.assertEqual((d['pair_common_once']['continuous_upper_bound'], d['pair_common_once']['binary_upper_bound']), (8504, 896))
        self.assertEqual((d['pair_common_once']['equality_upper_bound'], d['pair_common_once']['inequality_upper_bound']), (896, 1804))
        self.assertEqual((d['request_unique_factors_all_templates']['continuous_upper_bound'],
                          d['request_unique_factors_all_templates']['binary_upper_bound'],
                          d['request_unique_factors_all_templates']['total_constraint_row_upper_bound']), (22448, 3200, 9936))
        self.assertFalse(p['arbitrary_checker_accepted_package_bound'])
        self.assertIsNone(p['actual_nnz'])
        self.assertFalse(report['old_6684_scenario_is_new_producer_bound'])

    def test_duties_are_not_reduced_by_tiling(self):
        d = a.derive()['complete_obligations']
        self.assertEqual((d['pairs'], d['properties_per_pair'], d['duties']), (28, 9, 252))
        self.assertEqual((d['endpoints_min'], d['endpoints_max'], d['pair_queries_min'], d['pair_queries_max']), (252, 504, 9, 18))
        # Candidate capacity is local. Three tiles cover 18, not an 8-row truncation.
        self.assertEqual([min(8, 18-i) for i in range(0, 18, 8)], [8, 8, 2])

    def test_all_transitive_limits_remain(self):
        r = a.derive(); c = r['intersecting_capacity_contracts']
        self.assertEqual(c['old_source'], dict(factors=128, outputs=128, constraints=256, experts=4, classes=5))
        self.assertEqual(c['sparse_reference']['each_entire_vector_rows'], 16)
        self.assertEqual(c['sparse_reference']['target_factor_allowance'], 4128)
        self.assertFalse(c['sparse_reference']['chunk_size_is_whole_state_tiling'])
        self.assertTrue(c['full_producer_bound_exceeds_4096'])
        self.assertEqual(c['definite_recipe_refusals']['input_factors_and_output_rows'], 3072)
        self.assertEqual(r['source_schema'], 'CHECKED_SPARSE_HZ_BLOCK_SOURCE_V1')

    def test_storage_count_not_memory_claim(self):
        d = a.derive()['repeated_storage']
        self.assertEqual(d['support_snapshot_occurrences'], dict(common=28, entry=28, expert_templates=56, total=112))
        self.assertEqual(d['occurrences_per_expert'], 7)
        self.assertIsNone(a.derive()['actual_peak_memory'])

    def test_device_and_integration_not_admitted(self):
        r = a.derive(); d = r['device_boundary']
        for k in ('physical_block_kernel_implemented', 'flat_device_kernel_interchangeable',
                  'preproposal_capacity_cleared_by_gpu', 'source_entry_exposes_device'):
            self.assertFalse(d[k])
        self.assertEqual((d['updates'], d['dtype']), (128, 'float64'))
        self.assertEqual(r['integration']['conv_operators'], ['Conv2d', 'AvgPool2d'])
        self.assertEqual((r['integration']['conv_pairs'], r['integration']['conv_duties']), (6, 54))
        for k in ('new_model_loads', 'new_input_selections', 'new_propagations', 'new_solves', 'cuda_calls'):
            self.assertEqual(r[k], 0)
        self.assertFalse(r['source_or_output_certificate'])
        self.assertTrue(r['all_six_goal_gates_remain_open'])

    def test_topology_and_roster_mutations_refused(self):
        d = a.prior.prior.derive()['dense_recipe']
        for key, value in (('pairs', 27), ('duties', 251), ('input_dimensions', 1), ('classes', 9)):
            changed = copy.deepcopy(d); changed[key] = value
            with self.assertRaises(ValueError): a.producer_bounds(changed)
        changed = copy.deepcopy(d); changed['expert_widths'][-1] = 9
        with self.assertRaises(ValueError): a.producer_bounds(changed)

    def test_constant_and_literal_changes_refused(self):
        raw = 'MAX_FACTORS, MAX_ROWS, MAX_NNZ = 4096, 16, 262144\n'
        self.assertEqual(a.assignment(raw, '(MAX_FACTORS, MAX_ROWS, MAX_NNZ)'), (4096, 16, 262144))
        with self.assertRaises(ValueError): a.assignment(raw, 'OTHER')
        with self.assertRaises(ValueError): a.assignment(raw + raw, '(MAX_FACTORS, MAX_ROWS, MAX_NNZ)')
        code = 'def run(device):\n if device != "cpu": raise ValueError()\n'
        a.prior.prior.comparison(code, 'run', 'device != "cpu"')
        with self.assertRaises(ValueError): a.prior.prior.comparison(code, 'run', 'device != "cuda"')

    def test_source_device_signature_and_forwarding_changes_refused(self):
        raw = (a.ROOT/'scoped_source/hz_sparse_source.py').read_bytes().decode()
        self.assertFalse(a.source_device_contract(raw)['source_forwards_device'])
        for changed in (raw.replace('observe=None', 'observe=None, device=None'),
                        raw.replace('observe=None', 'observe=None, **kwargs'),
                        raw.replace('proof = propose_request(req,', 'proof = propose_request(req, device="cuda",')):
            with self.assertRaises(ValueError): a.source_device_contract(changed)

    def test_stdlib_and_metadata_imports_only(self):
        names = ('scripts/audit_hz_combined_capacity.py', 'scripts/audit_hz_block_capacity.py', 'scripts/audit_hz_real_intake.py')
        for name in names:
            tree = ast.parse((a.ROOT/name).read_bytes())
            imported = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            imported |= {v.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names}
            self.assertLessEqual(imported, {'argparse', 'ast', 'hashlib', 'json', 'math', 'pathlib', 'fractions', 'scripts'})

    def test_deterministic(self): self.assertEqual(a.derive(), a.derive())


if __name__ == '__main__': unittest.main()
