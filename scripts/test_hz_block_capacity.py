"""Static metadata and call-structure controls, no numerical library imports."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import audit_hz_block_capacity as a


class BlockCapacity(unittest.TestCase):
    def test_inventory_reads_only_explicit_repo_sources(self):
        seen = []; original = Path.read_bytes
        def read(path):
            name = path.relative_to(a.ROOT).as_posix()
            self.assertIn(name, a.FILES); seen.append(name)
            return original(path)
        with patch.object(Path, 'read_bytes', read): a.derive()
        self.assertEqual(set(seen), set(a.FILES))

    def test_complete_duties_and_template_work(self):
        d = a.derive()['dense_recipe']
        self.assertEqual((d['pairs'], d['duties'], d['endpoints_min'], d['endpoints_max']), (28, 252, 252, 504))
        self.assertEqual((d['pair_queries_min'], d['pair_queries_max']), (9, 18))
        self.assertEqual((d['old_per_pair_expert_propagations'], d['template_expert_propagations']), (56, 8))
        self.assertEqual((d['old_layer_records'], d['template_layer_records']), (340, 52))

    def test_snapshots_are_not_removed_constraints(self):
        d = a.derive()['dense_recipe']; r = d['all_unstable_row_scenario']
        self.assertEqual(d['support_snapshot_occurrences'], {'common': 28, 'entry': 28, 'expert_templates': 56, 'total': 112})
        self.assertEqual((r['stored_rows_in_four_snapshots'], r['executed_local_block_rows']), (3852, 2700))
        self.assertEqual(r['repeated_stored_prefix_rows'], 1152)
        self.assertFalse(d['old_factor_count_is_bound_for_new_lifts'])
        self.assertIsNone(d['new_lift_factors'])

    def test_reparse_pattern_not_assumed_after_code_change(self):
        raw = (a.ROOT/'scoped_source/check_hz_row_enclosure.py').read_bytes()
        self.assertEqual(a.repeated_parse_contract(raw)['whole_reference_parses_for_R_rows'], 'R+1')
        changed = raw.decode().replace('s,c,b=parse(reference)', 's,c,b=prepared(reference)')
        with self.assertRaises(ValueError): a.repeated_parse_contract(changed)
        changed = raw.decode().replace('ref,cm,bm=projected(reference,kind,i)', 'ref,cm,bm=prepared(reference,kind,i)')
        with self.assertRaises(ValueError): a.repeated_parse_contract(changed)

    def test_roster_and_literal_mutations_rejected(self):
        d = copy.deepcopy(a.prior.derive()['dense_recipe']); d['pairs'] -= 1
        with self.assertRaises(ValueError): a.topology_counts(d)
        raw = 'def check(queries):\n if not 1 <= len(queries) <= 8: raise ValueError()\n'
        a.prior.comparison(raw, 'check', '1 <= len(queries) <= 8')
        with self.assertRaises(ValueError): a.prior.comparison(raw, 'check', '1 <= len(queries) <= 18')

    def test_definite_and_unknown_limits_stay_separate(self):
        r = a.derive()
        self.assertEqual(r['definite_refusals']['input_factors_and_output_rows'], 3072)
        self.assertEqual(r['definite_refusals']['portable_bytes']['parameter_base64_lower_bound'], 74254592)
        self.assertFalse(r['wide_rows']['actual_wide_row_failure_observed'])
        self.assertIsNone(r['wide_rows']['actual_trained_row_widths'])
        self.assertEqual(r['wrapper_integration']['new_schema'], 'CHECKED_HZ_BLOCK_SOURCE_V1')
        self.assertFalse(r['wrapper_integration']['old_portable_has_block_checker'])
        for k in ('new_model_loads','new_input_selections','new_propagations','new_solves','cuda_calls'):
            self.assertEqual(r[k], 0)
        self.assertFalse(r['source_or_output_certificate'])
        self.assertTrue(r['all_six_goal_gates_remain_open'])

    def test_deterministic(self): self.assertEqual(a.derive(), a.derive())

    def test_stdlib_and_metadata_imports_only(self):
        for name in ('scripts/audit_hz_block_capacity.py', 'scripts/audit_hz_real_intake.py'):
            tree = ast.parse((a.ROOT/name).read_bytes())
            imports = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            imports |= {v.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names}
            self.assertLessEqual(imports, {'argparse','ast','hashlib','json','math','pathlib','fractions','scripts'})


if __name__ == '__main__': unittest.main()
