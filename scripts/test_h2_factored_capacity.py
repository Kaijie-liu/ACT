"""Metadata-only controls: no source tensor, model, matrix or solver is created."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Direct execution under -I -S, without importing the ACT package initializer.
PATH = Path(__file__).with_name('audit_h2_factored_capacity.py')
SPEC = importlib.util.spec_from_file_location('factored_capacity_audit', PATH)
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


class FactoredCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = m.derive()
        cls.raw = {n:(m.ROOT/n).read_bytes() for n in m.FILES}

    def test_registered_shape_counts(self):
        r = self.result; d = r['recipe_counts']
        self.assertEqual((r['parameters'], r['parameter_tensors']), (6961368, 52))
        self.assertEqual((d['source_nodes'], d['pair_variables'], d['pair_affine_equalities']), (9560, 4892, 924))
        self.assertEqual((d['pair_relu_nodes'], d['pair_guard_rows']), (896, 12))
        self.assertEqual((d['duties'], d['maximum_endpoint_LPs']), (252, 504))
        self.assertEqual(d['pair_affine_entries_if_dense'], 2036124)
        self.assertEqual(d['pair_relu_entries_if_all_unstable'], 4480)
        self.assertEqual(len(r['models']), 3)

    def test_64_byte_protocol_definite_rejection(self):
        v = self.result['frozen_64_byte_control_policy']
        self.assertEqual(v['chunks'], 870561)
        self.assertEqual(v['minimum_bytes_per_chunk_reference'], 125)
        self.assertEqual(v['source_header_ref_bytes_lower_bound'], 108820125)
        self.assertTrue(v['source_header_definitely_rejected'])
        self.assertFalse(v['full_path_admitted'])

    def test_default_is_only_necessary_not_full_admission(self):
        r = self.result; v = r['existing_1_mib_default_policy']; s = r['default_manifest_structure']
        self.assertEqual((v['chunks'], v['raw_tensor_bytes_with_center']), (95, 55715520))
        self.assertEqual(v['largest_tensor_elements'], 786432)
        self.assertFalse(v['source_header_definitely_rejected'])
        self.assertEqual(s['tensor_descriptor_list_bytes'], 18045)
        self.assertEqual(s['portable_members_excluding_outer_manifest'], 9698)
        self.assertEqual(s['portable_envelope_bytes_upper_bound'], 1136168)
        self.assertTrue(s['portable_envelope_necessary_test_passed'])
        for key in ('actual_source_header_bytes','actual_inner_proof_header_bytes',
                    'actual_node_member_bytes','actual_pair_member_bytes','actual_bundle_bytes'):
            self.assertIsNone(s[key])
        self.assertFalse(r['full_path_admitted'])
        self.assertEqual(r['status'], 'NO_REAL_FREEZE_CAPACITY_NOT_ADMITTED')
        self.assertFalse(r['policy']['supervisor']['real_intake_supported_by_protocol'])

    def test_per_tensor_rounding_and_limits(self):
        limits = self.result['policy']
        # 32+40 bytes fit one 80-byte aggregate, but tensors cannot share a chunk.
        v = m.chunk_summary({'a':[4], 'b':[5]}, 80, limits)
        self.assertEqual(v['chunks'], 2)
        self.assertEqual(v['raw_tensor_bytes_with_center'], 72)
        self.assertFalse(m.chunk_summary({'a':[1000001]}, 2**20, limits)['tensor_element_necessary_test_passed'])
        for invalid in (True, 0, 7, 65, 2**21):
            with self.assertRaises(ValueError): m.chunk_summary({'a':[1]}, invalid, limits)
        with self.assertRaises(ValueError):
            m.manifest_structure(self.result['tensor_shapes_including_center'], [], 28, 64, limits, ())

    def test_small_manifest_against_literal_encoding(self):
        p = self.result['policy']
        result = m.manifest_structure({'@center':[1, 2]}, ['input/0','input/1'], 1, 8, p, ())
        chunks = [{'file':f't/00000-{j:05d}.bin','bytes':8,'sha256':'a'*64,'offset':j*8} for j in range(2)]
        descriptors = [{'name':'@center','dtype':'torch.float64','shape':[1,2],'byte_order':'little','chunks':chunks}]
        self.assertEqual(result['tensor_descriptor_list_bytes'], len(m.compact(descriptors)))
        self.assertEqual(result['portable_members_excluding_outer_manifest'], 8)

    def test_no_dynamic_or_unbounded_policy_evaluation(self):
        self.assertEqual(m.integer(ast.parse('2*2**30', mode='eval').body), 2**31)
        for bad in ('True','-1','int(64)','x','2**100000','2**40*2','1/2'):
            with self.assertRaises(ValueError): m.integer(ast.parse(bad, mode='eval').body)
        for source in ('MEMBER_LIMIT = f()', 'MEMBER_LIMIT = 1\nMEMBER_LIMIT = 2', 'pass'):
            with self.assertRaises(ValueError): m.assignments(source, ('MEMBER_LIMIT',))

    def test_missing_or_changed_callsite_rejected(self):
        raw = dict(self.raw)
        raw[m.SOURCE] = raw[m.SOURCE].replace(b'chunk_bytes=BLOCK_LIMIT', b'chunk_bytes=64')
        with self.assertRaisesRegex(ValueError, 'pack default changed'): m.policy(raw)
        raw = dict(self.raw)
        raw[m.SUPERVISOR] = raw[m.SUPERVISOR].replace(b'0 < budget <= 300', b'0 < budget <= 600')
        with self.assertRaisesRegex(ValueError, 'contract changed'): m.policy(raw)
        raw = dict(self.raw)
        raw[m.VERIFY] = raw[m.VERIFY].replace(b'MEMBER_LIMIT=16*2**20', b'MEMBER_LIMIT=32*2**20')
        with self.assertRaisesRegex(ValueError, 'mismatched'): m.policy(raw)

    def test_recipe_mismatch_and_no_non_mlp_inference(self):
        cfg = json.loads(self.raw[m.TRAINING])['training']
        for key, value in (('dataset','MNIST'),('gate','hard_top1'),('top_k',1),('router_hidden',[True])):
            bad = {**cfg, key:value}
            with self.assertRaises(ValueError): m.shapes_from_recipe(bad)
        self.assertEqual(self.result['conv_status'], 'NOT_ADMITTED_UNSUPPORTED_CONV_POOL_CAPTURE')

    def test_inventory_member_and_factory_contracts(self):
        for file, before, after in (
                (m.IO, b'total_limit=TOTAL_LIMIT', b'total_limit=123'),
                ('scoped_source/factored_check.py', b'inventory(root,expected)', b'inventory(root,expected,999)'),
                ('scoped_source/factored_portable.py', b'0<size<=MEMBER_LIMIT', b'0<size'),
                ('act/back_end/moe/factory.py', b'index + 1 < len(widths) - 1', b'index < len(widths) - 1')):
            raw = dict(self.raw)
            self.assertIn(before, raw[file]); raw[file] = raw[file].replace(before, after)
            with self.assertRaises(ValueError): m.policy(raw)

    def test_metadata_mismatch_and_protocol_mutation(self):
        original = Path.read_bytes
        changed = json.loads(self.raw[m.SELECTION]); changed['models']['seed1']['model_state']['parameter_count'] += 1
        def reader(path):
            return json.dumps(changed).encode() if path == m.ROOT/m.SELECTION else original(path)
        with patch.object(Path,'read_bytes',reader):
            with self.assertRaisesRegex(ValueError, 'metadata versus recipe'): m.derive()
        raw = dict(self.raw); cfg = json.loads(raw[m.CONFIG]); cfg['source_chunk_bytes'] = 2**20
        raw[m.CONFIG] = json.dumps(cfg).encode()
        with self.assertRaisesRegex(ValueError, 'synthetic protocol changed'): m.policy(raw)

    def test_exact_read_allowlist_and_no_external_imports(self):
        original = Path.read_bytes; observed = set(); allowed = {m.ROOT/n for n in m.FILES}
        before = set(sys.modules)
        def reader(path):
            self.assertIn(path, allowed); observed.add(path); return original(path)
        with patch.object(Path,'read_bytes',reader): result = m.derive()
        self.assertEqual(observed, allowed)
        self.assertFalse({'act','torch','numpy','scipy','highspy'} & (set(sys.modules)-before))
        self.assertEqual((result['new_model_loads'],result['new_solves'],result['real_requests_started']), (0,0,0))

    def test_residency_is_not_fake_ram_or_performance(self):
        r = self.result
        self.assertEqual(len(r['residency_by_stage']), 14)
        for phase in r['residency_by_stage']:
            self.assertIn(phase['file'], r['source_bindings'])
            self.assertGreaterEqual(phase['last_line'], phase['first_line'])
            self.assertIsNone(phase['measured_peak_bytes'])
        self.assertFalse(r['performance_measurement'])
        self.assertFalse(r['mathematical_certificate'])
        self.assertTrue(r['remaining_unknowns'])


if __name__ == '__main__':
    unittest.main()
