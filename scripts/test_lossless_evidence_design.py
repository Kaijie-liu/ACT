"""Static design checks, not implementation/serialization/performance tests."""
import ast
from fractions import Fraction as F
import itertools
import json
import struct
import sys
import unittest
from unittest.mock import patch

from scripts import analyze_lossless_evidence_design as a


class Design(unittest.TestCase):
    def test_envelope_arithmetic_and_explicit_reserves(self):
        r=a.analyze(); one=r['single']; two=r['two_retained']
        self.assertEqual(one['components_per_copy'],dict(source=59909824,forest=18323200,
            raw_proof=11406592,exact_receipts_reserve=8388608,
            checker_code_execution_metadata_reserve=8388608,
            bounded_logs_failure_progress_terminal_reserve=4194304))
        self.assertEqual((one['total_bytes_upper'],one['headroom_bytes']),(110611136,23606592))
        self.assertEqual(two['total_bytes_upper'],221222272)
        self.assertTrue(one['within_retained_cap']); self.assertFalse(two['within_retained_cap'])
        self.assertEqual(one['exact_receipt_payload_upper'],4568768)
        self.assertLess(one['exact_receipt_payload_upper'],one['components_per_copy']['exact_receipts_reserve'])
        self.assertTrue(one['reserves_are_proposed_not_measured'])

    def test_all_logical_roles_and_roots_remain(self):
        r=a.analyze(); one=r['single']; v=r['inventory']
        self.assertEqual((v['property_duties'],v['endpoints_upper'],one['candidate_roles']),(252,504,1008))
        self.assertEqual(one['logical_dual_values_checked'],2721600)
        self.assertEqual(one['dense_two_role_f64_payload'],21772800)
        self.assertGreater(one['dense_two_role_f64_payload'],16*a.MIB)
        self.assertLess(one['components_per_copy']['raw_proof'],16*a.MIB)
        self.assertEqual(one['forest_components']['logical_roots'],16*136232)
        self.assertTrue(r['exactness_contract']['all_explicit_zero_values_checked'])

    def test_sequential_uniqueness_lemma_symbolic_cases(self):
        # Symbolic IDs only: no actual network/source is built. Each ReLU
        # output either retains its different parent or creates a fresh ID.
        parent=(0,1,2,3)
        for states in itertools.product(('active','inactive','unstable'),repeat=4):
            next_id=4; out=[]
            for old,kind in zip(parent,states):
                if kind=='active': out.append(old)
                else: out.append(next_id); next_id+=1
            self.assertEqual(len(set(out)),len(out))
        # Arbitrary DAG fan-out invalidates this premise; sums need not fit F64.
        q=F(1)+F(1,2**53)
        self.assertNotEqual(F(float(q)),q)

    def test_f64_value_exactness_not_general_rational_rounding(self):
        for x in (0.,-0.,.5,sys.float_info.max,-sys.float_info.max,
                  float.fromhex('0x0.0000000000001p-1022')):
            b=struct.pack('<d',x); y=struct.unpack('<d',b)[0]
            self.assertEqual(F(x),F(y)); self.assertEqual(struct.pack('<d',y),b)
        self.assertNotEqual(struct.pack('<d',0.),struct.pack('<d',-0.))
        self.assertNotEqual(F(float(F(1,3))),F(1,3))
        self.assertGreater((2**8192-1).bit_length(),4096)
        # A result needs general exact-integer encoding, not F64 or the
        # producer candidate's 4096-bit input gate.
        self.assertLessEqual(1+2+2+2*1024,2064)

    def test_named_native_payload_does_not_admit_memory(self):
        n=a.analyze()['native_named_payload']
        self.assertEqual(n['one_coo'],161585280)
        self.assertEqual(n['persistent_coo_and_transpose'],323170560)
        self.assertEqual(n['coalesce_named_arrays_and_sort_branch'],592479360)
        self.assertIsNone(n['total_native_peak_upper']); self.assertIsNone(n['total_rss_upper'])

    def test_no_policy_or_execution_change(self):
        r=a.analyze()
        for k in ('new_work_quota','new_absolute_ceiling','walltime_prediction'):
            self.assertIsNone(r[k])
        self.assertFalse(r['runtime_policy_changed']); self.assertEqual(r['request_budget_seconds'],300)
        for k in ('new_source_runs','model_loads','native_calls','solver_calls','gpu_calls'):
            self.assertEqual(r[k],0)
        for arm in ('single','two_retained'):
            for k in ('production_caps_changed','implementation_admitted','execution_admitted'):
                self.assertFalse(r[arm][k])
        adapter=r['legacy_checker_adapter']
        self.assertFalse(adapter['solves_peak_memory'])
        self.assertTrue(adapter['extra_source_pass_must_be_charged'])
        self.assertFalse(adapter['old_full_path_work_upper_reusable_without_recount'])

    def test_wrong_shape_copy_policy_and_parent_refused(self):
        v=a.analyze()['inventory']; v['expert_widths']=[3072,256,129,10]
        with self.assertRaisesRegex(ValueError,'sequential'): a.envelope(v)
        v=a.analyze()['inventory']
        for copies in (0,3,True,1.0):
            with self.assertRaisesRegex(ValueError,'copies'): a.envelope(v,copies=copies)
        with patch.object(a,'PARENT_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'identity'): a.analyze()

    def test_static_only_imports(self):
        tree=ast.parse((a.ROOT/'scripts/analyze_lossless_evidence_design.py').read_bytes())
        imports=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom): imports.add(node.module)
        self.assertEqual(imports,{'argparse','hashlib','json','pathlib','scripts.analyze_full_resource_ledger'})
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))

    def test_saved_report(self):
        saved=a.ROOT/'docs/hz_lossless_evidence_design_20261014.json'
        self.assertEqual(json.loads(saved.read_bytes()),a.analyze())


if __name__=='__main__': unittest.main()
