"""Standard-library metadata and exact-arithmetic diagnostic tests, no HZ work."""
import ast
from fractions import Fraction as F
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import audit_hz_real_intake as a


class IntakeTests(unittest.TestCase):
    def test_named_capacity_contract(self):
        code='def f(x):\n if not 1 <= x <= 8: raise ValueError()\n'
        self.assertEqual(a.comparison(code,'f','1 <= x <= 8')['line'],2)
        with self.assertRaises(ValueError):a.comparison(code,'f','1 <= x <= 9')

    def test_affine_witness(self):
        r=a.arithmetic_witnesses()
        required=F(1,2**104)+F(1,2**200)
        self.assertEqual(F(r['non_roundtrip_values']['affine_error_sum']),required)
        self.assertTrue(all(a.roundtrips(F(v)) for v in r['individual_affine_residuals']))
        self.assertFalse(a.roundtrips(required))

    def test_relational_arithmetic_not_closed(self):
        self.assertTrue(a.roundtrips(F(1,2**54)))
        self.assertFalse(a.roundtrips(F(1)-F(1,2**54)))
        self.assertFalse(a.roundtrips(F(1,2**56)-F(1,2)))
        self.assertFalse(a.roundtrips(F(1,2**1075)))
        self.assertFalse(a.roundtrips(F(2**2048)))

    def test_read_inventory(self):
        original=Path.read_bytes; seen=[]
        def read(path):
            self.assertIn(path.relative_to(a.ROOT).as_posix(),a.FILES)
            seen.append(path);return original(path)
        with patch.object(Path,'read_bytes',read):a.derive()
        self.assertEqual(len(seen),len(a.FILES))

    def test_hz_not_direct_node_counts(self):
        d=a.derive()['dense_recipe'];p=d['new_HZ_pair_scenario']
        self.assertEqual((d['pairs'],d['duties'],d['endpoint_targets_if_gate_nondegenerate']),(28,252,504))
        self.assertEqual((p['total_factors'],p['total_constraint_rows']),(6684,2700))
        self.assertEqual((p['continuous_factors'],p['binary_factors']),(5788,896))
        self.assertEqual((d['expert_propagations_in_current_all_pair_loop'],d['all_layer_trace_records']),(56,340))

    def test_source_bytes_and_scope(self):
        d=a.derive();self.assertEqual(d['source_bytes']['parameter_base64_lower_bound'],74254592)
        self.assertTrue(d['source_bytes']['source_lower_bound_exceeds_both'])
        self.assertEqual((d['conv']['all_pairs'],d['conv']['duties']),(6,54))
        self.assertFalse(d['source_or_output_certificate'])
        self.assertTrue(d['all_six_goal_gates_remain_open'])
        for k in ('new_model_loads','new_input_selections','new_propagations','new_solves','cuda_calls'):
            self.assertEqual(d[k],0)

    def test_deterministic_report(self):self.assertEqual(a.derive(),a.derive())

    def test_stdlib_only_imports(self):
        tree=ast.parse((a.ROOT/'scripts/audit_hz_real_intake.py').read_text())
        imports={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports|={v.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names}
        self.assertLessEqual(imports,{'argparse','ast','fractions','hashlib','json','math','pathlib'})


if __name__=='__main__':unittest.main()
