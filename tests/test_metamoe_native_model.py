import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from metamoe_native_model import load_component
from audit_metamoe_native_r4 import check_spec
from metamoe_component_native_r4 import parse_backend_result


class NativeControls(unittest.TestCase):
    def test_binding_rejected_before_pickle(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p = Path(tmp) / 'not_a_checkpoint'
            p.write_bytes(b'not a trusted pickle')
            with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
                load_component(tmp, p, '0' * 64)

    def test_late_backend_positive_rejected(self):
        value = parse_backend_result('Result: safe in 1 seconds\nFinal verified acc: 100% (total 1 examples)', 0, 301.)
        self.assertEqual(value['status'], 'TIMEOUT')

    def test_partial_property_rejected(self):
        with self.assertRaises(ValueError):
            check_spec('(declare-const X_0 Real)\n(assert (>= X_0 0))', 0)


if __name__ == '__main__':
    unittest.main()
