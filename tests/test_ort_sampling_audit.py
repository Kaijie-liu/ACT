"""Fail-closed contracts for the ORT sampling probe script."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_nn4sys_ort_containment.py"
SPEC = importlib.util.spec_from_file_location("audit_nn4sys_ort_containment", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class TestOrtAuditInputContract(unittest.TestCase):
    def test_native_rank_one_shape_is_preserved(self):
        self.assertEqual(audit._resolve_ort_input_shape((12296,), 12296), (12296,))

    def test_single_symbolic_dimension_is_inferred_without_rank_change(self):
        self.assertEqual(audit._resolve_ort_input_shape(("batch", 1), 1), (1, 1))

    def test_static_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "refusing reshape guess"):
            audit._resolve_ort_input_shape((3, 4), 13)

    def test_multiple_symbolic_dimensions_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "multiple unknown"):
            audit._resolve_ort_input_shape(("batch", "features"), 96)

    def test_dtype_is_taken_from_ort_contract(self):
        self.assertEqual(audit._ort_numpy_dtype("tensor(float)"), np.dtype(np.float32))
        self.assertEqual(audit._ort_numpy_dtype("tensor(double)"), np.dtype(np.float64))
        with self.assertRaises(NotImplementedError):
            audit._ort_numpy_dtype("tensor(int64)")


class TestOrtAuditResultLoading(unittest.TestCase):
    def test_directory_mode_unions_instance_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "per_instance_nn4sys_a.json").write_text(
                '{"per_instance":[{"official_instance_id":105,"cli_normalized":"CERTIFIED"}]}'
            )
            (root / "per_instance_nn4sys_b.json").write_text(
                '{"per_instance":[{"official_instance_id":106,"cli_normalized":"CERTIFIED"}]}'
            )
            loaded = audit._load_per_instance(root)
            self.assertEqual(set(loaded), {105, 106})


if __name__ == "__main__":
    unittest.main(verbosity=2)
