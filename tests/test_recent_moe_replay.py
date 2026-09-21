import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recent_moe_dual_rs_replay.py"
spec = importlib.util.spec_from_file_location("recount", SCRIPT)
recount = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recount)


class ReplayControls(unittest.TestCase):
    def test_strict_radius_and_abstention(self):
        with tempfile.TemporaryDirectory(prefix="dual-recount-") as d:
            root = Path(d)
            (root / "r.tsv").write_text("predict\tradius\n0\t1\n-1\t4\n1\t0.5\n")
            np.save(root / "e.npy", np.array([[2., 0., 99.], [2., 3., 99.], [0., 1., 99.]]))
            final, expert = recount.composed(root, "r.tsv", "e.npy", 2)
            np.testing.assert_array_equal(final, [1, 0, .5])
            self.assertEqual(expert.shape, (3, 2))
            self.assertEqual(recount.percentages(final, [1]), [0])

    def test_reject_auxiliary_column_as_expert(self):
        with tempfile.TemporaryDirectory(prefix="dual-recount-") as d:
            root = Path(d)
            (root / "r.tsv").write_text("predict\tradius\n2\t1\n")
            np.save(root / "e.npy", np.array([[1., 2., 99.]]))
            with self.assertRaisesRegex(ValueError, "invalid expert"):
                recount.composed(root, "r.tsv", "e.npy", 2)

    def test_reject_row_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="dual-recount-") as d:
            root = Path(d)
            (root / "r.tsv").write_text("predict\tradius\n0\t1\n")
            np.save(root / "e.npy", np.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, "alignment"):
                recount.composed(root, "r.tsv", "e.npy", 2)

    def test_reject_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix="dual-recount-") as d:
            root = Path(d)
            (root / "r.tsv").write_text("predict\tradius\n0\tnan\n")
            np.save(root / "e.npy", np.ones((1, 2)))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                recount.composed(root, "r.tsv", "e.npy", 2)


if __name__ == "__main__":
    unittest.main()
