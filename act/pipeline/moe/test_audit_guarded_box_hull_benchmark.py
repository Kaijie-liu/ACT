import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from act.pipeline.moe.audit_guarded_box_hull_benchmark import _artifact_issues
from act.pipeline.moe.benchmark_guarded_box_hull import (
    _bounds_sha256,
    _save_bounds_artifact,
)


class AuditGuardedBoxHullBenchmarkTests(unittest.TestCase):
    def test_artifact_recomputes_hashes_and_difference(self):
        arrays = {
            "highspy": (np.asarray([-1.0]), np.asarray([2.0])),
            "scipy": (np.asarray([-1.25]), np.asarray([2.5])),
        }
        with TemporaryDirectory(dir="/data1/Kane/MOE/cache/tmp") as directory:
            root = Path(directory)
            relative, digest = _save_bounds_artifact(root, "rank1:pair0-1", arrays)
            branch = {
                "bounds_artifact": relative,
                "bounds_artifact_sha256": digest,
                "bound_max_abs_diff": 0.5,
                "highspy": {"bounds_sha256": _bounds_sha256(*arrays["highspy"])},
                "scipy": {"bounds_sha256": _bounds_sha256(*arrays["scipy"])},
            }
            issues, difference = _artifact_issues(branch, root)
            self.assertEqual(issues, [])
            self.assertEqual(difference, 0.5)

    def test_artifact_detects_recorded_difference_mismatch(self):
        arrays = {
            "highspy": (np.asarray([-1.0]), np.asarray([2.0])),
            "scipy": (np.asarray([-1.0]), np.asarray([2.0])),
        }
        with TemporaryDirectory(dir="/data1/Kane/MOE/cache/tmp") as directory:
            root = Path(directory)
            relative, digest = _save_bounds_artifact(root, "rank1:pair0-1", arrays)
            branch = {
                "bounds_artifact": relative,
                "bounds_artifact_sha256": digest,
                "bound_max_abs_diff": 1.0,
                "highspy": {"bounds_sha256": _bounds_sha256(*arrays["highspy"])},
                "scipy": {"bounds_sha256": _bounds_sha256(*arrays["scipy"])},
            }
            issues, _ = _artifact_issues(branch, root)
            self.assertIn("recorded maximum bound difference mismatch", issues)


if __name__ == "__main__":
    unittest.main()
