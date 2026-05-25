"""Regression test for SATSidecar → ACT canonical receipt bridge.

History: cersyve sidecar (HyZor SATSidecar) discovered 3/12 strict-FAL
witnesses (iid 1, 5, 9) but the artifacts lived in SATSidecar JSON
schema, NOT counted in ACT canonical capability. The bridge at
``scripts/bridge_sidecar_to_act_receipt.py`` translates them into
``fal_receipt.write_receipt`` format. These tests pin the contract:

  * Only ``sat_zero_tol`` artifacts get bridged (no false promotion)
  * Bridge re-validates ``in_input_domain`` + ``ast_holds`` fail-CLOSED
  * x_star sha256 is re-hashed; mismatch → rejected
  * Output filename follows ACT convention (no collision with
    fal_receipt.write_receipt; receipt_dir can mix native + bridged)
  * Emitted JSON loads cleanly via ``fal_receipt.load_receipt``
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

# Make the bridge importable without exec — scripts/ isn't a package.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bridge_sidecar_to_act_receipt",
    REPO / "scripts" / "bridge_sidecar_to_act_receipt.py",
)
bridge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge_mod)
bridge = bridge_mod.bridge
BridgeError = bridge_mod.BridgeError


def _make_fake_sidecar_dir(tmp: Path, artifacts: list) -> Path:
    """Build a minimal SATSidecar-shaped run dir from spec dicts."""
    # Tests call this multiple times with the same tmp root; mkdir on
    # the second call would error. mkdir-exist_ok is the right semantic
    # because the second call writes fresh artifacts into the same dir.
    d = tmp / "sidecar"
    d.mkdir(exist_ok=True)

    # Need a fake onnx file too (bridge re-hashes model_sha256).
    fake_model = d / "fake.onnx"
    fake_model.write_bytes(b"\x00\x01\x02ONNX-FAKE")
    import hashlib
    model_sha = hashlib.sha256(fake_model.read_bytes()).hexdigest()

    manifest_entries = []
    for i, a in enumerate(artifacts):
        verdict = a.get("verdict", "sat_zero_tol")
        x = a.get("x", np.array([0.1, 0.2, 0.3], dtype=np.float64))
        y = a.get("y", np.array([-0.05, 0.05], dtype=np.float64))
        iid = a.get("instance_id", i)
        attempt = a.get("attempt", 100 + i)
        source = a.get("source", "d0_random_uniform")
        stem = f"cersyve_{iid}_{source}_{attempt}"
        x_npy = d / f"{stem}.x_star.npy"
        y_npy = d / f"{stem}.y_ort.npy"
        np.save(x_npy, x)
        np.save(y_npy, y)
        art = {
            "artifact_schema_version": 1,
            "sidecar_verdict": verdict,
            "in_input_domain": a.get("in_box", True),
            "input_domain_check_tol": 0.0,
            "model_path": str(fake_model),
            "model_sha256": model_sha,
            "spec_path": "/fake/spec.vnnlib",
            "spec_sha256": "deadbeef" * 8,
            "x_star_npy": x_npy.name,
            "x_star_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            "y_ort_npy": y_npy.name,
            "spec_result_zero_tol": {"ast_holds": a.get("zero_holds", True)},
            "spec_result_small_tol": {"ast_holds": a.get("small_holds", True)},
            "witness_id": {
                "benchmark": "cersyve",
                "instance_id": iid,
                "source": source,
                "attempt_index": attempt,
            },
            "run_id": "test",
            "hyzor_git_head": "abc123",
            "hyzor_worktree_dirty": False,
        }
        if a.get("tamper_x"):
            # corrupt the .npy after sha256 was recorded
            np.save(x_npy, x + 1.0)
        (d / f"{stem}.json").write_text(json.dumps(art, indent=2))
        manifest_entries.append({
            "file": f"{stem}.json",
            "verdict": verdict,
            "witness_id": art["witness_id"],
        })

    manifest = {
        "artifacts": manifest_entries,
        "totals": {"by_verdict": {}},
        "manifest_schema_version": 1,
    }
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return d


class TestSidecarBridge(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.receipt_dir = self.tmp / "act_receipts"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _bridge(self, artifacts):
        sidecar = _make_fake_sidecar_dir(self.tmp, artifacts)
        return bridge(sidecar, self.receipt_dir, "cersyve", REPO)

    def test_only_sat_zero_tol_bridged(self):
        """sat_replay_failed / sat_small_tol_only must NOT be promoted."""
        summary = self._bridge([
            {"verdict": "sat_zero_tol"},
            {"verdict": "sat_replay_failed"},
            {"verdict": "sat_small_tol_only"},
            {"verdict": "reject_pre_replay"},
            {"verdict": "sat_zero_tol"},
        ])
        self.assertEqual(summary["n_emitted"], 2)
        self.assertEqual(summary["n_skipped_non_strict"], 3)

    def test_in_box_false_is_rejected(self):
        """sat_zero_tol with in_input_domain=False MUST be rejected so
        a sidecar laxity cannot back-door false witnesses into ACT."""
        summary = self._bridge([
            {"verdict": "sat_zero_tol", "in_box": False},
        ])
        self.assertEqual(summary["n_emitted"], 0)
        self.assertEqual(summary["n_rejected_invariant"], 1)
        self.assertIn("in_input_domain", summary["rejected"][0]["reason"])

    def test_zero_holds_false_is_rejected(self):
        """If zero_tol ast_holds is somehow False, reject — sidecar
        verdict alone is not authoritative for ACT canonical."""
        summary = self._bridge([
            {"verdict": "sat_zero_tol", "zero_holds": False},
        ])
        self.assertEqual(summary["n_emitted"], 0)
        self.assertEqual(summary["n_rejected_invariant"], 1)

    def test_x_star_sha_mismatch_rejected(self):
        """If .npy was modified after sha was recorded, reject."""
        summary = self._bridge([
            {"verdict": "sat_zero_tol", "tamper_x": True},
        ])
        self.assertEqual(summary["n_emitted"], 0)
        self.assertEqual(summary["n_rejected_invariant"], 1)
        self.assertIn("sha256", summary["rejected"][0]["reason"])

    def test_emitted_receipt_loads_via_fal_receipt(self):
        """Bridged receipt MUST be parseable by ACT's fal_receipt.load_receipt."""
        from act.back_end.solver.fal_receipt import load_receipt
        self._bridge([{"verdict": "sat_zero_tol"}])
        emitted = sorted((self.receipt_dir).glob("cersyve_*_q*.json"))
        self.assertEqual(len(emitted), 1)
        d = load_receipt(emitted[0])
        # Required ACT receipt fields per fal_receipt.write_receipt schema.
        for field in ("schema_version", "witness_id", "model_sha256",
                      "spec_zero_tol_holds", "spec_small_tol_holds",
                      "input_box_holds", "input_box_reason",
                      "tol_zero", "tol_small", "x_star_sha256"):
            self.assertIn(field, d, f"bridged receipt missing required field: {field}")
        self.assertEqual(d["input_box_holds"], True)
        self.assertEqual(d["spec_zero_tol_holds"], True)
        self.assertEqual(d["input_box_reason"], "ok")

    def test_query_index_derived_from_source(self):
        """SATSidecar source 'd3_box_corner' → query_index=3."""
        self._bridge([
            {"verdict": "sat_zero_tol", "source": "d3_box_corner", "instance_id": 7, "attempt": 99},
        ])
        emitted = sorted(self.receipt_dir.glob("cersyve_7_q*_*.json"))
        self.assertEqual(len(emitted), 1)
        self.assertIn("cersyve_7_q3_", emitted[0].name)

    def test_no_filename_collision(self):
        """Two bridges into the same receipt_dir with same stem → second raises."""
        self._bridge([{"verdict": "sat_zero_tol"}])
        # second call with same artifact stem should hit collision in receipt_dir
        summary = self._bridge([{"verdict": "sat_zero_tol"}])
        # the second call uses a fresh sidecar but identical iid/source/attempt
        # stems → the emit raises BridgeError("receipt collision")
        self.assertEqual(summary["n_emitted"], 0)
        self.assertEqual(summary["n_rejected_invariant"], 1)
        self.assertIn("collision", summary["rejected"][0]["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
