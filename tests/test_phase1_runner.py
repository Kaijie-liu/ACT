"""Regression tests for the Phase 1 manifest-driven runner.

Round 8 (advisor 2026-05-24): the runner is the single point of audit
truth for Phase 1. These tests pin its pass/fail contract so future
refactors cannot silently weaken it.

The pure helpers ``load_seed``, ``join_with_seed``, and
``compute_phase1_verdict`` are tested directly. ``capture_code_snapshot``
is tested via a temp git repo.
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from act.pipeline.phase1_runner import (
    capture_code_snapshot,
    compute_phase1_verdict,
    join_with_seed,
    load_seed,
    select_iids_by_bench,
    sha256_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_seed(path: Path, rows):
    """Write minimal seed CSV (matches round4_sat61_proof_only.csv schema)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "instance_id", "official_zero", "official_small",
            "prev_status", "proof_only_status", "bucket", "elapsed",
        ])
        for r in rows:
            w.writerow([
                r["benchmark"], r["instance_id"], r["official_zero"],
                r["official_small"], "", "", "", "",
            ])


def _seed_unsat_sat(rows):
    """Build a uniform seed where every row is official_zero=unsat,
    official_small=sat (the actual P1 watchlist shape)."""
    return [
        {"benchmark": b, "instance_id": int(i),
         "official_zero": "unsat", "official_small": "sat"}
        for (b, i) in rows
    ]


# ---------------------------------------------------------------------------
# load_seed / SHA verification
# ---------------------------------------------------------------------------


class TestLoadSeed(unittest.TestCase):
    def test_loads_and_normalizes_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "seed.csv"
            rows = _seed_unsat_sat([("cifar100_2024", 1), ("cifar100_2024", 2)])
            _write_seed(p, rows)
            actual_sha = sha256_file(p)
            loaded = load_seed(p, actual_sha)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["benchmark"], "cifar100_2024")
            self.assertEqual(loaded[0]["official_instance_id"], 1)
            self.assertEqual(loaded[0]["official_zero"], "unsat")
            self.assertEqual(loaded[0]["official_small"], "sat")

    def test_sha_mismatch_refuses_load(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "seed.csv"
            _write_seed(p, _seed_unsat_sat([("x", 0)]))
            bad_sha = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                load_seed(p, bad_sha)
            self.assertIn("SHA mismatch", str(ctx.exception))


# ---------------------------------------------------------------------------
# select_iids_by_bench
# ---------------------------------------------------------------------------


class TestSelectIids(unittest.TestCase):
    def test_groups_and_sorts(self):
        seed = _seed_unsat_sat([
            ("cifar100_2024", 5), ("cifar100_2024", 1),
            ("tinyimagenet_2024", 3), ("tinyimagenet_2024", 1),
        ])
        # the parser converts instance_id to int; mimic that
        for r in seed:
            r["official_instance_id"] = r.pop("instance_id")
        out = select_iids_by_bench(seed)
        self.assertEqual(out["cifar100_2024"], [1, 5])
        self.assertEqual(out["tinyimagenet_2024"], [1, 3])


# ---------------------------------------------------------------------------
# join_with_seed + compute_phase1_verdict — the four contract tests
# ---------------------------------------------------------------------------


class TestPhase1Contract(unittest.TestCase):
    """The 4 advisor-mandated contract tests."""

    def _seed_iid(self, b, i):
        return {"benchmark": b, "official_instance_id": i,
                "official_zero": "unsat", "official_small": "sat"}

    # CONTRACT 1: seed unsat + formal FALSIFIED → whole run FAIL
    def test_unsat_with_formal_falsified_fails_whole_run(self):
        seed = [self._seed_iid("cifar100_2024", 42)]
        per_inst = {
            "cifar100_2024": [{
                "official_instance_id": 42,
                "internal_status": "SAT",
                "reportable_status": "FALSIFIED",
                "cli_normalized": "FALSIFIED",
                "count_bucket": "FALSIFIED",
                "wall_s": 1.0, "q_receipts": ["receipt0.json"],
            }],
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 0},
            bench_counts={"cifar100_2024": {"FALSIFIED": 1}},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(
            any("forbidden_count" in r and "FALSIFIED=1" in r for r in reasons),
            f"reasons={reasons}",
        )
        self.assertTrue(
            any("forbidden_reportable" in r for r in reasons),
            f"reasons={reasons}",
        )

    # CONTRACT 2: missing instance ID → FAIL
    def test_missing_instance_id_fails(self):
        seed = [self._seed_iid("cifar100_2024", 7),
                self._seed_iid("cifar100_2024", 8)]
        per_inst = {
            "cifar100_2024": [{
                "official_instance_id": 7,
                "internal_status": "UNSAT",
                "reportable_status": "CERTIFIED",
                "cli_normalized": "CERTIFIED",
                "count_bucket": "CERTIFIED",
                "wall_s": 1.0,
            }],
            # iid=8 missing
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 0},
            bench_counts={"cifar100_2024": {"CERTIFIED": 1}},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("coverage_gap" in r for r in reasons))
        self.assertTrue(any("missing_instance" in r and "iid=8" in r for r in reasons))

    # CONTRACT 2b: duplicate instance ID in per_instance → FAIL
    def test_duplicate_per_instance_fails(self):
        seed = [self._seed_iid("cifar100_2024", 3)]
        per_inst = {
            "cifar100_2024": [
                {"official_instance_id": 3, "internal_status": "UNSAT",
                 "reportable_status": "CERTIFIED", "cli_normalized": "CERTIFIED",
                 "count_bucket": "CERTIFIED", "wall_s": 1.0},
                {"official_instance_id": 3, "internal_status": "UNKNOWN",
                 "reportable_status": "UNKNOWN", "cli_normalized": "UNKNOWN",
                 "count_bucket": "UNKNOWN", "wall_s": 1.0},
            ],
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 0},
            bench_counts={"cifar100_2024": {"CERTIFIED": 1, "UNKNOWN": 1}},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("duplicate_per_instance" in r for r in reasons))

    # CONTRACT 2c: extra instance (CLI ran something not in seed) → FAIL
    def test_extra_instance_not_in_seed_fails(self):
        seed = [self._seed_iid("cifar100_2024", 1)]
        per_inst = {
            "cifar100_2024": [
                {"official_instance_id": 1, "internal_status": "UNSAT",
                 "reportable_status": "CERTIFIED", "cli_normalized": "CERTIFIED",
                 "count_bucket": "CERTIFIED", "wall_s": 1.0},
                {"official_instance_id": 99, "internal_status": "UNKNOWN",
                 "reportable_status": "UNKNOWN", "cli_normalized": "UNKNOWN",
                 "count_bucket": "UNKNOWN", "wall_s": 1.0},
            ],
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 0},
            bench_counts={"cifar100_2024": {"CERTIFIED": 1, "UNKNOWN": 1}},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("extra_instance" in r and "iid=99" in r for r in reasons))

    # CONTRACT 3: all CERTIFIED/UNKNOWN, no errors → PASS
    def test_all_certified_or_unknown_passes(self):
        seed = [self._seed_iid("cifar100_2024", 1),
                self._seed_iid("cifar100_2024", 2),
                self._seed_iid("tinyimagenet_2024", 10)]
        per_inst = {
            "cifar100_2024": [
                {"official_instance_id": 1, "internal_status": "UNSAT",
                 "reportable_status": "CERTIFIED", "cli_normalized": "CERTIFIED",
                 "count_bucket": "CERTIFIED", "wall_s": 1.0},
                {"official_instance_id": 2, "internal_status": "UNKNOWN",
                 "reportable_status": "UNKNOWN", "cli_normalized": "UNKNOWN",
                 "count_bucket": "UNKNOWN", "wall_s": 2.0},
            ],
            "tinyimagenet_2024": [
                {"official_instance_id": 10, "internal_status": "UNKNOWN",
                 "reportable_status": "UNKNOWN", "cli_normalized": "UNKNOWN",
                 "count_bucket": "UNKNOWN", "wall_s": 3.0},
            ],
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 0, "tinyimagenet_2024": 0},
            bench_counts={
                "cifar100_2024": {"CERTIFIED": 1, "UNKNOWN": 1},
                "tinyimagenet_2024": {"UNKNOWN": 1},
            },
        )
        self.assertEqual(verdict, "PASS", f"unexpected reasons: {reasons}")
        self.assertEqual(reasons, [])

    # CONTRACT 4: per-bench subprocess nonzero exit → FAIL
    def test_subprocess_nonzero_exit_fails(self):
        seed = [self._seed_iid("cifar100_2024", 1)]
        per_inst = {
            "cifar100_2024": [{
                "official_instance_id": 1,
                "internal_status": "UNSAT", "reportable_status": "CERTIFIED",
                "cli_normalized": "CERTIFIED", "count_bucket": "CERTIFIED",
                "wall_s": 1.0,
            }],
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 1},  # NONZERO
            bench_counts={"cifar100_2024": {"CERTIFIED": 1}},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("benchmark_subprocess_nonzero_exit" in r
                            and "exit=1" in r for r in reasons))

    # CONTRACT 5: ERROR_RECEIPT or ERROR present → FAIL
    def test_error_receipt_fails(self):
        seed = [self._seed_iid("cifar100_2024", 1)]
        per_inst = {
            "cifar100_2024": [{
                "official_instance_id": 1,
                "internal_status": "UNKNOWN", "reportable_status": "UNKNOWN",
                "cli_normalized": "UNKNOWN", "count_bucket": "UNKNOWN",
                "wall_s": 1.0,
            }],
        }
        joined, findings = join_with_seed(seed, per_inst)
        verdict, reasons = compute_phase1_verdict(
            joined, findings,
            bench_exits={"cifar100_2024": 0},
            bench_counts={"cifar100_2024": {"UNKNOWN": 1, "ERROR_RECEIPT": 1}},
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("ERROR_RECEIPT=1" in r for r in reasons))


# ---------------------------------------------------------------------------
# capture_code_snapshot — dirty worktree refusal
# ---------------------------------------------------------------------------


class TestCodeSnapshot(unittest.TestCase):
    def _make_repo(self, td: Path) -> Path:
        # init a tiny repo with one commit
        subprocess.check_call(["git", "init", "-q", str(td)])
        subprocess.check_call(["git", "-C", str(td), "config",
                               "user.email", "test@example.com"])
        subprocess.check_call(["git", "-C", str(td), "config",
                               "user.name", "test"])
        (td / "a.txt").write_text("hello\n")
        subprocess.check_call(["git", "-C", str(td), "add", "a.txt"])
        subprocess.check_call(["git", "-C", str(td), "commit", "-q", "-m", "init"])
        return td

    def test_clean_worktree_captures_head(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._make_repo(Path(td))
            snap = capture_code_snapshot(repo, allow_dirty=False)
            self.assertEqual(len(snap["git_head"]), 40)
            self.assertFalse(snap["worktree_dirty"])
            self.assertEqual(snap["dirty_diff_sha256"], "")

    def test_dirty_worktree_refused_without_flag(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._make_repo(Path(td))
            (repo / "a.txt").write_text("hello\nmodified\n")
            with self.assertRaises(RuntimeError) as ctx:
                capture_code_snapshot(repo, allow_dirty=False)
            self.assertIn("dirty", str(ctx.exception).lower())

    def test_dirty_worktree_allowed_with_flag_records_diff_sha(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._make_repo(Path(td))
            (repo / "a.txt").write_text("hello\nmodified\n")
            (repo / "untracked.txt").write_text("new\n")
            snap = capture_code_snapshot(repo, allow_dirty=True)
            self.assertTrue(snap["worktree_dirty"])
            self.assertEqual(len(snap["dirty_diff_sha256"]), 64)
            self.assertIn("modified", snap["dirty_diff_text"])
            self.assertIn("untracked.txt", snap["untracked_files"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
