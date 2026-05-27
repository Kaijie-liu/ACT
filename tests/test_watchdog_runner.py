"""Regression tests for the process-level watchdog runner.

The watchdog wraps the existing ACT verify CLI in an outer subprocess
with wall-clock + RSS caps. These tests pin three invariants that the
runner exists to enforce:

  1. A child that finishes within the wall budget returns ``OK`` and
     points at the CLI's own per-instance JSON.
  2. A child that exceeds the wall deadline is SIGTERM'd, then
     SIGKILL'd if necessary, and returns ``UNKNOWN_TIMEOUT`` — never
     ``OK`` or a CERTIFIED verdict.
  3. A child that exceeds the RSS cap is terminated and returns
     ``UNKNOWN_RESOURCE_LIMIT`` — never ``OK``.

We exercise the watchdog by substituting a tiny fake-CLI script (so we
do not depend on the heavy real verify path here). Two of the three
invariants are easy to test deterministically (timeout, OK); RSS-cap
testing uses a guaranteed allocator that crosses the cap quickly.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from act.pipeline.watchdog_runner import (
    WatchdogConfig, _normalize_status_for_aggregator, _tree_rss_mb,
    _write_synthetic_per_instance, is_acceptable, is_bounded_unknown,
)


# Helper: write a stand-in CLI script that mimics ACT's per-instance JSON
# convention (per_instance_<bench>_<UTC>.json under ACT_FORMAL_RESULTS_DIR)
# but does whatever the test asks (sleep, allocate, return).
def _fake_cli(out_dir: Path, name: str, body: str) -> Path:
    p = out_dir / f"fake_cli_{name}.py"
    p.write_text(textwrap.dedent(body))
    return p


def _run_under_watchdog(
    fake_script: Path, *, wall_s: float, rss_cap_gb=None,
    startup_grace_s=0.5, poll_interval_s=0.1, grace_kill_s=2.0,
    out_dir: Path, strict_bounded_failure: bool = False,
):
    """Invoke watchdog_runner.run_instance with a fake CLI by monkey-
    patching _build_cli_cmd / _build_cli_env to launch our fake script
    instead. This is the cleanest way to test the watchdog loop without
    standing up the entire verify path."""
    from act.pipeline import watchdog_runner as wd

    orig_cmd = wd._build_cli_cmd
    orig_env = wd._build_cli_env

    def fake_cmd(*, python_exe, benchmark, instance_id, timeout_s,
                 device, dtype, canonical_root, out_dir):
        return [sys.executable, str(fake_script), str(out_dir), benchmark, str(instance_id)]

    def fake_env(*, canonical_root, out_dir, formal_mode):
        env = dict(os.environ)
        return env

    wd._build_cli_cmd = fake_cmd
    wd._build_cli_env = fake_env
    try:
        config = WatchdogConfig(
            wall_s=wall_s, rss_cap_gb=rss_cap_gb,
            startup_grace_s=startup_grace_s,
            poll_interval_s=poll_interval_s,
            grace_kill_s=grace_kill_s,
        )
        return wd.run_instance(
            benchmark="fake_bench", instance_id=0, config=config,
            out_dir=out_dir, canonical_root=Path("/tmp"),
            strict_bounded_failure=strict_bounded_failure,
        )
    finally:
        wd._build_cli_cmd = orig_cmd
        wd._build_cli_env = orig_env


class TestWatchdogRunner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_clean_child_returns_ok(self):
        """A child that exits 0 and writes per_instance_<bench>_*.json
        within budget must return status=OK."""
        fake = _fake_cli(self.tmp, "clean", """
            import sys, json, os, time
            from pathlib import Path
            out, bench, iid = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
            out.mkdir(parents=True, exist_ok=True)
            now = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + 'Z'
            rec = {
                "schema_version": 1, "benchmark": bench, "formal_mode": True,
                "timestamp_utc": now, "wall_min": 0.0,
                "counts": {"CERTIFIED": 1, "FALSIFIED": 0, "UNKNOWN": 0, "ERROR": 0},
                "run_status": "PASSED",
                "per_instance": [{
                    "official_instance_id": int(iid),
                    "internal_status": "UNSAT", "reportable_status": "CERTIFIED",
                    "cli_normalized": "CERTIFIED", "wall_s": 0.05,
                    "queries": [], "q_statuses": [], "q_reportables": [], "q_receipts": [],
                }],
            }
            (out / f'per_instance_{bench}_{now}.json').write_text(json.dumps(rec))
            sys.exit(0)
        """)
        r = _run_under_watchdog(fake, wall_s=10.0, out_dir=self.tmp)
        self.assertEqual(r.status, "OK", f"got status={r.status} stdout={r.stdout_tail!r}")
        self.assertEqual(r.cli_normalized, "OK")
        self.assertIsNotNone(r.per_instance_json)
        d = json.loads(Path(r.per_instance_json).read_text())
        self.assertEqual(d["counts"]["CERTIFIED"], 1)

    def test_wall_timeout_returns_unknown_timeout(self):
        """A child that hangs past the wall deadline must be killed and
        return status=UNKNOWN_TIMEOUT, NOT promoted to CERTIFIED even
        if the child later would have."""
        fake = _fake_cli(self.tmp, "hang", """
            import time, sys
            # Sleep way past any test deadline.
            time.sleep(60)
        """)
        t0 = time.monotonic()
        r = _run_under_watchdog(
            fake, wall_s=1.0, startup_grace_s=0.5,
            poll_interval_s=0.1, grace_kill_s=1.0, out_dir=self.tmp,
        )
        elapsed = time.monotonic() - t0
        self.assertEqual(r.status, "UNKNOWN_TIMEOUT")
        self.assertEqual(r.cli_normalized, "UNKNOWN_TIMEOUT")
        self.assertLess(elapsed, 8.0,
                        f"watchdog took {elapsed:.1f}s; deadline was 1.5s + grace")
        # Synthetic record must record the UNKNOWN_TIMEOUT verdict and not
        # contain any CERTIFIED counts.
        self.assertIsNotNone(r.per_instance_json)
        d = json.loads(Path(r.per_instance_json).read_text())
        self.assertEqual(d["counts"]["CERTIFIED"], 0)
        self.assertEqual(d["counts"]["FALSIFIED"], 0)
        self.assertEqual(d["counts"]["UNKNOWN"], 1)
        self.assertEqual(
            d["per_instance"][0]["cli_normalized"], "UNKNOWN_TIMEOUT"
        )

    def test_wall_timeout_sigkills_child_that_ignores_sigterm(self):
        """The hard cutoff must not depend on cooperative SIGTERM handling.

        Long native solver calls may defer or ignore SIGTERM. Once the
        grace period expires, SIGKILL must make ``run_instance`` return
        promptly with authoritative UNKNOWN_TIMEOUT provenance.
        """
        fake = _fake_cli(self.tmp, "ignores_sigterm", """
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
        """)
        t0 = time.monotonic()
        r = _run_under_watchdog(
            fake, wall_s=0.2, startup_grace_s=0.1,
            poll_interval_s=0.05, grace_kill_s=0.2, out_dir=self.tmp,
            strict_bounded_failure=True,
        )
        elapsed = time.monotonic() - t0
        self.assertEqual(r.status, "UNKNOWN_TIMEOUT")
        self.assertEqual(r.returncode, -9)
        self.assertLess(elapsed, 3.0, f"hard kill returned after {elapsed:.2f}s")
        d = json.loads(Path(r.per_instance_json).read_text())
        self.assertEqual(d["run_status"], "FAILED")
        self.assertEqual(d["counts"]["UNKNOWN"], 1)

    def test_timeout_supersedes_child_certificate_with_unknown_record(self):
        """A child can write an apparent result and then hang. Once it is
        killed, the authoritative returned record must be synthetic UNKNOWN,
        never the prewritten CERTIFIED artifact."""
        fake = _fake_cli(self.tmp, "writes_then_hangs", """
            import json, sys, time
            from pathlib import Path
            out, bench, iid = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
            out.mkdir(parents=True, exist_ok=True)
            rec = {
                "counts": {"CERTIFIED": 1, "FALSIFIED": 0, "UNKNOWN": 0, "ERROR": 0},
                "run_status": "PASSED",
                "per_instance": [{
                    "official_instance_id": int(iid),
                    "cli_normalized": "CERTIFIED",
                    "internal_status": "UNSAT",
                    "reportable_status": "CERTIFIED",
                }],
            }
            (out / f'per_instance_{bench}_child.json').write_text(json.dumps(rec))
            time.sleep(60)
        """)
        r = _run_under_watchdog(
            fake, wall_s=0.2, startup_grace_s=0.2,
            poll_interval_s=0.05, grace_kill_s=0.5, out_dir=self.tmp,
        )
        self.assertEqual(r.status, "UNKNOWN_TIMEOUT")
        d = json.loads(Path(r.per_instance_json).read_text())
        self.assertTrue(d["watchdog_synthetic"])
        self.assertEqual(d["counts"]["CERTIFIED"], 0)
        self.assertEqual(d["counts"]["UNKNOWN"], 1)
        self.assertEqual(d["per_instance"][0]["cli_normalized"], "UNKNOWN_TIMEOUT")
        self.assertIn("per_instance_fake_bench_child.json",
                      d["superseded_child_per_instance_json"])

    def test_rss_cap_returns_unknown_resource_limit(self):
        """A child that allocates past the RSS cap must be killed and
        return UNKNOWN_RESOURCE_LIMIT, NOT silently OK."""
        fake = _fake_cli(self.tmp, "balloon", """
            import time
            # Hold a 50 MiB buffer; cap is set to 30 MiB in the test.
            blob = bytearray(50 * 1024 * 1024)
            for i in range(0, len(blob), 4096):
                blob[i] = (i % 256)
            time.sleep(30)
        """)
        # Cap at ~30 MiB so the 50 MiB child trips it; the watchdog also has a
        # generous wall budget so the kill cause is definitely RSS.
        r = _run_under_watchdog(
            fake, wall_s=30.0, rss_cap_gb=30 / 1024.0,
            startup_grace_s=0.5, poll_interval_s=0.1, grace_kill_s=1.0,
            out_dir=self.tmp,
        )
        self.assertEqual(r.status, "UNKNOWN_RESOURCE_LIMIT",
                         f"got status={r.status} peak_rss={r.peak_rss_mb:.1f}MiB")
        self.assertEqual(r.cli_normalized, "UNKNOWN_RESOURCE_LIMIT")
        # Synthetic record must NOT show CERTIFIED/FALSIFIED.
        d = json.loads(Path(r.per_instance_json).read_text())
        self.assertEqual(d["counts"]["CERTIFIED"], 0)
        self.assertEqual(d["counts"]["FALSIFIED"], 0)
        self.assertEqual(d["counts"]["UNKNOWN"], 1)

    def test_bounded_unknown_run_status_is_passed(self):
        """Watchdog-imposed bounded UNKNOWN (TIMEOUT, RESOURCE_LIMIT)
        records a successful auditable verdict — the synthetic per-
        instance JSON must read `run_status=PASSED` so downstream
        aggregators do not double-fault it as a crash."""
        out = self.tmp / "syn_to"
        out.mkdir()
        p = _write_synthetic_per_instance(
            benchmark="b", instance_id=0, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=1.0, peak_rss_mb=10.0,
            out_dir=out, returncode=-15,
        )
        d = json.loads(p.read_text())
        self.assertEqual(d["run_status"], "PASSED")
        out2 = self.tmp / "syn_rss"
        out2.mkdir()
        p2 = _write_synthetic_per_instance(
            benchmark="b", instance_id=0, status="UNKNOWN_RESOURCE_LIMIT",
            cli_normalized="UNKNOWN_RESOURCE_LIMIT", wall_s=1.0,
            peak_rss_mb=1024.0, out_dir=out2, returncode=-15,
        )
        d2 = json.loads(p2.read_text())
        self.assertEqual(d2["run_status"], "PASSED")

    def test_bounded_unknown_run_status_is_failed_under_strict_policy(self):
        """Strict qualification treats a bounded termination as a failed
        run while preserving the UNKNOWN verdict itself."""
        out = self.tmp / "syn_strict"
        out.mkdir()
        p = _write_synthetic_per_instance(
            benchmark="b", instance_id=0, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=1.0, peak_rss_mb=10.0,
            out_dir=out, returncode=-15, strict_bounded_failure=True,
        )
        d = json.loads(p.read_text())
        self.assertEqual(d["run_status"], "FAILED")
        self.assertTrue(d["strict_bounded_failure"])
        self.assertEqual(d["counts"]["UNKNOWN"], 1)
        self.assertEqual(d["per_instance"][0]["cli_normalized"], "UNKNOWN_TIMEOUT")

    def test_strict_timeout_propagates_policy_to_synthetic_record(self):
        """The end-to-end runner must not drop strict policy while
        creating a synthetic record for a killed child."""
        fake = _fake_cli(self.tmp, "strict_hang", """
            import time
            time.sleep(60)
        """)
        r = _run_under_watchdog(
            fake, wall_s=0.2, startup_grace_s=0.1,
            poll_interval_s=0.05, grace_kill_s=0.5, out_dir=self.tmp,
            strict_bounded_failure=True,
        )
        self.assertEqual(r.status, "UNKNOWN_TIMEOUT")
        d = json.loads(Path(r.per_instance_json).read_text())
        self.assertEqual(d["run_status"], "FAILED")
        self.assertTrue(d["strict_bounded_failure"])

    def test_real_error_run_status_is_failed(self):
        """A child that crashed (NO_OUTPUT, SPAWN_FAILED, EXIT_NONZERO)
        is NOT bounded UNKNOWN — its synthetic record must read
        `run_status=FAILED` so the run can't be silently accepted."""
        out = self.tmp / "syn_err"
        out.mkdir()
        for bad in ("NO_OUTPUT", "EXIT_NONZERO", "SPAWN_FAILED"):
            p = _write_synthetic_per_instance(
                benchmark="b", instance_id=0, status=bad,
                cli_normalized=f"ERROR_WATCHDOG_{bad}", wall_s=0.5,
                peak_rss_mb=5.0, out_dir=out, returncode=-1,
            )
            d = json.loads(p.read_text())
            self.assertEqual(d["run_status"], "FAILED",
                             f"status={bad} must produce run_status=FAILED")
            # Cleanup so the next iteration can reuse the dir.
            p.unlink()

    def test_is_acceptable_default_admits_bounded_unknown(self):
        """Default policy: OK + bounded UNKNOWN are acceptable;
        everything else is not."""
        self.assertTrue(is_acceptable("OK"))
        self.assertTrue(is_acceptable("UNKNOWN_TIMEOUT"))
        self.assertTrue(is_acceptable("UNKNOWN_RESOURCE_LIMIT"))
        self.assertFalse(is_acceptable("NO_OUTPUT"))
        self.assertFalse(is_acceptable("EXIT_NONZERO"))
        self.assertFalse(is_acceptable("SPAWN_FAILED"))

    def test_is_acceptable_strict_only_admits_ok(self):
        """Strict policy (used for paper-grade sentinel runs): any
        watchdog termination — including bounded UNKNOWN — fails the
        run, because the goal is to assert there was no need to
        terminate."""
        self.assertTrue(is_acceptable("OK", strict_bounded_failure=True))
        self.assertFalse(is_acceptable("UNKNOWN_TIMEOUT", strict_bounded_failure=True))
        self.assertFalse(is_acceptable("UNKNOWN_RESOURCE_LIMIT", strict_bounded_failure=True))
        self.assertFalse(is_acceptable("NO_OUTPUT", strict_bounded_failure=True))

    def test_is_bounded_unknown_exact_set(self):
        """Future watchdog statuses that *should* count as bounded
        UNKNOWN must be added to ``_BOUNDED_UNKNOWN_STATUSES``
        explicitly. The fail-closed invariant relies on this allowlist."""
        self.assertTrue(is_bounded_unknown("UNKNOWN_TIMEOUT"))
        self.assertTrue(is_bounded_unknown("UNKNOWN_RESOURCE_LIMIT"))
        # New, invented statuses must NOT silently slip into the bucket.
        self.assertFalse(is_bounded_unknown("UNKNOWN_SOMETHING"))
        self.assertFalse(is_bounded_unknown("ERROR_WATCHDOG_NO_OUTPUT"))

    def test_normalize_status_never_promotes(self):
        """The normalization helper must never return a string that
        prefix-matches the CERTIFIED / FALSIFIED tokens that downstream
        aggregators bucket on."""
        for s in (
            "UNKNOWN_TIMEOUT", "UNKNOWN_RESOURCE_LIMIT", "NO_OUTPUT",
            "EXIT_NONZERO", "SPAWN_FAILED", "OK",
        ):
            n = _normalize_status_for_aggregator(s)
            self.assertNotEqual(n, "CERTIFIED")
            self.assertNotEqual(n, "FALSIFIED")

    def test_tree_rss_mb_returns_nonnegative_for_self(self):
        """Sanity: the RSS sampler returns a positive number for the
        running test process and 0 for a definitely-dead pid."""
        rss = _tree_rss_mb(os.getpid())
        self.assertGreater(rss, 0)
        self.assertEqual(_tree_rss_mb(2_000_000_000), 0.0)

    def test_synthetic_per_instance_record_is_join_compatible(self):
        """phase1_runner joins per_instance records on
        official_instance_id; the synthetic record must carry that key."""
        out = self.tmp / "syn"
        out.mkdir()
        p = _write_synthetic_per_instance(
            benchmark="fake_bench", instance_id=42, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=12.5, peak_rss_mb=99.0,
            out_dir=out, returncode=-15,
        )
        d = json.loads(p.read_text())
        self.assertEqual(d["per_instance"][0]["official_instance_id"], 42)
        self.assertEqual(d["per_instance"][0]["cli_normalized"], "UNKNOWN_TIMEOUT")
        self.assertTrue(d["watchdog_synthetic"])

    def test_synthetic_paths_are_unique_by_instance_id(self):
        """Several quick kills in one benchmark directory must not overwrite
        each other's synthetic evidence in the same timestamp second."""
        out = self.tmp / "multi_iid"
        out.mkdir()
        p1 = _write_synthetic_per_instance(
            benchmark="b", instance_id=1, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=1.0, peak_rss_mb=1.0,
            out_dir=out, returncode=-15,
        )
        p2 = _write_synthetic_per_instance(
            benchmark="b", instance_id=2, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=1.0, peak_rss_mb=1.0,
            out_dir=out, returncode=-15,
        )
        self.assertNotEqual(p1, p2)
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())

    def test_synthetic_paths_are_unique_on_same_instance_retry(self):
        """Repeated termination records for one iid in one output
        directory must preserve both audit artifacts."""
        out = self.tmp / "same_iid_retry"
        out.mkdir()
        p1 = _write_synthetic_per_instance(
            benchmark="b", instance_id=1, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=1.0, peak_rss_mb=1.0,
            out_dir=out, returncode=-15,
        )
        p2 = _write_synthetic_per_instance(
            benchmark="b", instance_id=1, status="UNKNOWN_TIMEOUT",
            cli_normalized="UNKNOWN_TIMEOUT", wall_s=1.0, peak_rss_mb=1.0,
            out_dir=out, returncode=-15,
        )
        self.assertNotEqual(p1, p2)
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
