"""Process-level watchdog runner for ACT formal verification.

Motivation
==========
The in-process ``--timeout`` only feeds into HZ solver / LP consumption
phases. It does NOT:

  * interrupt a long-running ``analyze()`` call,
  * enforce a wall-clock cap on the per-instance VNNLIB conversion step,
  * stop memory growth in HZ propagation (e.g. cgan_2023 first instance
    consumed about 18.6 GB RSS and over an hour before being killed).

Three observed in-the-wild manifestations:

  * ``yolo_2023`` smoke with ``--timeout 30`` actually ran ~90 s per
    instance (5 instances → 7.8 min wall),
  * ``nn4sys`` ``lindex_200`` with ``--timeout 3`` did not return its
    first query within 45 s (external guard had to ``timeout 124`` it),
  * ``cgan_2023`` first instance ran > 1 h at ~18.6 GB RSS.

This module provides an out-of-process watchdog: each official instance
runs in its own subprocess, polled by the parent. Soft termination
(SIGTERM) at wall deadline, hard kill (SIGKILL) after a grace period.
RSS polled by walking the process tree; if the aggregate exceeds
``rss_cap_gb`` the subprocess is terminated.

A child that exits cleanly within budget yields ``status="OK"`` and the
unmodified per-instance JSON path; a watchdog termination yields a
synthetic per-instance record so downstream aggregators (phase1_runner,
SCORED_BENCHMARK_SUPPORT_MATRIX) can count the instance without losing
provenance.

Fail-closed invariants
======================
  * a wall-deadline kill records ``cli_normalized="UNKNOWN_TIMEOUT"`` —
    never promoted to CERTIFIED;
  * an RSS-cap kill records ``cli_normalized="UNKNOWN_RESOURCE_LIMIT"``
    — never promoted to CERTIFIED or FALSIFIED;
  * a non-zero exit that wrote NO per-instance JSON records
    ``cli_normalized="ERROR_WATCHDOG_CRASH"``;
  * the synthetic record carries ``wall_s`` and ``peak_rss_mb`` so a
    later audit can re-decide.

This is intentionally a thin wrapper around the existing CLI rather
than an alternative verifier — it preserves every existing soundness
guarantee from R9.3 et seq. and only adds an outer enforcement layer.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WatchdogConfig:
    """Per-instance watchdog parameters.

    wall_s: hard wall-clock deadline; SIGTERM is sent at wall_s,
        SIGKILL ``grace_kill_s`` seconds later if the process is still
        alive.
    rss_cap_gb: aggregate (parent + descendants) RSS cap in GiB; ``None``
        disables the RSS check. The cap is sampled at ``poll_interval_s``
        intervals; an over-cap reading terminates the subprocess.
    startup_grace_s: extra seconds added to ``wall_s`` to account for
        ONNX load + parser setup before the verification clock should
        start ticking. The wall deadline therefore is
        ``wall_s + startup_grace_s`` from process spawn.
    poll_interval_s: how often the watchdog wakes to check process state.
        Smaller values catch RSS spikes faster but cost more CPU on the
        watchdog itself.
    grace_kill_s: seconds between SIGTERM and SIGKILL after the soft
        deadline.
    """
    wall_s: float
    rss_cap_gb: Optional[float] = None
    startup_grace_s: float = 5.0
    poll_interval_s: float = 1.0
    grace_kill_s: float = 5.0


@dataclass
class WatchdogResult:
    benchmark: str
    instance_id: int
    status: str           # OK | UNKNOWN_TIMEOUT | UNKNOWN_RESOURCE_LIMIT | ERROR_WATCHDOG_CRASH | EXIT_NONZERO
    cli_normalized: str   # propagated for the per-instance aggregator
    wall_s: float
    peak_rss_mb: float
    returncode: Optional[int]
    per_instance_json: Optional[str]
    out_dir: str
    stdout_tail: str


# ---------------------------------------------------------------------------
# Process-tree RSS sampling


def _read_rss_kb(pid: int) -> int:
    """Return VmRSS (KiB) for one pid, 0 if the process is gone.

    /proc/<pid>/status is the canonical kernel-reported RSS. Note this
    is shared across forked children of the same image; we sum naively
    to give a conservative upper bound (worst-case overcount, never
    underreport)."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _children_pids(pid: int) -> List[int]:
    """List of direct child pids; empty if /proc unavailable.

    Used to roll up RSS for the verifier worker subprocesses that the
    ACT CLI may spawn (e.g. ORT replay child). The walk is one level
    deep — sufficient for the current CLI which does not deeply nest.
    """
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r") as f:
            txt = f.read().strip()
            return [int(c) for c in txt.split() if c.isdigit()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []


def _tree_rss_mb(root_pid: int) -> float:
    """Sum RSS over root + immediate children + grandchildren (3 levels)
    in MiB. Cheap enough at 1 Hz polling."""
    seen = {root_pid}
    frontier = [root_pid]
    for _depth in range(3):
        next_frontier: List[int] = []
        for p in frontier:
            for c in _children_pids(p):
                if c not in seen:
                    seen.add(c)
                    next_frontier.append(c)
        frontier = next_frontier
        if not frontier:
            break
    return sum(_read_rss_kb(p) for p in seen) / 1024.0


# ---------------------------------------------------------------------------
# Main per-instance runner


def _normalize_status_for_aggregator(status: str) -> str:
    """Map watchdog status into CLI's cli_normalized vocabulary. Anything
    that is not OK must NOT be promotable to CERTIFIED/FALSIFIED by a
    downstream aggregator that filters on cli_normalized prefix."""
    if status == "OK":
        return "OK"
    if status == "UNKNOWN_TIMEOUT":
        return "UNKNOWN_TIMEOUT"
    if status == "UNKNOWN_RESOURCE_LIMIT":
        return "UNKNOWN_RESOURCE_LIMIT"
    return f"ERROR_WATCHDOG_{status}"


# Statuses that the watchdog itself imposes when killing a child for
# resource/time reasons. These are *bounded* UNKNOWN: the verifier did
# not crash; the watchdog deliberately stopped a long/over-budget run.
# They are auditable verdicts (full provenance retained), so they should
# not be reported the same as a verifier crash or a missing-output bug.
_BOUNDED_UNKNOWN_STATUSES = frozenset({
    "UNKNOWN_TIMEOUT", "UNKNOWN_RESOURCE_LIMIT",
})


def is_bounded_unknown(status: str) -> bool:
    """A watchdog status counts as 'bounded UNKNOWN' iff the runner
    intentionally terminated the child for time/RSS. Verifier crashes,
    spawn failures, and missing-output errors are NOT bounded UNKNOWN —
    they need a real audit."""
    return status in _BOUNDED_UNKNOWN_STATUSES


def is_acceptable(status: str, *, strict_bounded_failure: bool = False) -> bool:
    """Whether a status counts as 'run acceptable'.

    Defaults: OK + bounded UNKNOWN are acceptable; verifier errors are
    not. With ``strict_bounded_failure=True`` only OK is acceptable —
    use this for paper-grade sentinel runs where any watchdog
    intervention is itself the failure signal."""
    if status == "OK":
        return True
    if strict_bounded_failure:
        return False
    return is_bounded_unknown(status)


def _build_cli_cmd(
    *, python_exe: str, benchmark: str, instance_id: int,
    timeout_s: float, device: str, dtype: str,
    canonical_root: Path, out_dir: Path,
) -> List[str]:
    """The exact CLI invocation the watchdog wraps; isolated for testing."""
    return [
        python_exe, "-m", "act.pipeline",
        "--verify", "vnnlib",
        "--category", benchmark,
        "--instance-ids", str(int(instance_id)),
        "--max-instances", "1",
        "--timeout", str(float(timeout_s)),
        "--device", device,
        "--dtype", dtype,
        "--solvers", "hybridz",
    ]


def _build_cli_env(
    *, canonical_root: Path, out_dir: Path, formal_mode: bool,
) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ACT_VNNLIB_ROOT": str(canonical_root),
        "ACT_FORMAL_RESULTS_DIR": str(out_dir),
        "PYTHONPATH": env.get("PYTHONPATH", "/data1/Kane/ACT"),
    })
    if formal_mode:
        env["ACT_FAL_RECEIPT_FORMAL"] = "1"
        env["ACT_FAL_RECEIPT_DIR"] = str(out_dir)
    return env


def _write_synthetic_per_instance(
    *, benchmark: str, instance_id: int, status: str,
    cli_normalized: str, wall_s: float, peak_rss_mb: float,
    out_dir: Path, returncode: Optional[int],
    strict_bounded_failure: bool = False,
    superseded_child_per_instance_json: Optional[str] = None,
) -> Path:
    """When the watchdog terminates a child mid-flight, the CLI may not
    have written its per_instance_<bench>_<UTC>.json. Write a synthetic
    record so phase1_runner can join on official_instance_id.

    ``strict_bounded_failure`` must match the enclosing watchdog policy:
    in strict mode a bounded UNKNOWN is still a fail-closed verdict, but
    the run has failed the qualification requirement and must not be
    recorded as ``PASSED``.
    """
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"per_instance_{benchmark}_watchdog_iid{instance_id}_{now}.json"
    run_status = (
        "PASSED"
        if is_acceptable(status, strict_bounded_failure=strict_bounded_failure)
        else "FAILED"
    )
    record = {
        "schema_version": 1,
        "benchmark": benchmark,
        "formal_mode": True,
        "timestamp_utc": now,
        "watchdog_synthetic": True,
        "strict_bounded_failure": bool(strict_bounded_failure),
        "superseded_child_per_instance_json": superseded_child_per_instance_json,
        "wall_min": wall_s / 60.0,
        "counts": {"CERTIFIED": 0, "FALSIFIED": 0, "UNKNOWN": 0, "ERROR": 0},
        "run_status": run_status,
        "per_instance": [{
            "official_instance_id": int(instance_id),
            "benchmark": benchmark,
            "internal_status": "UNKNOWN" if cli_normalized.startswith("UNKNOWN_") else "ERROR",
            "reportable_status": "UNKNOWN" if cli_normalized.startswith("UNKNOWN_") else "ERROR",
            "cli_normalized": cli_normalized,
            "wall_s": wall_s,
            "peak_rss_mb": peak_rss_mb,
            "returncode": returncode,
            "watchdog_status": status,
            "queries": [], "q_statuses": [], "q_reportables": [], "q_receipts": [],
            "error": (
                f"watchdog killed instance after {wall_s:.1f}s "
                f"(status={status}, peak_rss={peak_rss_mb:.1f} MiB)"
            ) if status != "OK" else None,
        }],
    }
    counts = record["per_instance"][0]
    if counts["internal_status"] == "UNKNOWN":
        record["counts"]["UNKNOWN"] = 1
    else:
        record["counts"]["ERROR"] = 1
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def run_instance(
    *, benchmark: str, instance_id: int, config: WatchdogConfig,
    out_dir: Path, canonical_root: Path,
    python_exe: str = "/data1/Kane/miniconda3/envs/act-py312/bin/python",
    device: str = "cpu", dtype: str = "float64",
    formal_mode: bool = True,
    stdout_path: Optional[Path] = None,
    strict_bounded_failure: bool = False,
) -> WatchdogResult:
    """Run one official instance under the watchdog. Returns a
    WatchdogResult with status and the per-instance JSON path (either
    written by the CLI itself or synthesized by the watchdog).

    Soundness contract: the function never returns ``status="OK"``
    unless the child exited 0; an OK result with a missing per-instance
    JSON is converted to ``ERROR_WATCHDOG_NO_OUTPUT`` because a missing
    receipt in formal mode is not safe to silently swallow.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cli_cmd(
        python_exe=python_exe, benchmark=benchmark, instance_id=instance_id,
        timeout_s=config.wall_s, device=device, dtype=dtype,
        canonical_root=canonical_root, out_dir=out_dir,
    )
    env = _build_cli_env(
        canonical_root=canonical_root, out_dir=out_dir, formal_mode=formal_mode,
    )

    stdout_path = stdout_path or (out_dir / f"watchdog_{benchmark}_{instance_id}.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    deadline_s = config.wall_s + config.startup_grace_s
    rss_cap_mb = (config.rss_cap_gb * 1024.0) if config.rss_cap_gb is not None else None
    peak_rss_mb = 0.0
    status = "OK"
    t0 = time.monotonic()

    # Start the child in its own session so kill propagates cleanly.
    log = open(stdout_path, "wb")
    try:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
    except Exception as e:
        log.close()
        # Spawn failure is its own error class; treat as ERROR_WATCHDOG_SPAWN.
        wd_status = "SPAWN_FAILED"
        cli_norm = f"ERROR_WATCHDOG_{wd_status}"
        synth = _write_synthetic_per_instance(
            benchmark=benchmark, instance_id=instance_id, status=wd_status,
            cli_normalized=cli_norm, wall_s=0.0, peak_rss_mb=0.0,
            out_dir=out_dir, returncode=None,
            strict_bounded_failure=strict_bounded_failure,
        )
        return WatchdogResult(
            benchmark=benchmark, instance_id=instance_id,
            status=wd_status, cli_normalized=cli_norm,
            wall_s=0.0, peak_rss_mb=0.0, returncode=None,
            per_instance_json=str(synth), out_dir=str(out_dir),
            stdout_tail=f"spawn failed: {e!r}",
        )

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            elapsed = time.monotonic() - t0
            # RSS sampling
            cur_rss = _tree_rss_mb(proc.pid)
            if cur_rss > peak_rss_mb:
                peak_rss_mb = cur_rss
            if rss_cap_mb is not None and cur_rss > rss_cap_mb:
                status = "UNKNOWN_RESOURCE_LIMIT"
                break
            if elapsed > deadline_s:
                status = "UNKNOWN_TIMEOUT"
                break
            time.sleep(config.poll_interval_s)
    finally:
        if proc.poll() is None:
            # Soft then hard kill, propagating to the whole session.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            t_term = time.monotonic()
            while proc.poll() is None and (time.monotonic() - t_term) < config.grace_kill_s:
                time.sleep(0.1)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait(timeout=2.0)
        log.close()

    wall_s = time.monotonic() - t0
    returncode = proc.returncode

    # Resolve per-instance JSON: the CLI writes one even on a clean OK.
    matches = sorted(out_dir.glob(f"per_instance_{benchmark}_*.json"))
    # Exclude any synthetic watchdog records we may have written earlier.
    matches = [p for p in matches if "watchdog" not in p.name]
    per_instance_json: Optional[Path] = matches[-1] if matches else None

    if status == "OK":
        if returncode != 0:
            status = "EXIT_NONZERO"
        elif per_instance_json is None:
            status = "NO_OUTPUT"

    cli_norm = _normalize_status_for_aggregator(status)

    if status != "OK":
        # A non-OK watchdog status is authoritative. Even if a killed child
        # happened to write a per-instance record before it exited, it is not
        # a completed run and must never leak CERTIFIED/FALSIFIED downstream.
        child_per_instance_json = per_instance_json
        per_instance_json = _write_synthetic_per_instance(
            benchmark=benchmark, instance_id=instance_id, status=status,
            cli_normalized=cli_norm, wall_s=wall_s, peak_rss_mb=peak_rss_mb,
            out_dir=out_dir, returncode=returncode,
            strict_bounded_failure=strict_bounded_failure,
            superseded_child_per_instance_json=(
                str(child_per_instance_json) if child_per_instance_json else None
            ),
        )

    stdout_tail = ""
    try:
        with open(stdout_path, "rb") as f:
            data = f.read()
        stdout_tail = data[-2000:].decode("utf-8", errors="replace")
    except OSError:
        pass

    return WatchdogResult(
        benchmark=benchmark, instance_id=instance_id,
        status=status, cli_normalized=cli_norm,
        wall_s=wall_s, peak_rss_mb=peak_rss_mb,
        returncode=returncode,
        per_instance_json=str(per_instance_json) if per_instance_json else None,
        out_dir=str(out_dir), stdout_tail=stdout_tail,
    )


# ---------------------------------------------------------------------------
# CLI


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one or more ACT instances under a process-level "
                    "wall-clock + RSS watchdog."
    )
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--instance-ids", required=True,
                    help="comma-separated official instance ids")
    ap.add_argument("--wall-s", type=float, required=True,
                    help="per-instance wall budget seconds (excluding startup grace)")
    ap.add_argument("--rss-cap-gb", type=float, default=None,
                    help="aggregate process-tree RSS cap in GiB (default: disabled)")
    ap.add_argument("--startup-grace-s", type=float, default=5.0)
    ap.add_argument("--poll-interval-s", type=float, default=1.0)
    ap.add_argument("--grace-kill-s", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--canonical-root", required=True, type=Path)
    ap.add_argument("--python-exe",
                    default="/data1/Kane/miniconda3/envs/act-py312/bin/python")
    ap.add_argument(
        "--raw-verdicts", action="store_true",
        help="Do not enable ACT_FAL_RECEIPT_FORMAL in the child process. "
             "Use this for GPU/CPU parity or throughput sweeps where the "
             "requested output is the verifier's raw CLI verdicts. The child "
             "still writes per_instance JSON via ACT_FORMAL_RESULTS_DIR; "
             "FALSIFIED rows should be audited separately before paper-grade "
             "sound SAT accounting.",
    )
    ap.add_argument(
        "--strict-bounded-failure", action="store_true",
        help="Treat any watchdog termination (including UNKNOWN_TIMEOUT / "
             "UNKNOWN_RESOURCE_LIMIT) as a non-zero exit. Default: bounded "
             "UNKNOWN is an acceptable auditable verdict and only verifier "
             "crashes / spawn failures cause non-zero exit.",
    )
    args = ap.parse_args()

    config = WatchdogConfig(
        wall_s=args.wall_s, rss_cap_gb=args.rss_cap_gb,
        startup_grace_s=args.startup_grace_s,
        poll_interval_s=args.poll_interval_s,
        grace_kill_s=args.grace_kill_s,
    )

    iids = [int(s) for s in args.instance_ids.split(",") if s.strip()]
    results: List[WatchdogResult] = []
    for iid in iids:
        r = run_instance(
            benchmark=args.benchmark, instance_id=iid, config=config,
            out_dir=args.out_dir, canonical_root=args.canonical_root,
            python_exe=args.python_exe, device=args.device, dtype=args.dtype,
            formal_mode=(not args.raw_verdicts),
            strict_bounded_failure=args.strict_bounded_failure,
        )
        results.append(r)
        print(json.dumps(asdict(r), default=str, indent=2))

    summary_path = args.out_dir / "watchdog_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_doc = {
        "schema_version": 2,
        "formal_mode": not bool(args.raw_verdicts),
        "raw_verdicts": bool(args.raw_verdicts),
        "strict_bounded_failure": bool(args.strict_bounded_failure),
        "counts": {
            "OK": sum(1 for r in results if r.status == "OK"),
            "UNKNOWN_TIMEOUT": sum(1 for r in results if r.status == "UNKNOWN_TIMEOUT"),
            "UNKNOWN_RESOURCE_LIMIT": sum(1 for r in results if r.status == "UNKNOWN_RESOURCE_LIMIT"),
            "ERROR": sum(1 for r in results
                         if not is_acceptable(r.status, strict_bounded_failure=False)),
        },
        "results": [asdict(r) for r in results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_doc, f, default=str, indent=2)
    print(f"[watchdog] summary written: {summary_path}")
    # Bounded UNKNOWN is acceptable by default; only verifier crashes etc.
    # flip the exit code. With --strict-bounded-failure, any non-OK fails.
    all_acceptable = all(
        is_acceptable(r.status, strict_bounded_failure=args.strict_bounded_failure)
        for r in results
    )
    return 0 if all_acceptable else 1


if __name__ == "__main__":
    sys.exit(main())
