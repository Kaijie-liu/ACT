"""Phase 1 runner: manifest-driven audit-strict batch verifier.

Built 2026-05-24 per advisor Round 8: do NOT launch Phase 1 by hand
with two raw CLI commands. Instead this module:

  1. Loads the frozen P1 watchlist seed (round4_sat61_proof_only.csv)
     and verifies its SHA256 verbatim before any other action.
  2. Captures the code snapshot: git HEAD + dirty diff SHA. Refuses to
     start if the worktree is dirty unless ``--allow-dirty-worktree``
     is passed; even then the dirty diff text is saved alongside the
     results so the run is bit-exactly reconstructable.
  3. Invokes the ACT formal CLI separately for each benchmark with
     ``--instance-ids`` listing the seed's official iids.
  4. Reads the per-benchmark ``per_instance_<bench>_<ts>.json`` written
     by the CLI, joins on ``(benchmark, official_instance_id)`` with
     the seed, and computes an aggregate Phase 1 verdict.
  5. Writes ``phase1_summary.json`` + ``phase1_per_instance.csv`` to
     the run dir and returns a non-zero exit on ANY of:
        - coverage_gap (seed instance missing from CLI output)
        - duplicate (benchmark, iid)
        - extra_instance (CLI ran something not in seed)
        - benchmark subprocess nonzero exit
        - any FALSIFIED / ERROR_RECEIPT / ERROR_INTERNAL_INCONSISTENCY
        - any ERROR (verifier/config exception)
        - any per-instance reportable_status outside {CERTIFIED, UNKNOWN}
          (the seed is 61 boundary-negative controls; a real FAL here
          would be a major scientific finding but ALSO a Phase 1 STOP).

The runner is itself the single point of audit truth: passing means
ALL of the above contracts hold; failing means the run cannot enter
the paper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PHASE1_SEED_CSV = "/data1/Kane/HyZor/audit_results/round4_sat61_proof_only.csv"
PHASE1_SEED_SHA = (
    "33d854e32a8c0e9a7a3c8de5e3c055f1219f3c7f5f122ec0c2c93802e428848a"
)
PHASE1_EXPECTED_BENCHMARKS = ("cifar100_2024", "tinyimagenet_2024")


# ---------------------------------------------------------------------------
# Pure helpers (testable without subprocess / git)
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def load_seed(csv_path: Path, expected_sha: str) -> List[dict]:
    """Read the seed manifest after SHA verification.

    Raises ValueError on SHA mismatch — Phase 1 manifest MUST be the
    frozen P1 watchlist verbatim, not an edited variant.
    """
    actual = sha256_file(csv_path)
    if actual != expected_sha:
        raise ValueError(
            f"seed SHA mismatch: expected {expected_sha}, got {actual}. "
            "Phase 1 manifest must be the frozen P1 watchlist verbatim."
        )
    rows: List[dict] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "benchmark": r["benchmark"].strip(),
                "official_instance_id": int(r["instance_id"]),
                "official_zero": r["official_zero"].strip().lower(),
                "official_small": r["official_small"].strip().lower(),
            })
    return rows


def select_iids_by_bench(seed: List[dict]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for r in seed:
        out.setdefault(r["benchmark"], []).append(r["official_instance_id"])
    for b in out:
        out[b] = sorted(set(out[b]))
    return out


def join_with_seed(
    seed: List[dict], per_instance_by_bench: Dict[str, List[dict]],
) -> Tuple[List[dict], List[dict]]:
    """Join seed rows with per-instance CLI output.

    Returns (joined_rows, audit_findings).
    audit_findings carries any structural problem (missing, duplicate,
    extra) but does NOT compute the pass/fail verdict — that's the job
    of ``compute_phase1_verdict``.
    """
    joined: List[dict] = []
    findings: List[dict] = []
    seen_in_seed = set()
    for s in seed:
        key = (s["benchmark"], s["official_instance_id"])
        if key in seen_in_seed:
            findings.append({"kind": "duplicate_seed_entry",
                             "detail": f"{s['benchmark']} iid={s['official_instance_id']}"})
        seen_in_seed.add(key)
        per_inst = per_instance_by_bench.get(s["benchmark"], [])
        match = [r for r in per_inst
                 if int(r.get("official_instance_id", -1)) == s["official_instance_id"]]
        if not match:
            findings.append({
                "kind": "missing_instance",
                "detail": f"{s['benchmark']} iid={s['official_instance_id']} not in per_instance JSON",
            })
            joined.append({**s, "joined": False})
            continue
        if len(match) > 1:
            findings.append({
                "kind": "duplicate_per_instance",
                "detail": f"{s['benchmark']} iid={s['official_instance_id']} appears {len(match)} times",
            })
        r = match[0]
        joined.append({
            **s,
            "joined": True,
            "internal_status": r.get("internal_status"),
            "reportable_status": r.get("reportable_status"),
            "cli_normalized": r.get("cli_normalized"),
            "count_bucket": r.get("count_bucket"),
            "wall_s": r.get("wall_s"),
            "q_receipts": r.get("q_receipts") or [],
            "q_statuses": r.get("q_statuses") or [],
            "q_reportables": r.get("q_reportables") or [],
        })
    # Flag per_instance entries NOT in seed (unexpected coverage)
    seed_keys = {(s["benchmark"], s["official_instance_id"]) for s in seed}
    for bench, rows in per_instance_by_bench.items():
        for r in rows:
            k = (bench, int(r.get("official_instance_id", -1)))
            if k not in seed_keys:
                findings.append({
                    "kind": "extra_instance",
                    "detail": f"{bench} iid={r.get('official_instance_id')} ran but not in seed",
                })
    return joined, findings


def compute_phase1_verdict(
    joined: List[dict], findings: List[dict],
    bench_exits: Dict[str, int], bench_counts: Dict[str, dict],
) -> Tuple[str, List[str]]:
    """Aggregate verdict for the whole Phase 1 batch.

    Returns ("PASS"|"FAIL", list_of_failure_reasons). Pass criterion
    (ALL must hold):
      - every seed instance was joined to a per_instance entry
      - no duplicate / extra / structural finding
      - every benchmark subprocess returned exit 0
      - counts.FALSIFIED == 0 across all benchmarks
      - counts.ERROR_RECEIPT == 0
      - counts.ERROR_INTERNAL_INCONSISTENCY == 0
      - counts.ERROR == 0
      - reportable_status for every joined row in {CERTIFIED, UNKNOWN}
    """
    reasons: List[str] = []
    n_missing = sum(1 for r in joined if not r.get("joined"))
    if n_missing:
        reasons.append(f"coverage_gap: {n_missing} seed instances missing from per_instance")
    for f in findings:
        reasons.append(f"{f['kind']}: {f['detail']}")
    for b, ec in bench_exits.items():
        if ec != 0:
            reasons.append(f"benchmark_subprocess_nonzero_exit: {b} exit={ec}")
    for b, c in bench_counts.items():
        for key in ("FALSIFIED", "ERROR_RECEIPT",
                    "ERROR_INTERNAL_INCONSISTENCY", "ERROR"):
            v = int(c.get(key, 0) or 0)
            if v > 0:
                reasons.append(f"forbidden_count: {b} {key}={v}")
    for r in joined:
        if not r.get("joined"):
            continue
        rs = r.get("reportable_status")
        if rs not in ("CERTIFIED", "UNKNOWN"):
            reasons.append(
                f"forbidden_reportable: {r['benchmark']} "
                f"iid={r['official_instance_id']} reportable={rs}"
            )
    return ("FAIL" if reasons else "PASS", reasons)


# ---------------------------------------------------------------------------
# IO / git / subprocess wrappers (kept thin so the pure helpers above
# can be tested without mocks)
# ---------------------------------------------------------------------------


def capture_code_snapshot(repo_root: Path, allow_dirty: bool) -> dict:
    """Capture git HEAD + dirty diff SHA.

    Refuses to start (raises RuntimeError) if the worktree is dirty
    unless ``allow_dirty=True``. Even then the full diff text is
    returned so the caller can persist it alongside the results.
    """
    def _git(args):
        return subprocess.check_output(
            ["git"] + args, cwd=str(repo_root), stderr=subprocess.DEVNULL,
        ).decode()
    head = _git(["rev-parse", "HEAD"]).strip()
    status = _git(["status", "--porcelain"])
    dirty = bool(status.strip())
    if dirty and not allow_dirty:
        raise RuntimeError(
            f"phase1_runner refuses to start: worktree at {head[:12]} is dirty.\n"
            f"Either commit the changes (preferred) or pass --allow-dirty-worktree.\n"
            f"git status --porcelain (first 1000 chars):\n{status[:1000]}"
        )
    diff_text = _git(["diff", "HEAD"]) if dirty else ""
    untracked = (_git(["ls-files", "--others", "--exclude-standard"])
                 if dirty else "")
    diff_sha = hashlib.sha256(
        (diff_text + "\n---UNTRACKED---\n" + untracked).encode()
    ).hexdigest() if dirty else ""
    return {
        "git_head": head,
        "worktree_dirty": dirty,
        "dirty_diff_sha256": diff_sha,
        "dirty_diff_text": diff_text,
        "untracked_files": untracked,
    }


def invoke_cli_for_bench(
    *, benchmark: str, iids: List[int], run_dir: Path,
    canonical_root: Path, timeout_s: int, dtype: str, device: str,
    python_exe: str, log_dir: Path,
) -> Tuple[int, Optional[Path]]:
    """Run ACT CLI for one benchmark with --instance-ids. Returns
    (exit_code, per_instance_json_path).

    The CLI writes ``per_instance_<bench>_<UTC>.json`` to
    ACT_FORMAL_RESULTS_DIR; we return the latest matching file.
    """
    if not iids:
        return 0, None
    iids_str = ",".join(str(i) for i in sorted(iids))
    env = dict(os.environ)
    env.update({
        "ACT_VNNLIB_ROOT": str(canonical_root),
        "ACT_FAL_RECEIPT_FORMAL": "1",
        "ACT_FAL_RECEIPT_DIR": str(run_dir),
        "ACT_FORMAL_RESULTS_DIR": str(run_dir),
        "PYTHONPATH": "/data1/Kane/ACT",
    })
    cmd = [
        python_exe, "-m", "act.pipeline",
        "--verify", "vnnlib",
        "--category", benchmark,
        "--instance-ids", iids_str,
        "--max-instances", str(len(iids)),
        "--timeout", str(timeout_s),
        "--device", device,
        "--dtype", dtype,
        "--solvers", "hybridz",
    ]
    log_path = log_dir / f"{benchmark}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as logf:
        p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    matches = sorted(run_dir.glob(f"per_instance_{benchmark}_*.json"))
    return p.returncode, (matches[-1] if matches else None)


def write_outputs(
    *, run_dir: Path, snapshot: dict, seed_sha_actual: str,
    joined: List[dict], findings: List[dict],
    bench_exits: Dict[str, int], bench_counts: Dict[str, dict],
    verdict: str, reasons: List[str],
) -> Tuple[Path, Path]:
    """Write phase1_per_instance.csv + phase1_summary.json.

    Also dumps the dirty diff text + untracked list as separate files
    so reviewers can re-apply the exact code state.
    """
    if snapshot.get("dirty_diff_text"):
        (run_dir / "dirty_diff.patch").write_text(snapshot["dirty_diff_text"])
    if snapshot.get("untracked_files"):
        (run_dir / "untracked_files.list").write_text(snapshot["untracked_files"])
    csv_path = run_dir / "phase1_per_instance.csv"
    cols = [
        "benchmark", "official_instance_id", "official_zero", "official_small",
        "joined", "internal_status", "reportable_status", "cli_normalized",
        "count_bucket", "wall_s", "q_receipts_n",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in joined:
            w.writerow([
                r.get("benchmark"), r.get("official_instance_id"),
                r.get("official_zero"), r.get("official_small"),
                int(bool(r.get("joined"))),
                r.get("internal_status", ""), r.get("reportable_status", ""),
                r.get("cli_normalized", ""), r.get("count_bucket", ""),
                r.get("wall_s", ""),
                len(r.get("q_receipts") or []),
            ])
    summary = {
        "schema_version": 1,
        "phase": "phase1_round4_sat61_proof_only",
        "seed_path": PHASE1_SEED_CSV,
        "seed_sha256_expected": PHASE1_SEED_SHA,
        "seed_sha256_actual": seed_sha_actual,
        "code_snapshot": {
            "git_head": snapshot["git_head"],
            "worktree_dirty": snapshot["worktree_dirty"],
            "dirty_diff_sha256": snapshot["dirty_diff_sha256"],
        },
        "per_benchmark_exit_codes": bench_exits,
        "per_benchmark_counts": bench_counts,
        "audit_findings": findings,
        "verdict": verdict,
        "failure_reasons": reasons,
        "n_seed": len(joined),
        "n_joined": sum(1 for r in joined if r.get("joined")),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = run_dir / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary_path, csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="fresh isolated directory; must not exist")
    parser.add_argument("--canonical-root", required=True, type=Path,
                        help="VNN-COMP canonical root, e.g. "
                             "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
    parser.add_argument("--seed-csv", default=PHASE1_SEED_CSV)
    parser.add_argument("--seed-sha256", default=PHASE1_SEED_SHA)
    parser.add_argument("--repo-root", default="/data1/Kane/ACT")
    parser.add_argument("--allow-dirty-worktree", action="store_true",
                        help="Permit running with uncommitted changes. Diff is "
                             "still SHA-recorded and dumped to run dir.")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--python-exe",
                        default="/data1/Kane/miniconda3/envs/act-py312/bin/python")
    args = parser.parse_args(argv)

    if args.run_dir.exists():
        print(f"ERROR: run_dir already exists: {args.run_dir}", file=sys.stderr)
        return 2
    args.run_dir.mkdir(parents=True)

    seed_sha_actual = sha256_file(args.seed_csv)
    print(f"[phase1] seed_sha256 = {seed_sha_actual}")
    seed = load_seed(Path(args.seed_csv), args.seed_sha256)
    print(f"[phase1] loaded {len(seed)} seed instances")

    # Code snapshot
    try:
        snapshot = capture_code_snapshot(
            Path(args.repo_root), allow_dirty=args.allow_dirty_worktree
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"[phase1] git_head={snapshot['git_head'][:12]}  "
          f"dirty={snapshot['worktree_dirty']}  "
          f"diff_sha={snapshot['dirty_diff_sha256'][:12] if snapshot['worktree_dirty'] else '-'}")

    # Save the seed CSV verbatim alongside the run
    (args.run_dir / "seed.csv").write_bytes(
        Path(args.seed_csv).read_bytes()
    )
    (args.run_dir / "seed_sha256.txt").write_text(
        f"{seed_sha_actual}  seed.csv\n"
    )

    # Per-benchmark CLI runs
    iids_by_bench = select_iids_by_bench(seed)
    bench_exits: Dict[str, int] = {}
    bench_counts: Dict[str, dict] = {}
    per_instance_by_bench: Dict[str, List[dict]] = {}
    log_dir = args.run_dir / "logs"
    log_dir.mkdir()
    for bench, iids in sorted(iids_by_bench.items()):
        print(f"[phase1] running {bench}  n_iids={len(iids)}", flush=True)
        ec, json_path = invoke_cli_for_bench(
            benchmark=bench, iids=iids, run_dir=args.run_dir,
            canonical_root=args.canonical_root,
            timeout_s=args.timeout_s, dtype=args.dtype, device=args.device,
            python_exe=args.python_exe, log_dir=log_dir,
        )
        bench_exits[bench] = ec
        if json_path is not None:
            data = json.loads(json_path.read_text())
            bench_counts[bench] = data.get("counts", {})
            per_instance_by_bench[bench] = data.get("per_instance", [])
            print(f"[phase1]   exit={ec}  counts={bench_counts[bench]}")
        else:
            bench_counts[bench] = {}
            per_instance_by_bench[bench] = []
            print(f"[phase1]   exit={ec}  NO per_instance JSON found")

    joined, findings = join_with_seed(seed, per_instance_by_bench)
    verdict, reasons = compute_phase1_verdict(
        joined, findings, bench_exits, bench_counts,
    )

    summary_path, csv_path = write_outputs(
        run_dir=args.run_dir, snapshot=snapshot,
        seed_sha_actual=seed_sha_actual, joined=joined, findings=findings,
        bench_exits=bench_exits, bench_counts=bench_counts,
        verdict=verdict, reasons=reasons,
    )
    print()
    print(f"[phase1] VERDICT: {verdict}")
    if reasons:
        for r in reasons[:20]:
            print(f"  - {r}")
        if len(reasons) > 20:
            print(f"  ... ({len(reasons) - 20} more)")
    print(f"[phase1] summary:    {summary_path}")
    print(f"[phase1] per_inst:   {csv_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
