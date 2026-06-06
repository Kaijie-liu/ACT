"""Resume Goal 2 pilot using subprocess-isolated per-iid runs.

Reason: iid 113 OOM-killed the parent process. Subprocess isolation gives
each iid a clean memory state; OOM only kills that subprocess and we
continue to the next iid.

Resumable: skips iids that already have a receipt.

Per advisor: K=∞ for headline. If subprocess OOMs at K=∞, mark UNK and
move on; do NOT auto-fallback to smaller K (that would silently mix
headline numbers).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ACT_ROOT = Path(__file__).resolve().parents[2]

CIFAR_IIDS = [0,2,6,8,24,29,57,72,86,113,118,130,145,156,168,180,185,190,195,199]
TINY_IIDS  = [0,6,12,24,30,45,60,73,86,99,116,130,145,158,170,180,190,195,198,199]

SINGLE_RUNNER = '''
import sys, json, time, resource
sys.path.insert(0, "/data1/Kane/ACT")
from research.sc_hz.goal2_phase_d_pilot import run_one_iid
from pathlib import Path
bench, iid, K, wall_s, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
r = run_one_iid(bench, iid, K, wall_s, Path(out))
print(f"DONE iid={iid} verdict={r.get('verdict')} wall={r.get('wall_total_s',0):.0f}s "
      f"ng={r.get('output_ng',0)} rss={r.get('peak_rss_gb',0):.1f}GB "
      f"max_excess={r.get('max_excess', 'n/a')}")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=100000)
    ap.add_argument("--wall-per-iid-s", type=int, default=600)
    ap.add_argument("--subprocess-timeout-s", type=int, default=800)
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    work = [("cifar100_2024", i) for i in CIFAR_IIDS] + \
            [("tinyimagenet_2024", i) for i in TINY_IIDS]
    print(f"Resume pilot: {len(work)} iids, K={args.K}, wall_subprocess={args.subprocess_timeout_s}s",
          flush=True)

    # Write the single-iid runner to a tempfile
    runner_path = out_root / "_single_iid_runner.py"
    runner_path.write_text(SINGLE_RUNNER)

    t0 = time.perf_counter()
    n_skipped = 0; n_oom = 0; n_done = 0
    for i, (bench, iid) in enumerate(work):
        receipt = out_root / bench / f"iid{iid:04d}.json"
        if receipt.exists():
            try:
                r = json.load(open(receipt))
                if r.get("verdict") in {"A_CONFIRMED", "CERT", "PHANTOM_LP_SAT", "UNK"}:
                    n_skipped += 1
                    continue
            except Exception:
                pass
        # Subprocess run
        print(f"\n[{i+1}/{len(work)}] {bench} iid {iid} (elapsed={time.perf_counter()-t0:.0f}s)...",
              flush=True)
        cmd = [
            "/data1/Kane/miniconda3/envs/act-py312/bin/python",
            str(runner_path),
            bench, str(iid), str(args.K), str(args.wall_per_iid_s), str(out_root),
        ]
        try:
            proc = subprocess.run(
                cmd, timeout=args.subprocess_timeout_s,
                capture_output=True, text=True,
            )
            n_done += 1
            tail = proc.stdout.strip().split("\n")[-3:] if proc.stdout else []
            for line in tail:
                print(f"  {line}", flush=True)
            if proc.returncode != 0:
                print(f"  RC={proc.returncode}, stderr: {proc.stderr[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            n_oom += 1
            # Write a fail-closed receipt
            rec = {"bench": bench, "iid": iid, "K": args.K,
                    "verdict": "UNK",
                    "fail_closed_reason": f"subprocess_timeout_{args.subprocess_timeout_s}s"}
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(rec, indent=2))
            print(f"  TIMEOUT after {args.subprocess_timeout_s}s — marked UNK",
                  flush=True)
        except Exception as e:
            n_oom += 1
            rec = {"bench": bench, "iid": iid, "K": args.K,
                    "verdict": "UNK",
                    "fail_closed_reason": f"subprocess_error: {type(e).__name__}: {str(e)[:200]}"}
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(rec, indent=2))
            print(f"  subprocess error: {e}", flush=True)

    print(f"\nresume run done: n_done={n_done}, n_skipped(existing)={n_skipped}, n_oom_or_timeout={n_oom}",
          flush=True)
    # Aggregate full summary
    results = []
    for bench, iid in work:
        receipt = out_root / bench / f"iid{iid:04d}.json"
        if receipt.exists():
            results.append(json.load(open(receipt)))
    by_bench = {}
    for b in ["cifar100_2024", "tinyimagenet_2024"]:
        sub = [r for r in results if r["bench"] == b]
        cc = Counter(r.get("verdict") for r in sub)
        a_iids = sorted(r["iid"] for r in sub if r.get("verdict") == "A_CONFIRMED")
        c_iids = sorted(r["iid"] for r in sub if r.get("verdict") == "CERT")
        by_bench[b] = {"n": len(sub), "verdict_counts": dict(cc),
                         "a_iids": a_iids, "cert_iids": c_iids}
        print(f"{b}: n={len(sub)}, {dict(cc)}", flush=True)
    summary = {
        "K": args.K, "subprocess_timeout_s": args.subprocess_timeout_s,
        "by_bench": by_bench,
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {out_root}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
