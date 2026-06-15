#!/usr/bin/env python
"""HZ full-run driver: verify all SUPPORTED VNN-COMP 2025 benchmarks with the
engine's best config, mem-safe + parallel + detailed logs.

Design for "fully parallel, never OOM-kill VS Code":
  * ONE subprocess per instance (hz_full_worker.py) -> total mem isolation; a
    runaway instance dies alone (RLIMIT_AS backstop inside the worker), never the
    host. The driver itself holds almost no memory.
  * Bounded ThreadPoolExecutor launches the subprocesses; per-instance hard wall
    timeout (subprocess kill). Host-memory governor pauses new launches if free
    RAM drops below a floor.
  * Best config (all defaults-optimal): tight_bounds=True, K=1, reduce on,
    hz_certify_spec verdict (scipy/HiGHS, Gurobi off), ORT = soundness guard only.
  * Detailed logs: per-benchmark JSONL (one line/instance) + master CSV + a live
    progress log + a final summary.
"""
import os, sys, json, time, csv, subprocess, threading, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/data1/Kane/ACT"
PY = "/data1/Kane/miniconda3/envs/act-py312/bin/python"
WORKER = f"{REPO}/scripts/hz_full_worker.py"
BENCH_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")

# Supported benchmarks present in this set, with per-benchmark wall timeout (s).
# Config is otherwise uniform: device=cpu, cap=4096 (narrow+affine carry; wide
# conv maps drop to the sound interval fallback -- no CPU OOM), mem_gb=8 backstop.
BENCHMARKS = {
    "malbeware":            180,
    "cersyve":              180,
    "safenlp_2024":         120,
    "sat_relu":             120,
    "tllverifybench_2023":  320,
    "acasxu_2023":          300,
    "cora_2024":            250,
    "lsnc_relu":            120,
    "linearizenn_2024":     180,
    "soundnessbench":       180,
    "relusplitter":         320,
    "dist_shift_2023":      250,
    "metaroom_2023":        250,
    "cgan_2023":            300,
    "collins_aerospace_benchmark": 180,
    # conv-heavy FULL-supported (ResNet/YOLO) -- at cap=4096 wide conv maps drop
    # to the sound interval fallback (cifar100 is the documented BaB-forward
    # ceiling); included for completeness of the supported set.
    "cifar100_2024":        250,
    "tinyimagenet_2024":    250,
    "yolo_2023":            300,
}

def n_instances(bench):
    f = BENCH_ROOT / bench / "instances.csv"
    return sum(1 for l in open(f) if l.strip())

_mem_lock = threading.Lock()
def free_gb():
    with open("/proc/meminfo") as f:
        m = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return m.get("MemAvailable", 0) / 1024**2

def run_one(bench, iid, timeout, cap, mem_gb, device, milp_timeout, logf):
    cmd = [PY, WORKER, bench, str(iid), "--device", device, "--cap", str(cap),
           "--mem-gb", str(mem_gb), "--milp-timeout", str(milp_timeout)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
        # the worker prints exactly one json line (last '{...}' line of stdout)
        line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), None)
        if line:
            r = json.loads(line)
        else:
            r = {"bench": bench, "iid": iid, "verdict": "ERROR",
                 "err": "no-json:" + (p.stderr.splitlines()[-1][:160] if p.stderr.strip() else "empty")}
    except subprocess.TimeoutExpired:
        r = {"bench": bench, "iid": iid, "verdict": "TIMEOUT", "time_s": round(time.time() - t0, 1)}
    except Exception as e:
        r = {"bench": bench, "iid": iid, "verdict": "ERROR", "err": f"{type(e).__name__}:{str(e)[:120]}"}
    r.setdefault("time_s", round(time.time() - t0, 1))
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--mem-gb", type=float, default=16.0,
                    help="RLIMIT_AS virtual cap/worker. 8 is too tight: torch+ort+"
                         "HiGHS reserve large VIRTUAL space, so thread-local alloc "
                         "ABORTs even though actual RSS is ~1GB. 16 fixes it without "
                         "raising real memory use (RLIMIT_AS caps virtual, not RSS).")
    ap.add_argument("--milp-timeout", type=float, default=15.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-floor-gb", type=float, default=20.0, help="pause launches below this free RAM")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", default=None, help="comma list to restrict benchmarks")
    ap.add_argument("--max-per-bench", type=int, default=0, help="0=all")
    a = ap.parse_args()

    stamp = subprocess.run(["date", "+%Y%m%d_%H%M%S"], capture_output=True, text=True).stdout.strip()
    outdir = Path(a.outdir or f"{REPO}/audit_results/hz_fullrun_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    masterf = outdir / "results_master.csv"
    logp = outdir / "run.log"
    benches = [b for b in BENCHMARKS if (not a.only or b in a.only.split(","))]

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logp, "a") as f: f.write(line + "\n")

    total = sum(min(n_instances(b), a.max_per_bench or 10**9) for b in benches)
    log(f"=== HZ full run: {len(benches)} benchmarks, {total} instances, "
        f"workers={a.workers}, cap={a.cap}, mem_gb={a.mem_gb}, device={a.device} ===")
    log(f"outdir={outdir}")

    master = open(masterf, "w", newline="")
    mw = csv.writer(master)
    mw.writerow(["bench", "iid", "verdict", "margin", "n_queries", "nc", "ng", "nb",
                 "time_s", "verify_s", "gt_cex", "p0", "err"])
    master.flush()

    grand = {}
    for bench in benches:
        N = n_instances(bench)
        if a.max_per_bench: N = min(N, a.max_per_bench)
        to = BENCHMARKS[bench]
        benchf = open(outdir / f"{bench}.jsonl", "w")
        counts = {"CERT": 0, "ADV": 0, "UNKNOWN": 0, "TIMEOUT": 0, "ERROR": 0, "P0": 0}
        t0 = time.time()
        log(f"--- {bench}: {N} instances, timeout={to}s ---")
        done = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {}
            for iid in range(N):
                # memory governor: block until free RAM is above the floor
                while free_gb() < a.mem_floor_gb:
                    log(f"  [mem-governor] free RAM {free_gb():.0f}GB < {a.mem_floor_gb}GB, pausing 10s")
                    time.sleep(10)
                futs[ex.submit(run_one, bench, iid, to, a.cap, a.mem_gb, a.device, a.milp_timeout, logp)] = iid
            for fut in as_completed(futs):
                r = fut.result()
                v = r.get("verdict", "ERROR")
                if r.get("p0"):
                    counts["P0"] += 1; counts["UNKNOWN"] += 1  # downgraded, flagged
                elif v == "CERT": counts["CERT"] += 1
                elif v == "ADV": counts["ADV"] += 1
                elif v == "TIMEOUT": counts["TIMEOUT"] += 1
                elif v.startswith("UNKNOWN"): counts["UNKNOWN"] += 1
                else: counts["ERROR"] += 1
                benchf.write(json.dumps(r) + "\n"); benchf.flush()
                mw.writerow([r.get("bench"), r.get("iid"), v, r.get("margin"), r.get("n_queries"),
                             r.get("nc"), r.get("ng"), r.get("nb"), r.get("time_s"),
                             r.get("verify_s"), r.get("gt_cex"), r.get("p0"), r.get("err")])
                master.flush()
                done += 1
                if done % 25 == 0 or done == N:
                    log(f"  {bench}: {done}/{N} | CERT={counts['CERT']} ADV={counts['ADV']} "
                        f"UNK={counts['UNKNOWN']} TO={counts['TIMEOUT']} ERR={counts['ERROR']} "
                        f"P0={counts['P0']} | freeRAM={free_gb():.0f}GB")
        benchf.close()
        dt = time.time() - t0
        grand[bench] = dict(counts, N=N, time_s=round(dt, 1))
        log(f"=== {bench} DONE in {dt:.0f}s: CERT={counts['CERT']} ADV={counts['ADV']} "
            f"UNKNOWN={counts['UNKNOWN']} TIMEOUT={counts['TIMEOUT']} ERROR={counts['ERROR']} "
            f"P0={counts['P0']} (V+A={counts['CERT']+counts['ADV']}) ===")

    master.close()
    # summary
    sumf = outdir / "summary.csv"
    with open(sumf, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bench", "N", "CERT", "ADV", "V+A", "UNKNOWN", "TIMEOUT", "ERROR", "P0", "time_s"])
        tc = ta = tu = tt = te = tp = tn = 0
        for b, c in grand.items():
            w.writerow([b, c["N"], c["CERT"], c["ADV"], c["CERT"] + c["ADV"], c["UNKNOWN"],
                        c["TIMEOUT"], c["ERROR"], c["P0"], c["time_s"]])
            tc += c["CERT"]; ta += c["ADV"]; tu += c["UNKNOWN"]; tt += c["TIMEOUT"]; te += c["ERROR"]; tp += c["P0"]; tn += c["N"]
        w.writerow(["TOTAL", tn, tc, ta, tc + ta, tu, tt, te, tp, ""])
    log(f"=== GRAND TOTAL: {tn} | CERT={tc} ADV={ta} V+A={tc+ta} UNKNOWN={tu} TIMEOUT={tt} ERROR={te} P0={tp} ===")
    log(f"summary={sumf}")

if __name__ == "__main__":
    main()
