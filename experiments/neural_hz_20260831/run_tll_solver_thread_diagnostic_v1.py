"""Fixed-order whole-family thread comparison; diagnostic only, no promotion."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neural_hz_20260831 import run_tll_current_source_replay_20260905_v2 as previous
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
TRACE_WORKER = EXPERIMENT / "milp_trace_worker_v1.py"
NEW_SOURCES = (
    Path(__file__), TRACE_WORKER,
    EXPERIMENT / "test_milp_trace_worker_v1.py",
    EXPERIMENT / "test_tll_solver_thread_diagnostic_v1.py",
    EXPERIMENT / "TLL_SOLVER_THREAD_DIAGNOSTIC_PREREG_20260905.md",
)


def classify(data):
    """A rejected proposal is not a reported ADV, and never earns a solve."""
    verdict = data.get("verdict")
    validations = data.get("concrete_validations", [])
    rejected = sum(item.get("valid") is not True for item in validations)
    valid_witness = any(all(item.get(key) is True for key in ("valid", "input_ok", "violates"))
                        for item in validations)
    reported_invalid = verdict == "ADV" and (rejected != 0 or not valid_witness)
    if verdict not in {"CERT", "ADV", "UNKNOWN", "ERROR"}:
        raise ValueError("unexpected verdict")
    if verdict in {"CERT", "ADV"} and data.get("output_hz_exact") is not True:
        raise ValueError("solved result lacks exact terminal HZ")
    return {"raw_verdict": verdict, "counted_verdict": "ERROR" if reported_invalid else verdict,
            "reported_invalid_adv": int(reported_invalid),
            "rejected_solver_proposals": rejected,
            "concrete_validation_count": len(validations)}


def check_drift(frozen):
    return (previous.shadow_worker._provenance(ROOT) != frozen["provenance"]
            or any(_sha256(ROOT / p) != h for p, h in frozen["extra_code_sha256"].items())
            or any(_sha256(ROOT / p) != h for p, h in frozen["diagnostic_sources"].items())
            or any(_sha256(Path(p)) != h for j in frozen["jobs"] for p, h in j["assets"].items()))


def run_one(directory, job, frozen, policy):
    iid = job["iid"]
    output, trace = (directory / f"iid{iid:02d}{suffix}" for suffix in (".json", ".trace.json"))
    record = {"iid": iid, "policy": policy, "formal_gain": 0, "counted_verdict": "ERROR"}
    command = [sys.executable, str(TRACE_WORKER), "--trace-output", str(trace),
               "--highs-thread-policy", policy, previous.FAMILY, str(iid),
               "--arm", previous.ARM, "--representation", "sparse", "--act-root", str(ROOT),
               "--bench-root", str(previous.BENCH_ROOT), "--vnnlib-root", str(previous.SPEC_ROOT),
               "--solver-timeout", "45", "--memory-gb", "16", "--output", str(output)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    try:
        if check_drift(frozen):
            raise ValueError("source or asset drift before execution")
        with (directory / f"iid{iid:02d}.log").open("x") as stream:
            child = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=240, check=False)
        record["exit_code"] = child.returncode
        if child.returncode:
            raise ValueError("nonzero worker exit")
        data = json.loads(output.read_text())
        diagnostic = json.loads(trace.read_text())
        if (data.get("iid"), data.get("bench"), data.get("arm"), data.get("provenance")) != (
            iid, previous.FAMILY, previous.ARM, frozen["provenance"]
        ):
            raise ValueError("worker identity/provenance mismatch")
        if (diagnostic.get("worker_provenance") != frozen["provenance"]
                or diagnostic.get("policy") != policy
                or diagnostic.get("trace_source_sha256") != frozen["diagnostic_sources"][str(TRACE_WORKER.relative_to(ROOT))]):
            raise ValueError("trace provenance mismatch")
        if not all(c.get("problem_unchanged") is True for c in diagnostic["calls"]):
            raise ValueError("MILP numerical inputs changed during solver call")
        for call in diagnostic["calls"]:
            if call["options"].get("threads") != (1 if policy == "one" else None):
                raise ValueError("thread policy mismatch")
        record.update(classify(data))
        record.update(error=data.get("error"), result_sha256=_sha256(output), trace_sha256=_sha256(trace),
                      verify_s=data.get("verify_s"), worker_wall_s=data.get("worker_wall_s"),
                      embedded_highs_version=diagnostic["embedded_highs_version"],
                      cpu_affinity=diagnostic["cpu_affinity"], calls=diagnostic["calls"])
    except subprocess.TimeoutExpired:
        record.update(counted_verdict="TIMEOUT", error="supervisor_wall_limit")
    except Exception as exc:
        record.update(counted_verdict="ERROR", error=f"{type(exc).__name__}: {exc}")
    record.update(supervisor_wall_s=time.monotonic() - started, command=command)
    _atomic_exclusive_json(directory / f"iid{iid:02d}.exit.json", record)
    return record


def summarize(records, jobs):
    solved = {r["iid"] for r in records if r["counted_verdict"] in {"CERT", "ADV"}}
    formal = {j["iid"] for j in jobs if j["baseline"] in {"CERT", "ADV"}}
    prior = {j["iid"] for j in jobs if j["prior_candidate_solved"]}
    calls = [c for r in records for c in r.get("calls", [])]
    return {"counts": dict(Counter(r["counted_verdict"] for r in records)),
            "completed": len(records), "solved": sorted(solved),
            "lost_formal_solved": sorted(formal - solved),
            "lost_prior_candidate_solved": sorted(prior - solved),
            "newly_solved_vs_formal": sorted(solved - formal),
            "reported_invalid_adv": sum(r.get("reported_invalid_adv", 0) for r in records),
            "rejected_solver_proposals": sum(r.get("rejected_solver_proposals", 0) for r in records),
            "solver_call_count": len(calls),
            "solver_wall_s_sum": sum(c["solve_wall_s"] for c in calls),
            "solver_cpu_s_sum": sum(c["solve_cpu_s"] for c in calls),
            "instrumentation_s_sum": sum(c["pre_solve_instrumentation_s"] + c["post_solve_instrumentation_s"] for c in calls),
            "peak_threads_including_monitor": max((c["observed_peak_threads_including_monitor"] for c in calls), default=0),
            "embedded_highs_versions": sorted({r["embedded_highs_version"] for r in records if "embedded_highs_version" in r}),
            "formal_gain": 0, "diagnostic_only": True, "promotion_passed": False}


def compare(auto, one):
    rows = []
    for a, b in zip(sorted(auto, key=lambda r: r["iid"]), sorted(one, key=lambda r: r["iid"])):
        if a["iid"] != b["iid"]:
            raise ValueError("arm universes differ")
        ac, bc = a.get("calls", []), b.get("calls", [])
        matches = [x["problem"]["sha256"] == y["problem"]["sha256"] for x, y in zip(ac, bc)]
        rows.append({"iid": a["iid"], "auto_verdict": a["counted_verdict"], "one_verdict": b["counted_verdict"],
                     "auto_calls": len(ac), "one_calls": len(bc), "compared_numeric_calls": len(matches),
                     "numeric_call_matches": matches,
                     "all_calls_identical": bool(matches) and len(ac) == len(bc) and all(matches)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.absolute()
    if directory.parent.resolve() != (EXPERIMENT / "results").resolve():
        raise ValueError("new directory must be directly under isolated results")
    if os.path.lexists(directory):
        raise FileExistsError(directory)
    subprocess.run(["sha256sum", "--check", "--quiet", "CHECKPOINT_20260905_SHA256SUMS"],
                   cwd=EXPERIMENT, check=True)
    frozen = previous.preflight()
    frozen.update(schema="tll_solver_thread_diagnostic_v1", diagnostic_only=True,
                  policy_order=["auto", "one"], numeric_threads_scope="OMP/BLAS/MKL only",
                  diagnostic_sources={str(p.relative_to(ROOT)): _sha256(p) for p in NEW_SOURCES})
    directory.mkdir()
    _atomic_exclusive_json(directory / "preregistered.json", frozen)
    arms, summaries = {}, {}
    for policy in frozen["policy_order"]:
        arm_dir = directory / policy
        arm_dir.mkdir()
        records, started = [], time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(run_one, arm_dir, j, frozen, policy) for j in frozen["jobs"]]
            for future in as_completed(pending):
                record = future.result()
                records.append(record)
                print(json.dumps({"policy": policy, "completed": len(records), "iid": record["iid"],
                                  "verdict": record["counted_verdict"], "error": record.get("error")}), flush=True)
        arms[policy] = sorted(records, key=lambda r: r["iid"])
        summary = summarize(records, frozen["jobs"])
        summary.update(policy=policy, batch_wall_s=time.monotonic() - started,
                       provenance_drift=check_drift(frozen), records=arms[policy])
        summaries[policy] = summary
        _atomic_exclusive_json(arm_dir / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
    comparison = compare(arms["auto"], arms["one"])
    _atomic_exclusive_json(directory / "comparison.json", {
        "schema": frozen["schema"], "formal_gain": 0, "promotion_passed": False,
        "controlled_speed_claim": False, "fixed_order_confound": "auto first, one second",
        "provenance_drift": check_drift(frozen), "rows": comparison,
        "summaries": {p: {k: v for k, v in s.items() if k != "records"} for p, s in summaries.items()},
    })


if __name__ == "__main__":
    main()
