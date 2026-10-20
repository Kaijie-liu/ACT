"""Read saved M1 controls without rerunning cases, a model or any solver."""
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import time

ROOT = Path("/data1/Kane/MOE/ACT")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_worker(value, freeze, deadline):
    require(time.monotonic() < deadline, "audit deadline")
    rows = value["cases"]
    require(value["status"] == "PASS" and [r["name"] for r in rows] == freeze["cases"]
            and all(r["status"] == "PASS" for r in rows), "complete fixed controls")
    require(all(type(n) is int and n >= 0 for n in value["work_counts"].values())
            and sum(value["work_counts"].values()) <= freeze["caps"]["work"], "finite cumulative work")
    details = {r["name"]: r["detail"] for r in rows}
    require(all(type(r["seconds"]) is float and math.isfinite(r["seconds"]) and r["seconds"] >= 0
                for r in rows), "observed finite costs")
    row = details["signed_not_safe"]["positive_combination"]
    lower = [Fraction(f["lower_bound"]) for f in row["proof_sources"]]
    gate = [Fraction(v) for v in row["gate_source"]["lambda_bounds"]]
    expected = min(t*lower[0]+(1-t)*lower[1] for t in gate)
    require(Fraction(row["checked_lower_bound"]) == expected
            and 0 < Fraction(row["accepted_minimum"]) <= expected
            and lower[0] < 0, "signed-to-weighted arithmetic")
    spec = importlib.util.spec_from_file_location("saved_m1_core", ROOT / "act/pipeline/moe/gate_weighted_facts.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    ledger = g.Work(deadline)
    # Continue the received ledger. Do not grant a fresh 8M allowance to reception.
    ledger.counts = dict(value["work_counts"])
    before = dict(ledger.counts)
    seen = 0
    for name, expected_calls in (("staged_stub_differential", (4, 2)),
                                 ("monolithic_stub_differential", (2, 1))):
        item = details[name]
        require(item["scope"] == "STUB_ONLY_NO_TIMING_OR_MODEL_EFFICACY", "stub guarantee")
        require((item["off_calls"]["solve"], item["on_calls"]["solve"]) == expected_calls,
                "fixed calls")
        require(item["off_calls"]["gate"] == item["on_calls"]["gate"] == 2
                and item["off_calls"]["propagation"] == item["on_calls"]["propagation"] == 2,
                "query/propagation invariant")
        evidence = item["evidence"]
        require(evidence["tier2"]["status"] == "UNKNOWN", "no fabricated complete SAFE")
        seen += g.audit_evidence(evidence, work=ledger)
    require(seen == 4, "all saved new positive facts")
    for path in details["old_positive_path"]["function_stub_paths"].values():
        require(path["status"] == "SAFE" and not any(path["calls"].values()), "old fast path")
    for path in ("staged", "monolithic"):
        tier2 = details["native_error_and_deadline"][path]["tier2"]
        require(tier2["status"] == "TIMEOUT" and tier2["partial_rows_censored"], "partial timeout")
    total_work = sum(ledger.counts.values())
    require(total_work <= freeze["caps"]["work"], "worker plus reception cumulative cap")
    require(time.monotonic() < deadline, "late audit")
    return dict(status="PASS", checked_saved_gate_facts=seen, case_groups=len(rows),
                reception_work_counts={k: ledger.counts[k]-before[k] for k in before},
                total_work=total_work,
                guarantee=g.GRADE, control_only=True)
