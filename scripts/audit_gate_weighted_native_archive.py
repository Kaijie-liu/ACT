"""Offline saved-record audit. Never upgrades the failed online M1b batch.

python -I -S -B scripts/audit_gate_weighted_native_archive.py <runtime-directory>
Standard library only; no model, tensor deserialization, checkpoint or solver.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scoped_proof.io import load, sha


def require(ok, text):
    if not ok:
        raise ValueError(text)


def check(root):
    started = time.monotonic()
    root = Path(root)
    freeze_path = ROOT / "docs/gate_weighted_native_freeze_20261020.json"
    freeze = load(freeze_path)
    terminal = load(root / "result.json")
    worker = load(root / "worker.json", terminal["worker"]["sha256"], 2**20)
    require(terminal["freeze_sha256"] == sha(freeze_path) == worker["freeze_sha256"],
            "freeze identity")
    require(terminal["status"] == "FAIL" and worker["status"] == "PASS",
            "retain original failure, no offline promotion")
    require(terminal["session"]["status"] == "ERROR"
            and terminal["session"]["execution_error"]["message"] == "live process RSS unavailable"
            and terminal["session"]["returncode"] == 0, "recorded failure boundary")
    require("saved_record_audit" not in terminal, "online reception was not reached")
    require(worker["shared_deadline"] == terminal["deadline"]
            and [f["name"] for f in worker["faults"]] == freeze["faults"]
            and all(f["status"] == "PASS" for f in worker["faults"]), "fixed faults/deadline")
    spec = importlib.util.spec_from_file_location(
        "saved_gate_core", ROOT / "act/pipeline/moe/gate_weighted_facts.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    # Separate OFFLINE audit allowance, not the online worker's execution.
    work = core.Work(started + 300)
    rows, identities = [], {}
    total = worker["work_counts"]
    keys = {"arithmetic", "row_visits", "coefficient_visits"}

    def counts(value):
        require(set(value) == keys and all(type(v) is int and v >= 0 for v in value.values())
                and sum(value.values()) <= freeze["caps"]["work"], "valid Work prefix")

    counts(total)
    previous = None
    checked = 0
    require(len(worker["calls"]) == len(freeze["calls"]) == 12, "fixed complete calls")
    for record, registered in zip(worker["calls"], freeze["calls"]):
        work.tick()
        i = record["ordinal"]
        require([record["case"], record["method"], record["enabled"]] == registered,
                "registered call order")
        counts(record["work_before"]); counts(record["work_after"])
        require(previous is None or previous == record["work_before"], "continuous Work")
        require(all(0 <= record["work_before"][k] <= record["work_after"][k] <= total[k] for k in keys)
                and record["work_delta"] == {
                    k: record["work_after"][k]-record["work_before"][k] for k in keys},
                "no omitted/reset Work")
        previous = record["work_after"]
        package = root / f"call_{i:02d}"
        e = load(package / "evidence.json", record["evidence_sha256"], 2**20)
        m = load(package / "manifest.json", record["manifest_sha256"], 2**20)
        require(e["request_id"] == m["request_id"] == record["request_id"]
                and m["evidence_sha256"] == record["evidence_sha256"]
                and e["verdict"]["status"] == record["status"] == m["status"]
                and record["audit"]["status"] == "PASS", "saved package/audit identity")
        for name in ("request", "witness"):
            if m.get(name):
                require(sha(package / f"{name}.pt") == m[name]["sha256"], "tensor artifact hash")
        c = record["f0_calls"]
        require(c == e["f0_calls"] and set(c) == {"attempted", "returned", "status0_returned"}
                and all(type(v) is int and v >= 0 for v in c.values())
                and c["status0_returned"] <= c["returned"] <= c["attempted"], "actual call counters")
        identity = dict(e["identity"])
        identity.pop("config_sha256")
        key = record["case"]
        require(key not in identities or identities[key] == identity, "same literal/input/property")
        identities[key] = identity
        require(e["route_coverage"]["route_sets_exact"]
                and e["route_coverage"]["coverage_complete"]
                and e["route_coverage"]["feasible_route_sets"] ==
                ([[0, 1], [0, 2], [1, 2]] if key == "B" else [[0, 1]]), "all legal pairs")
        require(e["gate_native_contract"]["deadline"] == terminal["deadline"],
                "same online deadline")
        prefix = e["gate_native_contract"]["work_counts"]
        counts(prefix)
        require(all(record["work_before"][k] <= prefix[k] <= record["work_after"][k] for k in keys),
                "evidence prefix interval")
        checked += core.audit_evidence(e, work=work)
        gate = e.get("gate_weighted_facts", {})
        rows.append(dict(case=key, method=record["method"], enabled=record["enabled"],
                         status=record["status"], f0_calls=c,
                         discharged=gate.get("discharged_pair_properties", 0),
                         unavailable_gates=gate.get("unavailable_gates", []),
                         observed_call_seconds=record["total_through_audit_seconds"]))
    for case in ("A", "B", "C"):
        for method in ("staged", "monolithic_f0"):
            off, on = sorted((r for r in rows if r["case"] == case and r["method"] == method),
                             key=lambda r: r["enabled"])
            require(off["status"] == on["status"], "observed terminal differential")
    members = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(root.rglob("*")) if p.is_file()}
    return dict(schema="GATE_NATIVE_OFFLINE_AUDIT_V1", status="PASS_SAVED_RECORDS_ONLY",
                online_status=terminal["status"], online_acceptance_recovered=False,
                freeze_sha256=sha(freeze_path), result_sha256=sha(root / "result.json"),
                worker_sha256=sha(root / "worker.json"), completed_calls=12,
                recorded_faults=len(worker["faults"]), checked_saved_gate_facts=checked,
                rows=rows, online_worker_work_counts=total,
                offline_audit_work_counts=dict(work.counts),
                members=members, retained_bytes=sum(x["bytes"] for x in members.values()),
                offline_audit_seconds=time.monotonic()-started,
                scope="hashes, saved structure/counters and M1 arithmetic; no new native replay or network proof")


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1]), sort_keys=True, indent=2))
