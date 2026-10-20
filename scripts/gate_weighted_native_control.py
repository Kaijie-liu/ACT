"""Once-only M1b entry: existing owned-session supervision, one shared deadline."""
import time
ENTRY = time.monotonic()
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"
FREEZE = ROOT / "docs/gate_weighted_native_freeze_20261020.json"
RUN = Path("/data1/Kane/MOE/results/gate_weighted_native_20261020_r1")
CAPS = dict(seconds=300, work=8000000, artifact_bytes=128*2**20,
            as_bytes=8*2**30, rss_bytes=2*2**30)


def require(ok, text):
    if not ok:
        raise ValueError(text)


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cap(deadline):
    # Reuse the already controlled hard-limit installation, not a new supervisor.
    old = module("scripts/gate_weighted_facts_control.py", "m1_limits")
    require(old.CAPS == CAPS, "unchanged caps")
    old.cap(deadline)


def bindings(expected):
    from scoped_proof.io import load, sha
    freeze = load(FREEZE, expected, 4*2**20)
    require(freeze["schema"] == "GATE_NATIVE_FINITE_FREEZE_V1"
            and freeze["caps"] == CAPS and freeze["attempts"] == 1
            and freeze["result_directory"] == str(RUN), "native freeze scope")
    for relative, digest in freeze["files"].items():
        require(sha(ROOT / relative) == digest, "source drift: " + relative)
    return freeze


def artifact_bytes():
    return sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())


def worker(expected, deadline, phase_deadline):
    cap(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import save
    from scoped_proof.owned_bounded import parent_rss
    freeze = bindings(expected)
    import importlib.metadata
    environment = dict(python=sys.version, packages={
        name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")})
    require(environment == freeze["environment"], "native environment drift")
    cases = module("scripts/gate_weighted_native_cases.py", "native_m1_cases")
    require(freeze["calls"] == [list(x) for x in cases.CALLS]
            and freeze["faults"] == cases.FAULTS, "fixed native controls")
    peak = 0

    def tick():
        nonlocal peak
        require(time.monotonic() < phase_deadline, "native worker phase cutoff")
        peak = max(peak, parent_rss())
        require(peak <= CAPS["rss_bytes"], "sampled native worker RSS")
        require(artifact_bytes() + 16*2**20 <= CAPS["artifact_bytes"],
                "artifact cap including publication/reception reserve")

    result = cases.run(ROOT, RUN, deadline, phase_deadline, tick)
    tick()
    result.update(freeze_sha256=expected, environment=environment,
                  worker_entry=ENTRY, shared_deadline=deadline,
                  worker_seconds_before_export=time.monotonic()-ENTRY, sampled_worker_rss=peak)
    require(len(json.dumps(result, allow_nan=False).encode()) <= 2**20, "bounded native result")
    receipt = save(RUN / "worker.json", result)
    tick()
    print(json.dumps(dict(worker=receipt, status=result["status"]), sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


def check_saved(value, freeze, deadline):
    """No Torch or solver: bind all records; recompute every saved M1 fact."""
    from scoped_proof.io import load, sha
    core = module("act/pipeline/moe/gate_weighted_facts.py", "native_saved_m1")

    def validate_work_counts(counts):
        require(set(counts) == {"row_visits", "coefficient_visits", "arithmetic"}
                and all(type(v) is int and v >= 0 for v in counts.values())
                and sum(counts.values()) <= CAPS["work"], "received cumulative Work")

    def validate_counts(counts):
        require(set(counts) == {"attempted", "returned", "status0_returned"}
                and all(type(v) is int and v >= 0 for v in counts.values())
                and counts["status0_returned"] <= counts["returned"] <= counts["attempted"],
                "received F0 counters")
    require(value["status"] == "PASS"
            and value["shared_deadline"] == deadline
            and len(value["calls"]) == 12
            and [f["name"] for f in value["faults"]] == freeze["faults"]
            and all(f["status"] == "PASS" for f in value["faults"]), "complete native controls")
    validate_work_counts(value["work_counts"])
    ledger = core.Work(deadline)
    ledger.counts = dict(value["work_counts"])  # Receipt-bound continuation, not a new allowance.
    before = dict(ledger.counts)
    facts = 0
    previous = None
    for ordinal, (record, spec) in enumerate(zip(value["calls"], freeze["calls"])):
        ledger.tick()
        require([record["case"], record["method"], record["enabled"]] == spec
                and record["ordinal"] == ordinal
                and record["audit"]["status"] == "PASS"
                and record["audit"]["online_same_deadline"] is True,
                "call identity/online audit")
        validate_counts(record["f0_calls"])
        for key in ("work_before", "work_after"):
            validate_work_counts(record[key])
        require(all(record["work_after"][k] >= record["work_before"][k] for k in before)
                and (previous is None or previous == record["work_before"]), "call ledger continuity")
        require(all(before[k] >= record["work_after"][k] for k in before)
                and record["work_delta"] == {
                    k: record["work_after"][k]-record["work_before"][k] for k in before},
                "worker final ledger cannot omit completed work")
        previous = record["work_after"]
        path = RUN / f"call_{ordinal:02d}"
        require(record["package"] == str(path), "fixed package location")
        evidence = load(path / "evidence.json", record["evidence_sha256"], 2**20)
        manifest = load(path / "manifest.json", record["manifest_sha256"], 2**20)
        require(manifest["request_id"] == evidence["request_id"] == record["request_id"]
                and manifest["evidence_sha256"] == record["evidence_sha256"]
                and evidence["f0_calls"] == record["f0_calls"], "bound package/counters")
        # Full tensor/source/witness checking occurred in the charged worker.
        # Reception verifies file identities without importing the native stack.
        for name in ("request", "witness"):
            item = manifest.get(name)
            if item:
                require(item["sha256"] == sha(path / f"{name}.pt"), "bound tensor artifact")
        require(evidence["gate_native_contract"]["deadline"] == deadline,
                "evidence online deadline")
        counts = evidence["gate_native_contract"]["work_counts"]
        validate_work_counts(counts)
        require(all(record["work_before"][k] <= counts[k] <= record["work_after"][k]
                    for k in before), "evidence ledger outside call prefix")
        facts += core.audit_evidence(evidence, work=ledger)
    for fault in value["faults"]:
        for key in ("work_before", "work_after"):
            if key in fault:
                validate_work_counts(fault[key])
                require(all(before[k] >= fault[key][k] for k in before), "omitted fault work")
    ledger.tick()
    return dict(status="PASS", checked_saved_gate_facts=facts,
                received_work_counts=before, final_work_counts=dict(ledger.counts),
                total_work=sum(ledger.counts.values()),
                guarantee="M1 arithmetic/identity checks plus charged worker structural audit; not network proof")


def main(expected):
    deadline = ENTRY + 300
    cap(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import load, save, sha
    from scoped_proof import owned_session as owned
    import shutil
    import subprocess
    result = dict(schema="GATE_NATIVE_FINITE_RESULT_V1", status="FAIL", caps=CAPS,
                  freeze_sha256=expected, entry=ENTRY, deadline=deadline,
                  full_cold_start_seconds=None, trained_model_executed=False, gpu_executed=False,
                  native_synthetic_scope=True, result_directory=str(RUN))
    created = False
    try:
        require(sys.executable == PYTHON, "existing environment only")
        freeze = bindings(expected)

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=5).strip()

        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and not git("status", "--porcelain"), "frozen clean branch")
        result["execution_head"] = git("rev-parse", "HEAD")
        available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines()
                         if x.startswith("MemAvailable:"))
        disk = shutil.disk_usage(RUN.parent).free
        require(available >= 4*2**30 and disk >= CAPS["artifact_bytes"], "fresh admission")
        require(time.monotonic() < ENTRY+60, "one bounded admission")
        result["admission"] = dict(available_bytes=available, disk_free_bytes=disk,
                                   elapsed_seconds=time.monotonic()-ENTRY)
        RUN.mkdir()
        created = True
        result["initial_fds"] = len(os.listdir("/proc/self/fd"))
        env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="")
        receipt = owned.execute(
            [PYTHON, "-I", "-B", str(Path(__file__).resolve()), "--worker", expected,
             str(deadline), str(ENTRY+240)],
            run_deadline=ENTRY+250, cleanup_deadline=ENTRY+270, env=env, rss_limit=CAPS["rss_bytes"])
        result["session"] = receipt
        if (RUN / "worker.json").is_file():
            result["worker"] = dict(sha256=sha(RUN / "worker.json"),
                                    bytes=(RUN / "worker.json").stat().st_size)
        require(receipt["status"] == "COMPLETED" and receipt["returncode"] == 0
                and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and receipt["execution_error"] is None and receipt["cleanup_error"] is None,
                "worker/session failure")
        require(result.get("worker") is not None, "missing native result")
        value = load(RUN / "worker.json", result["worker"]["sha256"], 2**20)
        require(value["freeze_sha256"] == expected, "worker freeze mismatch")
        result["saved_record_audit"] = check_saved(value, freeze, deadline)
        result["final_fds"] = len(os.listdir("/proc/self/fd"))
        require(result["final_fds"] == result["initial_fds"], "descriptor leak")
        require(owned.parent_rss() <= CAPS["rss_bytes"], "controller RSS")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        result["bytes_before_terminal"] = artifact_bytes()
        result["external_output_reserved_bytes"] = 32768
        result["elapsed_before_terminal_export"] = time.monotonic()-ENTRY
        raw = json.dumps(result, allow_nan=False).encode()
        require(artifact_bytes()+2*len(raw)+32768 <= CAPS["artifact_bytes"], "full artifact cap")
        require(time.monotonic() < deadline, "terminal cutoff")
        terminal = save(RUN / "result.json", result)
        through_export = time.monotonic()-ENTRY
        require(through_export < 300, "late terminal export")
        print(json.dumps(dict(status=result["status"], result=terminal,
                              observed_through_terminal_file_export_seconds=through_export,
                              stdout_flush_and_final_exit_in_duration=False,
                              result_directory=str(RUN)), sort_keys=True), flush=True)
        require(time.monotonic() < deadline, "late stdout")
    else:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-approved-once":
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit("explicit frozen once-only native opt in required")
