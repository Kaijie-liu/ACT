"""M3 once-only case/reception adapter using the existing public supervisor."""
import time
ENTRY = time.monotonic()
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"
FREEZE = ROOT / "docs/gate_margin_rows_freeze_20261020.json"
RUN = Path("/data1/Kane/MOE/results/gate_margin_rows_20261020_r1")
CAPS = dict(seconds=300, work=8000000, artifact_bytes=128*2**20,
            as_bytes=8*2**30, rss_bytes=2*2**30)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cap(deadline):
    old = module("scripts/gate_weighted_facts_control.py", "m3_unchanged_limits")
    require(old.CAPS == CAPS, "unchanged caps")
    old.cap(deadline)


def bindings(expected):
    from scoped_proof.io import load, sha
    value = load(FREEZE, expected, 4*2**20)
    require(value["schema"] == "GATE_MARGIN_FINITE_FREEZE_V1"
            and value["caps"] == CAPS and value["attempts"] == 1
            and value["result_directory"] == str(RUN), "frozen M3 scope")
    for relative, digest in value["files"].items():
        require(sha(ROOT / relative) == digest, "source drift: " + relative)
    return value


def artifact_bytes():
    return sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())


def worker(expected, deadline, phase_deadline):
    cap(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import save
    from scoped_proof.owned_bounded import parent_rss
    import importlib.metadata
    freeze = bindings(expected)
    environment = dict(python=sys.version, packages={
        name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")})
    require(environment == freeze["environment"], "native environment drift")
    cases = module("scripts/gate_margin_rows_cases.py", "m3_cases")
    require(freeze["paired_specs"] == cases.SPECS and freeze["reference_specs"] == cases.REFERENCES
            and freeze["faults"] == cases.FAULTS and freeze["fault_errors"] == cases.ERRORS, "fixed M3 roster")
    peak = 0

    def tick():
        nonlocal peak
        require(time.monotonic() < phase_deadline, "M3 worker phase cutoff")
        peak = max(peak, parent_rss())
        require(peak <= CAPS["rss_bytes"], "sampled native worker RSS")
        require(artifact_bytes()+16*2**20 <= CAPS["artifact_bytes"], "artifact publication reserve")

    result = cases.run(ROOT, RUN, deadline, phase_deadline, tick)
    tick()
    result.update(freeze_sha256=expected, environment=environment, worker_entry=ENTRY,
                  worker_seconds_before_export=time.monotonic()-ENTRY, sampled_worker_rss=peak)
    require(len(json.dumps(result, allow_nan=False).encode()) <= 8*2**20, "bounded worker result")
    receipt = save(RUN / "worker.json", result)
    tick()
    print(json.dumps(dict(worker=receipt, status=result["status"]), sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


def receive(value, freeze, deadline):
    """Continue the receipt-bound ledger; no ACT, checkpoint or optimizer import."""
    from scoped_source.hz_request_store import Work
    from scoped_proof.io import load
    cases = module("scripts/gate_margin_rows_cases.py", "m3_receiver_cases")
    require(value["status"] == "PASS" and value["shared_deadline"] == value["hard_deadline"] == deadline
            and value["counts"] == freeze["expected_counts"], "complete native counters/deadline")
    cases.inventory(value["records"], deadline, value["shared_deadline"], value["work_counts"])
    cases.all_faults(value["records"], value["faults"], value["work_counts"])
    require(len(value["record_receipts"]) == len(value["records"]), "saved-record roster")
    work = Work(deadline)
    work.counts = dict(value["work_counts"])  # Same spent ledger, not a new allowance.
    received = dict(work.counts)
    algebra = {}
    for row, receipt in zip(value["records"], value["record_receipts"]):
        work.tick()
        saved = load(RUN / (row["case"]+".json"), receipt["sha256"], 2**20)
        require(saved == row, "saved record identity")
        report = cases.verify_row(saved, work)
        require(report == saved["algebra"], "independent algebra reception")
        algebra[row["case"]] = report
    work.tick()
    return dict(status="PASS", same_deadline=deadline, received_work_counts=received,
                final_work_counts=dict(work.counts), total_work=sum(work.counts.values()),
                checked_records=len(algebra), algebra=algebra,
                guarantee="given-source row inclusion and synthetic LP checks; not complete MoE SAFE")


def main(expected, expected_head):
    deadline = ENTRY+300
    cap(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import save, sha, load
    from scoped_proof import owned_session as owned
    import subprocess
    import shutil
    result = dict(schema="GATE_MARGIN_FINITE_RESULT_V1", status="FAIL", freeze_sha256=expected,
                  caps=CAPS, entry=ENTRY, deadline=deadline, execution_head=expected_head,
                  result_directory=str(RUN), trained_model_executed=False, gpu_executed=False,
                  full_cold_start_seconds=None)
    created = False
    try:
        require(sys.executable == PYTHON, "existing environment")
        freeze = bindings(expected)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=5).strip()
        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and git("rev-parse", "HEAD") == expected_head and not git("status", "--porcelain"),
                "frozen clean execution identity")
        available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines()
                         if x.startswith("MemAvailable:"))
        free = shutil.disk_usage(RUN.parent).free
        require(available >= 4*2**30 and free >= CAPS["artifact_bytes"], "fresh admission")
        require(time.monotonic() < ENTRY+60, "one bounded admission")
        result["admission"] = dict(available_bytes=available, disk_free_bytes=free,
                                  elapsed_seconds=time.monotonic()-ENTRY)
        RUN.mkdir()
        created = True
        result["initial_fds"] = len(os.listdir("/proc/self/fd"))
        env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        receipt = owned.execute(
            [PYTHON, "-I", "-B", str(Path(__file__).resolve()), "--worker", expected,
             str(deadline), str(ENTRY+240)], run_deadline=ENTRY+250, cleanup_deadline=ENTRY+270,
            env=env, rss_limit=CAPS["rss_bytes"])
        result["session"] = receipt
        if (RUN/"worker.json").is_file():
            result["worker"] = dict(sha256=sha(RUN/"worker.json"), bytes=(RUN/"worker.json").stat().st_size)
        require(receipt["status"] == "COMPLETED" and receipt["returncode"] == 0
                and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and receipt["execution_error"] is None and receipt["cleanup_error"] is None,
                "worker/session failure")
        require(result.get("worker") is not None, "missing worker result")
        value = load(RUN/"worker.json", result["worker"]["sha256"], 8*2**20)
        require(value["freeze_sha256"] == expected, "worker freeze identity")
        result["reception"] = receive(value, freeze, deadline)
        result["final_fds"] = len(os.listdir("/proc/self/fd"))
        require(result["final_fds"] == result["initial_fds"], "descriptor leak")
        require(owned.parent_rss() <= CAPS["rss_bytes"], "controller RSS")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        result["bytes_before_terminal"] = artifact_bytes()
        result["elapsed_before_terminal_export"] = time.monotonic()-ENTRY
        raw = json.dumps(result, allow_nan=False).encode()
        require(artifact_bytes()+2*len(raw)+32768 <= CAPS["artifact_bytes"], "complete artifact cap")
        require(time.monotonic() < deadline, "terminal cutoff")
        terminal = save(RUN/"result.json", result)
        through_export = time.monotonic()-ENTRY
        require(through_export < 300, "late terminal export")
        print(json.dumps(dict(status=result["status"], result=terminal,
             observed_through_terminal_file_export_seconds=through_export,
             stdout_flush_and_final_exit_in_duration=False, result_directory=str(RUN)),
             sort_keys=True), flush=True)
        require(time.monotonic() < deadline, "late stdout")
    else:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
    if len(sys.argv) == 4 and sys.argv[1] == "--execute-approved-once":
        raise SystemExit(main(sys.argv[2], sys.argv[3]))
    raise SystemExit("explicit M3 frozen once-only entry required")
