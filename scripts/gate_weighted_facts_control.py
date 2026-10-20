"""One frozen synthetic control batch using the existing owned-session runner."""
import time
ENTRY = time.monotonic()
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"
FREEZE = ROOT / "docs/gate_weighted_facts_freeze_20261020.json"
RUN = Path("/data1/Kane/MOE/results/gate_weighted_facts_20261020_r1")
CAPS = dict(seconds=300, work=8000000, artifact_bytes=128*2**20,
            as_bytes=8*2**30, rss_bytes=2*2**30)


def require(ok, text):
    if not ok:
        raise ValueError(text)


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cap(deadline):
    for kind, value in ((resource.RLIMIT_AS, CAPS["as_bytes"]),
                        (resource.RLIMIT_FSIZE, CAPS["artifact_bytes"])):
        inherited = resource.getrlimit(kind)
        require(all(x == resource.RLIM_INFINITY or x >= value for x in inherited),
                "no raising inherited hard limits")
        resource.setrlimit(kind, (value, value))

    def hard_stop(_sig, _frame):
        os.write(2, b"M1_CONTROL_DEADLINE_NO_ACCEPTANCE\n")
        os._exit(124)
    signal.signal(signal.SIGALRM, hard_stop)
    require(time.monotonic() < deadline, "deadline before execution")
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())


def bindings(expected):
    from scoped_proof.io import load, sha
    freeze = load(FREEZE, expected, 131072)
    require(freeze["schema"] == "GATE_FACTS_FINITE_FREEZE_V1"
            and freeze["caps"] == CAPS and freeze["attempts"] == 1
            and freeze["result_directory"] == str(RUN), "finite freeze scope")
    for relative, digest in freeze["files"].items():
        require(sha(ROOT / relative) == digest, "source drift: " + relative)
    return freeze


def worker(expected, deadline):
    cap(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import save
    from scoped_proof.owned_bounded import parent_rss
    freeze = bindings(expected)
    cases = load_module("scripts/gate_weighted_facts_cases.py", "fixed_m1_cases")
    require(freeze["cases"] == list(cases.CASES), "fixed case order")
    peak = 0

    def tick():
        nonlocal peak
        require(time.monotonic() < deadline, "worker deadline")
        peak = max(peak, parent_rss())
        require(peak <= CAPS["rss_bytes"], "sampled worker RSS")
    result = cases.run(deadline, tick)
    tick()
    result.update(freeze_sha256=expected, worker_entry=ENTRY, worker_deadline=deadline,
                  worker_seconds_before_export=time.monotonic()-ENTRY, sampled_worker_rss=peak)
    require(len(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 2**20,
            "bounded fixed control member")
    receipt = save(RUN / "worker.json", result)
    require(time.monotonic() < deadline, "late worker export")
    print(json.dumps(dict(worker=receipt, status=result["status"]), sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


def main(expected):
    deadline = ENTRY + CAPS["seconds"]
    cap(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import load, save, sha
    from scoped_proof import owned_session as owned
    import shutil
    import subprocess
    started = time.monotonic()
    result = dict(schema="GATE_FACTS_FINITE_RESULT_V1", status="FAIL", caps=CAPS,
                  freeze_sha256=expected, entry=ENTRY, deadline=deadline,
                  controller_import_and_setup_seconds=started-ENTRY,
                  full_cold_start_seconds=None, model_executed=False, gpu_executed=False,
                  native_solver_executed=False, result_directory=str(RUN))
    created = False
    samples = []

    def parent_sample(phase):
        rss = owned.parent_rss()
        samples.append(dict(phase=phase, bytes=rss, elapsed_seconds=time.monotonic()-ENTRY))
        require(rss <= CAPS["rss_bytes"], "sampled reception/controller RSS")
        require(time.monotonic() < deadline, "controller phase deadline")
    try:
        require(sys.executable == PYTHON, "existing act-py312 only")
        freeze = bindings(expected)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           timeout=min(5., max(.001, deadline-time.monotonic()))).strip()
        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and not git("status", "--porcelain"), "frozen clean branch")
        result["execution_head"] = git("rev-parse", "HEAD")
        result["frozen_parent_head"] = freeze["parent_head"]
        available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines()
                         if x.startswith("MemAvailable:"))
        disk = shutil.disk_usage(RUN.parent).free
        require(available >= 4*2**30 and disk >= CAPS["artifact_bytes"], "fresh admission")
        require(time.monotonic() < ENTRY+60, "one bounded admission")
        result["admission"] = dict(available_bytes=available, disk_free_bytes=disk,
                                   elapsed_seconds=time.monotonic()-ENTRY)
        RUN.mkdir()  # Exclusive attempt claim; no resume or overwrite.
        created = True
        result["initial_fds"] = len(os.listdir("/proc/self/fd"))
        env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        receipt = owned.execute(
            [PYTHON, "-I", "-S", "-B", str(Path(__file__).resolve()),
             "--worker", expected, str(ENTRY+240)],
            run_deadline=ENTRY+250, cleanup_deadline=ENTRY+270,
            env=env, rss_limit=CAPS["rss_bytes"])
        result["session"] = receipt
        if (RUN / "worker.json").is_file():
            result["worker"] = dict(sha256=sha(RUN / "worker.json"),
                                     bytes=(RUN / "worker.json").stat().st_size)
        require(receipt["status"] == "COMPLETED" and receipt["returncode"] == 0
                and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and receipt["execution_error"] is None and receipt["cleanup_error"] is None,
                "worker or owned-session failure")
        require(result.get("worker") is not None, "missing candidate result")
        parent_sample("before_receive")
        value = load(RUN / "worker.json", result["worker"]["sha256"], 2**20)
        parent_sample("after_receive")
        require(value["freeze_sha256"] == expected and value["status"] == "PASS"
                and [c["name"] for c in value["cases"]] == freeze["cases"]
                and all(c["status"] == "PASS" for c in value["cases"]), "incomplete controls")
        require(sum(value["work_counts"].values()) <= CAPS["work"], "cumulative Work")
        require(time.monotonic() < deadline, "late acceptance")
        # Receipt reception, full saved-result audit and serialization are charged.
        audit = load_module("scripts/audit_gate_weighted_facts_control.py", "fixed_m1_audit")
        result["independent_saved_audit"] = audit.check_worker(value, freeze, deadline)
        parent_sample("after_saved_audit")
        result["final_fds"] = len(os.listdir("/proc/self/fd"))
        require(result["final_fds"] == result["initial_fds"], "descriptor leak")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        parent_sample("before_terminal_export")
        result["controller_rss_samples"] = samples
        result["bytes_before_terminal"] = sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())
        result["external_output_reserved_bytes"] = 32768
        result["elapsed_before_terminal_export"] = time.monotonic()-ENTRY
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        require(result["bytes_before_terminal"]+2*len(raw)+32768 <= CAPS["artifact_bytes"],
                "complete artifact and atomic-publication/output reserve cap")
        require(time.monotonic() < deadline, "terminal before deadline")
        terminal = save(RUN / "result.json", result)
        through_export = time.monotonic()-ENTRY
        require(through_export < CAPS["seconds"], "late terminal export")
        print(json.dumps(dict(status=result["status"], result=terminal,
                              observed_through_terminal_file_export_seconds=through_export,
                              stdout_flush_and_final_exit_in_duration=False,
                              result_directory=str(RUN)), sort_keys=True), flush=True)
        require(time.monotonic() < deadline, "late stdout reception")
    else:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3])))
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-approved-once":
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit("explicit frozen once-only opt in required")
