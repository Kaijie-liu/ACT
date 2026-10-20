"""Once-only stdlib adapter for the EXISTING owned-session supervisor.

No changes to process observation, RSS acceptance, quotas or native policies.
"""
import time
ENTRY = time.monotonic()
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"
FREEZE = ROOT/"docs/margin_support_freeze_20261021.json"
RUN = Path("/data1/Kane/MOE/results/margin_support_20261021_r1")
CAPS = dict(seconds=300, work=8000000, artifact_bytes=128*2**20,
            as_bytes=8*2**30, rss_bytes=2*2**30)


def require(ok, text):
    if not ok:
        raise ValueError(text)


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def limits(deadline):
    for kind, value in ((resource.RLIMIT_AS, CAPS["as_bytes"]),
                        (resource.RLIMIT_FSIZE, CAPS["artifact_bytes"])):
        old = resource.getrlimit(kind)
        require(all(x == resource.RLIM_INFINITY or x >= value for x in old), "inherited hard limits")
        resource.setrlimit(kind, (value, value))

    def stop(_sig, _frame):
        os.write(2, b"MARGIN_CONTROL_DEADLINE_NO_ACCEPTANCE\n")
        os._exit(124)
    signal.signal(signal.SIGALRM, stop)
    require(time.monotonic() < deadline, "deadline before setup")
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())


def binding(expected):
    from scoped_proof.io import load, sha
    spec = load(FREEZE, expected, 131072)
    require(spec["schema"] == "MARGIN_SUPPORT_FINITE_FREEZE_V1"
            and spec["caps"] == CAPS and spec["attempts"] == 1
            and spec["result_directory"] == str(RUN), "finite execution scope")
    for path, digest in spec["files"].items():
        require(sha(ROOT/path) == digest, "frozen file drift: "+path)
    return spec


def receive(value, freeze, deadline):
    from scoped_source.hz_request_store import Work, compact
    from scoped_source import check_margin_support as check
    require(time.monotonic() < deadline and value["deadline"] == deadline, "same request deadline")
    require(value["status"] == "PASS"
            and [r["name"] for r in value["cases"]] == freeze["cases"]
            and all(r["status"] == "PASS" for r in value["cases"]), "complete fixed case roster")
    work = Work(deadline)
    require(type(value["work_counts"]) is dict and set(value["work_counts"]) == set(work.counts)
            and all(type(v) is int and v >= 0 for v in value["work_counts"].values())
            and sum(value["work_counts"].values()) <= CAPS["work"], "continued Work")
    work.counts = dict(value["work_counts"]); before = sum(work.counts.values())
    certificates = value["certificates"]
    require([c["label"] for c in certificates] == freeze["certificates"], "saved source roster")
    receipts = []
    for item in certificates:
        frozen = freeze["sources"][item["label"]]
        require(item["source_sha256"] == frozen["sha256"], "frozen original source")
        receipt = check.check(compact(item["source"]), item["certificate"],
                              expected_source=frozen["sha256"], expected_mode=frozen["mode"],
                              expected_cells=frozen["cells"], work=work)
        require(receipt == item["receipt"], "saved receipt recomputation")
        receipts.append(dict(label=item["label"], **receipt))
    # Replay the additional single support obligation, not a whole source rerun.
    detail = value["cases"][6]["detail"]
    raw = compact(detail["slice_source"])
    from fractions import Fraction as F
    src = check.inspect(raw, freeze["sources"]["slice_support"]["sha256"], work)
    lower = check.checked_support(src, F(1, 4), F(0), detail["slice_support"], work)
    require(lower == F(1, 16) and detail["slice_lower"] == str(lower), "slice dual replay")
    work.tick()
    return dict(status="PASS", receipts=receipts, slice_lower=str(lower),
                total_work=sum(work.counts.values()),
                reception_work=sum(work.counts.values())-before,
                work_counts=work.counts, grade=check.GRADE)


def worker(expected, deadline, phase_deadline):
    limits(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import save
    from scoped_proof.owned_bounded import parent_rss
    freeze = binding(expected)
    cases = module("scripts/margin_support_cases.py", "finite_margin_cases")
    require(list(cases.CASES) == freeze["cases"], "case order")
    peak = 0

    def tick():
        nonlocal peak
        require(time.monotonic() < phase_deadline, "worker phase deadline")
        peak = max(peak, parent_rss())
        require(peak <= CAPS["rss_bytes"], "worker sampled RSS")
    result = cases.run(deadline, tick)
    require(not any(k in sys.modules for k in ("torch", "numpy", "scipy")),
            "standard-library-only worker")
    try:
        tick()
    except BaseException as exc:
        result.update(status="FAIL", terminal_error=repr(exc))
    result.update(freeze_sha256=expected, worker_entry=ENTRY, worker_deadline=phase_deadline,
                  sampled_worker_rss=peak, worker_seconds_before_export=time.monotonic()-ENTRY)
    require(len(json.dumps(result, sort_keys=True, allow_nan=False).encode()) <= 2**20,
            "worker member capacity")
    receipt = save(RUN/"worker.json", result)
    tick()
    print(json.dumps(dict(worker=receipt, status=result["status"])), flush=True)
    return 0 if result["status"] == "PASS" else 1


def main(expected):
    deadline = ENTRY+CAPS["seconds"]
    limits(deadline)
    sys.path.append(str(ROOT))
    from scoped_proof.io import load, save, sha
    from scoped_proof import owned_session as owned
    import shutil
    import subprocess
    result = dict(schema="MARGIN_SUPPORT_FINITE_RESULT_V1", status="FAIL", caps=CAPS,
                  entry=ENTRY, deadline=deadline, freeze_sha256=expected,
                  full_cold_start_seconds=None, model_executed=False, gpu_executed=False,
                  native_solver_executed=False, result_directory=str(RUN))
    created = False
    samples = []

    def sample(phase):
        value = owned.parent_rss()
        samples.append(dict(phase=phase, bytes=value, seconds=time.monotonic()-ENTRY))
        require(value <= CAPS["rss_bytes"] and time.monotonic() < deadline, "reception RSS/deadline")
    try:
        require(sys.executable == PYTHON, "existing act-py312 only")
        freeze = binding(expected)

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           timeout=min(5., max(.001, deadline-time.monotonic()))).strip()
        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and not git("status", "--porcelain"), "clean frozen branch")
        result["execution_head"] = git("rev-parse", "HEAD")
        available = next(int(r.split()[1])*1024 for r in Path("/proc/meminfo").read_text().splitlines()
                         if r.startswith("MemAvailable:"))
        disk = shutil.disk_usage(RUN.parent).free
        require(available >= 4*2**30 and disk >= CAPS["artifact_bytes"], "fresh bounded admission")
        require(time.monotonic() < ENTRY+60, "one admission window")
        result["admission"] = dict(available_bytes=available, disk_free_bytes=disk,
                                   seconds=time.monotonic()-ENTRY)
        RUN.mkdir()
        created = True
        result["initial_fds"] = len(os.listdir("/proc/self/fd"))
        result["controller_setup_seconds"] = time.monotonic()-ENTRY
        receipt = owned.execute(
            [PYTHON, "-I", "-S", "-B", str(Path(__file__).resolve()), "--worker", expected,
             str(deadline), str(ENTRY+240)],
            run_deadline=ENTRY+250, cleanup_deadline=ENTRY+270,
            env=dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                     OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            rss_limit=CAPS["rss_bytes"])
        result["session"] = receipt
        if (RUN/"worker.json").is_file():
            result["worker"] = dict(sha256=sha(RUN/"worker.json"), bytes=(RUN/"worker.json").stat().st_size)
        require(receipt["status"] == "COMPLETED" and receipt["returncode"] == 0
                and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
                and receipt["execution_error"] is None and receipt["cleanup_error"] is None,
                "owned-session or worker failure")
        sample("before_reception")
        require("worker" in result, "missing candidate")
        value = load(RUN/"worker.json", result["worker"]["sha256"], 2**20)
        require(value["freeze_sha256"] == expected, "candidate freeze binding")
        result["online_reception"] = receive(value, freeze, deadline)
        sample("after_reception")
        result["final_fds"] = len(os.listdir("/proc/self/fd"))
        require(result["final_fds"] == result["initial_fds"], "FD balance")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        try:
            sample("terminal")
        except BaseException as exc:
            result.update(status="FAIL", terminal_observation_error=repr(exc))
        result["controller_rss_samples"] = samples
        result["bytes_before_terminal"] = sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())
        result["external_output_reserved_bytes"] = 32768
        result["elapsed_before_terminal_export"] = time.monotonic()-ENTRY
        raw = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        require(result["bytes_before_terminal"]+2*len(raw)+32768 <= CAPS["artifact_bytes"],
                "complete artifacts and publication reserve")
        terminal = save(RUN/"result.json", result)
        through_export = time.monotonic()-ENTRY
        require(through_export < CAPS["seconds"], "late export")
        print(json.dumps(dict(status=result["status"], terminal=terminal,
                              observed_through_terminal_file_export_seconds=through_export,
                              stdout_flush_and_exit_in_observed_duration=False)), flush=True)
        require(time.monotonic() < deadline, "late stdout")
    else:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-approved-once":
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit("explicit once-only freeze argument required")
