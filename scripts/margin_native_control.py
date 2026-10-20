"""One frozen A1 batch on the existing supervisor with explicit development policy."""
import time
ENTRY = time.monotonic()
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
PYTHON = "/data1/Kane/miniconda3/envs/act-py312/bin/python"
FREEZE = ROOT/"docs/margin_native_freeze_20261021.json"
RUN = Path("/data1/Kane/MOE/results/margin_native_source_20261021_r1")
POLICY = "DEV_MARGIN_20261021_V1"
CAPS = dict(seconds=300, artifact_bytes=128*2**20, as_bytes=8*2**30,
            rss_gate=False, work_gate=False)
ENV = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def cases_module():
    spec = importlib.util.spec_from_file_location("margin_native_cases", ROOT/"scripts/margin_native_cases.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def limits(deadline):
    require(time.monotonic() < deadline and deadline-ENTRY <= 300, "single bounded deadline")
    for key, cap in ((resource.RLIMIT_AS, CAPS["as_bytes"]), (resource.RLIMIT_FSIZE, CAPS["artifact_bytes"])):
        old = resource.getrlimit(key)
        require(all(v == resource.RLIM_INFINITY or v >= cap for v in old), "inherited hard limits")
        resource.setrlimit(key, (cap, cap))
    def stop(*_):
        os.write(2, b"A1_HARD_DEADLINE_NO_ACCEPTANCE\n"); os._exit(124)
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())


def binding(anchor):
    from scoped_proof.io import load, sha
    spec = load(FREEZE, anchor, 2**20)
    require(spec["policy"] == POLICY and spec["caps"] == CAPS and spec["attempts"] == 1
            and spec["result_directory"] == str(RUN), "frozen A1 execution")
    for name, digest in spec["files"].items():
        require(sha(ROOT/name) == digest, "frozen code drift: "+name)
    for name, digest in spec["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/name).rglob("*.py"))}
        encoded = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(encoded).hexdigest() == digest, "frozen implementation tree: "+name)
    require(spec["cases"] == cases_module().CASES, "frozen cases")
    return spec


def artifact_bytes():
    return sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())


def save_bounded(path, value):
    from scoped_proof.io import save
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    require(len(raw) <= 4*2**20 and 2*(artifact_bytes()+2*len(raw)+2**20) <= CAPS["artifact_bytes"],
            "cumulative artifacts including publication reserve")
    return save(path, value)


def worker(anchor, deadline, phase_deadline):
    limits(phase_deadline)
    sys.path.insert(0, str(ROOT))
    binding(anchor)
    def prefix(result, work):
        save_bounded(RUN/("prefix_%02d.json" % len(result["cases"])),
                     dict(result, work_counts=dict(work.counts)))
    value = cases_module().run(deadline, prefix)
    value.update(freeze_sha256=anchor, phase_deadline=phase_deadline,
                 rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))
    receipt = save_bounded(RUN/"worker.json", value)
    require(time.monotonic() < phase_deadline, "worker export deadline")
    print(json.dumps(dict(status=value["status"], worker=receipt)), flush=True)
    return 0 if value["status"] == "PASS" else 1


def successful(receipt):
    return (receipt["status"] == "COMPLETED" and receipt["returncode"] == 0
            and receipt["execution_error"] is None and receipt["cleanup_error"] is None
            and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION"
            and receipt["wait4_usage"] is not None)


def main(anchor):
    deadline = ENTRY+300
    limits(deadline)
    sys.path.insert(0, str(ROOT))
    from scoped_proof.io import load, sha
    from scoped_proof import owned_session as owned
    from scoped_source.development_work import DevelopmentWork
    import subprocess
    result = dict(schema="MARGIN_NATIVE_CONTROL_RESULT_V1", status="FAIL", policy=POLICY,
                  freeze_sha256=anchor, entry=ENTRY, deadline=deadline, caps=CAPS,
                  full_cold_start_seconds=None, trained_source=False, gpu=False,
                  complete_request=False, scalar_aggregate_checked=False, sessions=[])
    created = False
    try:
        require(sys.executable == PYTHON, "existing act-py312")
        freeze = binding(anchor)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           timeout=min(5, max(.001, deadline-time.monotonic()))).strip()
        require(git("branch", "--show-current") == "feat/moe-route-verification"
                and not git("status", "--porcelain"), "clean frozen branch")
        result["execution_head"] = git("rev-parse", "HEAD")
        available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        disk = shutil.disk_usage(RUN.parent).free
        require(time.monotonic() < ENTRY+60 and available >= 4*2**30 and disk >= CAPS["artifact_bytes"],
                "one bounded fresh admission")
        result["admission"] = dict(seconds=time.monotonic()-ENTRY, available_bytes=available, disk_bytes=disk)
        RUN.mkdir(); created = True
        result["fds_before"] = len(os.listdir("/proc/self/fd"))
        cases = cases_module()
        result["observation_controls"] = cases.observation_controls(deadline)
        def launch(name, args, run_end, clean_end):
            receipt = owned.execute(args, run_deadline=run_end, cleanup_deadline=clean_end,
                                    env=ENV, rss_limit=None, development_policy=POLICY)
            result["sessions"].append(dict(name=name, receipt=receipt))
            save_bounded(RUN/(name+"_session.json"), receipt)
            return receipt
        receipt = launch("native", [PYTHON, "-I", "-B", __file__, "--worker", anchor,
                                     str(deadline), str(ENTRY+195)], ENTRY+205, ENTRY+220)
        require(successful(receipt), "native candidate session failed")
        result["worker_sha256"] = sha(RUN/"worker.json")
        value = load(RUN/"worker.json", result["worker_sha256"], 4*2**20)
        require(value["freeze_sha256"] == anchor and value["deadline"] == deadline
                and value["rlimit_as"] == [CAPS["as_bytes"]]*2, "worker scope/deadline/AS")
        work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(value["work_counts"])
        before = sum(work.counts.values()); started = time.monotonic()
        receipts = cases.receive(value, work)
        result["online_reception"] = dict(receipts=receipts, work_counts=dict(work.counts),
                                         work=sum(work.counts.values())-before,
                                         seconds=time.monotonic()-started)
        # Expired and partial reception are not rescued by valid partial supports.
        result["reception_faults"] = [cases.rejected(
            lambda: DevelopmentWork(time.monotonic()-1, policy=POLICY), "deadline"),
            cases.rejected(lambda: cases.receive(dict(value, supports=value["supports"][:-1]), work),
                           "all property receipts")]
        relocated = RUN/"relocated"; relocated.mkdir()
        files = freeze["portable_files"]
        started = time.monotonic()
        total = sum((ROOT/src).stat().st_size for src in files)+ (RUN/"worker.json").stat().st_size
        require(2*(artifact_bytes()+total+2**20) <= CAPS["artifact_bytes"], "relocation cumulative capacity")
        for src, dest in files.items():
            target = relocated/dest; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/src, target)
            require(sha(target) == freeze["files"][src], "relocated code identity")
        shutil.copyfile(RUN/"worker.json", relocated/"candidate.json")
        continuation = save_bounded(relocated/"continued.json", dict(work.counts))
        result["relocation_seconds"] = time.monotonic()-started
        receipt = launch("portable", [PYTHON, "-I", "-S", "-B", str(relocated/"margin_native_recheck.py"),
                                       result["worker_sha256"], continuation["sha256"], str(deadline)],
                         ENTRY+260, ENTRY+270)
        require(successful(receipt), "portable session failed")
        portable = load(relocated/"receipt.json", limit=2**20)
        require(portable["status"] == "PASS" and portable["receipts"] == receipts
                and portable["deadline"] == deadline
                and all(portable["work_counts"][k] >= work.counts[k] for k in work.counts),
                "portable complete same result with continued accounting")
        result["portable_reception"] = portable
        # Two fixed process faults; both are expected rejections, not retries.
        now = time.monotonic()
        receipt = launch("exit7", [PYTHON, "-I", "-S", "-c", "raise SystemExit(7)"],
                         min(now+3, ENTRY+275), min(now+5, ENTRY+278))
        require(receipt["status"] == "ERROR" and receipt["returncode"] == 7
                and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION", "explicit exit retained")
        now = time.monotonic()
        receipt = launch("deadline", [PYTHON, "-I", "-S", "-c", "import time; time.sleep(60)"],
                         min(now+.2, ENTRY+280), min(now+3, ENTRY+285))
        require(receipt["status"] == "TIMEOUT" and receipt["returncode"] == -9
                and receipt["cleanup_status"] == "LEADER_REAPED_NO_LIVE_SESSION", "deadline cleanup")
        result["fds_after"] = len(os.listdir("/proc/self/fd"))
        require(result["fds_before"] == result["fds_after"], "final FD balance")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        result["artifact_bytes_before_terminal"] = artifact_bytes()
        result["archive_reserve"] = "ONE_ADDITIONAL_COPY_OF_ALL_RUN_ARTIFACTS"
        result["elapsed_before_terminal_export"] = time.monotonic()-ENTRY
        receipt = save_bounded(RUN/"result.json", result)
        require(time.monotonic() < deadline, "late result cannot be accepted")
        print(json.dumps(dict(status=result["status"], receipt=receipt,
                              through_export_seconds=time.monotonic()-ENTRY,
                              stdout_flush_and_exit_in_duration=False)), flush=True)
    else:
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-approved-once":
        raise SystemExit(main(sys.argv[2]))
    raise SystemExit("explicit frozen once-only scope required")
