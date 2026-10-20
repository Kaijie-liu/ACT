"""One frozen sparse-scalar integration on the existing owned-session supervisor."""
import time
ENTRY = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

ROOT = Path("/data1/Kane/MOE/ACT")
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from margin_native_control import limits, successful, ENV, PYTHON, POLICY, CAPS
from scoped_proof.io import load, save, sha
from scoped_source.hz_request_store import require

FREEZE = ROOT/"docs/margin_scalar_freeze_20261021.json"
RUN = Path("/data1/Kane/MOE/results/margin_scalar_source_20261021_r1")


def binding(anchor):
    from margin_scalar_cases import CELLS, FAULTS, MODES
    spec = load(FREEZE, anchor, 2**20)
    require(spec["policy"] == POLICY and spec["caps"] == CAPS and spec["attempts"] == 1
            and spec["result_directory"] == str(RUN) and spec["cells"] == CELLS
            and spec["modes"] == list(MODES) and spec["faults"] == list(FAULTS)
            and spec["native_calls"] == 2 and spec["objective_slots"] == 6, "frozen scalar execution")
    for name, digest in spec["files"].items():
        require(sha(ROOT/name) == digest, "frozen code drift: "+name)
    for name, digest in spec["trees"].items():
        members = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/name).rglob("*.py"))}
        raw = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(raw).hexdigest() == digest, "frozen tree drift: "+name)
    return spec


def size():
    return sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())


def bounded_save(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    require(len(raw) <= 4*2**20 and 2*(size()+2*len(raw)+2**20) <= CAPS["artifact_bytes"],
            "scalar cumulative artifact and archive reserve")
    return save(path, value)


def worker(anchor, deadline, phase_deadline):
    limits(phase_deadline)
    binding(anchor)
    import margin_scalar_cases as cases
    def prefix(value, work):
        bounded_save(RUN/("prefix_%d.json" % len(value["records"])),
                     dict(value, work_counts=dict(work.counts)))
    value = cases.run(deadline, prefix)
    value.update(freeze_sha256=anchor, phase_deadline=phase_deadline,
                 rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))
    receipt = bounded_save(RUN/"worker.json", value)
    require(time.monotonic() < phase_deadline, "worker export deadline")
    print(json.dumps(dict(status=value["status"], worker=receipt)), flush=True)
    return 0 if value["status"] == "PASS" else 1


def main(anchor):
    deadline = ENTRY+300
    limits(deadline)
    from scoped_proof import owned_session as owned
    from scoped_source.development_work import DevelopmentWork
    import margin_scalar_cases as cases
    result = dict(schema="MARGIN_SCALAR_CONTROL_RESULT_V1", status="FAIL", policy=POLICY,
                  freeze_sha256=anchor, entry=ENTRY, deadline=deadline, caps=CAPS,
                  full_cold_start_seconds=None, trained_source=False, gpu=False,
                  complete_request=False, sessions=[])
    created = False
    try:
        require(sys.executable == PYTHON, "existing act-py312")
        frozen = binding(anchor)
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
        def launch(name, args, run_end, clean_end):
            receipt = owned.execute(args, run_deadline=run_end, cleanup_deadline=clean_end,
                                    env=ENV, rss_limit=None, development_policy=POLICY)
            result["sessions"].append(dict(name=name, receipt=receipt))
            bounded_save(RUN/(name+"_session.json"), receipt)
            return receipt
        receipt = launch("native", [PYTHON, "-I", "-B", __file__, "--worker", anchor,
                                     str(deadline), str(ENTRY+200)], ENTRY+210, ENTRY+220)
        require(successful(receipt), "native scalar session failed")
        result["worker_sha256"] = sha(RUN/"worker.json")
        value = load(RUN/"worker.json", result["worker_sha256"], 4*2**20)
        require(value["freeze_sha256"] == anchor and value["deadline"] == deadline
                and value["rlimit_as"] == [CAPS["as_bytes"]]*2, "worker identity/deadline/AS")
        work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(value["work_counts"])
        before = sum(work.counts.values()); started = time.monotonic()
        receipts = cases.receive(value, work)
        result["online_reception"] = dict(receipts=receipts, work_counts=dict(work.counts),
                                         work=sum(work.counts.values())-before,
                                         seconds=time.monotonic()-started)
        relocated = RUN/"relocated"; relocated.mkdir()
        started = time.monotonic()
        total = sum((ROOT/src).stat().st_size for src in frozen["portable_files"])+(RUN/"worker.json").stat().st_size
        require(2*(size()+total+2**20) <= CAPS["artifact_bytes"], "relocation archive reserve")
        for src, dest in frozen["portable_files"].items():
            target = relocated/dest; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/src, target)
            require(sha(target) == frozen["files"][src], "relocated code identity")
        shutil.copyfile(RUN/"worker.json", relocated/"candidate.json")
        continuation = bounded_save(relocated/"continued.json", dict(work.counts))
        result["relocation_seconds"] = time.monotonic()-started
        receipt = launch("portable", [PYTHON, "-I", "-S", "-B", str(relocated/"margin_scalar_recheck.py"),
                                       result["worker_sha256"], continuation["sha256"], str(deadline)],
                         ENTRY+260, ENTRY+280)
        require(successful(receipt), "portable scalar session failed")
        portable = load(relocated/"receipt.json", limit=2**20)
        require(portable["status"] == "PASS" and portable["receipts"] == receipts
                and portable["deadline"] == deadline
                and all(portable["work_counts"][k] >= work.counts[k] for k in work.counts),
                "same scalar result and continued accounting")
        result["portable_reception"] = portable
        result["fds_after"] = len(os.listdir("/proc/self/fd"))
        require(result["fds_before"] == result["fds_after"], "final FD balance")
        result["status"] = "PASS"
    except BaseException as exc:
        result.update(status="FAIL", error=repr(exc))
    if created:
        result["artifact_bytes_before_terminal"] = size()
        result["archive_reserve"] = "ONE_ADDITIONAL_COPY_OF_ALL_RUN_ARTIFACTS"
        result["elapsed_before_terminal_export"] = time.monotonic()-ENTRY
        receipt = bounded_save(RUN/"result.json", result)
        require(time.monotonic() < deadline, "late result rejected")
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
    raise SystemExit("explicit once-only frozen scalar scope required")
